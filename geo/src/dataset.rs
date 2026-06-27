use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, LargeBinaryArray,
    StructArray,
};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use arrow_json::LineDelimitedWriter;
use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index2DF32, Index3DBuilder, Index3DF32};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{EdgeInterpolationAlgorithm, LogicalType, Type as ParquetPhysicalType};
use parquet::file::metadata::{FileMetaData, ParquetMetaData};
use parquet::file::reader::ChunkReader;
use serde::Deserialize;

use crate::geoarrow;
use crate::manifest;
use crate::wkb::{self, GeometryBounds};
use crate::{
    AntimeridianPolicy, BuildRequest, ColumnCapabilities, ConvertRequest, CoordinateDims, CrsInfo,
    DeclaredExtent, DiscoveryWarning, EdgeAlgorithm, EdgeModel, EnvelopePolicy, FeatureRef,
    FileGeoMetadata, GeoArtifact, GeoArtifactManifest, GeoDiscovery, GeoError, GeoIndex,
    GeoIndex2D, GeoIndex3D, GeoIndexMetadata, GeometryColumn, GeometryColumnInfo, GeometryEncoding,
    GeometryMetadataSource, GeometryProfile, GeometryScan, GeometryScan2D, GeometryScan3D,
    GeometrySelectionReason, GeometrySelector, GeometryTypeSet, IndexBuildOptions,
    IndexDimsRequest, InspectRequest, NullPolicy, PayloadPlan, PropertyProjection, RowBoundsSource,
    SelectionStatus, StoragePrecision,
};

/// Content type used for [`PayloadPlan::RowRef`](crate::PayloadPlan::RowRef)
/// payload sections.
pub const FEATURE_REF_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.feature-ref";
/// Content type used for [`PayloadPlan::RowWkb`](crate::PayloadPlan::RowWkb)
/// payload sections.
pub const FEATURE_WKB_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.feature-wkb";
/// Content type used for [`PayloadPlan::FeatureJson`](crate::PayloadPlan::FeatureJson)
/// payload sections.
pub const FEATURE_JSON_CONTENT_TYPE: &str = "application/geo+json";
/// Byte length of the fixed-width [`FeatureRef`] payload record.
pub const FEATURE_REF_RECORD_LEN: usize = 24;

/// Open a GeoParquet or native Parquet geospatial dataset.
///
/// This performs metadata discovery only. Row data is read later by
/// [`GeoDataset::inspect`] with `exact: true`, [`GeoDataset::scan`],
/// [`GeoDataset::build`], or [`GeoDataset::convert`].
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::open;
///
/// let dataset = open(File::open("cities.parquet")?)?;
/// println!("{} rows", dataset.discovery().num_rows);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn open<R: ChunkReader + 'static>(reader: R) -> Result<GeoDataset<R>, GeoError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(reader)?;
    let (discovery, states) = discover_metadata(builder.metadata())?;
    let source_fingerprint = source_fingerprint(builder.metadata());
    Ok(GeoDataset {
        builder: Some(builder),
        discovery,
        states,
        source_fingerprint,
    })
}

/// Session object for one opened geospatial Parquet source.
///
/// `GeoDataset` owns the Parquet reader builder and exposes the high-level
/// workflow: discover columns, select/profile one geometry column, scan feature
/// envelopes, build an in-memory index, or convert to a streamable `PSINDEX`
/// artifact.
///
/// A dataset is consumed the first time rows are read. Call
/// [`GeoDataset::discovery`] and [`GeoDataset::select`] freely before scanning;
/// open a new dataset if you need to run multiple independent scans.
pub struct GeoDataset<R: ChunkReader> {
    builder: Option<ParquetRecordBatchReaderBuilder<R>>,
    discovery: GeoDiscovery,
    states: Vec<ColumnState>,
    source_fingerprint: String,
}

impl<R: ChunkReader + 'static> GeoDataset<R> {
    /// Return metadata-only discovery for all usable geometry columns.
    ///
    /// Discovery never scans geometry payloads. Unknown dimensions or geometry
    /// types in this value mean “not declared in metadata”; use
    /// [`GeoDataset::inspect`] with [`InspectRequest::exact`] when exact row
    /// inspection is needed.
    pub fn discovery(&self) -> &GeoDiscovery {
        &self.discovery
    }

    /// Resolve a selector to a concrete geometry column.
    ///
    /// This is a metadata-only operation. It applies the same default selection
    /// policy used by scan/build/convert: GeoParquet primary column first, then
    /// exactly one native Parquet geospatial column.
    pub fn select(&self, selector: GeometrySelector) -> Result<GeometryColumn, GeoError> {
        let state = self.select_state(&selector)?;
        Ok(GeometryColumn {
            name: state.info.name.clone(),
            info: state.info.clone(),
        })
    }

    /// Profile the selected geometry column.
    ///
    /// With the default request this returns metadata-derived information. Set
    /// [`InspectRequest::exact`] to scan rows when dimensions are unknown from
    /// metadata.
    pub fn inspect(&mut self, req: InspectRequest) -> Result<GeometryProfile, GeoError> {
        let state = self.select_state(&req.selector)?.clone();
        if req.exact && state.info.coordinate_dims == CoordinateDims::Unknown {
            let scan = self.scan(ScanRequestForInspect::from_request(req))?;
            return Ok(match scan {
                GeometryScan::D2(scan) => scan.profile,
                GeometryScan::D3(scan) => scan.profile,
            });
        }
        Ok(profile_from_state(&state, self.discovery.num_rows))
    }

    /// Scan feature envelopes, feature references, and optional payloads.
    ///
    /// The scan result contains one index entry per envelope. With geographic
    /// antimeridian splitting enabled, one source row may produce multiple
    /// entries with the same row number and different [`FeatureRef::part`]
    /// values.
    pub fn scan(&mut self, req: crate::ScanRequest) -> Result<GeometryScan, GeoError> {
        let state = self.select_state(&req.selector)?.clone();
        self.scan_selected(&state, req)
    }

    /// Build an in-memory [`GeoIndex`] over the selected geometry column.
    ///
    /// The returned index maps candidate hits back to [`FeatureRef`] values
    /// rather than compact item ids. Use [`GeoIndex2D::raw_index`] or
    /// [`GeoIndex3D::raw_index`] when you need direct access to the core index.
    pub fn build(&mut self, req: BuildRequest) -> Result<GeoIndex, GeoError> {
        let scan = self.scan(crate::ScanRequest {
            selector: req.selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: PayloadPlan::None,
        })?;
        Ok(match scan {
            GeometryScan::D2(scan) => {
                let mut builder = builder_2d(scan.boxes.len(), &req.build);
                for bbox in &scan.boxes {
                    builder.add(*bbox);
                }
                let metadata = GeoIndexMetadata {
                    profile: scan.profile,
                    feature_count: unique_feature_count(&scan.features),
                    index_entry_count: scan.boxes.len(),
                    entries_may_duplicate_rows: entries_may_duplicate_rows(&scan.features),
                };
                GeoIndex::D2(GeoIndex2D {
                    index: builder.finish()?,
                    features: scan.features,
                    metadata,
                })
            }
            GeometryScan::D3(scan) => {
                let mut builder = builder_3d(scan.boxes.len(), &req.build);
                for bbox in &scan.boxes {
                    builder.add(*bbox);
                }
                let metadata = GeoIndexMetadata {
                    profile: scan.profile,
                    feature_count: unique_feature_count(&scan.features),
                    index_entry_count: scan.boxes.len(),
                    entries_may_duplicate_rows: entries_may_duplicate_rows(&scan.features),
                };
                GeoIndex::D3(GeoIndex3D {
                    index: builder.finish()?,
                    features: scan.features,
                    metadata,
                })
            }
        })
    }

    /// Convert the selected geometry column into a streamable `PSINDEX` buffer.
    ///
    /// The generated bytes include the core index, optional payloads, and a
    /// `geoM` manifest describing the selected column, CRS, dimensions, payload
    /// plan, and feature-entry mapping. Existing contents of `out` are replaced.
    pub fn convert_into(
        &mut self,
        req: ConvertRequest,
        out: &mut Vec<u8>,
    ) -> Result<GeoArtifact, GeoError> {
        let selector = req.selector.clone();
        let scan = self.scan(crate::ScanRequest {
            selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: req.payload.clone(),
        })?;
        let manifest = match scan {
            GeometryScan::D2(scan) => {
                let mut builder = builder_2d(scan.boxes.len(), &req.build);
                for bbox in &scan.boxes {
                    builder.add(*bbox);
                }
                let payload = scan.payloads.as_deref();
                serialize_2d(
                    builder,
                    req.precision,
                    req.interleaved,
                    payload,
                    &scan.profile,
                    out,
                )?;
                artifact_manifest(
                    &scan.profile,
                    &req,
                    unique_feature_count(&scan.features),
                    scan.boxes.len(),
                    entries_may_duplicate_rows(&scan.features),
                    &self.source_fingerprint,
                )
            }
            GeometryScan::D3(scan) => {
                let mut builder = builder_3d(scan.boxes.len(), &req.build);
                for bbox in &scan.boxes {
                    builder.add(*bbox);
                }
                let payload = scan.payloads.as_deref();
                serialize_3d(
                    builder,
                    req.precision,
                    req.interleaved,
                    payload,
                    &scan.profile,
                    out,
                )?;
                artifact_manifest(
                    &scan.profile,
                    &req,
                    unique_feature_count(&scan.features),
                    scan.boxes.len(),
                    entries_may_duplicate_rows(&scan.features),
                    &self.source_fingerprint,
                )
            }
        };
        let base = std::mem::take(out);
        manifest::append_geo_manifest(&base, &manifest, out)?;
        Ok(GeoArtifact {
            manifest,
            bytes_len: out.len(),
        })
    }

    /// Convert the selected geometry column into a new `Vec<u8>`.
    ///
    /// This is a convenience wrapper around [`GeoDataset::convert_into`].
    pub fn convert(&mut self, req: ConvertRequest) -> Result<Vec<u8>, GeoError> {
        let mut out = Vec::new();
        self.convert_into(req, &mut out)?;
        Ok(out)
    }

    fn select_state(&self, selector: &GeometrySelector) -> Result<&ColumnState, GeoError> {
        match selector {
            GeometrySelector::Default => match &self.discovery.default_selection {
                SelectionStatus::Selected { column, .. } => self
                    .state_by_name(column)
                    .ok_or_else(|| GeoError::GeometryColumnNotFound(column.clone())),
                SelectionStatus::Ambiguous { columns } => Err(GeoError::AmbiguousGeometryColumn {
                    columns: columns.clone(),
                }),
                SelectionStatus::Missing { column } => {
                    Err(GeoError::GeometryColumnNotFound(column.clone()))
                }
                SelectionStatus::None => Err(GeoError::NoGeometryColumn),
            },
            GeometrySelector::Name(name) => self
                .state_by_name(name)
                .ok_or_else(|| GeoError::GeometryColumnNotFound(name.clone())),
            GeometrySelector::GeoParquetPrimary => {
                let Some(primary) = &self.discovery.file_metadata.geoparquet_primary_column else {
                    return Err(GeoError::NoGeometryColumn);
                };
                self.state_by_name(primary)
                    .ok_or_else(|| GeoError::GeometryColumnNotFound(primary.clone()))
            }
            GeometrySelector::SingleNativeParquet => {
                let native: Vec<_> = self
                    .states
                    .iter()
                    .filter(|state| state.info.encoding.is_native_parquet())
                    .collect();
                match native.as_slice() {
                    [one] => Ok(one),
                    [] => Err(GeoError::NoGeometryColumn),
                    many => Err(GeoError::AmbiguousGeometryColumn {
                        columns: many.iter().map(|state| state.info.name.clone()).collect(),
                    }),
                }
            }
            GeometrySelector::FirstUsable => self
                .states
                .iter()
                .find(|state| state.info.capabilities.can_scan_envelopes)
                .ok_or(GeoError::NoGeometryColumn),
        }
    }

    fn state_by_name(&self, name: &str) -> Option<&ColumnState> {
        self.states.iter().find(|state| state.info.name == name)
    }

    fn take_reader(
        &mut self,
    ) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader, GeoError> {
        let builder = self.builder.take().ok_or(GeoError::DatasetConsumed)?;
        builder.build().map_err(GeoError::from)
    }

    fn scan_selected(
        &mut self,
        state: &ColumnState,
        req: crate::ScanRequest,
    ) -> Result<GeometryScan, GeoError> {
        let batches = self.take_reader()?;
        let mut entries = Vec::new();
        let mut detected_dims = state.info.coordinate_dims;
        let collect_lons = matches!(req.envelope, EnvelopePolicy::Geographic { .. });
        let want_payload = !matches!(req.payload, PayloadPlan::None);
        let mut row_base = 0u64;

        for batch in batches {
            let batch = batch?;
            let geom = batch
                .column_by_name(&state.info.name)
                .ok_or_else(|| GeoError::GeometryColumnNotFound(state.info.name.clone()))?
                .clone();
            let binary = needs_binary(&state.info.encoding)
                .then(|| binary_column(&batch, &state.info.name))
                .transpose()?;
            let covering = covering_arrays(&batch, state.covering.as_ref())?;
            let property_columns = projection_columns(&batch, &state.info.name, &req.payload)?;
            for row in 0..batch.num_rows() {
                let row_number = row_base + row as u64;
                let scanned = scan_one_row(
                    state,
                    &geom,
                    binary.as_ref(),
                    covering.as_ref(),
                    row,
                    collect_lons,
                    want_payload,
                )?;
                let Some((bounds, wkb)) = scanned else {
                    match req.nulls {
                        NullPolicy::Skip => continue,
                        NullPolicy::Error => {
                            return Err(GeoError::NullGeometry {
                                row: row_number as usize,
                            });
                        }
                    }
                };
                detected_dims = detected_dims.merge(bounds.dims);
                let feature = FeatureRef::row(row_number);
                let property_json = match &req.payload {
                    PayloadPlan::FeatureJson { properties } => Some(feature_json_payload(
                        &feature,
                        wkb.as_deref(),
                        &batch,
                        row,
                        properties,
                        &property_columns,
                    )?),
                    PayloadPlan::RowRef => Some(encode_feature_ref(&feature)),
                    PayloadPlan::RowWkb => Some(encode_feature_wkb(
                        &feature,
                        wkb.as_deref().ok_or_else(|| {
                            GeoError::UnsupportedEncoding(format!(
                                "{} cannot emit WKB payload",
                                state.info.encoding
                            ))
                        })?,
                    )),
                    PayloadPlan::None => None,
                };
                entries.push(ScanEntry {
                    bounds,
                    feature,
                    payload: property_json,
                });
            }
            row_base += batch.num_rows() as u64;
        }

        let dims = resolve_scan_dims(req.dims, detected_dims, &entries)?;
        let mut profile = profile_from_state(state, self.discovery.num_rows);
        profile.coordinate_dims = detected_dims;

        match dims {
            ResolvedDims::D2 => {
                let mut boxes = Vec::new();
                let mut features = Vec::new();
                let mut payloads = req.payload_payloads();
                for entry in entries {
                    let parts = split_2d(&entry.bounds, req.envelope, entry.feature.row_number)?;
                    let has_parts = parts.len() > 1;
                    for (part_index, bbox) in parts.into_iter().enumerate() {
                        let mut feature = entry.feature.clone();
                        if has_parts {
                            feature.part = Some(part_index as u16);
                        }
                        boxes.push(bbox);
                        features.push(feature);
                        if let Some(payloads) = payloads.as_mut() {
                            payloads.push(entry.payload.clone().unwrap_or_default());
                        }
                    }
                }
                Ok(GeometryScan::D2(GeometryScan2D {
                    boxes,
                    features,
                    payloads,
                    profile,
                }))
            }
            ResolvedDims::D3 => {
                let mut boxes = Vec::new();
                let mut features = Vec::new();
                let mut payloads = req.payload_payloads();
                for entry in entries {
                    let parts = split_3d(&entry.bounds, req.envelope, entry.feature.row_number)?;
                    let has_parts = parts.len() > 1;
                    for (part_index, bbox) in parts.into_iter().enumerate() {
                        let mut feature = entry.feature.clone();
                        if has_parts {
                            feature.part = Some(part_index as u16);
                        }
                        boxes.push(bbox);
                        features.push(feature);
                        if let Some(payloads) = payloads.as_mut() {
                            payloads.push(entry.payload.clone().unwrap_or_default());
                        }
                    }
                }
                Ok(GeometryScan::D3(GeometryScan3D {
                    boxes,
                    features,
                    payloads,
                    profile,
                }))
            }
        }
    }
}

trait PayloadVec {
    fn payload_payloads(&self) -> Option<Vec<Vec<u8>>>;
}

impl PayloadVec for crate::ScanRequest {
    fn payload_payloads(&self) -> Option<Vec<Vec<u8>>> {
        (!matches!(self.payload, PayloadPlan::None)).then(Vec::new)
    }
}

struct ScanRequestForInspect;

impl ScanRequestForInspect {
    fn from_request(req: InspectRequest) -> crate::ScanRequest {
        crate::ScanRequest {
            selector: req.selector,
            dims: IndexDimsRequest::Auto,
            nulls: NullPolicy::Skip,
            envelope: EnvelopePolicy::Planar,
            payload: PayloadPlan::None,
        }
    }
}

#[derive(Debug, Clone)]
struct ColumnState {
    info: GeometryColumnInfo,
    covering: Option<GeoParquetBboxCovering>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeoParquetMetadata {
    version: String,
    primary_column: String,
    columns: HashMap<String, GeoParquetColumnMetadata>,
}

impl GeoParquetMetadata {
    fn from_parquet_meta(metadata: &FileMetaData) -> Option<Result<Self, GeoError>> {
        let value = metadata
            .key_value_metadata()?
            .iter()
            .find(|kv| kv.key == "geo")?
            .value
            .as_ref()?;
        Some(serde_json::from_str(value).map_err(|e| GeoError::Metadata(e.to_string())))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct GeoParquetColumnMetadata {
    encoding: String,
    #[serde(default)]
    geometry_types: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    crs: Option<Option<serde_json::Value>>,
    #[serde(default)]
    edges: Option<String>,
    #[serde(default)]
    bbox: Option<Vec<f64>>,
    #[serde(default)]
    covering: Option<GeoParquetCovering>,
}

fn deserialize_present_value<'de, D>(
    deserializer: D,
) -> Result<Option<Option<serde_json::Value>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer)
        .map(|value| Some(if value.is_null() { None } else { Some(value) }))
}

#[derive(Debug, Clone, Deserialize)]
struct GeoParquetCovering {
    bbox: GeoParquetBboxCovering,
}

#[derive(Debug, Clone, Deserialize)]
struct GeoParquetBboxCovering {
    xmin: Vec<String>,
    ymin: Vec<String>,
    #[serde(default)]
    zmin: Option<Vec<String>>,
    xmax: Vec<String>,
    ymax: Vec<String>,
    #[serde(default)]
    zmax: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct NativeColumn {
    name: String,
    encoding: GeometryEncoding,
    crs: CrsInfo,
    edges: EdgeModel,
    dims: CoordinateDims,
}

fn discover_metadata(meta: &ParquetMetaData) -> Result<(GeoDiscovery, Vec<ColumnState>), GeoError> {
    let file_meta = meta.file_metadata();
    let geo_meta = match GeoParquetMetadata::from_parquet_meta(file_meta) {
        Some(Ok(value)) => Some(value),
        Some(Err(err)) => return Err(err),
        None => None,
    };
    let native = native_geo_columns(meta);
    let mut states = Vec::new();
    let mut warnings = Vec::new();
    let mut geo_names = HashSet::new();

    if let Some(gpq) = &geo_meta {
        let mut names: Vec<_> = gpq.columns.keys().cloned().collect();
        names.sort();
        for name in names {
            let col = &gpq.columns[&name];
            let encoding = geoarrow::encoding_from_geoparquet(&col.encoding);
            if !geoarrow::is_supported_encoding(&encoding) {
                warnings.push(DiscoveryWarning::UnsupportedGeoParquetEncoding {
                    column: name.clone(),
                    encoding: col.encoding.clone(),
                });
            }
            let covering = col.covering.as_ref().map(|covering| covering.bbox.clone());
            let has_covering = covering.is_some();
            let dims = covering
                .as_ref()
                .and_then(|covering| {
                    (covering.zmin.is_some() && covering.zmax.is_some())
                        .then_some(CoordinateDims::Xyz)
                })
                .unwrap_or_else(|| CoordinateDims::from_geometry_types(&col.geometry_types));
            let can_scan = geoarrow::is_supported_encoding(&encoding) || has_covering;
            let info = GeometryColumnInfo {
                name: name.clone(),
                source: GeometryMetadataSource::GeoParquet,
                encoding: encoding.clone(),
                crs: geoparquet_crs(&col.crs),
                edges: geoparquet_edges(col.edges.as_deref()),
                coordinate_dims: dims,
                geometry_types: GeometryTypeSet {
                    types: col.geometry_types.clone(),
                },
                extent: col.bbox.clone().map(|values| DeclaredExtent { values }),
                row_bounds: row_bounds_sources(&encoding, has_covering),
                capabilities: ColumnCapabilities {
                    can_scan_envelopes: can_scan,
                    can_build_index: can_scan,
                    can_emit_row_wkb: encoding.is_wkb_payload()
                        || matches!(encoding, GeometryEncoding::GeoArrow { .. }),
                    can_emit_feature_json: encoding.is_wkb_payload()
                        || matches!(encoding, GeometryEncoding::GeoArrow { .. }),
                },
            };
            geo_names.insert(name);
            states.push(ColumnState { info, covering });
        }
    }

    let mut native_names = Vec::new();
    for candidate in native {
        if geo_names.contains(&candidate.name) {
            continue;
        }
        native_names.push(candidate.name.clone());
        states.push(ColumnState {
            info: GeometryColumnInfo {
                name: candidate.name,
                source: GeometryMetadataSource::ParquetGeospatial,
                encoding: candidate.encoding,
                crs: candidate.crs,
                edges: candidate.edges,
                coordinate_dims: candidate.dims,
                geometry_types: GeometryTypeSet::unknown(),
                extent: None,
                row_bounds: vec![RowBoundsSource::WkbEnvelope],
                capabilities: ColumnCapabilities {
                    can_scan_envelopes: true,
                    can_build_index: true,
                    can_emit_row_wkb: true,
                    can_emit_feature_json: true,
                },
            },
            covering: None,
        });
    }

    let default_selection = default_selection(&states, geo_meta.as_ref(), &native_names);
    if let Some(gpq) = &geo_meta
        && !states.iter().any(|state| {
            state.info.source == GeometryMetadataSource::GeoParquet
                && state.info.name == gpq.primary_column
        })
    {
        warnings.push(DiscoveryWarning::GeoParquetPrimaryMissing {
            column: gpq.primary_column.clone(),
        });
    }
    Ok((
        GeoDiscovery {
            num_rows: file_meta.num_rows().max(0) as u64,
            file_metadata: FileGeoMetadata {
                geoparquet_version: geo_meta.as_ref().map(|gpq| gpq.version.clone()),
                geoparquet_primary_column: geo_meta.as_ref().map(|gpq| gpq.primary_column.clone()),
                has_geoparquet_metadata: geo_meta.is_some(),
            },
            columns: states.iter().map(|state| state.info.clone()).collect(),
            default_selection,
            warnings,
        },
        states,
    ))
}

fn default_selection(
    states: &[ColumnState],
    geo_meta: Option<&GeoParquetMetadata>,
    native_names: &[String],
) -> SelectionStatus {
    if let Some(gpq) = geo_meta {
        return if states.iter().any(|state| {
            state.info.source == GeometryMetadataSource::GeoParquet
                && state.info.name == gpq.primary_column
        }) {
            SelectionStatus::Selected {
                column: gpq.primary_column.clone(),
                reason: GeometrySelectionReason::GeoParquetPrimary,
            }
        } else {
            SelectionStatus::None
        };
    }
    match native_names {
        [] => SelectionStatus::None,
        [one] => SelectionStatus::Selected {
            column: one.clone(),
            reason: GeometrySelectionReason::SingleNativeParquet,
        },
        many => SelectionStatus::Ambiguous {
            columns: many.to_vec(),
        },
    }
}

fn native_geo_columns(meta: &ParquetMetaData) -> Vec<NativeColumn> {
    meta.file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(column_index, col)| {
            let parts = col.path().parts();
            if parts.len() != 1
                || col.max_rep_level() != 0
                || col.physical_type() != ParquetPhysicalType::BYTE_ARRAY
            {
                return None;
            }
            match col.logical_type_ref()? {
                LogicalType::Geometry(geometry) => Some(NativeColumn {
                    name: parts[0].clone(),
                    encoding: GeometryEncoding::ParquetGeometry,
                    crs: native_crs(&geometry.crs),
                    edges: EdgeModel::Planar,
                    dims: native_dim_hint(meta, column_index),
                }),
                LogicalType::Geography(geography) => {
                    let algorithm = geography
                        .algorithm()
                        .map(edge_algorithm)
                        .unwrap_or(EdgeAlgorithm::Spherical);
                    Some(NativeColumn {
                        name: parts[0].clone(),
                        encoding: GeometryEncoding::ParquetGeography { algorithm },
                        crs: native_crs(&geography.crs),
                        edges: EdgeModel::Spherical,
                        dims: native_dim_hint(meta, column_index),
                    })
                }
                _ => None,
            }
        })
        .collect()
}

fn native_dim_hint(meta: &ParquetMetaData, column_index: usize) -> CoordinateDims {
    let mut saw_stats = false;
    let mut dims = CoordinateDims::Unknown;
    for row_group in meta.row_groups() {
        let Some(types) = row_group
            .column(column_index)
            .geo_statistics()
            .and_then(|stats| stats.geospatial_types())
        else {
            return CoordinateDims::Unknown;
        };
        saw_stats = true;
        for &ty in types {
            dims = dims.merge(dims_from_wkb_type(ty));
        }
    }
    if saw_stats {
        dims
    } else {
        CoordinateDims::Unknown
    }
}

fn dims_from_wkb_type(ty: i32) -> CoordinateDims {
    if (3000..4000).contains(&ty) {
        CoordinateDims::Xyzm
    } else if (2000..3000).contains(&ty) {
        CoordinateDims::Xym
    } else if (1000..2000).contains(&ty) {
        CoordinateDims::Xyz
    } else {
        CoordinateDims::Xy
    }
}

fn edge_algorithm(value: EdgeInterpolationAlgorithm) -> EdgeAlgorithm {
    match value {
        EdgeInterpolationAlgorithm::SPHERICAL => EdgeAlgorithm::Spherical,
        EdgeInterpolationAlgorithm::VINCENTY => EdgeAlgorithm::Vincenty,
        EdgeInterpolationAlgorithm::THOMAS => EdgeAlgorithm::Thomas,
        EdgeInterpolationAlgorithm::ANDOYER => EdgeAlgorithm::Andoyer,
        EdgeInterpolationAlgorithm::KARNEY => EdgeAlgorithm::Karney,
        EdgeInterpolationAlgorithm::_Unknown(_) => EdgeAlgorithm::Unknown,
    }
}

fn native_crs(value: &Option<String>) -> CrsInfo {
    value
        .as_ref()
        .map(|value| CrsInfo::PresentString {
            value: value.clone(),
        })
        .unwrap_or_else(|| CrsInfo::ImpliedDefault {
            value: "OGC:CRS84".to_string(),
        })
}

fn geoparquet_crs(value: &Option<Option<serde_json::Value>>) -> CrsInfo {
    match value {
        Some(Some(value)) => CrsInfo::Present {
            value: value.clone(),
        },
        Some(None) => CrsInfo::ExplicitNone,
        None => CrsInfo::ImpliedDefault {
            value: "OGC:CRS84".to_string(),
        },
    }
}

fn geoparquet_edges(value: Option<&str>) -> EdgeModel {
    match value {
        Some(edge) if edge.eq_ignore_ascii_case("spherical") => EdgeModel::Spherical,
        Some(edge) if edge.eq_ignore_ascii_case("planar") => EdgeModel::Planar,
        Some(_) => EdgeModel::Unknown,
        None => EdgeModel::Planar,
    }
}

fn row_bounds_sources(encoding: &GeometryEncoding, has_covering: bool) -> Vec<RowBoundsSource> {
    if has_covering {
        vec![RowBoundsSource::Covering]
    } else if encoding.is_wkb_payload() {
        vec![RowBoundsSource::WkbEnvelope]
    } else if matches!(encoding, GeometryEncoding::GeoArrow { .. }) {
        vec![RowBoundsSource::GeoArrowScan]
    } else {
        Vec::new()
    }
}

fn profile_from_state(state: &ColumnState, num_rows: u64) -> GeometryProfile {
    GeometryProfile {
        column: state.info.name.clone(),
        source: state.info.source,
        encoding: state.info.encoding.clone(),
        crs: state.info.crs.clone(),
        edges: state.info.edges,
        coordinate_dims: state.info.coordinate_dims,
        geometry_types: state.info.geometry_types.clone(),
        extent: state.info.extent.clone(),
        row_bounds: state.info.row_bounds.clone(),
        num_rows,
    }
}

enum WkbCol<'a> {
    Bin(&'a BinaryArray),
    Large(&'a LargeBinaryArray),
    View(&'a BinaryViewArray),
}

impl WkbCol<'_> {
    fn is_null(&self, row: usize) -> bool {
        match self {
            WkbCol::Bin(array) => array.is_null(row),
            WkbCol::Large(array) => array.is_null(row),
            WkbCol::View(array) => array.is_null(row),
        }
    }

    fn value(&self, row: usize) -> &[u8] {
        match self {
            WkbCol::Bin(array) => array.value(row),
            WkbCol::Large(array) => array.value(row),
            WkbCol::View(array) => array.value(row),
        }
    }
}

fn needs_binary(encoding: &GeometryEncoding) -> bool {
    encoding.is_wkb_payload()
}

fn binary_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<WkbCol<'a>, GeoError> {
    let arr = batch
        .column_by_name(name)
        .ok_or_else(|| GeoError::GeometryColumnNotFound(name.to_string()))?;
    if let Some(a) = arr.as_any().downcast_ref::<BinaryArray>() {
        Ok(WkbCol::Bin(a))
    } else if let Some(a) = arr.as_any().downcast_ref::<LargeBinaryArray>() {
        Ok(WkbCol::Large(a))
    } else if let Some(a) = arr.as_any().downcast_ref::<BinaryViewArray>() {
        Ok(WkbCol::View(a))
    } else {
        Err(GeoError::UnsupportedEncoding(format!(
            "geometry column `{name}` is not binary WKB ({:?})",
            arr.data_type()
        )))
    }
}

struct CoveringArrays {
    xmin: Vec<f64>,
    ymin: Vec<f64>,
    zmin: Option<Vec<f64>>,
    xmax: Vec<f64>,
    ymax: Vec<f64>,
    zmax: Option<Vec<f64>>,
}

fn covering_arrays(
    batch: &RecordBatch,
    covering: Option<&GeoParquetBboxCovering>,
) -> Result<Option<CoveringArrays>, GeoError> {
    let Some(covering) = covering else {
        return Ok(None);
    };
    Ok(Some(CoveringArrays {
        xmin: f64_path(batch, &covering.xmin)?,
        ymin: f64_path(batch, &covering.ymin)?,
        zmin: covering
            .zmin
            .as_ref()
            .map(|path| f64_path(batch, path))
            .transpose()?,
        xmax: f64_path(batch, &covering.xmax)?,
        ymax: f64_path(batch, &covering.ymax)?,
        zmax: covering
            .zmax
            .as_ref()
            .map(|path| f64_path(batch, path))
            .transpose()?,
    }))
}

fn f64_path(batch: &RecordBatch, path: &[String]) -> Result<Vec<f64>, GeoError> {
    let arr = descend(batch, path)?;
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        Ok((0..a.len()).map(|i| a.value(i)).collect())
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        Ok((0..a.len()).map(|i| a.value(i) as f64).collect())
    } else {
        Err(GeoError::Metadata(format!(
            "bbox covering path {path:?} is not a float column ({:?})",
            arr.data_type()
        )))
    }
}

fn descend<'a>(batch: &'a RecordBatch, path: &[String]) -> Result<&'a ArrayRef, GeoError> {
    let first = path
        .first()
        .ok_or_else(|| GeoError::Metadata("empty bbox covering path".to_string()))?;
    let mut cur = batch
        .column_by_name(first)
        .ok_or_else(|| GeoError::Metadata(format!("bbox covering column `{first}` not found")))?;
    for field in &path[1..] {
        let st = cur.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
            GeoError::Metadata(format!(
                "bbox covering path segment `{field}` is not a struct"
            ))
        })?;
        cur = st.column_by_name(field).ok_or_else(|| {
            GeoError::Metadata(format!("bbox covering field `{field}` not found"))
        })?;
    }
    Ok(cur)
}

type RowScanResult = Option<(GeometryBounds, Option<Vec<u8>>)>;

fn scan_one_row(
    state: &ColumnState,
    geom: &ArrayRef,
    binary: Option<&WkbCol<'_>>,
    covering: Option<&CoveringArrays>,
    row: usize,
    collect_lons: bool,
    want_payload: bool,
) -> Result<RowScanResult, GeoError> {
    if geom.is_null(row) {
        return Ok(None);
    }
    if let Some(binary) = binary {
        if binary.is_null(row) {
            return Ok(None);
        }
        let wkb = binary.value(row);
        let bounds = if let Some(covering) = covering.filter(|_| !want_payload) {
            bounds_from_covering(covering, row, collect_lons)
        } else {
            let Some(bounds) = wkb::bounds(wkb, collect_lons)? else {
                return Ok(None);
            };
            bounds
        };
        return Ok(Some((bounds, want_payload.then(|| wkb.to_vec()))));
    }
    let GeometryEncoding::GeoArrow { kind, .. } = state.info.encoding else {
        return Err(GeoError::UnsupportedEncoding(
            state.info.encoding.to_string(),
        ));
    };
    if let Some(covering) = covering.filter(|_| !want_payload) {
        return Ok(Some((
            bounds_from_covering(covering, row, collect_lons),
            None,
        )));
    }
    let Some(row) = geoarrow::scan_row(geom, kind, state.info.coordinate_dims, row, collect_lons)?
    else {
        return Ok(None);
    };
    Ok(Some((row.bounds, want_payload.then_some(row.wkb))))
}

fn bounds_from_covering(
    covering: &CoveringArrays,
    row: usize,
    collect_lons: bool,
) -> GeometryBounds {
    let mut bounds = GeometryBounds::new(collect_lons);
    bounds.min[0] = covering.xmin[row];
    bounds.min[1] = covering.ymin[row];
    bounds.max[0] = covering.xmax[row];
    bounds.max[1] = covering.ymax[row];
    if let (Some(zmin), Some(zmax)) = (&covering.zmin, &covering.zmax) {
        bounds.min[2] = zmin[row];
        bounds.max[2] = zmax[row];
        bounds.dims = CoordinateDims::Xyz;
    } else {
        bounds.dims = CoordinateDims::Xy;
    }
    bounds.any = true;
    if collect_lons {
        bounds.lon_values.push(bounds.min[0]);
        bounds.lon_values.push(bounds.max[0]);
    }
    bounds
}

#[derive(Debug)]
struct ScanEntry {
    bounds: GeometryBounds,
    feature: FeatureRef,
    payload: Option<Vec<u8>>,
}

enum ResolvedDims {
    D2,
    D3,
}

fn resolve_scan_dims(
    requested: IndexDimsRequest,
    detected: CoordinateDims,
    entries: &[ScanEntry],
) -> Result<ResolvedDims, GeoError> {
    let has_z = detected.has_z() || entries.iter().any(|entry| entry.bounds.dims.has_z());
    match requested {
        IndexDimsRequest::D2 if has_z => Err(GeoError::DimMismatch {
            expected: 2,
            found: 3,
        }),
        IndexDimsRequest::D2 => Ok(ResolvedDims::D2),
        IndexDimsRequest::D3 if !has_z => Err(GeoError::DimMismatch {
            expected: 3,
            found: 2,
        }),
        IndexDimsRequest::D3 => Ok(ResolvedDims::D3),
        IndexDimsRequest::Auto if has_z => Ok(ResolvedDims::D3),
        IndexDimsRequest::Auto => Ok(ResolvedDims::D2),
    }
}

fn split_2d(
    bounds: &GeometryBounds,
    policy: EnvelopePolicy,
    row: u64,
) -> Result<Vec<Box2D>, GeoError> {
    match policy {
        EnvelopePolicy::Planar => Ok(vec![Box2D::new(
            bounds.min[0],
            bounds.min[1],
            bounds.max[0],
            bounds.max[1],
        )]),
        EnvelopePolicy::Geographic { antimeridian } => {
            split_lon(bounds, antimeridian, row).map(|parts| {
                parts
                    .into_iter()
                    .map(|(xmin, xmax)| Box2D::new(xmin, bounds.min[1], xmax, bounds.max[1]))
                    .collect()
            })
        }
    }
}

fn split_3d(
    bounds: &GeometryBounds,
    policy: EnvelopePolicy,
    row: u64,
) -> Result<Vec<Box3D>, GeoError> {
    match policy {
        EnvelopePolicy::Planar => {
            let b = bounds.as_3d();
            Ok(vec![Box3D::new(b[0], b[1], b[2], b[3], b[4], b[5])])
        }
        EnvelopePolicy::Geographic { antimeridian } => {
            let zmin = if bounds.min[2].is_finite() {
                bounds.min[2]
            } else {
                0.0
            };
            let zmax = if bounds.max[2].is_finite() {
                bounds.max[2]
            } else {
                0.0
            };
            split_lon(bounds, antimeridian, row).map(|parts| {
                parts
                    .into_iter()
                    .map(|(xmin, xmax)| {
                        Box3D::new(xmin, bounds.min[1], zmin, xmax, bounds.max[1], zmax)
                    })
                    .collect()
            })
        }
    }
}

fn split_lon(
    bounds: &GeometryBounds,
    policy: AntimeridianPolicy,
    row: u64,
) -> Result<Vec<(f64, f64)>, GeoError> {
    let (start, end, crosses) = if bounds.min[0] > bounds.max[0] {
        (bounds.min[0], bounds.max[0], true)
    } else if bounds.lon_values.len() > 1 {
        minimal_lon_interval(&bounds.lon_values)
    } else {
        (bounds.min[0], bounds.max[0], false)
    };
    if !crosses {
        return Ok(vec![(start, end)]);
    }
    match policy {
        AntimeridianPolicy::Reject => Err(GeoError::Antimeridian { row }),
        AntimeridianPolicy::Split => Ok(vec![(start, 180.0), (-180.0, end)]),
        AntimeridianPolicy::ExpandToWorld => Ok(vec![(-180.0, 180.0)]),
    }
}

fn minimal_lon_interval(values: &[f64]) -> (f64, f64, bool) {
    let mut lons: Vec<f64> = values.iter().copied().map(normalize_lon).collect();
    lons.sort_by(|a, b| a.total_cmp(b));
    lons.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    if lons.len() <= 1 {
        let one = lons.first().copied().unwrap_or(0.0);
        return (one, one, false);
    }
    let mut max_gap = -1.0;
    let mut gap_index = 0usize;
    for i in 0..lons.len() {
        let next = if i + 1 == lons.len() {
            lons[0] + 360.0
        } else {
            lons[i + 1]
        };
        let gap = next - lons[i];
        if gap > max_gap {
            max_gap = gap;
            gap_index = i;
        }
    }
    let start = normalize_lon(lons[(gap_index + 1) % lons.len()]);
    let end = normalize_lon(lons[gap_index]);
    (start, end, start > end)
}

fn normalize_lon(value: f64) -> f64 {
    let mut v = value;
    while v < -180.0 {
        v += 360.0;
    }
    while v > 180.0 {
        v -= 360.0;
    }
    v
}

fn builder_2d(count: usize, opts: &IndexBuildOptions) -> Index2DBuilder {
    let mut builder = Index2DBuilder::new(count);
    if let Some(node_size) = opts.node_size {
        builder = builder.node_size(node_size);
    }
    builder = builder.parallel(opts.parallel);
    builder
}

fn builder_3d(count: usize, opts: &IndexBuildOptions) -> Index3DBuilder {
    let mut builder = Index3DBuilder::new(count);
    if let Some(node_size) = opts.node_size {
        builder = builder.node_size(node_size);
    }
    builder = builder.parallel(opts.parallel);
    builder
}

fn serialize_2d(
    builder: Index2DBuilder,
    precision: StoragePrecision,
    interleaved: bool,
    payload: Option<&[Vec<u8>]>,
    profile: &GeometryProfile,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let crs = profile.crs.as_index_crs();
    match precision {
        StoragePrecision::F64 => {
            let index = builder.finish()?;
            let mut serializer = index.serialize();
            if interleaved && payload.is_some() {
                serializer = serializer.interleaved();
            }
            if let Some(crs) = &crs {
                serializer = serializer.crs(crs);
            }
            if let Some(payload) = payload {
                serializer = serializer
                    .payloads(payload)
                    .content_type(content_type_for_payload(payload));
            }
            serializer.to_bytes_into(out)?;
        }
        StoragePrecision::F32 => {
            let index: Index2DF32 = builder.finish_f32()?;
            let mut serializer = index.serialize();
            if interleaved && payload.is_some() {
                serializer = serializer.interleaved();
            }
            if let Some(crs) = &crs {
                serializer = serializer.crs(crs);
            }
            if let Some(payload) = payload {
                serializer = serializer
                    .payloads(payload)
                    .content_type(content_type_for_payload(payload));
            }
            serializer.to_bytes_into(out)?;
        }
    }
    Ok(())
}

fn serialize_3d(
    builder: Index3DBuilder,
    precision: StoragePrecision,
    interleaved: bool,
    payload: Option<&[Vec<u8>]>,
    profile: &GeometryProfile,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let crs = profile.crs.as_index_crs();
    match precision {
        StoragePrecision::F64 => {
            let index = builder.finish()?;
            let mut serializer = index.serialize();
            if interleaved && payload.is_some() {
                serializer = serializer.interleaved();
            }
            if let Some(crs) = &crs {
                serializer = serializer.crs(crs);
            }
            if let Some(payload) = payload {
                serializer = serializer
                    .payloads(payload)
                    .content_type(content_type_for_payload(payload));
            }
            serializer.to_bytes_into(out)?;
        }
        StoragePrecision::F32 => {
            let index: Index3DF32 = builder.finish_f32()?;
            let mut serializer = index.serialize();
            if interleaved && payload.is_some() {
                serializer = serializer.interleaved();
            }
            if let Some(crs) = &crs {
                serializer = serializer.crs(crs);
            }
            if let Some(payload) = payload {
                serializer = serializer
                    .payloads(payload)
                    .content_type(content_type_for_payload(payload));
            }
            serializer.to_bytes_into(out)?;
        }
    }
    Ok(())
}

fn content_type_for_payload(payload: &[Vec<u8>]) -> &'static str {
    if payload
        .first()
        .is_some_and(|value| value.first().is_some_and(|b| *b == b'{'))
    {
        FEATURE_JSON_CONTENT_TYPE
    } else if payload
        .first()
        .is_some_and(|value| value.len() == FEATURE_REF_RECORD_LEN)
    {
        FEATURE_REF_CONTENT_TYPE
    } else {
        FEATURE_WKB_CONTENT_TYPE
    }
}

fn artifact_manifest(
    profile: &GeometryProfile,
    req: &ConvertRequest,
    feature_count: usize,
    index_entry_count: usize,
    entries_may_duplicate_rows: bool,
    source_fingerprint: &str,
) -> GeoArtifactManifest {
    GeoArtifactManifest {
        schema_version: 2,
        source_format: match profile.source {
            GeometryMetadataSource::GeoParquet => "geoparquet".to_string(),
            GeometryMetadataSource::ParquetGeospatial => "parquet-geospatial".to_string(),
        },
        source_fingerprint: source_fingerprint.to_string(),
        selected_column: profile.column.clone(),
        crs: profile.crs.clone(),
        edges: profile.edges,
        encoding: profile.encoding.clone(),
        dims: profile.coordinate_dims,
        storage_precision: req.precision,
        null_policy: req.nulls,
        antimeridian_policy: match req.envelope {
            EnvelopePolicy::Planar => AntimeridianPolicy::Reject,
            EnvelopePolicy::Geographic { antimeridian } => antimeridian,
        },
        payload_plan: req.payload.clone(),
        feature_count,
        index_entry_count,
        entries_may_duplicate_rows,
    }
}

fn projection_columns(
    batch: &RecordBatch,
    geometry_column: &str,
    payload: &PayloadPlan,
) -> Result<Vec<usize>, GeoError> {
    let properties = match payload {
        PayloadPlan::FeatureJson { properties } => properties,
        _ => return Ok(Vec::new()),
    };
    let fields = batch.schema().fields().clone();
    let names: Vec<_> = fields.iter().map(|field| field.name().clone()).collect();
    let selected: Vec<usize> = match properties {
        PropertyProjection::None => Vec::new(),
        PropertyProjection::AllNonGeometry => names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| (name != geometry_column).then_some(idx))
            .collect(),
        PropertyProjection::Include(include) => include
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| GeoError::PropertyColumnNotFound(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?,
        PropertyProjection::Exclude(exclude) => names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                (name != geometry_column && !exclude.iter().any(|excluded| excluded == name))
                    .then_some(idx)
            })
            .collect(),
    };
    Ok(selected)
}

fn feature_json_payload(
    feature: &FeatureRef,
    wkb: Option<&[u8]>,
    batch: &RecordBatch,
    row: usize,
    properties: &PropertyProjection,
    property_columns: &[usize],
) -> Result<Vec<u8>, GeoError> {
    let geometry = wkb
        .map(wkb::geometry_json)
        .transpose()?
        .unwrap_or(serde_json::Value::Null);
    let properties =
        if matches!(properties, PropertyProjection::None) || property_columns.is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            row_properties_json(batch, row, property_columns)?
        };
    let feature = serde_json::json!({
        "type": "Feature",
        "id": feature.feature_id.as_deref().unwrap_or(""),
        "feature_ref": feature,
        "geometry": geometry,
        "properties": properties,
    });
    serde_json::to_vec(&feature).map_err(|e| GeoError::Wkb(e.to_string()))
}

fn row_properties_json(
    batch: &RecordBatch,
    row: usize,
    property_columns: &[usize],
) -> Result<serde_json::Value, GeoError> {
    let mut fields = Vec::with_capacity(property_columns.len());
    let mut arrays = Vec::with_capacity(property_columns.len());
    for &idx in property_columns {
        fields.push(batch.schema().field(idx).clone());
        arrays.push(batch.column(idx).slice(row, 1));
    }
    let schema = Arc::new(Schema::new(fields));
    let projected = RecordBatch::try_new(schema, arrays)?;
    let mut buf = Vec::new();
    let mut writer = LineDelimitedWriter::new(&mut buf);
    writer.write(&projected)?;
    writer.finish()?;
    let value: serde_json::Value =
        serde_json::from_slice(buf.trim_ascii()).map_err(|e| GeoError::Wkb(e.to_string()))?;
    Ok(value)
}

fn encode_feature_ref(feature: &FeatureRef) -> Vec<u8> {
    let mut out = Vec::with_capacity(FEATURE_REF_RECORD_LEN);
    out.extend_from_slice(&feature.row_number.to_le_bytes());
    out.extend_from_slice(&feature.row_group.unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(&feature.row_in_group.unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(&feature.part.unwrap_or(u16::MAX).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

fn encode_feature_wkb(feature: &FeatureRef, wkb: &[u8]) -> Vec<u8> {
    let mut out = encode_feature_ref(feature);
    out.extend_from_slice(wkb);
    out
}

/// Decode a fixed-width [`FeatureRef`] payload.
///
/// Returns `None` if the payload is shorter than [`FEATURE_REF_RECORD_LEN`].
pub fn decode_feature_ref_payload(payload: &[u8]) -> Option<FeatureRef> {
    if payload.len() < FEATURE_REF_RECORD_LEN {
        return None;
    }
    let row_number = u64::from_le_bytes(payload[0..8].try_into().ok()?);
    let row_group = decode_u32_option(payload[8..12].try_into().ok()?);
    let row_in_group = decode_u32_option(payload[12..16].try_into().ok()?);
    let part = decode_u16_option(payload[16..18].try_into().ok()?);
    Some(FeatureRef {
        row_number,
        row_group,
        row_in_group,
        part,
        feature_id: None,
    })
}

/// Decode a [`FeatureRef`] followed by WKB bytes.
///
/// This is the payload shape generated by [`PayloadPlan::RowWkb`]. Returns
/// `None` when the fixed feature-ref prefix is truncated.
pub fn decode_feature_wkb_payload(payload: &[u8]) -> Option<(FeatureRef, &[u8])> {
    let feature = decode_feature_ref_payload(payload)?;
    Some((feature, &payload[FEATURE_REF_RECORD_LEN..]))
}

fn decode_u32_option(bytes: [u8; 4]) -> Option<u32> {
    match u32::from_le_bytes(bytes) {
        u32::MAX => None,
        value => Some(value),
    }
}

fn decode_u16_option(bytes: [u8; 2]) -> Option<u16> {
    match u16::from_le_bytes(bytes) {
        u16::MAX => None,
        value => Some(value),
    }
}

fn unique_feature_count(features: &[FeatureRef]) -> usize {
    features
        .iter()
        .map(|feature| feature.row_number)
        .collect::<HashSet<_>>()
        .len()
}

fn entries_may_duplicate_rows(features: &[FeatureRef]) -> bool {
    let mut seen = HashSet::new();
    features
        .iter()
        .any(|feature| !seen.insert(feature.row_number))
}

fn source_fingerprint(meta: &ParquetMetaData) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash = fnv(hash, &meta.file_metadata().num_rows().to_le_bytes());
    for col in meta.file_metadata().schema_descr().columns() {
        hash = fnv(hash, col.path().string().as_bytes());
        hash = fnv(hash, format!("{:?}", col.logical_type_ref()).as_bytes());
    }
    format!("fnv64:{hash:016x}")
}

fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

trait TrimAscii {
    fn trim_ascii(&self) -> &[u8];
}

impl TrimAscii for Vec<u8> {
    fn trim_ascii(&self) -> &[u8] {
        let mut start = 0;
        let mut end = self.len();
        while start < end && self[start].is_ascii_whitespace() {
            start += 1;
        }
        while end > start && self[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        &self[start..end]
    }
}
