use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BinaryViewArray, Float32Array, Float64Array,
    LargeBinaryArray, StructArray, UInt32Array, new_empty_array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use arrow_json::LineDelimitedWriter;
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use geo::Intersects;
use geozero::wkb::{FromWkb, WkbDialect};
use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index2DF32, Index3DBuilder, Index3DF32};
use parquet::arrow::{
    ProjectionMask,
    arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection},
};
use parquet::basic::{EdgeInterpolationAlgorithm, LogicalType, Type as ParquetPhysicalType};
use parquet::file::metadata::{FileMetaData, ParquetMetaData};
use parquet::file::reader::ChunkReader;
use serde::Deserialize;

use crate::geoarrow;
use crate::geodetic::SphericalRadius;
use crate::manifest;
use crate::validation;
use crate::wkb::{self, GeometryBounds};
use crate::{
    AntimeridianPolicy, BuildRequest, ColumnCapabilities, ConvertRequest, CoordinateDims, CrsInfo,
    DeclaredExtent, DiscoveryWarning, DuplicateFeatureRows, EdgeAlgorithm, EdgeModel,
    EnvelopePolicy, FeatureFilterRequest, FeatureReadOrder, FeatureReadRequest, FeatureRef,
    FeatureRows, FileGeoMetadata, GeoArtifact, GeoArtifactManifest, GeoDiscovery, GeoError,
    GeoIndex, GeoIndex2D, GeoIndex3D, GeoIndexMetadata, GeometryColumn, GeometryColumnInfo,
    GeometryEncoding, GeometryMetadataSource, GeometryProfile, GeometryReadMode, GeometryScan,
    GeometryScan2D, GeometryScan3D, GeometrySelectionReason, GeometrySelector, GeometryTypeSet,
    IndexBuildOptions, IndexDimsRequest, InspectRequest, NativeGeospatialStatsReport,
    NonPlanarExactPolicy, NullPolicy, PayloadPlan, PropertyProjection, QueryGeometry,
    RowBoundsSource, SelectionStatus, SpatialPredicate, StoragePrecision, ValidateRequest,
    ValidationCode, ValidationReport, ValidationSeverity,
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
    let native_stats = validation::native_geospatial_stats(builder.metadata());
    let (discovery, states) = discover_metadata(builder.metadata())?;
    let source_fingerprint = source_fingerprint(builder.metadata());
    let schema_columns = builder
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    Ok(GeoDataset {
        builder: Some(builder),
        discovery,
        states,
        native_stats,
        schema_columns,
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
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{open, BuildRequest, GeoIndex};
///
/// let mut dataset = open(File::open("cities.parquet")?)?;
/// let index = dataset.build(BuildRequest::default())?;
/// match index {
///     GeoIndex::D2(index) => println!("2D features: {}", index.metadata.feature_count),
///     GeoIndex::D3(index) => println!("3D features: {}", index.metadata.feature_count),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct GeoDataset<R: ChunkReader> {
    builder: Option<ParquetRecordBatchReaderBuilder<R>>,
    discovery: GeoDiscovery,
    states: Vec<ColumnState>,
    native_stats: Vec<NativeGeospatialStatsReport>,
    schema_columns: HashSet<String>,
    source_fingerprint: String,
}

impl<R: ChunkReader + 'static> GeoDataset<R> {
    /// Return metadata-only discovery for all usable geometry columns.
    ///
    /// Discovery never scans geometry payloads. Unknown dimensions or geometry
    /// types in this value mean “not declared in metadata”; use
    /// [`GeoDataset::inspect`] with [`InspectRequest::exact`] when exact row
    /// inspection is needed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, SelectionStatus};
    ///
    /// let dataset = open(File::open("cities.parquet")?)?;
    /// if let SelectionStatus::Selected { column, reason } = &dataset.discovery().default_selection {
    ///     println!("default geometry column: {column} ({reason:?})");
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn discovery(&self) -> &GeoDiscovery {
        &self.discovery
    }

    /// Resolve a selector to a concrete geometry column.
    ///
    /// This is a metadata-only operation. It applies the same default selection
    /// policy used by scan/build/convert: GeoParquet primary column first, then
    /// exactly one native Parquet geospatial column.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, GeometrySelector};
    ///
    /// let dataset = open(File::open("cities.parquet")?)?;
    /// let column = dataset.select(GeometrySelector::Name("geometry".to_string()))?;
    /// println!("selected {}", column.name);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, InspectRequest};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let profile = dataset.inspect(InspectRequest {
    ///     exact: true,
    ///     ..InspectRequest::default()
    /// })?;
    /// println!("{}: {}", profile.column, profile.coordinate_dims);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, GeometryScan, ScanRequest};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let scan = dataset.scan(ScanRequest::default())?;
    /// match scan {
    ///     GeometryScan::D2(scan) => println!("2D entries: {}", scan.boxes.len()),
    ///     GeometryScan::D3(scan) => println!("3D entries: {}", scan.boxes.len()),
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn scan(&mut self, req: crate::ScanRequest) -> Result<GeometryScan, GeoError> {
        let state = self.select_state(&req.selector)?.clone();
        self.scan_selected(&state, req)
    }

    /// Build an in-memory [`GeoIndex`] over the selected geometry column.
    ///
    /// The returned index maps candidate hits back to [`FeatureRef`] values
    /// rather than compact item ids. Use [`GeoIndex2D::raw_index`] or
    /// [`GeoIndex3D::raw_index`] when you need direct access to the core index.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, Box2D, BuildRequest, GeoIndex};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let GeoIndex::D2(index) = dataset.build(BuildRequest::default())? else {
    ///     panic!("expected 2D geometry");
    /// };
    /// let features = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0));
    /// println!("candidate features: {}", features.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, ConvertRequest};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let mut bytes = Vec::new();
    /// let artifact = dataset.convert_into(ConvertRequest::default(), &mut bytes)?;
    /// println!("{} bytes, {} features", artifact.bytes_len, artifact.manifest.feature_count);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, ConvertRequest};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let bytes = dataset.convert(ConvertRequest::default())?;
    /// std::fs::write("cities.psindex", bytes)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn convert(&mut self, req: ConvertRequest) -> Result<Vec<u8>, GeoError> {
        let mut out = Vec::new();
        self.convert_into(req, &mut out)?;
        Ok(out)
    }

    /// Filter candidate feature refs by an exact source-geometry predicate.
    ///
    /// This is the post-filter step after an index query. It reads only the
    /// selected source geometry rows, materializes WKB, and evaluates the
    /// requested predicate. Box queries are exact planar XY predicates.
    /// Spherical-radius queries are accepted only for spherical geography and
    /// currently support `Point` / `MultiPoint` geometries.
    ///
    /// Like [`GeoDataset::read_features`], this consumes the dataset reader.
    /// Open a fresh source dataset if you want to read projected rows after
    /// filtering.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{
    ///     open, Box2D, FeatureFilterRequest, FeatureReadRequest,
    /// };
    ///
    /// let candidates = vec![packed_spatial_index_geo::FeatureRef::row_number(42)];
    /// let query = Box2D::new(-10.0, 35.0, 20.0, 60.0);
    ///
    /// let mut filter_source = open(File::open("cities.parquet")?)?;
    /// let exact = filter_source.filter_features(
    ///     FeatureFilterRequest::intersects_box2d(candidates, query),
    /// )?;
    ///
    /// let mut read_source = open(File::open("cities.parquet")?)?;
    /// let rows = read_source.read_features(FeatureReadRequest::from_features(exact))?;
    /// println!("{} exact rows", rows.batch.num_rows());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn filter_features(
        &mut self,
        req: FeatureFilterRequest,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        let state = self.select_state(&req.selector)?.clone();
        if let Some(expected) = &req.expected_source_fingerprint
            && expected != &self.source_fingerprint
        {
            return Err(GeoError::SourceFingerprintMismatch {
                expected: expected.clone(),
                actual: self.source_fingerprint.clone(),
            });
        }
        let query = prepare_filter_query(&state, req.query, req.non_planar)?;

        let rows = self.read_features(FeatureReadRequest {
            features: req.features,
            selector: req.selector,
            properties: PropertyProjection::None,
            geometry: GeometryReadMode::Wkb,
            order: FeatureReadOrder::RequestOrder,
            duplicates: DuplicateFeatureRows::KeepParts,
            expected_source_fingerprint: req.expected_source_fingerprint,
        })?;
        if rows.features.is_empty() {
            return Ok(Vec::new());
        }

        let wkb = binary_column(&rows.batch, "geometry_wkb")?;
        let mut exact = Vec::new();
        for row in 0..rows.features.len() {
            if wkb.is_null(row) {
                continue;
            }
            let Some(geometry) = decode_geo_geometry(wkb.value(row))? else {
                continue;
            };
            if exact_predicate_matches(&geometry, query, req.predicate)? {
                exact.push(rows.features[row].clone());
            }
        }
        Ok(exact)
    }

    /// Read source Parquet rows for feature refs returned by a geo index query.
    ///
    /// This bridges the index back to the original source file: query a
    /// `GeoIndex` or `PSINDEX` artifact for [`FeatureRef`] values, open the
    /// source Parquet file again, and read the projected rows. Parquet is still
    /// read at row-group/page granularity; the index avoids a fresh geometry
    /// envelope scan and narrows the source rows to materialize.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{
    ///     open, Box2D, FeatureReadRequest, GeoArtifactIndex, SliceReader, open_geo_index,
    /// };
    ///
    /// let bytes = std::fs::read("cities.psindex")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    /// let features = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
    ///
    /// let mut source = open(File::open("cities.parquet")?)?;
    /// let rows = source.read_features(FeatureReadRequest::from_features(features))?;
    /// println!("{} rows", rows.batch.num_rows());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn read_features(&mut self, req: FeatureReadRequest) -> Result<FeatureRows, GeoError> {
        let state = self.select_state(&req.selector)?.clone();
        if let Some(expected) = &req.expected_source_fingerprint
            && expected != &self.source_fingerprint
        {
            return Err(GeoError::SourceFingerprintMismatch {
                expected: expected.clone(),
                actual: self.source_fingerprint.clone(),
            });
        }

        let builder = self.take_builder()?;
        let row_groups = row_group_spans(builder.metadata());
        let resolved = resolve_feature_refs(&req.features, &row_groups, self.discovery.num_rows)?;
        let output_features = output_feature_order(&resolved, req.order, req.duplicates);
        let read_rows = unique_source_rows(&output_features);

        let schema = builder.schema().clone();
        let property_roots = property_root_indices(&schema, &state.info.name, &req.properties)?;
        let geometry_root = match req.geometry {
            GeometryReadMode::Omit => None,
            GeometryReadMode::Wkb => Some(root_column_index(&schema, &state.info.name)?),
        };
        let mut read_roots = property_roots.clone();
        if let Some(root) = geometry_root
            && !read_roots.contains(&root)
        {
            read_roots.push(root);
        }
        read_roots.sort_unstable();
        read_roots.dedup();

        if read_rows.is_empty() {
            let batch = finish_feature_batch(
                &state,
                &schema,
                &property_roots,
                req.geometry,
                empty_read_batch(&schema, &read_roots, 0)?,
                &output_features,
            )?;
            return Ok(FeatureRows {
                features: output_features,
                batch,
            });
        }

        let read_plan = source_read_plan(&read_rows, &row_groups)?;
        let mask = ProjectionMask::roots(builder.parquet_schema(), read_roots.iter().copied());
        let reader = builder
            .with_row_groups(read_plan.row_groups)
            .with_row_selection(read_plan.selection)
            .with_projection(mask)
            .build()?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        let read_schema = batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(|| projected_schema(&schema, &read_roots));
        let mut read_batch = concat_batches(&read_schema, &batches)?;

        let take_indices = take_indices_for_features(&output_features, &read_rows)?;
        if needs_take(&take_indices) {
            read_batch = take_batch(&read_batch, &take_indices)?;
        }

        let batch = finish_feature_batch(
            &state,
            &schema,
            &property_roots,
            req.geometry,
            read_batch,
            &output_features,
        )?;
        Ok(FeatureRows {
            features: output_features,
            batch,
        })
    }

    /// Validate compatibility and requested geo operations for this dataset.
    ///
    /// The default request is metadata-only and does not consume rows. Set
    /// [`ValidateRequest::exact`] to scan rows and report scan/payload errors as
    /// validation issues.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, ValidateRequest};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let report = dataset.validate(ValidateRequest::default())?;
    /// println!("validation ok: {}", report.ok);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn validate(&mut self, req: ValidateRequest) -> Result<ValidationReport, GeoError> {
        let selected = self.selection_status(&req.selector);
        let mut issues = Vec::new();
        let mut profile = None;

        self.add_discovery_issues(&mut issues);
        self.add_selection_issues(&selected, &mut issues);

        let state = match &selected {
            SelectionStatus::Selected { column, .. } => self.state_by_name(column).cloned(),
            _ => None,
        };

        if let Some(state) = &state {
            profile = Some(profile_from_state(state, self.discovery.num_rows));
            self.add_capability_issues(state, &req, &mut issues);
            self.add_native_stats_issues(state, &req, &mut issues);
            add_coordinate_aabb_warning(state, &mut issues);

            if req.exact && !validation::has_errors(&issues) {
                match self.scan(crate::ScanRequest {
                    selector: req.selector.clone(),
                    dims: req.dims,
                    nulls: req.nulls,
                    envelope: req.envelope,
                    payload: req.payload.clone(),
                }) {
                    Ok(scan) => {
                        profile = Some(match scan {
                            GeometryScan::D2(scan) => scan.profile,
                            GeometryScan::D3(scan) => scan.profile,
                        });
                    }
                    Err(err) => issues.push(validation::issue(
                        scan_error_severity(&err),
                        scan_error_code(&err),
                        Some(state.info.name.clone()),
                        format!("exact validation scan failed: {err}"),
                    )),
                }
            }

            if let Some(profile) = &profile {
                add_profile_unknown_warnings(profile, &self.native_stats, &mut issues);
            }
        }

        let ok = !validation::has_errors(&issues);
        Ok(ValidationReport {
            discovery: self.discovery.clone(),
            selected,
            profile,
            native_stats: self.native_stats.clone(),
            issues,
            ok,
        })
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

    fn selection_status(&self, selector: &GeometrySelector) -> SelectionStatus {
        match selector {
            GeometrySelector::Default => self.discovery.default_selection.clone(),
            GeometrySelector::Name(name) => {
                if self.state_by_name(name).is_some() {
                    SelectionStatus::Selected {
                        column: name.clone(),
                        reason: GeometrySelectionReason::Explicit,
                    }
                } else {
                    SelectionStatus::Missing {
                        column: name.clone(),
                    }
                }
            }
            GeometrySelector::GeoParquetPrimary => {
                let Some(primary) = &self.discovery.file_metadata.geoparquet_primary_column else {
                    return SelectionStatus::None;
                };
                if self.state_by_name(primary).is_some() {
                    SelectionStatus::Selected {
                        column: primary.clone(),
                        reason: GeometrySelectionReason::GeoParquetPrimary,
                    }
                } else {
                    SelectionStatus::Missing {
                        column: primary.clone(),
                    }
                }
            }
            GeometrySelector::SingleNativeParquet => {
                let native: Vec<_> = self
                    .states
                    .iter()
                    .filter(|state| state.info.encoding.is_native_parquet())
                    .map(|state| state.info.name.clone())
                    .collect();
                match native.as_slice() {
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
            GeometrySelector::FirstUsable => self
                .states
                .iter()
                .find(|state| state.info.capabilities.can_scan_envelopes)
                .map(|state| SelectionStatus::Selected {
                    column: state.info.name.clone(),
                    reason: GeometrySelectionReason::FirstUsable,
                })
                .unwrap_or(SelectionStatus::None),
        }
    }

    fn add_discovery_issues(&self, issues: &mut Vec<crate::ValidationIssue>) {
        for warning in &self.discovery.warnings {
            match warning {
                DiscoveryWarning::GeoParquetPrimaryMissing { column } => {
                    issues.push(validation::issue(
                        ValidationSeverity::Warning,
                        ValidationCode::GeometryColumnNotFound,
                        Some(column.clone()),
                        format!("GeoParquet primary column `{column}` is not usable"),
                    ));
                }
                DiscoveryWarning::UnsupportedGeoParquetEncoding { column, encoding } => {
                    issues.push(validation::issue(
                        ValidationSeverity::Warning,
                        ValidationCode::UnsupportedEncoding,
                        Some(column.clone()),
                        format!(
                            "GeoParquet column `{column}` uses unsupported encoding `{encoding}`"
                        ),
                    ));
                }
                DiscoveryWarning::UnsupportedNativeColumn { column, reason } => {
                    issues.push(validation::issue(
                        ValidationSeverity::Warning,
                        ValidationCode::UnsupportedEncoding,
                        Some(column.clone()),
                        format!("native geospatial column `{column}` is not usable: {reason}"),
                    ));
                }
            }
        }
    }

    fn add_selection_issues(
        &self,
        selected: &SelectionStatus,
        issues: &mut Vec<crate::ValidationIssue>,
    ) {
        match selected {
            SelectionStatus::Selected { .. } => {}
            SelectionStatus::Ambiguous { columns } => issues.push(validation::issue(
                ValidationSeverity::Error,
                ValidationCode::AmbiguousGeometryColumn,
                None,
                format!("multiple geometry columns are usable; choose one explicitly: {columns:?}"),
            )),
            SelectionStatus::Missing { column } => issues.push(validation::issue(
                ValidationSeverity::Error,
                ValidationCode::GeometryColumnNotFound,
                Some(column.clone()),
                format!("geometry column `{column}` was not found or is not usable"),
            )),
            SelectionStatus::None => issues.push(validation::issue(
                ValidationSeverity::Error,
                ValidationCode::NoGeometryColumns,
                None,
                "no usable geometry column was found",
            )),
        }
    }

    fn add_capability_issues(
        &self,
        state: &ColumnState,
        req: &ValidateRequest,
        issues: &mut Vec<crate::ValidationIssue>,
    ) {
        if !state.info.capabilities.can_scan_envelopes {
            issues.push(validation::issue(
                ValidationSeverity::Error,
                ValidationCode::CannotScanEnvelopes,
                Some(state.info.name.clone()),
                format!(
                    "column `{}` cannot produce feature envelopes from {}",
                    state.info.name, state.info.encoding
                ),
            ));
        }

        match &req.payload {
            PayloadPlan::RowWkb if !state.info.capabilities.can_emit_row_wkb => {
                issues.push(validation::issue(
                    ValidationSeverity::Error,
                    ValidationCode::CannotEmitPayload,
                    Some(state.info.name.clone()),
                    format!(
                        "column `{}` cannot emit RowWkb payloads from {}",
                        state.info.name, state.info.encoding
                    ),
                ));
            }
            PayloadPlan::FeatureJson { properties } => {
                if !state.info.capabilities.can_emit_feature_json {
                    issues.push(validation::issue(
                        ValidationSeverity::Error,
                        ValidationCode::CannotEmitPayload,
                        Some(state.info.name.clone()),
                        format!(
                            "column `{}` cannot emit FeatureJson payloads from {}",
                            state.info.name, state.info.encoding
                        ),
                    ));
                }
                self.add_property_projection_issues(properties, issues);
            }
            PayloadPlan::None | PayloadPlan::RowRef | PayloadPlan::RowWkb => {}
        }
    }

    fn add_property_projection_issues(
        &self,
        properties: &PropertyProjection,
        issues: &mut Vec<crate::ValidationIssue>,
    ) {
        let PropertyProjection::Include(include) = properties else {
            return;
        };
        for name in include {
            if !self.schema_columns.contains(name) {
                issues.push(validation::issue(
                    ValidationSeverity::Error,
                    ValidationCode::ProjectedPropertyMissing,
                    Some(name.clone()),
                    format!("FeatureJson property projection references missing column `{name}`"),
                ));
            }
        }
    }

    fn add_native_stats_issues(
        &self,
        state: &ColumnState,
        req: &ValidateRequest,
        issues: &mut Vec<crate::ValidationIssue>,
    ) {
        let Some(stats) = self
            .native_stats
            .iter()
            .find(|stats| stats.column == state.info.name)
        else {
            if state.info.encoding.is_native_parquet() {
                issues.push(validation::issue(
                    ValidationSeverity::Warning,
                    ValidationCode::MissingNativeGeoStats,
                    Some(state.info.name.clone()),
                    format!(
                        "native column `{}` has no row-group geospatial statistics",
                        state.info.name
                    ),
                ));
            }
            return;
        };

        if stats.groups_with_stats == 0 {
            issues.push(validation::issue(
                ValidationSeverity::Warning,
                ValidationCode::MissingNativeGeoStats,
                Some(state.info.name.clone()),
                format!(
                    "column `{}` has no row-group geospatial statistics",
                    state.info.name
                ),
            ));
        } else if stats.groups_with_stats < stats.row_group_count {
            issues.push(validation::issue(
                ValidationSeverity::Warning,
                ValidationCode::MissingNativeGeoStats,
                Some(state.info.name.clone()),
                format!(
                    "column `{}` has geospatial statistics for {}/{} row groups",
                    state.info.name, stats.groups_with_stats, stats.row_group_count
                ),
            ));
        }

        if stats.has_antimeridian_wrap && !allows_antimeridian_wrap(req.envelope) {
            issues.push(validation::issue(
                ValidationSeverity::Warning,
                ValidationCode::AntimeridianWrap,
                Some(state.info.name.clone()),
                format!(
                    "column `{}` has native geospatial stats with xmin > xmax; use geographic split/world antimeridian handling for conservative indexing",
                    state.info.name
                ),
            ));
        }
    }

    fn take_reader(
        &mut self,
    ) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader, GeoError> {
        let builder = self.take_builder()?;
        builder.build().map_err(GeoError::from)
    }

    fn take_builder(&mut self) -> Result<ParquetRecordBatchReaderBuilder<R>, GeoError> {
        self.builder.take().ok_or(GeoError::DatasetConsumed)
    }

    fn scan_selected(
        &mut self,
        state: &ColumnState,
        req: crate::ScanRequest,
    ) -> Result<GeometryScan, GeoError> {
        let row_groups = self
            .builder
            .as_ref()
            .map(|builder| row_group_spans(builder.metadata()))
            .ok_or(GeoError::DatasetConsumed)?;
        let batches = self.take_reader()?;
        let mut entries = Vec::new();
        let mut detected_dims = state.info.coordinate_dims;
        let collect_lons = matches!(req.envelope, EnvelopePolicy::Geographic { .. });
        let want_payload = !matches!(req.payload, PayloadPlan::None);
        let mut row_base = 0u64;
        let mut row_group_cursor = 0usize;

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
                let (row_group, row_in_group) =
                    row_group_for_row(row_number, &row_groups, &mut row_group_cursor)?;
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
                let feature = FeatureRef::row_in_group(row_number, row_group, row_in_group);
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

fn add_profile_unknown_warnings(
    profile: &GeometryProfile,
    native_stats: &[NativeGeospatialStatsReport],
    issues: &mut Vec<crate::ValidationIssue>,
) {
    let stats = native_stats
        .iter()
        .find(|stats| stats.column == profile.column);
    if profile.coordinate_dims == CoordinateDims::Unknown {
        issues.push(validation::issue(
            ValidationSeverity::Warning,
            ValidationCode::UnknownDimensions,
            Some(profile.column.clone()),
            format!(
                "column `{}` has unknown coordinate dimensions in metadata/statistics",
                profile.column
            ),
        ));
    }
    if profile.geometry_types.types.is_empty()
        && stats.is_none_or(|stats| stats.groups_with_types == 0)
    {
        issues.push(validation::issue(
            ValidationSeverity::Warning,
            ValidationCode::UnknownDimensions,
            Some(profile.column.clone()),
            format!(
                "column `{}` has unknown geometry type metadata/statistics",
                profile.column
            ),
        ));
    }
}

fn add_coordinate_aabb_warning(state: &ColumnState, issues: &mut Vec<crate::ValidationIssue>) {
    let warn = matches!(
        state.info.encoding,
        GeometryEncoding::ParquetGeography { .. }
    ) || matches!(
        state.info.edges,
        EdgeModel::Spherical | EdgeModel::Ellipsoidal { .. } | EdgeModel::Unknown
    );
    if warn {
        issues.push(validation::issue(
            ValidationSeverity::Warning,
            ValidationCode::GeographyCoordinateAabb,
            Some(state.info.name.clone()),
            format!(
                "column `{}` is indexed as coordinate axis-aligned bounding boxes; exact spherical/ellipsoidal predicates are not evaluated",
                state.info.name
            ),
        ));
    }
}

fn allows_antimeridian_wrap(policy: EnvelopePolicy) -> bool {
    matches!(
        policy,
        EnvelopePolicy::Geographic {
            antimeridian: AntimeridianPolicy::Split | AntimeridianPolicy::ExpandToWorld
        }
    )
}

fn scan_error_severity(_err: &GeoError) -> ValidationSeverity {
    ValidationSeverity::Error
}

fn scan_error_code(err: &GeoError) -> ValidationCode {
    match err {
        GeoError::PropertyColumnNotFound(_) => ValidationCode::ProjectedPropertyMissing,
        GeoError::UnsupportedEncoding(_) => ValidationCode::UnsupportedEncoding,
        GeoError::AmbiguousGeometryColumn { .. } => ValidationCode::AmbiguousGeometryColumn,
        GeoError::GeometryColumnNotFound(_) => ValidationCode::GeometryColumnNotFound,
        GeoError::NoGeometryColumn => ValidationCode::NoGeometryColumns,
        GeoError::Antimeridian { .. } => ValidationCode::AntimeridianWrap,
        _ => ValidationCode::ExactScanFailed,
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
            dims = dims.merge(validation::coordinate_dims_from_wkb_type(ty));
        }
    }
    if saw_stats {
        dims
    } else {
        CoordinateDims::Unknown
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

fn reject_non_planar_exact(
    state: &ColumnState,
    policy: NonPlanarExactPolicy,
) -> Result<(), GeoError> {
    if matches!(policy, NonPlanarExactPolicy::TreatAsPlanar) {
        return Ok(());
    }
    if matches!(
        state.info.encoding,
        GeometryEncoding::ParquetGeography { .. }
    ) || !matches!(state.info.edges, EdgeModel::Planar)
    {
        return Err(GeoError::NonPlanarExactPredicate {
            column: state.info.name.clone(),
            edges: state.info.edges,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum PreparedFilterQuery {
    Box2D(Box2D),
    SphericalRadius(SphericalRadius),
}

fn prepare_filter_query(
    state: &ColumnState,
    query: QueryGeometry,
    non_planar: NonPlanarExactPolicy,
) -> Result<PreparedFilterQuery, GeoError> {
    match query {
        QueryGeometry::Box2D(bbox) => {
            reject_non_planar_exact(state, non_planar)?;
            Ok(PreparedFilterQuery::Box2D(bbox))
        }
        QueryGeometry::SphericalRadius {
            lon,
            lat,
            radius_metres,
        } => {
            let compatible_native = !matches!(
                state.info.encoding,
                GeometryEncoding::ParquetGeography { .. }
            ) || matches!(
                state.info.encoding,
                GeometryEncoding::ParquetGeography {
                    algorithm: EdgeAlgorithm::Spherical
                }
            );
            if !matches!(state.info.edges, EdgeModel::Spherical) || !compatible_native {
                return Err(GeoError::NonSphericalExactPredicate {
                    column: state.info.name.clone(),
                    edges: state.info.edges,
                });
            }
            Ok(PreparedFilterQuery::SphericalRadius(SphericalRadius::new(
                lon,
                lat,
                radius_metres,
            )?))
        }
    }
}

fn decode_geo_geometry(bytes: &[u8]) -> Result<Option<geo_types::Geometry<f64>>, GeoError> {
    let mut cursor = Cursor::new(bytes);
    match geo_types::Geometry::<f64>::from_wkb(&mut cursor, WkbDialect::Wkb) {
        Ok(geometry) => Ok(Some(geometry)),
        Err(err) => {
            let msg = err.to_string();
            let lower = msg.to_ascii_lowercase();
            if lower.contains("empty") || lower.contains("missing geometry") {
                Ok(None)
            } else {
                Err(GeoError::Wkb(msg))
            }
        }
    }
}

fn exact_predicate_matches(
    geometry: &geo_types::Geometry<f64>,
    query: PreparedFilterQuery,
    predicate: SpatialPredicate,
) -> Result<bool, GeoError> {
    match (query, predicate) {
        (PreparedFilterQuery::Box2D(bbox), SpatialPredicate::Intersects) => {
            let rect = geo_types::Rect::new(
                geo_types::Coord {
                    x: bbox.min_x,
                    y: bbox.min_y,
                },
                geo_types::Coord {
                    x: bbox.max_x,
                    y: bbox.max_y,
                },
            );
            Ok(geometry.intersects(&rect))
        }
        (PreparedFilterQuery::SphericalRadius(query), SpatialPredicate::Intersects) => {
            spherical_radius_matches(geometry, query)
        }
    }
}

fn spherical_radius_matches(
    geometry: &geo_types::Geometry<f64>,
    query: SphericalRadius,
) -> Result<bool, GeoError> {
    match geometry {
        geo_types::Geometry::Point(point) => Ok(query.contains_point(point.x(), point.y())),
        geo_types::Geometry::MultiPoint(points) => Ok(points
            .iter()
            .any(|point| query.contains_point(point.x(), point.y()))),
        geo_types::Geometry::Line(_) => {
            Err(GeoError::UnsupportedGeodeticGeometry("Line".to_string()))
        }
        geo_types::Geometry::LineString(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "LineString".to_string(),
        )),
        geo_types::Geometry::Polygon(_) => {
            Err(GeoError::UnsupportedGeodeticGeometry("Polygon".to_string()))
        }
        geo_types::Geometry::MultiLineString(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "MultiLineString".to_string(),
        )),
        geo_types::Geometry::MultiPolygon(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "MultiPolygon".to_string(),
        )),
        geo_types::Geometry::GeometryCollection(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "GeometryCollection".to_string(),
        )),
        geo_types::Geometry::Rect(_) => {
            Err(GeoError::UnsupportedGeodeticGeometry("Rect".to_string()))
        }
        geo_types::Geometry::Triangle(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "Triangle".to_string(),
        )),
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

#[derive(Debug, Clone, Copy)]
struct RowGroupSpan {
    index: u32,
    start: u64,
    len: u64,
}

impl RowGroupSpan {
    fn end(self) -> u64 {
        self.start + self.len
    }
}

#[derive(Debug, Clone)]
struct ResolvedFeature {
    feature: FeatureRef,
    row_group: u32,
    row_in_group: u32,
    original_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SourceRow {
    row_number: u64,
    row_group: u32,
    row_in_group: u32,
}

struct SourceReadPlan {
    row_groups: Vec<usize>,
    selection: RowSelection,
}

fn row_group_spans(meta: &ParquetMetaData) -> Vec<RowGroupSpan> {
    let mut start = 0u64;
    meta.row_groups()
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let len = group.num_rows().max(0) as u64;
            let span = RowGroupSpan {
                index: index as u32,
                start,
                len,
            };
            start += len;
            span
        })
        .collect()
}

fn row_group_for_row(
    row_number: u64,
    spans: &[RowGroupSpan],
    cursor: &mut usize,
) -> Result<(u32, u32), GeoError> {
    while *cursor + 1 < spans.len() && row_number >= spans[*cursor].end() {
        *cursor += 1;
    }
    let Some(span) = spans.get(*cursor).copied() else {
        return Err(GeoError::FeatureRowOutOfBounds {
            row_number,
            num_rows: 0,
        });
    };
    if row_number >= span.start && row_number < span.end() {
        return Ok((span.index, (row_number - span.start) as u32));
    }
    Err(GeoError::FeatureRowOutOfBounds {
        row_number,
        num_rows: spans.last().map(|span| span.end()).unwrap_or(0),
    })
}

fn resolve_feature_refs(
    features: &[FeatureRef],
    spans: &[RowGroupSpan],
    num_rows: u64,
) -> Result<Vec<ResolvedFeature>, GeoError> {
    features
        .iter()
        .enumerate()
        .map(|(original_index, feature)| {
            if feature.row_number >= num_rows {
                return Err(GeoError::FeatureRowOutOfBounds {
                    row_number: feature.row_number,
                    num_rows,
                });
            }
            let (row_group, row_in_group) = position_for_row(feature.row_number, spans)?;
            if let (Some(given_group), Some(given_row)) = (feature.row_group, feature.row_in_group)
            {
                let Some(span) = spans.get(given_group as usize).copied() else {
                    return Err(GeoError::FeatureRowPositionMismatch {
                        row_number: feature.row_number,
                        row_group: given_group,
                        row_in_group: given_row,
                    });
                };
                if given_row as u64 >= span.len
                    || span.start + given_row as u64 != feature.row_number
                {
                    return Err(GeoError::FeatureRowPositionMismatch {
                        row_number: feature.row_number,
                        row_group: given_group,
                        row_in_group: given_row,
                    });
                }
            }
            let mut resolved = feature.clone();
            resolved.row_group = Some(row_group);
            resolved.row_in_group = Some(row_in_group);
            Ok(ResolvedFeature {
                feature: resolved,
                row_group,
                row_in_group,
                original_index,
            })
        })
        .collect()
}

fn position_for_row(row_number: u64, spans: &[RowGroupSpan]) -> Result<(u32, u32), GeoError> {
    let mut lo = 0usize;
    let mut hi = spans.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let span = spans[mid];
        if row_number < span.start {
            hi = mid;
        } else if row_number >= span.end() {
            lo = mid + 1;
        } else {
            return Ok((span.index, (row_number - span.start) as u32));
        }
    }
    Err(GeoError::FeatureRowOutOfBounds {
        row_number,
        num_rows: spans.last().map(|span| span.end()).unwrap_or(0),
    })
}

fn output_feature_order(
    resolved: &[ResolvedFeature],
    order: FeatureReadOrder,
    duplicates: DuplicateFeatureRows,
) -> Vec<FeatureRef> {
    let mut selected: Vec<&ResolvedFeature> = Vec::new();
    let mut seen = HashSet::new();
    for feature in resolved {
        if matches!(duplicates, DuplicateFeatureRows::DedupRows)
            && !seen.insert(feature.feature.row_number)
        {
            continue;
        }
        selected.push(feature);
    }
    match order {
        FeatureReadOrder::SourceOrder => selected.sort_by_key(|feature| {
            (
                feature.feature.row_number,
                feature.row_group,
                feature.row_in_group,
                feature.original_index,
            )
        }),
        FeatureReadOrder::RequestOrder => selected.sort_by_key(|feature| feature.original_index),
    }
    selected
        .into_iter()
        .map(|feature| feature.feature.clone())
        .collect()
}

fn unique_source_rows(features: &[FeatureRef]) -> Vec<SourceRow> {
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for feature in features {
        if seen.insert(feature.row_number) {
            rows.push(SourceRow {
                row_number: feature.row_number,
                row_group: feature.row_group.expect("resolved feature row group"),
                row_in_group: feature.row_in_group.expect("resolved feature row offset"),
            });
        }
    }
    rows.sort_by_key(|row| (row.row_number, row.row_group, row.row_in_group));
    rows
}

fn source_read_plan(
    rows: &[SourceRow],
    spans: &[RowGroupSpan],
) -> Result<SourceReadPlan, GeoError> {
    let mut row_groups: Vec<usize> = rows.iter().map(|row| row.row_group as usize).collect();
    row_groups.sort_unstable();
    row_groups.dedup();

    let mut selected_offsets = HashMap::new();
    let mut total_rows = 0u64;
    for &group in &row_groups {
        let span = spans
            .get(group)
            .ok_or_else(|| GeoError::Metadata(format!("row group {group} not found")))?;
        selected_offsets.insert(span.index, total_rows);
        total_rows += span.len;
    }

    let mut offsets = Vec::with_capacity(rows.len());
    for row in rows {
        let base = *selected_offsets.get(&row.row_group).ok_or_else(|| {
            GeoError::Metadata(format!("row group {} not selected", row.row_group))
        })?;
        offsets.push(
            usize::try_from(base + row.row_in_group as u64).map_err(|_| {
                GeoError::Metadata("row selection offset does not fit usize".to_string())
            })?,
        );
    }
    offsets.sort_unstable();
    offsets.dedup();

    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for offset in offsets {
        if let Some(last) = ranges.last_mut()
            && last.end == offset
        {
            last.end += 1;
            continue;
        }
        ranges.push(offset..offset + 1);
    }
    let total_rows = usize::try_from(total_rows)
        .map_err(|_| GeoError::Metadata("row selection length does not fit usize".to_string()))?;
    Ok(SourceReadPlan {
        row_groups,
        selection: RowSelection::from_consecutive_ranges(ranges.into_iter(), total_rows),
    })
}

fn property_root_indices(
    schema: &Schema,
    geometry_column: &str,
    properties: &PropertyProjection,
) -> Result<Vec<usize>, GeoError> {
    let names: Vec<_> = schema.fields().iter().map(|field| field.name()).collect();
    match properties {
        PropertyProjection::None => Ok(Vec::new()),
        PropertyProjection::AllNonGeometry => Ok(names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| (*name != geometry_column).then_some(idx))
            .collect()),
        PropertyProjection::Include(include) => include
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|candidate| *candidate == name)
                    .ok_or_else(|| GeoError::PropertyColumnNotFound(name.clone()))
            })
            .collect(),
        PropertyProjection::Exclude(exclude) => Ok(names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                (*name != geometry_column && !exclude.iter().any(|excluded| excluded == *name))
                    .then_some(idx)
            })
            .collect()),
    }
}

fn root_column_index(schema: &Schema, name: &str) -> Result<usize, GeoError> {
    schema
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .ok_or_else(|| GeoError::GeometryColumnNotFound(name.to_string()))
}

fn projected_schema(schema: &Schema, roots: &[usize]) -> Arc<Schema> {
    Arc::new(Schema::new(
        roots
            .iter()
            .map(|&idx| schema.field(idx).clone())
            .collect::<Vec<_>>(),
    ))
}

fn empty_read_batch(
    source_schema: &Schema,
    roots: &[usize],
    row_count: usize,
) -> Result<RecordBatch, GeoError> {
    let schema = projected_schema(source_schema, roots);
    let columns = schema
        .fields()
        .iter()
        .map(|field| new_empty_array(field.data_type()))
        .collect();
    record_batch_with_len(schema, columns, row_count)
}

fn take_indices_for_features(
    features: &[FeatureRef],
    read_rows: &[SourceRow],
) -> Result<Vec<u32>, GeoError> {
    let positions: HashMap<_, _> = read_rows
        .iter()
        .enumerate()
        .map(|(idx, row)| (row.row_number, idx as u32))
        .collect();
    features
        .iter()
        .map(|feature| {
            positions.get(&feature.row_number).copied().ok_or_else(|| {
                GeoError::Metadata(format!(
                    "feature row {} was not read from source",
                    feature.row_number
                ))
            })
        })
        .collect()
}

fn needs_take(indices: &[u32]) -> bool {
    indices
        .iter()
        .enumerate()
        .any(|(idx, &value)| value as usize != idx)
}

fn take_batch(batch: &RecordBatch, indices: &[u32]) -> Result<RecordBatch, GeoError> {
    if batch.num_columns() == 0 {
        return record_batch_with_len(batch.schema(), Vec::new(), indices.len());
    }
    let indices = UInt32Array::from(indices.to_vec());
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(batch.schema(), columns).map_err(GeoError::from)
}

fn finish_feature_batch(
    state: &ColumnState,
    source_schema: &Schema,
    property_roots: &[usize],
    geometry: GeometryReadMode,
    read_batch: RecordBatch,
    features: &[FeatureRef],
) -> Result<RecordBatch, GeoError> {
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for &root in property_roots {
        let field = source_schema.field(root).clone();
        let name = field.name().clone();
        let column = read_batch
            .column_by_name(&name)
            .ok_or_else(|| GeoError::Metadata(format!("projected column `{name}` was not read")))?;
        fields.push(field);
        columns.push(column.clone());
    }
    if matches!(geometry, GeometryReadMode::Wkb) {
        fields.push(Field::new("geometry_wkb", DataType::Binary, false));
        columns.push(geometry_wkb_array(state, &read_batch, features)?);
    }
    record_batch_with_len(Arc::new(Schema::new(fields)), columns, features.len())
}

fn record_batch_with_len(
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    row_count: usize,
) -> Result<RecordBatch, GeoError> {
    if columns.is_empty() {
        RecordBatch::try_new_with_options(
            schema,
            columns,
            &RecordBatchOptions::new().with_row_count(Some(row_count)),
        )
        .map_err(GeoError::from)
    } else {
        RecordBatch::try_new(schema, columns).map_err(GeoError::from)
    }
}

fn geometry_wkb_array(
    state: &ColumnState,
    batch: &RecordBatch,
    features: &[FeatureRef],
) -> Result<ArrayRef, GeoError> {
    let geom = batch
        .column_by_name(&state.info.name)
        .ok_or_else(|| GeoError::GeometryColumnNotFound(state.info.name.clone()))?
        .clone();
    let binary = needs_binary(&state.info.encoding)
        .then(|| binary_column(batch, &state.info.name))
        .transpose()?;
    let mut builder = BinaryBuilder::new();
    for (row, feature) in features.iter().enumerate() {
        let wkb = wkb_payload_one_row(state, &geom, binary.as_ref(), row)?.ok_or_else(|| {
            GeoError::NullGeometry {
                row: feature.row_number as usize,
            }
        })?;
        builder.append_value(wkb);
    }
    Ok(Arc::new(builder.finish()))
}

fn wkb_payload_one_row(
    state: &ColumnState,
    geom: &ArrayRef,
    binary: Option<&WkbCol<'_>>,
    row: usize,
) -> Result<Option<Vec<u8>>, GeoError> {
    if geom.is_null(row) {
        return Ok(None);
    }
    if let Some(binary) = binary {
        if binary.is_null(row) {
            return Ok(None);
        }
        return Ok(Some(binary.value(row).to_vec()));
    }
    let GeometryEncoding::GeoArrow { kind, .. } = state.info.encoding else {
        return Err(GeoError::UnsupportedEncoding(
            state.info.encoding.to_string(),
        ));
    };
    Ok(geoarrow::scan_row(geom, kind, state.info.coordinate_dims, row, false)?.map(|row| row.wkb))
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
