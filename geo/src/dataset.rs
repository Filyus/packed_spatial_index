use std::collections::HashSet;

use arrow_select::concat::concat_batches;
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use parquet::file::reader::ChunkReader;
use serde::{Deserialize, Serialize};

use crate::PropertyProjection;
use crate::build::{BuildRequest, GeoArtifact, GeoIndex};
use crate::discovery::{self, ColumnState};
use crate::feature_read::{self, FeatureRows};
use crate::filter;
use crate::payload;
use crate::scan::{self, ScanRequestForInspect};
use crate::validation;
use crate::{
    AntimeridianPolicy, ConvertRequest, CoordinateDims, DiscoveryWarning, DuplicateFeatureRows,
    EdgeModel, EnvelopePolicy, FeatureFilterRequest, FeatureReadOrder, FeatureReadRequest,
    FeatureRef, GeoDiscovery, GeoError, GeometryColumn, GeometryEncoding, GeometryProfile,
    GeometryReadMode, GeometryScan, GeometrySelectionReason, GeometrySelector, IndexDimsRequest,
    NativeGeospatialStatsReport, NullPolicy, PayloadPlan, ScanRequest, SelectionStatus,
    ValidationCode, ValidationReport, ValidationSeverity,
};

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
    let (discovery, states) = discovery::discover_metadata(builder.metadata())?;
    let source_fingerprint = payload::source_fingerprint(builder.metadata());
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
///     GeoIndex::D2F32(_) | GeoIndex::D3F32(_) => println!("f32-precision index"),
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

    /// Return a stable fingerprint of the source Parquet file's metadata.
    ///
    /// Pass it as `expected_source_fingerprint` on [`FeatureFilterRequest`] or
    /// [`FeatureReadRequest`] to detect a source/artifact mismatch when
    /// reading features back from a different `GeoDataset` session than the
    /// one that built the index or artifact. Also needed by
    /// [`GeoArtifact::from_scan`] when converting a [`GeoDataset::scan`]
    /// result directly, without going through [`GeoDataset::convert_into`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::open;
    ///
    /// let dataset = open(File::open("cities.parquet")?)?;
    /// println!("source fingerprint: {}", dataset.source_fingerprint());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn source_fingerprint(&self) -> &str {
        &self.source_fingerprint
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
        Ok(discovery::profile_from_state(
            &state,
            self.discovery.num_rows,
        ))
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
    pub fn scan(&mut self, req: ScanRequest) -> Result<GeometryScan, GeoError> {
        let state = self.select_state(&req.selector)?.clone();
        scan::scan_selected(self, &state, req)
    }

    /// Build an in-memory [`GeoIndex`] over the selected geometry column.
    ///
    /// The returned index maps candidate hits back to [`FeatureRef`] values
    /// rather than compact item ids. Use
    /// [`GeoIndex2D::raw_index`](crate::GeoIndex2D::raw_index) or
    /// [`GeoIndex3D::raw_index`](crate::GeoIndex3D::raw_index) when you need
    /// direct access to the core index.
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
    /// let features = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
    /// println!("candidate features: {}", features.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build(&mut self, req: BuildRequest) -> Result<GeoIndex, GeoError> {
        let scan = self.scan(ScanRequest {
            selector: req.selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: PayloadPlan::None,
        })?;
        GeoIndex::from_scan(&scan, &req.build)
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
        let scan = self.scan(ScanRequest {
            selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: req.payload.clone(),
        })?;
        GeoArtifact::from_scan(&scan, &req, &self.source_fingerprint, out)
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
    /// requested predicate. Box and polygon queries are exact planar XY
    /// predicates. Spherical-radius queries are accepted only for spherical
    /// geography and currently support `Point` / `MultiPoint` geometries.
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
    ///     FeatureFilterRequest::intersects(candidates, query),
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
        let query = filter::prepare_filter_query(
            &state.info.encoding,
            state.info.edges,
            &state.info.name,
            req.query,
            req.non_planar,
        )?;

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

        let wkb = scan::binary_column(&rows.batch, "geometry_wkb")?;
        let mut exact = Vec::new();
        for row in 0..rows.features.len() {
            if wkb.is_null(row) {
                continue;
            }
            let Some(geometry) = filter::decode_geo_geometry(wkb.value(row))? else {
                continue;
            };
            if filter::exact_predicate_matches(&geometry, &query, req.predicate)? {
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
        let row_groups = feature_read::row_group_spans(builder.metadata());
        let resolved = feature_read::resolve_feature_refs(
            &req.features,
            &row_groups,
            self.discovery.num_rows,
        )?;
        let output_features =
            feature_read::output_feature_order(&resolved, req.order, req.duplicates);
        let read_rows = feature_read::unique_source_rows(&output_features);

        let schema = builder.schema().clone();
        let property_roots =
            feature_read::property_root_indices(&schema, &state.info.name, &req.properties)?;
        let geometry_root = match req.geometry {
            GeometryReadMode::Omit => None,
            GeometryReadMode::Wkb => {
                Some(feature_read::root_column_index(&schema, &state.info.name)?)
            }
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
            let batch = feature_read::finish_feature_batch(
                &state,
                &schema,
                &property_roots,
                req.geometry,
                feature_read::empty_read_batch(&schema, &read_roots, 0)?,
                &output_features,
            )?;
            return Ok(FeatureRows {
                features: output_features,
                batch,
            });
        }

        let read_plan = feature_read::source_read_plan(&read_rows, &row_groups)?;
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
            .unwrap_or_else(|| feature_read::projected_schema(&schema, &read_roots));
        let mut read_batch = concat_batches(&read_schema, &batches)?;

        let take_indices = feature_read::take_indices_for_features(&output_features, &read_rows)?;
        if feature_read::needs_take(&take_indices) {
            read_batch = feature_read::take_batch(&read_batch, &take_indices)?;
        }

        let batch = feature_read::finish_feature_batch(
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
            profile = Some(discovery::profile_from_state(
                state,
                self.discovery.num_rows,
            ));
            self.add_capability_issues(state, &req, &mut issues);
            self.add_native_stats_issues(state, &req, &mut issues);
            add_coordinate_aabb_warning(state, &mut issues);

            if req.exact && !validation::has_errors(&issues) {
                match self.scan(ScanRequest {
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
        if matches!(req.envelope, EnvelopePolicy::Geographic { .. })
            && state.info.crs.is_known_projected()
        {
            issues.push(validation::issue(
                ValidationSeverity::Error,
                ValidationCode::GeographyCoordinateAabb,
                Some(state.info.name.clone()),
                format!(
                    "column `{}` has a projected CRS; geographic antimeridian handling is only valid for lon/lat coordinates",
                    state.info.name
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

    pub(crate) fn take_builder(&mut self) -> Result<ParquetRecordBatchReaderBuilder<R>, GeoError> {
        self.builder.take().ok_or(GeoError::DatasetConsumed)
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

/// Request for [`GeoDataset::inspect`](crate::GeoDataset::inspect).
#[derive(Debug, Clone)]
pub struct InspectRequest {
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Scan rows when metadata alone cannot provide exact profile details.
    pub exact: bool,
}

impl Default for InspectRequest {
    fn default() -> Self {
        Self {
            selector: GeometrySelector::Default,
            exact: false,
        }
    }
}

/// Request for [`GeoDataset::validate`](crate::GeoDataset::validate).
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{open, ValidateRequest};
///
/// let mut dataset = open(File::open("cities.parquet")?)?;
/// let report = dataset.validate(ValidateRequest {
///     exact: true,
///     ..ValidateRequest::default()
/// })?;
/// assert!(report.ok);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidateRequest {
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Scan rows and validate scan/payload behavior instead of metadata only.
    pub exact: bool,
    /// Requested index dimensionality to validate.
    pub dims: IndexDimsRequest,
    /// Null/empty geometry policy to validate.
    pub nulls: NullPolicy,
    /// Envelope interpretation policy to validate.
    pub envelope: EnvelopePolicy,
    /// Payload plan to validate.
    pub payload: PayloadPlan,
}

impl Default for ValidateRequest {
    fn default() -> Self {
        Self {
            selector: GeometrySelector::Default,
            exact: false,
            dims: IndexDimsRequest::Auto,
            nulls: NullPolicy::Skip,
            envelope: EnvelopePolicy::Planar,
            payload: PayloadPlan::RowWkb,
        }
    }
}
