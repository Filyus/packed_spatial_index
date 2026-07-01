use std::ops::ControlFlow;

use packed_spatial_index::{
    EARTH_RADIUS_M, Index2D, Index2DBuilder, Index2DF32, Index3D, Index3DBuilder, Index3DF32,
    Point2D, Point3D, haversine_distance_2d,
};
use serde::{Deserialize, Serialize};

use crate::manifest;
use crate::payload;
use crate::{
    AntimeridianPolicy, EnvelopePolicy, FEATURE_JSON_CONTENT_TYPE, FEATURE_REF_CONTENT_TYPE,
    FEATURE_REF_RECORD_LEN, FEATURE_WKB_CONTENT_TYPE, FeatureRef, GeoArtifactManifest, GeoError,
    GeoQuery2D, GeoQuery3D, GeometryMetadataSource, GeometryProfile, GeometryScan,
    GeometrySelector, IndexDimsRequest, NullPolicy, PayloadPlan,
};

pub(crate) fn builder_2d(count: usize, opts: &IndexBuildOptions) -> Index2DBuilder {
    let mut builder = Index2DBuilder::new(count);
    if let Some(node_size) = opts.node_size {
        builder = builder.node_size(node_size);
    }
    builder = builder.parallel(opts.parallel);
    builder
}

pub(crate) fn builder_3d(count: usize, opts: &IndexBuildOptions) -> Index3DBuilder {
    let mut builder = Index3DBuilder::new(count);
    if let Some(node_size) = opts.node_size {
        builder = builder.node_size(node_size);
    }
    builder = builder.parallel(opts.parallel);
    builder
}

/// Collect up to `max_results` `(FeatureRef, distance)` pairs from a
/// nondecreasing-distance core neighbor visitor, stopping the traversal once
/// `max_results` have been found. `visit` should call the core index's own
/// `visit_neighbors`/`visit_neighbors_metric` with the closure it's given.
fn collect_nearest(
    features: &[FeatureRef],
    max_results: usize,
    visit: impl FnOnce(&mut dyn FnMut(usize, f64) -> ControlFlow<()>),
) -> Vec<(FeatureRef, f64)> {
    if max_results == 0 {
        return Vec::new();
    }
    let mut results = Vec::with_capacity(max_results);
    visit(&mut |id, distance| {
        if let Some(feature) = features.get(id) {
            results.push((feature.clone(), distance));
        }
        if results.len() >= max_results {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    results
}

/// Apply the shared interleaved/crs/payload configuration to a freshly built
/// `$index.serialize()` builder and write it to `$out`. The four index
/// serializers (2D/3D x f64/f32) duck-type the same builder methods but share
/// no common trait in the core crate, so this is a macro rather than a
/// generic function.
macro_rules! configure_and_write {
    ($index:expr, $interleaved:expr, $payload:expr, $crs:expr, $out:expr) => {{
        let mut serializer = $index.serialize();
        if $interleaved && $payload.is_some() {
            serializer = serializer.interleaved();
        }
        if let Some(crs) = &$crs {
            serializer = serializer.crs(crs);
        }
        if let Some(payload) = $payload {
            serializer = serializer
                .payloads(payload)
                .content_type(content_type_for_payload(payload));
        }
        serializer.to_bytes_into($out)?;
    }};
}

pub(crate) fn serialize_2d(
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
            configure_and_write!(index, interleaved, payload, crs, out)
        }
        StoragePrecision::F32 => {
            let index: Index2DF32 = builder.finish_f32()?;
            configure_and_write!(index, interleaved, payload, crs, out)
        }
    }
    Ok(())
}

pub(crate) fn serialize_3d(
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
            configure_and_write!(index, interleaved, payload, crs, out)
        }
        StoragePrecision::F32 => {
            let index: Index3DF32 = builder.finish_f32()?;
            configure_and_write!(index, interleaved, payload, crs, out)
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

pub(crate) fn artifact_manifest(
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

/// Coordinate storage precision for converted artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoragePrecision {
    /// Store coordinates as `f64`.
    F64,
    /// Store coordinates as `f32`; queries return a conservative superset.
    F32,
}

/// Options passed to the core index builder.
///
/// # Example
///
/// ```
/// use packed_spatial_index_geo::{IndexBuildOptions, StoragePrecision};
///
/// let opts = IndexBuildOptions {
///     precision: StoragePrecision::F32,
///     ..IndexBuildOptions::default()
/// };
/// assert_eq!(opts.precision, StoragePrecision::F32);
/// ```
#[derive(Debug, Clone)]
pub struct IndexBuildOptions {
    /// Optional node size override.
    pub node_size: Option<usize>,
    /// Whether to use parallel build when supported by the core crate.
    pub parallel: bool,
    /// In-memory index coordinate precision. Selects between [`GeoIndex::D2`]/
    /// [`GeoIndex::D3`] (`F64`, the default) and [`GeoIndex::D2F32`]/
    /// [`GeoIndex::D3F32`] (`F32`, half the box memory; `Box2D`/`Box3D` queries
    /// only — a `GeoQuery2D::Polygon` or `GeoQuery3D::Frustum3D` query is
    /// rejected against an `F32` index, since the underlying core index only
    /// implements a box-based search, not the generic query trait those
    /// variants need).
    pub precision: StoragePrecision,
}

impl Default for IndexBuildOptions {
    fn default() -> Self {
        Self {
            node_size: None,
            parallel: true,
            precision: StoragePrecision::F64,
        }
    }
}

/// Request for [`GeoDataset::build`](crate::GeoDataset::build).
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{
///     open, BuildRequest, GeometrySelector, IndexDimsRequest, NullPolicy,
/// };
///
/// let mut dataset = open(File::open("cities.parquet")?)?;
/// let index = dataset.build(BuildRequest {
///     selector: GeometrySelector::Name("geometry".to_string()),
///     dims: IndexDimsRequest::D2,
///     nulls: NullPolicy::Skip,
///     ..BuildRequest::default()
/// })?;
/// # let _ = index;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct BuildRequest {
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Requested index dimensionality.
    pub dims: IndexDimsRequest,
    /// Null/empty geometry policy.
    pub nulls: NullPolicy,
    /// Envelope interpretation policy.
    pub envelope: EnvelopePolicy,
    /// Core build options.
    pub build: IndexBuildOptions,
}

impl Default for BuildRequest {
    fn default() -> Self {
        Self {
            selector: GeometrySelector::Default,
            dims: IndexDimsRequest::Auto,
            nulls: NullPolicy::Error,
            envelope: EnvelopePolicy::Planar,
            build: IndexBuildOptions::default(),
        }
    }
}

/// Request for [`GeoDataset::convert`](crate::GeoDataset::convert) and
/// [`GeoDataset::convert_into`](crate::GeoDataset::convert_into).
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{open, ConvertRequest, StoragePrecision};
///
/// let mut dataset = open(File::open("cities.parquet")?)?;
/// let bytes = dataset.convert(ConvertRequest {
///     precision: StoragePrecision::F32,
///     ..ConvertRequest::default()
/// })?;
/// std::fs::write("cities.psindex", bytes)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ConvertRequest {
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Requested index dimensionality.
    pub dims: IndexDimsRequest,
    /// Null/empty geometry policy.
    pub nulls: NullPolicy,
    /// Envelope interpretation policy.
    pub envelope: EnvelopePolicy,
    /// Core build options.
    pub build: IndexBuildOptions,
    /// Artifact coordinate precision.
    pub precision: StoragePrecision,
    /// Payload plan.
    pub payload: PayloadPlan,
    /// Whether to use the stream-optimized interleaved artifact layout.
    pub interleaved: bool,
}

impl Default for ConvertRequest {
    fn default() -> Self {
        Self {
            selector: GeometrySelector::Default,
            dims: IndexDimsRequest::Auto,
            nulls: NullPolicy::Skip,
            envelope: EnvelopePolicy::Planar,
            build: IndexBuildOptions::default(),
            precision: StoragePrecision::F64,
            payload: PayloadPlan::RowWkb,
            interleaved: true,
        }
    }
}

/// In-memory geospatial index.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{open, Box2D, BuildRequest, GeoIndex};
///
/// let mut dataset = open(File::open("cities.parquet")?)?;
/// match dataset.build(BuildRequest::default())? {
///     GeoIndex::D2(index) => {
///         let hits = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
///         println!("{} candidate features", hits.len());
///     }
///     GeoIndex::D3(_) => println!("3D index"),
///     GeoIndex::D2F32(_) | GeoIndex::D3F32(_) => {
///         println!("f32-precision index (only built when requested)")
///     }
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub enum GeoIndex {
    /// 2D index, `f64` storage.
    D2(GeoIndex2D),
    /// 3D index, `f64` storage.
    D3(GeoIndex3D),
    /// 2D index, `f32` storage (see [`IndexBuildOptions::precision`]).
    D2F32(GeoIndex2DF32),
    /// 3D index, `f32` storage (see [`IndexBuildOptions::precision`]).
    D3F32(GeoIndex3DF32),
}

impl GeoIndex {
    /// Build an in-memory index from an already-computed [`GeometryScan`].
    ///
    /// [`GeoDataset::build`](crate::GeoDataset::build) and
    /// [`GeoDataset::convert_into`](crate::GeoDataset::convert_into) each call
    /// [`GeoDataset::scan`](crate::GeoDataset::scan) internally, so producing
    /// both an in-memory index and a converted artifact from one
    /// `GeoDataset` normally scans the source twice. Call
    /// [`GeoDataset::scan`](crate::GeoDataset::scan) once instead, then build
    /// both outputs from the result with this function and
    /// [`GeoArtifact::from_scan`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{
    ///     open, ConvertRequest, GeoArtifact, GeoIndex, IndexBuildOptions, PayloadPlan,
    ///     ScanRequest,
    /// };
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let scan = dataset.scan(ScanRequest {
    ///     payload: PayloadPlan::RowWkb,
    ///     ..ScanRequest::default()
    /// })?;
    /// let index = GeoIndex::from_scan(&scan, &IndexBuildOptions::default())?;
    ///
    /// let mut bytes = Vec::new();
    /// let artifact = GeoArtifact::from_scan(
    ///     &scan,
    ///     &ConvertRequest::default(),
    ///     dataset.source_fingerprint(),
    ///     &mut bytes,
    /// )?;
    ///
    /// let entries = match &index {
    ///     GeoIndex::D2(index) => index.features.len(),
    ///     GeoIndex::D3(index) => index.features.len(),
    ///     GeoIndex::D2F32(index) => index.features.len(),
    ///     GeoIndex::D3F32(index) => index.features.len(),
    /// };
    /// println!("{entries} entries, {} artifact bytes", artifact.bytes_len);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_scan(scan: &GeometryScan, opts: &IndexBuildOptions) -> Result<GeoIndex, GeoError> {
        Ok(match scan {
            GeometryScan::D2(scan) => {
                let mut builder = builder_2d(scan.boxes.len(), opts);
                for bbox in &scan.boxes {
                    builder.add(*bbox);
                }
                let metadata = GeoIndexMetadata {
                    profile: scan.profile.clone(),
                    feature_count: payload::unique_feature_count(&scan.features),
                    index_entry_count: scan.boxes.len(),
                    entries_may_duplicate_rows: payload::entries_may_duplicate_rows(&scan.features),
                };
                match opts.precision {
                    StoragePrecision::F64 => GeoIndex::D2(GeoIndex2D {
                        index: builder.finish()?,
                        features: scan.features.clone(),
                        metadata,
                    }),
                    StoragePrecision::F32 => GeoIndex::D2F32(GeoIndex2DF32 {
                        index: builder.finish_f32()?,
                        features: scan.features.clone(),
                        metadata,
                    }),
                }
            }
            GeometryScan::D3(scan) => {
                let mut builder = builder_3d(scan.boxes.len(), opts);
                for bbox in &scan.boxes {
                    builder.add(*bbox);
                }
                let metadata = GeoIndexMetadata {
                    profile: scan.profile.clone(),
                    feature_count: payload::unique_feature_count(&scan.features),
                    index_entry_count: scan.boxes.len(),
                    entries_may_duplicate_rows: payload::entries_may_duplicate_rows(&scan.features),
                };
                match opts.precision {
                    StoragePrecision::F64 => GeoIndex::D3(GeoIndex3D {
                        index: builder.finish()?,
                        features: scan.features.clone(),
                        metadata,
                    }),
                    StoragePrecision::F32 => GeoIndex::D3F32(GeoIndex3DF32 {
                        index: builder.finish_f32()?,
                        features: scan.features.clone(),
                        metadata,
                    }),
                }
            }
        })
    }
}

/// 2D in-memory geospatial index.
pub struct GeoIndex2D {
    /// Core index.
    pub index: Index2D,
    /// Feature reference per compact item id.
    pub features: Vec<FeatureRef>,
    /// Build metadata.
    pub metadata: GeoIndexMetadata,
}

impl GeoIndex2D {
    /// Search and return source feature references.
    pub fn search_features<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        let boxes = query.into().candidate_boxes_2d()?;
        let mut features = Vec::new();
        // A single core search yields each item id at most once, so duplicates
        // only arise across multiple candidate boxes (e.g. antimeridian splits).
        // Fast-path the common single-box query with no dedup bookkeeping; for
        // multi-box queries dedup by item id in O(1) via a set rather than the
        // former O(K^2) `Vec::contains` scan.
        if boxes.len() == 1 {
            for id in self.index.search(boxes[0]) {
                if let Some(feature) = self.features.get(id) {
                    features.push(feature.clone());
                }
            }
        } else {
            let mut seen = std::collections::HashSet::new();
            for bbox in boxes {
                for id in self.index.search(bbox) {
                    if seen.insert(id)
                        && let Some(feature) = self.features.get(id)
                    {
                        features.push(feature.clone());
                    }
                }
            }
        }
        Ok(features)
    }

    /// Up to `max_results` nearest features to `point`, planar Euclidean
    /// distance, nearest first, paired with each hit's squared distance.
    ///
    /// For lon/lat data, prefer [`nearest_features_haversine`][Self::nearest_features_haversine] —
    /// Euclidean distance on raw longitude/latitude degrees is not a
    /// geographic distance (a degree of longitude shrinks toward the poles).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, BuildRequest, GeoIndex, Point2D};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let GeoIndex::D2(index) = dataset.build(BuildRequest::default())? else {
    ///     panic!("expected a 2D index");
    /// };
    /// for (feature, dist_sq) in index.nearest_features(Point2D::new(13.4, 52.5), 3) {
    ///     println!("row {}: squared distance {dist_sq}", feature.row_number);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn nearest_features(&self, point: Point2D, max_results: usize) -> Vec<(FeatureRef, f64)> {
        collect_nearest(&self.features, max_results, |visitor| {
            let _ = self.index.visit_neighbors(point, f64::INFINITY, visitor);
        })
    }

    /// Up to `max_results` nearest features to a lon/lat query point by
    /// great-circle (haversine) distance in metres, nearest first, paired
    /// with each hit's distance in metres.
    ///
    /// Use for geographic data (`x` = longitude, `y` = latitude in degrees);
    /// see [`nearest_features`][Self::nearest_features] for planar Euclidean
    /// distance instead.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, BuildRequest, GeoIndex};
    ///
    /// let mut dataset = open(File::open("cities.parquet")?)?;
    /// let GeoIndex::D2(index) = dataset.build(BuildRequest::default())? else {
    ///     panic!("expected a 2D index");
    /// };
    /// for (feature, metres) in index.nearest_features_haversine(13.0, 52.4, 1, f64::INFINITY) {
    ///     println!("row {}: {metres:.0}m away", feature.row_number);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn nearest_features_haversine(
        &self,
        lon: f64,
        lat: f64,
        max_results: usize,
        max_distance_metres: f64,
    ) -> Vec<(FeatureRef, f64)> {
        collect_nearest(&self.features, max_results, |visitor| {
            let _ = self.index.visit_neighbors_metric(
                |bx| haversine_distance_2d((lon, lat), bx, EARTH_RADIUS_M),
                max_distance_metres,
                visitor,
            );
        })
    }

    /// Access the underlying core index.
    pub fn raw_index(&self) -> &Index2D {
        &self.index
    }
}

/// 2D in-memory geospatial index, `f32`-precision storage.
///
/// Built via [`IndexBuildOptions::precision`] set to
/// [`StoragePrecision::F32`](StoragePrecision::F32) — half the box memory of
/// [`GeoIndex2D`], at the cost of only supporting [`GeoQuery2D::Box2D`]
/// queries: the underlying core index (`Index2DF32`) takes a plain `Box2D`,
/// not the generic query trait a [`GeoQuery2D::Polygon`] or
/// [`GeoQuery2D::SphericalRadius`] search needs — a permanent limitation of
/// the f32-storage core index, not a TODO.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{
///     open, Box2D, BuildRequest, GeoIndex, IndexBuildOptions, StoragePrecision,
/// };
///
/// let mut dataset = open(File::open("cities.parquet")?)?;
/// let GeoIndex::D2F32(index) = dataset.build(BuildRequest {
///     build: IndexBuildOptions {
///         precision: StoragePrecision::F32,
///         ..IndexBuildOptions::default()
///     },
///     ..BuildRequest::default()
/// })?
/// else {
///     panic!("expected an f32 2D index");
/// };
/// let hits = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
/// println!("{} candidate features", hits.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct GeoIndex2DF32 {
    /// Core index.
    pub index: Index2DF32,
    /// Feature reference per compact item id.
    pub features: Vec<FeatureRef>,
    /// Build metadata.
    pub metadata: GeoIndexMetadata,
}

impl GeoIndex2DF32 {
    /// Search and return source feature references.
    ///
    /// Only [`GeoQuery2D::Box2D`] is supported; any other query variant
    /// returns [`GeoError::UnsupportedArtifact`].
    pub fn search_features<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        let GeoQuery2D::Box2D(bbox) = query.into() else {
            return Err(GeoError::UnsupportedArtifact(
                "f32-precision in-memory index only supports GeoQuery2D::Box2D queries; \
                 the underlying core index takes a plain Box2D, not the generic query \
                 trait a Polygon or SphericalRadius search needs"
                    .to_string(),
            ));
        };
        Ok(self
            .index
            .search(bbox)
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect())
    }

    /// Up to `max_results` nearest features to `point`, planar Euclidean
    /// distance, nearest first, paired with each hit's squared distance.
    ///
    /// Unlike [`search_features`](Self::search_features), this is not
    /// restricted to a query shape — the underlying core kNN search works on
    /// `f32`-precision storage the same way it does on `f64`. There is no
    /// haversine variant here: the core custom-metric kNN entry point
    /// (`neighbors_metric`) is not implemented for `f32`-precision indexes.
    pub fn nearest_features(&self, point: Point2D, max_results: usize) -> Vec<(FeatureRef, f64)> {
        collect_nearest(&self.features, max_results, |visitor| {
            let _ = self.index.visit_neighbors(point, f64::INFINITY, visitor);
        })
    }

    /// Access the underlying core index.
    pub fn raw_index(&self) -> &Index2DF32 {
        &self.index
    }
}

/// 3D in-memory geospatial index.
pub struct GeoIndex3D {
    /// Core index.
    pub index: Index3D,
    /// Feature reference per compact item id.
    pub features: Vec<FeatureRef>,
    /// Build metadata.
    pub metadata: GeoIndexMetadata,
}

impl GeoIndex3D {
    /// Search and return source feature references.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{Box3D, BuildRequest, GeoIndex, IndexDimsRequest, open};
    ///
    /// let mut dataset = open(File::open("elevations.parquet")?)?;
    /// let GeoIndex::D3(index) = dataset.build(BuildRequest {
    ///     dims: IndexDimsRequest::D3,
    ///     ..BuildRequest::default()
    /// })?
    /// else {
    ///     panic!("expected a 3D index");
    /// };
    /// let hits = index.search_features(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))?;
    /// println!("{} candidate features", hits.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_features<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        let ids = match query.into() {
            GeoQuery3D::Box3D(bbox) => self.index.search(bbox),
            GeoQuery3D::Frustum3D(frustum) => self.index.search(&frustum),
        };
        Ok(ids
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect())
    }

    /// Up to `max_results` nearest features to `point`, planar Euclidean
    /// distance, nearest first, paired with each hit's squared distance.
    ///
    /// There is no haversine variant for 3D data — core has no built-in
    /// geographic distance metric that also accounts for a third (elevation)
    /// coordinate.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use packed_spatial_index_geo::{open, BuildRequest, GeoIndex, IndexDimsRequest, Point3D};
    ///
    /// let mut dataset = open(File::open("elevations.parquet")?)?;
    /// let GeoIndex::D3(index) = dataset.build(BuildRequest {
    ///     dims: IndexDimsRequest::D3,
    ///     ..BuildRequest::default()
    /// })?
    /// else {
    ///     panic!("expected a 3D index");
    /// };
    /// for (feature, dist_sq) in index.nearest_features(Point3D::new(13.4, 52.5, 34.0), 3) {
    ///     println!("row {}: squared distance {dist_sq}", feature.row_number);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn nearest_features(&self, point: Point3D, max_results: usize) -> Vec<(FeatureRef, f64)> {
        collect_nearest(&self.features, max_results, |visitor| {
            let _ = self.index.visit_neighbors(point, f64::INFINITY, visitor);
        })
    }

    /// Access the underlying core index.
    pub fn raw_index(&self) -> &Index3D {
        &self.index
    }
}

/// 3D in-memory geospatial index, `f32`-precision storage.
///
/// Built via [`IndexBuildOptions::precision`] set to
/// [`StoragePrecision::F32`](StoragePrecision::F32) — half the box memory of
/// [`GeoIndex3D`], at the cost of only supporting [`GeoQuery3D::Box3D`]
/// queries: the underlying core index (`Index3DF32`) takes a plain `Box3D`,
/// not the generic query trait a non-box query needs — a permanent
/// limitation of the f32-storage core index, not a TODO.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{
///     open, Box3D, BuildRequest, GeoIndex, IndexBuildOptions, IndexDimsRequest, StoragePrecision,
/// };
///
/// let mut dataset = open(File::open("elevations.parquet")?)?;
/// let GeoIndex::D3F32(index) = dataset.build(BuildRequest {
///     dims: IndexDimsRequest::D3,
///     build: IndexBuildOptions {
///         precision: StoragePrecision::F32,
///         ..IndexBuildOptions::default()
///     },
///     ..BuildRequest::default()
/// })?
/// else {
///     panic!("expected an f32 3D index");
/// };
/// let hits = index.search_features(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))?;
/// println!("{} candidate features", hits.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct GeoIndex3DF32 {
    /// Core index.
    pub index: Index3DF32,
    /// Feature reference per compact item id.
    pub features: Vec<FeatureRef>,
    /// Build metadata.
    pub metadata: GeoIndexMetadata,
}

impl GeoIndex3DF32 {
    /// Search and return source feature references.
    ///
    /// Only [`GeoQuery3D::Box3D`] is supported; a [`GeoQuery3D::Frustum3D`]
    /// query returns [`GeoError::UnsupportedArtifact`].
    pub fn search_features<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        let GeoQuery3D::Box3D(bbox) = query.into() else {
            return Err(GeoError::UnsupportedArtifact(
                "f32-precision in-memory index only supports GeoQuery3D::Box3D queries; \
                 the underlying core index takes a plain Box3D, not the generic query \
                 trait a Frustum3D search needs"
                    .to_string(),
            ));
        };
        Ok(self
            .index
            .search(bbox)
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect())
    }

    /// Up to `max_results` nearest features to `point`, planar Euclidean
    /// distance, nearest first, paired with each hit's squared distance.
    ///
    /// Unlike [`search_features`](Self::search_features), this is not
    /// restricted to a query shape. There is no haversine variant: the core
    /// custom-metric kNN entry point is not implemented for `f32`-precision
    /// indexes.
    pub fn nearest_features(&self, point: Point3D, max_results: usize) -> Vec<(FeatureRef, f64)> {
        collect_nearest(&self.features, max_results, |visitor| {
            let _ = self.index.visit_neighbors(point, f64::INFINITY, visitor);
        })
    }

    /// Access the underlying core index.
    pub fn raw_index(&self) -> &Index3DF32 {
        &self.index
    }
}

/// Metadata for a built in-memory index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoIndexMetadata {
    /// Profile of the indexed column.
    pub profile: GeometryProfile,
    /// Number of unique source features represented.
    pub feature_count: usize,
    /// Number of index entries.
    pub index_entry_count: usize,
    /// Whether one source row may map to multiple entries.
    pub entries_may_duplicate_rows: bool,
}

/// Result metadata from converting to a `PSINDEX` artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoArtifact {
    /// Manifest embedded in the artifact.
    pub manifest: GeoArtifactManifest,
    /// Length of the generated byte buffer.
    pub bytes_len: usize,
}

impl GeoArtifact {
    /// Convert an already-computed [`GeometryScan`] into a `PSINDEX`
    /// artifact. Existing contents of `out` are replaced.
    ///
    /// Pairs with [`GeoIndex::from_scan`] to build both an in-memory index
    /// and a converted artifact from one
    /// [`GeoDataset::scan`](crate::GeoDataset::scan) call. `source_fingerprint`
    /// comes from
    /// [`GeoDataset::source_fingerprint`](crate::GeoDataset::source_fingerprint).
    ///
    /// # Example
    ///
    /// See [`GeoIndex::from_scan`].
    pub fn from_scan(
        scan: &GeometryScan,
        req: &ConvertRequest,
        source_fingerprint: &str,
        out: &mut Vec<u8>,
    ) -> Result<GeoArtifact, GeoError> {
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
                    req,
                    payload::unique_feature_count(&scan.features),
                    scan.boxes.len(),
                    payload::entries_may_duplicate_rows(&scan.features),
                    source_fingerprint,
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
                    req,
                    payload::unique_feature_count(&scan.features),
                    scan.boxes.len(),
                    payload::entries_may_duplicate_rows(&scan.features),
                    source_fingerprint,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::{GeometryScan2D, GeometryScan3D};
    use crate::{
        CoordinateDims, CrsInfo, EdgeModel, GeometryMetadataSource, GeometryTypeSet, WkbFlavor,
    };
    use packed_spatial_index::{Box2D, Box3D};

    fn test_profile() -> GeometryProfile {
        GeometryProfile {
            column: "geometry".to_string(),
            source: GeometryMetadataSource::GeoParquet,
            encoding: crate::GeometryEncoding::Wkb {
                flavor: WkbFlavor::Iso,
            },
            crs: CrsInfo::Missing,
            edges: EdgeModel::Planar,
            coordinate_dims: CoordinateDims::Xy,
            geometry_types: GeometryTypeSet::unknown(),
            extent: None,
            row_bounds: Vec::new(),
            num_rows: 3,
        }
    }

    fn scan_2d() -> GeometryScan {
        GeometryScan::D2(GeometryScan2D {
            boxes: vec![
                Box2D::new(0.0, 0.0, 1.0, 1.0),
                Box2D::new(5.0, 5.0, 6.0, 6.0),
                Box2D::new(10.0, 10.0, 11.0, 11.0),
            ],
            features: vec![
                FeatureRef::row_number(0),
                FeatureRef::row_number(1),
                FeatureRef::row_number(2),
            ],
            payloads: None,
            profile: test_profile(),
        })
    }

    fn scan_3d() -> GeometryScan {
        GeometryScan::D3(GeometryScan3D {
            boxes: vec![
                Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0),
            ],
            features: vec![FeatureRef::row_number(0), FeatureRef::row_number(1)],
            payloads: None,
            profile: test_profile(),
        })
    }

    #[test]
    fn from_scan_default_precision_is_f64() {
        let index = GeoIndex::from_scan(&scan_2d(), &IndexBuildOptions::default()).unwrap();
        assert!(matches!(index, GeoIndex::D2(_)));
        let index = GeoIndex::from_scan(&scan_3d(), &IndexBuildOptions::default()).unwrap();
        assert!(matches!(index, GeoIndex::D3(_)));
    }

    #[test]
    fn from_scan_f32_precision_yields_f32_variant_2d() {
        let opts = IndexBuildOptions {
            precision: StoragePrecision::F32,
            ..IndexBuildOptions::default()
        };
        let index = GeoIndex::from_scan(&scan_2d(), &opts).unwrap();
        let GeoIndex::D2F32(index) = index else {
            panic!("expected GeoIndex::D2F32");
        };
        let hits = index
            .search_features(Box2D::new(-1.0, -1.0, 2.0, 2.0))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].row_number, 0);
    }

    #[test]
    fn from_scan_f32_precision_yields_f32_variant_3d() {
        let opts = IndexBuildOptions {
            precision: StoragePrecision::F32,
            ..IndexBuildOptions::default()
        };
        let index = GeoIndex::from_scan(&scan_3d(), &opts).unwrap();
        let GeoIndex::D3F32(index) = index else {
            panic!("expected GeoIndex::D3F32");
        };
        let hits = index
            .search_features(Box3D::new(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].row_number, 0);
    }

    #[test]
    fn f32_2d_index_rejects_non_box_query() {
        let opts = IndexBuildOptions {
            precision: StoragePrecision::F32,
            ..IndexBuildOptions::default()
        };
        let GeoIndex::D2F32(index) = GeoIndex::from_scan(&scan_2d(), &opts).unwrap() else {
            panic!("expected GeoIndex::D2F32");
        };
        let polygon = GeoQuery2D::polygon(geo_types::Polygon::new(
            geo_types::LineString::from(vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.0, 0.0),
            ]),
            vec![],
        ));
        let err = index.search_features(polygon).unwrap_err();
        assert!(matches!(err, GeoError::UnsupportedArtifact(_)));
    }

    #[test]
    fn frustum3d_query_tightens_over_bounding_box_on_f64_3d_index() {
        use packed_spatial_index::Frustum3D;

        // A frustum that widens with z: at z=0, x/y in [-1,1]; at z=10, x/y in [-3,3].
        let frustum = Frustum3D::from_planes([
            [1.0, 0.0, 0.2, 1.0],
            [-1.0, 0.0, 0.2, 1.0],
            [0.0, 1.0, 0.2, 1.0],
            [0.0, -1.0, 0.2, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 10.0],
        ]);

        let scan = GeometryScan::D3(GeometryScan3D {
            boxes: vec![
                Box3D::new(0.0, 0.0, 0.0, 0.5, 0.5, 0.5),
                Box3D::new(2.0, 2.0, 0.0, 2.5, 2.5, 0.5),
                Box3D::new(-2.5, -2.5, 9.0, -2.0, -2.0, 9.5),
            ],
            features: vec![
                FeatureRef::row_number(0),
                FeatureRef::row_number(1),
                FeatureRef::row_number(2),
            ],
            payloads: None,
            profile: test_profile(),
        });

        let GeoIndex::D3(index) =
            GeoIndex::from_scan(&scan, &IndexBuildOptions::default()).unwrap()
        else {
            panic!("expected GeoIndex::D3");
        };

        let bbox = frustum.bounding_box().expect("non-degenerate frustum");
        let mut bbox_rows: Vec<u64> = index
            .search_features(bbox)
            .unwrap()
            .iter()
            .map(|f| f.row_number)
            .collect();
        bbox_rows.sort_unstable();
        assert_eq!(
            bbox_rows,
            vec![0, 1, 2],
            "bounding box covers all three boxes"
        );

        let mut frustum_rows: Vec<u64> = index
            .search_features(frustum)
            .unwrap()
            .iter()
            .map(|f| f.row_number)
            .collect();
        frustum_rows.sort_unstable();
        assert_eq!(
            frustum_rows,
            vec![0, 2],
            "frustum search excludes the box outside the narrow end, unlike its bounding box"
        );
    }

    #[test]
    fn f32_3d_index_rejects_frustum_query() {
        use packed_spatial_index::Frustum3D;

        let opts = IndexBuildOptions {
            precision: StoragePrecision::F32,
            ..IndexBuildOptions::default()
        };
        let GeoIndex::D3F32(index) = GeoIndex::from_scan(&scan_3d(), &opts).unwrap() else {
            panic!("expected GeoIndex::D3F32");
        };
        let frustum = Frustum3D::from_planes([
            [1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]);
        let err = index.search_features(frustum).unwrap_err();
        assert!(matches!(err, GeoError::UnsupportedArtifact(_)));
    }

    #[test]
    fn nearest_features_orders_by_planar_distance() {
        let GeoIndex::D2(index) =
            GeoIndex::from_scan(&scan_2d(), &IndexBuildOptions::default()).unwrap()
        else {
            panic!("expected GeoIndex::D2");
        };
        let hits = index.nearest_features(Point2D::new(0.0, 0.0), 2);
        let rows: Vec<u64> = hits.iter().map(|(f, _)| f.row_number).collect();
        assert_eq!(rows, vec![0, 1], "nearest boxes first, farthest last");
        // Distances are nondecreasing.
        assert!(hits[0].1 <= hits[1].1);
    }

    #[test]
    fn nearest_features_max_results_zero_is_empty() {
        let GeoIndex::D2(index) =
            GeoIndex::from_scan(&scan_2d(), &IndexBuildOptions::default()).unwrap()
        else {
            panic!("expected GeoIndex::D2");
        };
        assert!(index.nearest_features(Point2D::new(0.0, 0.0), 0).is_empty());
    }

    #[test]
    fn nearest_features_haversine_orders_by_great_circle_distance() {
        // Berlin and Paris, matching core's own neighbors_metric doc example.
        let scan = GeometryScan::D2(GeometryScan2D {
            boxes: vec![
                Box2D::from_point(packed_spatial_index::Point2D::new(13.40, 52.52)),
                Box2D::from_point(packed_spatial_index::Point2D::new(2.35, 48.86)),
            ],
            features: vec![FeatureRef::row_number(0), FeatureRef::row_number(1)],
            payloads: None,
            profile: test_profile(),
        });
        let GeoIndex::D2(index) =
            GeoIndex::from_scan(&scan, &IndexBuildOptions::default()).unwrap()
        else {
            panic!("expected GeoIndex::D2");
        };

        let hits = index.nearest_features_haversine(13.0, 52.4, 1, f64::INFINITY);
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].0.row_number, 0,
            "Berlin is nearer to the query point"
        );

        // A tight cutoff excludes even the nearest city.
        assert!(
            index
                .nearest_features_haversine(13.0, 52.4, 1, 1.0)
                .is_empty()
        );
    }

    #[test]
    fn f32_nearest_features_matches_f64_ordering() {
        let opts = IndexBuildOptions {
            precision: StoragePrecision::F32,
            ..IndexBuildOptions::default()
        };
        let GeoIndex::D2F32(index) = GeoIndex::from_scan(&scan_2d(), &opts).unwrap() else {
            panic!("expected GeoIndex::D2F32");
        };
        let hits = index.nearest_features(Point2D::new(0.0, 0.0), 2);
        let rows: Vec<u64> = hits.iter().map(|(f, _)| f.row_number).collect();
        assert_eq!(rows, vec![0, 1]);

        let GeoIndex::D3F32(index) = GeoIndex::from_scan(&scan_3d(), &opts).unwrap() else {
            panic!("expected GeoIndex::D3F32");
        };
        let hits = index.nearest_features(Point3D::new(0.0, 0.0, 0.0), 1);
        assert_eq!(hits[0].0.row_number, 0);
    }
}
