use packed_spatial_index::{
    Index2D, Index2DBuilder, Index2DF32, Index3D, Index3DBuilder, Index3DF32,
};
use serde::{Deserialize, Serialize};

use crate::{
    AntimeridianPolicy, EnvelopePolicy, FEATURE_JSON_CONTENT_TYPE, FEATURE_REF_CONTENT_TYPE,
    FEATURE_REF_RECORD_LEN, FEATURE_WKB_CONTENT_TYPE, FeatureRef, GeoArtifactManifest, GeoError,
    GeoQuery2D, GeoQuery3D, GeometryMetadataSource, GeometryProfile, GeometrySelector,
    IndexDimsRequest, NullPolicy, PayloadPlan,
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
#[derive(Debug, Clone)]
pub struct IndexBuildOptions {
    /// Optional node size override.
    pub node_size: Option<usize>,
    /// Whether to use parallel build when supported by the core crate.
    pub parallel: bool,
}

impl Default for IndexBuildOptions {
    fn default() -> Self {
        Self {
            node_size: None,
            parallel: true,
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
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub enum GeoIndex {
    /// 2D index.
    D2(GeoIndex2D),
    /// 3D index.
    D3(GeoIndex3D),
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

    /// Access the underlying core index.
    pub fn raw_index(&self) -> &Index2D {
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
    pub fn search_features<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .index
            .search(query.into().candidate_box_3d())
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect())
    }

    /// Access the underlying core index.
    pub fn raw_index(&self) -> &Index3D {
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
