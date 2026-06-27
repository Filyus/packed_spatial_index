use packed_spatial_index::{Box2D, Box3D, Index2D, Index3D};
use serde::{Deserialize, Serialize};

/// Source that made a geometry column discoverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryMetadataSource {
    /// GeoParquet `geo` file metadata.
    GeoParquet,
    /// Apache Parquet native `GEOMETRY` or `GEOGRAPHY` logical type.
    ParquetGeospatial,
}

/// Geometry encoding advertised by metadata or discovered from native Parquet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "encoding", rename_all = "snake_case")]
pub enum GeometryEncoding {
    /// WKB bytes.
    Wkb {
        /// WKB dialect.
        flavor: WkbFlavor,
    },
    /// GeoArrow nested/list encoding.
    GeoArrow {
        /// Geometry kind.
        kind: GeometryKind,
        /// Coordinate array layout.
        layout: CoordinateLayout,
    },
    /// Native Parquet `GEOMETRY` logical type.
    ParquetGeometry,
    /// Native Parquet `GEOGRAPHY` logical type.
    ParquetGeography {
        /// Declared geography edge interpolation algorithm.
        algorithm: EdgeAlgorithm,
    },
    /// Unknown or unsupported encoding string.
    Unknown(String),
}

impl GeometryEncoding {
    pub(crate) fn is_wkb_payload(&self) -> bool {
        matches!(
            self,
            GeometryEncoding::Wkb { .. }
                | GeometryEncoding::ParquetGeometry
                | GeometryEncoding::ParquetGeography { .. }
        )
    }

    pub(crate) fn is_native_parquet(&self) -> bool {
        matches!(
            self,
            GeometryEncoding::ParquetGeometry | GeometryEncoding::ParquetGeography { .. }
        )
    }
}

impl std::fmt::Display for GeometryEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryEncoding::Wkb { flavor } => write!(f, "{flavor}"),
            GeometryEncoding::GeoArrow { kind, .. } => write!(f, "{kind}"),
            GeometryEncoding::ParquetGeometry => f.write_str("GEOMETRY"),
            GeometryEncoding::ParquetGeography {
                algorithm: EdgeAlgorithm::Spherical,
            } => f.write_str("GEOGRAPHY(SPHERICAL)"),
            GeometryEncoding::ParquetGeography { algorithm } => {
                write!(f, "GEOGRAPHY({algorithm})")
            }
            GeometryEncoding::Unknown(value) => f.write_str(value),
        }
    }
}

/// WKB dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WkbFlavor {
    /// ISO WKB.
    Iso,
    /// Extended WKB.
    Ewkb,
    /// Not specified.
    Unknown,
}

impl std::fmt::Display for WkbFlavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WkbFlavor::Iso => f.write_str("WKB"),
            WkbFlavor::Ewkb => f.write_str("EWKB"),
            WkbFlavor::Unknown => f.write_str("WKB"),
        }
    }
}

/// Geometry kind for GeoArrow and profile metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryKind {
    /// Point geometry.
    Point,
    /// LineString geometry.
    LineString,
    /// Polygon geometry.
    Polygon,
    /// MultiPoint geometry.
    MultiPoint,
    /// MultiLineString geometry.
    MultiLineString,
    /// MultiPolygon geometry.
    MultiPolygon,
    /// Unknown geometry kind.
    Unknown,
}

impl GeometryKind {
    pub(crate) fn from_geoarrow_encoding(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "point" => Self::Point,
            "linestring" => Self::LineString,
            "polygon" => Self::Polygon,
            "multipoint" => Self::MultiPoint,
            "multilinestring" => Self::MultiLineString,
            "multipolygon" => Self::MultiPolygon,
            _ => Self::Unknown,
        }
    }

    pub(crate) fn list_depth(self) -> Option<usize> {
        match self {
            GeometryKind::Point => Some(0),
            GeometryKind::LineString | GeometryKind::MultiPoint => Some(1),
            GeometryKind::Polygon | GeometryKind::MultiLineString => Some(2),
            GeometryKind::MultiPolygon => Some(3),
            GeometryKind::Unknown => None,
        }
    }
}

impl std::fmt::Display for GeometryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryKind::Point => f.write_str("point"),
            GeometryKind::LineString => f.write_str("linestring"),
            GeometryKind::Polygon => f.write_str("polygon"),
            GeometryKind::MultiPoint => f.write_str("multipoint"),
            GeometryKind::MultiLineString => f.write_str("multilinestring"),
            GeometryKind::MultiPolygon => f.write_str("multipolygon"),
            GeometryKind::Unknown => f.write_str("unknown"),
        }
    }
}

/// Coordinate array layout for GeoArrow encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateLayout {
    /// Separate `x`, `y`, optional `z` / `m` fields.
    Struct,
    /// Fixed-size-list style interleaved coordinates.
    Interleaved,
    /// Layout is not known or not supported.
    Unknown,
}

/// Coordinate dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateDims {
    /// X/Y coordinates.
    Xy,
    /// X/Y/Z coordinates.
    Xyz,
    /// X/Y/M coordinates.
    Xym,
    /// X/Y/Z/M coordinates.
    Xyzm,
    /// Dimensions are not known from metadata.
    Unknown,
}

impl CoordinateDims {
    pub(crate) fn has_z(self) -> bool {
        matches!(self, CoordinateDims::Xyz | CoordinateDims::Xyzm)
    }

    pub(crate) fn has_m(self) -> bool {
        matches!(self, CoordinateDims::Xym | CoordinateDims::Xyzm)
    }

    pub(crate) fn index_dims(self) -> Option<u8> {
        match self {
            CoordinateDims::Xy | CoordinateDims::Xym => Some(2),
            CoordinateDims::Xyz | CoordinateDims::Xyzm => Some(3),
            CoordinateDims::Unknown => None,
        }
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        use CoordinateDims::{Unknown, Xy, Xym, Xyz, Xyzm};
        match (self, other) {
            (Unknown, v) | (v, Unknown) => v,
            (Xyzm, _) | (_, Xyzm) => Xyzm,
            (Xyz, Xym) | (Xym, Xyz) => Xyzm,
            (Xyz, _) | (_, Xyz) => Xyz,
            (Xym, _) | (_, Xym) => Xym,
            (Xy, Xy) => Xy,
        }
    }

    pub(crate) fn from_geometry_types(types: &[String]) -> Self {
        if types.is_empty() {
            return Self::Unknown;
        }
        let mut dims = Self::Xy;
        for ty in types {
            let lower = ty.to_ascii_lowercase();
            let one = if lower.ends_with(" zm") {
                Self::Xyzm
            } else if lower.ends_with(" z") {
                Self::Xyz
            } else if lower.ends_with(" m") {
                Self::Xym
            } else {
                Self::Xy
            };
            dims = dims.merge(one);
        }
        dims
    }
}

impl std::fmt::Display for CoordinateDims {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordinateDims::Xy => f.write_str("XY"),
            CoordinateDims::Xyz => f.write_str("XYZ"),
            CoordinateDims::Xym => f.write_str("XYM"),
            CoordinateDims::Xyzm => f.write_str("XYZM"),
            CoordinateDims::Unknown => f.write_str("unknown"),
        }
    }
}

/// Edge interpolation algorithm for native Parquet `GEOGRAPHY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeAlgorithm {
    /// Spherical interpolation.
    Spherical,
    /// Vincenty interpolation.
    Vincenty,
    /// Thomas interpolation.
    Thomas,
    /// Andoyer interpolation.
    Andoyer,
    /// Karney interpolation.
    Karney,
    /// Unknown interpolation algorithm.
    Unknown,
}

impl std::fmt::Display for EdgeAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeAlgorithm::Spherical => f.write_str("SPHERICAL"),
            EdgeAlgorithm::Vincenty => f.write_str("VINCENTY"),
            EdgeAlgorithm::Thomas => f.write_str("THOMAS"),
            EdgeAlgorithm::Andoyer => f.write_str("ANDOYER"),
            EdgeAlgorithm::Karney => f.write_str("KARNEY"),
            EdgeAlgorithm::Unknown => f.write_str("UNKNOWN"),
        }
    }
}

/// Edge model used by the geometry column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeModel {
    /// Planar coordinate edges.
    Planar,
    /// Great-circle/spherical geography edges.
    Spherical,
    /// Ellipsoidal geography edges.
    Ellipsoidal {
        /// Declared ellipsoidal interpolation algorithm.
        algorithm: EdgeAlgorithm,
    },
    /// Edge model is not known.
    Unknown,
}

/// CRS metadata for a geometry column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CrsInfo {
    /// CRS was present as structured JSON metadata.
    Present {
        /// CRS JSON value.
        value: serde_json::Value,
    },
    /// CRS was present as a string.
    PresentString {
        /// CRS string.
        value: String,
    },
    /// CRS was implied by the format default.
    ImpliedDefault {
        /// Implied CRS value.
        value: String,
    },
    /// Metadata explicitly declares no CRS.
    ExplicitNone,
    /// CRS metadata is absent.
    Missing,
    /// CRS state is unknown.
    Unknown,
}

impl CrsInfo {
    pub(crate) fn as_index_crs(&self) -> Option<String> {
        match self {
            CrsInfo::Present { value } => Some(value.to_string()),
            CrsInfo::PresentString { value } | CrsInfo::ImpliedDefault { value } => {
                Some(value.clone())
            }
            CrsInfo::ExplicitNone | CrsInfo::Missing | CrsInfo::Unknown => None,
        }
    }
}

/// Geometry type names known for a column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryTypeSet {
    /// Type names such as `"Point"`, `"Polygon"`, or `"Point Z"`.
    pub types: Vec<String>,
}

impl GeometryTypeSet {
    pub(crate) fn unknown() -> Self {
        Self { types: Vec::new() }
    }
}

/// Declared dataset or column extent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredExtent {
    /// Extent values as declared by metadata.
    pub values: Vec<f64>,
}

/// Source used to produce per-row bounds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowBoundsSource {
    /// GeoParquet bbox covering column.
    Covering,
    /// Envelope computed from WKB bytes.
    WkbEnvelope,
    /// Envelope computed by scanning GeoArrow arrays.
    GeoArrowScan,
    /// Native Parquet geospatial statistics.
    NativeGeospatialStats,
}

/// Operations supported by a geometry column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnCapabilities {
    /// Column can be scanned into per-feature envelopes.
    pub can_scan_envelopes: bool,
    /// Column can build an in-memory feature index.
    pub can_build_index: bool,
    /// Column can emit `RowWkb` payloads.
    pub can_emit_row_wkb: bool,
    /// Column can emit `FeatureJson` payloads.
    pub can_emit_feature_json: bool,
}

/// File-level geospatial metadata summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileGeoMetadata {
    /// GeoParquet metadata version, if present.
    pub geoparquet_version: Option<String>,
    /// GeoParquet primary column name, if present.
    pub geoparquet_primary_column: Option<String>,
    /// Whether the file contains GeoParquet `geo` metadata.
    pub has_geoparquet_metadata: bool,
}

/// Metadata-only discovery result for a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoDiscovery {
    /// Number of rows in the Parquet file.
    pub num_rows: u64,
    /// File-level metadata.
    pub file_metadata: FileGeoMetadata,
    /// Usable geometry columns.
    pub columns: Vec<GeometryColumnInfo>,
    /// Default selection status.
    pub default_selection: SelectionStatus,
    /// Non-fatal discovery warnings.
    pub warnings: Vec<DiscoveryWarning>,
}

/// Metadata and capabilities for one geometry column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryColumnInfo {
    /// Column name.
    pub name: String,
    /// Metadata source.
    pub source: GeometryMetadataSource,
    /// Geometry encoding.
    pub encoding: GeometryEncoding,
    /// CRS metadata.
    pub crs: CrsInfo,
    /// Edge model.
    pub edges: EdgeModel,
    /// Coordinate dimensions known from metadata.
    pub coordinate_dims: CoordinateDims,
    /// Geometry type names known from metadata.
    pub geometry_types: GeometryTypeSet,
    /// Declared extent, if any.
    pub extent: Option<DeclaredExtent>,
    /// Available row-bounds sources.
    pub row_bounds: Vec<RowBoundsSource>,
    /// Supported operations.
    pub capabilities: ColumnCapabilities,
}

/// Result of resolving a selector or default selection policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SelectionStatus {
    /// A column was selected.
    Selected {
        /// Selected column name.
        column: String,
        /// Why the column was selected.
        reason: GeometrySelectionReason,
    },
    /// Several candidates exist and no safe default is available.
    Ambiguous {
        /// Candidate column names.
        columns: Vec<String>,
    },
    /// Explicit column selection referenced a missing column.
    Missing {
        /// Missing column name.
        column: String,
    },
    /// No usable geometry columns were found.
    None,
}

/// Why a geometry column was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometrySelectionReason {
    /// Explicit selector.
    Explicit,
    /// GeoParquet primary column.
    GeoParquetPrimary,
    /// Only native Parquet geospatial column.
    SingleNativeParquet,
    /// First usable column.
    FirstUsable,
}

/// Non-fatal issue found during discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DiscoveryWarning {
    /// GeoParquet primary column was referenced but not usable.
    GeoParquetPrimaryMissing {
        /// Column name.
        column: String,
    },
    /// GeoParquet column encoding is not supported.
    UnsupportedGeoParquetEncoding {
        /// Column name.
        column: String,
        /// Encoding string.
        encoding: String,
    },
    /// Native Parquet column looked geospatial but did not satisfy reader rules.
    UnsupportedNativeColumn {
        /// Column name.
        column: String,
        /// Reason it was ignored.
        reason: String,
    },
}

/// Concrete selected geometry column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryColumn {
    /// Column name.
    pub name: String,
    /// Full column metadata.
    pub info: GeometryColumnInfo,
}

/// Geometry column selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometrySelector {
    /// GeoParquet primary, else exactly one native Parquet geospatial column.
    Default,
    /// Select by column name.
    Name(String),
    /// Select the GeoParquet primary column.
    GeoParquetPrimary,
    /// Select only if exactly one native Parquet geospatial column exists.
    SingleNativeParquet,
    /// Select the first usable geometry column.
    FirstUsable,
}

/// Requested index dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexDimsRequest {
    /// Infer dimensions.
    Auto,
    /// Force 2D envelopes.
    D2,
    /// Force 3D envelopes.
    D3,
}

/// Handling for null or empty geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullPolicy {
    /// Return an error.
    Error,
    /// Skip the geometry and preserve source row numbers in `FeatureRef`.
    Skip,
}

/// How to handle envelopes crossing the antimeridian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AntimeridianPolicy {
    /// Return an error for antimeridian-crossing envelopes.
    Reject,
    /// Split the feature into two index entries.
    Split,
    /// Expand the longitude interval to the whole world.
    ExpandToWorld,
}

/// Envelope interpretation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvelopePolicy {
    /// Treat coordinates as ordinary planar axes.
    Planar,
    /// Treat x as longitude and apply an antimeridian policy.
    Geographic {
        /// Antimeridian handling.
        antimeridian: AntimeridianPolicy,
    },
}

/// Stable reference back to a source feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureRef {
    /// Absolute source row number.
    pub row_number: u64,
    /// Source row group when known.
    pub row_group: Option<u32>,
    /// Row offset within the row group when known.
    pub row_in_group: Option<u32>,
    /// Split part for duplicated index entries.
    pub part: Option<u16>,
    /// Optional feature identifier.
    pub feature_id: Option<String>,
}

impl FeatureRef {
    pub(crate) fn row(row_number: u64) -> Self {
        Self {
            row_number,
            row_group: None,
            row_in_group: None,
            part: None,
            feature_id: None,
        }
    }
}

/// Payload to attach to converted artifact entries or scan results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PayloadPlan {
    /// Emit no payloads.
    None,
    /// Emit only fixed-width `FeatureRef` records.
    RowRef,
    /// Emit fixed-width `FeatureRef` records followed by WKB bytes.
    RowWkb,
    /// Emit GeoJSON Feature bytes with projected properties.
    FeatureJson {
        /// Property projection.
        properties: PropertyProjection,
    },
}

/// Property projection for `FeatureJson` payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PropertyProjection {
    /// Emit an empty properties object.
    None,
    /// Emit all non-geometry columns.
    AllNonGeometry,
    /// Emit only these property columns.
    Include(Vec<String>),
    /// Emit all non-geometry columns except these.
    Exclude(Vec<String>),
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

/// Request for [`GeoDataset::scan`](crate::GeoDataset::scan).
#[derive(Debug, Clone)]
pub struct ScanRequest {
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Requested envelope dimensionality.
    pub dims: IndexDimsRequest,
    /// Null/empty geometry policy.
    pub nulls: NullPolicy,
    /// Envelope interpretation policy.
    pub envelope: EnvelopePolicy,
    /// Payloads to emit for each scanned entry.
    pub payload: PayloadPlan,
}

impl Default for ScanRequest {
    fn default() -> Self {
        Self {
            selector: GeometrySelector::Default,
            dims: IndexDimsRequest::Auto,
            nulls: NullPolicy::Error,
            envelope: EnvelopePolicy::Planar,
            payload: PayloadPlan::None,
        }
    }
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

/// Profile of a selected geometry column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryProfile {
    /// Selected column name.
    pub column: String,
    /// Metadata source.
    pub source: GeometryMetadataSource,
    /// Geometry encoding.
    pub encoding: GeometryEncoding,
    /// CRS metadata.
    pub crs: CrsInfo,
    /// Edge model.
    pub edges: EdgeModel,
    /// Coordinate dimensions.
    pub coordinate_dims: CoordinateDims,
    /// Geometry types.
    pub geometry_types: GeometryTypeSet,
    /// Declared extent.
    pub extent: Option<DeclaredExtent>,
    /// Row-bounds sources used or available.
    pub row_bounds: Vec<RowBoundsSource>,
    /// Source row count.
    pub num_rows: u64,
}

/// Result of scanning feature envelopes.
#[derive(Debug, Clone)]
pub enum GeometryScan {
    /// 2D scan result.
    D2(GeometryScan2D),
    /// 3D scan result.
    D3(GeometryScan3D),
}

/// 2D scan result.
#[derive(Debug, Clone)]
pub struct GeometryScan2D {
    /// One bounding box per index entry.
    pub boxes: Vec<Box2D>,
    /// Feature reference for each box.
    pub features: Vec<FeatureRef>,
    /// Optional payload for each box.
    pub payloads: Option<Vec<Vec<u8>>>,
    /// Profile of the scanned column.
    pub profile: GeometryProfile,
}

/// 3D scan result.
#[derive(Debug, Clone)]
pub struct GeometryScan3D {
    /// One bounding box per index entry.
    pub boxes: Vec<Box3D>,
    /// Feature reference for each box.
    pub features: Vec<FeatureRef>,
    /// Optional payload for each box.
    pub payloads: Option<Vec<Vec<u8>>>,
    /// Profile of the scanned column.
    pub profile: GeometryProfile,
}

/// In-memory geospatial index.
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
    pub fn search_features(&self, query: Box2D) -> Vec<FeatureRef> {
        self.index
            .search(query)
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect()
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
    pub fn search_features(&self, query: Box3D) -> Vec<FeatureRef> {
        self.index
            .search(query)
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect()
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

/// Geospatial manifest embedded in a converted `PSINDEX` artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoArtifactManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Source format label.
    pub source_format: String,
    /// Stable source metadata fingerprint.
    pub source_fingerprint: String,
    /// Selected geometry column name.
    pub selected_column: String,
    /// CRS metadata.
    pub crs: CrsInfo,
    /// Edge model.
    pub edges: EdgeModel,
    /// Geometry encoding.
    pub encoding: GeometryEncoding,
    /// Coordinate dimensions.
    pub dims: CoordinateDims,
    /// Artifact coordinate precision.
    pub storage_precision: StoragePrecision,
    /// Null policy used during conversion.
    pub null_policy: NullPolicy,
    /// Antimeridian policy used during conversion.
    pub antimeridian_policy: AntimeridianPolicy,
    /// Payload plan used during conversion.
    pub payload_plan: PayloadPlan,
    /// Number of unique source features represented.
    pub feature_count: usize,
    /// Number of index entries.
    pub index_entry_count: usize,
    /// Whether one source row may map to multiple entries.
    pub entries_may_duplicate_rows: bool,
}
