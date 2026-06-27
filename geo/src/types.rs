use packed_spatial_index::{Box2D, Box3D, Index2D, Index3D};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryMetadataSource {
    GeoParquet,
    ParquetGeospatial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "encoding", rename_all = "snake_case")]
pub enum GeometryEncoding {
    Wkb {
        flavor: WkbFlavor,
    },
    GeoArrow {
        kind: GeometryKind,
        layout: CoordinateLayout,
    },
    ParquetGeometry,
    ParquetGeography {
        algorithm: EdgeAlgorithm,
    },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WkbFlavor {
    Iso,
    Ewkb,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryKind {
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateLayout {
    Struct,
    Interleaved,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateDims {
    Xy,
    Xyz,
    Xym,
    Xyzm,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeAlgorithm {
    Spherical,
    Vincenty,
    Thomas,
    Andoyer,
    Karney,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeModel {
    Planar,
    Spherical,
    Ellipsoidal { algorithm: EdgeAlgorithm },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CrsInfo {
    Present { value: serde_json::Value },
    PresentString { value: String },
    ImpliedDefault { value: String },
    ExplicitNone,
    Missing,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryTypeSet {
    pub types: Vec<String>,
}

impl GeometryTypeSet {
    pub(crate) fn unknown() -> Self {
        Self { types: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredExtent {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowBoundsSource {
    Covering,
    WkbEnvelope,
    GeoArrowScan,
    NativeGeospatialStats,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnCapabilities {
    pub can_scan_envelopes: bool,
    pub can_build_index: bool,
    pub can_emit_row_wkb: bool,
    pub can_emit_feature_json: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileGeoMetadata {
    pub geoparquet_version: Option<String>,
    pub geoparquet_primary_column: Option<String>,
    pub has_geoparquet_metadata: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoDiscovery {
    pub num_rows: u64,
    pub file_metadata: FileGeoMetadata,
    pub columns: Vec<GeometryColumnInfo>,
    pub default_selection: SelectionStatus,
    pub warnings: Vec<DiscoveryWarning>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryColumnInfo {
    pub name: String,
    pub source: GeometryMetadataSource,
    pub encoding: GeometryEncoding,
    pub crs: CrsInfo,
    pub edges: EdgeModel,
    pub coordinate_dims: CoordinateDims,
    pub geometry_types: GeometryTypeSet,
    pub extent: Option<DeclaredExtent>,
    pub row_bounds: Vec<RowBoundsSource>,
    pub capabilities: ColumnCapabilities,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SelectionStatus {
    Selected {
        column: String,
        reason: GeometrySelectionReason,
    },
    Ambiguous {
        columns: Vec<String>,
    },
    Missing {
        column: String,
    },
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometrySelectionReason {
    Explicit,
    GeoParquetPrimary,
    SingleNativeParquet,
    FirstUsable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DiscoveryWarning {
    GeoParquetPrimaryMissing { column: String },
    UnsupportedGeoParquetEncoding { column: String, encoding: String },
    UnsupportedNativeColumn { column: String, reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryColumn {
    pub name: String,
    pub info: GeometryColumnInfo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometrySelector {
    Default,
    Name(String),
    GeoParquetPrimary,
    SingleNativeParquet,
    FirstUsable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexDimsRequest {
    Auto,
    D2,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullPolicy {
    Error,
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AntimeridianPolicy {
    Reject,
    Split,
    ExpandToWorld,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvelopePolicy {
    Planar,
    Geographic { antimeridian: AntimeridianPolicy },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureRef {
    pub row_number: u64,
    pub row_group: Option<u32>,
    pub row_in_group: Option<u32>,
    pub part: Option<u16>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PayloadPlan {
    None,
    RowRef,
    RowWkb,
    FeatureJson { properties: PropertyProjection },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PropertyProjection {
    None,
    AllNonGeometry,
    Include(Vec<String>),
    Exclude(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoragePrecision {
    F64,
    F32,
}

#[derive(Debug, Clone)]
pub struct InspectRequest {
    pub selector: GeometrySelector,
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

#[derive(Debug, Clone)]
pub struct ScanRequest {
    pub selector: GeometrySelector,
    pub dims: IndexDimsRequest,
    pub nulls: NullPolicy,
    pub envelope: EnvelopePolicy,
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

#[derive(Debug, Clone)]
pub struct IndexBuildOptions {
    pub node_size: Option<usize>,
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

#[derive(Debug, Clone)]
pub struct BuildRequest {
    pub selector: GeometrySelector,
    pub dims: IndexDimsRequest,
    pub nulls: NullPolicy,
    pub envelope: EnvelopePolicy,
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

#[derive(Debug, Clone)]
pub struct ConvertRequest {
    pub selector: GeometrySelector,
    pub dims: IndexDimsRequest,
    pub nulls: NullPolicy,
    pub envelope: EnvelopePolicy,
    pub build: IndexBuildOptions,
    pub precision: StoragePrecision,
    pub payload: PayloadPlan,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryProfile {
    pub column: String,
    pub source: GeometryMetadataSource,
    pub encoding: GeometryEncoding,
    pub crs: CrsInfo,
    pub edges: EdgeModel,
    pub coordinate_dims: CoordinateDims,
    pub geometry_types: GeometryTypeSet,
    pub extent: Option<DeclaredExtent>,
    pub row_bounds: Vec<RowBoundsSource>,
    pub num_rows: u64,
}

#[derive(Debug, Clone)]
pub enum GeometryScan {
    D2(GeometryScan2D),
    D3(GeometryScan3D),
}

#[derive(Debug, Clone)]
pub struct GeometryScan2D {
    pub boxes: Vec<Box2D>,
    pub features: Vec<FeatureRef>,
    pub payloads: Option<Vec<Vec<u8>>>,
    pub profile: GeometryProfile,
}

#[derive(Debug, Clone)]
pub struct GeometryScan3D {
    pub boxes: Vec<Box3D>,
    pub features: Vec<FeatureRef>,
    pub payloads: Option<Vec<Vec<u8>>>,
    pub profile: GeometryProfile,
}

pub enum GeoIndex {
    D2(GeoIndex2D),
    D3(GeoIndex3D),
}

pub struct GeoIndex2D {
    pub index: Index2D,
    pub features: Vec<FeatureRef>,
    pub metadata: GeoIndexMetadata,
}

impl GeoIndex2D {
    pub fn search_features(&self, query: Box2D) -> Vec<FeatureRef> {
        self.index
            .search(query)
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect()
    }

    pub fn raw_index(&self) -> &Index2D {
        &self.index
    }
}

pub struct GeoIndex3D {
    pub index: Index3D,
    pub features: Vec<FeatureRef>,
    pub metadata: GeoIndexMetadata,
}

impl GeoIndex3D {
    pub fn search_features(&self, query: Box3D) -> Vec<FeatureRef> {
        self.index
            .search(query)
            .into_iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect()
    }

    pub fn raw_index(&self) -> &Index3D {
        &self.index
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoIndexMetadata {
    pub profile: GeometryProfile,
    pub feature_count: usize,
    pub index_entry_count: usize,
    pub entries_may_duplicate_rows: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoArtifact {
    pub manifest: GeoArtifactManifest,
    pub bytes_len: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoArtifactManifest {
    pub schema_version: u32,
    pub source_format: String,
    pub source_fingerprint: String,
    pub selected_column: String,
    pub crs: CrsInfo,
    pub edges: EdgeModel,
    pub encoding: GeometryEncoding,
    pub dims: CoordinateDims,
    pub storage_precision: StoragePrecision,
    pub null_policy: NullPolicy,
    pub antimeridian_policy: AntimeridianPolicy,
    pub payload_plan: PayloadPlan,
    pub feature_count: usize,
    pub index_entry_count: usize,
    pub entries_may_duplicate_rows: bool,
}
