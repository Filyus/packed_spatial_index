#[cfg(feature = "parquet")]
use std::collections::{HashMap, HashSet};

#[cfg(feature = "parquet")]
use parquet::basic::{EdgeInterpolationAlgorithm, LogicalType, Type as ParquetPhysicalType};
#[cfg(feature = "parquet")]
use parquet::file::metadata::{FileMetaData, ParquetMetaData};
use serde::{Deserialize, Serialize};

#[cfg(feature = "parquet")]
use crate::GeoError;
#[cfg(feature = "parquet")]
use crate::geoarrow;
#[cfg(feature = "parquet")]
use crate::validation;

#[cfg(feature = "parquet")]
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
    #[cfg(feature = "parquet")]
    pub(crate) fn is_wkb_payload(&self) -> bool {
        matches!(
            self,
            GeometryEncoding::Wkb { .. }
                | GeometryEncoding::ParquetGeometry
                | GeometryEncoding::ParquetGeography { .. }
        )
    }

    #[cfg(feature = "parquet")]
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
    #[cfg(feature = "parquet")]
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

    #[cfg(feature = "parquet")]
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
    #[cfg(feature = "parquet")]
    pub(crate) fn has_z(self) -> bool {
        matches!(self, CoordinateDims::Xyz | CoordinateDims::Xyzm)
    }

    #[cfg(feature = "parquet")]
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

    #[cfg(feature = "parquet")]
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

    #[cfg(feature = "parquet")]
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
    #[cfg(feature = "parquet")]
    pub(crate) fn as_index_crs(&self) -> Option<String> {
        match self {
            CrsInfo::Present { value } => Some(value.to_string()),
            CrsInfo::PresentString { value } | CrsInfo::ImpliedDefault { value } => {
                Some(value.clone())
            }
            CrsInfo::ExplicitNone | CrsInfo::Missing | CrsInfo::Unknown => None,
        }
    }

    #[cfg(feature = "parquet")]
    pub(crate) fn is_known_projected(&self) -> bool {
        let Some(value) = self.as_index_crs() else {
            return false;
        };
        let lower = value.to_ascii_lowercase();
        if lower.contains("crs84") || lower.contains("4326") {
            return false;
        }
        lower.contains("projected")
            || lower.contains("projcrs")
            || lower.contains("epsg:3857")
            || lower.contains("\"3857\"")
            || lower.contains("epsg:3395")
            || lower.contains("\"3395\"")
    }
}

#[cfg(feature = "parquet")]
/// Geometry type names known for a column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryTypeSet {
    /// Type names such as `"Point"`, `"Polygon"`, or `"Point Z"`.
    pub types: Vec<String>,
}

#[cfg(feature = "parquet")]
impl GeometryTypeSet {
    #[cfg(feature = "parquet")]
    pub(crate) fn unknown() -> Self {
        Self { types: Vec::new() }
    }
}

#[cfg(feature = "parquet")]
/// Declared dataset or column extent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredExtent {
    /// Extent values as declared by metadata.
    pub values: Vec<f64>,
}

#[cfg(feature = "parquet")]
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
}

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
/// Concrete selected geometry column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryColumn {
    /// Column name.
    pub name: String,
    /// Full column metadata.
    pub info: GeometryColumnInfo,
}

/// Geometry column selector.
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
#[derive(Debug, Clone)]
pub(crate) struct ColumnState {
    pub(crate) info: GeometryColumnInfo,
    pub(crate) covering: Option<GeoParquetBboxCovering>,
}

#[cfg(feature = "parquet")]
#[derive(Debug, Clone, Deserialize)]
struct GeoParquetMetadata {
    version: String,
    primary_column: String,
    columns: HashMap<String, GeoParquetColumnMetadata>,
}

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
fn deserialize_present_value<'de, D>(
    deserializer: D,
) -> Result<Option<Option<serde_json::Value>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer)
        .map(|value| Some(if value.is_null() { None } else { Some(value) }))
}

#[cfg(feature = "parquet")]
#[derive(Debug, Clone, Deserialize)]
struct GeoParquetCovering {
    bbox: GeoParquetBboxCovering,
}

#[cfg(feature = "parquet")]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GeoParquetBboxCovering {
    pub(crate) xmin: Vec<String>,
    pub(crate) ymin: Vec<String>,
    #[serde(default)]
    pub(crate) zmin: Option<Vec<String>>,
    pub(crate) xmax: Vec<String>,
    pub(crate) ymax: Vec<String>,
    #[serde(default)]
    pub(crate) zmax: Option<Vec<String>>,
}

#[cfg(feature = "parquet")]
#[derive(Debug, Clone)]
struct NativeColumn {
    name: String,
    encoding: GeometryEncoding,
    crs: CrsInfo,
    edges: EdgeModel,
    dims: CoordinateDims,
}

#[cfg(feature = "parquet")]
pub(crate) fn discover_metadata(
    meta: &ParquetMetaData,
) -> Result<(GeoDiscovery, Vec<ColumnState>), GeoError> {
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
fn geoparquet_edges(value: Option<&str>) -> EdgeModel {
    match value {
        Some(edge) if edge.eq_ignore_ascii_case("spherical") => EdgeModel::Spherical,
        Some(edge) if edge.eq_ignore_ascii_case("planar") => EdgeModel::Planar,
        Some(_) => EdgeModel::Unknown,
        None => EdgeModel::Planar,
    }
}

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
pub(crate) fn profile_from_state(state: &ColumnState, num_rows: u64) -> GeometryProfile {
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
