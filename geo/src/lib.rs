#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod artifact;
#[cfg(feature = "_source")]
mod build;
#[cfg(feature = "parquet")]
mod dataset;
mod discovery;
mod error;
#[cfg(feature = "parquet")]
mod feature_read;
#[cfg(feature = "flatgeobuf")]
mod fgb;
mod filter;
#[cfg(feature = "parquet")]
mod geoarrow;
mod geodetic;
#[cfg(feature = "geojson")]
mod geojson;
mod manifest;
mod payload;
mod query;
#[cfg(feature = "parquet")]
mod scan;
#[cfg(feature = "_source")]
mod scan_core;
#[cfg(feature = "parquet")]
mod validation;
mod wkb;

pub use artifact::{
    GeoArtifactDirectory, GeoArtifactIndex, GeoArtifactIndex2D, GeoArtifactIndex3D, GeoHit,
    GeoPayload, open_geo_index, open_geo_index_with_limits,
};
#[cfg(feature = "async")]
pub use artifact::{open_geo_index_async, open_geo_index_with_limits_async};
#[cfg(feature = "_source")]
pub use build::{
    BuildRequest, ConvertRequest, GeoArtifact, GeoIndex, GeoIndex2D, GeoIndex2DF32, GeoIndex3D,
    GeoIndex3DF32, GeoIndexMetadata, GeoSource, IndexBuildOptions,
};
#[cfg(feature = "parquet")]
pub use dataset::{GeoDataset, InspectRequest, ValidateRequest, open_geoparquet};
#[cfg(feature = "parquet")]
pub use discovery::{
    ColumnCapabilities, DiscoveryWarning, FileGeoMetadata, GeoDiscovery, GeometryColumn,
    GeometryColumnInfo, GeometrySelectionReason, SelectionStatus,
};
pub use discovery::{
    CoordinateDims, CoordinateLayout, CrsInfo, EdgeAlgorithm, EdgeModel, GeometryEncoding,
    GeometryKind, GeometrySelector, WkbFlavor,
};
#[cfg(feature = "_source")]
pub use discovery::{
    DeclaredExtent, GeometryMetadataSource, GeometryProfile, GeometryTypeSet, RowBoundsSource,
};
pub use error::GeoError;
#[cfg(feature = "parquet")]
pub use feature_read::FeatureRows;
#[cfg(feature = "flatgeobuf")]
pub use fgb::{FgbDataset, open_flatgeobuf};
pub use filter::FeatureFilterRequest;
pub use geodetic::{AntimeridianPolicy, EnvelopePolicy, NullPolicy};
#[cfg(feature = "geojson")]
pub use geojson::{
    GeoJsonDataset, build_geojson_stream, convert_geojson_stream, open_geojson, open_geojson_slice,
};
pub use manifest::{GeoArtifactManifest, StoragePrecision, read_geo_manifest};
pub use payload::{
    FEATURE_JSON_CONTENT_TYPE, FEATURE_REF_CONTENT_TYPE, FEATURE_REF_RECORD_LEN,
    FEATURE_WKB_CONTENT_TYPE, FeatureRef, PayloadPlan, PropertyProjection,
    decode_feature_ref_payload, decode_feature_wkb_payload,
};
pub use query::{GeoQuery2D, GeoQuery3D, NonPlanarExactPolicy, SpatialPredicate};
#[cfg(feature = "_source")]
pub use scan_core::{
    DuplicateFeatureRows, FeatureReadOrder, FeatureReadRequest, FeatureRecord, GeometryReadMode,
    GeometryScan, GeometryScan2D, GeometryScan3D, IndexDimsRequest, ScanRequest,
};
#[cfg(feature = "parquet")]
pub use validation::{
    NativeBoundingBox, NativeGeospatialStatsReport, RowGroupGeospatialStats, ValidationCode,
    ValidationIssue, ValidationReport, ValidationSeverity,
};

// Re-export `geo_types` so callers can build `GeoQuery2D::Polygon` queries
// without adding `geo-types` as a second direct dependency.
pub use geo_types;

// Re-export the core types this crate produces or names, so a caller can build,
// convert, load, and query entirely through `packed_spatial_index_geo` without
// adding `packed_spatial_index` as a second direct dependency.
#[cfg(feature = "async")]
pub use packed_spatial_index::AsyncRangeReader;
pub use packed_spatial_index::{
    Box2D, Box3D, ClipSpaceZ, EARTH_RADIUS_M, FileMetadata, Frustum3D, Index2D, Index2DF32,
    Index3D, Index3DF32, Point2D, Point3D, RangeReader, Ray2D, Ray3D, SliceReader, StreamIndex2D,
    StreamIndex2DF32, StreamIndex3D, StreamIndex3DF32, haversine_distance_2d, read_metadata,
};
