#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod artifact;
#[cfg(feature = "parquet")]
mod build;
#[cfg(feature = "parquet")]
mod dataset;
mod discovery;
mod error;
#[cfg(feature = "parquet")]
mod feature_read;
mod filter;
#[cfg(feature = "parquet")]
mod geoarrow;
mod geodetic;
mod manifest;
mod payload;
mod query;
#[cfg(feature = "parquet")]
mod scan;
#[cfg(feature = "parquet")]
mod validation;
mod wkb;

pub use artifact::{
    GeoArtifactIndex, GeoArtifactIndex2D, GeoArtifactIndex3D, GeoHit, GeoPayload, open_geo_index,
    open_geo_index_with_limits,
};
#[cfg(feature = "async")]
pub use artifact::{open_geo_index_async, open_geo_index_with_limits_async};
#[cfg(feature = "parquet")]
pub use build::{
    BuildRequest, ConvertRequest, GeoArtifact, GeoIndex, GeoIndex2D, GeoIndex2DF32, GeoIndex3D,
    GeoIndex3DF32, GeoIndexMetadata, IndexBuildOptions,
};
#[cfg(feature = "parquet")]
pub use dataset::{GeoDataset, IndexDimsRequest, InspectRequest, ValidateRequest, open};
#[cfg(feature = "parquet")]
pub use discovery::{
    ColumnCapabilities, DeclaredExtent, DiscoveryWarning, FileGeoMetadata, GeoDiscovery,
    GeometryColumn, GeometryColumnInfo, GeometryMetadataSource, GeometryProfile,
    GeometrySelectionReason, GeometryTypeSet, RowBoundsSource, SelectionStatus,
};
pub use discovery::{
    CoordinateDims, CoordinateLayout, CrsInfo, EdgeAlgorithm, EdgeModel, GeometryEncoding,
    GeometryKind, GeometrySelector, WkbFlavor,
};
pub use error::GeoError;
#[cfg(feature = "parquet")]
pub use feature_read::{
    DuplicateFeatureRows, FeatureReadOrder, FeatureReadRequest, FeatureRows, GeometryReadMode,
};
pub use filter::FeatureFilterRequest;
pub use geodetic::{AntimeridianPolicy, EnvelopePolicy, NullPolicy};
pub use manifest::{GeoArtifactManifest, StoragePrecision, read_geo_manifest};
pub use payload::{
    FEATURE_JSON_CONTENT_TYPE, FEATURE_REF_CONTENT_TYPE, FEATURE_REF_RECORD_LEN,
    FEATURE_WKB_CONTENT_TYPE, FeatureRef, PayloadPlan, PropertyProjection,
    decode_feature_ref_payload, decode_feature_wkb_payload,
};
pub use query::{GeoQuery2D, GeoQuery3D, NonPlanarExactPolicy, SpatialPredicate};
#[cfg(feature = "parquet")]
pub use scan::{GeometryScan, GeometryScan2D, GeometryScan3D, ScanRequest};
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
