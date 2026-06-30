#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod artifact;
mod build;
mod dataset;
mod discovery;
mod error;
mod filter;
mod geoarrow;
mod geodetic;
mod manifest;
mod payload;
mod query;
mod scan;
mod validation;
mod wkb;

pub use artifact::{
    GeoArtifactIndex, GeoArtifactIndex2D, GeoArtifactIndex3D, GeoHit, GeoPayload, open_geo_index,
    open_geo_index_with_limits,
};
pub use build::{
    BuildRequest, ConvertRequest, GeoArtifact, GeoIndex, GeoIndex2D, GeoIndex3D, GeoIndexMetadata,
    IndexBuildOptions, StoragePrecision,
};
pub use dataset::{
    DuplicateFeatureRows, FEATURE_JSON_CONTENT_TYPE, FEATURE_REF_CONTENT_TYPE,
    FEATURE_REF_RECORD_LEN, FEATURE_WKB_CONTENT_TYPE, FeatureReadOrder, FeatureReadRequest,
    FeatureRows, GeoDataset, GeometryReadMode, IndexDimsRequest, InspectRequest,
    PropertyProjection, ValidateRequest, open,
};
pub use discovery::{
    ColumnCapabilities, CoordinateDims, CoordinateLayout, CrsInfo, DeclaredExtent,
    DiscoveryWarning, EdgeAlgorithm, EdgeModel, FileGeoMetadata, GeoDiscovery, GeometryColumn,
    GeometryColumnInfo, GeometryEncoding, GeometryKind, GeometryMetadataSource, GeometryProfile,
    GeometrySelectionReason, GeometrySelector, GeometryTypeSet, RowBoundsSource, SelectionStatus,
    WkbFlavor,
};
pub use error::GeoError;
pub use filter::FeatureFilterRequest;
pub use geodetic::{AntimeridianPolicy, EnvelopePolicy, NullPolicy};
pub use manifest::{GeoArtifactManifest, read_geo_manifest};
pub use payload::{
    FeatureRef, PayloadPlan, decode_feature_ref_payload, decode_feature_wkb_payload,
};
pub use query::{GeoQuery2D, GeoQuery3D, NonPlanarExactPolicy, SpatialPredicate};
pub use scan::{GeometryScan, GeometryScan2D, GeometryScan3D, ScanRequest};
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
pub use packed_spatial_index::{
    Box2D, Box3D, FileMetadata, Index2D, Index3D, RangeReader, SliceReader, StreamIndex2D,
    StreamIndex2DF32, StreamIndex3D, StreamIndex3DF32, read_metadata,
};
