#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod artifact;
mod dataset;
mod error;
mod geoarrow;
mod manifest;
mod types;
mod wkb;

pub use artifact::{
    GeoArtifactIndex, GeoArtifactIndex2D, GeoArtifactIndex3D, GeoHit, GeoPayload, open_geo_index,
    open_geo_index_with_limits,
};
pub use dataset::{
    FEATURE_JSON_CONTENT_TYPE, FEATURE_REF_CONTENT_TYPE, FEATURE_REF_RECORD_LEN,
    FEATURE_WKB_CONTENT_TYPE, GeoDataset, decode_feature_ref_payload, decode_feature_wkb_payload,
    open,
};
pub use error::GeoError;
pub use manifest::read_geo_manifest;
pub use types::*;

// Re-export the core types this crate produces or names, so a caller can build,
// convert, load, and query entirely through `packed_spatial_index_geo` without
// adding `packed_spatial_index` as a second direct dependency.
pub use packed_spatial_index::{
    Box2D, Box3D, FileMetadata, Index2D, Index3D, RangeReader, SliceReader, StreamIndex2D,
    StreamIndex2DF32, StreamIndex3D, StreamIndex3DF32, read_metadata,
};
