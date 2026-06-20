#![doc = include_str!("../README.md")]

mod build;
mod convert;
mod read;

pub use build::{build_index_2d, build_index_3d};
pub use convert::{convert_2d, convert_2d_into, convert_3d, convert_3d_into};
pub use read::{GeoParquetInfo, detect_dims, inspect, read_bboxes_2d, read_bboxes_3d};

// Re-export the core types this crate produces or names, so a caller can build,
// convert, load, and query entirely through `packed_spatial_index_geo` without
// adding `packed_spatial_index` as a second direct dependency.
pub use packed_spatial_index::{
    Box2D, Box3D, FileMetadata, Index2D, Index3D, RangeReader, SliceReader, StreamIndex2D,
    StreamIndex2DF32, StreamIndex3D, StreamIndex3DF32, read_metadata,
};

/// Index build parameters, forwarded to `Index*DBuilder`.
#[derive(Debug, Clone)]
pub struct BuildOpts {
    /// Tree node fan-out. `None` keeps the crate default.
    pub node_size: Option<usize>,
    /// Build the index in parallel (requires the core `parallel` feature, on by
    /// default).
    pub parallel: bool,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            node_size: None,
            parallel: true,
        }
    }
}

/// Converter parameters for [`convert_2d`] / [`convert_3d`].
#[derive(Debug, Clone)]
pub struct ConvertOpts {
    /// Index build parameters.
    pub build: BuildOpts,
    /// Attach each row's WKB geometry as a leaf-ordered payload. When `false`
    /// only the index (bboxes + row ids) is serialized.
    pub include_payload: bool,
    /// Store coordinates as `f32` for a roughly half-size file. Queries become a
    /// conservative superset (box bounds are rounded outward); re-check exact hits
    /// against the payload geometry if you need precision.
    pub compact_f32: bool,
}

impl Default for ConvertOpts {
    fn default() -> Self {
        Self {
            build: BuildOpts::default(),
            include_payload: true,
            compact_f32: false,
        }
    }
}

/// Anything that can go wrong reading a GeoParquet source or building the index.
#[derive(Debug, thiserror::Error)]
pub enum GeoError {
    /// Error from the underlying `parquet` reader.
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Error from the underlying `arrow` layer.
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// Failed to parse or interpret the GeoParquet `geo` metadata.
    #[error("geoparquet metadata: {0}")]
    Metadata(String),
    /// Failed to decode a WKB geometry while computing its envelope.
    #[error("wkb: {0}")]
    Wkb(String),
    /// The index builder rejected the input.
    #[error(transparent)]
    Build(#[from] packed_spatial_index::BuildError),
    /// The serializer rejected the payload.
    #[error(transparent)]
    Payload(#[from] packed_spatial_index::PayloadError),
    /// The file has no primary geometry column.
    #[error("no GeoParquet geometry column")]
    NoGeometryColumn,
    /// A row has null or empty geometry. v1 keeps item id equal to the file row
    /// index, which has no room for skipped rows; filter such rows before
    /// indexing. (A skip/remap policy may be added later.)
    #[error("row {row} has null or empty geometry")]
    NullGeometry {
        /// The offending file row index.
        row: usize,
    },
    /// The geometry encoding is not one this crate can read.
    #[error("unsupported geometry encoding: {0}")]
    UnsupportedEncoding(String),
    /// Requested a dimensionality the file's geometry does not have.
    #[error("geometry is {found}D but {expected}D was requested")]
    DimMismatch {
        /// Dimensionality requested by the caller.
        expected: u8,
        /// Dimensionality detected in the file.
        found: u8,
    },
}
