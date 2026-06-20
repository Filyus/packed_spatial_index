//! Build a [`packed_spatial_index`] spatial index from a **GeoParquet** file.
//!
//! GeoParquet stores geometry plus, since 1.1, an optional per-row *bbox
//! covering* column — but it has no per-row spatial index, only per-row-group
//! statistics. This crate bridges that gap in two ways:
//!
//! * **Accelerator** — [`build_index_2d`] / [`build_index_3d`] build an in-memory
//!   index whose item id is the GeoParquet **row index**. Query results are row
//!   indices you can read back from the original file.
//! * **Converter** — [`convert_2d`] / [`convert_3d`] build the index *and* attach
//!   the WKB geometry as a leaf-ordered payload, serialized to a self-describing
//!   `PSINDEX` blob. That blob is queryable by the streaming engine straight from
//!   cloud storage (window / kNN / raycast returning the actual geometry in a
//!   handful of range reads).
//!
//! The heavy `arrow` / `parquet` / `geoparquet` dependencies live only here; the
//! `packed_spatial_index` core that *queries* the output stays lean (and wasm /
//! edge friendly). Build runs server-side; query runs anywhere.

mod build;
mod convert;
mod read;

pub use build::{build_index_2d, build_index_3d};
pub use convert::{convert_2d, convert_3d};
pub use read::{detect_dims, read_bboxes_2d, read_bboxes_3d};

// Re-export the core types that appear in this crate's signatures so callers
// don't have to depend on `packed_spatial_index` directly for the basics.
pub use packed_spatial_index::{Box2D, Box3D, Index2D, Index3D};

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
