#![doc = include_str!("../README.md")]

mod build;
mod convert;
mod read;

pub use build::{build_index_2d, build_index_3d};
pub use convert::{
    ConvertPayload, ROW_ID_CONTENT_TYPE, ROW_WKB_CONTENT_TYPE, convert_2d, convert_2d_into,
    convert_3d, convert_3d_into, decode_row_id_payload, decode_row_wkb_payload,
};
pub use read::{
    GeoParquetInfo, GeometryColumnInfo, GeometryColumnSelection, GeometryDiscovery,
    GeometrySelectionReason, detect_dims, detect_dims_with_opts, discover, discover_with_opts,
    inspect, inspect_with_opts, read_bboxes_2d, read_bboxes_2d_with_opts, read_bboxes_3d,
    read_bboxes_3d_with_opts,
};

// Re-export the core types this crate produces or names, so a caller can build,
// convert, load, and query entirely through `packed_spatial_index_geo` without
// adding `packed_spatial_index` as a second direct dependency.
pub use packed_spatial_index::{
    Box2D, Box3D, FileMetadata, Index2D, Index3D, RangeReader, SliceReader, StreamIndex2D,
    StreamIndex2DF32, StreamIndex3D, StreamIndex3DF32, read_metadata,
};

/// Geometry metadata source used to select and interpret the geometry column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryMetadataSource {
    /// GeoParquet `geo` key-value metadata.
    GeoParquet,
    /// Apache Parquet native `GEOMETRY` / `GEOGRAPHY` logical types.
    ParquetGeospatial,
}

/// Reader column-selection options.
#[derive(Debug, Clone, Default)]
pub struct ReadOpts {
    /// Geometry column to read. When omitted, GeoParquet `primary_column` wins;
    /// native Parquet geospatial files auto-select only if exactly one root
    /// `GEOMETRY` / `GEOGRAPHY` column exists.
    pub geometry_column: Option<String>,
}

/// Index build parameters, forwarded to `Index*DBuilder`.
#[derive(Debug, Clone)]
pub struct BuildOpts {
    /// Geometry column to index. See [`ReadOpts::geometry_column`].
    pub geometry_column: Option<String>,
    /// Tree node fan-out. `None` keeps the crate default.
    pub node_size: Option<usize>,
    /// Build the index in parallel (requires the core `parallel` feature, on by
    /// default).
    pub parallel: bool,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            geometry_column: None,
            node_size: None,
            parallel: true,
        }
    }
}

impl BuildOpts {
    pub(crate) fn read_opts(&self) -> ReadOpts {
        ReadOpts {
            geometry_column: self.geometry_column.clone(),
        }
    }
}

/// Converter parameters for [`convert_2d`] / [`convert_3d`].
#[derive(Debug, Clone)]
pub struct ConvertOpts {
    /// Geometry column to convert. See [`ReadOpts::geometry_column`].
    pub geometry_column: Option<String>,
    /// Index build parameters.
    pub build: BuildOpts,
    /// Attach a leaf-ordered payload section. When `false`, only the index
    /// (bboxes + compact item ids) is serialized.
    ///
    /// Kept as a coarse on/off switch for callers that already set it. When it
    /// is `true`, [`payload`](Self::payload) selects what the payload contains.
    pub include_payload: bool,
    /// What to attach when [`include_payload`](Self::include_payload) is `true`.
    ///
    /// The default is [`ConvertPayload::RowWkb`], preserving the original
    /// source row id next to each WKB geometry. This matters when
    /// [`skip_null`](Self::skip_null) compacts the output ids.
    pub payload: ConvertPayload,
    /// Store coordinates as `f32` for a roughly half-size file. Queries become a
    /// conservative superset (box bounds are rounded outward); re-check exact hits
    /// against the payload geometry if you need precision.
    pub compact_f32: bool,
    /// Drop rows whose geometry is null or empty instead of erroring. The output
    /// index covers the surviving rows; item ids are positions in the output, not
    /// original file row indices (the converter output is self-contained, so this
    /// is safe — unlike the accelerator, which keeps `id == row` and always
    /// errors on null).
    pub skip_null: bool,
    /// Interleave the geometry payload with the tree leaves so a streaming query
    /// fetches a leaf and its geometry in one contiguous range read. This is the
    /// right default for the converter's purpose (serving geometry over a network)
    /// — it cuts round-trips. Turn it off for a layout where the tree is read far
    /// more often than payloads are fetched.
    pub interleaved: bool,
}

impl Default for ConvertOpts {
    fn default() -> Self {
        Self {
            geometry_column: None,
            build: BuildOpts::default(),
            include_payload: true,
            payload: ConvertPayload::RowWkb,
            compact_f32: false,
            skip_null: false,
            interleaved: true,
        }
    }
}

impl ConvertOpts {
    pub(crate) fn read_opts(&self) -> ReadOpts {
        ReadOpts {
            geometry_column: self.geometry_column.clone(),
        }
    }
}

/// Anything that can go wrong reading a geospatial Parquet source or building
/// the index.
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
    #[error("no geometry column")]
    NoGeometryColumn,
    /// The requested geometry column is missing or not a supported geometry
    /// column.
    #[error("geometry column `{0}` not found")]
    GeometryColumnNotFound(String),
    /// The file has multiple native Parquet geospatial columns and no explicit
    /// selection was provided.
    #[error("ambiguous geometry column; choose one of: {columns:?}")]
    AmbiguousGeometryColumn {
        /// Candidate root-level native Parquet geospatial columns.
        columns: Vec<String>,
    },
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
