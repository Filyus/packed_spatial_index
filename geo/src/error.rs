#[derive(Debug, thiserror::Error)]
pub enum GeoError {
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("geoparquet metadata: {0}")]
    Metadata(String),
    #[error("wkb: {0}")]
    Wkb(String),
    #[error(transparent)]
    Build(#[from] packed_spatial_index::BuildError),
    #[error(transparent)]
    Payload(#[from] packed_spatial_index::PayloadError),
    #[error(transparent)]
    Stream(#[from] packed_spatial_index::StreamError),
    #[error("psindex container: {0}")]
    Container(String),
    #[error("PSINDEX artifact has no geoM manifest")]
    MissingGeoManifest,
    #[error("unsupported geo artifact: {0}")]
    UnsupportedArtifact(String),
    #[error("cannot decode geo payload: {0}")]
    PayloadDecode(String),
    #[error("dataset reader has already been consumed")]
    DatasetConsumed,
    #[error("no geometry column")]
    NoGeometryColumn,
    #[error("geometry column `{0}` not found")]
    GeometryColumnNotFound(String),
    #[error("ambiguous geometry column; choose one of: {columns:?}")]
    AmbiguousGeometryColumn { columns: Vec<String> },
    #[error("row {row} has null or empty geometry")]
    NullGeometry { row: usize },
    #[error("unsupported geometry encoding: {0}")]
    UnsupportedEncoding(String),
    #[error("geometry is {found}D but {expected}D was requested")]
    DimMismatch { expected: u8, found: u8 },
    #[error("row {row} crosses the antimeridian; choose split or world policy")]
    Antimeridian { row: u64 },
    #[error("properties projection references missing column `{0}`")]
    PropertyColumnNotFound(String),
}
