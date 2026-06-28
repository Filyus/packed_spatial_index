/// Error returned by geospatial Parquet discovery, scanning, conversion, and
/// artifact reading.
#[derive(Debug, thiserror::Error)]
pub enum GeoError {
    /// Parquet reader error.
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Arrow array/record-batch error.
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// Invalid or unsupported geospatial metadata.
    #[error("geoparquet metadata: {0}")]
    Metadata(String),
    /// WKB parse or envelope error.
    #[error("wkb: {0}")]
    Wkb(String),
    /// Core index build error.
    #[error(transparent)]
    Build(#[from] packed_spatial_index::BuildError),
    /// Core payload serialization error.
    #[error(transparent)]
    Payload(#[from] packed_spatial_index::PayloadError),
    /// Core stream reader error.
    #[error(transparent)]
    Stream(#[from] packed_spatial_index::StreamError),
    /// PSINDEX container framing error.
    #[error("psindex container: {0}")]
    Container(String),
    /// Converted artifact has no `geoM` manifest.
    #[error("PSINDEX artifact has no geoM manifest")]
    MissingGeoManifest,
    /// Artifact manifest or layout is not supported.
    #[error("unsupported geo artifact: {0}")]
    UnsupportedArtifact(String),
    /// Artifact payload could not be decoded according to the manifest.
    #[error("cannot decode geo payload: {0}")]
    PayloadDecode(String),
    /// Dataset rows have already been consumed by a scan/build/convert call.
    #[error("dataset reader has already been consumed")]
    DatasetConsumed,
    /// No usable geometry column exists.
    #[error("no geometry column")]
    NoGeometryColumn,
    /// Requested geometry column was not found or is not usable.
    #[error("geometry column `{0}` not found")]
    GeometryColumnNotFound(String),
    /// Multiple geometry columns match the default selector.
    #[error("ambiguous geometry column; choose one of: {columns:?}")]
    AmbiguousGeometryColumn {
        /// Candidate column names.
        columns: Vec<String>,
    },
    /// A row contains null or empty geometry and the null policy is `Error`.
    #[error("row {row} has null or empty geometry")]
    NullGeometry {
        /// Source row number.
        row: usize,
    },
    /// Geometry encoding is not supported for the requested operation.
    #[error("unsupported geometry encoding: {0}")]
    UnsupportedEncoding(String),
    /// Exact planar predicates were requested for a non-planar geometry column.
    #[error(
        "exact planar predicate requested for non-planar column `{column}` with edges {edges:?}; choose treat-as-planar to opt in"
    )]
    NonPlanarExactPredicate {
        /// Selected geometry column.
        column: String,
        /// Declared edge model.
        edges: crate::EdgeModel,
    },
    /// Exact spherical predicates were requested for a non-spherical geometry column.
    #[error(
        "exact spherical predicate requested for column `{column}` with edges {edges:?}; spherical radius filtering requires GEOGRAPHY(SPHERICAL)"
    )]
    NonSphericalExactPredicate {
        /// Selected geometry column.
        column: String,
        /// Declared edge model.
        edges: crate::EdgeModel,
    },
    /// Spherical radius query parameters are invalid.
    #[error("invalid spherical query: {0}")]
    InvalidSphericalQuery(String),
    /// The geometry type is not supported for spherical exact filtering.
    #[error("unsupported geometry for spherical exact filtering: {0}")]
    UnsupportedGeodeticGeometry(String),
    /// Geometry dimensionality does not match the requested index dimensions.
    #[error("geometry is {found}D but {expected}D was requested")]
    DimMismatch {
        /// Requested dimension count.
        expected: u8,
        /// Found dimension count.
        found: u8,
    },
    /// A geographic envelope crosses the antimeridian under `Reject` policy.
    #[error("row {row} crosses the antimeridian; choose split or world policy")]
    Antimeridian {
        /// Source row number.
        row: u64,
    },
    /// A `FeatureJson` property projection references a missing column.
    #[error("properties projection references missing column `{0}`")]
    PropertyColumnNotFound(String),
    /// Expected source fingerprint does not match the opened dataset.
    #[error("source fingerprint mismatch: expected {expected}, found {actual}")]
    SourceFingerprintMismatch {
        /// Expected fingerprint.
        expected: String,
        /// Actual fingerprint of the opened source.
        actual: String,
    },
    /// A feature reference points outside the source row count.
    #[error("feature ref row {row_number} is outside source row count {num_rows}")]
    FeatureRowOutOfBounds {
        /// Referenced absolute source row.
        row_number: u64,
        /// Number of rows in the opened source.
        num_rows: u64,
    },
    /// A feature reference carries inconsistent row-group coordinates.
    #[error(
        "feature ref row {row_number} does not match row group {row_group} offset {row_in_group}"
    )]
    FeatureRowPositionMismatch {
        /// Referenced absolute source row.
        row_number: u64,
        /// Referenced row group.
        row_group: u32,
        /// Referenced row offset within the row group.
        row_in_group: u32,
    },
}
