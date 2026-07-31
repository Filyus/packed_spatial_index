/// Error returned by geospatial source discovery, scanning, conversion, and
/// artifact reading.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GeoError {
    /// Parquet reader error.
    #[cfg(feature = "parquet")]
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Arrow array/record-batch error.
    #[cfg(feature = "parquet")]
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// FlatGeobuf read or parse error.
    #[cfg(feature = "flatgeobuf")]
    #[error("flatgeobuf: {0}")]
    FlatGeobuf(String),
    /// GeoJSON parse error.
    #[cfg(feature = "geojson")]
    #[error("geojson: {0}")]
    GeoJson(String),
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
    /// A `GeometryScan` was built with a different
    /// payload plan than the `ConvertRequest` passed to
    /// `GeoArtifact::from_scan` asks for. The
    /// payload bytes are already fixed by the scan, so the request cannot change
    /// them; scan the source with the payload plan you want in the artifact (or
    /// use `GeoDataset::convert` /
    /// `GeoDataset::convert_into`, which scan
    /// and convert in one step).
    #[error(
        "scan was built with payload plan {scanned:?} but the ConvertRequest asks for {requested:?}; scan with the payload plan you want in the artifact"
    )]
    ScanPayloadMismatch {
        /// Payload plan the scan actually produced.
        scanned: crate::PayloadPlan,
        /// Payload plan the `ConvertRequest` asked for.
        requested: crate::PayloadPlan,
    },
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
    /// A polygon query geometry has no coordinates.
    #[error("polygon query geometry is empty")]
    EmptyQueryPolygon,
    /// The geometry type is not supported for spherical exact filtering.
    #[error("unsupported geometry for spherical exact filtering: {0}")]
    UnsupportedGeodeticGeometry(String),
    /// Geometry dimensionality does not match the requested index dimensions.
    ///
    /// Scanned envelopes, not declared geometry types, decide this: a bbox
    /// covering with no z bounds yields 2D envelopes even for a `Point Z`
    /// column. Request a payload plan that reads geometry (`RowWkb` or
    /// `FeatureJson`) to index the real z extents.
    #[error(
        "geometry is {found}D but {expected}D was requested (scanned envelopes decide this; a bbox covering without z bounds yields 2D envelopes)"
    )]
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

/// Classification of a [`GeoError`] returned by a query, independent of any
/// HTTP framework: whether it describes a problem in the query the caller can
/// act on (and which one), or an artifact/server-side fault.
///
/// Shared by every artifact query frontend so a case handled by one is not
/// silently missing from another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeoErrorClass {
    /// The query exceeded the caller's configured per-query cost limits.
    QueryTooLarge,
    /// An exact predicate was requested against a geometry model that cannot
    /// support one (non-planar/non-spherical exact predicate).
    UnsupportedQuery,
    /// The query's bbox or polygon input is invalid.
    InvalidBbox,
    /// The requested predicate cannot be evaluated for this artifact's
    /// geometry (spherical exact filtering against unsupported geometry).
    UnsupportedPredicate,
    /// Everything else: an artifact- or server-side fault, not a property of
    /// the request.
    ArtifactError,
}

impl GeoErrorClass {
    /// Classify a [`GeoError`] returned by a query.
    pub fn classify(err: &GeoError) -> Self {
        use packed_spatial_index::StreamError;
        match err {
            GeoError::Stream(StreamError::LimitExceeded) => Self::QueryTooLarge,
            GeoError::NonPlanarExactPredicate { .. }
            | GeoError::NonSphericalExactPredicate { .. } => Self::UnsupportedQuery,
            GeoError::InvalidSphericalQuery(_) | GeoError::EmptyQueryPolygon => Self::InvalidBbox,
            GeoError::UnsupportedGeodeticGeometry(_) => Self::UnsupportedPredicate,
            _ => Self::ArtifactError,
        }
    }

    /// Suggested HTTP status code for this class.
    pub fn http_status(self) -> u16 {
        match self {
            Self::QueryTooLarge | Self::UnsupportedQuery | Self::UnsupportedPredicate => 422,
            Self::InvalidBbox => 400,
            Self::ArtifactError => 500,
        }
    }

    /// Stable machine-readable error code for this class.
    pub fn code(self) -> &'static str {
        match self {
            Self::QueryTooLarge => "query_too_large",
            Self::UnsupportedQuery => "unsupported_query",
            Self::InvalidBbox => "invalid_bbox",
            Self::UnsupportedPredicate => "unsupported_predicate",
            Self::ArtifactError => "artifact_error",
        }
    }
}

/// Shorthand for [`GeoErrorClass::classify`].
pub fn classify_geo_error(err: &GeoError) -> GeoErrorClass {
    GeoErrorClass::classify(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_maps_every_known_case() {
        use packed_spatial_index::StreamError;

        assert_eq!(
            GeoErrorClass::classify(&GeoError::Stream(StreamError::LimitExceeded)),
            GeoErrorClass::QueryTooLarge
        );
        assert_eq!(
            GeoErrorClass::classify(&GeoError::NonPlanarExactPredicate {
                column: "geometry".to_string(),
                edges: crate::EdgeModel::Planar,
            }),
            GeoErrorClass::UnsupportedQuery
        );
        assert_eq!(
            GeoErrorClass::classify(&GeoError::InvalidSphericalQuery("bad".into())),
            GeoErrorClass::InvalidBbox
        );
        assert_eq!(
            GeoErrorClass::classify(&GeoError::EmptyQueryPolygon),
            GeoErrorClass::InvalidBbox
        );
        // The regression case: the Worker demo's own copy of this
        // classification used to fall through to a 500 here.
        assert_eq!(
            GeoErrorClass::classify(&GeoError::UnsupportedGeodeticGeometry("multipoint".into())),
            GeoErrorClass::UnsupportedPredicate
        );
        assert_eq!(
            GeoErrorClass::classify(&GeoError::MissingGeoManifest),
            GeoErrorClass::ArtifactError
        );
    }

    #[test]
    fn class_status_and_code_are_stable() {
        assert_eq!(GeoErrorClass::QueryTooLarge.http_status(), 422);
        assert_eq!(GeoErrorClass::QueryTooLarge.code(), "query_too_large");
        assert_eq!(GeoErrorClass::UnsupportedQuery.http_status(), 422);
        assert_eq!(GeoErrorClass::UnsupportedQuery.code(), "unsupported_query");
        assert_eq!(GeoErrorClass::InvalidBbox.http_status(), 400);
        assert_eq!(GeoErrorClass::InvalidBbox.code(), "invalid_bbox");
        assert_eq!(GeoErrorClass::UnsupportedPredicate.http_status(), 422);
        assert_eq!(
            GeoErrorClass::UnsupportedPredicate.code(),
            "unsupported_predicate"
        );
        assert_eq!(GeoErrorClass::ArtifactError.http_status(), 500);
        assert_eq!(GeoErrorClass::ArtifactError.code(), "artifact_error");
    }
}
