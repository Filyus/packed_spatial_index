use std::{io, path::PathBuf};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// Error type used by the local PSINDEX server.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Catalog or startup configuration is invalid.
    #[error("configuration error: {0}")]
    Config(String),
    /// A collection id was not found.
    #[error("collection `{0}` was not found")]
    CollectionNotFound(String),
    /// The request path matches no route.
    #[error("no route for `{0}`")]
    RouteNotFound(String),
    /// The route exists but does not accept the request method.
    #[error("method not allowed: {0}")]
    MethodNotAllowed(String),
    /// The query string could not be deserialized, typically an unknown
    /// parameter name.
    #[error("invalid query: {0}")]
    InvalidQuery(String),
    /// The client supplied an invalid bbox.
    #[error("invalid bbox: {0}")]
    InvalidBbox(String),
    /// The client supplied an invalid limit.
    #[error("invalid limit: {0}")]
    InvalidLimit(String),
    /// The client supplied an invalid offset.
    #[error("invalid offset: {0}")]
    InvalidOffset(String),
    /// The client supplied an invalid spatial predicate.
    #[error("invalid predicate: {0}")]
    InvalidPredicate(String),
    /// The client supplied an invalid result level.
    #[error("invalid level: {0}")]
    InvalidLevel(String),
    /// The client supplied an invalid payload mode.
    #[error("invalid payload mode: {0}")]
    InvalidPayload(String),
    /// The client supplied an invalid identity mode.
    #[error("invalid identity mode: {0}")]
    InvalidIdentity(String),
    /// The client supplied an invalid count mode.
    #[error("invalid count mode: {0}")]
    InvalidCount(String),
    /// The client supplied invalid frustum planes.
    #[error("invalid frustum: {0}")]
    InvalidFrustum(String),
    /// The client supplied a query the artifact's edge/encoding model rejects.
    /// 422, not 500: these describe a client request that cannot be satisfied
    /// by this collection, not a server-side fault.
    #[error("unsupported query: {0}")]
    UnsupportedQuery(String),
    /// The artifact payload cannot support the requested operation.
    #[error("unsupported payload: {0}")]
    UnsupportedPayload(String),
    /// The requested spatial predicate cannot run for this collection.
    #[error("unsupported predicate: {0}")]
    UnsupportedPredicate(String),
    /// The requested result level cannot run for this collection.
    #[error("unsupported level: {0}")]
    UnsupportedLevel(String),
    /// The query exceeded the catalog's per-query cost limits.
    #[error("query too large: {0}")]
    QueryTooLarge(String),
    /// The blocking task running an artifact query did not finish.
    #[error("query task failed: {0}")]
    QueryTask(String),
    /// File I/O failed.
    #[error("I/O error for {path}: {source}")]
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// TOML parsing failed.
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    /// Geospatial artifact/query error.
    #[error("geo artifact error: {0}")]
    Geo(#[from] packed_spatial_index_geo::GeoError),
}

impl ServerError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Classify a [`GeoError`](packed_spatial_index_geo::GeoError) returned by a
    /// query as a client-side problem (422) when it describes an artifact/query
    /// incompatibility, otherwise fall back to the generic 500 mapping.
    ///
    /// Used after `filter_matches` / exact-predicate evaluation: a spherical
    /// column with `predicate=intersects`, or a payload the artifact cannot
    /// decode, is a property of the request, not a server fault.
    pub(crate) fn from_geo(err: packed_spatial_index_geo::GeoError) -> Self {
        use packed_spatial_index_geo::GeoErrorClass as Class;
        match packed_spatial_index_geo::classify_geo_error(&err) {
            // The artifact is fine and the server is fine; the query asked for
            // more work than the catalog allows.
            Class::QueryTooLarge => Self::QueryTooLarge(
                "the query exceeded this server's per-query cost limits; narrow the bbox"
                    .to_string(),
            ),
            Class::UnsupportedQuery => Self::UnsupportedQuery(err.to_string()),
            Class::InvalidBbox => Self::InvalidBbox(err.to_string()),
            Class::UnsupportedPredicate => Self::UnsupportedPredicate(err.to_string()),
            // Covers `ArtifactError` and any class added later behind
            // `GeoErrorClass`'s `#[non_exhaustive]`.
            _ => Self::Geo(err),
        }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            ServerError::InvalidQuery(_)
            | ServerError::InvalidBbox(_)
            | ServerError::InvalidLimit(_)
            | ServerError::InvalidOffset(_)
            | ServerError::InvalidPredicate(_)
            | ServerError::InvalidLevel(_)
            | ServerError::InvalidPayload(_)
            | ServerError::InvalidIdentity(_)
            | ServerError::InvalidCount(_)
            | ServerError::InvalidFrustum(_) => StatusCode::BAD_REQUEST,
            ServerError::CollectionNotFound(_) | ServerError::RouteNotFound(_) => {
                StatusCode::NOT_FOUND
            }
            ServerError::MethodNotAllowed(_) => StatusCode::METHOD_NOT_ALLOWED,
            ServerError::UnsupportedQuery(_)
            | ServerError::UnsupportedPayload(_)
            | ServerError::UnsupportedPredicate(_)
            | ServerError::UnsupportedLevel(_)
            | ServerError::QueryTooLarge(_) => StatusCode::UNPROCESSABLE_ENTITY,
            // A catalog is parsed once at startup, never in a request, so a
            // TOML error can only ever be a misconfigured server.
            ServerError::Config(_)
            | ServerError::Toml(_)
            | ServerError::Io { .. }
            | ServerError::Geo(_)
            | ServerError::QueryTask(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            ServerError::InvalidQuery(_) => "invalid_query",
            ServerError::InvalidBbox(_) => "invalid_bbox",
            ServerError::InvalidLimit(_) => "invalid_limit",
            ServerError::InvalidOffset(_) => "invalid_offset",
            ServerError::InvalidPredicate(_) => "invalid_predicate",
            ServerError::InvalidLevel(_) => "invalid_level",
            ServerError::InvalidPayload(_) => "invalid_payload",
            ServerError::InvalidIdentity(_) => "invalid_identity",
            ServerError::InvalidCount(_) => "invalid_count",
            ServerError::InvalidFrustum(_) => "invalid_frustum",
            ServerError::CollectionNotFound(_) => "collection_not_found",
            ServerError::RouteNotFound(_) => "not_found",
            ServerError::MethodNotAllowed(_) => "method_not_allowed",
            ServerError::UnsupportedQuery(_) => "unsupported_query",
            ServerError::UnsupportedPayload(_) => "unsupported_payload",
            ServerError::UnsupportedPredicate(_) => "unsupported_predicate",
            ServerError::UnsupportedLevel(_) => "unsupported_level",
            ServerError::QueryTooLarge(_) => "query_too_large",
            ServerError::Config(_) | ServerError::Toml(_) => "configuration",
            ServerError::Io { .. } => "io",
            ServerError::QueryTask(_) => "query_task",
            ServerError::Geo(_) => "artifact_error",
        }
    }
}

/// JSON error body returned by HTTP handlers.
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    /// Error object.
    pub error: ErrorInfo,
}

/// Stable HTTP error details.
#[derive(Debug, Serialize)]
pub struct ErrorInfo {
    /// Machine-readable error class.
    pub code: &'static str,
    /// Human-readable error message.
    pub message: String,
}

impl ServerError {
    /// Message safe to return to a client.
    ///
    /// Only [`ServerError::Io`] differs from [`Display`](std::fmt::Display):
    /// its path describes where this server keeps its files, which is the
    /// operator's business rather than the caller's. Everything else stays
    /// verbatim — `Geo(_)` in particular describes the artifact the client
    /// asked about (column names, row numbers, the source fingerprint) and
    /// carries no filesystem paths, so redacting it would only make failures
    /// harder to report.
    fn client_message(&self) -> String {
        match self {
            ServerError::Io { source, .. } => format!("I/O error reading the artifact: {source}"),
            other => other.to_string(),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        if status.is_server_error() {
            // The full form, path included, belongs in the operator's log.
            tracing::error!(error = %self, "request failed");
        }
        let body = ErrorBody {
            error: ErrorInfo {
                code: self.code(),
                message: self.client_message(),
            },
        };
        (status, Json(body)).into_response()
    }
}
