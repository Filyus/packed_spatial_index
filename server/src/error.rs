use std::{io, path::PathBuf};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use packed_spatial_index_geo::GeoError;
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
    /// The artifact payload cannot support the requested operation.
    #[error("unsupported payload: {0}")]
    UnsupportedPayload(String),
    /// The requested spatial predicate cannot run for this collection.
    #[error("unsupported predicate: {0}")]
    UnsupportedPredicate(String),
    /// The requested result level cannot run for this collection.
    #[error("unsupported level: {0}")]
    UnsupportedLevel(String),
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
    Geo(#[from] GeoError),
}

impl ServerError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            ServerError::InvalidBbox(_)
            | ServerError::InvalidLimit(_)
            | ServerError::InvalidOffset(_)
            | ServerError::InvalidPredicate(_)
            | ServerError::InvalidLevel(_)
            | ServerError::InvalidPayload(_)
            | ServerError::Toml(_) => StatusCode::BAD_REQUEST,
            ServerError::CollectionNotFound(_) => StatusCode::NOT_FOUND,
            ServerError::UnsupportedPayload(_)
            | ServerError::UnsupportedPredicate(_)
            | ServerError::UnsupportedLevel(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ServerError::Config(_) | ServerError::Io { .. } | ServerError::Geo(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }

    fn code(&self) -> &'static str {
        match self {
            ServerError::InvalidBbox(_) => "invalid_bbox",
            ServerError::InvalidLimit(_) => "invalid_limit",
            ServerError::InvalidOffset(_) => "invalid_offset",
            ServerError::InvalidPredicate(_) => "invalid_predicate",
            ServerError::InvalidLevel(_) => "invalid_level",
            ServerError::InvalidPayload(_) => "invalid_payload",
            ServerError::Toml(_) => "bad_request",
            ServerError::CollectionNotFound(_) => "collection_not_found",
            ServerError::UnsupportedPayload(_) => "unsupported_payload",
            ServerError::UnsupportedPredicate(_) => "unsupported_predicate",
            ServerError::UnsupportedLevel(_) => "unsupported_level",
            ServerError::Config(_) => "configuration",
            ServerError::Io { .. } => "io",
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

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ErrorBody {
            error: ErrorInfo {
                code: self.code(),
                message: self.to_string(),
            },
        };
        (status, Json(body)).into_response()
    }
}
