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
    NotFound(String),
    /// The client supplied an invalid request.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// The artifact cannot support the requested operation.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
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
            ServerError::BadRequest(_) | ServerError::Toml(_) => StatusCode::BAD_REQUEST,
            ServerError::NotFound(_) => StatusCode::NOT_FOUND,
            ServerError::Unsupported(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ServerError::Config(_) | ServerError::Io { .. } | ServerError::Geo(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }

    fn code(&self) -> &'static str {
        match self {
            ServerError::BadRequest(_) | ServerError::Toml(_) => "bad_request",
            ServerError::NotFound(_) => "not_found",
            ServerError::Unsupported(_) => "unsupported",
            ServerError::Config(_) => "configuration",
            ServerError::Io { .. } => "io",
            ServerError::Geo(_) => "geo",
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
