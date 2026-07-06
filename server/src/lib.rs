//! Local native HTTP server for querying geospatial PSINDEX artifacts.
//!
//! The server is artifact-first: it opens `.psindex` files at startup, caches
//! their [`packed_spatial_index_geo::GeoArtifactDirectory`] values, and
//! attaches fresh local file readers per request.

#![warn(missing_docs)]

/// Catalog parsing and path resolution.
pub mod catalog;
mod collection;
mod error;
mod http;
mod query;

pub use catalog::{Catalog, CollectionConfig, ServerConfig};
pub use collection::{Collection, ServerState};
pub use error::{ErrorBody, ServerError};
pub use http::{router, serve};
