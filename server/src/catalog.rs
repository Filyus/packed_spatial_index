use std::{
    collections::HashSet,
    fs,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
};

use packed_spatial_index::StreamLimits;
use serde::Deserialize;

use crate::ServerError;

/// Resolved server catalog.
#[derive(Debug, Clone)]
pub struct Catalog {
    /// Server bind configuration.
    pub server: ServerConfig,
    /// Resolved collection entries.
    pub collections: Vec<CollectionConfig>,
}

/// Server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Socket address used by the binary unless overridden on the CLI.
    #[serde(default = "default_addr")]
    pub addr: SocketAddr,
    /// Per-query cost limits applied to every artifact query.
    #[serde(default)]
    pub limits: LimitsConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: default_addr(),
            limits: LimitsConfig::default(),
        }
    }
}

fn default_addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 3000)
}

/// Per-query cost limits, in catalog vocabulary.
///
/// A query that would exceed any of these aborts with `query_too_large` rather
/// than running unbounded over a broad window. The defaults are deliberately
/// generous but finite: a `bbox` covering the whole artifact still materializes
/// its match set, and nothing else bounds that. Set a field to `0` to lift the
/// limit.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitsConfig {
    /// Maximum range reads per query. Defaults to unlimited: reads against a
    /// local file are cheap, and the byte and item caps already bound the work.
    /// The knob exists for slow or metered storage.
    #[serde(default)]
    pub max_reads: usize,
    /// Maximum bytes read per query.
    #[serde(default = "default_max_read_bytes")]
    pub max_read_bytes: u64,
    /// Maximum matches a query may produce before pagination.
    #[serde(default = "default_max_items")]
    pub max_items: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_reads: 0,
            max_read_bytes: default_max_read_bytes(),
            max_items: default_max_items(),
        }
    }
}

impl LimitsConfig {
    /// Translate to core stream limits, mapping `0` to unbounded.
    pub fn to_stream_limits(self) -> StreamLimits {
        StreamLimits {
            max_reads: (self.max_reads > 0).then_some(self.max_reads),
            max_read_bytes: (self.max_read_bytes > 0).then_some(self.max_read_bytes),
            max_items: (self.max_items > 0).then_some(self.max_items),
            ..StreamLimits::default()
        }
    }
}

fn default_max_read_bytes() -> u64 {
    512 * 1024 * 1024
}

fn default_max_items() -> usize {
    1_000_000
}

/// Collection entry from the catalog.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionConfig {
    /// URL-safe collection id.
    pub id: String,
    /// Human-readable title.
    #[serde(default)]
    pub title: Option<String>,
    /// Human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// Resolved path to the `.psindex` artifact.
    pub artifact: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCatalog {
    #[serde(default)]
    server: ServerConfig,
    #[serde(default)]
    collections: Vec<CollectionConfig>,
}

impl Catalog {
    /// Read and validate a catalog from a TOML file.
    ///
    /// Relative artifact paths are resolved against the catalog file's parent
    /// directory. Artifact files themselves are opened later by
    /// [`crate::ServerState::from_catalog`].
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ServerError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|e| ServerError::io(path, e))?;
        Self::from_toml_str(&text, path.parent().unwrap_or_else(|| Path::new(".")))
    }

    /// Parse and validate catalog TOML using `base_dir` for relative artifacts.
    pub fn from_toml_str(text: &str, base_dir: impl AsRef<Path>) -> Result<Self, ServerError> {
        let base_dir = base_dir.as_ref();
        let mut raw: RawCatalog = toml::from_str(text)?;
        if raw.collections.is_empty() {
            return Err(ServerError::Config(
                "catalog must contain at least one [[collections]] entry".to_string(),
            ));
        }
        let mut seen = HashSet::new();
        for collection in &mut raw.collections {
            validate_collection_id(&collection.id)?;
            if !seen.insert(collection.id.clone()) {
                return Err(ServerError::Config(format!(
                    "duplicate collection id `{}`",
                    collection.id
                )));
            }
            if collection.artifact.as_os_str().is_empty() {
                return Err(ServerError::Config(format!(
                    "collection `{}` has an empty artifact path",
                    collection.id
                )));
            }
            if collection.artifact.is_relative() {
                collection.artifact = base_dir.join(&collection.artifact);
            }
        }
        Ok(Self {
            server: raw.server,
            collections: raw.collections,
        })
    }
}

fn validate_collection_id(id: &str) -> Result<(), ServerError> {
    if id.is_empty() {
        return Err(ServerError::Config("collection id is empty".to_string()));
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err(ServerError::Config(format!(
            "collection id `{id}` must contain only ASCII letters, digits, `_`, or `-`"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_artifact_paths() {
        let catalog = Catalog::from_toml_str(
            r#"
            [[collections]]
            id = "places"
            artifact = "data/places.psindex"
            "#,
            Path::new("fixtures"),
        )
        .unwrap();
        assert_eq!(
            catalog.collections[0].artifact,
            PathBuf::from("fixtures").join("data/places.psindex")
        );
    }

    #[test]
    fn rejects_duplicate_collection_ids() {
        let err = Catalog::from_toml_str(
            r#"
            [[collections]]
            id = "places"
            artifact = "a.psindex"

            [[collections]]
            id = "places"
            artifact = "b.psindex"
            "#,
            Path::new("."),
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate collection id"));
    }

    #[test]
    fn rejects_unknown_catalog_keys() {
        let err = Catalog::from_toml_str(
            r#"
            [[collections]]
            id = "places"
            titel = "Places"
            artifact = "a.psindex"
            "#,
            Path::new("."),
        )
        .unwrap_err();
        assert!(err.to_string().contains("titel"), "{err}");

        let err = Catalog::from_toml_str(
            r#"
            [server.limits]
            max_item = 5

            [[collections]]
            id = "places"
            artifact = "a.psindex"
            "#,
            Path::new("."),
        )
        .unwrap_err();
        assert!(err.to_string().contains("max_item"), "{err}");
    }

    #[test]
    fn rejects_invalid_collection_ids() {
        let err = Catalog::from_toml_str(
            r#"
            [[collections]]
            id = "bad/id"
            artifact = "a.psindex"
            "#,
            Path::new("."),
        )
        .unwrap_err();
        assert!(err.to_string().contains("must contain only ASCII"));
    }
}
