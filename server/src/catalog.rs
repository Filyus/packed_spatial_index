use std::{
    collections::HashSet,
    fs,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
};

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
pub struct ServerConfig {
    /// Socket address used by the binary unless overridden on the CLI.
    #[serde(default = "default_addr")]
    pub addr: SocketAddr,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: default_addr(),
        }
    }
}

fn default_addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 3000)
}

/// Collection entry from the catalog.
#[derive(Debug, Clone, Deserialize)]
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
