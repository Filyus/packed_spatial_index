use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use packed_spatial_index::{FileReader, Index2D, Index3D, RangeReader, StreamLimits};
use packed_spatial_index_geo::{
    CoordinateDims, GeoArtifactDirectory, GeoArtifactIndex, GeoArtifactManifest, PayloadPlan,
    open_geo_index,
};

use crate::{Catalog, CollectionConfig, ServerError};

/// Shared server state.
#[derive(Clone)]
pub struct ServerState {
    collections: Arc<HashMap<String, Arc<Collection>>>,
}

impl ServerState {
    /// Open all catalog artifacts and cache their artifact directories.
    pub fn from_catalog(catalog: Catalog) -> Result<Self, ServerError> {
        let limits = catalog.server.limits.to_stream_limits();
        let mut collections = HashMap::with_capacity(catalog.collections.len());
        for config in catalog.collections {
            let collection = Arc::new(Collection::open(config, limits)?);
            collections.insert(collection.id().to_owned(), collection);
        }
        Ok(Self {
            collections: Arc::new(collections),
        })
    }

    /// Return a collection by id.
    pub fn collection(&self, id: &str) -> Option<Arc<Collection>> {
        self.collections.get(id).cloned()
    }

    /// Return all collections sorted by id.
    pub fn collections(&self) -> Vec<Arc<Collection>> {
        let mut collections = self.collections.values().cloned().collect::<Vec<_>>();
        collections.sort_by(|a, b| a.id().cmp(b.id()));
        collections
    }
}

/// One configured PSINDEX collection.
pub struct Collection {
    id: String,
    title: Option<String>,
    description: Option<String>,
    artifact_path: PathBuf,
    directory: GeoArtifactDirectory,
    limits: StreamLimits,
    /// An in-memory owned index for queries the streaming artifact does not
    /// carry (the distance join), loaded once on the first request that needs
    /// it. Artifacts are immutable per the catalog contract, so the index never
    /// goes stale; a failed load is remembered too, so a broken file costs one
    /// attempt, not one per request.
    join_index: OnceLock<Result<JoinIndex, String>>,
}

impl Collection {
    fn open(config: CollectionConfig, limits: StreamLimits) -> Result<Self, ServerError> {
        if !Path::new(&config.artifact).is_file() {
            return Err(ServerError::Config(format!(
                "collection `{}` artifact does not exist or is not a file: {}",
                config.id,
                config.artifact.display()
            )));
        }
        let reader =
            FileReader::open(&config.artifact).map_err(|e| ServerError::io(&config.artifact, e))?;
        let index = open_geo_index(reader)?;
        let (directory, _reader) = index.into_directory();
        Ok(Self {
            id: config.id,
            title: config.title,
            description: config.description,
            artifact_path: config.artifact,
            directory,
            limits,
            join_index: OnceLock::new(),
        })
    }

    /// Collection id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Optional human-readable title.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Optional human-readable description.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Path to the local `.psindex` artifact.
    pub fn artifact_path(&self) -> &Path {
        &self.artifact_path
    }

    /// Attach a fresh local file reader to the cached artifact directory.
    pub fn open_local_index(&self) -> Result<GeoArtifactIndex<FileReader>, ServerError> {
        let reader = FileReader::open(&self.artifact_path)
            .map_err(|e| ServerError::io(&self.artifact_path, e))?;
        self.attach_reader(reader)
    }

    /// Attach an arbitrary range reader to the cached artifact directory.
    ///
    /// The catalog's per-query cost limits ride along, so every query issued
    /// through the returned index is bounded.
    pub fn attach_reader<R: RangeReader>(
        &self,
        reader: R,
    ) -> Result<GeoArtifactIndex<R>, ServerError> {
        Ok(GeoArtifactIndex::from_directory_with_limits(
            &self.directory,
            reader,
            self.limits,
        )?)
    }

    /// Cached geospatial artifact manifest.
    pub fn manifest(&self) -> &GeoArtifactManifest {
        self.directory.manifest()
    }

    /// Number of indexed entries.
    pub fn entry_count(&self) -> usize {
        self.directory.num_entries()
    }

    /// Packed node size in the artifact.
    pub fn node_size(&self) -> usize {
        self.directory.node_size()
    }

    /// Whether one source row can appear as several index entries, as
    /// antimeridian splitting produces.
    pub fn entries_may_duplicate_rows(&self) -> bool {
        self.manifest().entries_may_duplicate_rows
    }

    /// Whether `predicate=intersects` can run from artifact payloads alone.
    pub fn supports_intersects_predicate(&self) -> bool {
        matches!(
            self.manifest().dims,
            CoordinateDims::Xy | CoordinateDims::Xym
        ) && matches!(
            self.manifest().payload_plan,
            PayloadPlan::RowWkb | PayloadPlan::FeatureJson { .. }
        )
    }

    /// An in-memory owned index for the distance join, loaded once and cached.
    ///
    /// The load reads the whole artifact and parses it with the core owned
    /// loader, which accepts both serializable layouts — including the
    /// interleaved layout the `geo` convert writes by default. The
    /// dimensionality comes from the `geoM` manifest.
    pub(crate) fn join_index(&self) -> Result<&JoinIndex, ServerError> {
        self.join_index
            .get_or_init(|| {
                let bytes = std::fs::read(&self.artifact_path).map_err(|e| e.to_string())?;
                let index = match self.manifest().dims {
                    CoordinateDims::Xy | CoordinateDims::Xym => JoinIndex::D2(
                        Index2D::from_bytes(&bytes)
                            .map_err(|e| format!("artifact failed to load: {e}"))?,
                    ),
                    CoordinateDims::Xyz | CoordinateDims::Xyzm => JoinIndex::D3(
                        Index3D::from_bytes(&bytes)
                            .map_err(|e| format!("artifact failed to load: {e}"))?,
                    ),
                    CoordinateDims::Unknown => {
                        return Err(format!(
                            "collection `{}` has unknown dimensions; a distance join needs a 2D or 3D artifact",
                            self.id
                        ));
                    }
                };
                Ok(index)
            })
            .as_ref()
            .map_err(|e| ServerError::QueryTask(e.clone()))
    }
}

/// An owned core index over a collection's whole artifact, dimension-dispatched.
pub(crate) enum JoinIndex {
    /// 2D artifact.
    D2(Index2D),
    /// 3D artifact.
    D3(Index3D),
}

#[cfg(test)]
mod tests {
    use std::{
        fs, io,
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use packed_spatial_index::Box2D;
    use packed_spatial_index_geo::{
        ConvertRequest, PayloadPlan, SliceReader, open_geo_index, open_geojson_slice,
    };
    use tempfile::tempdir;

    use super::*;

    struct CountingReader<T> {
        inner: SliceReader<T>,
        reads: Arc<AtomicUsize>,
    }

    impl<T: AsRef<[u8]>> CountingReader<T> {
        fn new(data: T, reads: Arc<AtomicUsize>) -> Self {
            Self {
                inner: SliceReader::new(data),
                reads,
            }
        }
    }

    impl<T: AsRef<[u8]>> RangeReader for CountingReader<T> {
        fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.inner.read_exact_at(offset, buf)
        }

        fn len(&self) -> Option<u64> {
            self.inner.len()
        }
    }

    fn write_artifact(path: &Path) -> Vec<u8> {
        let doc = br#"{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                    "properties": {}
                }
            ]
        }"#;
        let mut source = open_geojson_slice(doc).unwrap();
        let bytes = source
            .convert(ConvertRequest {
                payload: PayloadPlan::RowRef,
                ..ConvertRequest::default()
            })
            .unwrap();
        fs::write(path, &bytes).unwrap();
        bytes
    }

    #[test]
    fn server_state_rejects_missing_artifact() {
        let dir = tempdir().unwrap();
        let catalog = Catalog {
            server: Default::default(),
            collections: vec![CollectionConfig {
                id: "places".to_string(),
                title: None,
                description: None,
                artifact: dir.path().join("missing.psindex"),
            }],
        };
        let err = match ServerState::from_catalog(catalog) {
            Ok(_) => panic!("missing artifact should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("artifact does not exist"));
    }

    #[test]
    fn server_state_rejects_invalid_artifact() {
        let dir = tempdir().unwrap();
        let artifact = dir.path().join("bad.psindex");
        fs::write(&artifact, b"not a psindex").unwrap();
        let catalog = Catalog {
            server: Default::default(),
            collections: vec![CollectionConfig {
                id: "places".to_string(),
                title: None,
                description: None,
                artifact,
            }],
        };
        assert!(ServerState::from_catalog(catalog).is_err());
    }

    #[test]
    fn attach_reader_performs_no_reads_until_query() {
        let dir = tempdir().unwrap();
        let artifact = dir.path().join("places.psindex");
        let bytes = write_artifact(&artifact);
        let catalog = Catalog {
            server: Default::default(),
            collections: vec![CollectionConfig {
                id: "places".to_string(),
                title: None,
                description: None,
                artifact,
            }],
        };
        let state = ServerState::from_catalog(catalog).unwrap();
        let collection = state.collection("places").unwrap();

        let reads = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader::new(bytes, Arc::clone(&reads));
        let index = collection.attach_reader(reader).unwrap();
        assert_eq!(reads.load(Ordering::SeqCst), 0);

        let GeoArtifactIndex::D2(index) = index else {
            panic!("expected 2D artifact");
        };
        let refs = index
            .search_feature_refs(Box2D::new(0.0, 0.0, 2.0, 2.0))
            .unwrap();
        assert_eq!(refs.len(), 1);
        assert!(reads.load(Ordering::SeqCst) > 0);
    }

    /// `count=only` is only worth a query parameter if counting really is
    /// cheaper than the search it replaces. The response shape cannot show
    /// that -- an empty `matches` array proves nothing about what was read --
    /// so compare the reads the artifact actually issues.
    #[test]
    fn counting_reads_less_than_materializing_the_same_matches() {
        let dir = tempdir().unwrap();
        let artifact = dir.path().join("places.psindex");
        let bytes = write_artifact(&artifact);
        let query = Box2D::new(0.0, 0.0, 2.0, 2.0);

        let counted_reads = Arc::new(AtomicUsize::new(0));
        let index = open_geo_index(CountingReader::new(
            bytes.clone(),
            Arc::clone(&counted_reads),
        ))
        .unwrap();
        let GeoArtifactIndex::D2(index) = index else {
            panic!("expected 2D artifact");
        };
        let opened = counted_reads.load(Ordering::SeqCst);
        let counted = index.count_entries(query).unwrap();
        let count_only = counted_reads.load(Ordering::SeqCst) - opened;

        let header_reads = Arc::new(AtomicUsize::new(0));
        let index = open_geo_index(CountingReader::new(bytes, Arc::clone(&header_reads))).unwrap();
        let GeoArtifactIndex::D2(index) = index else {
            panic!("expected 2D artifact");
        };
        let opened = header_reads.load(Ordering::SeqCst);
        let headers = index.search_match_headers(query).unwrap();
        let materialized = header_reads.load(Ordering::SeqCst) - opened;

        assert_eq!(
            counted,
            headers.len(),
            "the two paths must agree on the count"
        );
        assert!(
            count_only < materialized,
            "counting issued {count_only} reads, materializing {materialized}:              count=only is not buying anything"
        );
    }

    #[test]
    fn open_geo_index_would_read_the_counting_reader() {
        let dir = tempdir().unwrap();
        let artifact = dir.path().join("places.psindex");
        let bytes = write_artifact(&artifact);
        let reads = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader::new(bytes, Arc::clone(&reads));
        let _index = open_geo_index(reader).unwrap();
        assert!(reads.load(Ordering::SeqCst) > 0);
    }
}
