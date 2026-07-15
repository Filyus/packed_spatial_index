use geo::{BoundingRect, Intersects};
use geo_types::{Coord, MultiPolygon, Rect};
use geozero::ToGeo;
use geozero::geojson::GeoJson;
#[cfg(feature = "async")]
use packed_spatial_index::AsyncRangeReader;
use packed_spatial_index::{
    Box2D, Overlaps2D, PayloadPrefix, RangeReader, StreamDirectory, StreamError, StreamIndex2D,
    StreamIndex2DF32, StreamIndex3D, StreamIndex3DF32, StreamLimits,
};

use crate::{
    FEATURE_REF_RECORD_LEN, FeatureRef, GeoArtifactManifest, GeoError, GeoQuery2D, GeoQuery3D,
    NonPlanarExactPolicy, PayloadPlan, SpatialPredicate, StoragePrecision,
    decode_feature_ref_payload, decode_feature_wkb_payload, feature_json_body,
    filter::{
        decode_geo_geometry, exact_predicate_matches, exact_wkb_predicate_matches,
        prepare_filter_query,
    },
    manifest::{
        CHUNK_ENTRY_LEN, FORMAT_MAGIC, FORMAT_VERSION, SUPERBLOCK_LEN, TAG_GEO_MANIFEST,
        read_geo_manifest_content, read_u32, read_u64,
    },
};

/// Adapts a geo polygon query to the core [`Overlaps2D`] trait so a polygon can
/// drive the streaming region traversal — pruning subtrees outside the polygon
/// during the descent, instead of fetching everything in its bounding box.
struct PolygonRegion<'a>(&'a MultiPolygon<f64>);

impl Overlaps2D for PolygonRegion<'_> {
    fn overlaps_box(&self, bx: Box2D) -> bool {
        let rect = Rect::new(
            Coord {
                x: bx.min_x,
                y: bx.min_y,
            },
            Coord {
                x: bx.max_x,
                y: bx.max_y,
            },
        );
        self.0.intersects(&rect)
    }
}

/// Open a converted geospatial `PSINDEX` artifact.
///
/// The reader first fetches the `geoM` manifest from the container directory,
/// then opens the matching 2D/3D and f64/f32 streaming index. The backing
/// [`RangeReader`] does not need to report a length; this works with local files
/// and HTTP range-style readers.
///
/// # Example
///
/// ```no_run
/// use packed_spatial_index_geo::{open_geo_index, Box2D, GeoArtifactIndex, SliceReader};
///
/// let bytes = std::fs::read("cities.psindex")?;
/// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
///     panic!("expected a 2D artifact");
/// };
/// let refs = index.search_feature_refs(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn open_geo_index<R: RangeReader>(reader: R) -> Result<GeoArtifactIndex<R>, GeoError> {
    open_geo_index_with_limits(reader, StreamLimits::default())
}

/// Open a converted geospatial `PSINDEX` artifact with explicit stream limits.
///
/// Use this when opening untrusted or externally hosted artifacts and you want
/// to cap accepted item counts, directory sizes, or range-read sizes through
/// [`StreamLimits`].
pub fn open_geo_index_with_limits<R: RangeReader>(
    reader: R,
    limits: StreamLimits,
) -> Result<GeoArtifactIndex<R>, GeoError> {
    let manifest = read_manifest_from_reader(&reader)?;
    validate_manifest(&manifest)?;
    let artifact = match (manifest.dims.index_dims(), manifest.storage_precision) {
        (Some(2), StoragePrecision::F64) => GeoArtifactIndex::D2(GeoArtifactIndex2D {
            index: GeoStreamIndex2D::F64(StreamIndex2D::open_with_limits(reader, limits)?),
            manifest,
        }),
        (Some(2), StoragePrecision::F32) => GeoArtifactIndex::D2(GeoArtifactIndex2D {
            index: GeoStreamIndex2D::F32(StreamIndex2DF32::open_with_limits(reader, limits)?),
            manifest,
        }),
        (Some(3), StoragePrecision::F64) => GeoArtifactIndex::D3(GeoArtifactIndex3D {
            index: GeoStreamIndex3D::F64(StreamIndex3D::open_with_limits(reader, limits)?),
            manifest,
        }),
        (Some(3), StoragePrecision::F32) => GeoArtifactIndex::D3(GeoArtifactIndex3D {
            index: GeoStreamIndex3D::F32(StreamIndex3DF32::open_with_limits(reader, limits)?),
            manifest,
        }),
        (None, _) => {
            return Err(GeoError::UnsupportedArtifact(format!(
                "artifact has unknown coordinate dimensions {:?}",
                manifest.dims
            )));
        }
        (Some(other), _) => {
            return Err(GeoError::UnsupportedArtifact(format!(
                "artifact has unsupported coordinate dimension count {other}"
            )));
        }
    };
    validate_manifest_entry_count(artifact.manifest(), artifact.num_entries())?;
    validate_manifest_payload_presence(artifact.manifest(), artifact.has_payload())?;
    Ok(artifact)
}

/// Open a converted geospatial `PSINDEX` artifact from async range I/O.
///
/// Async opening mirrors [`open_geo_index`]: it fetches the `geoM` manifest,
/// opens the matching streaming index, and keeps all per-query reads async.
#[cfg(feature = "async")]
pub async fn open_geo_index_async<R: AsyncRangeReader>(
    reader: R,
) -> Result<GeoArtifactIndex<R>, GeoError> {
    open_geo_index_with_limits_async(reader, StreamLimits::default()).await
}

/// Open a converted geospatial `PSINDEX` artifact from async range I/O with
/// explicit stream limits.
#[cfg(feature = "async")]
pub async fn open_geo_index_with_limits_async<R: AsyncRangeReader>(
    reader: R,
    limits: StreamLimits,
) -> Result<GeoArtifactIndex<R>, GeoError> {
    let manifest = read_manifest_from_reader_async(&reader).await?;
    validate_manifest(&manifest)?;
    let artifact = match (manifest.dims.index_dims(), manifest.storage_precision) {
        (Some(2), StoragePrecision::F64) => GeoArtifactIndex::D2(GeoArtifactIndex2D {
            index: GeoStreamIndex2D::F64(
                StreamIndex2D::open_with_limits_async(reader, limits).await?,
            ),
            manifest,
        }),
        (Some(2), StoragePrecision::F32) => GeoArtifactIndex::D2(GeoArtifactIndex2D {
            index: GeoStreamIndex2D::F32(
                StreamIndex2DF32::open_with_limits_async(reader, limits).await?,
            ),
            manifest,
        }),
        (Some(3), StoragePrecision::F64) => GeoArtifactIndex::D3(GeoArtifactIndex3D {
            index: GeoStreamIndex3D::F64(
                StreamIndex3D::open_with_limits_async(reader, limits).await?,
            ),
            manifest,
        }),
        (Some(3), StoragePrecision::F32) => GeoArtifactIndex::D3(GeoArtifactIndex3D {
            index: GeoStreamIndex3D::F32(
                StreamIndex3DF32::open_with_limits_async(reader, limits).await?,
            ),
            manifest,
        }),
        (None, _) => {
            return Err(GeoError::UnsupportedArtifact(format!(
                "artifact has unknown coordinate dimensions {:?}",
                manifest.dims
            )));
        }
        (Some(other), _) => {
            return Err(GeoError::UnsupportedArtifact(format!(
                "artifact has unsupported coordinate dimension count {other}"
            )));
        }
    };
    validate_manifest_entry_count(artifact.manifest(), artifact.num_entries())?;
    validate_manifest_payload_presence(artifact.manifest(), artifact.has_payload())?;
    Ok(artifact)
}

fn validate_manifest(manifest: &GeoArtifactManifest) -> Result<(), GeoError> {
    if manifest.schema_version != 2 {
        return Err(GeoError::UnsupportedArtifact(format!(
            "unsupported geoM schema version {}",
            manifest.schema_version
        )));
    }
    if manifest.feature_count > manifest.index_entry_count {
        return Err(GeoError::UnsupportedArtifact(format!(
            "geoM feature_count {} exceeds index_entry_count {}",
            manifest.feature_count, manifest.index_entry_count
        )));
    }
    if !manifest.entries_may_duplicate_rows && manifest.feature_count != manifest.index_entry_count
    {
        return Err(GeoError::UnsupportedArtifact(format!(
            "geoM says rows do not duplicate, but feature_count {} differs from index_entry_count {}",
            manifest.feature_count, manifest.index_entry_count
        )));
    }
    Ok(())
}

fn validate_manifest_entry_count(
    manifest: &GeoArtifactManifest,
    actual: usize,
) -> Result<(), GeoError> {
    if manifest.index_entry_count != actual {
        return Err(GeoError::UnsupportedArtifact(format!(
            "geoM index_entry_count {} differs from stream entry count {actual}",
            manifest.index_entry_count
        )));
    }
    Ok(())
}

fn validate_manifest_payload_presence(
    manifest: &GeoArtifactManifest,
    has_payload: bool,
) -> Result<(), GeoError> {
    let manifest_has_payload = !matches!(manifest.payload_plan, PayloadPlan::None);
    if manifest_has_payload != has_payload {
        return Err(GeoError::UnsupportedArtifact(format!(
            "geoM payload plan {:?} disagrees with stream payload presence",
            manifest.payload_plan
        )));
    }
    Ok(())
}

/// A streamable geospatial index opened from a converted artifact.
///
/// Match on `D2` or `D3` to query with the corresponding query type. The variant
/// is chosen from the artifact's `geoM` manifest.
pub enum GeoArtifactIndex<R> {
    /// 2D artifact.
    D2(GeoArtifactIndex2D<R>),
    /// 3D artifact.
    D3(GeoArtifactIndex3D<R>),
}

impl<R> GeoArtifactIndex<R> {
    /// Return the parsed `geoM` manifest.
    pub fn manifest(&self) -> &GeoArtifactManifest {
        match self {
            GeoArtifactIndex::D2(index) => index.manifest(),
            GeoArtifactIndex::D3(index) => index.manifest(),
        }
    }

    fn num_entries(&self) -> usize {
        match self {
            GeoArtifactIndex::D2(index) => index.num_entries(),
            GeoArtifactIndex::D3(index) => index.num_entries(),
        }
    }

    fn has_payload(&self) -> bool {
        match self {
            GeoArtifactIndex::D2(index) => index.has_payload(),
            GeoArtifactIndex::D3(index) => index.has_payload(),
        }
    }

    /// Split off the reader, keeping a reusable [`GeoArtifactDirectory`]. No I/O.
    ///
    /// Cache the directory and rebuild a fresh artifact index per request with
    /// [`from_directory`](Self::from_directory) to skip the container directory,
    /// `geoM` manifest, and inner stream-directory reads on warm requests.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("places.psindex")?;
    /// let artifact = open_geo_index(SliceReader::new(bytes.clone()))?;
    /// let (directory, _reader) = artifact.into_directory();
    ///
    /// let warm = GeoArtifactIndex::from_directory(&directory, SliceReader::new(bytes))?;
    /// if let GeoArtifactIndex::D2(index) = warm {
    ///     let ids = index.search_entry_ids(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
    ///     println!("{} matching entries", ids.len());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_directory(self) -> (GeoArtifactDirectory, R) {
        match self {
            GeoArtifactIndex::D2(index) => index.into_directory(),
            GeoArtifactIndex::D3(index) => index.into_directory(),
        }
    }

    /// Rebuild an artifact index from a cached [`GeoArtifactDirectory`] and a
    /// fresh reader. No I/O: the manifest and stream directory were parsed when
    /// the artifact was first opened.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{GeoArtifactIndex, SliceReader, open_geo_index};
    /// # let bytes = std::fs::read("places.psindex")?;
    /// # let artifact = open_geo_index(SliceReader::new(bytes.clone()))?;
    /// # let (directory, _reader) = artifact.into_directory();
    /// let warm = GeoArtifactIndex::from_directory(&directory, SliceReader::new(bytes))?;
    /// println!("artifact has {} index entries", warm.manifest().index_entry_count);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_directory(dir: &GeoArtifactDirectory, reader: R) -> Result<Self, GeoError> {
        Self::from_directory_with_limits(dir, reader, StreamLimits::default())
    }

    /// [`from_directory`](Self::from_directory) with per-query [`StreamLimits`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{GeoArtifactIndex, SliceReader, StreamLimits, open_geo_index};
    /// # let bytes = std::fs::read("places.psindex")?;
    /// # let artifact = open_geo_index(SliceReader::new(bytes.clone()))?;
    /// # let (directory, _reader) = artifact.into_directory();
    /// let limits = StreamLimits {
    ///     max_read_bytes: Some(8 * 1024 * 1024),
    ///     ..StreamLimits::default()
    /// };
    /// let warm = GeoArtifactIndex::from_directory_with_limits(
    ///     &directory,
    ///     SliceReader::new(bytes),
    ///     limits,
    /// )?;
    /// println!("artifact has {} index entries", warm.manifest().index_entry_count);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_directory_with_limits(
        dir: &GeoArtifactDirectory,
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, GeoError> {
        match dir.manifest.dims.index_dims() {
            Some(2) => Ok(GeoArtifactIndex::D2(
                GeoArtifactIndex2D::from_directory_with_limits(dir, reader, limits)?,
            )),
            Some(3) => Ok(GeoArtifactIndex::D3(
                GeoArtifactIndex3D::from_directory_with_limits(dir, reader, limits)?,
            )),
            None => Err(GeoError::UnsupportedArtifact(format!(
                "cached directory has unknown coordinate dimensions {:?}",
                dir.manifest.dims
            ))),
            Some(other) => Err(GeoError::UnsupportedArtifact(format!(
                "cached directory has unsupported coordinate dimension count {other}"
            ))),
        }
    }
}

/// A 2D geospatial artifact index.
///
/// Use [`GeoArtifactIndex2D::search_matches`] when payloads are present and you
/// want decoded [`FeatureRef`] values plus typed payload data.
pub struct GeoArtifactIndex2D<R> {
    index: GeoStreamIndex2D<R>,
    manifest: GeoArtifactManifest,
}

impl<R> GeoArtifactIndex2D<R> {
    /// Return the parsed `geoM` manifest.
    pub fn manifest(&self) -> &GeoArtifactManifest {
        &self.manifest
    }

    fn num_entries(&self) -> usize {
        match &self.index {
            GeoStreamIndex2D::F64(index) => index.num_items(),
            GeoStreamIndex2D::F32(index) => index.num_items(),
        }
    }

    fn has_payload(&self) -> bool {
        match &self.index {
            GeoStreamIndex2D::F64(index) => index.has_payload(),
            GeoStreamIndex2D::F32(index) => index.has_payload(),
        }
    }

    /// Split off the reader, keeping a reusable [`GeoArtifactDirectory`]. No I/O.
    pub fn into_directory(self) -> (GeoArtifactDirectory, R) {
        let (inner, reader) = match self.index {
            GeoStreamIndex2D::F64(index) => index.into_directory(),
            GeoStreamIndex2D::F32(index) => index.into_directory(),
        };
        (
            GeoArtifactDirectory {
                inner,
                manifest: self.manifest,
            },
            reader,
        )
    }

    /// Rebuild a 2D artifact index from a cached [`GeoArtifactDirectory`] and a
    /// fresh reader. No I/O.
    pub fn from_directory(dir: &GeoArtifactDirectory, reader: R) -> Result<Self, GeoError> {
        Self::from_directory_with_limits(dir, reader, StreamLimits::default())
    }

    /// [`from_directory`](Self::from_directory) with per-query [`StreamLimits`].
    pub fn from_directory_with_limits(
        dir: &GeoArtifactDirectory,
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, GeoError> {
        if dir.manifest.dims.index_dims() != Some(2) {
            return Err(GeoError::UnsupportedArtifact(format!(
                "cached directory is not 2D (dims {:?})",
                dir.manifest.dims
            )));
        }
        let index = match dir.manifest.storage_precision {
            StoragePrecision::F64 => GeoStreamIndex2D::F64(
                StreamIndex2D::from_directory_with_limits(&dir.inner, reader, limits)?,
            ),
            StoragePrecision::F32 => GeoStreamIndex2D::F32(
                StreamIndex2DF32::from_directory_with_limits(&dir.inner, reader, limits)?,
            ),
        };
        Ok(Self {
            index,
            manifest: dir.manifest.clone(),
        })
    }

    /// Exactly filter geo matches by the geometry stored in their payloads —
    /// the post-filter step for the streaming path, with no source re-read.
    ///
    /// Index search narrows by bounding box; this keeps only the matches whose
    /// geometry actually satisfies `query` under `predicate`, removing the bbox
    /// false-positives over holes and concavities. Because it tests the geometry
    /// already fetched by [`GeoArtifactIndex2D::search_matches`], it avoids the
    /// candidate geometry re-read that `GeoDataset::filter_features` (`parquet`
    /// feature) performs.
    ///
    /// Needs a payload that carries geometry — `RowWkb` or `FeatureJson`. A
    /// `RowRef` payload stores no geometry, so it returns
    /// [`GeoError::PayloadDecode`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{
    ///     GeoArtifactIndex, GeoQuery2D, NonPlanarExactPolicy, SliceReader, SpatialPredicate,
    ///     open_geo_index,
    /// };
    /// use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};
    ///
    /// let bytes = std::fs::read("places.psi")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    ///
    /// let triangle = Polygon::new(
    ///     LineString::new(vec![
    ///         Coord { x: 0.0, y: 0.0 },
    ///         Coord { x: 10.0, y: 0.0 },
    ///         Coord { x: 0.0, y: 10.0 },
    ///         Coord { x: 0.0, y: 0.0 },
    ///     ]),
    ///     vec![],
    /// );
    ///
    /// // Bounding-box candidates from the artifact, then exact polygon filtering
    /// // on the geometry already in their payloads (no source re-read).
    /// let matches = index.search_matches(GeoQuery2D::polygon(triangle.clone()))?;
    /// let exact = index.filter_matches(
    ///     matches,
    ///     GeoQuery2D::polygon(triangle),
    ///     SpatialPredicate::Intersects,
    ///     NonPlanarExactPolicy::Reject,
    /// )?;
    /// println!("{} exact matches", exact.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    pub fn filter_matches<Q: Into<GeoQuery2D>>(
        &self,
        matches: Vec<GeoMatch>,
        query: Q,
        predicate: SpatialPredicate,
        non_planar: NonPlanarExactPolicy,
    ) -> Result<Vec<GeoMatch>, GeoError> {
        let prepared = prepare_filter_query(
            &self.manifest.encoding,
            self.manifest.edges,
            &self.manifest.selected_column,
            query.into(),
            non_planar,
        )?;
        let mut kept = Vec::new();
        for m in matches {
            if let GeoPayload::RowWkb(wkb) = &m.payload
                && let Some(matched) = exact_wkb_predicate_matches(wkb, &prepared, predicate)?
            {
                if matched {
                    kept.push(m);
                }
                continue;
            }
            let geometry = match &m.payload {
                GeoPayload::RowWkb(wkb) => decode_geo_geometry(wkb)?,
                GeoPayload::FeatureJson(feature) => feature_json_geometry(feature)?,
                GeoPayload::RowRef => {
                    return Err(GeoError::PayloadDecode(
                        "filter_matches needs a geometry payload (RowWkb or FeatureJson); RowRef has none"
                            .to_string(),
                    ));
                }
            };
            let Some(geometry) = geometry else {
                continue;
            };
            if exact_predicate_matches(&geometry, &prepared, predicate)? {
                kept.push(m);
            }
        }
        Ok(kept)
    }
}

impl<R: RangeReader> GeoArtifactIndex2D<R> {
    /// Search the underlying core index and return index entry ids.
    ///
    /// This does not decode geo payloads and therefore also works for
    /// [`PayloadPlan::None`] artifacts.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("places.psindex")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    /// let entry_ids = index.search_entry_ids(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
    /// println!("{} matching index entries", entry_ids.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_entry_ids<Q: Into<GeoQuery2D>>(&self, query: Q) -> Result<Vec<usize>, GeoError> {
        let query = query.into();
        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            return match &self.index {
                GeoStreamIndex2D::F64(index) => Ok(index.search_region(&region)?),
                GeoStreamIndex2D::F32(index) => Ok(index.search_region(&region)?),
            };
        }

        let boxes = query.candidate_boxes_2d()?;
        // Duplicates only arise across multiple candidate boxes; a single box
        // yields each entry once. Skip dedup bookkeeping in the common single-box
        // case, and dedup by entry id in O(1) (not O(K^2) `iter().any`) otherwise.
        let dedup = boxes.len() > 1;
        let mut items = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let raw = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search(bbox)?,
                GeoStreamIndex2D::F32(index) => index.search(bbox)?,
            };
            for item in raw {
                if !dedup || seen.insert(item) {
                    items.push(item);
                }
            }
        }
        Ok(items)
    }

    /// Search and return source feature references.
    ///
    /// This requires an artifact payload plan that stores feature refs
    /// (`RowRef`, `RowWkb`, or `FeatureJson`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("places.psindex")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    /// for feature in index.search_feature_refs(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    ///     println!("row {} part {:?}", feature.row_number, feature.part);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_feature_refs<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_matches(query)?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Count matching index entries without materializing ids or payloads.
    ///
    /// Works for every payload plan, including [`PayloadPlan::None`]. Counts
    /// index entries, not source features — a split feature counts once per
    /// part. Single-box and polygon queries stream the count through the core
    /// visitor; a query that expands to several candidate boxes (for example
    /// an antimeridian-crossing box) falls back to
    /// [`search_entry_ids`](Self::search_entry_ids), since the same entry can
    /// match more than one candidate box and must be counted once. There is
    /// no async variant — the async layer exposes no visitor; use
    /// `search_entry_ids_async().await?.len()` there.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("places.psindex")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    /// let count = index.count_entries(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
    /// println!("{count} matching index entries");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn count_entries<Q: Into<GeoQuery2D>>(&self, query: Q) -> Result<usize, GeoError> {
        let query = query.into();
        let mut count = 0usize;
        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            match &self.index {
                GeoStreamIndex2D::F64(index) => index.visit_region(&region, |_| count += 1)?,
                GeoStreamIndex2D::F32(index) => index.visit_region(&region, |_| count += 1)?,
            }
            return Ok(count);
        }
        let boxes = query.candidate_boxes_2d()?;
        if boxes.len() > 1 {
            return Ok(self.search_entry_ids(query)?.len());
        }
        for bbox in boxes {
            match &self.index {
                GeoStreamIndex2D::F64(index) => index.visit(bbox, |_| count += 1)?,
                GeoStreamIndex2D::F32(index) => index.visit(bbox, |_| count += 1)?,
            }
        }
        Ok(count)
    }

    /// Search and return one deduplicated [`FeatureRef`] per matched source
    /// feature.
    ///
    /// Unlike [`search_feature_refs`](Self::search_feature_refs), which is
    /// entry-level (a split feature yields one ref per index entry), this
    /// collapses split entries: results are feature-level, `part` is `None`,
    /// order is deterministic. Requires a feature-ref-bearing payload plan
    /// (`RowRef`, `RowWkb`, or `FeatureJson`); [`PayloadPlan::None`] artifacts
    /// return an error (the artifact stores no payload section).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("places.psindex")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    /// for feature in index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    ///     assert!(feature.part.is_none());
    ///     println!("row {}", feature.row_number);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_features<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_feature_matches(query)?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Search and return one [`GeoMatch`] per matched source feature.
    ///
    /// Feature-level counterpart of [`search_matches`](Self::search_matches):
    /// split index entries collapse into one match per source feature via
    /// [`GeoMatch::dedupe_by_feature`] — the lowest-part entry is kept as the
    /// representative (its `entry_id` and payload), `feature.part` is `None`.
    /// To combine deduplication with your own filtering (for example
    /// [`filter_matches`](Self::filter_matches) between search and dedupe),
    /// use [`search_matches`](Self::search_matches) plus
    /// [`GeoMatch::dedupe_by_feature`] directly.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, GeoPayload, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("places.psindex")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    /// for m in index.search_feature_matches(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    ///     assert!(m.feature.part.is_none());
    ///     if let GeoPayload::RowWkb(wkb) = &m.payload {
    ///         println!("row {}: {} WKB bytes", m.feature.row_number, wkb.len());
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_feature_matches<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatch>, GeoError> {
        let mut matches = self.search_matches(query)?;
        GeoMatch::dedupe_by_feature(&mut matches);
        Ok(matches)
    }

    /// Search and return decoded geo matches.
    ///
    /// Each match includes the index entry id, the source [`FeatureRef`], and
    /// the decoded [`GeoPayload`] described by the artifact manifest.
    ///
    /// A [`GeoQuery2D::Polygon`] query prunes subtrees that fall outside the
    /// polygon during the streamed descent, so it fetches only the leaves the
    /// polygon overlaps — less data than its bounding box. Box and
    /// spherical-radius queries narrow by candidate bounding boxes.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, GeoPayload, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("cities.psi")?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    /// for m in index.search_matches(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    ///     match &m.payload {
    ///         GeoPayload::RowWkb(wkb) => {
    ///             println!("{}: {} WKB bytes", m.feature.row_number, wkb.len())
    ///         }
    ///         GeoPayload::FeatureJson(feature) => println!("{}: {feature}", m.feature.row_number),
    ///         GeoPayload::RowRef => println!("{}: no geometry payload", m.feature.row_number),
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_matches<Q: Into<GeoQuery2D>>(&self, query: Q) -> Result<Vec<GeoMatch>, GeoError> {
        let query = query.into();
        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            let raw = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search_payloads_region(&region)?,
                GeoStreamIndex2D::F32(index) => index.search_payloads_region(&region)?,
            };
            return decode_matches(&self.manifest.payload_plan, raw);
        }

        let boxes = query.candidate_boxes_2d()?;
        // Duplicates only arise across multiple candidate boxes; a single box
        // yields each entry once. Skip dedup bookkeeping in the common single-box
        // case, and dedup by entry id in O(1) (not O(K^2) `iter().any`) otherwise.
        let dedup = boxes.len() > 1;
        let mut decoded = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let raw = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search_payloads(bbox)?,
                GeoStreamIndex2D::F32(index) => index.search_payloads(bbox)?,
            };
            for m in decode_matches(&self.manifest.payload_plan, raw)? {
                if !dedup || seen.insert(m.entry_id) {
                    decoded.push(m);
                }
            }
        }
        Ok(decoded)
    }

    /// Search and return lightweight [`GeoMatchHeader`] records — identity and
    /// payload size per matched index entry, without reading payload bodies.
    ///
    /// Headers carry everything sorting, deduplication, and pagination need;
    /// feed a page of them to [`fetch_matches`](Self::fetch_matches) to
    /// materialize full [`GeoMatch`] values for just that page. Supported for
    /// `RowRef`, `RowWkb`, and current `FeatureJson` artifacts, whose payloads
    /// start with the fixed feature-ref record. Legacy raw-JSON `FeatureJson`
    /// artifacts remain readable through [`search_matches`](Self::search_matches)
    /// but must be rebuilt for header-only search. [`PayloadPlan::None`]
    /// artifacts return [`GeoError::UnsupportedArtifact`].
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index_geo::{
    ///     open_geo_index, open_geojson_slice, Box2D, ConvertRequest, GeoArtifactIndex,
    ///     GeoMatchHeader, SliceReader,
    /// };
    ///
    /// let geojson = br#"{
    ///   "type": "FeatureCollection",
    ///   "features": [
    ///     {
    ///       "type": "Feature",
    ///       "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
    ///       "properties": {"name": "one"}
    ///     },
    ///     {
    ///       "type": "Feature",
    ///       "geometry": {"type": "Point", "coordinates": [5.0, 6.0]},
    ///       "properties": {"name": "two"}
    ///     }
    ///   ]
    /// }"#;
    ///
    /// let mut source = open_geojson_slice(geojson)?;
    /// let bytes = source.convert(ConvertRequest::default())?;
    /// let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 2D artifact");
    /// };
    ///
    /// let mut headers = index.search_match_headers(Box2D::new(0.0, 0.0, 10.0, 10.0))?;
    /// GeoMatchHeader::dedupe_by_feature(&mut headers);
    ///
    /// let page = &headers[..headers.len().min(1)];
    /// let matches = index.fetch_matches(page)?;
    /// assert_eq!(matches.len(), page.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_match_headers<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatchHeader>, GeoError> {
        ensure_header_capable_plan(&self.manifest.payload_plan)?;
        let query = query.into();
        let mut headers = Vec::new();
        let mut short_payload = false;
        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            let collect =
                |p: PayloadPrefix<'_>| collect_header(p, &mut headers, &mut short_payload);
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index.visit_payload_prefixes_region(&region, FEATURE_REF_RECORD_LEN, collect)?
                }
                GeoStreamIndex2D::F32(index) => {
                    index.visit_payload_prefixes_region(&region, FEATURE_REF_RECORD_LEN, collect)?
                }
            }
            return finish_headers(headers, short_payload);
        }

        let boxes = query.candidate_boxes_2d()?;
        let dedup = boxes.len() > 1;
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let mut batch = Vec::new();
            let collect = |p: PayloadPrefix<'_>| collect_header(p, &mut batch, &mut short_payload);
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index.visit_payload_prefixes(bbox, FEATURE_REF_RECORD_LEN, collect)?
                }
                GeoStreamIndex2D::F32(index) => {
                    index.visit_payload_prefixes(bbox, FEATURE_REF_RECORD_LEN, collect)?
                }
            }
            for header in batch {
                if !dedup || seen.insert(header.entry_id) {
                    headers.push(header);
                }
            }
        }
        finish_headers(headers, short_payload)
    }

    /// Fetch and decode full [`GeoMatch`] values for the given headers
    /// (typically one page), preserving the input header order.
    ///
    /// `RowRef` headers rebuild their matches with no I/O — the feature ref is
    /// the whole payload. `RowWkb` headers fetch payload bodies for exactly
    /// the given headers' leaf ranks, in coalesced reads.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{Box2D, GeoArtifactIndex, SliceReader, open_geo_index};
    /// # let bytes = std::fs::read("places.psindex")?;
    /// # let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    /// #     panic!("expected a 2D artifact");
    /// # };
    /// let headers = index.search_match_headers(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
    /// let page = &headers[..headers.len().min(100)];
    /// let matches = index.fetch_matches(page)?;
    /// assert_eq!(matches.len(), page.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fetch_matches(&self, headers: &[GeoMatchHeader]) -> Result<Vec<GeoMatch>, GeoError> {
        match &self.manifest.payload_plan {
            PayloadPlan::RowRef => Ok(headers
                .iter()
                .map(GeoMatchHeader::to_row_ref_match)
                .collect()),
            PayloadPlan::RowWkb | PayloadPlan::FeatureJson { .. } => {
                let ranks: Vec<usize> = headers.iter().map(|h| h.leaf_rank).collect();
                let mut by_rank = std::collections::HashMap::new();
                let collect = |rank: usize, blob: &[u8]| {
                    by_rank.insert(rank, blob.to_vec());
                };
                match &self.index {
                    GeoStreamIndex2D::F64(index) => {
                        index.visit_payloads_at_ranks(&ranks, collect)?
                    }
                    GeoStreamIndex2D::F32(index) => {
                        index.visit_payloads_at_ranks(&ranks, collect)?
                    }
                }
                assemble_matches(&self.manifest.payload_plan, headers, &by_rank)
            }
            plan => Err(GeoError::UnsupportedArtifact(format!(
                "fetch_matches supports RowRef, RowWkb, and FeatureJson payload plans, not {plan:?}"
            ))),
        }
    }
}

#[cfg(feature = "async")]
impl<R: AsyncRangeReader> GeoArtifactIndex2D<R> {
    /// Search the underlying core index over async range I/O and return compact
    /// index entry ids.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box2D, GeoArtifactIndex, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D2(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 2D artifact");
    /// # };
    /// let entry_ids = index
    ///     .search_entry_ids_async(Box2D::new(-10.0, 35.0, 20.0, 60.0))
    ///     .await?;
    /// println!("{} matching index entries", entry_ids.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_entry_ids_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<usize>, GeoError> {
        let query = query.into();
        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            return match &self.index {
                GeoStreamIndex2D::F64(index) => Ok(index.search_region_async(&region).await?),
                GeoStreamIndex2D::F32(index) => Ok(index.search_region_async(&region).await?),
            };
        }

        let boxes = query.candidate_boxes_2d()?;
        // Same dedup rationale as the sync `search_entry_ids`: single-box queries
        // need none; multi-box (for example antimeridian-split) dedup in O(1).
        let dedup = boxes.len() > 1;
        let mut items = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let raw = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search_async(bbox).await?,
                GeoStreamIndex2D::F32(index) => index.search_async(bbox).await?,
            };
            for item in raw {
                if !dedup || seen.insert(item) {
                    items.push(item);
                }
            }
        }
        Ok(items)
    }

    /// Search and return lightweight async [`GeoMatchHeader`] records without
    /// reading payload bodies.
    pub async fn search_match_headers_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatchHeader>, GeoError> {
        ensure_header_capable_plan(&self.manifest.payload_plan)?;
        let query = query.into();
        let mut headers = Vec::new();
        let mut short_payload = false;

        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            let collect = |p: PayloadPrefix<'_>| {
                collect_header(p, &mut headers, &mut short_payload);
            };
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index
                        .visit_payload_prefixes_region_async(
                            &region,
                            FEATURE_REF_RECORD_LEN,
                            collect,
                        )
                        .await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index
                        .visit_payload_prefixes_region_async(
                            &region,
                            FEATURE_REF_RECORD_LEN,
                            collect,
                        )
                        .await?
                }
            }
            return finish_headers(headers, short_payload);
        }

        let boxes = query.candidate_boxes_2d()?;
        let dedup = boxes.len() > 1;
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let mut batch = Vec::new();
            let collect = |p: PayloadPrefix<'_>| {
                collect_header(p, &mut batch, &mut short_payload);
            };
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index
                        .visit_payload_prefixes_async(bbox, FEATURE_REF_RECORD_LEN, collect)
                        .await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index
                        .visit_payload_prefixes_async(bbox, FEATURE_REF_RECORD_LEN, collect)
                        .await?
                }
            }
            for header in batch {
                if !dedup || seen.insert(header.entry_id) {
                    headers.push(header);
                }
            }
        }
        finish_headers(headers, short_payload)
    }

    /// Search and return one deterministic entry-level page of match headers
    /// together with the total number of matching entries.
    ///
    /// Headers use the same feature / part / entry-id order as
    /// [`GeoMatchHeader::sort_by_entry`]. For a single candidate box or a
    /// polygon query, the search retains at most `offset + limit` headers while
    /// counting all matches. This method does not collapse split entries to
    /// feature level; use [`search_match_headers_async`](Self::search_match_headers_async)
    /// plus [`GeoMatchHeader::dedupe_by_feature`] when feature-level results are
    /// required.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{AsyncRangeReader, Box2D, GeoArtifactIndex2D};
    /// # async fn query<R: AsyncRangeReader>(
    /// #     index: &GeoArtifactIndex2D<R>,
    /// # ) -> Result<(), Box<dyn std::error::Error>> {
    /// let page = index
    ///     .search_match_headers_page_async(
    ///         Box2D::new(-10.0, 35.0, 20.0, 60.0),
    ///         0,
    ///         100,
    ///     )
    ///     .await?;
    /// let matches = index.fetch_matches_async(&page.headers).await?;
    /// assert_eq!(matches.len(), page.headers.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_match_headers_page_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
        offset: usize,
        limit: usize,
    ) -> Result<GeoMatchHeaderPage, GeoError> {
        ensure_header_capable_plan(&self.manifest.payload_plan)?;
        let query = query.into();
        let mut page = HeaderPageCollector::new(offset, limit);
        let mut short_payload = false;

        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            let collect = |p: PayloadPrefix<'_>| {
                collect_header_page(p, &mut page, &mut short_payload);
            };
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index
                        .visit_payload_prefixes_region_async(
                            &region,
                            FEATURE_REF_RECORD_LEN,
                            collect,
                        )
                        .await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index
                        .visit_payload_prefixes_region_async(
                            &region,
                            FEATURE_REF_RECORD_LEN,
                            collect,
                        )
                        .await?
                }
            }
            return finish_match_header_page(page, short_payload);
        }

        let boxes = query.candidate_boxes_2d()?;
        let dedup = boxes.len() > 1;
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let collect = |p: PayloadPrefix<'_>| match GeoMatchHeader::from_prefix(p) {
                Some(header) => {
                    if !dedup || seen.insert(header.entry_id) {
                        page.push(header);
                    }
                }
                None => short_payload = true,
            };
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index
                        .visit_payload_prefixes_async(bbox, FEATURE_REF_RECORD_LEN, collect)
                        .await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index
                        .visit_payload_prefixes_async(bbox, FEATURE_REF_RECORD_LEN, collect)
                        .await?
                }
            }
        }
        finish_match_header_page(page, short_payload)
    }

    /// Fetch and decode full async [`GeoMatch`] values for headers returned by
    /// [`search_match_headers_async`](Self::search_match_headers_async).
    pub async fn fetch_matches_async(
        &self,
        headers: &[GeoMatchHeader],
    ) -> Result<Vec<GeoMatch>, GeoError> {
        match &self.manifest.payload_plan {
            PayloadPlan::RowRef => Ok(headers
                .iter()
                .map(GeoMatchHeader::to_row_ref_match)
                .collect()),
            PayloadPlan::RowWkb | PayloadPlan::FeatureJson { .. } => {
                let ranks: Vec<usize> = headers.iter().map(|h| h.leaf_rank).collect();
                let mut by_rank = std::collections::HashMap::new();
                let collect = |rank: usize, blob: &[u8]| {
                    by_rank.insert(rank, blob.to_vec());
                };
                match &self.index {
                    GeoStreamIndex2D::F64(index) => {
                        index.visit_payloads_at_ranks_async(&ranks, collect).await?
                    }
                    GeoStreamIndex2D::F32(index) => {
                        index.visit_payloads_at_ranks_async(&ranks, collect).await?
                    }
                }
                assemble_matches(&self.manifest.payload_plan, headers, &by_rank)
            }
            plan => Err(GeoError::UnsupportedArtifact(format!(
                "fetch_matches_async supports RowRef, RowWkb, and FeatureJson payload plans, not {plan:?}"
            ))),
        }
    }

    /// Search and return payload locations without reading payload bodies or
    /// decoding feature refs.
    pub async fn search_payload_headers_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoPayloadHeader>, GeoError> {
        if matches!(self.manifest.payload_plan, PayloadPlan::None) {
            return Err(GeoError::UnsupportedArtifact(
                "search_payload_headers_async needs an artifact with payloads".to_string(),
            ));
        }
        let query = query.into();
        let mut headers = Vec::new();

        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            let collect = |p: PayloadPrefix<'_>| {
                headers.push(GeoPayloadHeader::from_prefix(p));
            };
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index
                        .visit_payload_prefixes_region_async(&region, 0, collect)
                        .await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index
                        .visit_payload_prefixes_region_async(&region, 0, collect)
                        .await?
                }
            }
            return Ok(headers);
        }

        let boxes = query.candidate_boxes_2d()?;
        let dedup = boxes.len() > 1;
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let mut batch = Vec::new();
            let collect = |p: PayloadPrefix<'_>| {
                batch.push(GeoPayloadHeader::from_prefix(p));
            };
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index.visit_payload_prefixes_async(bbox, 0, collect).await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index.visit_payload_prefixes_async(bbox, 0, collect).await?
                }
            }
            for header in batch {
                if !dedup || seen.insert(header.entry_id) {
                    headers.push(header);
                }
            }
        }
        Ok(headers)
    }

    /// Search and return one deterministic page of payload headers together
    /// with the total number of matching entries.
    ///
    /// Headers are ordered by `entry_id`. For a single candidate box or a
    /// polygon query, the search retains at most `offset + limit` headers while
    /// counting all matches, instead of collecting every matching header before
    /// pagination. Queries split into multiple candidate boxes additionally
    /// retain matched entry ids to remove cross-box duplicates.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{AsyncRangeReader, Box2D, GeoArtifactIndex2D};
    /// # async fn query<R: AsyncRangeReader>(
    /// #     index: &GeoArtifactIndex2D<R>,
    /// # ) -> Result<(), Box<dyn std::error::Error>> {
    /// let page = index
    ///     .search_payload_headers_page_async(
    ///         Box2D::new(-10.0, 35.0, 20.0, 60.0),
    ///         0,
    ///         100,
    ///     )
    ///     .await?;
    /// let matches = index
    ///     .fetch_payload_header_matches_async(&page.headers)
    ///     .await?;
    /// assert_eq!(matches.len(), page.headers.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_payload_headers_page_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
        offset: usize,
        limit: usize,
    ) -> Result<GeoPayloadHeaderPage, GeoError> {
        if matches!(self.manifest.payload_plan, PayloadPlan::None) {
            return Err(GeoError::UnsupportedArtifact(
                "search_payload_headers_page_async needs an artifact with payloads".to_string(),
            ));
        }
        let query = query.into();
        let mut page = HeaderPageCollector::new(offset, limit);

        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            let collect = |p: PayloadPrefix<'_>| page.push(GeoPayloadHeader::from_prefix(p));
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index
                        .visit_payload_prefixes_region_async(&region, 0, collect)
                        .await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index
                        .visit_payload_prefixes_region_async(&region, 0, collect)
                        .await?
                }
            }
            return Ok(finish_payload_header_page(page));
        }

        let boxes = query.candidate_boxes_2d()?;
        let dedup = boxes.len() > 1;
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let collect = |p: PayloadPrefix<'_>| {
                let header = GeoPayloadHeader::from_prefix(p);
                if !dedup || seen.insert(header.entry_id) {
                    page.push(header);
                }
            };
            match &self.index {
                GeoStreamIndex2D::F64(index) => {
                    index.visit_payload_prefixes_async(bbox, 0, collect).await?
                }
                GeoStreamIndex2D::F32(index) => {
                    index.visit_payload_prefixes_async(bbox, 0, collect).await?
                }
            }
        }
        Ok(finish_payload_header_page(page))
    }

    /// Fetch full matches for payload headers returned by
    /// [`search_payload_headers_async`](Self::search_payload_headers_async).
    pub async fn fetch_payload_header_matches_async(
        &self,
        headers: &[GeoPayloadHeader],
    ) -> Result<Vec<GeoMatch>, GeoError> {
        match &self.manifest.payload_plan {
            PayloadPlan::RowRef | PayloadPlan::RowWkb | PayloadPlan::FeatureJson { .. } => {
                let ranks: Vec<usize> = headers.iter().map(|h| h.leaf_rank).collect();
                let mut by_rank = std::collections::HashMap::new();
                let collect = |rank: usize, blob: &[u8]| {
                    by_rank.insert(rank, blob.to_vec());
                };
                match &self.index {
                    GeoStreamIndex2D::F64(index) => {
                        index.visit_payloads_at_ranks_async(&ranks, collect).await?
                    }
                    GeoStreamIndex2D::F32(index) => {
                        index.visit_payloads_at_ranks_async(&ranks, collect).await?
                    }
                }
                headers
                    .iter()
                    .map(|header| {
                        let payload =
                            payload_at_rank(&by_rank, header.leaf_rank, header.payload_len)?;
                        let (feature, payload) =
                            decode_payload(&self.manifest.payload_plan, payload)?;
                        Ok(GeoMatch {
                            entry_id: header.entry_id,
                            feature,
                            payload,
                        })
                    })
                    .collect()
            }
            PayloadPlan::None => Err(GeoError::UnsupportedArtifact(
                "fetch_payload_header_matches_async needs an artifact with payloads".to_string(),
            )),
        }
    }

    /// Search over async range I/O and return source feature references.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box2D, GeoArtifactIndex, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D2(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 2D artifact");
    /// # };
    /// let features = index
    ///     .search_feature_refs_async(Box2D::new(-10.0, 35.0, 20.0, 60.0))
    ///     .await?;
    /// println!("{} matching feature refs", features.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_feature_refs_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_matches_async(query)
            .await?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Search over async range I/O and return one deduplicated [`FeatureRef`]
    /// per matched source feature.
    ///
    /// Async counterpart of [`search_features`](Self::search_features): split
    /// index entries collapse, `part` is `None`, order is deterministic.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box2D, GeoArtifactIndex, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D2(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 2D artifact");
    /// # };
    /// for feature in index
    ///     .search_features_async(Box2D::new(-10.0, 35.0, 20.0, 60.0))
    ///     .await?
    /// {
    ///     assert!(feature.part.is_none());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_features_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_feature_matches_async(query)
            .await?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Search over async range I/O and return one [`GeoMatch`] per matched
    /// source feature.
    ///
    /// Async counterpart of
    /// [`search_feature_matches`](Self::search_feature_matches).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box2D, GeoArtifactIndex, GeoPayload, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D2(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 2D artifact");
    /// # };
    /// for m in index
    ///     .search_feature_matches_async(Box2D::new(-10.0, 35.0, 20.0, 60.0))
    ///     .await?
    /// {
    ///     assert!(m.feature.part.is_none());
    ///     if let GeoPayload::RowWkb(wkb) = &m.payload {
    ///         println!("row {}: {} WKB bytes", m.feature.row_number, wkb.len());
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_feature_matches_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatch>, GeoError> {
        let mut matches = self.search_matches_async(query).await?;
        GeoMatch::dedupe_by_feature(&mut matches);
        Ok(matches)
    }

    /// Search over async range I/O and return decoded geo matches.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box2D, GeoArtifactIndex, GeoPayload, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D2(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 2D artifact");
    /// # };
    /// for m in index
    ///     .search_matches_async(Box2D::new(-10.0, 35.0, 20.0, 60.0))
    ///     .await?
    /// {
    ///     match &m.payload {
    ///         GeoPayload::RowWkb(wkb) => println!("{} WKB bytes", wkb.len()),
    ///         GeoPayload::FeatureJson(feature) => println!("{feature}"),
    ///         GeoPayload::RowRef => println!("row {}", m.feature.row_number),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_matches_async<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatch>, GeoError> {
        let query = query.into();
        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            ensure_polygon_query_not_empty(multi_polygon)?;
            let region = PolygonRegion(multi_polygon);
            let raw = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search_payloads_region_async(&region).await?,
                GeoStreamIndex2D::F32(index) => index.search_payloads_region_async(&region).await?,
            };
            return decode_matches(&self.manifest.payload_plan, raw);
        }

        let boxes = query.candidate_boxes_2d()?;
        let dedup = boxes.len() > 1;
        let mut decoded = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let raw = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search_payloads_async(bbox).await?,
                GeoStreamIndex2D::F32(index) => index.search_payloads_async(bbox).await?,
            };
            for m in decode_matches(&self.manifest.payload_plan, raw)? {
                if !dedup || seen.insert(m.entry_id) {
                    decoded.push(m);
                }
            }
        }
        Ok(decoded)
    }
}

fn ensure_polygon_query_not_empty(multi_polygon: &MultiPolygon<f64>) -> Result<(), GeoError> {
    multi_polygon
        .bounding_rect()
        .ok_or(GeoError::EmptyQueryPolygon)?;
    Ok(())
}

/// Decode the geometry of a `FeatureJson` payload (a GeoJSON `Feature`) into
/// `geo_types`. Returns `Ok(None)` for a missing or null geometry member.
fn feature_json_geometry(
    feature: &serde_json::Value,
) -> Result<Option<geo_types::Geometry<f64>>, GeoError> {
    let Some(geometry) = feature.get("geometry") else {
        return Ok(None);
    };
    if geometry.is_null() {
        return Ok(None);
    }
    let json =
        serde_json::to_string(geometry).map_err(|e| GeoError::PayloadDecode(e.to_string()))?;
    GeoJson(&json)
        .to_geo()
        .map(Some)
        .map_err(|e| GeoError::PayloadDecode(e.to_string()))
}

/// A 3D geospatial artifact index.
pub struct GeoArtifactIndex3D<R> {
    index: GeoStreamIndex3D<R>,
    manifest: GeoArtifactManifest,
}

impl<R> GeoArtifactIndex3D<R> {
    /// Return the parsed `geoM` manifest.
    pub fn manifest(&self) -> &GeoArtifactManifest {
        &self.manifest
    }

    fn num_entries(&self) -> usize {
        match &self.index {
            GeoStreamIndex3D::F64(index) => index.num_items(),
            GeoStreamIndex3D::F32(index) => index.num_items(),
        }
    }

    fn has_payload(&self) -> bool {
        match &self.index {
            GeoStreamIndex3D::F64(index) => index.has_payload(),
            GeoStreamIndex3D::F32(index) => index.has_payload(),
        }
    }

    /// Split off the reader, keeping a reusable [`GeoArtifactDirectory`]. No I/O.
    pub fn into_directory(self) -> (GeoArtifactDirectory, R) {
        let (inner, reader) = match self.index {
            GeoStreamIndex3D::F64(index) => index.into_directory(),
            GeoStreamIndex3D::F32(index) => index.into_directory(),
        };
        (
            GeoArtifactDirectory {
                inner,
                manifest: self.manifest,
            },
            reader,
        )
    }

    /// Rebuild a 3D artifact index from a cached [`GeoArtifactDirectory`] and a
    /// fresh reader. No I/O.
    pub fn from_directory(dir: &GeoArtifactDirectory, reader: R) -> Result<Self, GeoError> {
        Self::from_directory_with_limits(dir, reader, StreamLimits::default())
    }

    /// [`from_directory`](Self::from_directory) with per-query [`StreamLimits`].
    pub fn from_directory_with_limits(
        dir: &GeoArtifactDirectory,
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, GeoError> {
        if dir.manifest.dims.index_dims() != Some(3) {
            return Err(GeoError::UnsupportedArtifact(format!(
                "cached directory is not 3D (dims {:?})",
                dir.manifest.dims
            )));
        }
        let index = match dir.manifest.storage_precision {
            StoragePrecision::F64 => GeoStreamIndex3D::F64(
                StreamIndex3D::from_directory_with_limits(&dir.inner, reader, limits)?,
            ),
            StoragePrecision::F32 => GeoStreamIndex3D::F32(
                StreamIndex3DF32::from_directory_with_limits(&dir.inner, reader, limits)?,
            ),
        };
        Ok(Self {
            index,
            manifest: dir.manifest.clone(),
        })
    }
}

/// Reusable, reader-independent metadata for an opened geospatial artifact.
///
/// This caches the inner core [`StreamDirectory`] plus the parsed `geoM`
/// manifest. Split one off with [`GeoArtifactIndex::into_directory`], then
/// rebuild a fresh index from it with [`GeoArtifactIndex::from_directory`] and a
/// new reader. Reattaching performs no I/O, so a server or worker can pay the
/// container, manifest, and stream-directory reads once per warm cache entry.
#[derive(Clone)]
pub struct GeoArtifactDirectory {
    inner: StreamDirectory,
    manifest: GeoArtifactManifest,
}

impl GeoArtifactDirectory {
    /// Return the parsed `geoM` manifest cached with the directory.
    pub fn manifest(&self) -> &GeoArtifactManifest {
        &self.manifest
    }

    /// Number of index entries.
    pub fn num_entries(&self) -> usize {
        self.inner.num_items()
    }

    /// Whether the artifact has no index entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Packed node size of the artifact index.
    pub fn node_size(&self) -> usize {
        self.inner.node_size()
    }

    /// Whether the artifact carries a payload section.
    pub fn has_payload(&self) -> bool {
        self.inner.has_payload()
    }
}

impl<R: RangeReader> GeoArtifactIndex3D<R> {
    /// Search the underlying core index and return index entry ids.
    ///
    /// This does not decode geo payloads and therefore also works for
    /// [`PayloadPlan::None`] artifacts.
    ///
    /// A [`GeoQuery3D::Frustum3D`] query prunes subtrees that fall outside
    /// the frustum during the streamed descent, for both `f64`- and
    /// `f32`-precision artifacts.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box3D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("elevations.psindex")?;
    /// let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 3D artifact");
    /// };
    /// let entry_ids = index.search_entry_ids(Box3D::new(
    ///     -10.0, 35.0, 0.0, 20.0, 60.0, 100.0,
    /// ))?;
    /// println!("{} matching index entries", entry_ids.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_entry_ids<Q: Into<GeoQuery3D>>(&self, query: Q) -> Result<Vec<usize>, GeoError> {
        match query.into() {
            GeoQuery3D::Box3D(bbox) => match &self.index {
                GeoStreamIndex3D::F64(index) => Ok(index.search(bbox)?),
                GeoStreamIndex3D::F32(index) => Ok(index.search(bbox)?),
            },
            GeoQuery3D::Frustum3D(frustum) => match &self.index {
                GeoStreamIndex3D::F64(index) => Ok(index.search_region(&frustum)?),
                GeoStreamIndex3D::F32(index) => Ok(index.search_region(&frustum)?),
            },
        }
    }

    /// Search and return source feature references.
    ///
    /// This requires an artifact payload plan that stores feature refs
    /// (`RowRef`, `RowWkb`, or `FeatureJson`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box3D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("elevations.psindex")?;
    /// let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 3D artifact");
    /// };
    /// for feature in index.search_feature_refs(Box3D::new(
    ///     -10.0, 35.0, 0.0, 20.0, 60.0, 100.0,
    /// ))? {
    ///     println!("row {} part {:?}", feature.row_number, feature.part);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_feature_refs<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_matches(query)?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Count matching index entries without materializing ids or payloads;
    /// the 3D counterpart of [`GeoArtifactIndex2D::count_entries`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box3D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("elevations.psindex")?;
    /// let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 3D artifact");
    /// };
    /// let count = index.count_entries(Box3D::new(
    ///     -10.0, 35.0, 0.0, 20.0, 60.0, 100.0,
    /// ))?;
    /// println!("{count} matching index entries");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn count_entries<Q: Into<GeoQuery3D>>(&self, query: Q) -> Result<usize, GeoError> {
        let mut count = 0usize;
        match query.into() {
            GeoQuery3D::Box3D(bbox) => match &self.index {
                GeoStreamIndex3D::F64(index) => index.visit(bbox, |_| count += 1)?,
                GeoStreamIndex3D::F32(index) => index.visit(bbox, |_| count += 1)?,
            },
            GeoQuery3D::Frustum3D(frustum) => match &self.index {
                GeoStreamIndex3D::F64(index) => index.visit_region(&frustum, |_| count += 1)?,
                GeoStreamIndex3D::F32(index) => index.visit_region(&frustum, |_| count += 1)?,
            },
        }
        Ok(count)
    }

    /// Search and return one deduplicated [`FeatureRef`] per matched source
    /// feature.
    ///
    /// Unlike [`search_feature_refs`](Self::search_feature_refs), which is
    /// entry-level, this collapses split index entries: `part` is `None`,
    /// order is deterministic. [`PayloadPlan::None`] artifacts return an
    /// error (the artifact stores no payload section).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box3D, GeoArtifactIndex, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("elevations.psindex")?;
    /// let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 3D artifact");
    /// };
    /// for feature in index.search_features(Box3D::new(
    ///     -10.0, 35.0, 0.0, 20.0, 60.0, 100.0,
    /// ))? {
    ///     assert!(feature.part.is_none());
    ///     println!("row {}", feature.row_number);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_features<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_feature_matches(query)?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Search and return one [`GeoMatch`] per matched source feature.
    ///
    /// Feature-level counterpart of [`search_matches`](Self::search_matches);
    /// see [`GeoMatch::dedupe_by_feature`] for the collapse rules.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box3D, GeoArtifactIndex, GeoPayload, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("elevations.psindex")?;
    /// let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 3D artifact");
    /// };
    /// for m in index.search_feature_matches(Box3D::new(
    ///     -10.0, 35.0, 0.0, 20.0, 60.0, 100.0,
    /// ))? {
    ///     assert!(m.feature.part.is_none());
    ///     if let GeoPayload::RowWkb(wkb) = &m.payload {
    ///         println!("row {}: {} WKB bytes", m.feature.row_number, wkb.len());
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_feature_matches<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatch>, GeoError> {
        let mut matches = self.search_matches(query)?;
        GeoMatch::dedupe_by_feature(&mut matches);
        Ok(matches)
    }

    /// Search and return decoded geo matches.
    ///
    /// Each match includes the index entry id, the source [`FeatureRef`], and
    /// the decoded [`GeoPayload`] described by the artifact manifest.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{Box3D, GeoArtifactIndex, GeoPayload, SliceReader, open_geo_index};
    ///
    /// let bytes = std::fs::read("elevations.psi")?;
    /// let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 3D artifact");
    /// };
    /// for m in index.search_matches(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))? {
    ///     match &m.payload {
    ///         GeoPayload::RowWkb(wkb) => {
    ///             println!("{}: {} WKB bytes", m.feature.row_number, wkb.len())
    ///         }
    ///         GeoPayload::FeatureJson(feature) => println!("{}: {feature}", m.feature.row_number),
    ///         GeoPayload::RowRef => println!("{}: no geometry payload", m.feature.row_number),
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_matches<Q: Into<GeoQuery3D>>(&self, query: Q) -> Result<Vec<GeoMatch>, GeoError> {
        match query.into() {
            GeoQuery3D::Box3D(bbox) => {
                let raw = match &self.index {
                    GeoStreamIndex3D::F64(index) => index.search_payloads(bbox)?,
                    GeoStreamIndex3D::F32(index) => index.search_payloads(bbox)?,
                };
                decode_matches(&self.manifest.payload_plan, raw)
            }
            GeoQuery3D::Frustum3D(frustum) => {
                let raw = match &self.index {
                    GeoStreamIndex3D::F64(index) => index.search_payloads_region(&frustum)?,
                    GeoStreamIndex3D::F32(index) => index.search_payloads_region(&frustum)?,
                };
                decode_matches(&self.manifest.payload_plan, raw)
            }
        }
    }

    /// Search and return lightweight [`GeoMatchHeader`] records; the 3D
    /// counterpart of [`GeoArtifactIndex2D::search_match_headers`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use packed_spatial_index_geo::{
    ///     Box3D, GeoArtifactIndex, GeoMatchHeader, SliceReader, open_geo_index,
    /// };
    ///
    /// let bytes = std::fs::read("elevations.psindex")?;
    /// let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    ///     panic!("expected a 3D artifact");
    /// };
    /// let mut headers = index.search_match_headers(Box3D::new(
    ///     -10.0, 35.0, 0.0, 20.0, 60.0, 100.0,
    /// ))?;
    /// GeoMatchHeader::dedupe_by_feature(&mut headers);
    ///
    /// let page = &headers[..headers.len().min(100)];
    /// let matches = index.fetch_matches(page)?;
    /// assert_eq!(matches.len(), page.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_match_headers<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatchHeader>, GeoError> {
        ensure_header_capable_plan(&self.manifest.payload_plan)?;
        let mut headers = Vec::new();
        let mut short_payload = false;
        let collect = |p: PayloadPrefix<'_>| collect_header(p, &mut headers, &mut short_payload);
        match query.into() {
            GeoQuery3D::Box3D(bbox) => match &self.index {
                GeoStreamIndex3D::F64(index) => {
                    index.visit_payload_prefixes(bbox, FEATURE_REF_RECORD_LEN, collect)?
                }
                GeoStreamIndex3D::F32(index) => {
                    index.visit_payload_prefixes(bbox, FEATURE_REF_RECORD_LEN, collect)?
                }
            },
            GeoQuery3D::Frustum3D(frustum) => match &self.index {
                GeoStreamIndex3D::F64(index) => index.visit_payload_prefixes_region(
                    &frustum,
                    FEATURE_REF_RECORD_LEN,
                    collect,
                )?,
                GeoStreamIndex3D::F32(index) => index.visit_payload_prefixes_region(
                    &frustum,
                    FEATURE_REF_RECORD_LEN,
                    collect,
                )?,
            },
        }
        finish_headers(headers, short_payload)
    }

    /// Fetch and decode full [`GeoMatch`] values for the given headers; the 3D
    /// counterpart of [`GeoArtifactIndex2D::fetch_matches`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{Box3D, GeoArtifactIndex, SliceReader, open_geo_index};
    /// # let bytes = std::fs::read("elevations.psindex")?;
    /// # let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes))? else {
    /// #     panic!("expected a 3D artifact");
    /// # };
    /// let headers = index.search_match_headers(Box3D::new(
    ///     -10.0, 35.0, 0.0, 20.0, 60.0, 100.0,
    /// ))?;
    /// let page = &headers[..headers.len().min(100)];
    /// let matches = index.fetch_matches(page)?;
    /// assert_eq!(matches.len(), page.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fetch_matches(&self, headers: &[GeoMatchHeader]) -> Result<Vec<GeoMatch>, GeoError> {
        match &self.manifest.payload_plan {
            PayloadPlan::RowRef => Ok(headers
                .iter()
                .map(GeoMatchHeader::to_row_ref_match)
                .collect()),
            PayloadPlan::RowWkb | PayloadPlan::FeatureJson { .. } => {
                let ranks: Vec<usize> = headers.iter().map(|h| h.leaf_rank).collect();
                let mut by_rank = std::collections::HashMap::new();
                let collect = |rank: usize, blob: &[u8]| {
                    by_rank.insert(rank, blob.to_vec());
                };
                match &self.index {
                    GeoStreamIndex3D::F64(index) => {
                        index.visit_payloads_at_ranks(&ranks, collect)?
                    }
                    GeoStreamIndex3D::F32(index) => {
                        index.visit_payloads_at_ranks(&ranks, collect)?
                    }
                }
                assemble_matches(&self.manifest.payload_plan, headers, &by_rank)
            }
            plan => Err(GeoError::UnsupportedArtifact(format!(
                "fetch_matches supports RowRef, RowWkb, and FeatureJson payload plans, not {plan:?}"
            ))),
        }
    }
}

/// Reject payload plans whose entries cannot be described by a header.
fn ensure_header_capable_plan(plan: &PayloadPlan) -> Result<(), GeoError> {
    match plan {
        PayloadPlan::RowRef | PayloadPlan::RowWkb | PayloadPlan::FeatureJson { .. } => Ok(()),
        PayloadPlan::None => Err(GeoError::UnsupportedArtifact(
            "search_match_headers needs a feature-ref-bearing payload (RowRef, RowWkb, or FeatureJson); \
             this artifact stores no payload"
                .to_string(),
        )),
    }
}

/// Decode one payload prefix into a header, flagging too-short payloads.
fn collect_header(p: PayloadPrefix<'_>, out: &mut Vec<GeoMatchHeader>, short: &mut bool) {
    match GeoMatchHeader::from_prefix(p) {
        Some(header) => out.push(header),
        None => *short = true,
    }
}

#[cfg(feature = "async")]
fn collect_header_page(
    p: PayloadPrefix<'_>,
    out: &mut HeaderPageCollector<GeoMatchHeader>,
    short: &mut bool,
) {
    match GeoMatchHeader::from_prefix(p) {
        Some(header) => out.push(header),
        None => *short = true,
    }
}

fn ensure_complete_header_payload(short_payload: bool) -> Result<(), GeoError> {
    if short_payload {
        return Err(GeoError::PayloadDecode(
            "a matched payload is shorter than a feature-ref record".to_string(),
        ));
    }
    Ok(())
}

fn finish_headers(
    headers: Vec<GeoMatchHeader>,
    short_payload: bool,
) -> Result<Vec<GeoMatchHeader>, GeoError> {
    ensure_complete_header_payload(short_payload)?;
    Ok(headers)
}

/// Build full matches for `headers` from blobs fetched by rank, preserving the
/// input header order.
fn assemble_matches(
    plan: &PayloadPlan,
    headers: &[GeoMatchHeader],
    by_rank: &std::collections::HashMap<usize, Vec<u8>>,
) -> Result<Vec<GeoMatch>, GeoError> {
    headers
        .iter()
        .map(|header| {
            let payload = payload_at_rank(by_rank, header.leaf_rank, header.payload_len)?;
            let (mut feature, payload) = decode_payload(plan, payload)?;
            ensure_header_feature_matches(&header.feature, &feature)?;
            feature.part = header.feature.part;
            Ok(GeoMatch {
                entry_id: header.entry_id,
                feature,
                payload,
            })
        })
        .collect()
}

fn payload_at_rank(
    by_rank: &std::collections::HashMap<usize, Vec<u8>>,
    leaf_rank: usize,
    expected_len: usize,
) -> Result<&[u8], GeoError> {
    let payload = by_rank.get(&leaf_rank).ok_or_else(|| {
        GeoError::PayloadDecode("missing payload for a header's leaf rank".to_string())
    })?;
    if payload.len() != expected_len {
        return Err(GeoError::PayloadDecode(format!(
            "payload length changed after header read: expected {expected_len}, got {}",
            payload.len()
        )));
    }
    Ok(payload)
}

fn ensure_header_feature_matches(
    header: &FeatureRef,
    payload: &FeatureRef,
) -> Result<(), GeoError> {
    // Headers cannot carry feature_id; part=None may also be the dedupe marker.
    let same_fixed_identity = header.row_number == payload.row_number
        && header.row_group == payload.row_group
        && header.row_in_group == payload.row_in_group;
    let same_part = header.part.is_none() || header.part == payload.part;
    if !same_fixed_identity || !same_part {
        return Err(GeoError::PayloadDecode(
            "payload feature_ref disagrees with its match header".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "async")]
impl<R: AsyncRangeReader> GeoArtifactIndex3D<R> {
    /// Search the underlying core index over async range I/O and return compact
    /// index entry ids.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box3D, GeoArtifactIndex, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D3(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 3D artifact");
    /// # };
    /// let entry_ids = index
    ///     .search_entry_ids_async(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))
    ///     .await?;
    /// println!("{} matching index entries", entry_ids.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_entry_ids_async<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<usize>, GeoError> {
        match query.into() {
            GeoQuery3D::Box3D(bbox) => match &self.index {
                GeoStreamIndex3D::F64(index) => Ok(index.search_async(bbox).await?),
                GeoStreamIndex3D::F32(index) => Ok(index.search_async(bbox).await?),
            },
            GeoQuery3D::Frustum3D(frustum) => match &self.index {
                GeoStreamIndex3D::F64(index) => Ok(index.search_region_async(&frustum).await?),
                GeoStreamIndex3D::F32(index) => Ok(index.search_region_async(&frustum).await?),
            },
        }
    }

    /// Search over async range I/O and return source feature references.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box3D, GeoArtifactIndex, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D3(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 3D artifact");
    /// # };
    /// let features = index
    ///     .search_feature_refs_async(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))
    ///     .await?;
    /// println!("{} matching feature refs", features.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_feature_refs_async<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_matches_async(query)
            .await?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Search over async range I/O and return one deduplicated [`FeatureRef`]
    /// per matched source feature.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box3D, GeoArtifactIndex, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D3(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 3D artifact");
    /// # };
    /// for feature in index
    ///     .search_features_async(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))
    ///     .await?
    /// {
    ///     assert!(feature.part.is_none());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_features_async<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_feature_matches_async(query)
            .await?
            .into_iter()
            .map(|m| m.feature)
            .collect())
    }

    /// Search over async range I/O and return one [`GeoMatch`] per matched
    /// source feature.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box3D, GeoArtifactIndex, GeoPayload, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D3(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 3D artifact");
    /// # };
    /// for m in index
    ///     .search_feature_matches_async(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))
    ///     .await?
    /// {
    ///     assert!(m.feature.part.is_none());
    ///     if let GeoPayload::RowWkb(wkb) = &m.payload {
    ///         println!("row {}: {} WKB bytes", m.feature.row_number, wkb.len());
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_feature_matches_async<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatch>, GeoError> {
        let mut matches = self.search_matches_async(query).await?;
        GeoMatch::dedupe_by_feature(&mut matches);
        Ok(matches)
    }

    /// Search over async range I/O and return decoded geo matches.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use packed_spatial_index_geo::{
    /// #     AsyncRangeReader, Box3D, GeoArtifactIndex, GeoPayload, open_geo_index_async,
    /// # };
    /// # async fn query<R: AsyncRangeReader>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    /// # let GeoArtifactIndex::D3(index) = open_geo_index_async(reader).await? else {
    /// #     panic!("expected a 3D artifact");
    /// # };
    /// for m in index
    ///     .search_matches_async(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))
    ///     .await?
    /// {
    ///     match &m.payload {
    ///         GeoPayload::RowWkb(wkb) => println!("{} WKB bytes", wkb.len()),
    ///         GeoPayload::FeatureJson(feature) => println!("{feature}"),
    ///         GeoPayload::RowRef => println!("row {}", m.feature.row_number),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_matches_async<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<GeoMatch>, GeoError> {
        match query.into() {
            GeoQuery3D::Box3D(bbox) => {
                let raw = match &self.index {
                    GeoStreamIndex3D::F64(index) => index.search_payloads_async(bbox).await?,
                    GeoStreamIndex3D::F32(index) => index.search_payloads_async(bbox).await?,
                };
                decode_matches(&self.manifest.payload_plan, raw)
            }
            GeoQuery3D::Frustum3D(frustum) => {
                let raw = match &self.index {
                    GeoStreamIndex3D::F64(index) => {
                        index.search_payloads_region_async(&frustum).await?
                    }
                    GeoStreamIndex3D::F32(index) => {
                        index.search_payloads_region_async(&frustum).await?
                    }
                };
                decode_matches(&self.manifest.payload_plan, raw)
            }
        }
    }
}

/// One query match from a converted geospatial artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoMatch {
    /// Index entry id in the artifact (the core crate calls these compact
    /// item ids). Stable for one artifact build, not across rebuilds.
    pub entry_id: usize,
    /// Source feature reference stored in the artifact payload.
    pub feature: FeatureRef,
    /// Decoded payload for the match.
    pub payload: GeoPayload,
}

impl GeoMatch {
    /// Sort by feature identity, then `part`, then `entry_id` — the canonical
    /// deterministic entry-level order.
    pub fn sort_by_entry(matches: &mut [GeoMatch]) {
        matches.sort_by(|a, b| {
            a.feature
                .cmp_entry(&b.feature)
                .then_with(|| a.entry_id.cmp(&b.entry_id))
        });
    }

    /// Sort, then collapse split index entries to one match per source
    /// feature.
    ///
    /// The lowest-part entry survives as the representative (its `entry_id`
    /// and payload are kept) and its `feature.part` is set to `None`, since a
    /// part number is meaningless once split entries collapse. Standalone on
    /// purpose: run it after any per-entry filtering (for example
    /// [`GeoArtifactIndex2D::filter_matches`]) — a split part can satisfy a
    /// geometry predicate while another part of the same feature does not.
    pub fn dedupe_by_feature(matches: &mut Vec<GeoMatch>) {
        Self::sort_by_entry(matches);
        matches.dedup_by(|b, a| a.feature.same_feature(&b.feature));
        for m in matches {
            m.feature.part = None;
        }
    }
}

/// Lightweight header for one matched index entry: identity and payload size
/// without the payload body.
///
/// Produced by 2D or 3D `search_match_headers` methods (and the async 2D
/// counterpart); sort, dedupe, and page headers cheaply, then feed the page to
/// the corresponding `fetch_matches` method to materialize full [`GeoMatch`]
/// values for just those entries.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoMatchHeader {
    /// Index entry id, as in [`GeoMatch::entry_id`].
    pub entry_id: usize,
    /// Source feature reference decoded from the payload record prefix.
    pub feature: FeatureRef,
    /// Full payload byte length. For `RowWkb` the WKB geometry occupies
    /// `payload_len - FEATURE_REF_RECORD_LEN` bytes.
    pub payload_len: usize,
    /// Position in the leaf-ordered payload section; build-local, used by
    /// `fetch_matches` to read this entry's payload body.
    leaf_rank: usize,
}

impl GeoMatchHeader {
    /// Sort by feature identity, then `part`, then `entry_id` — the same
    /// canonical order as [`GeoMatch::sort_by_entry`].
    pub fn sort_by_entry(headers: &mut [GeoMatchHeader]) {
        headers.sort_by(Self::entry_order);
    }

    /// Sort, then collapse split index entries to one header per source
    /// feature — the same collapse as [`GeoMatch::dedupe_by_feature`]: the
    /// lowest-part representative survives and its `feature.part` becomes
    /// `None`.
    pub fn dedupe_by_feature(headers: &mut Vec<GeoMatchHeader>) {
        Self::sort_by_entry(headers);
        headers.dedup_by(|b, a| a.feature.same_feature(&b.feature));
        for header in headers {
            header.feature.part = None;
        }
    }

    /// Length of the payload body after the fixed feature-ref record.
    ///
    /// The whole payload is `payload_len` bytes; the leading
    /// [`FEATURE_REF_RECORD_LEN`] hold the [`FeatureRef`], so a `RowWkb`
    /// payload's WKB geometry occupies `payload_len - FEATURE_REF_RECORD_LEN`
    /// bytes. Returns `None` when the stored payload is shorter than a
    /// feature-ref record (a corrupt or truncated artifact).
    pub fn body_byte_len(&self) -> Option<usize> {
        self.payload_len.checked_sub(FEATURE_REF_RECORD_LEN)
    }

    fn entry_order(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.feature
            .cmp_entry(&b.feature)
            .then_with(|| a.entry_id.cmp(&b.entry_id))
    }

    fn from_prefix(prefix: PayloadPrefix<'_>) -> Option<Self> {
        decode_feature_ref_payload(prefix.prefix).map(|feature| Self {
            entry_id: prefix.id,
            feature,
            payload_len: prefix.payload_len,
            leaf_rank: prefix.leaf_rank,
        })
    }

    /// Rebuild the full match for a `RowRef` header — the feature ref is the
    /// entire payload, so no I/O is needed.
    fn to_row_ref_match(&self) -> GeoMatch {
        GeoMatch {
            entry_id: self.entry_id,
            feature: self.feature.clone(),
            payload: GeoPayload::RowRef,
        }
    }
}

/// One deterministic entry-level page from an async match-header search.
#[cfg(feature = "async")]
#[derive(Debug, Clone, PartialEq)]
pub struct GeoMatchHeaderPage {
    /// Total number of matching entries before pagination.
    pub number_matched: usize,
    /// Requested page in feature / part / entry-id order.
    pub headers: Vec<GeoMatchHeader>,
}

/// Lightweight payload location for one matched index entry.
///
/// Unlike [`GeoMatchHeader`], this does not decode a [`FeatureRef`]. It is useful
/// for artifacts where the manifest guarantees entries do not duplicate rows:
/// callers can count, sort, and page by `entry_id`, then fetch full payloads for
/// the page by rank.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoPayloadHeader {
    /// Index entry id, as in [`GeoMatch::entry_id`].
    pub entry_id: usize,
    /// Full payload byte length.
    pub payload_len: usize,
    /// Position in the leaf-ordered payload section.
    leaf_rank: usize,
}

impl GeoPayloadHeader {
    /// Sort by index entry id for deterministic entry-level pagination.
    pub fn sort_by_entry(headers: &mut [GeoPayloadHeader]) {
        headers.sort_by(Self::entry_order);
    }

    /// Length after subtracting one fixed FeatureRef prefix.
    ///
    /// Use this only for payload layouts that the artifact metadata identifies
    /// as feature-ref-prefixed. Legacy raw-JSON `FeatureJson` payloads do not
    /// have that prefix; for them, `payload_len` is already the JSON length.
    pub fn body_byte_len(&self) -> Option<usize> {
        self.payload_len.checked_sub(FEATURE_REF_RECORD_LEN)
    }

    #[cfg(feature = "async")]
    fn from_prefix(prefix: PayloadPrefix<'_>) -> Self {
        Self {
            entry_id: prefix.id,
            payload_len: prefix.payload_len,
            leaf_rank: prefix.leaf_rank,
        }
    }

    fn entry_order(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.entry_id
            .cmp(&b.entry_id)
            .then_with(|| a.leaf_rank.cmp(&b.leaf_rank))
            .then_with(|| a.payload_len.cmp(&b.payload_len))
    }
}

/// One deterministic page from an async payload-header search.
#[cfg(feature = "async")]
#[derive(Debug, Clone, PartialEq)]
pub struct GeoPayloadHeaderPage {
    /// Total number of matching entries before pagination.
    pub number_matched: usize,
    /// Requested page, ordered by entry id.
    pub headers: Vec<GeoPayloadHeader>,
}

#[cfg(feature = "async")]
trait PageHeader {
    fn page_order(a: &Self, b: &Self) -> std::cmp::Ordering;
}

#[cfg(feature = "async")]
impl PageHeader for GeoMatchHeader {
    fn page_order(a: &Self, b: &Self) -> std::cmp::Ordering {
        Self::entry_order(a, b)
    }
}

#[cfg(feature = "async")]
impl PageHeader for GeoPayloadHeader {
    fn page_order(a: &Self, b: &Self) -> std::cmp::Ordering {
        Self::entry_order(a, b)
    }
}

#[cfg(feature = "async")]
struct HeaderPageCollector<H: PageHeader> {
    number_matched: usize,
    offset: usize,
    keep: usize,
    headers: std::collections::BinaryHeap<HeaderByOrder<H>>,
}

#[cfg(feature = "async")]
impl<H: PageHeader> HeaderPageCollector<H> {
    fn new(offset: usize, limit: usize) -> Self {
        Self {
            number_matched: 0,
            offset,
            keep: if limit == 0 {
                0
            } else {
                offset.saturating_add(limit)
            },
            headers: std::collections::BinaryHeap::new(),
        }
    }

    fn push(&mut self, header: H) {
        self.number_matched += 1;
        if self.keep == 0 {
            return;
        }
        if self.headers.len() < self.keep {
            self.headers.push(HeaderByOrder(header));
            return;
        }
        if self
            .headers
            .peek()
            .is_some_and(|last| H::page_order(&header, &last.0).is_lt())
        {
            self.headers.pop();
            self.headers.push(HeaderByOrder(header));
        }
    }

    fn finish(self) -> (usize, Vec<H>) {
        let mut headers: Vec<_> = self.headers.into_iter().map(|item| item.0).collect();
        headers.sort_by(H::page_order);
        let headers = headers.into_iter().skip(self.offset).collect();
        (self.number_matched, headers)
    }
}

#[cfg(feature = "async")]
fn finish_match_header_page(
    page: HeaderPageCollector<GeoMatchHeader>,
    short_payload: bool,
) -> Result<GeoMatchHeaderPage, GeoError> {
    ensure_complete_header_payload(short_payload)?;
    let (number_matched, headers) = page.finish();
    Ok(GeoMatchHeaderPage {
        number_matched,
        headers,
    })
}

#[cfg(feature = "async")]
fn finish_payload_header_page(page: HeaderPageCollector<GeoPayloadHeader>) -> GeoPayloadHeaderPage {
    let (number_matched, headers) = page.finish();
    GeoPayloadHeaderPage {
        number_matched,
        headers,
    }
}

#[cfg(feature = "async")]
struct HeaderByOrder<H: PageHeader>(H);

#[cfg(feature = "async")]
impl<H: PageHeader> PartialEq for HeaderByOrder<H> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

#[cfg(feature = "async")]
impl<H: PageHeader> Eq for HeaderByOrder<H> {}

#[cfg(feature = "async")]
impl<H: PageHeader> PartialOrd for HeaderByOrder<H> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "async")]
impl<H: PageHeader> Ord for HeaderByOrder<H> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        H::page_order(&self.0, &other.0)
    }
}

/// Decoded artifact payload.
#[derive(Debug, Clone, PartialEq)]
pub enum GeoPayload {
    /// A `RowRef` payload; the feature ref is available on [`GeoMatch::feature`].
    RowRef,
    /// A `RowWkb` payload containing WKB geometry bytes.
    RowWkb(Vec<u8>),
    /// A `FeatureJson` payload decoded as a GeoJSON Feature value.
    FeatureJson(serde_json::Value),
}

enum GeoStreamIndex2D<R> {
    F64(StreamIndex2D<R>),
    F32(StreamIndex2DF32<R>),
}

enum GeoStreamIndex3D<R> {
    F64(StreamIndex3D<R>),
    F32(StreamIndex3DF32<R>),
}

const MAX_CONTAINER_CHUNKS_WITHOUT_LEN: usize = 1024;
const MAX_GEO_MANIFEST_BYTES_WITHOUT_LEN: usize = 1024 * 1024;

fn checked_directory_span(
    chunk_count: usize,
    total_len: Option<u64>,
) -> Result<(usize, usize), GeoError> {
    if total_len.is_none() && chunk_count > MAX_CONTAINER_CHUNKS_WITHOUT_LEN {
        return Err(GeoError::Container(
            "too many chunks without a known length".to_string(),
        ));
    }
    let dir_len = chunk_count
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let dir_end = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    if let Some(total_len) = total_len
        && total_len < dir_end as u64
    {
        return Err(GeoError::Container("truncated directory".to_string()));
    }
    Ok((dir_len, dir_end))
}

fn check_manifest_len(len: usize, total_len: Option<u64>) -> Result<(), GeoError> {
    if total_len.is_none() && len > MAX_GEO_MANIFEST_BYTES_WITHOUT_LEN {
        return Err(GeoError::Container(
            "geoM manifest is too large".to_string(),
        ));
    }
    Ok(())
}

fn read_manifest_from_reader<R: RangeReader>(reader: &R) -> Result<GeoArtifactManifest, GeoError> {
    let mut head = [0u8; SUPERBLOCK_LEN];
    reader
        .read_exact_at(0, &mut head)
        .map_err(StreamError::Io)?;
    if &head[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(GeoError::Container("bad magic".to_string()));
    }
    if read_u64(&head, 8)? != FORMAT_VERSION {
        return Err(GeoError::Container("unsupported version".to_string()));
    }

    let total_len = reader.len();
    let chunk_count = read_u32(&head, 16)? as usize;
    let (dir_len, dir_end) = checked_directory_span(chunk_count, total_len)?;
    let mut dir = vec![0; dir_len];
    reader
        .read_exact_at(SUPERBLOCK_LEN as u64, &mut dir)
        .map_err(StreamError::Io)?;

    for i in 0..chunk_count {
        let base = i * CHUNK_ENTRY_LEN;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&dir[base..base + 4]);
        let offset = usize::try_from(read_u64(&dir, base + 8)?)
            .map_err(|_| GeoError::Container("offset overflow".to_string()))?;
        let len = usize::try_from(read_u64(&dir, base + 16)?)
            .map_err(|_| GeoError::Container("length overflow".to_string()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| GeoError::Container("chunk overflow".to_string()))?;
        if offset < dir_end {
            return Err(GeoError::Container("chunk range outside file".to_string()));
        }
        if let Some(total_len) = total_len
            && end as u64 > total_len
        {
            return Err(GeoError::Container("chunk range outside file".to_string()));
        }
        if tag == TAG_GEO_MANIFEST {
            check_manifest_len(len, total_len)?;
            let mut content = vec![0; len];
            reader
                .read_exact_at(offset as u64, &mut content)
                .map_err(StreamError::Io)?;
            return read_geo_manifest_content(&content);
        }
    }

    Err(GeoError::MissingGeoManifest)
}

#[cfg(feature = "async")]
async fn read_manifest_from_reader_async<R: AsyncRangeReader>(
    reader: &R,
) -> Result<GeoArtifactManifest, GeoError> {
    let mut head = [0u8; SUPERBLOCK_LEN];
    reader
        .read_exact_at(0, &mut head)
        .await
        .map_err(StreamError::Io)?;
    if &head[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(GeoError::Container("bad magic".to_string()));
    }
    if read_u64(&head, 8)? != FORMAT_VERSION {
        return Err(GeoError::Container("unsupported version".to_string()));
    }

    let total_len = reader.len();
    let chunk_count = read_u32(&head, 16)? as usize;
    let (dir_len, dir_end) = checked_directory_span(chunk_count, total_len)?;
    let mut dir = vec![0; dir_len];
    reader
        .read_exact_at(SUPERBLOCK_LEN as u64, &mut dir)
        .await
        .map_err(StreamError::Io)?;

    for i in 0..chunk_count {
        let base = i * CHUNK_ENTRY_LEN;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&dir[base..base + 4]);
        let offset = usize::try_from(read_u64(&dir, base + 8)?)
            .map_err(|_| GeoError::Container("offset overflow".to_string()))?;
        let len = usize::try_from(read_u64(&dir, base + 16)?)
            .map_err(|_| GeoError::Container("length overflow".to_string()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| GeoError::Container("chunk overflow".to_string()))?;
        if offset < dir_end {
            return Err(GeoError::Container("chunk range outside file".to_string()));
        }
        if let Some(total_len) = total_len
            && end as u64 > total_len
        {
            return Err(GeoError::Container("chunk range outside file".to_string()));
        }
        if tag == TAG_GEO_MANIFEST {
            check_manifest_len(len, total_len)?;
            let mut content = vec![0; len];
            reader
                .read_exact_at(offset as u64, &mut content)
                .await
                .map_err(StreamError::Io)?;
            return read_geo_manifest_content(&content);
        }
    }

    Err(GeoError::MissingGeoManifest)
}

fn decode_matches(
    plan: &PayloadPlan,
    raw: Vec<(usize, Vec<u8>)>,
) -> Result<Vec<GeoMatch>, GeoError> {
    raw.into_iter()
        .map(|(entry_id, payload)| {
            let (feature, payload) = decode_payload(plan, &payload)?;
            Ok(GeoMatch {
                entry_id,
                feature,
                payload,
            })
        })
        .collect()
}

fn decode_payload(
    plan: &PayloadPlan,
    payload: &[u8],
) -> Result<(FeatureRef, GeoPayload), GeoError> {
    match plan {
        PayloadPlan::RowRef => {
            let feature = decode_feature_ref_payload(payload).ok_or_else(|| {
                GeoError::PayloadDecode("row-ref payload is truncated".to_string())
            })?;
            Ok((feature, GeoPayload::RowRef))
        }
        PayloadPlan::RowWkb => {
            let (feature, wkb) = decode_feature_wkb_payload(payload).ok_or_else(|| {
                GeoError::PayloadDecode("row-wkb payload is truncated".to_string())
            })?;
            Ok((feature, GeoPayload::RowWkb(wkb.to_vec())))
        }
        PayloadPlan::FeatureJson { .. } => {
            let prefix_feature = decode_feature_ref_payload(payload);
            let json: serde_json::Value = serde_json::from_slice(feature_json_body(payload))
                .map_err(|e| GeoError::PayloadDecode(e.to_string()))?;
            let json_feature = json
                .get("feature_ref")
                .cloned()
                .map(|value| {
                    serde_json::from_value(value)
                        .map_err(|e| GeoError::PayloadDecode(e.to_string()))
                })
                .transpose()?;
            let feature = match (prefix_feature, json_feature) {
                (Some(prefix), Some(json_feature)) => {
                    if !feature_ref_record_fields_match(&prefix, &json_feature) {
                        return Err(GeoError::PayloadDecode(
                            "feature_json prefix disagrees with its JSON feature_ref".to_string(),
                        ));
                    }
                    json_feature
                }
                (Some(prefix), None) => prefix,
                (None, Some(json_feature)) => json_feature,
                (None, None) => {
                    return Err(GeoError::PayloadDecode(
                        "feature_json payload has no feature_ref".to_string(),
                    ));
                }
            };
            Ok((feature, GeoPayload::FeatureJson(json)))
        }
        PayloadPlan::None => Err(GeoError::UnsupportedArtifact(
            "artifact payload does not contain feature refs".to_string(),
        )),
    }
}

fn feature_ref_record_fields_match(a: &FeatureRef, b: &FeatureRef) -> bool {
    a.row_number == b.row_number
        && a.row_group == b.row_group
        && a.row_in_group == b.row_in_group
        && a.part == b.part
}
