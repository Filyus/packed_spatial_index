use geo::Intersects;
use geo_types::{Coord, MultiPolygon, Rect};
use geozero::ToGeo;
use geozero::geojson::GeoJson;
use packed_spatial_index::{
    Box2D, Overlaps2D, RangeReader, StreamError, StreamIndex2D, StreamIndex2DF32, StreamIndex3D,
    StreamIndex3DF32, StreamLimits,
};

use crate::{
    FeatureRef, GeoArtifactManifest, GeoError, GeoQuery2D, GeoQuery3D, NonPlanarExactPolicy,
    PayloadPlan, SpatialPredicate, StoragePrecision, decode_feature_ref_payload,
    decode_feature_wkb_payload,
    filter::{decode_geo_geometry, exact_predicate_matches, prepare_filter_query},
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
/// let hits = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
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
    if manifest.schema_version != 2 {
        return Err(GeoError::UnsupportedArtifact(format!(
            "unsupported geoM schema version {}",
            manifest.schema_version
        )));
    }
    match (manifest.dims.index_dims(), manifest.storage_precision) {
        (Some(2), StoragePrecision::F64) => Ok(GeoArtifactIndex::D2(GeoArtifactIndex2D {
            index: GeoStreamIndex2D::F64(StreamIndex2D::open_with_limits(reader, limits)?),
            manifest,
        })),
        (Some(2), StoragePrecision::F32) => Ok(GeoArtifactIndex::D2(GeoArtifactIndex2D {
            index: GeoStreamIndex2D::F32(StreamIndex2DF32::open_with_limits(reader, limits)?),
            manifest,
        })),
        (Some(3), StoragePrecision::F64) => Ok(GeoArtifactIndex::D3(GeoArtifactIndex3D {
            index: GeoStreamIndex3D::F64(StreamIndex3D::open_with_limits(reader, limits)?),
            manifest,
        })),
        (Some(3), StoragePrecision::F32) => Ok(GeoArtifactIndex::D3(GeoArtifactIndex3D {
            index: GeoStreamIndex3D::F32(StreamIndex3DF32::open_with_limits(reader, limits)?),
            manifest,
        })),
        (None, _) => Err(GeoError::UnsupportedArtifact(format!(
            "artifact has unknown coordinate dimensions {:?}",
            manifest.dims
        ))),
        (Some(other), _) => Err(GeoError::UnsupportedArtifact(format!(
            "artifact has unsupported coordinate dimension count {other}"
        ))),
    }
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
}

/// A 2D geospatial artifact index.
///
/// Use [`GeoArtifactIndex2D::search_hits`] when payloads are present and you
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
}

impl<R: RangeReader> GeoArtifactIndex2D<R> {
    /// Search the underlying core index and return compact item ids.
    ///
    /// This does not decode geo payloads and therefore also works for
    /// [`PayloadPlan::None`] artifacts.
    pub fn search_items<Q: Into<GeoQuery2D>>(&self, query: Q) -> Result<Vec<usize>, GeoError> {
        let mut items = Vec::new();
        for bbox in query.into().candidate_boxes_2d()? {
            let hits = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search(bbox)?,
                GeoStreamIndex2D::F32(index) => index.search(bbox)?,
            };
            for item in hits {
                if !items.contains(&item) {
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
    pub fn search_features<Q: Into<GeoQuery2D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_hits(query)?
            .into_iter()
            .map(|hit| hit.feature)
            .collect())
    }

    /// Search and return decoded geo hits.
    ///
    /// Each hit includes the compact item id, the source [`FeatureRef`], and
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
    /// for hit in index.search_hits(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    ///     match &hit.payload {
    ///         GeoPayload::RowWkb(wkb) => {
    ///             println!("{}: {} WKB bytes", hit.feature.row_number, wkb.len())
    ///         }
    ///         GeoPayload::FeatureJson(feature) => println!("{}: {feature}", hit.feature.row_number),
    ///         GeoPayload::RowRef => println!("{}: no geometry payload", hit.feature.row_number),
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_hits<Q: Into<GeoQuery2D>>(&self, query: Q) -> Result<Vec<GeoHit>, GeoError> {
        let query = query.into();
        if let GeoQuery2D::Polygon(multi_polygon) = &query {
            let region = PolygonRegion(multi_polygon);
            let hits = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search_payloads_region(&region)?,
                GeoStreamIndex2D::F32(index) => index.search_payloads_region(&region)?,
            };
            return decode_hits(&self.manifest.payload_plan, hits);
        }

        let boxes = query.candidate_boxes_2d()?;
        // Duplicates only arise across multiple candidate boxes; a single box
        // yields each item once. Skip dedup bookkeeping in the common single-box
        // case, and dedup by item id in O(1) (not O(K^2) `iter().any`) otherwise.
        let dedup = boxes.len() > 1;
        let mut decoded = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for bbox in boxes {
            let hits = match &self.index {
                GeoStreamIndex2D::F64(index) => index.search_payloads(bbox)?,
                GeoStreamIndex2D::F32(index) => index.search_payloads(bbox)?,
            };
            for hit in decode_hits(&self.manifest.payload_plan, hits)? {
                if !dedup || seen.insert(hit.item) {
                    decoded.push(hit);
                }
            }
        }
        Ok(decoded)
    }

    /// Exactly filter geo hits by the geometry stored in their payloads — the
    /// post-filter step for the streaming path, with no source re-read.
    ///
    /// Index search narrows by bounding box; this keeps only the hits whose
    /// geometry actually satisfies `query` under `predicate`, removing the bbox
    /// false-positives over holes and concavities. Because it tests the geometry
    /// already fetched by [`GeoArtifactIndex2D::search_hits`], it avoids the
    /// candidate geometry re-read that [`GeoDataset::filter_features`] performs.
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
    /// let hits = index.search_hits(GeoQuery2D::polygon(triangle.clone()))?;
    /// let exact = index.filter_hits(
    ///     hits,
    ///     GeoQuery2D::polygon(triangle),
    ///     SpatialPredicate::Intersects,
    ///     NonPlanarExactPolicy::Reject,
    /// )?;
    /// println!("{} exact hits", exact.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// [`GeoDataset::filter_features`]: crate::GeoDataset::filter_features
    pub fn filter_hits<Q: Into<GeoQuery2D>>(
        &self,
        hits: Vec<GeoHit>,
        query: Q,
        predicate: SpatialPredicate,
        non_planar: NonPlanarExactPolicy,
    ) -> Result<Vec<GeoHit>, GeoError> {
        let prepared = prepare_filter_query(
            &self.manifest.encoding,
            self.manifest.edges,
            &self.manifest.selected_column,
            query.into(),
            non_planar,
        )?;
        let mut kept = Vec::new();
        for hit in hits {
            let geometry = match &hit.payload {
                GeoPayload::RowWkb(wkb) => decode_geo_geometry(wkb)?,
                GeoPayload::FeatureJson(feature) => feature_json_geometry(feature)?,
                GeoPayload::RowRef => {
                    return Err(GeoError::PayloadDecode(
                        "filter_hits needs a geometry payload (RowWkb or FeatureJson); RowRef has none"
                            .to_string(),
                    ));
                }
            };
            let Some(geometry) = geometry else {
                continue;
            };
            if exact_predicate_matches(&geometry, &prepared, predicate)? {
                kept.push(hit);
            }
        }
        Ok(kept)
    }
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
}

impl<R: RangeReader> GeoArtifactIndex3D<R> {
    /// Search the underlying core index and return compact item ids.
    ///
    /// This does not decode geo payloads and therefore also works for
    /// [`PayloadPlan::None`] artifacts.
    pub fn search_items<Q: Into<GeoQuery3D>>(&self, query: Q) -> Result<Vec<usize>, GeoError> {
        let bbox = query.into().candidate_box_3d();
        match &self.index {
            GeoStreamIndex3D::F64(index) => Ok(index.search(bbox)?),
            GeoStreamIndex3D::F32(index) => Ok(index.search(bbox)?),
        }
    }

    /// Search and return source feature references.
    ///
    /// This requires an artifact payload plan that stores feature refs
    /// (`RowRef`, `RowWkb`, or `FeatureJson`).
    pub fn search_features<Q: Into<GeoQuery3D>>(
        &self,
        query: Q,
    ) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_hits(query)?
            .into_iter()
            .map(|hit| hit.feature)
            .collect())
    }

    /// Search and return decoded geo hits.
    ///
    /// Each hit includes the compact item id, the source [`FeatureRef`], and
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
    /// for hit in index.search_hits(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0))? {
    ///     match &hit.payload {
    ///         GeoPayload::RowWkb(wkb) => {
    ///             println!("{}: {} WKB bytes", hit.feature.row_number, wkb.len())
    ///         }
    ///         GeoPayload::FeatureJson(feature) => println!("{}: {feature}", hit.feature.row_number),
    ///         GeoPayload::RowRef => println!("{}: no geometry payload", hit.feature.row_number),
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_hits<Q: Into<GeoQuery3D>>(&self, query: Q) -> Result<Vec<GeoHit>, GeoError> {
        let bbox = query.into().candidate_box_3d();
        let hits = match &self.index {
            GeoStreamIndex3D::F64(index) => index.search_payloads(bbox)?,
            GeoStreamIndex3D::F32(index) => index.search_payloads(bbox)?,
        };
        decode_hits(&self.manifest.payload_plan, hits)
    }
}

/// One query hit from a converted geospatial artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoHit {
    /// Compact item id in the core index.
    pub item: usize,
    /// Source feature reference stored in the artifact payload.
    pub feature: FeatureRef,
    /// Decoded payload for the hit.
    pub payload: GeoPayload,
}

/// Decoded artifact payload.
#[derive(Debug, Clone, PartialEq)]
pub enum GeoPayload {
    /// A `RowRef` payload; the feature ref is available on [`GeoHit::feature`].
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

    let chunk_count = read_u32(&head, 16)? as usize;
    let dir_len = chunk_count
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let dir_end = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
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
        if let Some(total_len) = reader.len()
            && end as u64 > total_len
        {
            return Err(GeoError::Container("chunk range outside file".to_string()));
        }
        if tag == TAG_GEO_MANIFEST {
            let mut content = vec![0; len];
            reader
                .read_exact_at(offset as u64, &mut content)
                .map_err(StreamError::Io)?;
            return read_geo_manifest_content(&content);
        }
    }

    Err(GeoError::MissingGeoManifest)
}

fn decode_hits(plan: &PayloadPlan, hits: Vec<(usize, Vec<u8>)>) -> Result<Vec<GeoHit>, GeoError> {
    hits.into_iter()
        .map(|(item, payload)| {
            let (feature, payload) = decode_payload(plan, &payload)?;
            Ok(GeoHit {
                item,
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
            let json: serde_json::Value = serde_json::from_slice(payload)
                .map_err(|e| GeoError::PayloadDecode(e.to_string()))?;
            let feature = json
                .get("feature_ref")
                .cloned()
                .ok_or_else(|| {
                    GeoError::PayloadDecode("feature_json payload has no feature_ref".to_string())
                })
                .and_then(|value| {
                    serde_json::from_value(value)
                        .map_err(|e| GeoError::PayloadDecode(e.to_string()))
                })?;
            Ok((feature, GeoPayload::FeatureJson(json)))
        }
        PayloadPlan::None => Err(GeoError::UnsupportedArtifact(
            "artifact payload does not contain feature refs".to_string(),
        )),
    }
}
