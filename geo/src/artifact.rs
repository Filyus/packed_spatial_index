use packed_spatial_index::{
    Box2D, Box3D, RangeReader, StreamError, StreamIndex2D, StreamIndex2DF32, StreamIndex3D,
    StreamIndex3DF32, StreamLimits,
};

use crate::{
    FeatureRef, GeoArtifactManifest, GeoError, PayloadPlan, StoragePrecision,
    decode_feature_ref_payload, decode_feature_wkb_payload,
    manifest::{
        CHUNK_ENTRY_LEN, FORMAT_MAGIC, FORMAT_VERSION, SUPERBLOCK_LEN, TAG_GEO_MANIFEST,
        read_geo_manifest_content, read_u32, read_u64,
    },
};

pub fn open_geo_index<R: RangeReader>(reader: R) -> Result<GeoArtifactIndex<R>, GeoError> {
    open_geo_index_with_limits(reader, StreamLimits::default())
}

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

pub enum GeoArtifactIndex<R> {
    D2(GeoArtifactIndex2D<R>),
    D3(GeoArtifactIndex3D<R>),
}

impl<R> GeoArtifactIndex<R> {
    pub fn manifest(&self) -> &GeoArtifactManifest {
        match self {
            GeoArtifactIndex::D2(index) => index.manifest(),
            GeoArtifactIndex::D3(index) => index.manifest(),
        }
    }
}

pub struct GeoArtifactIndex2D<R> {
    index: GeoStreamIndex2D<R>,
    manifest: GeoArtifactManifest,
}

impl<R> GeoArtifactIndex2D<R> {
    pub fn manifest(&self) -> &GeoArtifactManifest {
        &self.manifest
    }
}

impl<R: RangeReader> GeoArtifactIndex2D<R> {
    pub fn search_items(&self, query: Box2D) -> Result<Vec<usize>, GeoError> {
        match &self.index {
            GeoStreamIndex2D::F64(index) => Ok(index.search(query)?),
            GeoStreamIndex2D::F32(index) => Ok(index.search(query)?),
        }
    }

    pub fn search_features(&self, query: Box2D) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_hits(query)?
            .into_iter()
            .map(|hit| hit.feature)
            .collect())
    }

    pub fn search_hits(&self, query: Box2D) -> Result<Vec<GeoHit>, GeoError> {
        let hits = match &self.index {
            GeoStreamIndex2D::F64(index) => index.search_payloads(query)?,
            GeoStreamIndex2D::F32(index) => index.search_payloads(query)?,
        };
        decode_hits(&self.manifest.payload_plan, hits)
    }
}

pub struct GeoArtifactIndex3D<R> {
    index: GeoStreamIndex3D<R>,
    manifest: GeoArtifactManifest,
}

impl<R> GeoArtifactIndex3D<R> {
    pub fn manifest(&self) -> &GeoArtifactManifest {
        &self.manifest
    }
}

impl<R: RangeReader> GeoArtifactIndex3D<R> {
    pub fn search_items(&self, query: Box3D) -> Result<Vec<usize>, GeoError> {
        match &self.index {
            GeoStreamIndex3D::F64(index) => Ok(index.search(query)?),
            GeoStreamIndex3D::F32(index) => Ok(index.search(query)?),
        }
    }

    pub fn search_features(&self, query: Box3D) -> Result<Vec<FeatureRef>, GeoError> {
        Ok(self
            .search_hits(query)?
            .into_iter()
            .map(|hit| hit.feature)
            .collect())
    }

    pub fn search_hits(&self, query: Box3D) -> Result<Vec<GeoHit>, GeoError> {
        let hits = match &self.index {
            GeoStreamIndex3D::F64(index) => index.search_payloads(query)?,
            GeoStreamIndex3D::F32(index) => index.search_payloads(query)?,
        };
        decode_hits(&self.manifest.payload_plan, hits)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeoHit {
    pub item: usize,
    pub feature: FeatureRef,
    pub payload: GeoPayload,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeoPayload {
    RowRef,
    RowWkb(Vec<u8>),
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
