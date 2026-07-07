use serde::{Deserialize, Serialize};

use crate::{
    AntimeridianPolicy, CoordinateDims, CrsInfo, EdgeModel, GeoError, GeometryEncoding, NullPolicy,
    PayloadPlan,
};

/// Coordinate storage precision of a converted artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoragePrecision {
    /// Store coordinates as `f64`.
    F64,
    /// Store coordinates as `f32`; queries return a conservative superset.
    F32,
}

/// Geospatial manifest embedded in a converted `PSINDEX` artifact.
///
/// # Example
///
/// ```no_run
/// use packed_spatial_index_geo::read_geo_manifest;
///
/// let bytes = std::fs::read("cities.psindex")?;
/// if let Some(manifest) = read_geo_manifest(&bytes)? {
///     println!(
///         "{}: {} features",
///         manifest.selected_column,
///         manifest.feature_count
///     );
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoArtifactManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Source format label.
    pub source_format: String,
    /// Stable source metadata fingerprint.
    pub source_fingerprint: String,
    /// Selected geometry column name.
    pub selected_column: String,
    /// CRS metadata.
    pub crs: CrsInfo,
    /// Edge model.
    pub edges: EdgeModel,
    /// Geometry encoding.
    pub encoding: GeometryEncoding,
    /// Coordinate dimensions.
    pub dims: CoordinateDims,
    /// Artifact coordinate precision.
    pub storage_precision: StoragePrecision,
    /// Null policy used during conversion.
    pub null_policy: NullPolicy,
    /// Antimeridian policy used during conversion.
    pub antimeridian_policy: AntimeridianPolicy,
    /// Payload plan used during conversion.
    pub payload_plan: PayloadPlan,
    /// Number of unique source features represented.
    pub feature_count: usize,
    /// Number of index entries.
    pub index_entry_count: usize,
    /// Whether one source row may map to multiple entries.
    pub entries_may_duplicate_rows: bool,
}

pub(crate) const FORMAT_MAGIC: &[u8; 8] = b"PSINDEX\0";
pub(crate) const FORMAT_VERSION: u64 = 2;
pub(crate) const SUPERBLOCK_LEN: usize = 32;
pub(crate) const CHUNK_ENTRY_LEN: usize = 24;
#[cfg(feature = "_source")]
const CHUNK_FLAG_CRITICAL: u32 = 1;
pub(crate) const TAG_GEO_MANIFEST: [u8; 4] = *b"geoM";

#[derive(Debug, Clone)]
struct Chunk {
    tag: [u8; 4],
    #[cfg_attr(not(feature = "_source"), allow(dead_code))]
    flags: u32,
    content: Vec<u8>,
}

#[cfg(feature = "_source")]
#[derive(Debug, Clone, Copy)]
struct ChunkRef {
    tag: [u8; 4],
    flags: u32,
    offset: usize,
    len: usize,
}

/// Read the embedded `geoM` manifest from a converted `PSINDEX` byte buffer.
///
/// Returns `Ok(None)` when the container has no `geoM` chunk. Use
/// [`open_geo_index`](crate::open_geo_index) when you want to query the artifact
/// as a geospatial index instead of only reading metadata.
///
/// Unlike [`open_geo_index_with_limits`](crate::open_geo_index_with_limits), this
/// parses the whole buffer in memory with no size caps, so it is meant for
/// caller-trusted data (a file you read yourself, a fixture). For untrusted or
/// externally hosted artifacts open them through
/// [`open_geo_index_with_limits`](crate::open_geo_index_with_limits), which
/// applies [`StreamLimits`](packed_spatial_index::StreamLimits).
pub fn read_geo_manifest(bytes: &[u8]) -> Result<Option<GeoArtifactManifest>, GeoError> {
    let chunks = parse_chunks(bytes)?;
    let Some(chunk) = chunks.iter().find(|chunk| chunk.tag == TAG_GEO_MANIFEST) else {
        return Ok(None);
    };
    read_geo_manifest_content(&chunk.content).map(Some)
}

pub(crate) fn read_geo_manifest_content(content: &[u8]) -> Result<GeoArtifactManifest, GeoError> {
    let value: serde_json::Value =
        serde_json::from_slice(content).map_err(|e| GeoError::Container(e.to_string()))?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            GeoError::UnsupportedArtifact("geoM manifest has no schema_version".to_string())
        })?;
    if schema_version != 2 {
        return Err(GeoError::UnsupportedArtifact(format!(
            "unsupported geoM schema version {schema_version}"
        )));
    }
    serde_json::from_value(value).map_err(|e| GeoError::Container(e.to_string()))
}

#[cfg(feature = "_source")]
pub(crate) fn append_geo_manifest(
    out: &mut Vec<u8>,
    manifest: &GeoArtifactManifest,
) -> Result<(), GeoError> {
    let refs = parse_chunk_refs(out)?;
    let content = serde_json::to_vec(manifest).map_err(|e| GeoError::Container(e.to_string()))?;
    let chunks: Vec<ChunkRef> = refs
        .into_iter()
        .filter(|chunk| chunk.tag != TAG_GEO_MANIFEST)
        .collect();
    let mut lengths: Vec<usize> = chunks.iter().map(|chunk| chunk.len).collect();
    lengths.push(content.len());
    let (offsets, total) = plan_offsets_from_lengths(&lengths)?;
    let moves: Vec<(usize, usize, usize)> = chunks
        .iter()
        .zip(offsets.iter().copied())
        .map(|(chunk, new_offset)| (chunk.offset, new_offset, chunk.len))
        .collect();

    // Adding a new directory entry normally shifts all chunk bodies to the
    // right. Replacing an existing geoM can shift chunks left; keep that rare
    // path simple and conservative.
    if moves
        .iter()
        .any(|(old_offset, new_offset, _)| new_offset < old_offset)
    {
        let bytes = out.clone();
        let mut owned = parse_chunks(&bytes)?;
        owned.retain(|chunk| chunk.tag != TAG_GEO_MANIFEST);
        owned.push(Chunk {
            tag: TAG_GEO_MANIFEST,
            flags: 0,
            content,
        });
        return write_chunks(&owned, out);
    }

    out.resize(total, 0);
    for (old_offset, new_offset, len) in moves.iter().rev().copied() {
        out.copy_within(old_offset..old_offset + len, new_offset);
    }
    let manifest_offset = *offsets
        .last()
        .ok_or_else(|| GeoError::Container("missing manifest offset".to_string()))?;
    out[manifest_offset..manifest_offset + content.len()].copy_from_slice(&content);

    zero_padding(out, &offsets, &lengths, total)?;
    write_superblock_and_directory(out, &chunks, &offsets, content.len())?;
    Ok(())
}

fn parse_chunks(bytes: &[u8]) -> Result<Vec<Chunk>, GeoError> {
    if bytes.len() < SUPERBLOCK_LEN {
        return Err(GeoError::Container("truncated superblock".to_string()));
    }
    if &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(GeoError::Container("bad magic".to_string()));
    }
    if read_u64(bytes, 8)? != FORMAT_VERSION {
        return Err(GeoError::Container("unsupported version".to_string()));
    }
    let chunk_count = read_u32(bytes, 16)? as usize;
    let dir_len = chunk_count
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let dir_end = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    if bytes.len() < dir_end {
        return Err(GeoError::Container("truncated directory".to_string()));
    }
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut max_end = dir_end;
    for i in 0..chunk_count {
        let base = SUPERBLOCK_LEN + i * CHUNK_ENTRY_LEN;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&bytes[base..base + 4]);
        let flags = read_u32(bytes, base + 4)?;
        let offset = usize::try_from(read_u64(bytes, base + 8)?)
            .map_err(|_| GeoError::Container("offset overflow".to_string()))?;
        let len = usize::try_from(read_u64(bytes, base + 16)?)
            .map_err(|_| GeoError::Container("length overflow".to_string()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| GeoError::Container("chunk overflow".to_string()))?;
        if offset < dir_end || end > bytes.len() {
            return Err(GeoError::Container("chunk range outside file".to_string()));
        }
        max_end = max_end.max(end);
        chunks.push(Chunk {
            tag,
            flags,
            content: bytes[offset..end].to_vec(),
        });
    }
    if bytes.len() > align8(max_end)? {
        return Err(GeoError::Container(
            "trailing bytes outside directory".to_string(),
        ));
    }
    Ok(chunks)
}

#[cfg(feature = "_source")]
fn parse_chunk_refs(bytes: &[u8]) -> Result<Vec<ChunkRef>, GeoError> {
    if bytes.len() < SUPERBLOCK_LEN {
        return Err(GeoError::Container("truncated superblock".to_string()));
    }
    if &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(GeoError::Container("bad magic".to_string()));
    }
    if read_u64(bytes, 8)? != FORMAT_VERSION {
        return Err(GeoError::Container("unsupported version".to_string()));
    }
    let chunk_count = read_u32(bytes, 16)? as usize;
    let dir_len = chunk_count
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let dir_end = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    if bytes.len() < dir_end {
        return Err(GeoError::Container("truncated directory".to_string()));
    }
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut max_end = dir_end;
    for i in 0..chunk_count {
        let base = SUPERBLOCK_LEN + i * CHUNK_ENTRY_LEN;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&bytes[base..base + 4]);
        let flags = read_u32(bytes, base + 4)?;
        let offset = usize::try_from(read_u64(bytes, base + 8)?)
            .map_err(|_| GeoError::Container("offset overflow".to_string()))?;
        let len = usize::try_from(read_u64(bytes, base + 16)?)
            .map_err(|_| GeoError::Container("length overflow".to_string()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| GeoError::Container("chunk overflow".to_string()))?;
        if offset < dir_end || end > bytes.len() {
            return Err(GeoError::Container("chunk range outside file".to_string()));
        }
        max_end = max_end.max(end);
        chunks.push(ChunkRef {
            tag,
            flags,
            offset,
            len,
        });
    }
    if bytes.len() > align8(max_end)? {
        return Err(GeoError::Container(
            "trailing bytes outside directory".to_string(),
        ));
    }
    Ok(chunks)
}

#[cfg(feature = "_source")]
fn write_chunks(chunks: &[Chunk], out: &mut Vec<u8>) -> Result<(), GeoError> {
    let offsets = plan_offsets(chunks)?;
    let total = offsets
        .last()
        .zip(chunks.last())
        .map(|(offset, chunk)| offset + chunk.content.len())
        .unwrap_or(SUPERBLOCK_LEN + chunks.len() * CHUNK_ENTRY_LEN);
    let total = align8(total)?;
    out.clear();
    out.resize(total, 0);
    out[..FORMAT_MAGIC.len()].copy_from_slice(FORMAT_MAGIC);
    write_u64(out, 8, FORMAT_VERSION);
    write_u32(out, 16, chunks.len() as u32);
    for (i, chunk) in chunks.iter().enumerate() {
        let base = SUPERBLOCK_LEN + i * CHUNK_ENTRY_LEN;
        out[base..base + 4].copy_from_slice(&chunk.tag);
        write_u32(out, base + 4, chunk.flags & CHUNK_FLAG_CRITICAL);
        write_u64(out, base + 8, offsets[i] as u64);
        write_u64(out, base + 16, chunk.content.len() as u64);
        let start = offsets[i];
        out[start..start + chunk.content.len()].copy_from_slice(&chunk.content);
    }
    Ok(())
}

#[cfg(feature = "_source")]
fn write_superblock_and_directory(
    out: &mut [u8],
    chunks: &[ChunkRef],
    offsets: &[usize],
    manifest_len: usize,
) -> Result<(), GeoError> {
    let chunk_count = chunks
        .len()
        .checked_add(1)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    if offsets.len() != chunk_count {
        return Err(GeoError::Container("directory offset mismatch".to_string()));
    }
    out[..FORMAT_MAGIC.len()].copy_from_slice(FORMAT_MAGIC);
    write_u64(out, 8, FORMAT_VERSION);
    write_u32(out, 16, chunk_count as u32);
    for (i, chunk) in chunks.iter().enumerate() {
        write_chunk_entry(out, i, chunk.tag, chunk.flags, offsets[i], chunk.len);
    }
    write_chunk_entry(
        out,
        chunks.len(),
        TAG_GEO_MANIFEST,
        0,
        offsets[chunks.len()],
        manifest_len,
    );
    Ok(())
}

#[cfg(feature = "_source")]
fn write_chunk_entry(
    out: &mut [u8],
    index: usize,
    tag: [u8; 4],
    flags: u32,
    offset: usize,
    len: usize,
) {
    let base = SUPERBLOCK_LEN + index * CHUNK_ENTRY_LEN;
    out[base..base + 4].copy_from_slice(&tag);
    write_u32(out, base + 4, flags & CHUNK_FLAG_CRITICAL);
    write_u64(out, base + 8, offset as u64);
    write_u64(out, base + 16, len as u64);
}

#[cfg(feature = "_source")]
fn plan_offsets(chunks: &[Chunk]) -> Result<Vec<usize>, GeoError> {
    let dir_len = chunks
        .len()
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let mut cur = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let mut offsets = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        cur = align8(cur)?;
        offsets.push(cur);
        cur = cur
            .checked_add(chunk.content.len())
            .ok_or_else(|| GeoError::Container("container length overflow".to_string()))?;
    }
    Ok(offsets)
}

#[cfg(feature = "_source")]
fn plan_offsets_from_lengths(lengths: &[usize]) -> Result<(Vec<usize>, usize), GeoError> {
    let dir_len = lengths
        .len()
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let mut cur = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let mut offsets = Vec::with_capacity(lengths.len());
    for len in lengths {
        cur = align8(cur)?;
        offsets.push(cur);
        cur = cur
            .checked_add(*len)
            .ok_or_else(|| GeoError::Container("container length overflow".to_string()))?;
    }
    Ok((offsets, align8(cur)?))
}

#[cfg(feature = "_source")]
fn zero_padding(
    out: &mut [u8],
    offsets: &[usize],
    lengths: &[usize],
    total: usize,
) -> Result<(), GeoError> {
    let dir_end = SUPERBLOCK_LEN
        .checked_add(
            lengths
                .len()
                .checked_mul(CHUNK_ENTRY_LEN)
                .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?,
        )
        .ok_or_else(|| GeoError::Container("directory overflow".to_string()))?;
    let mut pos = dir_end;
    for (&offset, &len) in offsets.iter().zip(lengths) {
        if offset > pos {
            out[pos..offset].fill(0);
        }
        pos = offset
            .checked_add(len)
            .ok_or_else(|| GeoError::Container("container length overflow".to_string()))?;
    }
    if total > pos {
        out[pos..total].fill(0);
    }
    Ok(())
}

fn align8(value: usize) -> Result<usize, GeoError> {
    value
        .checked_add(7)
        .map(|v| v & !7)
        .ok_or_else(|| GeoError::Container("alignment overflow".to_string()))
}

pub(crate) fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, GeoError> {
    let end = offset + 4;
    let Some(slice) = bytes.get(offset..end) else {
        return Err(GeoError::Container("truncated u32".to_string()));
    };
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

pub(crate) fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, GeoError> {
    let end = offset + 8;
    let Some(slice) = bytes.get(offset..end) else {
        return Err(GeoError::Container("truncated u64".to_string()));
    };
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

#[cfg(feature = "_source")]
fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "_source")]
fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
