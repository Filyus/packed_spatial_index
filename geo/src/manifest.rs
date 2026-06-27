use crate::{GeoArtifactManifest, GeoError};

const FORMAT_MAGIC: &[u8; 8] = b"PSINDEX\0";
const FORMAT_VERSION: u64 = 2;
const SUPERBLOCK_LEN: usize = 32;
const CHUNK_ENTRY_LEN: usize = 24;
const CHUNK_FLAG_CRITICAL: u32 = 1;
const TAG_GEO_MANIFEST: [u8; 4] = *b"geoM";

#[derive(Debug, Clone)]
struct Chunk {
    tag: [u8; 4],
    flags: u32,
    content: Vec<u8>,
}

pub fn read_geo_manifest(bytes: &[u8]) -> Result<Option<GeoArtifactManifest>, GeoError> {
    let chunks = parse_chunks(bytes)?;
    let Some(chunk) = chunks.iter().find(|chunk| chunk.tag == TAG_GEO_MANIFEST) else {
        return Ok(None);
    };
    serde_json::from_slice(&chunk.content)
        .map(Some)
        .map_err(|e| GeoError::Container(e.to_string()))
}

pub(crate) fn append_geo_manifest(
    bytes: &[u8],
    manifest: &GeoArtifactManifest,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let mut chunks = parse_chunks(bytes)?;
    chunks.retain(|chunk| chunk.tag != TAG_GEO_MANIFEST);
    chunks.push(Chunk {
        tag: TAG_GEO_MANIFEST,
        flags: 0,
        content: serde_json::to_vec(manifest).map_err(|e| GeoError::Container(e.to_string()))?,
    });
    write_chunks(&chunks, out)
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

fn align8(value: usize) -> Result<usize, GeoError> {
    value
        .checked_add(7)
        .map(|v| v & !7)
        .ok_or_else(|| GeoError::Container("alignment overflow".to_string()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, GeoError> {
    let end = offset + 4;
    let Some(slice) = bytes.get(offset..end) else {
        return Err(GeoError::Container("truncated u32".to_string()));
    };
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, GeoError> {
    let end = offset + 8;
    let Some(slice) = bytes.get(offset..end) else {
        return Err(GeoError::Container("truncated u64".to_string()));
    };
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
