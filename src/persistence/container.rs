use super::{ByteWriter, LoadError, read_u32_at, read_u64_at, usize_from_u64};

// Packed Spatial Index file signature, at the start of the superblock; the
// format version follows it as a little-endian u64.
pub(super) const FORMAT_MAGIC: &[u8; 8] = b"PSINDEX\0";

/// Stored `version` value. Bumped only on a breaking layout change; a reader
/// rejects any other value.
pub(crate) const FORMAT_VERSION: u64 = 2;
/// Fixed superblock length: magic(8) + version(8) + chunk_count(4) + reserved(12).
pub(crate) const SUPERBLOCK_LEN: usize = 32;
/// One chunk-directory entry: tag(4) + flags(4) + offset(8) + length(8).
pub(crate) const CHUNK_ENTRY_LEN: usize = 24;
/// `flags` bit marking a chunk a reader must understand (else reject the file).
pub(crate) const CHUNK_FLAG_CRITICAL: u32 = 1;

/// A located chunk from the directory. Criticality is enforced during parsing
/// (an unknown critical chunk is rejected), so it is not retained here.
#[derive(Debug)]
pub(super) struct ChunkRef {
    pub(crate) tag: [u8; 4],
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

/// Parse and validate the superblock + chunk directory. Every entry's byte range
/// is checked against the buffer; an unknown **critical** chunk (tag not in
/// `known_critical`) is rejected. Does not read chunk contents.
pub(super) fn parse_container(
    bytes: &[u8],
    known_critical: &[[u8; 4]],
) -> Result<Vec<ChunkRef>, LoadError> {
    if bytes.len() < SUPERBLOCK_LEN {
        return Err(LoadError::Truncated);
    }
    if &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(LoadError::BadMagic);
    }
    if read_u64_at(bytes, 8)? != FORMAT_VERSION {
        return Err(LoadError::UnsupportedVersion);
    }
    let chunk_count = read_u32_at(bytes, 16)? as usize;

    let dir_len = chunk_count
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or(LoadError::IntegerOverflow)?;
    let dir_end = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or(LoadError::IntegerOverflow)?;
    if bytes.len() < dir_end {
        return Err(LoadError::Truncated);
    }

    let mut chunks = Vec::with_capacity(chunk_count);
    let mut max_end = dir_end;
    for i in 0..chunk_count {
        let base = SUPERBLOCK_LEN + i * CHUNK_ENTRY_LEN;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&bytes[base..base + 4]);
        let flags = read_u32_at(bytes, base + 4)?;
        let offset = read_u64_at(bytes, base + 8).and_then(usize_from_u64)?;
        let len = read_u64_at(bytes, base + 16).and_then(usize_from_u64)?;
        let end = offset.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
        if offset < dir_end || end > bytes.len() {
            return Err(LoadError::InvalidTree);
        }
        max_end = max_end.max(end);
        let critical = flags & CHUNK_FLAG_CRITICAL != 0;
        if critical && !known_critical.contains(&tag) {
            return Err(LoadError::UnsupportedVersion);
        }
        chunks.push(ChunkRef { tag, offset, len });
    }
    // Reject trailing bytes the directory does not account for (beyond the
    // last chunk's 8-byte alignment pad).
    if bytes.len() > ((max_end + 7) & !7) {
        return Err(LoadError::LengthMismatch {
            expected: max_end,
            actual: bytes.len(),
        });
    }
    Ok(chunks)
}

/// Find the first chunk with `tag`.
pub(super) fn find_chunk(chunks: &[ChunkRef], tag: [u8; 4]) -> Option<&ChunkRef> {
    chunks.iter().find(|c| c.tag == tag)
}

/// Round up to the next 8-byte boundary (chunks are 8-aligned).
fn align8(x: usize) -> Result<usize, LoadError> {
    x.checked_add(7)
        .map(|v| v & !7)
        .ok_or(LoadError::IntegerOverflow)
}

/// Plan a container from each chunk's content length: returns the total file
/// length and the absolute byte offset of each chunk's content (8-aligned).
pub(crate) fn plan_container(content_lens: &[usize]) -> Result<(usize, Vec<usize>), LoadError> {
    let dir_len = content_lens
        .len()
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or(LoadError::IntegerOverflow)?;
    let mut cur = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let mut offsets = Vec::with_capacity(content_lens.len());
    for &len in content_lens {
        cur = align8(cur)?;
        offsets.push(cur);
        cur = cur.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
    }
    cur = align8(cur)?;
    Ok((cur, offsets))
}

impl ByteWriter<'_> {
    pub(crate) fn write_superblock(&mut self, chunk_count: u32) {
        self.write_magic();
        self.write_u64(FORMAT_VERSION);
        self.write_u32(chunk_count);
        self.write_zeros(12);
    }

    pub(crate) fn write_chunk_entry(
        &mut self,
        tag: &[u8; 4],
        critical: bool,
        offset: usize,
        len: usize,
    ) {
        self.write_bytes(tag);
        self.write_u32(if critical { CHUNK_FLAG_CRITICAL } else { 0 });
        self.write_u64(offset as u64);
        self.write_u64(len as u64);
    }
}
