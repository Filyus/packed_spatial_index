use super::ByteWriter;
#[cfg(feature = "stream")]
use super::{LoadError, read_u32_at};

/// The optional payload-prefix section: a contiguous, leaf-rank-indexed copy of
/// the first `record_stride` bytes of every payload blob.
///
/// Its whole reason to exist is read count. Those leading bytes are otherwise
/// strided through variable-length bodies, so a query that wants only them
/// cannot coalesce and issues one range read per match. Copied out into a dense
/// array they are read in a handful of runs, like the offset table beside them.
///
/// Optional and purely derived — every byte in it is also still at the head of
/// its blob, so a reader that skips this chunk gets the same answers from the
/// payload section, just with more reads.
pub(crate) const TAG_PFIX: [u8; 4] = *b"PFIX";

/// Minimum `PFIX` descriptor length an older reader must tolerate (`desc_len`
/// floor). Readers accept any `desc_len >= PFIX_DESC_LEN` and skip to the body.
pub(crate) const PFIX_DESC_LEN: usize = 12;

/// Decoded `PFIX` descriptor; returns it plus the slice that follows (the
/// `num_items * record_stride` prefix bytes).
#[cfg(feature = "stream")]
pub(crate) struct PfixDesc {
    /// Where the prefix bytes start relative to the chunk.
    pub(crate) desc_len: usize,
    /// Bytes per item. Always non-zero — a zero-stride section carries nothing
    /// and is never written.
    pub(crate) record_stride: usize,
}

#[cfg(feature = "stream")]
pub(crate) fn parse_pfix_chunk(chunk: &[u8]) -> Result<PfixDesc, LoadError> {
    if chunk.len() < PFIX_DESC_LEN {
        return Err(LoadError::Truncated);
    }
    let desc_len = read_u32_at(chunk, 0)? as usize;
    if desc_len < PFIX_DESC_LEN {
        return Err(LoadError::InvalidTree);
    }
    let ordering = chunk[4];
    let compression = chunk[5];
    // Only leaf-rank ordering, uncompressed prefixes exist in this version.
    if ordering != 0 || compression != 0 {
        return Err(LoadError::UnsupportedVersion);
    }
    let record_stride = read_u32_at(chunk, 8)? as usize;
    if record_stride == 0 {
        return Err(LoadError::InvalidTree);
    }
    Ok(PfixDesc {
        desc_len,
        record_stride,
    })
}

impl ByteWriter<'_> {
    /// Write the `PFIX` descriptor. Deliberately the same shape as
    /// [`write_pyld_desc`](Self::write_pyld_desc), so the two parse paths stay
    /// twins and a later field can be appended by growing `desc_len`.
    pub(crate) fn write_pfix_desc(&mut self, record_stride: u32) {
        self.write_u32(PFIX_DESC_LEN as u32);
        self.write_u8(0); // ordering = leaf rank
        self.write_u8(0); // compression = none
        self.write_u16(0); // reserved
        self.write_u32(record_stride);
    }
}
