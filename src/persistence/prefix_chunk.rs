use super::ByteWriter;

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
