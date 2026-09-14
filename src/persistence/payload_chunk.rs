use super::{ByteWriter, LoadError, read_u16_at, read_u32_at};

/// The optional payload section (descriptor + offset table + blobs). Optional —
/// an index-only reader skips it.
pub(crate) const TAG_PYLD: [u8; 4] = *b"PYLD";

/// Minimum `PYLD` descriptor length an older reader must tolerate (`desc_len`
/// floor). Readers accept any `desc_len >= PYLD_DESC_LEN` and skip to the body.
pub(crate) const PYLD_DESC_LEN: usize = 8;
/// `PYLD` descriptor length this version writes: the 8-byte base plus the
/// `record_stride` u32. A reader before this field existed reads only the base
/// and treats the payload as variable-width (`record_stride = 0`).
pub(crate) const PYLD_DESC_LEN_FIXED: usize = 12;

/// Flags bit (in the descriptor's reserved `u16`): the fixed width was *inferred*
/// from uniform blob lengths rather than declared by the caller.
///
/// The two produce the same bytes and the same `payload(id)`, but only a declared
/// width means the writer intended a record array — so a reader must not
/// reinterpret an inferred payload as typed records. Older readers never looked at
/// the reserved field, and every file they wrote has it `0` (= declared), so this
/// costs no bytes and changes nothing about how they are read.
pub(crate) const PYLD_FLAG_STRIDE_INFERRED: u16 = 1 << 0;

/// Decoded `PYLD` descriptor; returns it plus the slice that follows (offset
/// table + blobs).
pub(crate) struct PyldDesc {
    /// Only the streaming reader consults this (to locate the body); the
    /// in-memory parser uses the body slice it comes paired with.
    #[cfg_attr(not(feature = "stream"), allow(dead_code))]
    pub(crate) desc_len: usize,
    /// Fixed record stride in bytes, or `0` for a variable-width payload (offset
    /// table present). Read from the descriptor when `desc_len` covers it.
    pub(crate) record_stride: usize,
    /// The stride was inferred from uniform blob lengths, not declared. Always
    /// `false` for a variable-width payload and for every file written before the
    /// flag existed. See [`PYLD_FLAG_STRIDE_INFERRED`].
    pub(crate) stride_inferred: bool,
}

pub(crate) fn parse_pyld_chunk(chunk: &[u8]) -> Result<(PyldDesc, &[u8]), LoadError> {
    if chunk.len() < PYLD_DESC_LEN {
        return Err(LoadError::Truncated);
    }
    let desc_len = read_u32_at(chunk, 0)? as usize;
    if desc_len < PYLD_DESC_LEN || desc_len > chunk.len() {
        return Err(LoadError::InvalidTree);
    }
    let ordering = chunk[4];
    let compression = chunk[5];
    // Only leaf-rank ordering, uncompressed blobs exist in this version.
    if ordering != 0 || compression != 0 {
        return Err(LoadError::UnsupportedVersion);
    }
    // `record_stride` was appended after the 8-byte base; an older file without
    // it has `desc_len == 8` and is read as variable-width (stride 0).
    let record_stride = if desc_len >= PYLD_DESC_LEN_FIXED {
        read_u32_at(chunk, 8)? as usize
    } else {
        0
    };
    // Unknown flag bits stay ignored: the reserved field has never been validated,
    // and rejecting on it now would strand files a later version writes.
    let flags = read_u16_at(chunk, 6)?;
    let stride_inferred = record_stride != 0 && flags & PYLD_FLAG_STRIDE_INFERRED != 0;
    Ok((
        PyldDesc {
            desc_len,
            record_stride,
            stride_inferred,
        },
        &chunk[desc_len..],
    ))
}

impl ByteWriter<'_> {
    /// Write the `PYLD` descriptor. A variable-width payload keeps the original
    /// 8-byte descriptor (byte-identical to older files); a fixed-width one
    /// appends the `record_stride` field, growing `desc_len` to 12.
    ///
    /// `stride_inferred` records whether that width was deduced from uniform blobs
    /// rather than declared, and is meaningful only alongside a stride.
    pub(crate) fn write_pyld_desc(&mut self, record_stride: Option<u32>, stride_inferred: bool) {
        let desc_len = match record_stride {
            Some(_) => PYLD_DESC_LEN_FIXED,
            None => PYLD_DESC_LEN,
        };
        self.write_u32(desc_len as u32);
        self.write_u8(0); // ordering = leaf rank
        self.write_u8(0); // compression = none
        self.write_u16(if stride_inferred && record_stride.is_some() {
            PYLD_FLAG_STRIDE_INFERRED
        } else {
            0
        });
        if let Some(stride) = record_stride {
            self.write_u32(stride);
        }
    }
}
