use super::{LoadError, read_u64_at, read_u64_le_unchecked, usize_from_u64};

/// Validated slices of the optional trailing payload section. Borrowed by the
/// zero-copy views to serve `payload(id)`.
pub(crate) struct ParsedPayload<'a> {
    /// `(num_items + 1)` little-endian `u64` prefix offsets into `blobs`. Empty
    /// for a fixed-width payload (`stride != 0`), which needs no table.
    pub(crate) offsets: &'a [u8],
    /// Concatenated per-item payload bytes.
    pub(crate) blobs: &'a [u8],
    /// Fixed record stride in bytes, or `0` for a variable-width payload (use
    /// `offsets`). When non-zero every blob is exactly `stride` bytes, so the
    /// blob at leaf rank `r` is `blobs[r * stride ..][.. stride]`.
    pub(crate) stride: usize,
    /// The writer inferred `stride` from uniform blob lengths instead of being
    /// told it. The bytes are addressed identically either way; what differs is
    /// intent, so an API that reinterprets the blobs as typed records must refuse
    /// an inferred payload — uniform width is not a declaration of record type.
    pub(crate) stride_inferred: bool,
}

/// Whether this payload is a *declared* array of `stride`-byte records — the only
/// case in which reinterpreting the blobs as a typed record is warranted.
///
/// A width the writer merely inferred from uniform blobs says nothing about what
/// the blobs are: a corpus of 24-byte feature references is uniformly 24 bytes,
/// which is also the `f32` 2D triangle stride. The format carries no type tag, so
/// the declaration is the whole of the evidence and an inferred width is not one.
#[inline]
pub(crate) fn declares_records(payload: &ParsedPayload<'_>, stride: usize) -> bool {
    payload.stride == stride && !payload.stride_inferred
}

/// Slice the payload at leaf rank `r`: by arithmetic for a fixed-width payload,
/// or out of the leaf-ordered offset table for a variable-width one.
#[inline]
pub(crate) fn payload_slice<'a>(payload: &ParsedPayload<'a>, r: usize) -> &'a [u8] {
    if payload.stride != 0 {
        let start = r * payload.stride;
        &payload.blobs[start..start + payload.stride]
    } else {
        let start = read_u64_le_unchecked(payload.offsets, r * 8) as usize;
        let end = read_u64_le_unchecked(payload.offsets, (r + 1) * 8) as usize;
        &payload.blobs[start..end]
    }
}

/// Build the `insertion id -> leaf rank` map by inverting the leaf entries of
/// `indices` (which map leaf rank -> insertion id). The payload section is
/// leaf-ordered, so a view needs this to serve random `payload(id)` lookups.
pub(crate) fn build_id_to_leaf(indices: &[u8], num_items: usize) -> Vec<u32> {
    let mut id_to_leaf = vec![0u32; num_items];
    for r in 0..num_items {
        let id = read_u64_le_unchecked(indices, r * 8) as usize;
        id_to_leaf[id] = r as u32;
    }
    id_to_leaf
}

/// Validate and slice a `PYLD` chunk's post-descriptor bytes. A fixed-width
/// payload (`stride != 0`) is just `num_items * stride` blob bytes with no table;
/// a variable-width one is a `(num_items + 1)` prefix-offset table followed by the
/// blob region.
pub(crate) fn parse_payload_body(
    body: &[u8],
    num_items: usize,
    stride: usize,
    stride_inferred: bool,
) -> Result<ParsedPayload<'_>, LoadError> {
    if stride != 0 {
        let total = num_items
            .checked_mul(stride)
            .ok_or(LoadError::IntegerOverflow)?;
        if body.len() != total {
            return Err(LoadError::LengthMismatch {
                expected: total,
                actual: body.len(),
            });
        }
        return Ok(ParsedPayload {
            offsets: &[],
            blobs: body,
            stride,
            stride_inferred,
        });
    }
    let offsets_len = num_items
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or(LoadError::IntegerOverflow)?;
    if body.len() < offsets_len {
        return Err(LoadError::Truncated);
    }
    let offsets = &body[..offsets_len];
    let mut prev = 0u64;
    for i in 0..=num_items {
        let off = read_u64_at(offsets, i * 8)?;
        if (i == 0 && off != 0) || off < prev {
            return Err(LoadError::InvalidTree);
        }
        prev = off;
    }
    let blob_total = usize_from_u64(prev)?;
    let blobs = &body[offsets_len..];
    if blobs.len() != blob_total {
        return Err(LoadError::LengthMismatch {
            expected: offsets_len + blob_total,
            actual: body.len(),
        });
    }
    Ok(ParsedPayload {
        offsets,
        blobs,
        stride: 0,
        stride_inferred: false,
    })
}
