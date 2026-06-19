use crate::persistence::{LoadError, read_u64_le_unchecked};

use super::limits::Budget;
use super::{StreamError, read_index};

/// Byte locations of a streamed index's payload section.
#[derive(Clone)]
pub(crate) struct PayloadSection {
    /// Byte offset of the `(num_items + 1)` u64 prefix-offset table. Unused (and
    /// `0`) for a fixed-width payload, which has no table.
    pub(crate) offsets_start: u64,
    /// Byte offset of the blob region.
    pub(crate) blobs_start: u64,
    /// Total blob bytes (validated against the file length at open).
    pub(crate) blob_total: u64,
    /// Fixed record stride in bytes, or `0` for a variable-width payload (read
    /// the offset table). When non-zero the blob at leaf rank `r` is at
    /// `blobs_start + r * stride`, no table read needed.
    pub(crate) stride: u64,
}

/// Index of the last leaf position coalesced into the run starting at `j` (leaf
/// positions whose offset-table byte gap is within budget read together).
pub(super) fn payload_run_end(leaf_positions: &[usize], j: usize, max_gap: u64) -> usize {
    let mut k = j;
    while k + 1 < leaf_positions.len() {
        let gap = (leaf_positions[k + 1] - leaf_positions[k]) as u64 * 8;
        if gap > max_gap {
            break;
        }
        k += 1;
    }
    k
}

/// Validate and return the blob byte span `[blob_lo, blob_hi)` for a run whose
/// offset table `[lo ..= hi+1]` was read into `off_buf`.
pub(super) fn payload_blob_span(
    off_buf: &[u8],
    lo: usize,
    hi: usize,
    blob_total: u64,
) -> Result<(u64, u64), StreamError> {
    let blob_lo = read_u64_le_unchecked(off_buf, 0);
    let blob_hi = read_u64_le_unchecked(off_buf, (hi + 1 - lo) * 8);
    if blob_hi < blob_lo || blob_hi > blob_total {
        return Err(StreamError::Format(LoadError::InvalidTree));
    }
    Ok((blob_lo, blob_hi))
}

/// Emit `(insertion id, blob)` for every survivor in a payload run, slicing each
/// blob out of the run's fetched `blob_buf` and validating offsets/ids.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_run_payloads<F: FnMut(usize, &[u8])>(
    leaf_positions: &[usize],
    indices: &[u8],
    j: usize,
    k: usize,
    lo: usize,
    off_buf: &[u8],
    blob_lo: u64,
    blob_hi: u64,
    blob_buf: &[u8],
    num_items: usize,
    budget: &mut Budget,
    emit: &mut F,
) -> Result<(), StreamError> {
    for (offset, &p) in leaf_positions[j..=k].iter().enumerate() {
        let i = j + offset;
        let o0 = read_u64_le_unchecked(off_buf, (p - lo) * 8);
        let o1 = read_u64_le_unchecked(off_buf, (p + 1 - lo) * 8);
        // Untrusted offsets: the stream never validates the whole table, so a
        // run's entries may be out of order. Require `blob_lo <= o0 <= o1 <=
        // blob_hi` so the blob slice stays in `blob_buf` (a missing `o0 >=
        // blob_lo` check would underflow `o0 - blob_lo`).
        if o0 < blob_lo || o1 < o0 || o1 > blob_hi {
            return Err(StreamError::Format(LoadError::InvalidTree));
        }
        let id = read_index(indices, i)?;
        if id >= num_items {
            return Err(StreamError::Format(LoadError::InvalidTree));
        }
        budget.charge_item()?;
        emit(
            id,
            &blob_buf[(o0 - blob_lo) as usize..(o1 - blob_lo) as usize],
        );
    }
    Ok(())
}

/// Last leaf position coalesced into a fixed-width run starting at `j`: leaf
/// positions whose blob byte gap (`gap * stride`) is within budget read together.
pub(super) fn payload_run_end_fixed(
    leaf_positions: &[usize],
    j: usize,
    stride: usize,
    max_gap: u64,
) -> usize {
    let mut k = j;
    while k + 1 < leaf_positions.len() {
        let gap = (leaf_positions[k + 1] - leaf_positions[k]) as u64 * stride as u64;
        if gap > max_gap {
            break;
        }
        k += 1;
    }
    k
}

/// Emit `(insertion id, blob)` for a fixed-width payload run already read into
/// `blob_buf` (records `lo` through `leaf_positions[k]`, each `stride` bytes).
/// No offset table: blob of rank `p` is at `(p - lo) * stride`.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_run_payloads_fixed<F: FnMut(usize, &[u8])>(
    leaf_positions: &[usize],
    indices: &[u8],
    j: usize,
    k: usize,
    lo: usize,
    stride: usize,
    blob_buf: &[u8],
    num_items: usize,
    budget: &mut Budget,
    emit: &mut F,
) -> Result<(), StreamError> {
    for (offset, &p) in leaf_positions[j..=k].iter().enumerate() {
        let i = j + offset;
        // `leaf_positions` are sorted leaf ranks in `[lo, leaf_positions[k]]`, so
        // `within + stride` stays inside `blob_buf` (length `(k_pos + 1 - lo) *
        // stride`).
        let within = (p - lo) * stride;
        let id = read_index(indices, i)?;
        if id >= num_items {
            return Err(StreamError::Format(LoadError::InvalidTree));
        }
        budget.charge_item()?;
        emit(id, &blob_buf[within..within + stride]);
    }
    Ok(())
}
