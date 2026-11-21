use crate::persistence::{LoadError, read_u64_le_unchecked};

use super::limits::Budget;
use super::{PayloadPrefix, StreamError, read_index};

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

/// Byte locations of a streamed index's optional payload-prefix section: a
/// dense `num_items * stride` array of blob heads in leaf-rank order.
///
/// Purely a read-shape optimization. The same bytes are still at the head of
/// each payload blob, so a reader that ignores this answers identically — it
/// just pays one read per match instead of one per run.
#[derive(Clone)]
pub(crate) struct PrefixSection {
    /// Byte offset of the prefix array.
    pub(crate) start: u64,
    /// Bytes per item; the prefix for leaf rank `r` is at `start + r * stride`.
    pub(crate) stride: usize,
}

/// Sans-IO wave state machine for payload, prefix, and rank-paged reads.
///
/// Each wave produces independent `(offset, len)` ranges; the sync driver reads
/// them sequentially into a reused scratch buffer while the async driver issues
/// the entire wave concurrently via `read_batch_async`. The driver must feed
/// wave responses in ascending `run_idx` order (`0..reads.len()`).
pub(super) trait WaveMachine<F> {
    /// Populate `reads` with the next wave of independent `(offset, len)` ranges
    /// and return `Ok(true)`, or complete any zero-I/O final phase and return
    /// `Ok(false)`.
    fn next_wave(
        &mut self,
        reads: &mut Vec<(u64, usize)>,
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<bool, StreamError>;

    /// Process the fetched bytes for `reads[run_idx]` of the current wave.
    fn feed_read(
        &mut self,
        run_idx: usize,
        buf: &[u8],
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<(), StreamError>;
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

/// Coalesce sorted `leaf_positions` into `(start_idx, end_idx)` index runs
/// using the offset-table gap.
fn plan_offset_runs(leaf_positions: &[usize], max_gap: u64) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut j = 0;
    while j < leaf_positions.len() {
        let k = payload_run_end(leaf_positions, j, max_gap);
        runs.push((j, k));
        j = k + 1;
    }
    runs
}

/// Coalesce sorted `leaf_positions` into `(start_idx, end_idx)` index runs
/// for fixed-width records of `stride` bytes.
fn plan_fixed_runs(leaf_positions: &[usize], stride: usize, max_gap: u64) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut j = 0;
    while j < leaf_positions.len() {
        let k = payload_run_end_fixed(leaf_positions, j, stride, max_gap);
        runs.push((j, k));
        j = k + 1;
    }
    runs
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PayloadsPhase {
    NeedFixedBlobs { stride: usize },
    ReadingFixedBlobs { stride: usize },
    NeedOffsets,
    ReadingOffsets,
    ReadingBlobs,
    Done,
}

/// Sans-IO wave planner for streaming payloads of surviving leaves (fixed-width
/// or variable-width).
pub(super) struct PayloadsMachine<'a> {
    section: &'a PayloadSection,
    leaf_positions: &'a [usize],
    indices: &'a [u8],
    num_items: usize,
    runs: Vec<(usize, usize)>,
    blob_spans: Vec<(u64, u64)>,
    item_slices: Vec<(usize, usize)>,
    phase: PayloadsPhase,
}

impl<'a> PayloadsMachine<'a> {
    pub(super) fn new(
        section: &'a PayloadSection,
        leaf_positions: &'a [usize],
        indices: &'a [u8],
        num_items: usize,
        coalesce_gap: u64,
    ) -> Self {
        if leaf_positions.is_empty() {
            return Self {
                section,
                leaf_positions,
                indices,
                num_items,
                runs: Vec::new(),
                blob_spans: Vec::new(),
                item_slices: Vec::new(),
                phase: PayloadsPhase::Done,
            };
        }
        if section.stride != 0 {
            let stride = section.stride as usize;
            let runs = plan_fixed_runs(leaf_positions, stride, coalesce_gap);
            Self {
                section,
                leaf_positions,
                indices,
                num_items,
                runs,
                blob_spans: Vec::new(),
                item_slices: Vec::new(),
                phase: PayloadsPhase::NeedFixedBlobs { stride },
            }
        } else {
            let runs = plan_offset_runs(leaf_positions, coalesce_gap);
            let blob_spans = vec![(0, 0); runs.len()];
            let item_slices = vec![(0, 0); leaf_positions.len()];
            Self {
                section,
                leaf_positions,
                indices,
                num_items,
                runs,
                blob_spans,
                item_slices,
                phase: PayloadsPhase::NeedOffsets,
            }
        }
    }
}

impl<'a, F: FnMut(usize, &[u8])> WaveMachine<F> for PayloadsMachine<'a> {
    fn next_wave(
        &mut self,
        reads: &mut Vec<(u64, usize)>,
        _budget: &mut Budget,
        _emit: &mut F,
    ) -> Result<bool, StreamError> {
        reads.clear();
        match self.phase {
            PayloadsPhase::NeedFixedBlobs { stride } => {
                for &(j, k) in &self.runs {
                    let lo = self.leaf_positions[j];
                    reads.push((
                        self.section.blobs_start + (lo * stride) as u64,
                        (self.leaf_positions[k] + 1 - lo) * stride,
                    ));
                }
                self.phase = PayloadsPhase::ReadingFixedBlobs { stride };
                Ok(true)
            }
            PayloadsPhase::NeedOffsets => {
                for &(j, k) in &self.runs {
                    let lo = self.leaf_positions[j];
                    reads.push((
                        self.section.offsets_start + (lo * 8) as u64,
                        (self.leaf_positions[k] + 2 - lo) * 8,
                    ));
                }
                self.phase = PayloadsPhase::ReadingOffsets;
                Ok(true)
            }
            PayloadsPhase::ReadingOffsets => {
                for &(lo, hi) in &self.blob_spans {
                    reads.push((self.section.blobs_start + lo, (hi - lo) as usize));
                }
                self.phase = PayloadsPhase::ReadingBlobs;
                Ok(true)
            }
            PayloadsPhase::ReadingFixedBlobs { .. }
            | PayloadsPhase::ReadingBlobs
            | PayloadsPhase::Done => {
                self.phase = PayloadsPhase::Done;
                Ok(false)
            }
        }
    }

    fn feed_read(
        &mut self,
        run_idx: usize,
        buf: &[u8],
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<(), StreamError> {
        match self.phase {
            PayloadsPhase::ReadingFixedBlobs { stride } => {
                let (j, k) = self.runs[run_idx];
                emit_run_payloads_fixed(
                    self.leaf_positions,
                    self.indices,
                    j,
                    k,
                    self.leaf_positions[j],
                    stride,
                    buf,
                    self.num_items,
                    budget,
                    emit,
                )
            }
            PayloadsPhase::ReadingOffsets => {
                let (j, k) = self.runs[run_idx];
                let lo = self.leaf_positions[j];
                let hi = self.leaf_positions[k];
                let (blob_lo, blob_hi) = payload_blob_span(buf, lo, hi, self.section.blob_total)?;
                self.blob_spans[run_idx] = (blob_lo, blob_hi);
                for (offset, &p) in self.leaf_positions[j..=k].iter().enumerate() {
                    let o0 = read_u64_le_unchecked(buf, (p - lo) * 8);
                    let o1 = read_u64_le_unchecked(buf, (p + 1 - lo) * 8);
                    // Untrusted offsets: the stream never validates the whole table, so a
                    // run's entries may be out of order. Require `blob_lo <= o0 <= o1 <=
                    // blob_hi` so the blob slice stays in `blob_buf` (a missing `o0 >=
                    // blob_lo` check would underflow `o0 - blob_lo`).
                    if o0 < blob_lo || o1 < o0 || o1 > blob_hi {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    self.item_slices[j + offset] =
                        ((o0 - blob_lo) as usize, (o1 - blob_lo) as usize);
                }
                Ok(())
            }
            PayloadsPhase::ReadingBlobs => {
                let (j, k) = self.runs[run_idx];
                for i in j..=k {
                    let id = read_index(self.indices, i)?;
                    if id >= self.num_items {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    budget.charge_item()?;
                    let (start, end) = self.item_slices[i];
                    emit(id, &buf[start..end]);
                }
                Ok(())
            }
            _ => unreachable!("feed_read called in non-reading phase"),
        }
    }
}

/// One matching leaf's payload location resolved during a prefix visit.
struct PrefixSpan {
    /// Position within the leaf batch (indexes into the gathered id bytes).
    run_index: usize,
    /// Leaf rank — position in the leaf-ordered payload section.
    leaf_rank: usize,
    /// Blob start, relative to the blob region.
    blob_start: u64,
    /// Full payload byte length.
    payload_len: usize,
}

impl PrefixSpan {
    /// Exclusive end of the prefix read for this span.
    fn prefix_end(&self, prefix_len: usize) -> u64 {
        self.blob_start + self.payload_len.min(prefix_len) as u64
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrefixesPhase {
    NeedOffsets,
    ReadingOffsets,
    NeedPrefixes,
    ReadingPfix { stride: usize },
    ReadingBlobPrefixes,
    Done,
}

/// Sans-IO wave planner for streaming payload prefixes (fixed/variable-width,
/// with or without a dense `PFIX` chunk).
pub(super) struct PrefixesMachine<'a> {
    section: &'a PayloadSection,
    prefix_section: Option<&'a PrefixSection>,
    leaf_positions: &'a [usize],
    indices: &'a [u8],
    prefix_len: usize,
    num_items: usize,
    coalesce_gap: u64,
    prefix_coalesce_gap: u64,
    runs: Vec<(usize, usize)>,
    spans: Vec<PrefixSpan>,
    ranks: Vec<usize>,
    prefix_runs: Vec<(usize, usize, u64, u64)>,
    phase: PrefixesPhase,
}

impl<'a> PrefixesMachine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        section: &'a PayloadSection,
        prefix_section: Option<&'a PrefixSection>,
        leaf_positions: &'a [usize],
        indices: &'a [u8],
        prefix_len: usize,
        num_items: usize,
        coalesce_gap: u64,
        prefix_coalesce_gap: u64,
    ) -> Self {
        if leaf_positions.is_empty() {
            return Self {
                section,
                prefix_section,
                leaf_positions,
                indices,
                prefix_len,
                num_items,
                coalesce_gap,
                prefix_coalesce_gap,
                runs: Vec::new(),
                spans: Vec::new(),
                ranks: Vec::new(),
                prefix_runs: Vec::new(),
                phase: PrefixesPhase::Done,
            };
        }
        if section.stride != 0 {
            let stride = section.stride as usize;
            let spans = leaf_positions
                .iter()
                .enumerate()
                .map(|(i, &p)| PrefixSpan {
                    run_index: i,
                    leaf_rank: p,
                    blob_start: (p * stride) as u64,
                    payload_len: stride,
                })
                .collect();
            Self {
                section,
                prefix_section,
                leaf_positions,
                indices,
                prefix_len,
                num_items,
                coalesce_gap,
                prefix_coalesce_gap,
                runs: Vec::new(),
                spans,
                ranks: Vec::new(),
                prefix_runs: Vec::new(),
                phase: PrefixesPhase::NeedPrefixes,
            }
        } else {
            let runs = plan_offset_runs(leaf_positions, coalesce_gap);
            Self {
                section,
                prefix_section,
                leaf_positions,
                indices,
                prefix_len,
                num_items,
                coalesce_gap,
                prefix_coalesce_gap,
                runs,
                spans: Vec::with_capacity(leaf_positions.len()),
                ranks: Vec::new(),
                prefix_runs: Vec::new(),
                phase: PrefixesPhase::NeedOffsets,
            }
        }
    }
}

impl<'a, F: FnMut(PayloadPrefix<'_>)> WaveMachine<F> for PrefixesMachine<'a> {
    fn next_wave(
        &mut self,
        reads: &mut Vec<(u64, usize)>,
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<bool, StreamError> {
        reads.clear();
        match self.phase {
            PrefixesPhase::NeedOffsets => {
                for &(j, k) in &self.runs {
                    let lo = self.leaf_positions[j];
                    reads.push((
                        self.section.offsets_start + (lo * 8) as u64,
                        (self.leaf_positions[k] + 2 - lo) * 8,
                    ));
                }
                self.phase = PrefixesPhase::ReadingOffsets;
                Ok(true)
            }
            PrefixesPhase::ReadingOffsets | PrefixesPhase::NeedPrefixes => {
                if self.prefix_len == 0 {
                    for span in &self.spans {
                        let id = read_index(self.indices, span.run_index)?;
                        if id >= self.num_items {
                            return Err(StreamError::Format(LoadError::InvalidTree));
                        }
                        budget.charge_item()?;
                        emit(PayloadPrefix {
                            id,
                            leaf_rank: span.leaf_rank,
                            prefix: &[],
                            payload_len: span.payload_len,
                        });
                    }
                    self.phase = PrefixesPhase::Done;
                    return Ok(false);
                }

                // With a prefix section the same bytes sit in a dense array, so
                // the scan reads rank runs instead of one strided range per match.
                // The ordinary record gap applies: what lies between two prefixes
                // here is other prefixes, not bodies, so over-reading is cheap.
                if let Some(pfix) = self.prefix_section.filter(|p| self.prefix_len <= p.stride) {
                    let stride = pfix.stride;
                    self.ranks = self.spans.iter().map(|s| s.leaf_rank).collect();
                    self.runs = plan_fixed_runs(&self.ranks, stride, self.coalesce_gap);
                    for &(j, k) in &self.runs {
                        reads.push((
                            pfix.start + (self.ranks[j] * stride) as u64,
                            (self.ranks[k] + 1 - self.ranks[j]) * stride,
                        ));
                    }
                    self.phase = PrefixesPhase::ReadingPfix { stride };
                    return Ok(true);
                }

                let gap = self.prefix_coalesce_gap;
                self.prefix_runs.clear();
                let mut j = 0;
                while j < self.spans.len() {
                    let run_start = self.spans[j].blob_start;
                    let mut run_end = self.spans[j].prefix_end(self.prefix_len);
                    let mut k = j;
                    while k + 1 < self.spans.len() {
                        let next = &self.spans[k + 1];
                        if next.blob_start < run_start
                            || next.blob_start.saturating_sub(run_end) > gap
                        {
                            break;
                        }
                        run_end = run_end.max(next.prefix_end(self.prefix_len));
                        k += 1;
                    }
                    self.prefix_runs.push((j, k, run_start, run_end));
                    reads.push((
                        self.section.blobs_start + run_start,
                        (run_end - run_start) as usize,
                    ));
                    j = k + 1;
                }
                self.phase = PrefixesPhase::ReadingBlobPrefixes;
                Ok(true)
            }
            PrefixesPhase::ReadingPfix { .. }
            | PrefixesPhase::ReadingBlobPrefixes
            | PrefixesPhase::Done => {
                self.phase = PrefixesPhase::Done;
                Ok(false)
            }
        }
    }

    fn feed_read(
        &mut self,
        run_idx: usize,
        buf: &[u8],
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<(), StreamError> {
        match self.phase {
            PrefixesPhase::ReadingOffsets => {
                let (j, k) = self.runs[run_idx];
                debug_assert_eq!(self.spans.len(), j);
                let lo = self.leaf_positions[j];
                for (offset, &p) in self.leaf_positions[j..=k].iter().enumerate() {
                    let o0 = read_u64_le_unchecked(buf, (p - lo) * 8);
                    let o1 = read_u64_le_unchecked(buf, (p + 1 - lo) * 8);
                    if o1 < o0 || o1 > self.section.blob_total {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    self.spans.push(PrefixSpan {
                        run_index: j + offset,
                        leaf_rank: p,
                        blob_start: o0,
                        payload_len: (o1 - o0) as usize,
                    });
                }
                Ok(())
            }
            PrefixesPhase::ReadingPfix { stride } => {
                let (j, k) = self.runs[run_idx];
                let lo = self.ranks[j];
                for span in &self.spans[j..=k] {
                    let id = read_index(self.indices, span.run_index)?;
                    if id >= self.num_items {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    budget.charge_item()?;
                    let at = (span.leaf_rank - lo) * stride;
                    let take = span.payload_len.min(self.prefix_len);
                    emit(PayloadPrefix {
                        id,
                        leaf_rank: span.leaf_rank,
                        prefix: &buf[at..at + take],
                        payload_len: span.payload_len,
                    });
                }
                Ok(())
            }
            PrefixesPhase::ReadingBlobPrefixes => {
                let (j, k, run_start, _) = self.prefix_runs[run_idx];
                for span in &self.spans[j..=k] {
                    let id = read_index(self.indices, span.run_index)?;
                    if id >= self.num_items {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    budget.charge_item()?;
                    let at = (span.blob_start - run_start) as usize;
                    let take = span.payload_len.min(self.prefix_len);
                    emit(PayloadPrefix {
                        id,
                        leaf_rank: span.leaf_rank,
                        prefix: &buf[at..at + take],
                        payload_len: span.payload_len,
                    });
                }
                Ok(())
            }
            _ => unreachable!("feed_read called in non-reading phase"),
        }
    }
}

struct RankBlobSpan {
    rank: usize,
    blob_start: u64,
    blob_end: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RankPhase {
    NeedFixedBlobs { stride: usize },
    ReadingFixedBlobs { stride: usize },
    NeedOffsets,
    ReadingOffsets,
    ReadingBlobs,
    Done,
}

/// Sans-IO wave planner for fetching full payloads at explicit leaf ranks.
pub(super) struct RankPayloadsMachine<'a> {
    section: &'a PayloadSection,
    coalesce_gap: u64,
    ranks: Vec<usize>,
    runs: Vec<(usize, usize)>,
    blob_spans: Vec<RankBlobSpan>,
    blob_runs: Vec<(usize, usize, u64, u64)>,
    phase: RankPhase,
}

impl<'a> RankPayloadsMachine<'a> {
    pub(super) fn new(
        section: &'a PayloadSection,
        leaf_ranks: &[usize],
        num_items: usize,
        coalesce_gap: u64,
    ) -> Result<Self, StreamError> {
        let mut ranks = leaf_ranks.to_vec();
        ranks.sort_unstable();
        ranks.dedup();
        if ranks.last().is_some_and(|&max| max >= num_items) {
            return Err(StreamError::InvalidRank);
        }
        if ranks.is_empty() {
            return Ok(Self {
                section,
                coalesce_gap,
                ranks,
                runs: Vec::new(),
                blob_spans: Vec::new(),
                blob_runs: Vec::new(),
                phase: RankPhase::Done,
            });
        }
        if section.stride != 0 {
            let stride = section.stride as usize;
            let runs = plan_fixed_runs(&ranks, stride, coalesce_gap);
            Ok(Self {
                section,
                coalesce_gap,
                ranks,
                runs,
                blob_spans: Vec::new(),
                blob_runs: Vec::new(),
                phase: RankPhase::NeedFixedBlobs { stride },
            })
        } else {
            let runs = plan_offset_runs(&ranks, coalesce_gap);
            let cap = ranks.len();
            Ok(Self {
                section,
                coalesce_gap,
                ranks,
                runs,
                blob_spans: Vec::with_capacity(cap),
                blob_runs: Vec::new(),
                phase: RankPhase::NeedOffsets,
            })
        }
    }
}

impl<'a, F: FnMut(usize, &[u8])> WaveMachine<F> for RankPayloadsMachine<'a> {
    fn next_wave(
        &mut self,
        reads: &mut Vec<(u64, usize)>,
        _budget: &mut Budget,
        _emit: &mut F,
    ) -> Result<bool, StreamError> {
        reads.clear();
        match self.phase {
            RankPhase::NeedFixedBlobs { stride } => {
                for &(j, k) in &self.runs {
                    reads.push((
                        self.section.blobs_start + (self.ranks[j] * stride) as u64,
                        (self.ranks[k] + 1 - self.ranks[j]) * stride,
                    ));
                }
                self.phase = RankPhase::ReadingFixedBlobs { stride };
                Ok(true)
            }
            RankPhase::NeedOffsets => {
                for &(j, k) in &self.runs {
                    reads.push((
                        self.section.offsets_start + (self.ranks[j] * 8) as u64,
                        (self.ranks[k] + 2 - self.ranks[j]) * 8,
                    ));
                }
                self.phase = RankPhase::ReadingOffsets;
                Ok(true)
            }
            RankPhase::ReadingOffsets => {
                let gap = self.coalesce_gap;
                self.blob_runs.clear();
                let mut j = 0;
                while j < self.blob_spans.len() {
                    let run_start = self.blob_spans[j].blob_start;
                    let mut run_end = self.blob_spans[j].blob_end;
                    let mut k = j;
                    while k + 1 < self.blob_spans.len() {
                        let next = &self.blob_spans[k + 1];
                        if next.blob_start < run_start
                            || next.blob_start.saturating_sub(run_end) > gap
                        {
                            break;
                        }
                        run_end = run_end.max(next.blob_end);
                        k += 1;
                    }
                    self.blob_runs.push((j, k, run_start, run_end));
                    reads.push((
                        self.section.blobs_start + run_start,
                        (run_end - run_start) as usize,
                    ));
                    j = k + 1;
                }
                self.phase = RankPhase::ReadingBlobs;
                Ok(true)
            }
            RankPhase::ReadingFixedBlobs { .. } | RankPhase::ReadingBlobs | RankPhase::Done => {
                self.phase = RankPhase::Done;
                Ok(false)
            }
        }
    }

    fn feed_read(
        &mut self,
        run_idx: usize,
        buf: &[u8],
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<(), StreamError> {
        match self.phase {
            RankPhase::ReadingFixedBlobs { stride } => {
                let (j, k) = self.runs[run_idx];
                let lo = self.ranks[j];
                for &p in &self.ranks[j..=k] {
                    budget.charge_item()?;
                    let within = (p - lo) * stride;
                    emit(p, &buf[within..within + stride]);
                }
                Ok(())
            }
            RankPhase::ReadingOffsets => {
                let (j, k) = self.runs[run_idx];
                let lo = self.ranks[j];
                for &rank in &self.ranks[j..=k] {
                    let o0 = read_u64_le_unchecked(buf, (rank - lo) * 8);
                    let o1 = read_u64_le_unchecked(buf, (rank + 1 - lo) * 8);
                    if o1 < o0 || o1 > self.section.blob_total {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    self.blob_spans.push(RankBlobSpan {
                        rank,
                        blob_start: o0,
                        blob_end: o1,
                    });
                }
                Ok(())
            }
            RankPhase::ReadingBlobs => {
                let (j, k, blob_lo, _) = self.blob_runs[run_idx];
                for span in &self.blob_spans[j..=k] {
                    budget.charge_item()?;
                    emit(
                        span.rank,
                        &buf[(span.blob_start - blob_lo) as usize
                            ..(span.blob_end - blob_lo) as usize],
                    );
                }
                Ok(())
            }
            _ => unreachable!("feed_read called in non-reading phase"),
        }
    }
}
