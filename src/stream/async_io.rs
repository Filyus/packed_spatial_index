use std::io;

use crate::estimate::{Estimate, box_fraction_2d, box_fraction_3d, subtree_leaf_range};
use crate::geometry::{Box2D, Box3D, Overlaps2D, Overlaps3D};
use crate::persistence::{LoadError, SUPERBLOCK_LEN, read_u64_le_unchecked};

use super::limits::{Budget, COALESCE_GAP_BYTES};
use super::open::{OpenBytes, OpenStep, head_len, plan_open};
use super::payload::{
    PayloadSection, emit_run_payloads, emit_run_payloads_fixed, payload_blob_span, payload_run_end,
    payload_run_end_fixed,
};
use super::planner::{apply_gather_run, expand_frontier, plan_gather};
use super::{
    PayloadPrefix, StreamCore, StreamError, StreamIndex2D, StreamIndex2DF32, StreamIndex3D,
    StreamIndex3DF32, StreamLimits, parse_box2d, parse_box2d_f32, parse_box3d, parse_box3d_f32,
    read_index,
};

// ---- Async streaming (behind the `async` feature) ----
//
// Mirror of the synchronous traversal for sources whose reads are async (browser
// / edge worker over HTTP range or object storage). The descent logic is the
// same — only the reads are awaited; the overlap test and the result sink stay
// synchronous closures so no async closures are needed. (The sync and async
// paths are kept in lockstep by an equivalence test; a future sans-io refactor
// could share one core.)

/// Async counterpart of [`RangeReader`](super::RangeReader): read a byte range,
/// returning a future.
///
/// Implement this to query an index that lives behind async I/O — an HTTP range
/// request from WebAssembly, an object-storage `get(range)` in an edge worker.
/// The returned futures need not be `Send` (edge/browser executors are
/// single-threaded). See [`RangeReader`](super::RangeReader) for the sync
/// analogue and an HTTP implementation sketch.
#[cfg(feature = "async")]
#[allow(async_fn_in_trait, clippy::len_without_is_empty)]
pub trait AsyncRangeReader {
    /// Read exactly `buf.len()` bytes starting at `offset`.
    async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Total length in bytes, if known.
    fn len(&self) -> Option<u64> {
        None
    }
}

/// What a traversal collects at the leaves.
#[cfg(feature = "async")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    Ids,
    Payloads,
}

#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamCore<R> {
    /// Async mirror of [`open`](StreamCore::open) over the same [`plan_open`],
    /// except that each batch of reads is issued concurrently and the variable
    /// payload's total length is fetched speculatively with the descriptors:
    /// a cold open is the head plus one batch.
    async fn open_async(
        reader: R,
        dimensions: usize,
        coord_bytes: usize,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        let file_len = reader.len();
        let mut bytes = OpenBytes::default();
        let mut head = vec![0u8; head_len(&limits, file_len)];
        match reader.read_exact_at(0, &mut head).await {
            Ok(()) => bytes.insert(0, head),
            // See the sync `open`: a short source of unknown length.
            Err(e)
                if e.kind() == io::ErrorKind::UnexpectedEof
                    && file_len.is_none()
                    && head.len() > SUPERBLOCK_LEN =>
            {
                head.truncate(SUPERBLOCK_LEN);
                reader.read_exact_at(0, &mut head).await?;
                bytes.insert(0, head);
            }
            Err(e) => return Err(e.into()),
        }
        let gap = limits.coalesce_gap_bytes.unwrap_or(COALESCE_GAP_BYTES);
        loop {
            match plan_open(
                &mut bytes,
                dimensions,
                coord_bytes,
                &limits,
                file_len,
                gap,
                true,
            )? {
                OpenStep::Read(reads) => {
                    let mut bufs: Vec<Vec<u8>> =
                        reads.iter().map(|&(_, len)| vec![0u8; len]).collect();
                    let fetches = reads
                        .iter()
                        .zip(bufs.iter_mut())
                        .map(|(&(offset, _), buf)| {
                            reader.read_exact_at(offset, buf.as_mut_slice())
                        });
                    futures_util::future::try_join_all(fetches).await?;
                    for (&(offset, _), buf) in reads.iter().zip(bufs) {
                        bytes.insert(offset, buf);
                    }
                }
                OpenStep::Ready(layout) => return Ok(Self::from_layout(*layout, reader, limits)),
            }
        }
    }

    /// Async mirror of [`gather`](StreamCore::gather), but issues all of a
    /// level's coalesced runs concurrently (one buffer each). On a
    /// single-threaded async executor this puts several range fetches in flight
    /// at once, so the level's latency is one round trip rather than the sum.
    async fn gather_async(
        &self,
        positions: &[usize],
        section0: u64,
        stride: usize,
        cache: &[u8],
        out: &mut Vec<u8>,
        budget: &mut Budget,
    ) -> Result<(), StreamError> {
        let runs = plan_gather(
            positions,
            section0,
            stride,
            self.dir_node_start,
            cache,
            out,
            self.coalesce_gap(),
        );
        for run in &runs {
            budget.charge_read(run.len)?;
        }
        let mut bufs: Vec<Vec<u8>> = runs.iter().map(|run| vec![0u8; run.len]).collect();
        let reads = runs
            .iter()
            .zip(bufs.iter_mut())
            .map(|(run, buf)| self.reader.read_exact_at(run.offset, buf.as_mut_slice()));
        futures_util::future::try_join_all(reads).await?;
        for (run, buf) in runs.iter().zip(&bufs) {
            apply_gather_run(out, run, buf, stride);
        }
        Ok(())
    }

    /// Async mirror of [`gather_payloads`](StreamCore::gather_payloads). Reads
    /// every run's offset table concurrently, then every run's blobs
    /// concurrently — two round trips for the whole leaf frontier rather than two
    /// per run.
    async fn gather_payloads_async<F>(
        &self,
        section: &PayloadSection,
        leaf_positions: &[usize],
        indices: &[u8],
        budget: &mut Budget,
        sink: &mut F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        // Group leaf positions into coalesced runs.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut j = 0;
        while j < leaf_positions.len() {
            let k = payload_run_end(leaf_positions, j, self.coalesce_gap());
            runs.push((j, k));
            j = k + 1;
        }

        // Phase 1: read every run's offset table concurrently.
        let mut off_bufs: Vec<Vec<u8>> = runs
            .iter()
            .map(|&(j, k)| vec![0u8; (leaf_positions[k] + 2 - leaf_positions[j]) * 8])
            .collect();
        for buf in &off_bufs {
            budget.charge_read(buf.len())?;
        }
        let off_reads = runs.iter().zip(off_bufs.iter_mut()).map(|(&(j, _), buf)| {
            let lo = leaf_positions[j];
            self.reader
                .read_exact_at(section.offsets_start + (lo * 8) as u64, buf.as_mut_slice())
        });
        futures_util::future::try_join_all(off_reads).await?;

        // Validate each run's blob span.
        let mut spans = Vec::with_capacity(runs.len());
        for (&(j, k), off_buf) in runs.iter().zip(&off_bufs) {
            spans.push(payload_blob_span(
                off_buf,
                leaf_positions[j],
                leaf_positions[k],
                section.blob_total,
            )?);
        }

        // Phase 2: read every run's blobs concurrently (empty spans are no-ops).
        let mut blob_bufs: Vec<Vec<u8>> = spans
            .iter()
            .map(|&(lo, hi)| vec![0u8; (hi - lo) as usize])
            .collect();
        for buf in &blob_bufs {
            if !buf.is_empty() {
                budget.charge_read(buf.len())?;
            }
        }
        let blob_reads = spans
            .iter()
            .zip(blob_bufs.iter_mut())
            .map(|(&(lo, _), buf)| {
                self.reader
                    .read_exact_at(section.blobs_start + lo, buf.as_mut_slice())
            });
        futures_util::future::try_join_all(blob_reads).await?;

        // Emit every run.
        for ((&(j, k), off_buf), (&(blob_lo, blob_hi), blob_buf)) in
            runs.iter().zip(&off_bufs).zip(spans.iter().zip(&blob_bufs))
        {
            emit_run_payloads(
                leaf_positions,
                indices,
                j,
                k,
                leaf_positions[j],
                off_buf,
                blob_lo,
                blob_hi,
                blob_buf,
                self.num_items,
                budget,
                sink,
            )?;
        }
        Ok(())
    }

    /// Fixed-width async payload gather: one contiguous blob read per coalesced
    /// run, all runs issued concurrently. No offset-table phase (the variable
    /// `gather_payloads_async` needs two round trips; this needs one).
    async fn gather_payloads_fixed_async<F>(
        &self,
        section: &PayloadSection,
        leaf_positions: &[usize],
        indices: &[u8],
        budget: &mut Budget,
        sink: &mut F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        let stride = section.stride as usize;
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut j = 0;
        while j < leaf_positions.len() {
            let k = payload_run_end_fixed(leaf_positions, j, stride, self.coalesce_gap());
            runs.push((j, k));
            j = k + 1;
        }

        let mut blob_bufs: Vec<Vec<u8>> = runs
            .iter()
            .map(|&(j, k)| vec![0u8; (leaf_positions[k] + 1 - leaf_positions[j]) * stride])
            .collect();
        for buf in &blob_bufs {
            budget.charge_read(buf.len())?;
        }
        let reads = runs.iter().zip(blob_bufs.iter_mut()).map(|(&(j, _), buf)| {
            let lo = leaf_positions[j];
            self.reader.read_exact_at(
                section.blobs_start + (lo * stride) as u64,
                buf.as_mut_slice(),
            )
        });
        futures_util::future::try_join_all(reads).await?;

        for (&(j, k), blob_buf) in runs.iter().zip(&blob_bufs) {
            emit_run_payloads_fixed(
                leaf_positions,
                indices,
                j,
                k,
                leaf_positions[j],
                stride,
                blob_buf,
                self.num_items,
                budget,
                sink,
            )?;
        }
        Ok(())
    }

    async fn visit_payload_prefixes_async<O, F>(
        &self,
        overlaps: O,
        prefix_len: usize,
        mut emit: F,
    ) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(PayloadPrefix<'_>),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        if self.num_items == 0 {
            return Ok(());
        }

        let mut budget = Budget::new(self.limits);
        let mut frontier = vec![self.num_nodes - 1];
        let mut level = self.level_count - 1;
        let mut boxes = Vec::new();
        let mut indices = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();

        loop {
            self.gather_async(
                &frontier,
                self.box0,
                self.box_stride,
                &self.dir_boxes,
                &mut boxes,
                &mut budget,
            )
            .await?;
            survivors.clear();
            indices.clear();
            for (i, &pos) in frontier.iter().enumerate() {
                let slot = i * self.box_stride;
                if overlaps(&boxes[slot..slot + self.record]) {
                    survivors.push(pos);
                    if self.interleaved {
                        indices
                            .extend_from_slice(&boxes[slot + self.record..slot + self.record + 8]);
                    }
                }
            }
            if survivors.is_empty() {
                return Ok(());
            }

            if !self.interleaved {
                self.gather_async(
                    &survivors,
                    self.idx0,
                    8,
                    &self.dir_indices,
                    &mut indices,
                    &mut budget,
                )
                .await?;
            }

            if level == 0 {
                self.gather_payload_prefixes_async(
                    section,
                    &survivors,
                    &indices,
                    prefix_len,
                    &mut budget,
                    &mut emit,
                )
                .await?;
                return Ok(());
            }

            frontier = expand_frontier(
                &self.level_bounds,
                self.node_size,
                level,
                survivors.len(),
                &indices,
            )?;
            level -= 1;
        }
    }

    async fn gather_payload_prefixes_async<F>(
        &self,
        section: &PayloadSection,
        leaf_positions: &[usize],
        indices: &[u8],
        prefix_len: usize,
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(PayloadPrefix<'_>),
    {
        let mut spans = Vec::with_capacity(leaf_positions.len());
        if section.stride != 0 {
            let stride = section.stride as usize;
            for (i, &p) in leaf_positions.iter().enumerate() {
                spans.push(AsyncPrefixSpan {
                    run_index: i,
                    leaf_rank: p,
                    blob_start: (p * stride) as u64,
                    payload_len: stride,
                });
            }
        } else {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut j = 0;
            while j < leaf_positions.len() {
                let k = payload_run_end(leaf_positions, j, self.coalesce_gap());
                runs.push((j, k));
                j = k + 1;
            }

            let mut off_bufs: Vec<Vec<u8>> = runs
                .iter()
                .map(|&(j, k)| vec![0u8; (leaf_positions[k] + 2 - leaf_positions[j]) * 8])
                .collect();
            for buf in &off_bufs {
                budget.charge_read(buf.len())?;
            }
            let off_reads = runs.iter().zip(off_bufs.iter_mut()).map(|(&(j, _), buf)| {
                let lo = leaf_positions[j];
                self.reader
                    .read_exact_at(section.offsets_start + (lo * 8) as u64, buf.as_mut_slice())
            });
            futures_util::future::try_join_all(off_reads).await?;

            for (&(j, k), off_buf) in runs.iter().zip(&off_bufs) {
                let lo = leaf_positions[j];
                for (offset, &p) in leaf_positions[j..=k].iter().enumerate() {
                    let o0 = read_u64_le_unchecked(off_buf, (p - lo) * 8);
                    let o1 = read_u64_le_unchecked(off_buf, (p + 1 - lo) * 8);
                    if o1 < o0 || o1 > section.blob_total {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    spans.push(AsyncPrefixSpan {
                        run_index: j + offset,
                        leaf_rank: p,
                        blob_start: o0,
                        payload_len: (o1 - o0) as usize,
                    });
                }
            }
        }

        if prefix_len == 0 {
            for span in &spans {
                let id = read_index(indices, span.run_index)?;
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
            return Ok(());
        }

        // With a prefix section the same bytes sit in a dense array, so the scan
        // reads rank runs instead of one strided range per match. The ordinary
        // record gap applies: what lies between two prefixes here is other
        // prefixes, not bodies, so over-reading is cheap.
        if let Some(pfix) = self.prefix.as_ref().filter(|p| prefix_len <= p.stride) {
            let stride = pfix.stride;
            let ranks: Vec<usize> = spans.iter().map(|s| s.leaf_rank).collect();
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut j = 0;
            while j < ranks.len() {
                let k = payload_run_end_fixed(&ranks, j, stride, self.coalesce_gap());
                runs.push((j, k));
                j = k + 1;
            }
            let mut bufs: Vec<Vec<u8>> = runs
                .iter()
                .map(|&(j, k)| vec![0u8; (ranks[k] + 1 - ranks[j]) * stride])
                .collect();
            for buf in &bufs {
                budget.charge_read(buf.len())?;
            }
            let reads = runs.iter().zip(bufs.iter_mut()).map(|(&(j, _), buf)| {
                self.reader
                    .read_exact_at(pfix.start + (ranks[j] * stride) as u64, buf.as_mut_slice())
            });
            futures_util::future::try_join_all(reads).await?;

            for (&(j, k), read_buf) in runs.iter().zip(&bufs) {
                let lo = ranks[j];
                for span in &spans[j..=k] {
                    let id = read_index(indices, span.run_index)?;
                    if id >= self.num_items {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    budget.charge_item()?;
                    let at = (span.leaf_rank - lo) * stride;
                    let take = span.payload_len.min(prefix_len);
                    emit(PayloadPrefix {
                        id,
                        leaf_rank: span.leaf_rank,
                        prefix: &read_buf[at..at + take],
                        payload_len: span.payload_len,
                    });
                }
            }
            return Ok(());
        }

        let mut prefix_runs: Vec<(usize, usize, u64, u64)> = Vec::new();
        let gap = self.prefix_coalesce_gap(prefix_len);
        let mut j = 0;
        while j < spans.len() {
            let run_start = spans[j].blob_start;
            let mut run_end = spans[j].prefix_end(prefix_len);
            let mut k = j;
            while k + 1 < spans.len() {
                let next = &spans[k + 1];
                if next.blob_start < run_start || next.blob_start.saturating_sub(run_end) > gap {
                    break;
                }
                run_end = run_end.max(next.prefix_end(prefix_len));
                k += 1;
            }
            prefix_runs.push((j, k, run_start, run_end));
            j = k + 1;
        }

        let mut bufs: Vec<Vec<u8>> = prefix_runs
            .iter()
            .map(|&(_, _, start, end)| vec![0u8; (end - start) as usize])
            .collect();
        for buf in &bufs {
            if !buf.is_empty() {
                budget.charge_read(buf.len())?;
            }
        }
        let reads = prefix_runs
            .iter()
            .zip(bufs.iter_mut())
            .filter(|(_, buf)| !buf.is_empty())
            .map(|(&(_, _, start, _), buf)| {
                self.reader
                    .read_exact_at(section.blobs_start + start, buf.as_mut_slice())
            });
        futures_util::future::try_join_all(reads).await?;

        for (&(j, k, run_start, _), read_buf) in prefix_runs.iter().zip(&bufs) {
            for span in &spans[j..=k] {
                let id = read_index(indices, span.run_index)?;
                if id >= self.num_items {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                budget.charge_item()?;
                let at = (span.blob_start - run_start) as usize;
                let take = span.payload_len.min(prefix_len);
                emit(PayloadPrefix {
                    id,
                    leaf_rank: span.leaf_rank,
                    prefix: &read_buf[at..at + take],
                    payload_len: span.payload_len,
                });
            }
        }
        Ok(())
    }

    async fn visit_payloads_at_ranks_async<F>(
        &self,
        leaf_ranks: &[usize],
        mut emit: F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        let mut ranks = leaf_ranks.to_vec();
        ranks.sort_unstable();
        ranks.dedup();
        if ranks.last().is_some_and(|&max| max >= self.num_items) {
            return Err(StreamError::InvalidRank);
        }
        let mut budget = Budget::new(self.limits);

        if section.stride != 0 {
            let stride = section.stride as usize;
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut j = 0;
            while j < ranks.len() {
                let k = payload_run_end_fixed(&ranks, j, stride, self.coalesce_gap());
                runs.push((j, k));
                j = k + 1;
            }
            let mut bufs: Vec<Vec<u8>> = runs
                .iter()
                .map(|&(j, k)| vec![0u8; (ranks[k] + 1 - ranks[j]) * stride])
                .collect();
            for buf in &bufs {
                budget.charge_read(buf.len())?;
            }
            let reads = runs.iter().zip(bufs.iter_mut()).map(|(&(j, _), buf)| {
                let lo = ranks[j];
                self.reader.read_exact_at(
                    section.blobs_start + (lo * stride) as u64,
                    buf.as_mut_slice(),
                )
            });
            futures_util::future::try_join_all(reads).await?;
            for (&(j, k), buf) in runs.iter().zip(&bufs) {
                let lo = ranks[j];
                for &p in &ranks[j..=k] {
                    budget.charge_item()?;
                    let within = (p - lo) * stride;
                    emit(p, &buf[within..within + stride]);
                }
            }
            return Ok(());
        }

        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut j = 0;
        while j < ranks.len() {
            let k = payload_run_end(&ranks, j, self.coalesce_gap());
            runs.push((j, k));
            j = k + 1;
        }

        let mut off_bufs: Vec<Vec<u8>> = runs
            .iter()
            .map(|&(j, k)| vec![0u8; (ranks[k] + 2 - ranks[j]) * 8])
            .collect();
        for buf in &off_bufs {
            budget.charge_read(buf.len())?;
        }
        let off_reads = runs.iter().zip(off_bufs.iter_mut()).map(|(&(j, _), buf)| {
            let lo = ranks[j];
            self.reader
                .read_exact_at(section.offsets_start + (lo * 8) as u64, buf.as_mut_slice())
        });
        futures_util::future::try_join_all(off_reads).await?;

        let mut blob_spans = Vec::with_capacity(ranks.len());
        for (&(j, k), off_buf) in runs.iter().zip(&off_bufs) {
            let lo = ranks[j];
            for &rank in &ranks[j..=k] {
                let o0 = read_u64_le_unchecked(off_buf, (rank - lo) * 8);
                let o1 = read_u64_le_unchecked(off_buf, (rank + 1 - lo) * 8);
                if o1 < o0 || o1 > section.blob_total {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                blob_spans.push(AsyncRankBlobSpan {
                    rank,
                    blob_start: o0,
                    blob_end: o1,
                });
            }
        }

        let mut blob_runs: Vec<(usize, usize, u64, u64)> = Vec::new();
        let gap = self.coalesce_gap();
        let mut j = 0;
        while j < blob_spans.len() {
            let run_start = blob_spans[j].blob_start;
            let mut run_end = blob_spans[j].blob_end;
            let mut k = j;
            while k + 1 < blob_spans.len() {
                let next = &blob_spans[k + 1];
                if next.blob_start < run_start || next.blob_start.saturating_sub(run_end) > gap {
                    break;
                }
                run_end = run_end.max(next.blob_end);
                k += 1;
            }
            blob_runs.push((j, k, run_start, run_end));
            j = k + 1;
        }

        let mut blob_bufs: Vec<Vec<u8>> = blob_runs
            .iter()
            .map(|&(_, _, lo, hi)| vec![0u8; (hi - lo) as usize])
            .collect();
        for buf in &blob_bufs {
            if !buf.is_empty() {
                budget.charge_read(buf.len())?;
            }
        }
        let blob_reads = blob_runs
            .iter()
            .zip(blob_bufs.iter_mut())
            .map(|(&(_, _, lo, _), buf)| {
                self.reader
                    .read_exact_at(section.blobs_start + lo, buf.as_mut_slice())
            });
        futures_util::future::try_join_all(blob_reads).await?;

        for (&(j, k, blob_lo, _blob_hi), blob_buf) in blob_runs.iter().zip(&blob_bufs) {
            for span in &blob_spans[j..=k] {
                budget.charge_item()?;
                emit(
                    span.rank,
                    &blob_buf
                        [(span.blob_start - blob_lo) as usize..(span.blob_end - blob_lo) as usize],
                );
            }
        }
        Ok(())
    }

    /// Async mirror of [`estimate`](StreamCore::estimate): the same level-by-level
    /// bracket, stopping at `stop_level`, with each level below the directory
    /// floor fetched as one concurrent gather. At or above the floor it awaits
    /// nothing that reads.
    pub(crate) async fn estimate_async<O, C, Fr>(
        &self,
        stop_level: usize,
        overlaps: O,
        contains: C,
        fraction: Fr,
    ) -> Result<Estimate, StreamError>
    where
        O: Fn(&[u8]) -> bool,
        C: Fn(&[u8]) -> bool,
        Fr: Fn(&[u8]) -> f64,
    {
        let mut out = Estimate {
            lower: 0,
            upper: 0,
            estimate: 0.0,
            nodes_tested: 0,
        };
        if self.num_items == 0 {
            return Ok(out);
        }

        let mut budget = Budget::new(self.limits);
        let mut frontier = vec![self.num_nodes - 1];
        let mut level = self.level_count - 1;
        let mut boxes = Vec::new();
        let mut indices = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();

        loop {
            self.gather_async(
                &frontier,
                self.box0,
                self.box_stride,
                &self.dir_boxes,
                &mut boxes,
                &mut budget,
            )
            .await?;
            let level_start = if level == 0 {
                0
            } else {
                self.level_bounds[level - 1]
            };
            survivors.clear();
            indices.clear();
            for (i, &pos) in frontier.iter().enumerate() {
                let slot = i * self.box_stride;
                let record = &boxes[slot..slot + self.record];
                out.nodes_tested += 1;
                if !overlaps(record) {
                    continue;
                }
                let (start, end) =
                    subtree_leaf_range(pos, level, level_start, self.node_size, self.num_items);
                let size = end - start;
                if level == 0 || contains(record) {
                    out.lower += size;
                    out.upper += size;
                    out.estimate += size as f64;
                    continue;
                }
                if level <= stop_level {
                    out.upper += size;
                    out.estimate += size as f64 * fraction(record);
                    continue;
                }
                survivors.push(pos);
                if self.interleaved {
                    indices.extend_from_slice(&boxes[slot + self.record..slot + self.record + 8]);
                }
            }
            if survivors.is_empty() {
                break;
            }
            if !self.interleaved {
                self.gather_async(
                    &survivors,
                    self.idx0,
                    8,
                    &self.dir_indices,
                    &mut indices,
                    &mut budget,
                )
                .await?;
            }
            frontier = expand_frontier(
                &self.level_bounds,
                self.node_size,
                level,
                survivors.len(),
                &indices,
            )?;
            level -= 1;
        }
        out.estimate = out.estimate.clamp(out.lower as f64, out.upper as f64);
        Ok(out)
    }

    /// Async mirror of the synchronous traversal, parameterized by `want` (ids or
    /// id+payload). `overlaps` and `sink` are synchronous; only reads are awaited.
    async fn traverse_async<O, F>(
        &self,
        overlaps: O,
        want: Want,
        mut sink: F,
    ) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize, &[u8]),
    {
        let section = if want == Want::Payloads {
            Some(self.payload.as_ref().ok_or(StreamError::NoPayload)?)
        } else {
            None
        };
        if self.num_items == 0 {
            return Ok(());
        }

        let mut budget = Budget::new(self.limits);
        let mut frontier = vec![self.num_nodes - 1];
        let mut level = self.level_count - 1;
        let mut boxes = Vec::new();
        let mut indices = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();

        loop {
            // One gather per level fetches each frontier node's box (interleaved:
            // box + index together; SoA: box only).
            self.gather_async(
                &frontier,
                self.box0,
                self.box_stride,
                &self.dir_boxes,
                &mut boxes,
                &mut budget,
            )
            .await?;
            survivors.clear();
            indices.clear();
            for (i, &pos) in frontier.iter().enumerate() {
                let slot = i * self.box_stride;
                if overlaps(&boxes[slot..slot + self.record]) {
                    survivors.push(pos);
                    if self.interleaved {
                        indices
                            .extend_from_slice(&boxes[slot + self.record..slot + self.record + 8]);
                    }
                }
            }
            if survivors.is_empty() {
                return Ok(());
            }

            if !self.interleaved {
                self.gather_async(
                    &survivors,
                    self.idx0,
                    8,
                    &self.dir_indices,
                    &mut indices,
                    &mut budget,
                )
                .await?;
            }

            if level == 0 {
                match section {
                    Some(section) if section.stride != 0 => {
                        self.gather_payloads_fixed_async(
                            section,
                            &survivors,
                            &indices,
                            &mut budget,
                            &mut sink,
                        )
                        .await?;
                    }
                    Some(section) => {
                        self.gather_payloads_async(
                            section,
                            &survivors,
                            &indices,
                            &mut budget,
                            &mut sink,
                        )
                        .await?;
                    }
                    None => {
                        for i in 0..survivors.len() {
                            let id = read_index(&indices, i)?;
                            if id >= self.num_items {
                                return Err(StreamError::Format(LoadError::InvalidTree));
                            }
                            budget.charge_item()?;
                            sink(id, &[]);
                        }
                    }
                }
                return Ok(());
            }

            frontier = expand_frontier(
                &self.level_bounds,
                self.node_size,
                level,
                survivors.len(),
                &indices,
            )?;
            level -= 1;
        }
    }
}

struct AsyncPrefixSpan {
    run_index: usize,
    leaf_rank: usize,
    blob_start: u64,
    payload_len: usize,
}

struct AsyncRankBlobSpan {
    rank: usize,
    blob_start: u64,
    blob_end: u64,
}

impl AsyncPrefixSpan {
    fn prefix_end(&self, prefix_len: usize) -> u64 {
        self.blob_start + self.payload_len.min(prefix_len) as u64
    }
}

/// Streaming reader for a 2D `f64` index over async I/O. Mirrors
/// [`StreamIndex2D`]; use it when reads return futures (e.g. browser / edge
/// worker). Behind the `async` feature.
#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamIndex2D<R> {
    /// Open and validate a 2D `f64` index from an async `reader`.
    pub async fn open_async(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits_async(reader, StreamLimits::default()).await
    }

    /// Open from an async `reader` with per-query [`StreamLimits`]. See
    /// [`StreamIndex2D::open_with_limits`].
    pub async fn open_with_limits_async(
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open_async(reader, 2, 8, limits).await?,
        })
    }

    /// Stream the indices of every item whose box intersects `query`.
    pub async fn search_async(&self, query: Box2D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box2d(r).overlaps(query),
                Want::Ids,
                |id, _| out.push(id),
            )
            .await?;
        Ok(out)
    }

    /// Return the number of items overlapping `query`, counted without
    /// collecting them.
    pub async fn count_async(&self, query: Box2D) -> Result<usize, StreamError> {
        self.count_region_async(&query).await
    }

    /// Async mirror of [`estimate_count`](Self::estimate_count): bracket and
    /// estimate how many items `query` would hit, from node boxes, stopping at
    /// `stop_level`. With `stop_level >= directory_floor()` nothing is read, so
    /// a worker can decide whether a query is worth its round trips first.
    pub async fn estimate_count_async(
        &self,
        query: Box2D,
        stop_level: usize,
    ) -> Result<Estimate, StreamError> {
        self.core
            .estimate_async(
                stop_level,
                |record| parse_box2d(record).overlaps(query),
                |record| query.contains(parse_box2d(record)),
                |record| box_fraction_2d(parse_box2d(record), query),
            )
            .await
    }

    /// Stream `(item index, payload blob)` for every item intersecting `query`.
    pub async fn search_payloads_async(
        &self,
        query: Box2D,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box2d(r).overlaps(query),
                Want::Payloads,
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose box overlaps the region `query` —
    /// any [`Overlaps2D`] shape, not just a box.
    pub async fn visit_region_async<Q, F>(
        &self,
        query: &Q,
        mut visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box2d(r)),
                Want::Ids,
                |id, _| visitor(id),
            )
            .await
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub async fn search_region_async<Q: Overlaps2D>(
        &self,
        query: &Q,
    ) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region_async(query, |index| out.push(index))
            .await?;
        Ok(out)
    }

    /// Return the number of items whose box overlaps the region `query`.
    pub async fn count_region_async<Q: Overlaps2D>(&self, query: &Q) -> Result<usize, StreamError> {
        let mut count = 0usize;
        self.visit_region_async(query, |_| count += 1).await?;
        Ok(count)
    }

    /// Visit `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub async fn visit_payloads_region_async<Q, F>(
        &self,
        query: &Q,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize, &[u8]),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box2d(r)),
                Want::Payloads,
                visitor,
            )
            .await
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub async fn search_payloads_region_async<Q: Overlaps2D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region_async(query, |id, blob| out.push((id, blob.to_vec())))
            .await?;
        Ok(out)
    }

    /// Async counterpart of [`StreamIndex2D::search_payload_prefixes_each`].
    pub async fn visit_payload_prefixes_async<F: FnMut(PayloadPrefix<'_>)>(
        &self,
        query: Box2D,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payload_prefixes_async(
                |record| parse_box2d(record).overlaps(query),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex2D::search_payload_prefixes_region_each`].
    pub async fn visit_payload_prefixes_region_async<Q, F>(
        &self,
        query: &Q,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(PayloadPrefix<'_>),
    {
        self.core
            .visit_payload_prefixes_async(
                |record| query.overlaps_box(parse_box2d(record)),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex2D::payloads_at_ranks_each`].
    pub async fn visit_payloads_at_ranks_async<F: FnMut(usize, &[u8])>(
        &self,
        leaf_ranks: &[usize],
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads_at_ranks_async(leaf_ranks, visitor)
            .await
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload_async(&self) -> bool {
        self.core.has_payload()
    }
}

/// Streaming reader for a 3D `f64` index over async I/O. See [`StreamIndex2D`]'s
/// async methods. Behind the `async` feature.
#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamIndex3D<R> {
    /// Open and validate a 3D `f64` index from an async `reader`.
    pub async fn open_async(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits_async(reader, StreamLimits::default()).await
    }

    /// Open from an async `reader` with per-query [`StreamLimits`].
    pub async fn open_with_limits_async(
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open_async(reader, 3, 8, limits).await?,
        })
    }

    /// Stream the indices of every item whose box intersects `query`.
    pub async fn search_async(&self, query: Box3D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box3d(r).overlaps(query),
                Want::Ids,
                |id, _| out.push(id),
            )
            .await?;
        Ok(out)
    }

    /// Return the number of items overlapping `query`, counted without
    /// collecting them.
    pub async fn count_async(&self, query: Box3D) -> Result<usize, StreamError> {
        self.count_region_async(&query).await
    }

    /// Async mirror of [`estimate_count`](Self::estimate_count): bracket and
    /// estimate how many items `query` would hit, from node boxes, stopping at
    /// `stop_level`. With `stop_level >= directory_floor()` nothing is read, so
    /// a worker can decide whether a query is worth its round trips first.
    pub async fn estimate_count_async(
        &self,
        query: Box3D,
        stop_level: usize,
    ) -> Result<Estimate, StreamError> {
        self.core
            .estimate_async(
                stop_level,
                |record| parse_box3d(record).overlaps(query),
                |record| query.contains(parse_box3d(record)),
                |record| box_fraction_3d(parse_box3d(record), query),
            )
            .await
    }

    /// Stream `(item index, payload blob)` for every item intersecting `query`.
    pub async fn search_payloads_async(
        &self,
        query: Box3D,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box3d(r).overlaps(query),
                Want::Payloads,
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose box overlaps the region `query` —
    /// any [`Overlaps3D`] shape, not just a box.
    pub async fn visit_region_async<Q, F>(
        &self,
        query: &Q,
        mut visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box3d(r)),
                Want::Ids,
                |id, _| visitor(id),
            )
            .await
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub async fn search_region_async<Q: Overlaps3D>(
        &self,
        query: &Q,
    ) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region_async(query, |index| out.push(index))
            .await?;
        Ok(out)
    }

    /// Return the number of items whose box overlaps the region `query`.
    pub async fn count_region_async<Q: Overlaps3D>(&self, query: &Q) -> Result<usize, StreamError> {
        let mut count = 0usize;
        self.visit_region_async(query, |_| count += 1).await?;
        Ok(count)
    }

    /// Visit `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub async fn visit_payloads_region_async<Q, F>(
        &self,
        query: &Q,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize, &[u8]),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box3d(r)),
                Want::Payloads,
                visitor,
            )
            .await
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub async fn search_payloads_region_async<Q: Overlaps3D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region_async(query, |id, blob| out.push((id, blob.to_vec())))
            .await?;
        Ok(out)
    }

    /// Async counterpart of [`StreamIndex3D::search_payload_prefixes_each`].
    pub async fn visit_payload_prefixes_async<F: FnMut(PayloadPrefix<'_>)>(
        &self,
        query: Box3D,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payload_prefixes_async(
                |record| parse_box3d(record).overlaps(query),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex3D::search_payload_prefixes_region_each`].
    pub async fn visit_payload_prefixes_region_async<Q, F>(
        &self,
        query: &Q,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(PayloadPrefix<'_>),
    {
        self.core
            .visit_payload_prefixes_async(
                |record| query.overlaps_box(parse_box3d(record)),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex3D::payloads_at_ranks_each`].
    pub async fn visit_payloads_at_ranks_async<F: FnMut(usize, &[u8])>(
        &self,
        leaf_ranks: &[usize],
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads_at_ranks_async(leaf_ranks, visitor)
            .await
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload_async(&self) -> bool {
        self.core.has_payload()
    }
}

/// Async streaming reader for a compact `f32` 2D index. Mirrors
/// [`StreamIndex2DF32`]'s sync methods over async I/O. Behind the `async` feature.
#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamIndex2DF32<R> {
    /// Open and validate a 2D `f32` index from an async `reader`.
    pub async fn open_async(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits_async(reader, StreamLimits::default()).await
    }

    /// Open from an async `reader` with per-query [`StreamLimits`].
    pub async fn open_with_limits_async(
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open_async(reader, 2, 4, limits).await?,
        })
    }

    /// Stream the indices of every item whose (rounded) box intersects `query`.
    pub async fn search_async(&self, query: Box2D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box2d_f32(r).overlaps(query),
                Want::Ids,
                |id, _| out.push(id),
            )
            .await?;
        Ok(out)
    }

    /// Return the number of items overlapping `query`, counted without
    /// collecting them.
    pub async fn count_async(&self, query: Box2D) -> Result<usize, StreamError> {
        self.count_region_async(&query).await
    }

    /// Async mirror of [`estimate_count`](Self::estimate_count): bracket and
    /// estimate how many items `query` would hit, from node boxes, stopping at
    /// `stop_level`. With `stop_level >= directory_floor()` nothing is read, so
    /// a worker can decide whether a query is worth its round trips first.
    pub async fn estimate_count_async(
        &self,
        query: Box2D,
        stop_level: usize,
    ) -> Result<Estimate, StreamError> {
        self.core
            .estimate_async(
                stop_level,
                |record| parse_box2d_f32(record).overlaps(query),
                |record| query.contains(parse_box2d_f32(record)),
                |record| box_fraction_2d(parse_box2d_f32(record), query),
            )
            .await
    }

    /// Stream `(item index, payload blob)` for every item intersecting `query`.
    pub async fn search_payloads_async(
        &self,
        query: Box2D,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box2d_f32(r).overlaps(query),
                Want::Payloads,
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose (rounded) box overlaps the region
    /// `query` — any [`Overlaps2D`] shape.
    pub async fn visit_region_async<Q, F>(
        &self,
        query: &Q,
        mut visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box2d_f32(r)),
                Want::Ids,
                |id, _| visitor(id),
            )
            .await
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub async fn search_region_async<Q: Overlaps2D>(
        &self,
        query: &Q,
    ) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region_async(query, |index| out.push(index))
            .await?;
        Ok(out)
    }

    /// Return the number of items whose box overlaps the region `query`.
    pub async fn count_region_async<Q: Overlaps2D>(&self, query: &Q) -> Result<usize, StreamError> {
        let mut count = 0usize;
        self.visit_region_async(query, |_| count += 1).await?;
        Ok(count)
    }

    /// Visit `(item index, payload blob)` for every item whose (rounded) box
    /// overlaps the region `query`.
    pub async fn visit_payloads_region_async<Q, F>(
        &self,
        query: &Q,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize, &[u8]),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box2d_f32(r)),
                Want::Payloads,
                visitor,
            )
            .await
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub async fn search_payloads_region_async<Q: Overlaps2D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region_async(query, |id, blob| out.push((id, blob.to_vec())))
            .await?;
        Ok(out)
    }

    /// Async counterpart of [`StreamIndex2DF32::search_payload_prefixes_each`].
    pub async fn visit_payload_prefixes_async<F: FnMut(PayloadPrefix<'_>)>(
        &self,
        query: Box2D,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payload_prefixes_async(
                |record| parse_box2d_f32(record).overlaps(query),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex2DF32::search_payload_prefixes_region_each`].
    pub async fn visit_payload_prefixes_region_async<Q, F>(
        &self,
        query: &Q,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(PayloadPrefix<'_>),
    {
        self.core
            .visit_payload_prefixes_async(
                |record| query.overlaps_box(parse_box2d_f32(record)),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex2DF32::payloads_at_ranks_each`].
    pub async fn visit_payloads_at_ranks_async<F: FnMut(usize, &[u8])>(
        &self,
        leaf_ranks: &[usize],
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads_at_ranks_async(leaf_ranks, visitor)
            .await
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload_async(&self) -> bool {
        self.core.has_payload()
    }
}

/// Async streaming reader for a compact `f32` 3D index. See
/// [`StreamIndex2DF32`]'s async methods. Behind the `async` feature.
#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamIndex3DF32<R> {
    /// Open and validate a 3D `f32` index from an async `reader`.
    pub async fn open_async(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits_async(reader, StreamLimits::default()).await
    }

    /// Open from an async `reader` with per-query [`StreamLimits`].
    pub async fn open_with_limits_async(
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open_async(reader, 3, 4, limits).await?,
        })
    }

    /// Stream the indices of every item whose (rounded) box intersects `query`.
    pub async fn search_async(&self, query: Box3D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box3d_f32(r).overlaps(query),
                Want::Ids,
                |id, _| out.push(id),
            )
            .await?;
        Ok(out)
    }

    /// Return the number of items overlapping `query`, counted without
    /// collecting them.
    pub async fn count_async(&self, query: Box3D) -> Result<usize, StreamError> {
        self.count_region_async(&query).await
    }

    /// Async mirror of [`estimate_count`](Self::estimate_count): bracket and
    /// estimate how many items `query` would hit, from node boxes, stopping at
    /// `stop_level`. With `stop_level >= directory_floor()` nothing is read, so
    /// a worker can decide whether a query is worth its round trips first.
    pub async fn estimate_count_async(
        &self,
        query: Box3D,
        stop_level: usize,
    ) -> Result<Estimate, StreamError> {
        self.core
            .estimate_async(
                stop_level,
                |record| parse_box3d_f32(record).overlaps(query),
                |record| query.contains(parse_box3d_f32(record)),
                |record| box_fraction_3d(parse_box3d_f32(record), query),
            )
            .await
    }

    /// Stream `(item index, payload blob)` for every item intersecting `query`.
    pub async fn search_payloads_async(
        &self,
        query: Box3D,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box3d_f32(r).overlaps(query),
                Want::Payloads,
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose (rounded) box overlaps the region
    /// `query` — any [`Overlaps3D`] shape.
    pub async fn visit_region_async<Q, F>(
        &self,
        query: &Q,
        mut visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box3d_f32(r)),
                Want::Ids,
                |id, _| visitor(id),
            )
            .await
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub async fn search_region_async<Q: Overlaps3D>(
        &self,
        query: &Q,
    ) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region_async(query, |index| out.push(index))
            .await?;
        Ok(out)
    }

    /// Return the number of items whose box overlaps the region `query`.
    pub async fn count_region_async<Q: Overlaps3D>(&self, query: &Q) -> Result<usize, StreamError> {
        let mut count = 0usize;
        self.visit_region_async(query, |_| count += 1).await?;
        Ok(count)
    }

    /// Visit `(item index, payload blob)` for every item whose (rounded) box
    /// overlaps the region `query`.
    pub async fn visit_payloads_region_async<Q, F>(
        &self,
        query: &Q,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize, &[u8]),
    {
        self.core
            .traverse_async(
                |r| query.overlaps_box(parse_box3d_f32(r)),
                Want::Payloads,
                visitor,
            )
            .await
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub async fn search_payloads_region_async<Q: Overlaps3D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region_async(query, |id, blob| out.push((id, blob.to_vec())))
            .await?;
        Ok(out)
    }

    /// Async counterpart of [`StreamIndex3DF32::search_payload_prefixes_each`].
    pub async fn visit_payload_prefixes_async<F: FnMut(PayloadPrefix<'_>)>(
        &self,
        query: Box3D,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payload_prefixes_async(
                |record| parse_box3d_f32(record).overlaps(query),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex3DF32::search_payload_prefixes_region_each`].
    pub async fn visit_payload_prefixes_region_async<Q, F>(
        &self,
        query: &Q,
        prefix_len: usize,
        visitor: F,
    ) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(PayloadPrefix<'_>),
    {
        self.core
            .visit_payload_prefixes_async(
                |record| query.overlaps_box(parse_box3d_f32(record)),
                prefix_len,
                visitor,
            )
            .await
    }

    /// Async counterpart of [`StreamIndex3DF32::payloads_at_ranks_each`].
    pub async fn visit_payloads_at_ranks_async<F: FnMut(usize, &[u8])>(
        &self,
        leaf_ranks: &[usize],
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads_at_ranks_async(leaf_ranks, visitor)
            .await
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload_async(&self) -> bool {
        self.core.has_payload()
    }
}
