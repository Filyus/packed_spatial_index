use std::io;
use std::sync::Arc;

use crate::geometry::{Box2D, Box3D, Overlaps2D, Overlaps3D};
use crate::persistence::{
    CHUNK_ENTRY_LEN, CHUNK_FLAG_CRITICAL, FORMAT_VERSION, LoadError, PYLD_DESC_LEN,
    PYLD_DESC_LEN_FIXED, SUPERBLOCK_LEN, TAG_PYLD, TAG_TREE, TREE_DESC_LEN, derive_level_bounds,
    expected_tree_shape, parse_pyld_chunk, parse_tree_chunk, read_u32_at, read_u64_at,
};

use super::core::{align8_u64, checked_directory_span};
use super::directory::directory_start;
use super::limits::{Budget, directory_node_budget};
use super::payload::{
    PayloadSection, emit_run_payloads, emit_run_payloads_fixed, payload_blob_span, payload_run_end,
    payload_run_end_fixed,
};
use super::planner::{apply_gather_run, expand_frontier, plan_gather};
use super::{
    StreamCore, StreamError, StreamIndex2D, StreamIndex2DF32, StreamIndex3D, StreamIndex3DF32,
    StreamLimits, parse_box2d, parse_box2d_f32, parse_box3d, parse_box3d_f32, read_index,
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
    async fn open_async(
        reader: R,
        dimensions: usize,
        coord_bytes: usize,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        let mut head = [0u8; SUPERBLOCK_LEN];
        reader.read_exact_at(0, &mut head).await?;
        if &head[..8] != b"PSINDEX\0" {
            return Err(StreamError::Format(LoadError::BadMagic));
        }
        if u64::from_le_bytes(head[8..16].try_into().unwrap()) != FORMAT_VERSION {
            return Err(StreamError::Format(LoadError::UnsupportedVersion));
        }
        let chunk_count = read_u32_at(&head, 16)? as usize;
        let file_len = reader.len();
        let (dir_len, dir_end) = checked_directory_span(chunk_count, file_len)?;
        let mut dir = vec![0u8; dir_len];
        reader
            .read_exact_at(SUPERBLOCK_LEN as u64, &mut dir)
            .await?;

        let mut max_end = dir_end;
        let mut tree: Option<(u64, u64)> = None;
        let mut pyld: Option<(u64, u64)> = None;
        for i in 0..chunk_count {
            let base = i * CHUNK_ENTRY_LEN;
            let mut tag = [0u8; 4];
            tag.copy_from_slice(&dir[base..base + 4]);
            let flags = read_u32_at(&dir, base + 4)?;
            let offset = read_u64_at(&dir, base + 8)?;
            let len = read_u64_at(&dir, base + 16)?;
            let end = offset.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
            if file_len.is_some_and(|fl| end > fl) {
                return Err(StreamError::Format(LoadError::InvalidTree));
            }
            max_end = max_end.max(end);
            if tag == TAG_TREE {
                tree = Some((offset, len));
            } else if tag == TAG_PYLD {
                pyld = Some((offset, len));
            } else if flags & CHUNK_FLAG_CRITICAL != 0 {
                return Err(StreamError::Format(LoadError::UnsupportedVersion));
            }
        }

        // Reject a file longer than the last chunk plus its alignment pad — a
        // stray trailing byte the directory does not account for.
        let aligned_end = align8_u64(max_end)?;
        if let Some(fl) = file_len
            && fl > aligned_end
        {
            return Err(StreamError::Format(LoadError::LengthMismatch {
                expected: max_end as usize,
                actual: fl as usize,
            }));
        }
        let (toff, tlen) = tree.ok_or(LoadError::InvalidTree)?;
        if tlen < TREE_DESC_LEN as u64 {
            return Err(StreamError::Format(LoadError::Truncated));
        }
        let mut desc = [0u8; TREE_DESC_LEN];
        reader.read_exact_at(toff, &mut desc).await?;
        let (td, _) = parse_tree_chunk(&desc)?;
        if td.dimensions != dimensions || td.coord_bytes != coord_bytes {
            return Err(StreamError::Format(LoadError::UnsupportedVersion));
        }
        let (num_nodes, level_count) = expected_tree_shape(td.num_items, td.node_size)?;
        let record = dimensions
            .checked_mul(2 * coord_bytes)
            .ok_or(LoadError::IntegerOverflow)?;
        let box_stride = if td.interleaved { record + 8 } else { record };
        let box0 = toff + td.desc_len as u64;
        let node_len = num_nodes
            .checked_mul(box_stride + if td.interleaved { 0 } else { 8 })
            .ok_or(LoadError::IntegerOverflow)?;
        if tlen != td.desc_len as u64 + node_len as u64 {
            return Err(StreamError::Format(LoadError::InvalidTree));
        }
        let idx0 = if td.interleaved {
            box0
        } else {
            box0 + (num_nodes * record) as u64
        };
        let level_bounds = derive_level_bounds(td.num_items, td.node_size, level_count);

        let payload = match pyld {
            Some((poff, plen)) => {
                if plen < PYLD_DESC_LEN as u64 {
                    return Err(StreamError::Format(LoadError::Truncated));
                }
                let dn = (PYLD_DESC_LEN_FIXED as u64).min(plen) as usize;
                let mut pd = [0u8; PYLD_DESC_LEN_FIXED];
                reader.read_exact_at(poff, &mut pd[..dn]).await?;
                let (pdesc, _) = parse_pyld_chunk(&pd[..dn])?;
                let body0 = poff + pdesc.desc_len as u64;
                if pdesc.record_stride != 0 {
                    let stride = pdesc.record_stride as u64;
                    let blob_total = (td.num_items as u64)
                        .checked_mul(stride)
                        .ok_or(StreamError::Format(LoadError::IntegerOverflow))?;
                    let need = pdesc.desc_len as u64 + blob_total;
                    if plen != need {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    Some(PayloadSection {
                        offsets_start: 0,
                        blobs_start: body0,
                        blob_total,
                        stride,
                    })
                } else {
                    let offsets_start = body0;
                    let last_at = offsets_start + (td.num_items as u64) * 8;
                    let mut last = [0u8; 8];
                    reader.read_exact_at(last_at, &mut last).await?;
                    let blob_total = u64::from_le_bytes(last);
                    let blobs_start = offsets_start + (td.num_items as u64 + 1) * 8;
                    let need = pdesc.desc_len as u64 + (td.num_items as u64 + 1) * 8 + blob_total;
                    if plen != need {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    Some(PayloadSection {
                        offsets_start,
                        blobs_start,
                        blob_total,
                        stride: 0,
                    })
                }
            }
            None => None,
        };

        // Directory prefetch (mirror of the sync `open` epilogue).
        let budget = directory_node_budget(&limits, box_stride, td.interleaved);
        let dir_node_start = directory_start(&level_bounds, level_count, budget);
        let cached_nodes = num_nodes - dir_node_start;
        let mut dir_boxes = vec![0u8; cached_nodes * box_stride];
        if !dir_boxes.is_empty() {
            let offset = box0 + (dir_node_start * box_stride) as u64;
            reader.read_exact_at(offset, &mut dir_boxes).await?;
        }
        let mut dir_indices = if td.interleaved {
            Vec::new()
        } else {
            vec![0u8; cached_nodes * 8]
        };
        if !dir_indices.is_empty() {
            let offset = idx0 + (dir_node_start * 8) as u64;
            reader.read_exact_at(offset, &mut dir_indices).await?;
        }
        let dir_boxes: Arc<[u8]> = dir_boxes.into();
        let dir_indices: Arc<[u8]> = dir_indices.into();
        Ok(StreamCore {
            reader,
            node_size: td.node_size,
            num_items: td.num_items,
            num_nodes,
            level_count,
            level_bounds,
            record,
            box_stride,
            interleaved: td.interleaved,
            box0,
            idx0,
            dir_node_start,
            dir_boxes,
            dir_indices,
            payload,
            limits,
        })
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

    /// Whether this index was written with a payload section.
    pub fn has_payload_async(&self) -> bool {
        self.core.has_payload()
    }
}
