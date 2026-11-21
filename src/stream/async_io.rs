use std::io;

use crate::estimate::{Estimate, box_fraction_2d, box_fraction_3d};
use crate::geometry::{Box2D, Box3D, Overlaps2D, Overlaps3D};
use crate::persistence::SUPERBLOCK_LEN;

use super::core::{EstimateMachine, EstimateStep, TraverseMachine, TraverseStep, emit_leaf_ids};
use super::limits::{Budget, COALESCE_GAP_BYTES};
use super::open::{OpenBytes, OpenStep, head_len, plan_open};
use super::payload::{PayloadsMachine, PrefixesMachine, RankPayloadsMachine, WaveMachine};
use super::planner::{AlignedPlan, GatherBatch, apply_gather_run, plan_aligned};
use super::{
    PayloadPrefix, StreamCore, StreamError, StreamIndex2D, StreamIndex2DF32, StreamIndex3D,
    StreamIndex3DF32, StreamLimits, parse_box2d, parse_box2d_f32, parse_box3d, parse_box3d_f32,
};

// ---- Async streaming (behind the `async` feature) ----
//
// Thin async I/O driver over the shared sans-IO traversal, estimation, and
// payload state machines in `core.rs`, `open.rs`, and `payload.rs`. Each wave
// of independent byte ranges is issued concurrently (`read_batch_async` via
// `try_join_all`), so per-level round trips match the wave depth while
// predicates and callbacks stay synchronous.

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

    /// Fetch a batch of independent `(offset, len)` reads concurrently,
    /// charging `budget` and return one buffer per read (empty reads cost
    /// nothing). With [`StreamLimits::align_bytes`] set, each read widens to
    /// whole blocks and reads sharing a block share one fetch.
    async fn read_batch_async(
        &self,
        reads: &[(u64, usize)],
        budget: &mut Budget,
    ) -> Result<Vec<Vec<u8>>, StreamError> {
        let Some(block) = self.align_block() else {
            let mut bufs: Vec<Vec<u8>> = reads.iter().map(|&(_, len)| vec![0u8; len]).collect();
            for buf in bufs.iter().filter(|buf| !buf.is_empty()) {
                budget.charge_read(buf.len())?;
            }
            let fetches = reads
                .iter()
                .zip(bufs.iter_mut())
                .filter(|(_, buf)| !buf.is_empty())
                .map(|(&(offset, _), buf)| self.reader.read_exact_at(offset, buf.as_mut_slice()));
            futures_util::future::try_join_all(fetches).await?;
            return Ok(bufs);
        };
        let AlignedPlan { fetches, place } = plan_aligned(reads, block, self.data_end);
        let mut fetched: Vec<Vec<u8>> = fetches.iter().map(|&(_, len)| vec![0u8; len]).collect();
        for buf in &fetched {
            budget.charge_read(buf.len())?;
        }
        let pending = fetches
            .iter()
            .zip(fetched.iter_mut())
            .map(|(&(offset, _), buf)| self.reader.read_exact_at(offset, buf.as_mut_slice()));
        futures_util::future::try_join_all(pending).await?;
        Ok(reads
            .iter()
            .zip(place)
            .map(|(&(_, len), (at, within))| {
                if len == 0 {
                    Vec::new()
                } else {
                    fetched[at][within..within + len].to_vec()
                }
            })
            .collect())
    }

    /// Execute one planned gather wave concurrently (`try_join_all`).
    async fn run_gather_async(
        &self,
        batch: GatherBatch<'_>,
        budget: &mut Budget,
    ) -> Result<(), StreamError> {
        let reads: Vec<(u64, usize)> = batch.runs.iter().map(|run| (run.offset, run.len)).collect();
        let bufs = self.read_batch_async(&reads, budget).await?;
        for (run, buf) in batch.runs.iter().zip(&bufs) {
            apply_gather_run(batch.out, run, buf, batch.stride);
        }
        Ok(())
    }

    /// Execute a sans-IO [`WaveMachine`], issuing each wave's reads concurrently.
    async fn run_waves_async<M: WaveMachine<F>, F>(
        &self,
        mut machine: M,
        budget: &mut Budget,
        mut emit: F,
    ) -> Result<(), StreamError> {
        let mut reads = Vec::new();
        while machine.next_wave(&mut reads, budget, &mut emit)? {
            if reads.iter().all(|&(_, len)| len == 0) {
                for i in 0..reads.len() {
                    machine.feed_read(i, &[], budget, &mut emit)?;
                }
            } else {
                let bufs = self.read_batch_async(&reads, budget).await?;
                for (i, buf) in bufs.iter().enumerate() {
                    machine.feed_read(i, buf, budget, &mut emit)?;
                }
            }
        }
        Ok(())
    }

    /// Descend the tree level by level via [`TraverseMachine`], returning the
    /// surviving leaf positions and gathered index bytes, or `None` if empty.
    async fn traverse_leaves_async<O>(
        &self,
        overlaps: O,
        budget: &mut Budget,
    ) -> Result<Option<(Vec<usize>, Vec<u8>)>, StreamError>
    where
        O: Fn(&[u8]) -> bool,
    {
        let mut machine = TraverseMachine::new(self, overlaps);
        let found = loop {
            match machine.step()? {
                TraverseStep::Gather(batch) => self.run_gather_async(batch, budget).await?,
                TraverseStep::Leaves => break true,
                TraverseStep::Empty => break false,
            }
        };
        if !found {
            return Ok(None);
        }
        Ok(Some(machine.into_leaves()))
    }

    /// Async counterpart of [`visit_ids`](StreamCore::visit_ids).
    pub(crate) async fn visit_ids_async<O, F>(
        &self,
        overlaps: O,
        emit: F,
    ) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize),
    {
        let mut budget = Budget::new(self.limits);
        let Some((survivors, indices)) = self.traverse_leaves_async(overlaps, &mut budget).await?
        else {
            return Ok(());
        };
        emit_leaf_ids(&survivors, &indices, self.num_items, &mut budget, emit)
    }

    /// Async counterpart of [`search_payloads_each`](StreamCore::search_payloads_each).
    pub(crate) async fn search_payloads_each_async<O, F>(
        &self,
        overlaps: O,
        emit: F,
    ) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize, &[u8]),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        let mut budget = Budget::new(self.limits);
        let Some((survivors, indices)) = self.traverse_leaves_async(overlaps, &mut budget).await?
        else {
            return Ok(());
        };
        let machine = PayloadsMachine::new(
            section,
            &survivors,
            &indices,
            self.num_items,
            self.coalesce_gap(),
        );
        self.run_waves_async(machine, &mut budget, emit).await
    }

    async fn visit_payload_prefixes_async<O, F>(
        &self,
        overlaps: O,
        prefix_len: usize,
        emit: F,
    ) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(PayloadPrefix<'_>),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        let mut budget = Budget::new(self.limits);
        let Some((survivors, indices)) = self.traverse_leaves_async(overlaps, &mut budget).await?
        else {
            return Ok(());
        };
        let machine = PrefixesMachine::new(
            section,
            self.prefix.as_ref(),
            &survivors,
            &indices,
            prefix_len,
            self.num_items,
            self.coalesce_gap(),
            self.prefix_coalesce_gap(prefix_len),
        );
        self.run_waves_async(machine, &mut budget, emit).await
    }

    async fn visit_payloads_at_ranks_async<F>(
        &self,
        leaf_ranks: &[usize],
        emit: F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        let machine =
            RankPayloadsMachine::new(section, leaf_ranks, self.num_items, self.coalesce_gap())?;
        let mut budget = Budget::new(self.limits);
        self.run_waves_async(machine, &mut budget, emit).await
    }

    /// Async counterpart of [`estimate`](StreamCore::estimate): drives the same
    /// [`EstimateMachine`] with concurrent per-level gathers below the
    /// directory floor.
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
        let mut budget = Budget::new(self.limits);
        let mut machine = EstimateMachine::new(self, stop_level, overlaps, contains, fraction);
        loop {
            match machine.step()? {
                EstimateStep::Gather(batch) => self.run_gather_async(batch, &mut budget).await?,
                EstimateStep::Done(out) => return Ok(out),
            }
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
            .visit_ids_async(|r| parse_box2d(r).overlaps(query), |id| out.push(id))
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
            .search_payloads_each_async(
                |r| parse_box2d(r).overlaps(query),
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose box overlaps the region `query` —
    /// any [`Overlaps2D`] shape, not just a box.
    pub async fn visit_region_async<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize),
    {
        self.core
            .visit_ids_async(|r| query.overlaps_box(parse_box2d(r)), visitor)
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
            .search_payloads_each_async(|r| query.overlaps_box(parse_box2d(r)), visitor)
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
            .visit_ids_async(|r| parse_box3d(r).overlaps(query), |id| out.push(id))
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
            .search_payloads_each_async(
                |r| parse_box3d(r).overlaps(query),
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose box overlaps the region `query` —
    /// any [`Overlaps3D`] shape, not just a box.
    pub async fn visit_region_async<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize),
    {
        self.core
            .visit_ids_async(|r| query.overlaps_box(parse_box3d(r)), visitor)
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
            .search_payloads_each_async(|r| query.overlaps_box(parse_box3d(r)), visitor)
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
            .visit_ids_async(|r| parse_box2d_f32(r).overlaps(query), |id| out.push(id))
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
            .search_payloads_each_async(
                |r| parse_box2d_f32(r).overlaps(query),
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose (rounded) box overlaps the region
    /// `query` — any [`Overlaps2D`] shape.
    pub async fn visit_region_async<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize),
    {
        self.core
            .visit_ids_async(|r| query.overlaps_box(parse_box2d_f32(r)), visitor)
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
            .search_payloads_each_async(|r| query.overlaps_box(parse_box2d_f32(r)), visitor)
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
            .visit_ids_async(|r| parse_box3d_f32(r).overlaps(query), |id| out.push(id))
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
            .search_payloads_each_async(
                |r| parse_box3d_f32(r).overlaps(query),
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Stream the indices of every item whose (rounded) box overlaps the region
    /// `query` — any [`Overlaps3D`] shape.
    pub async fn visit_region_async<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize),
    {
        self.core
            .visit_ids_async(|r| query.overlaps_box(parse_box3d_f32(r)), visitor)
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
            .search_payloads_each_async(|r| query.overlaps_box(parse_box3d_f32(r)), visitor)
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
