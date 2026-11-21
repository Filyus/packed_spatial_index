use std::io;
use std::sync::Arc;

use crate::estimate::{Estimate, subtree_leaf_range};
use crate::persistence::{CHUNK_ENTRY_LEN, LoadError, SUPERBLOCK_LEN, read_u64_le_unchecked};

use super::StreamError;
use super::directory::StreamCoreParts;
use super::limits::{Budget, COALESCE_GAP_BYTES, StreamLimits};
use super::open::{OpenBytes, OpenLayout, OpenStep, head_len, plan_open};
use super::payload::{
    PayloadSection, PayloadsMachine, PrefixSection, PrefixesMachine, RankPayloadsMachine,
    WaveMachine,
};
use super::planner::{
    GatherBatch, GatherRun, align_span, apply_gather_run, expand_frontier, plan_gather,
};
use super::readers::RangeReader;

const MAX_CONTAINER_CHUNKS_WITHOUT_LEN: usize = 1024;

/// Dimension-independent streaming state: validated header counts, section
/// offsets, the parsed level bounds, and the cached upper-level directory.
///
/// Both the 2D and (future) 3D streaming indexes wrap one of these; only box
/// parsing and query traversal differ between dimensions.
pub(crate) struct StreamCore<R> {
    pub(crate) reader: R,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    /// Exclusive end offset of each level, in node positions (`level_bounds[i]`).
    pub(crate) level_bounds: Vec<usize>,
    /// Box record size in bytes.
    pub(crate) record: usize,
    /// Byte stride from one node's box to the next: `record` for the SoA layout,
    /// `record + 8` for the interleaved layout (box immediately followed by its
    /// index). The box of node `n` is always its first `record` bytes.
    pub(crate) box_stride: usize,
    /// Whether the node section is interleaved (box + index per node). When set,
    /// a node's index is read from its own record, so `traverse` issues no second
    /// index gather. That gather is *dependent* — its positions are the survivors
    /// of the box tests — so dropping it halves the descent's round-trip depth,
    /// not its read count.
    pub(crate) interleaved: bool,
    /// Byte offset of the box / node section.
    pub(crate) box0: u64,
    /// Byte offset of the separate index section (SoA layout; unused interleaved).
    pub(crate) idx0: u64,
    /// First node position covered by the cached directory.
    pub(crate) dir_node_start: usize,
    /// End of the last chunk: block-aligned query reads are clamped to it.
    pub(crate) data_end: u64,
    /// Cached box (or node, when interleaved) bytes for positions
    /// `[dir_node_start, num_nodes)`, strided by `box_stride`. `Arc` so a
    /// directory split off with `into_parts` reattaches by a refcount bump, not
    /// a copy (cheap reuse across queries; no growth of sticky wasm memory).
    pub(crate) dir_boxes: Arc<[u8]>,
    /// Cached index bytes for the same positions (SoA layout only; empty when
    /// interleaved, where indices live inside `dir_boxes`).
    pub(crate) dir_indices: Arc<[u8]>,
    /// Optional payload section. `None` when the index carries no payload.
    pub(crate) payload: Option<PayloadSection>,
    /// Optional payload-prefix section: a dense copy of the blob heads, so a
    /// prefix scan reads runs instead of one range per match.
    pub(crate) prefix: Option<PrefixSection>,
    /// Per-query cost limits applied to every query (default: unbounded).
    pub(crate) limits: StreamLimits,
}

impl<R> StreamCore<R> {
    /// Whether the index carries a payload section. No I/O, so available for
    /// both sync and async readers.
    pub(crate) fn has_payload(&self) -> bool {
        self.payload.is_some()
    }

    /// The lowest level whose nodes the cached directory holds entirely.
    /// An estimate stopping at this level or above reads nothing. No I/O, so
    /// available for both sync and async readers.
    pub(crate) fn directory_floor(&self) -> usize {
        if self.num_nodes == 0 {
            return 0;
        }
        crate::traversal::upper_bound_level(&self.level_bounds, self.dir_node_start)
    }

    /// Byte gap below which records coalesce into one read (the caller's
    /// [`StreamLimits::coalesce_gap_bytes`] or the built-in default).
    pub(crate) fn coalesce_gap(&self) -> u64 {
        self.limits.coalesce_gap_bytes.unwrap_or(COALESCE_GAP_BYTES)
    }

    /// Block size for aligned query reads, or `None` to read exact ranges.
    pub(crate) fn align_block(&self) -> Option<u64> {
        self.limits.align_bytes.filter(|&b| b > 0)
    }

    /// Byte gap below which payload *prefixes* coalesce into one read.
    ///
    /// Defaults to `prefix_len`: never skip more than one prefix worth of
    /// bytes, so asking for 24 bytes cannot drag a whole body along. That makes
    /// the scan strided once bodies exceed `2 * prefix_len`, which is the right
    /// trade on a local file and the wrong one over a network, hence the
    /// override. The ordinary record gap stays the ceiling either way.
    pub(crate) fn prefix_coalesce_gap(&self, prefix_len: usize) -> u64 {
        let requested = self
            .limits
            .prefix_coalesce_gap_bytes
            .unwrap_or(prefix_len as u64);
        self.coalesce_gap().min(requested)
    }

    /// Split off the reader, keeping the reusable directory. No I/O.
    pub(crate) fn into_parts(self) -> (StreamCoreParts, R) {
        let parts = StreamCoreParts {
            node_size: self.node_size,
            num_items: self.num_items,
            num_nodes: self.num_nodes,
            level_count: self.level_count,
            level_bounds: self.level_bounds,
            record: self.record,
            box_stride: self.box_stride,
            interleaved: self.interleaved,
            box0: self.box0,
            idx0: self.idx0,
            dir_node_start: self.dir_node_start,
            data_end: self.data_end,
            dir_boxes: self.dir_boxes,
            dir_indices: self.dir_indices,
            payload: self.payload,
            prefix: self.prefix,
        };
        (parts, self.reader)
    }

    /// Assemble an opened core from a validated [`OpenLayout`]. No I/O.
    pub(super) fn from_layout(layout: OpenLayout, reader: R, limits: StreamLimits) -> Self {
        StreamCore {
            reader,
            node_size: layout.node_size,
            num_items: layout.num_items,
            num_nodes: layout.num_nodes,
            level_count: layout.level_count,
            level_bounds: layout.level_bounds,
            record: layout.record,
            box_stride: layout.box_stride,
            interleaved: layout.interleaved,
            box0: layout.box0,
            idx0: layout.idx0,
            dir_node_start: layout.dir_node_start,
            data_end: layout.data_end,
            dir_boxes: layout.dir_boxes.into(),
            dir_indices: layout.dir_indices.into(),
            payload: layout.payload,
            prefix: layout.prefix,
            limits,
        }
    }

    /// Reattach a reader to a previously split directory. No I/O.
    pub(crate) fn from_parts(parts: StreamCoreParts, reader: R, limits: StreamLimits) -> Self {
        StreamCore {
            reader,
            node_size: parts.node_size,
            num_items: parts.num_items,
            num_nodes: parts.num_nodes,
            level_count: parts.level_count,
            level_bounds: parts.level_bounds,
            record: parts.record,
            box_stride: parts.box_stride,
            interleaved: parts.interleaved,
            box0: parts.box0,
            idx0: parts.idx0,
            dir_node_start: parts.dir_node_start,
            data_end: parts.data_end,
            dir_boxes: parts.dir_boxes,
            dir_indices: parts.dir_indices,
            payload: parts.payload,
            prefix: parts.prefix,
            limits,
        }
    }
}

pub(super) fn checked_directory_span(
    chunk_count: usize,
    file_len: Option<u64>,
) -> Result<(usize, u64), LoadError> {
    if file_len.is_none() && chunk_count > MAX_CONTAINER_CHUNKS_WITHOUT_LEN {
        return Err(LoadError::InvalidTree);
    }
    let dir_len = chunk_count
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or(LoadError::IntegerOverflow)?;
    let dir_end = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or(LoadError::IntegerOverflow)?;
    if let Some(file_len) = file_len
        && file_len < dir_end as u64
    {
        return Err(LoadError::Truncated);
    }
    Ok((dir_len, dir_end as u64))
}

pub(super) fn align8_u64(value: u64) -> Result<u64, LoadError> {
    value
        .checked_add(7)
        .map(|v| v & !7)
        .ok_or(LoadError::IntegerOverflow)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DescentPhase {
    NeedBoxes,
    BoxesReady,
    IndicesReady,
    Done,
}

enum EstimateAction {
    Boxes(Vec<GatherRun>),
    Indices(Vec<GatherRun>),
    Done,
}

pub(super) enum EstimateStep<'a> {
    Gather(GatherBatch<'a>),
    Done(Estimate),
}

/// Sans-IO state machine for level-by-level selectivity estimation.
pub(super) struct EstimateMachine<'a, R, O, C, Fr> {
    core: &'a StreamCore<R>,
    stop_level: usize,
    overlaps: O,
    contains: C,
    fraction: Fr,
    out: Estimate,
    frontier: Vec<usize>,
    level: usize,
    boxes: Vec<u8>,
    indices: Vec<u8>,
    survivors: Vec<usize>,
    phase: DescentPhase,
}

impl<'a, R, O, C, Fr> EstimateMachine<'a, R, O, C, Fr>
where
    O: Fn(&[u8]) -> bool,
    C: Fn(&[u8]) -> bool,
    Fr: Fn(&[u8]) -> f64,
{
    pub(super) fn new(
        core: &'a StreamCore<R>,
        stop_level: usize,
        overlaps: O,
        contains: C,
        fraction: Fr,
    ) -> Self {
        let (frontier, level, phase) = if core.num_items == 0 {
            (Vec::new(), 0, DescentPhase::Done)
        } else {
            (
                vec![core.num_nodes - 1],
                core.level_count - 1,
                DescentPhase::NeedBoxes,
            )
        };
        Self {
            core,
            stop_level,
            overlaps,
            contains,
            fraction,
            out: Estimate {
                lower: 0,
                upper: 0,
                estimate: 0.0,
                nodes_tested: 0,
            },
            frontier,
            level,
            boxes: Vec::new(),
            indices: Vec::new(),
            survivors: Vec::new(),
            phase,
        }
    }

    #[inline]
    pub(super) fn step(&mut self) -> Result<EstimateStep<'_>, StreamError> {
        let action = if self.phase == DescentPhase::Done {
            self.out.estimate = self
                .out
                .estimate
                .clamp(self.out.lower as f64, self.out.upper as f64);
            EstimateAction::Done
        } else {
            loop {
                if self.phase == DescentPhase::NeedBoxes {
                    let runs = plan_gather(
                        &self.frontier,
                        self.core.box0,
                        self.core.box_stride,
                        self.core.dir_node_start,
                        &self.core.dir_boxes,
                        &mut self.boxes,
                        self.core.coalesce_gap(),
                    );
                    self.phase = DescentPhase::BoxesReady;
                    if !runs.is_empty() {
                        break EstimateAction::Boxes(runs);
                    }
                }
                if self.phase == DescentPhase::BoxesReady {
                    let level = self.level;
                    let stop_level = self.stop_level;
                    let level_start = if level == 0 {
                        0
                    } else {
                        self.core.level_bounds[level - 1]
                    };
                    let box_stride = self.core.box_stride;
                    let record_len = self.core.record;
                    let interleaved = self.core.interleaved;
                    let node_size = self.core.node_size;
                    let num_items = self.core.num_items;
                    let boxes = self.boxes.as_slice();
                    let survivors = &mut self.survivors;
                    let indices = &mut self.indices;
                    let overlaps = &self.overlaps;
                    let contains = &self.contains;
                    let fraction = &self.fraction;
                    let out = &mut self.out;
                    survivors.clear();
                    indices.clear();
                    for (i, &pos) in self.frontier.iter().enumerate() {
                        let slot = i * box_stride;
                        let record = &boxes[slot..slot + record_len];
                        out.nodes_tested += 1;
                        if !overlaps(record) {
                            continue;
                        }
                        let (start, end) =
                            subtree_leaf_range(pos, level, level_start, node_size, num_items);
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
                        if interleaved {
                            indices.extend_from_slice(
                                &boxes[slot + record_len..slot + record_len + 8],
                            );
                        }
                    }
                    if self.survivors.is_empty() {
                        self.phase = DescentPhase::Done;
                        self.out.estimate = self
                            .out
                            .estimate
                            .clamp(self.out.lower as f64, self.out.upper as f64);
                        break EstimateAction::Done;
                    }
                    self.phase = DescentPhase::IndicesReady;
                    if !self.core.interleaved {
                        let runs = plan_gather(
                            &self.survivors,
                            self.core.idx0,
                            8,
                            self.core.dir_node_start,
                            &self.core.dir_indices,
                            &mut self.indices,
                            self.core.coalesce_gap(),
                        );
                        if !runs.is_empty() {
                            break EstimateAction::Indices(runs);
                        }
                    }
                }
                self.frontier = expand_frontier(
                    &self.core.level_bounds,
                    self.core.node_size,
                    self.level,
                    self.survivors.len(),
                    &self.indices,
                )?;
                self.level -= 1;
                self.phase = DescentPhase::NeedBoxes;
            }
        };
        Ok(match action {
            EstimateAction::Boxes(runs) => EstimateStep::Gather(GatherBatch {
                runs,
                stride: self.core.box_stride,
                out: &mut self.boxes,
            }),
            EstimateAction::Indices(runs) => EstimateStep::Gather(GatherBatch {
                runs,
                stride: 8,
                out: &mut self.indices,
            }),
            EstimateAction::Done => EstimateStep::Done(self.out),
        })
    }
}

enum TraverseAction {
    Boxes(Vec<GatherRun>),
    Indices(Vec<GatherRun>),
    Leaves,
    Empty,
}

pub(super) enum TraverseStep<'a> {
    Gather(GatherBatch<'a>),
    Leaves,
    Empty,
}

/// Sans-IO state machine for level-by-level tree descent to matching leaves.
pub(super) struct TraverseMachine<'a, R, O> {
    core: &'a StreamCore<R>,
    overlaps: O,
    frontier: Vec<usize>,
    level: usize,
    boxes: Vec<u8>,
    indices: Vec<u8>,
    survivors: Vec<usize>,
    phase: DescentPhase,
}

impl<'a, R, O> TraverseMachine<'a, R, O>
where
    O: Fn(&[u8]) -> bool,
{
    pub(super) fn new(core: &'a StreamCore<R>, overlaps: O) -> Self {
        let (frontier, level, phase) = if core.num_items == 0 {
            (Vec::new(), 0, DescentPhase::Done)
        } else {
            (
                vec![core.num_nodes - 1],
                core.level_count - 1,
                DescentPhase::NeedBoxes,
            )
        };
        Self {
            core,
            overlaps,
            frontier,
            level,
            boxes: Vec::new(),
            indices: Vec::new(),
            survivors: Vec::new(),
            phase,
        }
    }

    pub(super) fn into_leaves(self) -> (Vec<usize>, Vec<u8>) {
        (self.survivors, self.indices)
    }

    #[inline]
    pub(super) fn step(&mut self) -> Result<TraverseStep<'_>, StreamError> {
        let action = if self.phase == DescentPhase::Done {
            TraverseAction::Empty
        } else {
            loop {
                if self.phase == DescentPhase::NeedBoxes {
                    let runs = plan_gather(
                        &self.frontier,
                        self.core.box0,
                        self.core.box_stride,
                        self.core.dir_node_start,
                        &self.core.dir_boxes,
                        &mut self.boxes,
                        self.core.coalesce_gap(),
                    );
                    self.phase = DescentPhase::BoxesReady;
                    if !runs.is_empty() {
                        break TraverseAction::Boxes(runs);
                    }
                }
                if self.phase == DescentPhase::BoxesReady {
                    let box_stride = self.core.box_stride;
                    let record_len = self.core.record;
                    let interleaved = self.core.interleaved;
                    let boxes = self.boxes.as_slice();
                    let survivors = &mut self.survivors;
                    let indices = &mut self.indices;
                    let overlaps = &self.overlaps;
                    survivors.clear();
                    indices.clear();
                    for (i, &pos) in self.frontier.iter().enumerate() {
                        let slot = i * box_stride;
                        if overlaps(&boxes[slot..slot + record_len]) {
                            survivors.push(pos);
                            // Interleaved: the index trails the box in the same record,
                            // so no second gather is needed.
                            if interleaved {
                                indices.extend_from_slice(
                                    &boxes[slot + record_len..slot + record_len + 8],
                                );
                            }
                        }
                    }
                    if self.survivors.is_empty() {
                        self.phase = DescentPhase::Done;
                        break TraverseAction::Empty;
                    }
                    self.phase = DescentPhase::IndicesReady;
                    if !self.core.interleaved {
                        let runs = plan_gather(
                            &self.survivors,
                            self.core.idx0,
                            8,
                            self.core.dir_node_start,
                            &self.core.dir_indices,
                            &mut self.indices,
                            self.core.coalesce_gap(),
                        );
                        if !runs.is_empty() {
                            break TraverseAction::Indices(runs);
                        }
                    }
                }
                if self.level == 0 {
                    self.phase = DescentPhase::Done;
                    break TraverseAction::Leaves;
                }
                self.frontier = expand_frontier(
                    &self.core.level_bounds,
                    self.core.node_size,
                    self.level,
                    self.survivors.len(),
                    &self.indices,
                )?;
                self.level -= 1;
                self.phase = DescentPhase::NeedBoxes;
            }
        };
        Ok(match action {
            TraverseAction::Boxes(runs) => TraverseStep::Gather(GatherBatch {
                runs,
                stride: self.core.box_stride,
                out: &mut self.boxes,
            }),
            TraverseAction::Indices(runs) => TraverseStep::Gather(GatherBatch {
                runs,
                stride: 8,
                out: &mut self.indices,
            }),
            TraverseAction::Leaves => TraverseStep::Leaves,
            TraverseAction::Empty => TraverseStep::Empty,
        })
    }
}

/// Validate gathered leaf insertion ids and emit each while charging `budget`.
pub(super) fn emit_leaf_ids<F: FnMut(usize)>(
    survivors: &[usize],
    indices: &[u8],
    num_items: usize,
    budget: &mut Budget,
    mut emit: F,
) -> Result<(), StreamError> {
    for i in 0..survivors.len() {
        let id = read_index(indices, i)?;
        if id >= num_items {
            return Err(StreamError::Format(LoadError::InvalidTree));
        }
        budget.charge_item()?;
        emit(id);
    }
    Ok(())
}

impl<R: RangeReader> StreamCore<R> {
    /// Open and validate a chunk-container index from `reader`: check the
    /// superblock, read the directory, locate the `TREE` (and optional `PYLD`)
    /// chunk, derive the tree shape, and prefetch the upper-level directory.
    ///
    /// The validation lives in [`plan_open`]; this driver fetches what it asks
    /// for. The first read is a speculative head (see
    /// [`StreamLimits::open_head_bytes`]) and each later batch is coalesced
    /// by the ordinary record gap.
    pub(crate) fn open(
        reader: R,
        dimensions: usize,
        coord_bytes: usize,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        let file_len = reader.len();
        let mut bytes = OpenBytes::default();
        let mut head = vec![0u8; head_len(&limits, file_len)];
        match reader.read_exact_at(0, &mut head) {
            Ok(()) => bytes.insert(0, head),
            // A source of unknown length shorter than the head: fall back to
            // the superblock and read the rest as the planner locates it.
            Err(e)
                if e.kind() == io::ErrorKind::UnexpectedEof
                    && file_len.is_none()
                    && head.len() > SUPERBLOCK_LEN =>
            {
                head.truncate(SUPERBLOCK_LEN);
                reader.read_exact_at(0, &mut head)?;
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
                false,
            )? {
                OpenStep::Read(reads) => {
                    for (offset, len) in reads {
                        let mut buf = vec![0u8; len];
                        reader.read_exact_at(offset, &mut buf)?;
                        bytes.insert(offset, buf);
                    }
                }
                OpenStep::Ready(layout) => return Ok(Self::from_layout(*layout, reader, limits)),
            }
        }
    }

    /// Cached box record bytes for node `position`, if the directory covers it.
    /// The box is the first `record` bytes of the node's `box_stride`-byte slot
    /// (interleaved nodes carry their index in the trailing 8 bytes).
    pub(crate) fn cached_box_bytes(&self, position: usize) -> Option<&[u8]> {
        if position < self.dir_node_start || position >= self.num_nodes {
            return None;
        }
        let start = (position - self.dir_node_start) * self.box_stride;
        self.dir_boxes.get(start..start + self.record)
    }

    /// Read `buf.len()` bytes at `offset` for a query, charging `budget`. With
    /// [`StreamLimits::align_bytes`] set, the read widens to whole blocks
    /// (clamped to the data end) and the part the query's previous aligned
    /// fetch already holds is served from it, so runs sharing a block cost one
    /// read. An empty read is a no-op.
    fn read_charged(
        &self,
        offset: u64,
        buf: &mut [u8],
        budget: &mut Budget,
    ) -> Result<(), StreamError> {
        if buf.is_empty() {
            return Ok(());
        }
        let Some(block) = self.align_block() else {
            budget.charge_read(buf.len())?;
            self.reader.read_exact_at(offset, buf)?;
            return Ok(());
        };
        let end = offset + buf.len() as u64;
        let (mut lo, hi) = align_span(offset, end, block, self.data_end);
        // Serve the head of the read from the previous aligned fetch.
        let mut done = 0usize;
        if let Some((start, bytes)) = &budget.last_block {
            let span_end = start + bytes.len() as u64;
            if *start <= offset && offset < span_end {
                done = (span_end.min(end) - offset) as usize;
                let within = (offset - start) as usize;
                buf[..done].copy_from_slice(&bytes[within..within + done]);
                lo = span_end;
            }
        }
        if done == buf.len() {
            return Ok(());
        }
        let mut fetched = vec![0u8; (hi - lo) as usize];
        budget.charge_read(fetched.len())?;
        self.reader.read_exact_at(lo, &mut fetched)?;
        let within = (offset + done as u64 - lo) as usize;
        let rest = buf.len() - done;
        buf[done..].copy_from_slice(&fetched[within..within + rest]);
        budget.last_block = Some((lo, fetched));
        Ok(())
    }

    /// Execute one planned gather wave sequentially, reusing `scratch`.
    #[inline]
    fn run_gather(
        &self,
        batch: GatherBatch<'_>,
        scratch: &mut Vec<u8>,
        budget: &mut Budget,
    ) -> Result<(), StreamError> {
        for run in &batch.runs {
            scratch.clear();
            scratch.resize(run.len, 0);
            self.read_charged(run.offset, scratch, budget)?;
            apply_gather_run(batch.out, run, scratch, batch.stride);
        }
        Ok(())
    }

    /// Execute a sans-IO [`WaveMachine`] sequentially, reusing `reads` and
    /// `scratch` across waves.
    fn run_waves<M: WaveMachine<F>, F>(
        &self,
        mut machine: M,
        budget: &mut Budget,
        mut emit: F,
    ) -> Result<(), StreamError> {
        let mut reads = Vec::new();
        let mut scratch = Vec::new();
        while machine.next_wave(&mut reads, budget, &mut emit)? {
            for (run_idx, &(offset, len)) in reads.iter().enumerate() {
                if len == 0 {
                    machine.feed_read(run_idx, &[], budget, &mut emit)?;
                } else {
                    scratch.clear();
                    scratch.resize(len, 0);
                    self.read_charged(offset, &mut scratch, budget)?;
                    machine.feed_read(run_idx, &scratch, budget, &mut emit)?;
                }
            }
        }
        Ok(())
    }

    /// Descend the tree level by level via [`TraverseMachine`], calling `leaf`
    /// once at the leaf level with the surviving leaf positions (sorted) and
    /// their gathered index bytes (the insertion ids, in the same order).
    fn traverse<O, L>(&self, overlaps: O, leaf: L) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        L: FnOnce(&[usize], &[u8], &mut Budget) -> Result<(), StreamError>,
    {
        let mut budget = Budget::new(self.limits);
        let mut scratch = Vec::new();
        let mut machine = TraverseMachine::new(self, overlaps);
        let found = loop {
            match machine.step()? {
                TraverseStep::Gather(batch) => {
                    self.run_gather(batch, &mut scratch, &mut budget)?;
                }
                TraverseStep::Leaves => break true,
                TraverseStep::Empty => break false,
            }
        };
        if !found {
            return Ok(());
        }
        let (survivors, indices) = machine.into_leaves();
        leaf(&survivors, &indices, &mut budget)
    }

    /// Bracket and estimate a window's hit count from node boxes, descending
    /// level by level via [`EstimateMachine`] and stopping at `stop_level`.
    /// Levels the directory caches cost no reads; the budget is charged only
    /// for levels fetched below it.
    pub(crate) fn estimate<O, C, Fr>(
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
        let mut scratch = Vec::new();
        let mut machine = EstimateMachine::new(self, stop_level, overlaps, contains, fraction);
        loop {
            match machine.step()? {
                EstimateStep::Gather(batch) => {
                    self.run_gather(batch, &mut scratch, &mut budget)?;
                }
                EstimateStep::Done(out) => return Ok(out),
            }
        }
    }

    /// Visit the insertion id of every leaf whose box satisfies `overlaps`.
    pub(crate) fn visit_ids<O, F>(&self, overlaps: O, visit: F) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize),
    {
        self.traverse(overlaps, |survivors, indices, budget| {
            emit_leaf_ids(survivors, indices, self.num_items, budget, visit)
        })
    }

    /// Visit `(insertion id, payload blob)` for every leaf whose box satisfies
    /// `overlaps`, streaming the payload section in leaf order during the leaf
    /// pass so the offset table and blobs are read in coalesced runs.
    pub(crate) fn search_payloads_each<O, F>(&self, overlaps: O, emit: F) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize, &[u8]),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        let coalesce_gap = self.coalesce_gap();
        self.traverse(overlaps, |survivors, indices, budget| {
            let machine =
                PayloadsMachine::new(section, survivors, indices, self.num_items, coalesce_gap);
            self.run_waves(machine, budget, emit)
        })
    }

    /// Visit a [`PayloadPrefix`] for every leaf whose box satisfies `overlaps`:
    /// the insertion id, its leaf rank, the payload's full byte length, and its
    /// first `prefix_len` bytes — without reading payload bodies past the
    /// prefix. Lengths come from the offset table (or the fixed stride), so a
    /// variable-width payload's body bytes beyond the prefix are never fetched.
    pub(crate) fn search_payload_prefixes_each<O, F>(
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
        let coalesce_gap = self.coalesce_gap();
        let prefix_coalesce_gap = self.prefix_coalesce_gap(prefix_len);
        self.traverse(overlaps, |survivors, indices, budget| {
            let machine = PrefixesMachine::new(
                section,
                self.prefix.as_ref(),
                survivors,
                indices,
                prefix_len,
                self.num_items,
                coalesce_gap,
                prefix_coalesce_gap,
            );
            self.run_waves(machine, budget, emit)
        })
    }

    /// Visit `(leaf rank, payload blob)` for an explicit set of leaf ranks —
    /// random-access payload reads for ranks captured earlier by
    /// [`search_payload_prefixes_each`](Self::search_payload_prefixes_each). Input ranks
    /// are sorted and deduplicated internally so the payload section is read
    /// in coalesced ascending runs; blobs are emitted in ascending rank order.
    /// A rank at or past the item count fails with [`StreamError::InvalidRank`].
    pub(crate) fn payloads_at_ranks_each<F>(
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
        self.run_waves(machine, &mut budget, emit)
    }
}

/// One matching leaf seen by payload-prefix visits: identity plus payload size,
/// with only the leading payload bytes fetched.
#[derive(Debug)]
pub struct PayloadPrefix<'a> {
    /// Item insertion id.
    pub id: usize,
    /// Leaf rank — the item's position in the leaf-ordered payload section.
    /// Stable for one serialized index, not across rebuilds. Feed ranks to
    /// [`StreamIndex2D::payloads_at_ranks_each`](super::StreamIndex2D::payloads_at_ranks_each)
    /// to fetch full payloads later.
    pub leaf_rank: usize,
    /// The first `min(prefix_len, payload_len)` payload bytes.
    pub prefix: &'a [u8],
    /// Full payload byte length; the body past `prefix` was not read.
    pub payload_len: usize,
}

/// Read index entry `i` (a little-endian `u64`) from gathered index bytes.
pub(crate) fn read_index(bytes: &[u8], i: usize) -> Result<usize, StreamError> {
    let value = read_u64_le_unchecked(bytes, i * 8);
    usize::try_from(value).map_err(|_| StreamError::Format(LoadError::IntegerOverflow))
}
