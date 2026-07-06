use std::sync::Arc;

use crate::persistence::{
    CHUNK_ENTRY_LEN, CHUNK_FLAG_CRITICAL, FORMAT_VERSION, LoadError, PYLD_DESC_LEN,
    PYLD_DESC_LEN_FIXED, SUPERBLOCK_LEN, TAG_PYLD, TAG_TREE, TREE_DESC_LEN, derive_level_bounds,
    expected_tree_shape, parse_pyld_chunk, parse_tree_chunk, read_u32_at, read_u64_at,
    read_u64_le_unchecked,
};

use super::StreamError;
use super::directory::{StreamCoreParts, directory_start};
use super::limits::{Budget, COALESCE_GAP_BYTES, StreamLimits, directory_node_budget};
use super::payload::{
    PayloadSection, emit_run_payloads, emit_run_payloads_fixed, payload_blob_span, payload_run_end,
    payload_run_end_fixed,
};
use super::planner::{apply_gather_run, expand_frontier, plan_gather};
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
    /// a node's index is read from its own record, so no separate index gather is
    /// issued — one coalesced read per level instead of two.
    pub(crate) interleaved: bool,
    /// Byte offset of the box / node section.
    pub(crate) box0: u64,
    /// Byte offset of the separate index section (SoA layout; unused interleaved).
    pub(crate) idx0: u64,
    /// First node position covered by the cached directory.
    pub(crate) dir_node_start: usize,
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
    /// Per-query cost limits applied to every query (default: unbounded).
    pub(crate) limits: StreamLimits,
}

impl<R> StreamCore<R> {
    /// Whether the index carries a payload section. No I/O, so available for
    /// both sync and async readers.
    pub(crate) fn has_payload(&self) -> bool {
        self.payload.is_some()
    }

    /// Byte gap below which records coalesce into one read (the caller's
    /// [`StreamLimits::coalesce_gap_bytes`] or the built-in default).
    pub(crate) fn coalesce_gap(&self) -> u64 {
        self.limits.coalesce_gap_bytes.unwrap_or(COALESCE_GAP_BYTES)
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
            dir_boxes: self.dir_boxes,
            dir_indices: self.dir_indices,
            payload: self.payload,
        };
        (parts, self.reader)
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
            dir_boxes: parts.dir_boxes,
            dir_indices: parts.dir_indices,
            payload: parts.payload,
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

impl<R: RangeReader> StreamCore<R> {
    /// Open and validate a chunk-container index from `reader`: check the
    /// superblock, read the directory, locate the `TREE` (and optional `PYLD`)
    /// chunk, derive the tree shape, and prefetch the upper-level directory.
    pub(crate) fn open(
        reader: R,
        dimensions: usize,
        coord_bytes: usize,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        // One leading read covers the superblock (magic + version + chunk_count).
        let mut head = [0u8; SUPERBLOCK_LEN];
        reader.read_exact_at(0, &mut head)?;
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
        reader.read_exact_at(SUPERBLOCK_LEN as u64, &mut dir)?;

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

        // TREE descriptor.
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
        reader.read_exact_at(toff, &mut desc)?;
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

        // Optional payload chunk.
        let payload = match pyld {
            Some((poff, plen)) => {
                if plen < PYLD_DESC_LEN as u64 {
                    return Err(StreamError::Format(LoadError::Truncated));
                }
                let dn = (PYLD_DESC_LEN_FIXED as u64).min(plen) as usize;
                let mut pd = [0u8; PYLD_DESC_LEN_FIXED];
                reader.read_exact_at(poff, &mut pd[..dn])?;
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
                    reader.read_exact_at(last_at, &mut last)?;
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

        // Directory: cache the upper levels (a contiguous suffix of the node
        // section) up to the byte budget.
        let budget = directory_node_budget(&limits, box_stride, td.interleaved);
        let dir_node_start = directory_start(&level_bounds, level_count, budget);
        let cached_nodes = num_nodes - dir_node_start;

        let mut dir_boxes = vec![0u8; cached_nodes * box_stride];
        if !dir_boxes.is_empty() {
            let offset = box0 + (dir_node_start * box_stride) as u64;
            reader.read_exact_at(offset, &mut dir_boxes)?;
        }
        // The interleaved layout carries indices inside the node records, so the
        // separate index cache is read only for the SoA layout.
        let mut dir_indices = if td.interleaved {
            Vec::new()
        } else {
            vec![0u8; cached_nodes * 8]
        };
        if !dir_indices.is_empty() {
            let offset = idx0 + (dir_node_start * 8) as u64;
            reader.read_exact_at(offset, &mut dir_indices)?;
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

    /// Gather `stride`-byte records for `positions` (sorted) from the section at
    /// `section0` into `out`. The planning and scatter live in [`plan_gather`] /
    /// [`apply_gather_run`] (shared with the async path); here we just read each
    /// coalesced run.
    #[allow(clippy::too_many_arguments)]
    fn gather(
        &self,
        positions: &[usize],
        section0: u64,
        stride: usize,
        cache: &[u8],
        out: &mut Vec<u8>,
        scratch: &mut Vec<u8>,
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
            scratch.clear();
            scratch.resize(run.len, 0);
            self.reader.read_exact_at(run.offset, scratch)?;
            apply_gather_run(out, run, scratch, stride);
        }
        Ok(())
    }

    /// Descend the tree level by level, calling `leaf` once at the leaf level
    /// with the surviving leaf positions (sorted) and their gathered index bytes
    /// (the insertion ids, in the same order). `overlaps` decides box
    /// intersection; this keeps the traversal dimension- and payload-independent.
    ///
    /// At each level the frontier's boxes are fetched (cached or
    /// coalesced-streamed) and tested; survivors expand to their child groups,
    /// and a parent that fails the test prunes its whole subtree.
    fn traverse<O, L>(&self, overlaps: O, mut leaf: L) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        L: FnMut(&[usize], &[u8], &mut Budget) -> Result<(), StreamError>,
    {
        if self.num_items == 0 {
            return Ok(());
        }

        let mut budget = Budget::new(self.limits);
        let mut frontier = vec![self.num_nodes - 1];
        let mut level = self.level_count - 1;
        let mut boxes = Vec::new();
        let mut indices = Vec::new();
        let mut scratch = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();

        loop {
            // One gather fetches each frontier node's box (interleaved: box +
            // index in the same `box_stride`-byte record; SoA: box only).
            self.gather(
                &frontier,
                self.box0,
                self.box_stride,
                &self.dir_boxes,
                &mut boxes,
                &mut scratch,
                &mut budget,
            )?;
            survivors.clear();
            indices.clear();
            for (i, &pos) in frontier.iter().enumerate() {
                let slot = i * self.box_stride;
                if overlaps(&boxes[slot..slot + self.record]) {
                    survivors.push(pos);
                    // Interleaved: the index trails the box in the same record, so
                    // no second gather is needed.
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
                self.gather(
                    &survivors,
                    self.idx0,
                    8,
                    &self.dir_indices,
                    &mut indices,
                    &mut scratch,
                    &mut budget,
                )?;
            }

            if level == 0 {
                // `survivors` are sorted leaf positions; `indices` their ids.
                return leaf(&survivors, &indices, &mut budget);
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

    /// Visit the insertion id of every leaf whose box satisfies `overlaps`.
    pub(crate) fn visit_ids<O, F>(&self, overlaps: O, mut visit: F) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize),
    {
        self.traverse(overlaps, |survivors, indices, budget| {
            for i in 0..survivors.len() {
                let id = read_index(indices, i)?;
                if id >= self.num_items {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                budget.charge_item()?;
                visit(id);
            }
            Ok(())
        })
    }

    /// Visit `(insertion id, payload blob)` for every leaf whose box satisfies
    /// `overlaps`, streaming the payload section in leaf order during the leaf
    /// pass so the offset table and blobs are read in coalesced runs.
    pub(crate) fn visit_payloads<O, F>(&self, overlaps: O, mut emit: F) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize, &[u8]),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        let mut off_buf = Vec::new();
        let mut blob_buf = Vec::new();
        self.traverse(overlaps, |survivors, indices, budget| {
            if section.stride != 0 {
                self.gather_payloads_fixed(
                    section,
                    survivors,
                    indices,
                    &mut blob_buf,
                    budget,
                    &mut emit,
                )
            } else {
                self.gather_payloads(
                    section,
                    survivors,
                    indices,
                    &mut off_buf,
                    &mut blob_buf,
                    budget,
                    &mut emit,
                )
            }
        })
    }

    /// Stream the blobs for `leaf_positions` (sorted leaf ranks) and their
    /// `indices` (insertion ids, same order), coalescing the leaf-ordered offset
    /// table and blob region into runs. Emits `(id, blob)` per leaf.
    #[allow(clippy::too_many_arguments)]
    fn gather_payloads<F>(
        &self,
        section: &PayloadSection,
        leaf_positions: &[usize],
        indices: &[u8],
        off_buf: &mut Vec<u8>,
        blob_buf: &mut Vec<u8>,
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        let mut j = 0;
        while j < leaf_positions.len() {
            let k = payload_run_end(leaf_positions, j, self.coalesce_gap());
            let lo = leaf_positions[j];
            let hi = leaf_positions[k];

            off_buf.clear();
            off_buf.resize((hi + 2 - lo) * 8, 0);
            budget.charge_read(off_buf.len())?;
            self.reader
                .read_exact_at(section.offsets_start + (lo * 8) as u64, off_buf)?;
            let (blob_lo, blob_hi) = payload_blob_span(off_buf, lo, hi, section.blob_total)?;

            blob_buf.clear();
            blob_buf.resize((blob_hi - blob_lo) as usize, 0);
            if !blob_buf.is_empty() {
                budget.charge_read(blob_buf.len())?;
                self.reader
                    .read_exact_at(section.blobs_start + blob_lo, blob_buf)?;
            }

            emit_run_payloads(
                leaf_positions,
                indices,
                j,
                k,
                lo,
                off_buf,
                blob_lo,
                blob_hi,
                blob_buf,
                self.num_items,
                budget,
                emit,
            )?;
            j = k + 1;
        }
        Ok(())
    }

    /// Fixed-width payload variant of [`gather_payloads`](Self::gather_payloads):
    /// no offset table, so each coalesced run is one contiguous blob read whose
    /// byte span is pure arithmetic (`lo * stride`).
    fn gather_payloads_fixed<F>(
        &self,
        section: &PayloadSection,
        leaf_positions: &[usize],
        indices: &[u8],
        blob_buf: &mut Vec<u8>,
        budget: &mut Budget,
        emit: &mut F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        let stride = section.stride as usize;
        let mut j = 0;
        while j < leaf_positions.len() {
            let k = payload_run_end_fixed(leaf_positions, j, stride, self.coalesce_gap());
            let lo = leaf_positions[j];
            let hi = leaf_positions[k];
            let span = (hi + 1 - lo) * stride;

            blob_buf.clear();
            blob_buf.resize(span, 0);
            budget.charge_read(span)?;
            self.reader
                .read_exact_at(section.blobs_start + (lo * stride) as u64, blob_buf)?;

            emit_run_payloads_fixed(
                leaf_positions,
                indices,
                j,
                k,
                lo,
                stride,
                blob_buf,
                self.num_items,
                budget,
                emit,
            )?;
            j = k + 1;
        }
        Ok(())
    }

    /// Visit a [`PayloadPrefix`] for every leaf whose box satisfies `overlaps`:
    /// the insertion id, its leaf rank, the payload's full byte length, and its
    /// first `prefix_len` bytes — without reading payload bodies past the
    /// prefix. Lengths come from the offset table (or the fixed stride), so a
    /// variable-width payload's body bytes beyond the prefix are never fetched.
    pub(crate) fn visit_payload_prefixes<O, F>(
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
        let mut off_buf = Vec::new();
        let mut spans: Vec<PrefixSpan> = Vec::new();
        let mut read_buf = Vec::new();
        self.traverse(overlaps, |survivors, indices, budget| {
            // Resolve each survivor's blob start and full length first.
            spans.clear();
            if section.stride != 0 {
                let stride = section.stride as usize;
                for (i, &p) in survivors.iter().enumerate() {
                    spans.push(PrefixSpan {
                        run_index: i,
                        leaf_rank: p,
                        blob_start: (p * stride) as u64,
                        payload_len: stride,
                    });
                }
            } else {
                let mut j = 0;
                while j < survivors.len() {
                    let k = payload_run_end(survivors, j, self.coalesce_gap());
                    let lo = survivors[j];
                    let hi = survivors[k];
                    off_buf.clear();
                    off_buf.resize((hi + 2 - lo) * 8, 0);
                    budget.charge_read(off_buf.len())?;
                    self.reader
                        .read_exact_at(section.offsets_start + (lo * 8) as u64, &mut off_buf)?;
                    for (offset, &p) in survivors[j..=k].iter().enumerate() {
                        let o0 = read_u64_le_unchecked(&off_buf, (p - lo) * 8);
                        let o1 = read_u64_le_unchecked(&off_buf, (p + 1 - lo) * 8);
                        if o1 < o0 || o1 > section.blob_total {
                            return Err(StreamError::Format(LoadError::InvalidTree));
                        }
                        spans.push(PrefixSpan {
                            run_index: j + offset,
                            leaf_rank: p,
                            blob_start: o0,
                            payload_len: (o1 - o0) as usize,
                        });
                    }
                    j = k + 1;
                }
            }

            // Coalesce the prefix byte spans into runs and emit each survivor.
            let gap = self.coalesce_gap();
            let mut j = 0;
            while j < spans.len() {
                let run_start = spans[j].blob_start;
                let mut run_end = spans[j].prefix_end(prefix_len);
                let mut k = j;
                while k + 1 < spans.len() {
                    let next = &spans[k + 1];
                    if next.blob_start < run_start || next.blob_start.saturating_sub(run_end) > gap
                    {
                        break;
                    }
                    run_end = run_end.max(next.prefix_end(prefix_len));
                    k += 1;
                }
                read_buf.clear();
                read_buf.resize((run_end - run_start) as usize, 0);
                if !read_buf.is_empty() {
                    budget.charge_read(read_buf.len())?;
                    self.reader
                        .read_exact_at(section.blobs_start + run_start, &mut read_buf)?;
                }
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
                j = k + 1;
            }
            Ok(())
        })
    }

    /// Visit `(leaf rank, payload blob)` for an explicit set of leaf ranks —
    /// random-access payload reads for ranks captured earlier by
    /// [`visit_payload_prefixes`](Self::visit_payload_prefixes). Input ranks
    /// are sorted and deduplicated internally so the payload section is read
    /// in coalesced ascending runs; blobs are emitted in ascending rank order.
    /// A rank at or past the item count fails with [`StreamError::InvalidRank`].
    pub(crate) fn visit_payloads_at_ranks<F>(
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
        let mut off_buf = Vec::new();
        let mut blob_buf = Vec::new();
        if section.stride != 0 {
            let stride = section.stride as usize;
            let mut j = 0;
            while j < ranks.len() {
                let k = payload_run_end_fixed(&ranks, j, stride, self.coalesce_gap());
                let lo = ranks[j];
                let hi = ranks[k];
                let span = (hi + 1 - lo) * stride;
                blob_buf.clear();
                blob_buf.resize(span, 0);
                budget.charge_read(span)?;
                self.reader
                    .read_exact_at(section.blobs_start + (lo * stride) as u64, &mut blob_buf)?;
                for &p in &ranks[j..=k] {
                    budget.charge_item()?;
                    let within = (p - lo) * stride;
                    emit(p, &blob_buf[within..within + stride]);
                }
                j = k + 1;
            }
        } else {
            let mut j = 0;
            while j < ranks.len() {
                let k = payload_run_end(&ranks, j, self.coalesce_gap());
                let lo = ranks[j];
                let hi = ranks[k];
                off_buf.clear();
                off_buf.resize((hi + 2 - lo) * 8, 0);
                budget.charge_read(off_buf.len())?;
                self.reader
                    .read_exact_at(section.offsets_start + (lo * 8) as u64, &mut off_buf)?;
                let (blob_lo, blob_hi) = payload_blob_span(&off_buf, lo, hi, section.blob_total)?;
                blob_buf.clear();
                blob_buf.resize((blob_hi - blob_lo) as usize, 0);
                if !blob_buf.is_empty() {
                    budget.charge_read(blob_buf.len())?;
                    self.reader
                        .read_exact_at(section.blobs_start + blob_lo, &mut blob_buf)?;
                }
                for &p in &ranks[j..=k] {
                    let o0 = read_u64_le_unchecked(&off_buf, (p - lo) * 8);
                    let o1 = read_u64_le_unchecked(&off_buf, (p + 1 - lo) * 8);
                    if o0 < blob_lo || o1 < o0 || o1 > blob_hi {
                        return Err(StreamError::Format(LoadError::InvalidTree));
                    }
                    budget.charge_item()?;
                    emit(
                        p,
                        &blob_buf[(o0 - blob_lo) as usize..(o1 - blob_lo) as usize],
                    );
                }
                j = k + 1;
            }
        }
        Ok(())
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

/// One matching leaf seen by
/// [`visit_payload_prefixes`](StreamCore::visit_payload_prefixes): identity
/// plus payload size, with only the leading payload bytes fetched.
#[derive(Debug)]
pub struct PayloadPrefix<'a> {
    /// Item insertion id.
    pub id: usize,
    /// Leaf rank — the item's position in the leaf-ordered payload section.
    /// Stable for one serialized index, not across rebuilds. Feed ranks to
    /// [`StreamIndex2D::visit_payloads_at_ranks`](super::StreamIndex2D::visit_payloads_at_ranks)
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
