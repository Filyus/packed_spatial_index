//! SoA index variant with SIMD searches (available with the `simd` feature).
//!
//! items are stored as four separate arrays (`min_x[]`, `min_y[]`, `max_x[]`,
//! `max_y[]`). The tree is built exactly like the AoS version; only the layout
//! and search implementation differ.

use std::{collections::BinaryHeap, ops::ControlFlow};

use wide::{CmpGe, CmpLe, f64x4};

use crate::{
    build::BuildError,
    builder2d::BuildConfig,
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::{Box2D, Point2D},
    neighbors::{NeighborNodeState, NeighborState, NeighborWorkspace, max_distance_squared},
    persistence::{
        ByteWriter, LoadError, parse_index_bytes, read_f64_le_unchecked, read_u64_le_unchecked,
        serialized_len,
    },
    sort2d::{SortKeyContext, encode_sort_by_key},
    traversal::{SearchWorkspace, prefetch_read, upper_bound_level},
    tree::{TreeLayout, try_compute_tree_layout},
};

type Num = f64;

pub(crate) fn build_simd_index(
    config: BuildConfig,
    items: Vec<Box2D>,
) -> Result<SimdIndex2D, BuildError> {
    let node_size = config.node_size;
    let num_items = config.num_items;
    let TreeLayout {
        level_bounds,
        num_nodes,
    } = try_compute_tree_layout(num_items, node_size)?;

    if num_items == 0 {
        return Ok(SimdIndex2D {
            node_size,
            num_items,
            level_bounds,
            min_xs: Vec::new(),
            min_ys: Vec::new(),
            max_xs: Vec::new(),
            max_ys: Vec::new(),
            indices: Vec::new(),
        });
    }

    if num_items <= node_size {
        return Ok(build_single_node_soa(
            node_size,
            num_items,
            level_bounds,
            items,
        ));
    }

    let mut min_xs = vec![0.0f64; num_nodes];
    let mut min_ys = vec![0.0f64; num_nodes];
    let mut max_xs = vec![0.0f64; num_nodes];
    let mut max_ys = vec![0.0f64; num_nodes];
    let mut indices = vec![0usize; num_nodes];

    let (mut e_min_x, mut e_min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut e_max_x, mut e_max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for b in &items {
        e_min_x = e_min_x.min(b.min_x);
        e_min_y = e_min_y.min(b.min_y);
        e_max_x = e_max_x.max(b.max_x);
        e_max_y = e_max_y.max(b.max_y);
    }
    let scaled_width = u16::MAX as f64 / (e_max_x - e_min_x);
    let scaled_height = u16::MAX as f64 / (e_max_y - e_min_y);

    #[cfg(feature = "parallel")]
    let use_parallel = config.parallel && num_items >= config.parallel_min_items;

    let context = SortKeyContext {
        scaled_width,
        scaled_height,
        min_x: e_min_x,
        min_y: e_min_y,
        radix: config.radix,
        radix_bits: config.radix_bits,
        #[cfg(feature = "parallel")]
        use_parallel,
    };
    let order = encode_sort_by_key(&items, config.sort_key, context);

    #[cfg(feature = "parallel")]
    let scattered_in_parallel = if use_parallel {
        reorder_parallel_soa_2d(
            &mut min_xs[..num_items],
            &mut min_ys[..num_items],
            &mut max_xs[..num_items],
            &mut max_ys[..num_items],
            &mut indices[..num_items],
            &order,
            &items,
        );
        true
    } else {
        false
    };
    #[cfg(not(feature = "parallel"))]
    let scattered_in_parallel = false;

    if !scattered_in_parallel {
        for (slot, &(_, orig)) in order.iter().enumerate() {
            let b = items[orig as usize];
            min_xs[slot] = b.min_x;
            min_ys[slot] = b.min_y;
            max_xs[slot] = b.max_x;
            max_ys[slot] = b.max_y;
            indices[slot] = orig as usize;
        }
    }

    let mut read_pos = 0usize;
    let mut write_pos = num_items;
    for &level_end in &level_bounds[0..level_bounds.len() - 1] {
        while read_pos < level_end {
            let node_index = read_pos;
            let (mut nmnx, mut nmny) = (f64::INFINITY, f64::INFINITY);
            let (mut nmxx, mut nmxy) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            let mut j = 0;
            while j < node_size && read_pos < level_end {
                nmnx = nmnx.min(min_xs[read_pos]);
                nmny = nmny.min(min_ys[read_pos]);
                nmxx = nmxx.max(max_xs[read_pos]);
                nmxy = nmxy.max(max_ys[read_pos]);
                read_pos += 1;
                j += 1;
            }
            min_xs[write_pos] = nmnx;
            min_ys[write_pos] = nmny;
            max_xs[write_pos] = nmxx;
            max_ys[write_pos] = nmxy;
            indices[write_pos] = node_index;
            write_pos += 1;
        }
    }

    Ok(SimdIndex2D {
        node_size,
        num_items,
        level_bounds,
        min_xs,
        min_ys,
        max_xs,
        max_ys,
        indices,
    })
}

fn build_single_node_soa(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    items: Vec<Box2D>,
) -> SimdIndex2D {
    let mut min_xs = Vec::with_capacity(num_items + 1);
    let mut min_ys = Vec::with_capacity(num_items + 1);
    let mut max_xs = Vec::with_capacity(num_items + 1);
    let mut max_ys = Vec::with_capacity(num_items + 1);
    let mut indices = Vec::with_capacity(num_items + 1);

    let (mut root_min_x, mut root_min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut root_max_x, mut root_max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (idx, b) in items.into_iter().enumerate() {
        min_xs.push(b.min_x);
        min_ys.push(b.min_y);
        max_xs.push(b.max_x);
        max_ys.push(b.max_y);
        indices.push(idx);

        root_min_x = root_min_x.min(b.min_x);
        root_min_y = root_min_y.min(b.min_y);
        root_max_x = root_max_x.max(b.max_x);
        root_max_y = root_max_y.max(b.max_y);
    }

    min_xs.push(root_min_x);
    min_ys.push(root_min_y);
    max_xs.push(root_max_x);
    max_ys.push(root_max_y);
    indices.push(0);

    SimdIndex2D {
        node_size,
        num_items,
        level_bounds,
        min_xs,
        min_ys,
        max_xs,
        max_ys,
        indices,
    }
}

/// Finished read-only SIMD index.
///
/// Created through [`Index2DBuilder::finish_simd`](crate::Index2DBuilder::finish_simd).
/// It has the same public search and nearest-neighbor API as [`Index2D`](crate::Index2D),
/// but stores box coordinates in structure-of-arrays form for SIMD traversal.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, Box2D};
///
/// let mut builder = Index2DBuilder::new(1);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
///
/// let index = builder.finish_simd().unwrap();
/// assert_eq!(index.search(Box2D::new(0.5, 0.5, 0.5, 0.5)), vec![0]);
/// ```
pub struct SimdIndex2D {
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    min_xs: Vec<Num>,
    min_ys: Vec<Num>,
    max_xs: Vec<Num>,
    max_ys: Vec<Num>,
    indices: Vec<usize>,
}

impl SimdIndex2D {
    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total extent of indexed items, or `None` for an empty index.
    pub fn extent(&self) -> Option<Box2D> {
        if self.num_items == 0 {
            None
        } else {
            let last = self.min_xs.len() - 1;
            Some(Box2D::new(
                self.min_xs[last],
                self.min_ys[last],
                self.max_xs[last],
                self.max_ys[last],
            ))
        }
    }

    /// Return the packed node size used by this index.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Serialize into the stable little-endian `PSINDEX` format.
    ///
    /// The output is byte-identical to [`Index2D::to_bytes`](crate::Index2D::to_bytes)
    /// for the same items, so a `SimdIndex2D` and an `Index2D` are interchangeable on
    /// disk: either can load bytes produced by the other.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.to_bytes_into(&mut out);
        out
    }

    /// Serialize into a caller-provided buffer, reusing its allocation.
    ///
    /// Equivalent to [`to_bytes`](Self::to_bytes) but writes into `out` (cleared first).
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        let level_count = self.level_bounds.len();
        let num_nodes = self.min_xs.len();
        let len = serialized_len(level_count, num_nodes).expect("serialized index is too large");
        let mut bytes = ByteWriter::new(out, len);
        bytes.write_magic();
        bytes.write_format_version();
        bytes.write_header_len();
        bytes.write_flags();
        bytes.write_u64(self.node_size as u64);
        bytes.write_u64(self.num_items as u64);
        bytes.write_u64(num_nodes as u64);
        bytes.write_u64(level_count as u64);
        bytes.write_usize_slice_as_u64(&self.level_bounds);
        bytes.write_soa_boxes_2d(&self.min_xs, &self.min_ys, &self.max_xs, &self.max_ys);
        bytes.write_usize_slice_as_u64(&self.indices);
        bytes.finish();
    }

    /// Load a SIMD index from bytes produced by [`to_bytes`](Self::to_bytes) or by
    /// [`Index2D::to_bytes`](crate::Index2D::to_bytes); the AoS box records are
    /// scattered into the structure-of-arrays columns.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let parsed = parse_index_bytes(bytes)?;
        let num_nodes = parsed.num_nodes;

        let mut level_bounds = Vec::with_capacity(parsed.level_count);
        for i in 0..parsed.level_count {
            level_bounds.push(read_u64_le_unchecked(parsed.level_bounds, i * 8) as usize);
        }

        let mut min_xs = Vec::with_capacity(num_nodes);
        let mut min_ys = Vec::with_capacity(num_nodes);
        let mut max_xs = Vec::with_capacity(num_nodes);
        let mut max_ys = Vec::with_capacity(num_nodes);
        let mut indices = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let off = i * 32; // four f64 per 2D box record
            min_xs.push(read_f64_le_unchecked(parsed.entries, off));
            min_ys.push(read_f64_le_unchecked(parsed.entries, off + 8));
            max_xs.push(read_f64_le_unchecked(parsed.entries, off + 16));
            max_ys.push(read_f64_le_unchecked(parsed.entries, off + 24));
            indices.push(read_u64_le_unchecked(parsed.indices, i * 8) as usize);
        }

        Ok(SimdIndex2D {
            node_size: parsed.node_size,
            num_items: parsed.num_items,
            level_bounds,
            min_xs,
            min_ys,
            max_xs,
            max_ys,
            indices,
        })
    }

    #[inline]
    fn prefetch_node(&self, node_index: usize) {
        if node_index < self.min_xs.len() {
            prefetch_read(self.min_xs.as_ptr().wrapping_add(node_index));
            prefetch_read(self.min_ys.as_ptr().wrapping_add(node_index));
            prefetch_read(self.max_xs.as_ptr().wrapping_add(node_index));
            prefetch_read(self.max_ys.as_ptr().wrapping_add(node_index));
            prefetch_read(self.indices.as_ptr().wrapping_add(node_index));
        }
        let next_line = node_index.saturating_add((64 / std::mem::size_of::<Num>()).max(1));
        if self.node_size > 1 && next_line < self.min_xs.len() {
            prefetch_read(self.min_xs.as_ptr().wrapping_add(next_line));
            prefetch_read(self.min_ys.as_ptr().wrapping_add(next_line));
            prefetch_read(self.max_xs.as_ptr().wrapping_add(next_line));
            prefetch_read(self.max_ys.as_ptr().wrapping_add(next_line));
            prefetch_read(self.indices.as_ptr().wrapping_add(next_line));
        }
    }

    /// Return the indices of all items whose boxes intersect `query`.
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(query, &mut out);
        out
    }

    /// Search with a reusable result buffer.
    ///
    /// This automatically chooses the widest available SIMD implementation: AVX-512
    /// on supporting x86-64 CPUs, otherwise AVX2/SSE through `wide`.
    pub fn search_into(&self, query: Box2D, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_avx512(query, out, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'a>(&self, query: Box2D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.search_avx512(query, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `query`.
    pub fn any(&self, query: Box2D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    pub fn first(&self, query: Box2D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Return up to `max_results` item indices nearest to `point`.
    pub fn neighbors(&self, point: Point2D, max_results: usize) -> Vec<usize> {
        self.neighbors_within(point, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of `point`.
    pub fn neighbors_within(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        self.neighbors_into(point, max_results, max_distance, &mut results);
        results
    }

    /// Nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_into(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
    ) {
        results.clear();
        if max_results == 0 {
            return;
        }
        if max_results == 1 {
            let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
            if let Some(index) = self.nearest_one_with_queue(point, max_distance, &mut queue) {
                results.push(index);
            }
            return;
        }

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(point, max_results, max_distance, results, &mut queue);
    }

    /// Nearest-neighbor search with reusable result and priority-queue buffers.
    pub fn neighbors_with<'a>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        workspace: &'a mut NeighborWorkspace,
    ) -> &'a [usize] {
        workspace.results.clear();
        if max_results == 0 {
            workspace.queue.clear();
            workspace.node_queue.clear();
            return &workspace.results;
        }
        if max_results == 1 {
            workspace.queue.clear();
            if let Some(index) =
                self.nearest_one_with_queue(point, max_distance, &mut workspace.node_queue)
            {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            point,
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
        );
        &workspace.results
    }

    /// Visit items in nondecreasing squared-distance order from `point`.
    pub fn visit_neighbors<B, F>(
        &self,
        point: Point2D,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.visit_neighbors_with_queue(point, max_distance, &mut queue, &mut visitor)
    }

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_avx512(query, &mut stack, visitor)
    }

    fn collect_neighbors_with_queue(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        queue: &mut BinaryHeap<NeighborState>,
    ) {
        queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if self.num_items == 0 {
            return;
        }

        let mut node_index = self.min_xs.len() - 1;
        loop {
            let upper_bound_level = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist > max_dist_sq {
                    continue;
                }
                queue.push(NeighborState::new(self.indices[pos], is_leaf, dist));
            }

            let mut continue_search = false;
            while let Some(state) = queue.pop() {
                if state.dist > max_dist_sq {
                    queue.clear();
                    return;
                }
                if state.is_leaf {
                    results.push(state.index);
                    if results.len() == max_results {
                        return;
                    }
                } else {
                    node_index = state.index;
                    continue_search = true;
                    break;
                }
            }

            if !continue_search {
                return;
            }
        }
    }

    fn nearest_one_with_queue(
        &self,
        point: Point2D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        queue.clear();
        let mut best_dist = max_distance_squared(max_distance)?;
        if self.num_items == 0 {
            return None;
        }

        let mut best_index = None;
        let mut node_index = self.min_xs.len() - 1;
        loop {
            let upper_bound_level = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist > best_dist {
                    continue;
                }
                if is_leaf {
                    if dist == 0.0 {
                        return Some(self.indices[pos]);
                    }
                    best_dist = dist;
                    best_index = Some(self.indices[pos]);
                } else {
                    queue.push(NeighborNodeState::new(self.indices[pos], dist));
                }
            }

            match queue.pop() {
                Some(state) if state.dist <= best_dist => node_index = state.index,
                _ => return best_index,
            }
        }
    }

    fn visit_neighbors_with_queue<B, F>(
        &self,
        point: Point2D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborState>,
        visitor: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return ControlFlow::Continue(());
        };
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.min_xs.len() - 1;
        loop {
            let upper_bound_level = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist > max_dist_sq {
                    continue;
                }
                queue.push(NeighborState::new(self.indices[pos], is_leaf, dist));
            }

            let mut continue_search = false;
            while let Some(state) = queue.pop() {
                if state.dist > max_dist_sq {
                    queue.clear();
                    return ControlFlow::Continue(());
                }
                if state.is_leaf {
                    visitor(state.index, state.dist)?;
                } else {
                    node_index = state.index;
                    continue_search = true;
                    break;
                }
            }

            if !continue_search {
                return ControlFlow::Continue(());
            }
        }
    }

    #[inline]
    fn distance_squared_to(&self, pos: usize, point: Point2D) -> f64 {
        let dx = axis_distance(point.x, self.min_xs[pos], self.max_xs[pos]);
        let dy = axis_distance(point.y, self.min_ys[pos], self.max_ys[pos]);
        dx * dx + dy * dy
    }

    /// Same as [`visit`](SimdIndex2D::visit), but the traversal stack is reused by the caller.
    #[doc(hidden)]
    pub fn visit_simd<B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_simd_impl::<false, B, F>(query, stack, visitor)
    }

    /// Hidden prefetch variant of [`visit_simd`](SimdIndex2D::visit_simd).
    #[doc(hidden)]
    pub fn visit_simd_prefetch<B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_simd_impl::<true, B, F>(query, stack, visitor)
    }

    /// AVX-512 visitor path, falling back to [`visit_simd`](SimdIndex2D::visit_simd).
    #[doc(hidden)]
    pub fn visit_avx512<B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: this branch is selected only after checking avx512f availability.
                return unsafe { self.visit_avx512_impl::<B, F>(query, stack, visitor) };
            }
        }
        self.visit_simd(query, stack, visitor)
    }

    /// Element-by-element traversal (SoA layout, branchless `overlaps`).
    #[doc(hidden)]
    pub fn search_scalar(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                let hit = (self.min_xs[pos] <= query.max_x)
                    & (self.max_xs[pos] >= query.min_x)
                    & (self.min_ys[pos] <= query.max_y)
                    & (self.max_ys[pos] >= query.min_y);
                if !hit {
                    continue;
                }
                let index = self.indices[pos];
                if is_leaf {
                    out.push(index);
                } else {
                    stack.push(index);
                    stack.push(level - 1);
                }
            }
            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }

    /// AVX2/SSE path through `wide::f64x4`.
    #[doc(hidden)]
    pub fn search_simd(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        self.search_simd_impl::<false>(query, out, stack);
    }

    /// AVX2/SSE path with prefetch for the next node from the stack.
    #[doc(hidden)]
    pub fn search_simd_prefetch(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        self.search_simd_impl::<true>(query, out, stack);
    }

    fn search_simd_impl<const PREFETCH: bool>(
        &self,
        query: Box2D,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = f64x4::splat(query.max_x);
        let qmnx_v = f64x4::splat(query.min_x);
        let qmxy_v = f64x4::splat(query.max_y);
        let qmny_v = f64x4::splat(query.min_y);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 4 <= end {
                let mnx = load4(&self.min_xs, pos);
                let mxx = load4(&self.max_xs, pos);
                let mny = load4(&self.min_ys, pos);
                let mxy = load4(&self.max_ys, pos);
                let mask = mnx.simd_le(qmxx_v)
                    & mxx.simd_ge(qmnx_v)
                    & mny.simd_le(qmxy_v)
                    & mxy.simd_ge(qmny_v);
                let bits = mask.to_bitmask();
                if bits != 0 {
                    for k in 0..4 {
                        if bits & (1 << k) != 0 {
                            let p = pos + k;
                            let index = self.indices[p];
                            if is_leaf {
                                out.push(index);
                            } else {
                                stack.push(index);
                                stack.push(level - 1);
                            }
                        }
                    }
                }
                pos += 4;
            }

            while pos < end {
                let hit = (self.min_xs[pos] <= query.max_x)
                    & (self.max_xs[pos] >= query.min_x)
                    & (self.min_ys[pos] <= query.max_y)
                    & (self.max_ys[pos] >= query.min_y);
                if hit {
                    let index = self.indices[pos];
                    if is_leaf {
                        out.push(index);
                    } else {
                        stack.push(index);
                        stack.push(level - 1);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                if PREFETCH {
                    self.prefetch_node(stack[stack.len() - 2]);
                }
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }

    fn visit_simd_impl<const PREFETCH: bool, B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }
        let qmxx_v = f64x4::splat(query.max_x);
        let qmnx_v = f64x4::splat(query.min_x);
        let qmxy_v = f64x4::splat(query.max_y);
        let qmny_v = f64x4::splat(query.min_y);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 4 <= end {
                let mnx = load4(&self.min_xs, pos);
                let mxx = load4(&self.max_xs, pos);
                let mny = load4(&self.min_ys, pos);
                let mxy = load4(&self.max_ys, pos);
                let mask = mnx.simd_le(qmxx_v)
                    & mxx.simd_ge(qmnx_v)
                    & mny.simd_le(qmxy_v)
                    & mxy.simd_ge(qmny_v);
                let bits = mask.to_bitmask();
                if bits != 0 {
                    for k in 0..4 {
                        if bits & (1 << k) != 0 {
                            let p = pos + k;
                            let index = self.indices[p];
                            if is_leaf {
                                visitor(index)?;
                            } else {
                                stack.push(index);
                                stack.push(level - 1);
                            }
                        }
                    }
                }
                pos += 4;
            }

            while pos < end {
                let hit = (self.min_xs[pos] <= query.max_x)
                    & (self.max_xs[pos] >= query.min_x)
                    & (self.min_ys[pos] <= query.max_y)
                    & (self.max_ys[pos] >= query.min_y);
                if hit {
                    let index = self.indices[pos];
                    if is_leaf {
                        visitor(index)?;
                    } else {
                        stack.push(index);
                        stack.push(level - 1);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                if PREFETCH {
                    self.prefetch_node(stack[stack.len() - 2]);
                }
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return ControlFlow::Continue(());
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn visit_avx512_impl<B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        use std::arch::x86_64::*;

        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }
        let qmxx_v = _mm512_set1_pd(query.max_x);
        let qmnx_v = _mm512_set1_pd(query.min_x);
        let qmxy_v = _mm512_set1_pd(query.max_y);
        let qmny_v = _mm512_set1_pd(query.min_y);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end`, and `end` is bounded by the array length.
                let (mnx, mxx, mny, mxy) = unsafe {
                    (
                        _mm512_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_ys.as_ptr().add(pos)),
                    )
                };
                let m1 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mnx, qmxx_v);
                let m2 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mxx, qmnx_v);
                let m3 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mny, qmxy_v);
                let m4 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mxy, qmny_v);
                let mut bits: u8 = m1 & m2 & m3 & m4;
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    let index = self.indices[pos + k];
                    if is_leaf {
                        visitor(index)?;
                    } else {
                        stack.push(index);
                        stack.push(level - 1);
                    }
                    bits &= bits - 1;
                }
                pos += 8;
            }

            while pos < end {
                let hit = (self.min_xs[pos] <= query.max_x)
                    & (self.max_xs[pos] >= query.min_x)
                    & (self.min_ys[pos] <= query.max_y)
                    & (self.max_ys[pos] >= query.min_y);
                if hit {
                    let index = self.indices[pos];
                    if is_leaf {
                        visitor(index)?;
                    } else {
                        stack.push(index);
                        stack.push(level - 1);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return ControlFlow::Continue(());
            }
        }
    }

    /// AVX-512 path, falling back to [`search_simd`](SimdIndex2D::search_simd).
    #[doc(hidden)]
    pub fn search_avx512(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: this branch is selected only after checking avx512f availability.
                unsafe { self.search_avx512_impl(query, out, stack) };
                return;
            }
        }
        self.search_simd(query, out, stack);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn search_avx512_impl(
        &self,
        query: Box2D,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        use std::arch::x86_64::*;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = _mm512_set1_pd(query.max_x);
        let qmnx_v = _mm512_set1_pd(query.min_x);
        let qmxy_v = _mm512_set1_pd(query.max_y);
        let qmny_v = _mm512_set1_pd(query.min_y);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end`, and `end` is bounded by the array length.
                let (mnx, mxx, mny, mxy) = unsafe {
                    (
                        _mm512_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_ys.as_ptr().add(pos)),
                    )
                };
                let m1 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mnx, qmxx_v);
                let m2 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mxx, qmnx_v);
                let m3 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mny, qmxy_v);
                let m4 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mxy, qmny_v);
                let mut bits: u8 = m1 & m2 & m3 & m4;
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    let index = self.indices[pos + k];
                    if is_leaf {
                        out.push(index);
                    } else {
                        stack.push(index);
                        stack.push(level - 1);
                    }
                    bits &= bits - 1;
                }
                pos += 8;
            }

            while pos < end {
                let hit = (self.min_xs[pos] <= query.max_x)
                    & (self.max_xs[pos] >= query.min_x)
                    & (self.min_ys[pos] <= query.max_y)
                    & (self.max_ys[pos] >= query.min_y);
                if hit {
                    let index = self.indices[pos];
                    if is_leaf {
                        out.push(index);
                    } else {
                        stack.push(index);
                        stack.push(level - 1);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }
}

#[inline]
fn axis_distance(point: f64, min: f64, max: f64) -> f64 {
    if point < min {
        min - point
    } else if point > max {
        point - max
    } else {
        0.0
    }
}

#[inline]
fn load4(a: &[f64], p: usize) -> f64x4 {
    f64x4::from([a[p], a[p + 1], a[p + 2], a[p + 3]])
}

/// Scatter the Hilbert-ordered items into the SoA leaf columns in parallel. Each
/// output slot is written exactly once, so the columns can be filled independently.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn reorder_parallel_soa_2d(
    min_xs: &mut [f64],
    min_ys: &mut [f64],
    max_xs: &mut [f64],
    max_ys: &mut [f64],
    indices: &mut [usize],
    order: &[(u32, u32)],
    items: &[Box2D],
) {
    use rayon::prelude::*;

    min_xs
        .par_iter_mut()
        .zip(min_ys.par_iter_mut())
        .zip(max_xs.par_iter_mut())
        .zip(max_ys.par_iter_mut())
        .zip(indices.par_iter_mut())
        .zip(order.par_iter())
        .for_each(|(((((mnx, mny), mxx), mxy), idx), &(_, orig))| {
            let b = items[orig as usize];
            *mnx = b.min_x;
            *mny = b.min_y;
            *mxx = b.max_x;
            *mxy = b.max_y;
            *idx = orig as usize;
        });
}

/// Byte size of one persisted 2D box record (`[min_x, min_y, max_x, max_y]`).
const RECORD_2D: usize = 32;

/// Assemble one coordinate column for four consecutive 2D box records into a SIMD
/// vector. The four records are contiguous (128 bytes), so the strided reads stay
/// within the same cache lines.
#[inline]
fn lane4_2d(entries: &[u8], base: usize, field: usize) -> f64x4 {
    let o = base + field;
    f64x4::from([
        read_f64_le_unchecked(entries, o),
        read_f64_le_unchecked(entries, o + RECORD_2D),
        read_f64_le_unchecked(entries, o + 2 * RECORD_2D),
        read_f64_le_unchecked(entries, o + 3 * RECORD_2D),
    ])
}

/// Zero-copy SIMD view over bytes produced by [`SimdIndex2D::to_bytes`] or
/// [`Index2D::to_bytes`](crate::Index2D::to_bytes).
///
/// Like [`Index2DView`](crate::Index2DView) it borrows the buffer without
/// allocating owned tree storage, but the traversal uses `wide::f64x4` overlap
/// tests, assembling lane vectors from four contiguous box records. Ideal for
/// querying memory-mapped indexes without allocating.
///
/// Nearest-neighbor results are returned in nondecreasing distance order. Ties
/// between equal-distance items are not stable across index layouts.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, SimdIndex2DView, Box2D};
///
/// let mut builder = Index2DBuilder::new(1);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// let bytes = builder.finish_simd().unwrap().to_bytes();
///
/// let view = SimdIndex2DView::from_bytes(&bytes)?;
/// assert_eq!(view.search(Box2D::new(0.5, 0.5, 0.5, 0.5)), vec![0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SimdIndex2DView<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    level_bounds: &'a [u8],
    entries: &'a [u8],
    indices: &'a [u8],
}

impl<'a> SimdIndex2DView<'a> {
    /// Borrow a zero-copy view over the canonical `PSINDEX` 2D bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let parsed = parse_index_bytes(bytes)?;
        Ok(Self {
            node_size: parsed.node_size,
            num_items: parsed.num_items,
            num_nodes: parsed.num_nodes,
            level_count: parsed.level_count,
            level_bounds: parsed.level_bounds,
            entries: parsed.entries,
            indices: parsed.indices,
        })
    }

    /// Return the number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the packed node size.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Return the total extent of indexed items, or `None` for an empty view.
    pub fn extent(&self) -> Option<Box2D> {
        if self.num_items == 0 {
            None
        } else {
            Some(self.box_at(self.num_nodes - 1))
        }
    }

    #[inline]
    fn index_at(&self, pos: usize) -> usize {
        read_u64_le_unchecked(self.indices, pos * 8) as usize
    }

    #[inline]
    fn box_at(&self, pos: usize) -> Box2D {
        let b = pos * RECORD_2D;
        Box2D::new(
            read_f64_le_unchecked(self.entries, b),
            read_f64_le_unchecked(self.entries, b + 8),
            read_f64_le_unchecked(self.entries, b + 16),
            read_f64_le_unchecked(self.entries, b + 24),
        )
    }

    #[inline]
    fn level_bound_unchecked(&self, index: usize) -> usize {
        read_u64_le_unchecked(self.level_bounds, index * 8) as usize
    }

    #[inline]
    fn upper_bound_level(&self, node_index: usize) -> usize {
        let mut lo = 0usize;
        let mut hi = self.level_count - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.level_bound_unchecked(mid) > node_index {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    /// Return the indices of all items whose boxes intersect `query`.
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(query, &mut out);
        out
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, query: Box2D, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        out.clear();
        let _: ControlFlow<()> = self.try_visit(query, &mut stack, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'b>(&self, query: Box2D, workspace: &'b mut SearchWorkspace) -> &'b [usize] {
        workspace.results.clear();
        let results = &mut workspace.results;
        let _: ControlFlow<()> = self.try_visit(query, &mut workspace.stack, |index| {
            results.push(index);
            ControlFlow::Continue(())
        });
        &workspace.results
    }

    /// Return `true` if at least one item intersects `query`.
    pub fn any(&self, query: Box2D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    pub fn first(&self, query: Box2D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.try_visit(query, &mut stack, visitor)
    }

    fn try_visit<B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }
        let qmxx = f64x4::splat(query.max_x);
        let qmnx = f64x4::splat(query.min_x);
        let qmxy = f64x4::splat(query.max_y);
        let qmny = f64x4::splat(query.min_y);

        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 4 <= end {
                let base = pos * RECORD_2D;
                let mask = lane4_2d(self.entries, base, 0).simd_le(qmxx)
                    & lane4_2d(self.entries, base, 16).simd_ge(qmnx)
                    & lane4_2d(self.entries, base, 8).simd_le(qmxy)
                    & lane4_2d(self.entries, base, 24).simd_ge(qmny);
                let bits = mask.to_bitmask();
                if bits != 0 {
                    for k in 0..4 {
                        if bits & (1 << k) != 0 {
                            let p = pos + k;
                            let index = self.index_at(p);
                            if is_leaf {
                                visitor(index)?;
                            } else {
                                stack.push(index);
                                stack.push(level - 1);
                            }
                        }
                    }
                }
                pos += 4;
            }

            while pos < end {
                if self.box_at(pos).overlaps(query) {
                    let index = self.index_at(pos);
                    if is_leaf {
                        visitor(index)?;
                    } else {
                        stack.push(index);
                        stack.push(level - 1);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return ControlFlow::Continue(());
            }
        }
    }

    #[inline]
    fn distance_squared_to(&self, pos: usize, point: Point2D) -> f64 {
        let b = self.box_at(pos);
        let dx = axis_distance(point.x, b.min_x, b.max_x);
        let dy = axis_distance(point.y, b.min_y, b.max_y);
        dx * dx + dy * dy
    }

    /// Return up to `max_results` item indices nearest to `point`.
    pub fn neighbors(&self, point: Point2D, max_results: usize) -> Vec<usize> {
        self.neighbors_within(point, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of `point`.
    pub fn neighbors_within(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        self.neighbors_into(point, max_results, max_distance, &mut results);
        results
    }

    /// Nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_into(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
    ) {
        results.clear();
        if max_results == 0 {
            return;
        }
        if max_results == 1 {
            let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
            if let Some(index) = self.nearest_one_with_queue(point, max_distance, &mut queue) {
                results.push(index);
            }
            return;
        }
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(point, max_results, max_distance, results, &mut queue);
    }

    /// Nearest-neighbor search with reusable result and priority-queue buffers.
    pub fn neighbors_with<'b>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        workspace: &'b mut NeighborWorkspace,
    ) -> &'b [usize] {
        workspace.results.clear();
        if max_results == 0 {
            workspace.queue.clear();
            workspace.node_queue.clear();
            return &workspace.results;
        }
        if max_results == 1 {
            workspace.queue.clear();
            if let Some(index) =
                self.nearest_one_with_queue(point, max_distance, &mut workspace.node_queue)
            {
                workspace.results.push(index);
            }
            return &workspace.results;
        }
        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            point,
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
        );
        &workspace.results
    }

    /// Visit items in nondecreasing squared-distance order from `point`.
    pub fn visit_neighbors<B, F>(
        &self,
        point: Point2D,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.visit_neighbors_with_queue(point, max_distance, &mut queue, &mut visitor)
    }

    fn collect_neighbors_with_queue(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        queue: &mut BinaryHeap<NeighborState>,
    ) {
        queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if self.num_items == 0 {
            return;
        }

        let mut node_index = self.num_nodes - 1;
        loop {
            let upper = self.upper_bound_level(node_index);
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(upper));
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist > max_dist_sq {
                    continue;
                }
                queue.push(NeighborState::new(self.index_at(pos), is_leaf, dist));
            }

            let mut continue_search = false;
            while let Some(state) = queue.pop() {
                if state.dist > max_dist_sq {
                    queue.clear();
                    return;
                }
                if state.is_leaf {
                    results.push(state.index);
                    if results.len() == max_results {
                        return;
                    }
                } else {
                    node_index = state.index;
                    continue_search = true;
                    break;
                }
            }
            if !continue_search {
                return;
            }
        }
    }

    fn visit_neighbors_with_queue<B, F>(
        &self,
        point: Point2D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborState>,
        visitor: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return ControlFlow::Continue(());
        };
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.num_nodes - 1;
        loop {
            let upper = self.upper_bound_level(node_index);
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(upper));
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist > max_dist_sq {
                    continue;
                }
                queue.push(NeighborState::new(self.index_at(pos), is_leaf, dist));
            }

            let mut continue_search = false;
            while let Some(state) = queue.pop() {
                if state.dist > max_dist_sq {
                    queue.clear();
                    return ControlFlow::Continue(());
                }
                if state.is_leaf {
                    visitor(state.index, state.dist)?;
                } else {
                    node_index = state.index;
                    continue_search = true;
                    break;
                }
            }
            if !continue_search {
                return ControlFlow::Continue(());
            }
        }
    }

    fn nearest_one_with_queue(
        &self,
        point: Point2D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        queue.clear();
        let mut best_dist = max_distance_squared(max_distance)?;
        if self.num_items == 0 {
            return None;
        }
        let mut best_index = None;
        let mut node_index = self.num_nodes - 1;
        loop {
            let upper = self.upper_bound_level(node_index);
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(upper));
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist > best_dist {
                    continue;
                }
                if is_leaf {
                    if dist == 0.0 {
                        return Some(self.index_at(pos));
                    }
                    best_dist = dist;
                    best_index = Some(self.index_at(pos));
                } else {
                    queue.push(NeighborNodeState::new(self.index_at(pos), dist));
                }
            }
            match queue.pop() {
                Some(state) if state.dist <= best_dist => node_index = state.index,
                _ => return best_index,
            }
        }
    }
}
