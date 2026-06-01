//! SoA index variant with SIMD searches (available with the `simd` feature).
//!
//! Boxes are stored as four separate arrays (`min_x[]`, `min_y[]`, `max_x[]`,
//! `max_y[]`). The tree is built exactly like the AoS version; only the layout
//! and search implementation differ.

use std::{collections::BinaryHeap, ops::ControlFlow};

use wide::{CmpGe, CmpLe, f64x4};

#[cfg(feature = "parallel")]
use crate::sort::encode_sort_parallel;
use crate::{
    builder::BuildConfig,
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::{Point, Rect},
    index::{SearchWorkspace, prefetch_read, upper_bound_level},
    neighbors::{NeighborNodeState, NeighborState, NeighborWorkspace, max_distance_squared},
    sort::{encode_sort_serial, hilbert_coord},
};

type Num = f64;

pub(crate) fn build_simd_index(config: BuildConfig, boxes: Vec<Rect>) -> SimdIndex {
    let node_size = config.node_size;
    let num_items = config.num_items;
    let mut level_bounds: Vec<usize> = Vec::new();
    let mut num_nodes = num_items;
    let mut n = num_items;
    level_bounds.push(n);
    if num_items > 0 {
        loop {
            n = (n as f64 / node_size as f64).ceil() as usize;
            num_nodes += n;
            level_bounds.push(num_nodes);
            if n == 1 {
                break;
            }
        }
    }

    if num_items == 0 {
        return SimdIndex {
            node_size,
            num_items,
            level_bounds,
            min_xs: Vec::new(),
            min_ys: Vec::new(),
            max_xs: Vec::new(),
            max_ys: Vec::new(),
            indices: Vec::new(),
        };
    }

    if num_items <= node_size {
        return build_single_node_soa(node_size, num_items, level_bounds, boxes);
    }

    let mut min_xs = vec![0.0f64; num_nodes];
    let mut min_ys = vec![0.0f64; num_nodes];
    let mut max_xs = vec![0.0f64; num_nodes];
    let mut max_ys = vec![0.0f64; num_nodes];
    let mut indices = vec![0usize; num_nodes];

    let (mut e_min_x, mut e_min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut e_max_x, mut e_max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for b in &boxes {
        e_min_x = e_min_x.min(b.min_x);
        e_min_y = e_min_y.min(b.min_y);
        e_max_x = e_max_x.max(b.max_x);
        e_max_y = e_max_y.max(b.max_y);
    }
    let scaled_width = u16::MAX as f64 / (e_max_x - e_min_x);
    let scaled_height = u16::MAX as f64 / (e_max_y - e_min_y);

    let sort_key = config.sort_key;
    let encode = |i: usize, b: &Rect| -> (u32, u32) {
        let hx = hilbert_coord(scaled_width, b.min_x, b.max_x, e_min_x);
        let hy = hilbert_coord(scaled_height, b.min_y, b.max_y, e_min_y);
        (sort_key.encode(hx, hy), i as u32)
    };

    #[cfg(feature = "parallel")]
    let use_parallel = config.parallel && num_items >= config.parallel_min_items;

    #[cfg(feature = "parallel")]
    let order: Vec<(u32, u32)> = if use_parallel {
        encode_sort_parallel(&boxes, &encode)
    } else {
        encode_sort_serial(&boxes, &encode, config.radix, config.radix_bits)
    };
    #[cfg(not(feature = "parallel"))]
    let order: Vec<(u32, u32)> =
        encode_sort_serial(&boxes, &encode, config.radix, config.radix_bits);

    for (slot, &(_, orig)) in order.iter().enumerate() {
        let b = boxes[orig as usize];
        min_xs[slot] = b.min_x;
        min_ys[slot] = b.min_y;
        max_xs[slot] = b.max_x;
        max_ys[slot] = b.max_y;
        indices[slot] = orig as usize;
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

    SimdIndex {
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

fn build_single_node_soa(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    boxes: Vec<Rect>,
) -> SimdIndex {
    let mut min_xs = Vec::with_capacity(num_items + 1);
    let mut min_ys = Vec::with_capacity(num_items + 1);
    let mut max_xs = Vec::with_capacity(num_items + 1);
    let mut max_ys = Vec::with_capacity(num_items + 1);
    let mut indices = Vec::with_capacity(num_items + 1);

    let (mut root_min_x, mut root_min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut root_max_x, mut root_max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (idx, b) in boxes.into_iter().enumerate() {
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

    SimdIndex {
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
/// Created through [`IndexBuilder::finish_simd`](crate::IndexBuilder::finish_simd).
/// It has the same public search and nearest-neighbor API as [`Index`](crate::Index),
/// but stores bounds in structure-of-arrays form for SIMD traversal.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{IndexBuilder, Rect};
///
/// let mut builder = IndexBuilder::new(1);
/// builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
///
/// let index = builder.finish_simd().unwrap();
/// assert_eq!(index.search(Rect::new(0.5, 0.5, 0.5, 0.5)), vec![0]);
/// ```
pub struct SimdIndex {
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    min_xs: Vec<Num>,
    min_ys: Vec<Num>,
    max_xs: Vec<Num>,
    max_ys: Vec<Num>,
    indices: Vec<usize>,
}

impl SimdIndex {
    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total bounds of indexed items, or `None` for an empty index.
    pub fn bounds(&self) -> Option<Rect> {
        if self.num_items == 0 {
            None
        } else {
            let last = self.min_xs.len() - 1;
            Some(Rect::new(
                self.min_xs[last],
                self.min_ys[last],
                self.max_xs[last],
                self.max_ys[last],
            ))
        }
    }

    /// Return the packed node size.
    #[doc(hidden)]
    pub fn node_size(&self) -> usize {
        self.node_size
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

    /// Return the indices of all items whose rectangles intersect `rect`.
    pub fn search(&self, rect: Rect) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(rect, &mut out);
        out
    }

    /// Return the indices of all items intersecting raw bounds.
    pub fn search_bounds(&self, min_x: Num, min_y: Num, max_x: Num, max_y: Num) -> Vec<usize> {
        self.search(Rect::new(min_x, min_y, max_x, max_y))
    }

    /// Search with a reusable result buffer.
    ///
    /// This automatically chooses the widest available SIMD implementation: AVX-512
    /// on supporting x86-64 CPUs, otherwise AVX2/SSE through `wide`.
    pub fn search_into(&self, rect: Rect, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_avx512(rect, out, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'a>(&self, rect: Rect, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.search_avx512(rect, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `rect`.
    pub fn any(&self, rect: Rect) -> bool {
        self.visit(rect, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    pub fn first(&self, rect: Rect) -> Option<usize> {
        match self.visit(rect, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Return up to `max_results` item indices nearest to `point`.
    pub fn neighbors(&self, point: Point, max_results: usize) -> Vec<usize> {
        self.neighbors_within(point, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of `point`.
    pub fn neighbors_within(
        &self,
        point: Point,
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
        point: Point,
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
        point: Point,
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
        point: Point,
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
    pub fn visit<B, F>(&self, rect: Rect, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_simd(rect, &mut stack, visitor)
    }

    fn collect_neighbors_with_queue(
        &self,
        point: Point,
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
        point: Point,
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
        point: Point,
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
    fn distance_squared_to(&self, pos: usize, point: Point) -> f64 {
        let dx = axis_distance(point.x, self.min_xs[pos], self.max_xs[pos]);
        let dy = axis_distance(point.y, self.min_ys[pos], self.max_ys[pos]);
        dx * dx + dy * dy
    }

    /// Same as [`visit`](SimdIndex::visit), but the traversal stack is reused by the caller.
    #[doc(hidden)]
    pub fn visit_simd<B, F>(&self, rect: Rect, stack: &mut Vec<usize>, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_simd_impl::<false, B, F>(rect, stack, visitor)
    }

    /// Experimental prefetch variant of [`visit_simd`](SimdIndex::visit_simd).
    #[doc(hidden)]
    pub fn visit_simd_prefetch<B, F>(
        &self,
        rect: Rect,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_simd_impl::<true, B, F>(rect, stack, visitor)
    }

    /// Element-by-element traversal (SoA layout, branchless `overlaps`).
    #[doc(hidden)]
    pub fn search_scalar(&self, rect: Rect, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
                let hit = (self.min_xs[pos] <= rect.max_x)
                    & (self.max_xs[pos] >= rect.min_x)
                    & (self.min_ys[pos] <= rect.max_y)
                    & (self.max_ys[pos] >= rect.min_y);
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
    pub fn search_simd(&self, rect: Rect, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        self.search_simd_impl::<false>(rect, out, stack);
    }

    /// AVX2/SSE path with prefetch for the next node from the stack.
    #[doc(hidden)]
    pub fn search_simd_prefetch(&self, rect: Rect, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        self.search_simd_impl::<true>(rect, out, stack);
    }

    fn search_simd_impl<const PREFETCH: bool>(
        &self,
        rect: Rect,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = f64x4::splat(rect.max_x);
        let qmnx_v = f64x4::splat(rect.min_x);
        let qmxy_v = f64x4::splat(rect.max_y);
        let qmny_v = f64x4::splat(rect.min_y);

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
                let hit = (self.min_xs[pos] <= rect.max_x)
                    & (self.max_xs[pos] >= rect.min_x)
                    & (self.min_ys[pos] <= rect.max_y)
                    & (self.max_ys[pos] >= rect.min_y);
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
        rect: Rect,
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
        let qmxx_v = f64x4::splat(rect.max_x);
        let qmnx_v = f64x4::splat(rect.min_x);
        let qmxy_v = f64x4::splat(rect.max_y);
        let qmny_v = f64x4::splat(rect.min_y);

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
                let hit = (self.min_xs[pos] <= rect.max_x)
                    & (self.max_xs[pos] >= rect.min_x)
                    & (self.min_ys[pos] <= rect.max_y)
                    & (self.max_ys[pos] >= rect.min_y);
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

    /// AVX-512 path, falling back to [`search_simd`](SimdIndex::search_simd).
    #[doc(hidden)]
    pub fn search_avx512(&self, rect: Rect, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: this branch is selected only after checking avx512f availability.
                unsafe { self.search_avx512_impl(rect, out, stack) };
                return;
            }
        }
        self.search_simd(rect, out, stack);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn search_avx512_impl(&self, rect: Rect, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        use std::arch::x86_64::*;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = _mm512_set1_pd(rect.max_x);
        let qmnx_v = _mm512_set1_pd(rect.min_x);
        let qmxy_v = _mm512_set1_pd(rect.max_y);
        let qmny_v = _mm512_set1_pd(rect.min_y);

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
                let hit = (self.min_xs[pos] <= rect.max_x)
                    & (self.max_xs[pos] >= rect.min_x)
                    & (self.min_ys[pos] <= rect.max_y)
                    & (self.max_ys[pos] >= rect.min_y);
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
