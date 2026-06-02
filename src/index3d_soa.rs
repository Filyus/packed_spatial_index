//! SoA 3D index variant with SIMD searches (available with the `simd` feature).
//!
//! Boxes are stored as six separate arrays (`min_x[]`, `min_y[]`, `min_z[]`,
//! `max_x[]`, `max_y[]`, `max_z[]`). The tree is built exactly like the AoS
//! [`Index3D`](crate::Index3D); only the layout and search implementation differ.
//! This mirrors [`SimdIndex2D`](crate::SimdIndex2D) with an added Z axis.

use std::{collections::BinaryHeap, ops::ControlFlow};

use wide::{CmpGe, CmpLe, f64x4};

use crate::{
    builder3d::BuildConfig3D,
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::{Bounds3D, Point3D},
    neighbors::{NeighborNodeState, NeighborState, NeighborWorkspace, max_distance_squared},
    sort3d::{SortKey3DContext, encode_sort_by_key_3d},
    traversal::{SearchWorkspace, upper_bound_level},
    tree::{TreeLayout, compute_tree_layout},
};

type Num = f64;

pub(crate) fn build_simd_index_3d(config: BuildConfig3D, boxes: Vec<Bounds3D>) -> SimdIndex3D {
    let node_size = config.node_size;
    let num_items = config.num_items;
    let TreeLayout {
        level_bounds,
        num_nodes,
    } = compute_tree_layout(num_items, node_size);

    if num_items == 0 {
        return SimdIndex3D::empty(node_size, num_items, level_bounds);
    }

    if num_items <= node_size {
        return build_single_node_soa_3d(node_size, num_items, level_bounds, boxes);
    }

    let mut min_xs = vec![0.0f64; num_nodes];
    let mut min_ys = vec![0.0f64; num_nodes];
    let mut min_zs = vec![0.0f64; num_nodes];
    let mut max_xs = vec![0.0f64; num_nodes];
    let mut max_ys = vec![0.0f64; num_nodes];
    let mut max_zs = vec![0.0f64; num_nodes];
    let mut indices = vec![0usize; num_nodes];

    let extent = extent_3d(&boxes);

    #[cfg(feature = "parallel")]
    let use_parallel = config.parallel && num_items >= config.parallel_min_items;

    let context = SortKey3DContext::new(extent, config.radix, config.radix_bits);
    #[cfg(feature = "parallel")]
    let context = context.parallel(use_parallel);
    let order = encode_sort_by_key_3d(&boxes, config.sort_key, context);

    for (slot, &(_, orig)) in order.iter().enumerate() {
        let b = boxes[orig];
        min_xs[slot] = b.min_x;
        min_ys[slot] = b.min_y;
        min_zs[slot] = b.min_z;
        max_xs[slot] = b.max_x;
        max_ys[slot] = b.max_y;
        max_zs[slot] = b.max_z;
        indices[slot] = orig;
    }

    let mut read_pos = 0usize;
    let mut write_pos = num_items;
    for &level_end in &level_bounds[0..level_bounds.len() - 1] {
        while read_pos < level_end {
            let node_index = read_pos;
            let (mut nmnx, mut nmny, mut nmnz) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let (mut nmxx, mut nmxy, mut nmxz) =
                (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            let mut j = 0;
            while j < node_size && read_pos < level_end {
                nmnx = nmnx.min(min_xs[read_pos]);
                nmny = nmny.min(min_ys[read_pos]);
                nmnz = nmnz.min(min_zs[read_pos]);
                nmxx = nmxx.max(max_xs[read_pos]);
                nmxy = nmxy.max(max_ys[read_pos]);
                nmxz = nmxz.max(max_zs[read_pos]);
                read_pos += 1;
                j += 1;
            }
            min_xs[write_pos] = nmnx;
            min_ys[write_pos] = nmny;
            min_zs[write_pos] = nmnz;
            max_xs[write_pos] = nmxx;
            max_ys[write_pos] = nmxy;
            max_zs[write_pos] = nmxz;
            indices[write_pos] = node_index;
            write_pos += 1;
        }
    }

    SimdIndex3D {
        node_size,
        num_items,
        level_bounds,
        min_xs,
        min_ys,
        min_zs,
        max_xs,
        max_ys,
        max_zs,
        indices,
    }
}

fn build_single_node_soa_3d(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    boxes: Vec<Bounds3D>,
) -> SimdIndex3D {
    let mut min_xs = Vec::with_capacity(num_items + 1);
    let mut min_ys = Vec::with_capacity(num_items + 1);
    let mut min_zs = Vec::with_capacity(num_items + 1);
    let mut max_xs = Vec::with_capacity(num_items + 1);
    let mut max_ys = Vec::with_capacity(num_items + 1);
    let mut max_zs = Vec::with_capacity(num_items + 1);
    let mut indices = Vec::with_capacity(num_items + 1);

    let (mut rmnx, mut rmny, mut rmnz) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let (mut rmxx, mut rmxy, mut rmxz) = (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (idx, b) in boxes.into_iter().enumerate() {
        min_xs.push(b.min_x);
        min_ys.push(b.min_y);
        min_zs.push(b.min_z);
        max_xs.push(b.max_x);
        max_ys.push(b.max_y);
        max_zs.push(b.max_z);
        indices.push(idx);

        rmnx = rmnx.min(b.min_x);
        rmny = rmny.min(b.min_y);
        rmnz = rmnz.min(b.min_z);
        rmxx = rmxx.max(b.max_x);
        rmxy = rmxy.max(b.max_y);
        rmxz = rmxz.max(b.max_z);
    }

    min_xs.push(rmnx);
    min_ys.push(rmny);
    min_zs.push(rmnz);
    max_xs.push(rmxx);
    max_ys.push(rmxy);
    max_zs.push(rmxz);
    indices.push(0);

    SimdIndex3D {
        node_size,
        num_items,
        level_bounds,
        min_xs,
        min_ys,
        min_zs,
        max_xs,
        max_ys,
        max_zs,
        indices,
    }
}

fn extent_3d(boxes: &[Bounds3D]) -> Bounds3D {
    let (mut mnx, mut mny, mut mnz) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let (mut mxx, mut mxy, mut mxz) = (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for b in boxes {
        mnx = mnx.min(b.min_x);
        mny = mny.min(b.min_y);
        mnz = mnz.min(b.min_z);
        mxx = mxx.max(b.max_x);
        mxy = mxy.max(b.max_y);
        mxz = mxz.max(b.max_z);
    }
    Bounds3D::new(mnx, mny, mnz, mxx, mxy, mxz)
}

/// Finished read-only SIMD 3D index.
///
/// Created through [`Index3DBuilder::finish_simd`](crate::Index3DBuilder::finish_simd).
/// It has the same public search and nearest-neighbor API as [`Index3D`](crate::Index3D),
/// but stores bounds in structure-of-arrays form for SIMD traversal.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index3DBuilder, Bounds3D};
///
/// let mut builder = Index3DBuilder::new(1);
/// builder.add(Bounds3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
///
/// let index = builder.finish_simd().unwrap();
/// assert_eq!(index.search(Bounds3D::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)), vec![0]);
/// ```
pub struct SimdIndex3D {
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    min_xs: Vec<Num>,
    min_ys: Vec<Num>,
    min_zs: Vec<Num>,
    max_xs: Vec<Num>,
    max_ys: Vec<Num>,
    max_zs: Vec<Num>,
    indices: Vec<usize>,
}

impl SimdIndex3D {
    fn empty(node_size: usize, num_items: usize, level_bounds: Vec<usize>) -> Self {
        SimdIndex3D {
            node_size,
            num_items,
            level_bounds,
            min_xs: Vec::new(),
            min_ys: Vec::new(),
            min_zs: Vec::new(),
            max_xs: Vec::new(),
            max_ys: Vec::new(),
            max_zs: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total extent of indexed items, or `None` for an empty index.
    pub fn extent(&self) -> Option<Bounds3D> {
        if self.num_items == 0 {
            None
        } else {
            let last = self.min_xs.len() - 1;
            Some(Bounds3D::new(
                self.min_xs[last],
                self.min_ys[last],
                self.min_zs[last],
                self.max_xs[last],
                self.max_ys[last],
                self.max_zs[last],
            ))
        }
    }

    /// Return the packed node size used by this index.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Return the indices of all items whose bounds intersect `bounds`.
    pub fn search(&self, bounds: Bounds3D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(bounds, &mut out);
        out
    }

    /// Search with a reusable result buffer.
    ///
    /// This automatically chooses the widest available SIMD implementation: AVX-512
    /// on supporting x86-64 CPUs, otherwise AVX2/SSE through `wide`.
    pub fn search_into(&self, bounds: Bounds3D, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_avx512(bounds, out, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'a>(
        &self,
        bounds: Bounds3D,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize] {
        self.search_avx512(bounds, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `bounds`.
    pub fn any(&self, bounds: Bounds3D) -> bool {
        self.visit(bounds, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    pub fn first(&self, bounds: Bounds3D) -> Option<usize> {
        match self.visit(bounds, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Return up to `max_results` item indices nearest to `point`.
    pub fn neighbors(&self, point: Point3D, max_results: usize) -> Vec<usize> {
        self.neighbors_within(point, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of `point`.
    pub fn neighbors_within(
        &self,
        point: Point3D,
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
        point: Point3D,
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
        point: Point3D,
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
        point: Point3D,
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
    pub fn visit<B, F>(&self, bounds: Bounds3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_simd(bounds, &mut stack, visitor)
    }

    fn collect_neighbors_with_queue(
        &self,
        point: Point3D,
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
        point: Point3D,
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
        point: Point3D,
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
    fn distance_squared_to(&self, pos: usize, point: Point3D) -> f64 {
        let dx = axis_distance(point.x, self.min_xs[pos], self.max_xs[pos]);
        let dy = axis_distance(point.y, self.min_ys[pos], self.max_ys[pos]);
        let dz = axis_distance(point.z, self.min_zs[pos], self.max_zs[pos]);
        dx * dx + dy * dy + dz * dz
    }

    /// Same as [`visit`](SimdIndex3D::visit), but the traversal stack is reused by the caller.
    #[doc(hidden)]
    pub fn visit_simd<B, F>(
        &self,
        bounds: Bounds3D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_simd_impl::<B, F>(bounds, stack, visitor)
    }

    /// Element-by-element traversal (SoA layout, branchless `overlaps`).
    #[doc(hidden)]
    pub fn search_scalar(&self, bounds: Bounds3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
                if !self.hit_scalar(pos, bounds) {
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

    #[inline]
    fn hit_scalar(&self, pos: usize, bounds: Bounds3D) -> bool {
        (self.min_xs[pos] <= bounds.max_x)
            & (self.max_xs[pos] >= bounds.min_x)
            & (self.min_ys[pos] <= bounds.max_y)
            & (self.max_ys[pos] >= bounds.min_y)
            & (self.min_zs[pos] <= bounds.max_z)
            & (self.max_zs[pos] >= bounds.min_z)
    }

    /// AVX2/SSE path through `wide::f64x4`.
    #[doc(hidden)]
    pub fn search_simd(&self, bounds: Bounds3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = f64x4::splat(bounds.max_x);
        let qmnx_v = f64x4::splat(bounds.min_x);
        let qmxy_v = f64x4::splat(bounds.max_y);
        let qmny_v = f64x4::splat(bounds.min_y);
        let qmxz_v = f64x4::splat(bounds.max_z);
        let qmnz_v = f64x4::splat(bounds.min_z);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 4 <= end {
                let mask = load4(&self.min_xs, pos).simd_le(qmxx_v)
                    & load4(&self.max_xs, pos).simd_ge(qmnx_v)
                    & load4(&self.min_ys, pos).simd_le(qmxy_v)
                    & load4(&self.max_ys, pos).simd_ge(qmny_v)
                    & load4(&self.min_zs, pos).simd_le(qmxz_v)
                    & load4(&self.max_zs, pos).simd_ge(qmnz_v);
                let bits = mask.to_bitmask();
                if bits != 0 {
                    for k in 0..4 {
                        if bits & (1 << k) != 0 {
                            let index = self.indices[pos + k];
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
                if self.hit_scalar(pos, bounds) {
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

    fn visit_simd_impl<B, F>(
        &self,
        bounds: Bounds3D,
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
        let qmxx_v = f64x4::splat(bounds.max_x);
        let qmnx_v = f64x4::splat(bounds.min_x);
        let qmxy_v = f64x4::splat(bounds.max_y);
        let qmny_v = f64x4::splat(bounds.min_y);
        let qmxz_v = f64x4::splat(bounds.max_z);
        let qmnz_v = f64x4::splat(bounds.min_z);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 4 <= end {
                let mask = load4(&self.min_xs, pos).simd_le(qmxx_v)
                    & load4(&self.max_xs, pos).simd_ge(qmnx_v)
                    & load4(&self.min_ys, pos).simd_le(qmxy_v)
                    & load4(&self.max_ys, pos).simd_ge(qmny_v)
                    & load4(&self.min_zs, pos).simd_le(qmxz_v)
                    & load4(&self.max_zs, pos).simd_ge(qmnz_v);
                let bits = mask.to_bitmask();
                if bits != 0 {
                    for k in 0..4 {
                        if bits & (1 << k) != 0 {
                            let index = self.indices[pos + k];
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
                if self.hit_scalar(pos, bounds) {
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

    /// AVX-512 path, falling back to [`search_simd`](SimdIndex3D::search_simd).
    #[doc(hidden)]
    pub fn search_avx512(&self, bounds: Bounds3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: this branch is selected only after checking avx512f availability.
                unsafe { self.search_avx512_impl(bounds, out, stack) };
                return;
            }
        }
        self.search_simd(bounds, out, stack);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn search_avx512_impl(
        &self,
        bounds: Bounds3D,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        use std::arch::x86_64::*;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = _mm512_set1_pd(bounds.max_x);
        let qmnx_v = _mm512_set1_pd(bounds.min_x);
        let qmxy_v = _mm512_set1_pd(bounds.max_y);
        let qmny_v = _mm512_set1_pd(bounds.min_y);
        let qmxz_v = _mm512_set1_pd(bounds.max_z);
        let qmnz_v = _mm512_set1_pd(bounds.min_z);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end`, and `end` is bounded by the array length.
                let (mnx, mxx, mny, mxy, mnz, mxz) = unsafe {
                    (
                        _mm512_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_zs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_zs.as_ptr().add(pos)),
                    )
                };
                let m1 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mnx, qmxx_v);
                let m2 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mxx, qmnx_v);
                let m3 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mny, qmxy_v);
                let m4 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mxy, qmny_v);
                let m5 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mnz, qmxz_v);
                let m6 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mxz, qmnz_v);
                let mut bits: u8 = m1 & m2 & m3 & m4 & m5 & m6;
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
                if self.hit_scalar(pos, bounds) {
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
