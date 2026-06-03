//! SoA 3D index variant with SIMD searches (available with the `simd` feature).
//!
//! items are stored as six separate arrays (`min_x[]`, `min_y[]`, `min_z[]`,
//! `max_x[]`, `max_y[]`, `max_z[]`). The tree is built exactly like the AoS
//! [`Index3D`](crate::Index3D); only the layout and search implementation differ.
//! This mirrors [`SimdIndex2D`](crate::SimdIndex2D) with an added Z axis.

use std::{collections::BinaryHeap, ops::ControlFlow};

use wide::{CmpGe, CmpLe, f64x4};

use crate::{
    build::BuildError,
    builder3d::BuildConfig3D,
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::{Box3D, Point3D},
    neighbors::{NeighborNodeState, NeighborState, NeighborWorkspace, max_distance_squared},
    persistence::{
        ByteWriter, LoadError, parse_index3d_bytes, read_f64_le_unchecked, read_u64_le_unchecked,
        serialized_len_3d,
    },
    sort3d::{SortKey3DContext, encode_sort_by_key_3d},
    traversal::{SearchWorkspace, upper_bound_level},
    tree::{TreeLayout, try_compute_tree_layout},
};

type Num = f64;

pub(crate) fn build_simd_index_3d(
    config: BuildConfig3D,
    items: Vec<Box3D>,
) -> Result<SimdIndex3D, BuildError> {
    let node_size = config.node_size;
    let num_items = config.num_items;
    let TreeLayout {
        level_bounds,
        num_nodes,
    } = try_compute_tree_layout(num_items, node_size)?;

    if num_items == 0 {
        return Ok(SimdIndex3D::empty(node_size, num_items, level_bounds));
    }

    if num_items <= node_size {
        return Ok(build_single_node_soa_3d(
            node_size,
            num_items,
            level_bounds,
            items,
        ));
    }

    let mut min_xs = vec![0.0f64; num_nodes];
    let mut min_ys = vec![0.0f64; num_nodes];
    let mut min_zs = vec![0.0f64; num_nodes];
    let mut max_xs = vec![0.0f64; num_nodes];
    let mut max_ys = vec![0.0f64; num_nodes];
    let mut max_zs = vec![0.0f64; num_nodes];
    let mut indices = vec![0usize; num_nodes];

    let extent = extent_3d(&items);

    #[cfg(feature = "parallel")]
    let use_parallel = config.parallel && num_items >= config.parallel_min_items;

    let context = SortKey3DContext::new(extent, config.radix, config.radix_bits);
    #[cfg(feature = "parallel")]
    let context = context.parallel(use_parallel);
    let order = encode_sort_by_key_3d(&items, config.sort_key, context);

    #[cfg(feature = "parallel")]
    let scattered_in_parallel = if use_parallel {
        reorder_parallel_soa_3d(
            &mut min_xs[..num_items],
            &mut min_ys[..num_items],
            &mut min_zs[..num_items],
            &mut max_xs[..num_items],
            &mut max_ys[..num_items],
            &mut max_zs[..num_items],
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
            let b = items[orig];
            min_xs[slot] = b.min_x;
            min_ys[slot] = b.min_y;
            min_zs[slot] = b.min_z;
            max_xs[slot] = b.max_x;
            max_ys[slot] = b.max_y;
            max_zs[slot] = b.max_z;
            indices[slot] = orig;
        }
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

    Ok(SimdIndex3D {
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
    })
}

fn build_single_node_soa_3d(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    items: Vec<Box3D>,
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
    for (idx, b) in items.into_iter().enumerate() {
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

fn extent_3d(items: &[Box3D]) -> Box3D {
    let (mut mnx, mut mny, mut mnz) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let (mut mxx, mut mxy, mut mxz) = (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for b in items {
        mnx = mnx.min(b.min_x);
        mny = mny.min(b.min_y);
        mnz = mnz.min(b.min_z);
        mxx = mxx.max(b.max_x);
        mxy = mxy.max(b.max_y);
        mxz = mxz.max(b.max_z);
    }
    Box3D::new(mnx, mny, mnz, mxx, mxy, mxz)
}

/// Finished read-only SIMD 3D index.
///
/// Created through [`Index3DBuilder::finish_simd`](crate::Index3DBuilder::finish_simd).
/// It has the same public search and nearest-neighbor API as [`Index3D`](crate::Index3D),
/// but stores box coordinates in structure-of-arrays form for SIMD traversal.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index3DBuilder, Box3D};
///
/// let mut builder = Index3DBuilder::new(1);
/// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
///
/// let index = builder.finish_simd().unwrap();
/// assert_eq!(index.search(Box3D::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)), vec![0]);
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
    pub fn extent(&self) -> Option<Box3D> {
        if self.num_items == 0 {
            None
        } else {
            let last = self.min_xs.len() - 1;
            Some(Box3D::new(
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

    /// Serialize into the stable little-endian `PSINDEX` 3D format.
    ///
    /// The output is byte-identical to [`Index3D::to_bytes`](crate::Index3D::to_bytes)
    /// for the same items, so a `SimdIndex3D` and an `Index3D` are interchangeable on
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
        let len = serialized_len_3d(level_count, num_nodes).expect("serialized index is too large");
        let mut bytes = ByteWriter::new(out, len);
        bytes.write_magic();
        bytes.write_format_version();
        bytes.write_header_len();
        bytes.write_3d_flags();
        bytes.write_u64(self.node_size as u64);
        bytes.write_u64(self.num_items as u64);
        bytes.write_u64(num_nodes as u64);
        bytes.write_u64(level_count as u64);
        bytes.write_usize_slice_as_u64(&self.level_bounds);
        bytes.write_soa_boxes_3d(
            &self.min_xs,
            &self.min_ys,
            &self.min_zs,
            &self.max_xs,
            &self.max_ys,
            &self.max_zs,
        );
        bytes.write_usize_slice_as_u64(&self.indices);
        bytes.finish();
    }

    /// Load a SIMD 3D index from bytes produced by [`to_bytes`](Self::to_bytes) or by
    /// [`Index3D::to_bytes`](crate::Index3D::to_bytes); the AoS box records are
    /// scattered into the structure-of-arrays columns.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let parsed = parse_index3d_bytes(bytes)?;
        let num_nodes = parsed.num_nodes;

        let mut level_bounds = Vec::with_capacity(parsed.level_count);
        for i in 0..parsed.level_count {
            level_bounds.push(read_u64_le_unchecked(parsed.level_bounds, i * 8) as usize);
        }

        let mut min_xs = Vec::with_capacity(num_nodes);
        let mut min_ys = Vec::with_capacity(num_nodes);
        let mut min_zs = Vec::with_capacity(num_nodes);
        let mut max_xs = Vec::with_capacity(num_nodes);
        let mut max_ys = Vec::with_capacity(num_nodes);
        let mut max_zs = Vec::with_capacity(num_nodes);
        let mut indices = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let off = i * 48; // six f64 per 3D box record
            min_xs.push(read_f64_le_unchecked(parsed.entries, off));
            min_ys.push(read_f64_le_unchecked(parsed.entries, off + 8));
            min_zs.push(read_f64_le_unchecked(parsed.entries, off + 16));
            max_xs.push(read_f64_le_unchecked(parsed.entries, off + 24));
            max_ys.push(read_f64_le_unchecked(parsed.entries, off + 32));
            max_zs.push(read_f64_le_unchecked(parsed.entries, off + 40));
            indices.push(read_u64_le_unchecked(parsed.indices, i * 8) as usize);
        }

        Ok(SimdIndex3D {
            node_size: parsed.node_size,
            num_items: parsed.num_items,
            level_bounds,
            min_xs,
            min_ys,
            min_zs,
            max_xs,
            max_ys,
            max_zs,
            indices,
        })
    }

    /// Return the indices of all items whose boxes intersect `query`.
    pub fn search(&self, query: Box3D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(query, &mut out);
        out
    }

    /// Search with a reusable result buffer.
    ///
    /// This automatically chooses the widest available SIMD implementation: AVX-512
    /// on supporting x86-64 CPUs, otherwise AVX2/SSE through `wide`.
    pub fn search_into(&self, query: Box3D, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_avx512(query, out, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'a>(&self, query: Box3D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.search_avx512(query, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `query`.
    pub fn any(&self, query: Box3D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    pub fn first(&self, query: Box3D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
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
    pub fn visit<B, F>(&self, query: Box3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_avx512(query, &mut stack, visitor)
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
        query: Box3D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_simd_impl::<B, F>(query, stack, visitor)
    }

    /// AVX-512 visitor path, falling back to [`visit_simd`](SimdIndex3D::visit_simd).
    #[doc(hidden)]
    pub fn visit_avx512<B, F>(
        &self,
        query: Box3D,
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
    pub fn search_scalar(&self, query: Box3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
                if !self.hit_scalar(pos, query) {
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
    fn hit_scalar(&self, pos: usize, query: Box3D) -> bool {
        (self.min_xs[pos] <= query.max_x)
            & (self.max_xs[pos] >= query.min_x)
            & (self.min_ys[pos] <= query.max_y)
            & (self.max_ys[pos] >= query.min_y)
            & (self.min_zs[pos] <= query.max_z)
            & (self.max_zs[pos] >= query.min_z)
    }

    /// AVX2/SSE path through `wide::f64x4`.
    #[doc(hidden)]
    pub fn search_simd(&self, query: Box3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = f64x4::splat(query.max_x);
        let qmnx_v = f64x4::splat(query.min_x);
        let qmxy_v = f64x4::splat(query.max_y);
        let qmny_v = f64x4::splat(query.min_y);
        let qmxz_v = f64x4::splat(query.max_z);
        let qmnz_v = f64x4::splat(query.min_z);

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
                if self.hit_scalar(pos, query) {
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
        query: Box3D,
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
        let qmxz_v = f64x4::splat(query.max_z);
        let qmnz_v = f64x4::splat(query.min_z);

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
                if self.hit_scalar(pos, query) {
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

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn visit_avx512_impl<B, F>(
        &self,
        query: Box3D,
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
        let qmxz_v = _mm512_set1_pd(query.max_z);
        let qmnz_v = _mm512_set1_pd(query.min_z);

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
                if self.hit_scalar(pos, query) {
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
    pub fn search_avx512(&self, query: Box3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
        query: Box3D,
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
        let qmxz_v = _mm512_set1_pd(query.max_z);
        let qmnz_v = _mm512_set1_pd(query.min_z);

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
                if self.hit_scalar(pos, query) {
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
fn reorder_parallel_soa_3d(
    min_xs: &mut [f64],
    min_ys: &mut [f64],
    min_zs: &mut [f64],
    max_xs: &mut [f64],
    max_ys: &mut [f64],
    max_zs: &mut [f64],
    indices: &mut [usize],
    order: &[(u64, usize)],
    items: &[Box3D],
) {
    use rayon::prelude::*;

    min_xs
        .par_iter_mut()
        .zip(min_ys.par_iter_mut())
        .zip(min_zs.par_iter_mut())
        .zip(max_xs.par_iter_mut())
        .zip(max_ys.par_iter_mut())
        .zip(max_zs.par_iter_mut())
        .zip(indices.par_iter_mut())
        .zip(order.par_iter())
        .for_each(
            |(((((((mnx, mny), mnz), mxx), mxy), mxz), idx), &(_, orig))| {
                let b = items[orig];
                *mnx = b.min_x;
                *mny = b.min_y;
                *mnz = b.min_z;
                *mxx = b.max_x;
                *mxy = b.max_y;
                *mxz = b.max_z;
                *idx = orig;
            },
        );
}

/// Byte size of one persisted 3D box record (`[min_x, min_y, min_z, max_x, max_y, max_z]`).
const RECORD_3D: usize = 48;

/// Assemble one coordinate column for four consecutive 3D box records into a SIMD
/// vector. The four records are contiguous (192 bytes), so the strided reads stay
/// within a handful of cache lines.
#[inline]
fn lane4_3d(entries: &[u8], base: usize, field: usize) -> f64x4 {
    let o = base + field;
    f64x4::from([
        read_f64_le_unchecked(entries, o),
        read_f64_le_unchecked(entries, o + RECORD_3D),
        read_f64_le_unchecked(entries, o + 2 * RECORD_3D),
        read_f64_le_unchecked(entries, o + 3 * RECORD_3D),
    ])
}

/// Zero-copy SIMD view over bytes produced by [`SimdIndex3D::to_bytes`] or
/// [`Index3D::to_bytes`](crate::Index3D::to_bytes).
///
/// Like [`Index3DView`](crate::Index3DView) it borrows the buffer without
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
/// use packed_spatial_index::{Index3DBuilder, SimdIndex3DView, Box3D};
///
/// let mut builder = Index3DBuilder::new(1);
/// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
/// let bytes = builder.finish_simd().unwrap().to_bytes();
///
/// let view = SimdIndex3DView::from_bytes(&bytes)?;
/// assert_eq!(view.search(Box3D::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)), vec![0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SimdIndex3DView<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    level_bounds: &'a [u8],
    entries: &'a [u8],
    indices: &'a [u8],
}

impl<'a> SimdIndex3DView<'a> {
    /// Borrow a zero-copy view over the canonical `PSINDEX` 3D bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let parsed = parse_index3d_bytes(bytes)?;
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
    pub fn extent(&self) -> Option<Box3D> {
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
    fn box_at(&self, pos: usize) -> Box3D {
        let b = pos * RECORD_3D;
        Box3D::new(
            read_f64_le_unchecked(self.entries, b),
            read_f64_le_unchecked(self.entries, b + 8),
            read_f64_le_unchecked(self.entries, b + 16),
            read_f64_le_unchecked(self.entries, b + 24),
            read_f64_le_unchecked(self.entries, b + 32),
            read_f64_le_unchecked(self.entries, b + 40),
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
    pub fn search(&self, query: Box3D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(query, &mut out);
        out
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, query: Box3D, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        out.clear();
        let _: ControlFlow<()> = self.try_visit(query, &mut stack, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'b>(&self, query: Box3D, workspace: &'b mut SearchWorkspace) -> &'b [usize] {
        workspace.results.clear();
        let results = &mut workspace.results;
        let _: ControlFlow<()> = self.try_visit(query, &mut workspace.stack, |index| {
            results.push(index);
            ControlFlow::Continue(())
        });
        &workspace.results
    }

    /// Return `true` if at least one item intersects `query`.
    pub fn any(&self, query: Box3D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    pub fn first(&self, query: Box3D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.try_visit(query, &mut stack, visitor)
    }

    fn try_visit<B, F>(
        &self,
        query: Box3D,
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
        let qmxz = f64x4::splat(query.max_z);
        let qmnz = f64x4::splat(query.min_z);

        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 4 <= end {
                let base = pos * RECORD_3D;
                let mask = lane4_3d(self.entries, base, 0).simd_le(qmxx)
                    & lane4_3d(self.entries, base, 24).simd_ge(qmnx)
                    & lane4_3d(self.entries, base, 8).simd_le(qmxy)
                    & lane4_3d(self.entries, base, 32).simd_ge(qmny)
                    & lane4_3d(self.entries, base, 16).simd_le(qmxz)
                    & lane4_3d(self.entries, base, 40).simd_ge(qmnz);
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
    fn distance_squared_to(&self, pos: usize, point: Point3D) -> f64 {
        let b = self.box_at(pos);
        let dx = axis_distance(point.x, b.min_x, b.max_x);
        let dy = axis_distance(point.y, b.min_y, b.max_y);
        let dz = axis_distance(point.z, b.min_z, b.max_z);
        dx * dx + dy * dy + dz * dz
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
    pub fn neighbors_with<'b>(
        &self,
        point: Point3D,
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
