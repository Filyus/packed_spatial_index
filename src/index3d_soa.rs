//! SoA 3D index variant with SIMD searches (available with the `simd` feature).
//!
//! items are stored as six separate arrays (`min_x[]`, `min_y[]`, `min_z[]`,
//! `max_x[]`, `max_y[]`, `max_z[]`). The tree is built exactly like the AoS
//! [`Index3D`](crate::Index3D); only the layout and search implementation differ.
//! This mirrors [`SimdIndex2D`](crate::SimdIndex2D) with an added Z axis.

use std::{collections::BinaryHeap, ops::ControlFlow};

use wide::f64x4;

#[cfg(target_arch = "x86_64")]
use crate::leftpack::leftpack4;
use crate::{
    build::BuildError,
    builder3d::BuildConfig3D,
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::{Box3D, Point3D},
    join::{join_core, self_join_core},
    neighbors::{NeighborNodeState, NeighborQuery3D, NeighborState, NeighborWorkspace, best_first},
    persistence::{
        ByteWriter, CHUNK_ENTRY_LEN, LoadError, SUPERBLOCK_LEN, TAG_TREE, TREE_DESC_LEN,
        parse_index, plan_container, read_f64_le_unchecked, read_u64_le_unchecked,
    },
    ray::Ray3D,
    sort3d::{SortKey3DContext, encode_sort_by_key_3d},
    traversal::{SearchWorkspace, upper_bound_level},
    tree::{TreeLayout, try_compute_tree_layout},
    tree_access::TreeAccess,
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
        let num_nodes = self.min_xs.len();
        let tree_len = TREE_DESC_LEN + num_nodes * 48 + num_nodes * 8;
        let (total, off) = plan_container(&[tree_len]).expect("serialized index is too large");
        let mut bytes = ByteWriter::new(out, total);
        bytes.write_superblock(1);
        bytes.write_chunk_entry(&TAG_TREE, true, off[0], tree_len);
        bytes.write_zeros(off[0] - (SUPERBLOCK_LEN + CHUNK_ENTRY_LEN));
        bytes.write_tree_desc(3, 8, false, self.num_items, self.node_size);
        bytes.write_soa_boxes_3d(
            &self.min_xs,
            &self.min_ys,
            &self.min_zs,
            &self.max_xs,
            &self.max_ys,
            &self.max_zs,
        );
        bytes.write_usize_slice_as_u64(&self.indices);
        bytes.write_zeros(total - (off[0] + tree_len));
        bytes.finish();
    }

    /// Load a SIMD 3D index from bytes produced by [`to_bytes`](Self::to_bytes) or by
    /// [`Index3D::to_bytes`](crate::Index3D::to_bytes); the AoS box records are
    /// scattered into the structure-of-arrays columns.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 3, 8)?;
        if payload.is_some() {
            return Err(LoadError::UnsupportedVersion);
        }
        let num_nodes = parsed.num_nodes;
        let level_bounds = parsed.level_bounds;

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
    /// This automatically dispatches to the widest available kernel at runtime:
    /// AVX-512 (`VPCOMPRESSQ` collection), then an explicit AVX2 tier (left-pack
    /// collection), then the SSE2 `wide` fallback.
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
            if let Some(index) =
                self.nearest_one_with_queue(NeighborQuery3D::Point(point), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Point(point),
            max_results,
            max_distance,
            results,
            &mut queue,
        );
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
            if let Some(index) = self.nearest_one_with_queue(
                NeighborQuery3D::Point(point),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Point(point),
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
        self.visit_neighbors_with_queue(
            NeighborQuery3D::Point(point),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    /// Return up to `max_results` item indices nearest to the box `query`.
    /// See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box(&self, query: Box3D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box_within(
        &self,
        query: Box3D,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        self.neighbors_of_box_into(query, max_results, max_distance, &mut results);
        results
    }

    /// Box-query nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_of_box_into(
        &self,
        query: Box3D,
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
            if let Some(index) =
                self.nearest_one_with_queue(NeighborQuery3D::Box(query), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            results,
            &mut queue,
        );
    }

    /// Box-query nearest-neighbor search with reusable result and
    /// priority-queue buffers.
    pub fn neighbors_of_box_with<'na>(
        &self,
        query: Box3D,
        max_results: usize,
        max_distance: f64,
        workspace: &'na mut NeighborWorkspace,
    ) -> &'na [usize] {
        workspace.results.clear();
        if max_results == 0 {
            workspace.queue.clear();
            workspace.node_queue.clear();
            return &workspace.results;
        }
        if max_results == 1 {
            workspace.queue.clear();
            if let Some(index) = self.nearest_one_with_queue(
                NeighborQuery3D::Box(query),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
        );
        &workspace.results
    }

    /// Visit items in nondecreasing box-to-box distance order from `query`.
    /// See [`Index2D::visit_neighbors_of_box`](crate::Index2D::visit_neighbors_of_box).
    pub fn visit_neighbors_of_box<B, F>(
        &self,
        query: Box3D,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.visit_neighbors_with_queue(
            NeighborQuery3D::Box(query),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_avx512(query, &mut stack, visitor)
    }

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`. See [`Index2D::join`](crate::Index2D::join).
    pub fn join(&self, other: &SimdIndex3D) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other`. See
    /// [`Index2D::join_with`](crate::Index2D::join_with).
    pub fn join_with<B, F>(&self, other: &SimdIndex3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, visitor)
    }

    /// Return every unordered pair of distinct intersecting items within this
    /// index, each pair exactly once. See
    /// [`Index2D::self_join`](crate::Index2D::self_join).
    pub fn self_join(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.self_join_with(|i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every unordered pair of distinct intersecting items within this
    /// index. See [`Index2D::self_join_with`](crate::Index2D::self_join_with).
    pub fn self_join_with<B, F>(&self, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, visitor)
    }

    fn collect_neighbors_with_queue(
        &self,
        query: NeighborQuery3D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        queue: &mut BinaryHeap<NeighborState>,
    ) {
        best_first::collect_neighbors(
            self.min_xs.len(),
            self.num_items,
            self.node_size,
            |n| self.level_bounds[upper_bound_level(&self.level_bounds, n)],
            |p| self.indices[p],
            max_results,
            max_distance,
            |pos| self.distance_squared_at(pos, query),
            results,
            queue,
        );
    }

    fn nearest_one_with_queue(
        &self,
        query: NeighborQuery3D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        best_first::nearest_one(
            self.min_xs.len(),
            self.num_items,
            self.node_size,
            |n| self.level_bounds[upper_bound_level(&self.level_bounds, n)],
            |p| self.indices[p],
            max_distance,
            |pos| self.distance_squared_at(pos, query),
            queue,
        )
    }

    fn visit_neighbors_with_queue<B, F>(
        &self,
        query: NeighborQuery3D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborState>,
        visitor: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        best_first::visit_neighbors(
            self.min_xs.len(),
            self.num_items,
            self.node_size,
            |n| self.level_bounds[upper_bound_level(&self.level_bounds, n)],
            |p| self.indices[p],
            max_distance,
            |pos| self.distance_squared_at(pos, query),
            queue,
            visitor,
        )
    }

    #[inline]
    fn distance_squared_at(&self, pos: usize, query: NeighborQuery3D) -> f64 {
        query.distance_squared_to(Box3D::new(
            self.min_xs[pos],
            self.min_ys[pos],
            self.min_zs[pos],
            self.max_xs[pos],
            self.max_ys[pos],
            self.max_zs[pos],
        ))
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
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: selected only after checking avx2 availability.
                return unsafe { self.visit_avx2_impl::<B, F>(query, stack, visitor) };
            }
        }
        self.visit_simd(query, stack, visitor)
    }

    /// Force the AVX2 visit path (doc-hidden; for benchmarks/tests).
    #[doc(hidden)]
    pub fn visit_avx2<B, F>(
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
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the avx2 feature check.
                return unsafe { self.visit_avx2_impl::<B, F>(query, stack, visitor) };
            }
        }
        self.visit_simd(query, stack, visitor)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn visit_avx2_impl<B, F>(
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
        if query.contains(self.root_box()) {
            for &index in &self.indices[..self.num_items] {
                visitor(index)?;
            }
            return ControlFlow::Continue(());
        }
        let qmxx_v = _mm256_set1_pd(query.max_x);
        let qmnx_v = _mm256_set1_pd(query.min_x);
        let qmxy_v = _mm256_set1_pd(query.max_y);
        let qmny_v = _mm256_set1_pd(query.min_y);
        let qmxz_v = _mm256_set1_pd(query.max_z);
        let qmnz_v = _mm256_set1_pd(query.min_z);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                let start = self.leaf_start_for_entry(node_index, level);
                let end = if end < self.level_bounds[level] {
                    self.leaf_start_for_entry(end, level)
                } else {
                    self.num_items
                };
                for &index in &self.indices[start..end] {
                    visitor(index)?;
                }
            } else {
                let child_level = if is_leaf { 0 } else { level - 1 };
                let mut pos = node_index;
                while pos + 4 <= end {
                    // SAFETY: `pos + 4 <= end`, and `end` is bounded by the array length.
                    let (mnx, mxx, mny, mxy, mnz, mxz) = unsafe {
                        (
                            _mm256_loadu_pd(self.min_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.min_ys.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_ys.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.min_zs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_zs.as_ptr().add(pos)),
                        )
                    };
                    let overlap = _mm256_and_pd(
                        _mm256_and_pd(
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mnx, qmxx_v),
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mxx, qmnx_v),
                            ),
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mny, qmxy_v),
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mxy, qmny_v),
                            ),
                        ),
                        _mm256_and_pd(
                            _mm256_cmp_pd::<_CMP_LE_OQ>(mnz, qmxz_v),
                            _mm256_cmp_pd::<_CMP_GE_OQ>(mxz, qmnz_v),
                        ),
                    );
                    let mut bits = _mm256_movemask_pd(overlap) as usize;
                    if is_leaf {
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            visitor(self.indices[pos + k])?;
                            bits &= bits - 1;
                        }
                    } else {
                        let contains = _mm256_and_pd(
                            _mm256_and_pd(
                                _mm256_and_pd(
                                    _mm256_cmp_pd::<_CMP_GE_OQ>(mnx, qmnx_v),
                                    _mm256_cmp_pd::<_CMP_LE_OQ>(mxx, qmxx_v),
                                ),
                                _mm256_and_pd(
                                    _mm256_cmp_pd::<_CMP_GE_OQ>(mny, qmny_v),
                                    _mm256_cmp_pd::<_CMP_LE_OQ>(mxy, qmxy_v),
                                ),
                            ),
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mnz, qmnz_v),
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mxz, qmxz_v),
                            ),
                        );
                        let cbits = _mm256_movemask_pd(contains) as usize;
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            stack.push(self.indices[pos + k]);
                            stack.push(encode_level(child_level, cbits & (1 << k) != 0));
                            bits &= bits - 1;
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
                            stack.push(encode_level(
                                child_level,
                                self.query_contains_node(query, pos),
                            ));
                        }
                    }
                    pos += 1;
                }
            }

            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return ControlFlow::Continue(());
            }
        }
    }

    /// Element-by-element traversal (SoA layout, branchless `overlaps`).
    #[doc(hidden)]
    pub fn search_scalar(&self, query: Box3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        if query.contains(self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
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

    /// Total extent box, stored as the last node. Callers must ensure the index is
    /// non-empty.
    #[inline]
    fn root_box(&self) -> Box3D {
        let last = self.min_xs.len() - 1;
        Box3D::new(
            self.min_xs[last],
            self.min_ys[last],
            self.min_zs[last],
            self.max_xs[last],
            self.max_ys[last],
            self.max_zs[last],
        )
    }

    /// True when `query` fully contains the box stored at `pos`.
    #[inline]
    fn query_contains_node(&self, query: Box3D, pos: usize) -> bool {
        query.min_x <= self.min_xs[pos]
            && self.max_xs[pos] <= query.max_x
            && query.min_y <= self.min_ys[pos]
            && self.max_ys[pos] <= query.max_y
            && query.min_z <= self.min_zs[pos]
            && self.max_zs[pos] <= query.max_z
    }

    /// Append every leaf index under the entry at `node_index` (a node at `level`)
    /// without per-item overlap tests, used when the query fully contains the node.
    #[inline]
    fn extend_contained_leaf_indices(
        &self,
        node_index: usize,
        end: usize,
        level: usize,
        out: &mut Vec<usize>,
    ) {
        let start = self.leaf_start_for_entry(node_index, level);
        let end = if end < self.level_bounds[level] {
            self.leaf_start_for_entry(end, level)
        } else {
            self.num_items
        };
        out.extend_from_slice(&self.indices[start..end]);
    }

    /// Walk a node entry down to the leaf-array position where its subtree begins.
    #[inline]
    fn leaf_start_for_entry(&self, mut index: usize, mut level: usize) -> usize {
        while level > 0 {
            index = self.indices[index];
            level -= 1;
        }
        index
    }

    /// AVX2/SSE path through `wide::f64x4`.
    #[doc(hidden)]
    pub fn search_simd(&self, query: Box3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        if query.contains(self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
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
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                self.extend_contained_leaf_indices(node_index, end, level, out);
            } else {
                let child_level = if is_leaf { 0 } else { level - 1 };
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
                                let p = pos + k;
                                let index = self.indices[p];
                                if is_leaf {
                                    out.push(index);
                                } else {
                                    stack.push(index);
                                    stack.push(encode_level(
                                        child_level,
                                        self.query_contains_node(query, p),
                                    ));
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
                            stack.push(encode_level(
                                child_level,
                                self.query_contains_node(query, pos),
                            ));
                        }
                    }
                    pos += 1;
                }
            }

            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
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
        if query.contains(self.root_box()) {
            for &index in &self.indices[..self.num_items] {
                visitor(index)?;
            }
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
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                let start = self.leaf_start_for_entry(node_index, level);
                let end = if end < self.level_bounds[level] {
                    self.leaf_start_for_entry(end, level)
                } else {
                    self.num_items
                };
                for &index in &self.indices[start..end] {
                    visitor(index)?;
                }
            } else {
                // Guarded against underflow for a single leaf-level node (`level == 0`);
                // `child_level` is only read on the internal-node push paths.
                let child_level = if is_leaf { 0 } else { level - 1 };
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
                                let p = pos + k;
                                let index = self.indices[p];
                                if is_leaf {
                                    visitor(index)?;
                                } else {
                                    stack.push(index);
                                    stack.push(encode_level(
                                        child_level,
                                        self.query_contains_node(query, p),
                                    ));
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
                            stack.push(encode_level(
                                child_level,
                                self.query_contains_node(query, pos),
                            ));
                        }
                    }
                    pos += 1;
                }
            }

            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
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
        if query.contains(self.root_box()) {
            for &index in &self.indices[..self.num_items] {
                visitor(index)?;
            }
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
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                let start = self.leaf_start_for_entry(node_index, level);
                let end = if end < self.level_bounds[level] {
                    self.leaf_start_for_entry(end, level)
                } else {
                    self.num_items
                };
                for &index in &self.indices[start..end] {
                    visitor(index)?;
                }
            } else {
                // Guarded against underflow for a single leaf-level node (`level == 0`);
                // `child_level` is only read on the internal-node push paths.
                let child_level = if is_leaf { 0 } else { level - 1 };
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
                    if is_leaf {
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            visitor(self.indices[pos + k])?;
                            bits &= bits - 1;
                        }
                    } else {
                        // query contains child: qmin <= cmin && cmax <= qmax on all axes.
                        let c1 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mnx, qmnx_v);
                        let c2 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxx, qmxx_v);
                        let c3 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mny, qmny_v);
                        let c4 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxy, qmxy_v);
                        let c5 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mnz, qmnz_v);
                        let c6 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxz, qmxz_v);
                        let cbits: u8 = c1 & c2 & c3 & c4 & c5 & c6;
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            stack.push(self.indices[pos + k]);
                            stack.push(encode_level(child_level, cbits & (1 << k) != 0));
                            bits &= bits - 1;
                        }
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
                            stack.push(encode_level(
                                child_level,
                                self.query_contains_node(query, pos),
                            ));
                        }
                    }
                    pos += 1;
                }
            }

            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
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
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: selected only after checking avx2 availability.
                unsafe { self.search_avx2_impl(query, out, stack) };
                return;
            }
        }
        self.search_simd(query, out, stack);
    }

    /// AVX2 (256-bit, 4 boxes/chunk) range search — the runtime tier between the
    /// `wide` fallback and AVX-512. AVX2 lacks `VPCOMPRESSQ`, so leaf results use
    /// the AVX2 left-pack (`VPERMD` + LUT, see `crate::leftpack`). Doc-hidden.
    #[doc(hidden)]
    pub fn search_avx2(&self, query: Box3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the avx2 feature check above.
                unsafe { self.search_avx2_impl(query, out, stack) };
                return;
            }
        }
        self.search_simd(query, out, stack);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_avx2_impl(&self, query: Box3D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        use std::arch::x86_64::*;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        if query.contains(self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }
        let qmxx_v = _mm256_set1_pd(query.max_x);
        let qmnx_v = _mm256_set1_pd(query.min_x);
        let qmxy_v = _mm256_set1_pd(query.max_y);
        let qmny_v = _mm256_set1_pd(query.min_y);
        let qmxz_v = _mm256_set1_pd(query.max_z);
        let qmnz_v = _mm256_set1_pd(query.min_z);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                self.extend_contained_leaf_indices(node_index, end, level, out);
            } else {
                let child_level = if is_leaf { 0 } else { level - 1 };
                if is_leaf {
                    out.reserve(end - node_index + 4);
                }
                let mut pos = node_index;
                while pos + 4 <= end {
                    // SAFETY: `pos + 4 <= end`, and `end` is bounded by the array length.
                    let (mnx, mxx, mny, mxy, mnz, mxz) = unsafe {
                        (
                            _mm256_loadu_pd(self.min_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.min_ys.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_ys.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.min_zs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_zs.as_ptr().add(pos)),
                        )
                    };
                    let overlap = _mm256_and_pd(
                        _mm256_and_pd(
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mnx, qmxx_v),
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mxx, qmnx_v),
                            ),
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mny, qmxy_v),
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mxy, qmny_v),
                            ),
                        ),
                        _mm256_and_pd(
                            _mm256_cmp_pd::<_CMP_LE_OQ>(mnz, qmxz_v),
                            _mm256_cmp_pd::<_CMP_GE_OQ>(mxz, qmnz_v),
                        ),
                    );
                    let mut bits = _mm256_movemask_pd(overlap) as usize;
                    if is_leaf {
                        if bits != 0 {
                            // SAFETY: `pos + 4 <= end <= indices.len()`; `out` has
                            // `end - node_index + 4` slack reserved.
                            unsafe {
                                let added = leftpack4(
                                    self.indices.as_ptr().add(pos),
                                    bits as u32,
                                    out.as_mut_ptr().add(out.len()),
                                );
                                out.set_len(out.len() + added);
                            }
                        }
                    } else {
                        let contains = _mm256_and_pd(
                            _mm256_and_pd(
                                _mm256_and_pd(
                                    _mm256_cmp_pd::<_CMP_GE_OQ>(mnx, qmnx_v),
                                    _mm256_cmp_pd::<_CMP_LE_OQ>(mxx, qmxx_v),
                                ),
                                _mm256_and_pd(
                                    _mm256_cmp_pd::<_CMP_GE_OQ>(mny, qmny_v),
                                    _mm256_cmp_pd::<_CMP_LE_OQ>(mxy, qmxy_v),
                                ),
                            ),
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mnz, qmnz_v),
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mxz, qmxz_v),
                            ),
                        );
                        let cbits = _mm256_movemask_pd(contains) as usize;
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            stack.push(self.indices[pos + k]);
                            stack.push(encode_level(child_level, cbits & (1 << k) != 0));
                            bits &= bits - 1;
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
                            stack.push(encode_level(
                                child_level,
                                self.query_contains_node(query, pos),
                            ));
                        }
                    }
                    pos += 1;
                }
            }

            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
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
        if query.contains(self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
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
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                self.extend_contained_leaf_indices(node_index, end, level, out);
            } else {
                let child_level = if is_leaf { 0 } else { level - 1 };
                // Reserve the whole node's worth of results up front so the
                // compress-store below writes through a stable base pointer (no
                // reallocation mid-node).
                if is_leaf {
                    out.reserve(end - node_index);
                }
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
                    if is_leaf {
                        // VPCOMPRESSQ: pack the matching index lanes contiguously
                        // into `out` in one instruction (capacity reserved above).
                        // SAFETY: `pos + 8 <= end <= indices.len()`; `out` has at
                        // least `end - node_index` slack reserved, so the store of
                        // up to 8 elements past `len` stays in bounds.
                        unsafe {
                            let dst = out.as_mut_ptr().add(out.len()) as *mut i64;
                            let vidx =
                                _mm512_loadu_epi64(self.indices.as_ptr().add(pos) as *const i64);
                            _mm512_mask_compressstoreu_epi64(dst, bits, vidx);
                            out.set_len(out.len() + bits.count_ones() as usize);
                        }
                    } else {
                        // query contains child: qmin <= cmin && cmax <= qmax on all axes.
                        let c1 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mnx, qmnx_v);
                        let c2 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxx, qmxx_v);
                        let c3 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mny, qmny_v);
                        let c4 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxy, qmxy_v);
                        let c5 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mnz, qmnz_v);
                        let c6 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxz, qmxz_v);
                        let cbits: u8 = c1 & c2 & c3 & c4 & c5 & c6;
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            stack.push(self.indices[pos + k]);
                            stack.push(encode_level(child_level, cbits & (1 << k) != 0));
                            bits &= bits - 1;
                        }
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
                            stack.push(encode_level(
                                child_level,
                                self.query_contains_node(query, pos),
                            ));
                        }
                    }
                    pos += 1;
                }
            }

            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }
}

#[inline]
fn load4(a: &[f64], p: usize) -> f64x4 {
    f64x4::from([a[p], a[p + 1], a[p + 2], a[p + 3]])
}

/// High bit of the stacked level word, set when the query fully contains a node so
/// its whole subtree can be collected without further overlap tests.
const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
const LEVEL_MASK: usize = !CONTAINED_FLAG;

#[inline]
fn encode_level(level: usize, contained: bool) -> usize {
    if contained {
        level | CONTAINED_FLAG
    } else {
        level
    }
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
    /// Derived at load (not stored), so owned rather than borrowed.
    level_bounds: Vec<usize>,
    entries: &'a [u8],
    indices: &'a [u8],
}

impl<'a> SimdIndex3DView<'a> {
    /// Borrow a zero-copy view over the canonical `PSINDEX` 3D bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 3, 8)?;
        if payload.is_some() {
            return Err(LoadError::UnsupportedVersion);
        }
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
        self.level_bounds[index]
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

    /// Walk a node entry down to the leaf-array position where its subtree begins.
    #[inline]
    fn leaf_start_for_entry(&self, mut index: usize, mut level: usize) -> usize {
        while level > 0 {
            index = self.index_at(index);
            level -= 1;
        }
        index
    }

    /// Leaf-array `[start, end)` range covered by the entry at `node_index`
    /// (a node at `level`), used when the query fully contains that node.
    #[inline]
    fn contained_leaf_range(&self, node_index: usize, end: usize, level: usize) -> (usize, usize) {
        let start = self.leaf_start_for_entry(node_index, level);
        let end = if end < self.level_bound_unchecked(level) {
            self.leaf_start_for_entry(end, level)
        } else {
            self.num_items
        };
        (start, end)
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

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`. See [`Index2D::join`](crate::Index2D::join).
    pub fn join(&self, other: &SimdIndex3DView<'_>) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other`. See
    /// [`Index2D::join_with`](crate::Index2D::join_with).
    pub fn join_with<B, F>(&self, other: &SimdIndex3DView<'_>, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, visitor)
    }

    /// Return every unordered pair of distinct intersecting items within this
    /// view, each pair exactly once. See
    /// [`Index2D::self_join`](crate::Index2D::self_join).
    pub fn self_join(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.self_join_with(|i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every unordered pair of distinct intersecting items within this
    /// view. See [`Index2D::self_join_with`](crate::Index2D::self_join_with).
    pub fn self_join_with<B, F>(&self, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, visitor)
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
        if query.contains(self.box_at(self.num_nodes - 1)) {
            for pos in 0..self.num_items {
                visitor(self.index_at(pos))?;
            }
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
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;

            if contained {
                let (start, end) = self.contained_leaf_range(node_index, end, level);
                for pos in start..end {
                    visitor(self.index_at(pos))?;
                }
            } else {
                let child_level = if is_leaf { 0 } else { level - 1 };
                let mut pos = node_index;
                while pos + 4 <= end {
                    let base = pos * RECORD_3D;
                    let mnx = lane4_3d(self.entries, base, 0);
                    let mxx = lane4_3d(self.entries, base, 24);
                    let mny = lane4_3d(self.entries, base, 8);
                    let mxy = lane4_3d(self.entries, base, 32);
                    let mnz = lane4_3d(self.entries, base, 16);
                    let mxz = lane4_3d(self.entries, base, 40);
                    let mask = mnx.simd_le(qmxx)
                        & mxx.simd_ge(qmnx)
                        & mny.simd_le(qmxy)
                        & mxy.simd_ge(qmny)
                        & mnz.simd_le(qmxz)
                        & mxz.simd_ge(qmnz);
                    let bits = mask.to_bitmask();
                    if bits != 0 {
                        // query contains child: qmin <= cmin && cmax <= qmax on all axes.
                        let cmask = mnx.simd_ge(qmnx)
                            & mxx.simd_le(qmxx)
                            & mny.simd_ge(qmny)
                            & mxy.simd_le(qmxy)
                            & mnz.simd_ge(qmnz)
                            & mxz.simd_le(qmxz);
                        let cbits = cmask.to_bitmask();
                        for k in 0..4 {
                            if bits & (1 << k) != 0 {
                                let p = pos + k;
                                let index = self.index_at(p);
                                if is_leaf {
                                    visitor(index)?;
                                } else {
                                    stack.push(index);
                                    stack.push(encode_level(child_level, cbits & (1 << k) != 0));
                                }
                            }
                        }
                    }
                    pos += 4;
                }

                while pos < end {
                    let b = self.box_at(pos);
                    if b.overlaps(query) {
                        let index = self.index_at(pos);
                        if is_leaf {
                            visitor(index)?;
                        } else {
                            stack.push(index);
                            stack.push(encode_level(child_level, query.contains(b)));
                        }
                    }
                    pos += 1;
                }
            }

            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return ControlFlow::Continue(());
            }
        }
    }

    #[inline]
    fn distance_squared_at(&self, pos: usize, query: NeighborQuery3D) -> f64 {
        query.distance_squared_to(self.box_at(pos))
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
            if let Some(index) =
                self.nearest_one_with_queue(NeighborQuery3D::Point(point), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Point(point),
            max_results,
            max_distance,
            results,
            &mut queue,
        );
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
            if let Some(index) = self.nearest_one_with_queue(
                NeighborQuery3D::Point(point),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }
        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Point(point),
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
        self.visit_neighbors_with_queue(
            NeighborQuery3D::Point(point),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    /// Return up to `max_results` item indices nearest to the box `query`.
    /// See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box(&self, query: Box3D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box_within(
        &self,
        query: Box3D,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        self.neighbors_of_box_into(query, max_results, max_distance, &mut results);
        results
    }

    /// Box-query nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_of_box_into(
        &self,
        query: Box3D,
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
            if let Some(index) =
                self.nearest_one_with_queue(NeighborQuery3D::Box(query), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            results,
            &mut queue,
        );
    }

    /// Box-query nearest-neighbor search with reusable result and
    /// priority-queue buffers.
    pub fn neighbors_of_box_with<'na>(
        &self,
        query: Box3D,
        max_results: usize,
        max_distance: f64,
        workspace: &'na mut NeighborWorkspace,
    ) -> &'na [usize] {
        workspace.results.clear();
        if max_results == 0 {
            workspace.queue.clear();
            workspace.node_queue.clear();
            return &workspace.results;
        }
        if max_results == 1 {
            workspace.queue.clear();
            if let Some(index) = self.nearest_one_with_queue(
                NeighborQuery3D::Box(query),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
        );
        &workspace.results
    }

    /// Visit items in nondecreasing box-to-box distance order from `query`.
    /// See [`Index2D::visit_neighbors_of_box`](crate::Index2D::visit_neighbors_of_box).
    pub fn visit_neighbors_of_box<B, F>(
        &self,
        query: Box3D,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.visit_neighbors_with_queue(
            NeighborQuery3D::Box(query),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    fn collect_neighbors_with_queue(
        &self,
        query: NeighborQuery3D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        queue: &mut BinaryHeap<NeighborState>,
    ) {
        best_first::collect_neighbors(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |n| self.level_bound_unchecked(self.upper_bound_level(n)),
            |p| self.index_at(p),
            max_results,
            max_distance,
            |pos| self.distance_squared_at(pos, query),
            results,
            queue,
        );
    }

    fn visit_neighbors_with_queue<B, F>(
        &self,
        query: NeighborQuery3D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborState>,
        visitor: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        best_first::visit_neighbors(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |n| self.level_bound_unchecked(self.upper_bound_level(n)),
            |p| self.index_at(p),
            max_distance,
            |pos| self.distance_squared_at(pos, query),
            queue,
            visitor,
        )
    }

    fn nearest_one_with_queue(
        &self,
        query: NeighborQuery3D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        best_first::nearest_one(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |n| self.level_bound_unchecked(self.upper_bound_level(n)),
            |p| self.index_at(p),
            max_distance,
            |pos| self.distance_squared_at(pos, query),
            queue,
        )
    }
}

impl TreeAccess for SimdIndex3D {
    type Bounds = Box3D;

    #[inline]
    fn tree_num_items(&self) -> usize {
        self.num_items
    }
    #[inline]
    fn tree_num_nodes(&self) -> usize {
        self.min_xs.len()
    }
    #[inline]
    fn tree_node_size(&self) -> usize {
        self.node_size
    }
    #[inline]
    fn tree_level_count(&self) -> usize {
        self.level_bounds.len()
    }
    #[inline]
    fn tree_level_bound(&self, level: usize) -> usize {
        self.level_bounds[level]
    }
    #[inline]
    fn tree_bounds(&self, pos: usize) -> Box3D {
        Box3D::new(
            self.min_xs[pos],
            self.min_ys[pos],
            self.min_zs[pos],
            self.max_xs[pos],
            self.max_ys[pos],
            self.max_zs[pos],
        )
    }
    #[inline]
    fn tree_index(&self, pos: usize) -> usize {
        self.indices[pos]
    }
    #[inline]
    fn bounds_overlap(a: Box3D, b: Box3D) -> bool {
        a.overlaps(b)
    }
    #[inline]
    fn bounds_contain(outer: Box3D, inner: Box3D) -> bool {
        outer.contains(inner)
    }
}

impl TreeAccess for SimdIndex3DView<'_> {
    type Bounds = Box3D;

    #[inline]
    fn tree_num_items(&self) -> usize {
        self.num_items
    }
    #[inline]
    fn tree_num_nodes(&self) -> usize {
        self.num_nodes
    }
    #[inline]
    fn tree_node_size(&self) -> usize {
        self.node_size
    }
    #[inline]
    fn tree_level_count(&self) -> usize {
        self.level_count
    }
    #[inline]
    fn tree_level_bound(&self, level: usize) -> usize {
        self.level_bound_unchecked(level)
    }
    #[inline]
    fn tree_bounds(&self, pos: usize) -> Box3D {
        self.box_at(pos)
    }
    #[inline]
    fn tree_index(&self, pos: usize) -> usize {
        self.index_at(pos)
    }
    #[inline]
    fn bounds_overlap(a: Box3D, b: Box3D) -> bool {
        a.overlaps(b)
    }
    #[inline]
    fn bounds_contain(outer: Box3D, inner: Box3D) -> bool {
        outer.contains(inner)
    }
}

impl SimdIndex3D {
    #[inline]
    fn box_at_soa(&self, pos: usize) -> Box3D {
        Box3D::new(
            self.min_xs[pos],
            self.min_ys[pos],
            self.min_zs[pos],
            self.max_xs[pos],
            self.max_ys[pos],
            self.max_zs[pos],
        )
    }

    /// SoA/SIMD ordered closest-hit raycast: same result as
    /// [`Index3D::raycast_closest_with`](crate::Index3D::raycast_closest_with), but the
    /// ray/AABB slab test is evaluated four children at a time with `wide::f64x4`. The
    /// dominant cost of sparse-scene ray traversal is the per-child slab arithmetic, so
    /// the SoA columns let it run 4-wide.
    ///
    /// Axis-parallel rays are handled by a masked slab test to avoid `0 * inf = NaN`
    /// at box faces: AVX-512 when available, otherwise `wide::f64x4`.
    pub fn raycast_closest_with(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // The plain AVX-512 slab is multiply-only (fastest, but not NaN-safe for
                // axis-parallel rays); the masked variant handles a zero direction.
                // SAFETY: only reached after confirming avx512f is available.
                return unsafe {
                    if ray.has_zero_direction() {
                        self.raycast_closest_avx512_masked(ray, workspace)
                    } else {
                        self.raycast_closest_avx512(ray, workspace)
                    }
                };
            }
        }
        self.raycast_closest_wide(ray, workspace)
    }

    fn raycast_closest_wide(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        let queue = &mut workspace.node_queue;
        queue.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return None;
        }
        let root = self.min_xs.len() - 1;
        let root_t = ray.enter_t(self.box_at_soa(root))?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        let ox = f64x4::splat(ray.origin.x);
        let oy = f64x4::splat(ray.origin.y);
        let oz = f64x4::splat(ray.origin.z);
        let ix = f64x4::splat(ray.inv_dir_x);
        let iy = f64x4::splat(ray.inv_dir_y);
        let iz = f64x4::splat(ray.inv_dir_z);
        let zero = f64x4::splat(0.0);
        let maxd = f64x4::splat(ray.max_distance);
        let pos_inf = f64x4::splat(f64::INFINITY);
        let neg_inf = f64x4::splat(f64::NEG_INFINITY);
        // Direction is constant for the whole ray, so degeneracy is a per-axis flag,
        // not per-lane. For a zero-direction axis the slab imposes no `t` bound when the
        // origin is inside (inclusive, so a ray exactly on a face still hits) and an
        // empty interval otherwise — computed with `blend` to avoid the `0 * inf = NaN`
        // that the multiply path would hit.
        let (zx, zy, zz) = (ray.dir_x == 0.0, ray.dir_y == 0.0, ray.dir_z == 0.0);
        let axis = |mn: f64x4, mx: f64x4, o: f64x4, inv: f64x4, degenerate: bool| {
            if degenerate {
                let inside = mn.simd_le(o) & o.simd_le(mx);
                (
                    inside.blend(neg_inf, pos_inf),
                    inside.blend(pos_inf, neg_inf),
                )
            } else {
                let t1 = (mn - o) * inv;
                let t2 = (mx - o) * inv;
                (t1.fast_min(t2), t1.fast_max(t2))
            }
        };

        while let Some(node) = queue.pop() {
            if node.dist >= best_t {
                break;
            }
            let upper = upper_bound_level(&self.level_bounds, node.index);
            let end = (node.index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node.index < self.num_items;

            let mut pos = node.index;
            while pos + 4 <= end {
                let (nx, fx) = axis(
                    load4(&self.min_xs, pos),
                    load4(&self.max_xs, pos),
                    ox,
                    ix,
                    zx,
                );
                let (ny, fy) = axis(
                    load4(&self.min_ys, pos),
                    load4(&self.max_ys, pos),
                    oy,
                    iy,
                    zy,
                );
                let (nz, fz) = axis(
                    load4(&self.min_zs, pos),
                    load4(&self.max_zs, pos),
                    oz,
                    iz,
                    zz,
                );
                let near = nx.fast_max(ny).fast_max(nz).fast_max(zero);
                let far = fx.fast_min(fy).fast_min(fz).fast_min(maxd);
                let bits = near.simd_le(far).to_bitmask();
                if bits != 0 {
                    let tn = near.to_array();
                    // `k` indexes `tn` and selects the mask bit, so a range loop is clearest.
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..4 {
                        if bits & (1 << k) != 0 && tn[k] < best_t {
                            if is_leaf {
                                best_t = tn[k];
                                best_index = Some(self.indices[pos + k]);
                            } else {
                                queue.push(NeighborNodeState::new(self.indices[pos + k], tn[k]));
                            }
                        }
                    }
                }
                pos += 4;
            }
            while pos < end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos))
                    && t < best_t
                {
                    if is_leaf {
                        best_t = t;
                        best_index = Some(self.indices[pos]);
                    } else {
                        queue.push(NeighborNodeState::new(self.indices[pos], t));
                    }
                }
                pos += 1;
            }
        }

        best_index.map(|index| (index, best_t))
    }

    /// AVX-512 closest-hit: the slab test runs eight children at a time. With
    /// `node_size == 8` that is exactly one vector op per node.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn raycast_closest_avx512(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        use std::arch::x86_64::*;

        let queue = &mut workspace.node_queue;
        queue.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return None;
        }
        let root = self.min_xs.len() - 1;
        let root_t = ray.enter_t(self.box_at_soa(root))?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        let ox = _mm512_set1_pd(ray.origin.x);
        let oy = _mm512_set1_pd(ray.origin.y);
        let oz = _mm512_set1_pd(ray.origin.z);
        let ix = _mm512_set1_pd(ray.inv_dir_x);
        let iy = _mm512_set1_pd(ray.inv_dir_y);
        let iz = _mm512_set1_pd(ray.inv_dir_z);
        let zero = _mm512_setzero_pd();
        let maxd = _mm512_set1_pd(ray.max_distance);

        while let Some(node) = queue.pop() {
            if node.dist >= best_t {
                break;
            }
            let upper = upper_bound_level(&self.level_bounds, node.index);
            let end = (node.index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node.index < self.num_items;
            let mut pos = node.index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end <= len`, so all eight lanes are in bounds.
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
                let t1x = _mm512_mul_pd(_mm512_sub_pd(mnx, ox), ix);
                let t2x = _mm512_mul_pd(_mm512_sub_pd(mxx, ox), ix);
                let t1y = _mm512_mul_pd(_mm512_sub_pd(mny, oy), iy);
                let t2y = _mm512_mul_pd(_mm512_sub_pd(mxy, oy), iy);
                let t1z = _mm512_mul_pd(_mm512_sub_pd(mnz, oz), iz);
                let t2z = _mm512_mul_pd(_mm512_sub_pd(mxz, oz), iz);
                let near = _mm512_max_pd(
                    _mm512_max_pd(
                        _mm512_max_pd(_mm512_min_pd(t1x, t2x), _mm512_min_pd(t1y, t2y)),
                        _mm512_min_pd(t1z, t2z),
                    ),
                    zero,
                );
                let far = _mm512_min_pd(
                    _mm512_min_pd(
                        _mm512_min_pd(_mm512_max_pd(t1x, t2x), _mm512_max_pd(t1y, t2y)),
                        _mm512_max_pd(t1z, t2z),
                    ),
                    maxd,
                );
                let mut bits: u8 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(near, far);
                if bits != 0 {
                    let mut tn = [0.0f64; 8];
                    // SAFETY: `tn` holds eight `f64`, matching the 512-bit store.
                    unsafe { _mm512_storeu_pd(tn.as_mut_ptr(), near) };
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if tn[k] < best_t {
                            if is_leaf {
                                best_t = tn[k];
                                best_index = Some(self.indices[pos + k]);
                            } else {
                                queue.push(NeighborNodeState::new(self.indices[pos + k], tn[k]));
                            }
                        }
                    }
                }
                pos += 8;
            }
            while pos < end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos))
                    && t < best_t
                {
                    if is_leaf {
                        best_t = t;
                        best_index = Some(self.indices[pos]);
                    } else {
                        queue.push(NeighborNodeState::new(self.indices[pos], t));
                    }
                }
                pos += 1;
            }
        }

        best_index.map(|index| (index, best_t))
    }

    /// AVX-512 closest-hit for axis-parallel rays: a zero-direction axis is handled with
    /// `_mm512_mask_blend_pd` over an inclusive inside-test, so it is NaN-safe even when
    /// the ray grazes a box face. Only invoked for rays with a zero direction component.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn raycast_closest_avx512_masked(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        use std::arch::x86_64::*;

        let queue = &mut workspace.node_queue;
        queue.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return None;
        }
        let root = self.min_xs.len() - 1;
        let root_t = ray.enter_t(self.box_at_soa(root))?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        let ox = _mm512_set1_pd(ray.origin.x);
        let oy = _mm512_set1_pd(ray.origin.y);
        let oz = _mm512_set1_pd(ray.origin.z);
        let ix = _mm512_set1_pd(ray.inv_dir_x);
        let iy = _mm512_set1_pd(ray.inv_dir_y);
        let iz = _mm512_set1_pd(ray.inv_dir_z);
        let zero = _mm512_setzero_pd();
        let maxd = _mm512_set1_pd(ray.max_distance);
        let pos_inf = _mm512_set1_pd(f64::INFINITY);
        let neg_inf = _mm512_set1_pd(f64::NEG_INFINITY);
        let (zx, zy, zz) = (ray.dir_x == 0.0, ray.dir_y == 0.0, ray.dir_z == 0.0);

        // `(near, far)` interval for one axis across eight children.
        let axis = |mn, mx, o, inv, degenerate: bool| {
            if degenerate {
                let inside = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mn, o)
                    & _mm512_cmp_pd_mask::<_CMP_LE_OQ>(o, mx);
                (
                    _mm512_mask_blend_pd(inside, pos_inf, neg_inf),
                    _mm512_mask_blend_pd(inside, neg_inf, pos_inf),
                )
            } else {
                let t1 = _mm512_mul_pd(_mm512_sub_pd(mn, o), inv);
                let t2 = _mm512_mul_pd(_mm512_sub_pd(mx, o), inv);
                (_mm512_min_pd(t1, t2), _mm512_max_pd(t1, t2))
            }
        };

        while let Some(node) = queue.pop() {
            if node.dist >= best_t {
                break;
            }
            let upper = upper_bound_level(&self.level_bounds, node.index);
            let end = (node.index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node.index < self.num_items;

            let mut pos = node.index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end <= len`, so all eight lanes are in bounds.
                let (nx, fx, ny, fy, nz, fz) = unsafe {
                    let (mnx, mxx, mny, mxy, mnz, mxz) = (
                        _mm512_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_zs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_zs.as_ptr().add(pos)),
                    );
                    let (nx, fx) = axis(mnx, mxx, ox, ix, zx);
                    let (ny, fy) = axis(mny, mxy, oy, iy, zy);
                    let (nz, fz) = axis(mnz, mxz, oz, iz, zz);
                    (nx, fx, ny, fy, nz, fz)
                };
                let near = _mm512_max_pd(_mm512_max_pd(_mm512_max_pd(nx, ny), nz), zero);
                let far = _mm512_min_pd(_mm512_min_pd(_mm512_min_pd(fx, fy), fz), maxd);
                let mut bits: u8 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(near, far);
                if bits != 0 {
                    let mut tn = [0.0f64; 8];
                    // SAFETY: `tn` holds eight `f64`, matching the 512-bit store.
                    unsafe { _mm512_storeu_pd(tn.as_mut_ptr(), near) };
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if tn[k] < best_t {
                            if is_leaf {
                                best_t = tn[k];
                                best_index = Some(self.indices[pos + k]);
                            } else {
                                queue.push(NeighborNodeState::new(self.indices[pos + k], tn[k]));
                            }
                        }
                    }
                }
                pos += 8;
            }
            while pos < end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos))
                    && t < best_t
                {
                    if is_leaf {
                        best_t = t;
                        best_index = Some(self.indices[pos]);
                    } else {
                        queue.push(NeighborNodeState::new(self.indices[pos], t));
                    }
                }
                pos += 1;
            }
        }

        best_index.map(|index| (index, best_t))
    }

    /// Return the nearest item whose box the ray segment enters, as
    /// `(item index, entry t)`, or `None` when the segment hits nothing.
    ///
    /// Nodes are visited front-to-back by entry distance and pruned once a
    /// closer hit is known, so the cost is roughly independent of
    /// `max_distance` after the first hit. `t` is `0.0` when the ray origin
    /// starts inside the item's box.
    pub fn raycast_closest(&self, ray: Ray3D) -> Option<(usize, f64)> {
        let mut workspace = NeighborWorkspace::new();
        self.raycast_closest_with(ray, &mut workspace)
    }

    /// Return the indices of all items whose boxes the ray segment touches.
    pub fn raycast(&self, ray: Ray3D) -> Vec<usize> {
        let mut results = Vec::new();
        self.raycast_into(ray, &mut results);
        results
    }

    /// Raycast with a reusable result buffer.
    pub fn raycast_into(&self, ray: Ray3D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.raycast_into_stack(ray, results, &mut stack);
    }

    /// Raycast with reusable result and traversal buffers.
    pub fn raycast_with<'a>(&self, ray: Ray3D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.raycast_into_stack(ray, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Buffer-explicit raycast (mirrors `search_into_stack`). The per-node slab
    /// test is vectorized: AVX-512 (eight children at a time) for non-degenerate
    /// rays where available, otherwise `wide::f64x4`. Axis-parallel rays always
    /// take the `wide` path, whose `blend` kernel is NaN-safe at box faces.
    #[doc(hidden)]
    pub fn raycast_into_stack(&self, ray: Ray3D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        #[cfg(target_arch = "x86_64")]
        {
            if !ray.has_zero_direction() {
                if std::is_x86_feature_detected!("avx512f") {
                    // SAFETY: reached only after confirming avx512f is available.
                    unsafe { self.raycast_collect_avx512(ray, results, stack) };
                    return;
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: reached only after confirming avx2 is available.
                    unsafe { self.raycast_collect_avx2(ray, results, stack) };
                    return;
                }
            }
        }
        self.raycast_collect_wide(ray, results, stack);
    }

    /// Force the `wide` all-hits raycast path (doc-hidden; benchmarks/tests).
    #[doc(hidden)]
    pub fn raycast_wide_into(&self, ray: Ray3D, results: &mut Vec<usize>) {
        let mut stack = Vec::new();
        self.raycast_collect_wide(ray, results, &mut stack);
    }

    /// Force the AVX2 all-hits raycast path (doc-hidden; benchmarks/tests).
    #[doc(hidden)]
    pub fn raycast_avx2_into(&self, ray: Ray3D, results: &mut Vec<usize>) {
        let mut stack = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if !ray.has_zero_direction() && std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the avx2 feature check.
                unsafe { self.raycast_collect_avx2(ray, results, &mut stack) };
                return;
            }
        }
        self.raycast_collect_wide(ray, results, &mut stack);
    }

    fn raycast_collect_wide(&self, ray: Ray3D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        results.clear();
        stack.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return;
        }

        let ox = f64x4::splat(ray.origin.x);
        let oy = f64x4::splat(ray.origin.y);
        let oz = f64x4::splat(ray.origin.z);
        let ix = f64x4::splat(ray.inv_dir_x);
        let iy = f64x4::splat(ray.inv_dir_y);
        let iz = f64x4::splat(ray.inv_dir_z);
        let zero = f64x4::splat(0.0);
        let maxd = f64x4::splat(ray.max_distance);
        let pos_inf = f64x4::splat(f64::INFINITY);
        let neg_inf = f64x4::splat(f64::NEG_INFINITY);
        // A zero-direction axis imposes no `t` bound when the origin is inside
        // (inclusive, so a ray on a face still hits) and an empty interval
        // otherwise, computed with `blend` to dodge the `0 * inf = NaN` of the
        // multiply path.
        let (zx, zy, zz) = (ray.dir_x == 0.0, ray.dir_y == 0.0, ray.dir_z == 0.0);
        let axis = |mn: f64x4, mx: f64x4, o: f64x4, inv: f64x4, degenerate: bool| {
            if degenerate {
                let inside = mn.simd_le(o) & o.simd_le(mx);
                (
                    inside.blend(neg_inf, pos_inf),
                    inside.blend(pos_inf, neg_inf),
                )
            } else {
                let t1 = (mn - o) * inv;
                let t2 = (mx - o) * inv;
                (t1.fast_min(t2), t1.fast_max(t2))
            }
        };

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let child_level = level.wrapping_sub(1);

            let mut pos = node_index;
            while pos + 4 <= end {
                let (nx, fx) = axis(
                    load4(&self.min_xs, pos),
                    load4(&self.max_xs, pos),
                    ox,
                    ix,
                    zx,
                );
                let (ny, fy) = axis(
                    load4(&self.min_ys, pos),
                    load4(&self.max_ys, pos),
                    oy,
                    iy,
                    zy,
                );
                let (nz, fz) = axis(
                    load4(&self.min_zs, pos),
                    load4(&self.max_zs, pos),
                    oz,
                    iz,
                    zz,
                );
                let near = nx.fast_max(ny).fast_max(nz).fast_max(zero);
                let far = fx.fast_min(fy).fast_min(fz).fast_min(maxd);
                let mut bits = near.simd_le(far).to_bitmask();
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let index = self.indices[pos + k];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
                    }
                }
                pos += 4;
            }
            while pos < end {
                if ray.intersects_box(self.box_at_soa(pos)) {
                    let index = self.indices[pos];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
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

    /// AVX-512 all-hits slab test, eight children at a time. Only called for
    /// non-degenerate rays (no zero direction component), so the multiply-only
    /// slab is NaN-safe.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn raycast_collect_avx512(
        &self,
        ray: Ray3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        use std::arch::x86_64::*;

        results.clear();
        stack.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return;
        }

        let ox = _mm512_set1_pd(ray.origin.x);
        let oy = _mm512_set1_pd(ray.origin.y);
        let oz = _mm512_set1_pd(ray.origin.z);
        let ix = _mm512_set1_pd(ray.inv_dir_x);
        let iy = _mm512_set1_pd(ray.inv_dir_y);
        let iz = _mm512_set1_pd(ray.inv_dir_z);
        let zero = _mm512_setzero_pd();
        let maxd = _mm512_set1_pd(ray.max_distance);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let child_level = level.wrapping_sub(1);
            if is_leaf {
                results.reserve(end - node_index);
            }

            let mut pos = node_index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end <= len`, so all eight lanes are in bounds.
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
                let t1x = _mm512_mul_pd(_mm512_sub_pd(mnx, ox), ix);
                let t2x = _mm512_mul_pd(_mm512_sub_pd(mxx, ox), ix);
                let t1y = _mm512_mul_pd(_mm512_sub_pd(mny, oy), iy);
                let t2y = _mm512_mul_pd(_mm512_sub_pd(mxy, oy), iy);
                let t1z = _mm512_mul_pd(_mm512_sub_pd(mnz, oz), iz);
                let t2z = _mm512_mul_pd(_mm512_sub_pd(mxz, oz), iz);
                let near = _mm512_max_pd(
                    _mm512_max_pd(
                        _mm512_max_pd(_mm512_min_pd(t1x, t2x), _mm512_min_pd(t1y, t2y)),
                        _mm512_min_pd(t1z, t2z),
                    ),
                    zero,
                );
                let far = _mm512_min_pd(
                    _mm512_min_pd(
                        _mm512_min_pd(_mm512_max_pd(t1x, t2x), _mm512_max_pd(t1y, t2y)),
                        _mm512_max_pd(t1z, t2z),
                    ),
                    maxd,
                );
                let mut bits: u8 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(near, far);
                if is_leaf {
                    // VPCOMPRESSQ pack the hit indices (capacity reserved above).
                    // SAFETY: `pos + 8 <= end <= indices.len()`; `results` has
                    // `end - node_index` slack.
                    unsafe {
                        let dst = results.as_mut_ptr().add(results.len()) as *mut i64;
                        let vidx = _mm512_loadu_epi64(self.indices.as_ptr().add(pos) as *const i64);
                        _mm512_mask_compressstoreu_epi64(dst, bits, vidx);
                        results.set_len(results.len() + bits.count_ones() as usize);
                    }
                } else {
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        stack.push(self.indices[pos + k]);
                        stack.push(child_level);
                    }
                }
                pos += 8;
            }
            while pos < end {
                if ray.intersects_box(self.box_at_soa(pos)) {
                    let index = self.indices[pos];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
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

    /// AVX2 all-hits raycast (4-wide slab test, AVX2 left-pack leaf collection).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn raycast_collect_avx2(
        &self,
        ray: Ray3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        use std::arch::x86_64::*;

        results.clear();
        stack.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return;
        }

        let ox = _mm256_set1_pd(ray.origin.x);
        let oy = _mm256_set1_pd(ray.origin.y);
        let oz = _mm256_set1_pd(ray.origin.z);
        let ix = _mm256_set1_pd(ray.inv_dir_x);
        let iy = _mm256_set1_pd(ray.inv_dir_y);
        let iz = _mm256_set1_pd(ray.inv_dir_z);
        let zero = _mm256_setzero_pd();
        let maxd = _mm256_set1_pd(ray.max_distance);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let child_level = level.wrapping_sub(1);
            if is_leaf {
                results.reserve(end - node_index + 4);
            }

            let mut pos = node_index;
            while pos + 4 <= end {
                // SAFETY: `pos + 4 <= end <= len`, so all four lanes are in bounds.
                let (mnx, mxx, mny, mxy, mnz, mxz) = unsafe {
                    (
                        _mm256_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.min_zs.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.max_zs.as_ptr().add(pos)),
                    )
                };
                let t1x = _mm256_mul_pd(_mm256_sub_pd(mnx, ox), ix);
                let t2x = _mm256_mul_pd(_mm256_sub_pd(mxx, ox), ix);
                let t1y = _mm256_mul_pd(_mm256_sub_pd(mny, oy), iy);
                let t2y = _mm256_mul_pd(_mm256_sub_pd(mxy, oy), iy);
                let t1z = _mm256_mul_pd(_mm256_sub_pd(mnz, oz), iz);
                let t2z = _mm256_mul_pd(_mm256_sub_pd(mxz, oz), iz);
                let near = _mm256_max_pd(
                    _mm256_max_pd(
                        _mm256_max_pd(_mm256_min_pd(t1x, t2x), _mm256_min_pd(t1y, t2y)),
                        _mm256_min_pd(t1z, t2z),
                    ),
                    zero,
                );
                let far = _mm256_min_pd(
                    _mm256_min_pd(
                        _mm256_min_pd(_mm256_max_pd(t1x, t2x), _mm256_max_pd(t1y, t2y)),
                        _mm256_max_pd(t1z, t2z),
                    ),
                    maxd,
                );
                let mut bits = _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LE_OQ>(near, far)) as usize;
                if is_leaf {
                    if bits != 0 {
                        // SAFETY: `pos + 4 <= end <= indices.len()`; `results` has
                        // `end - node_index + 4` slack reserved.
                        unsafe {
                            let added = leftpack4(
                                self.indices.as_ptr().add(pos),
                                bits as u32,
                                results.as_mut_ptr().add(results.len()),
                            );
                            results.set_len(results.len() + added);
                        }
                    }
                } else {
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        stack.push(self.indices[pos + k]);
                        stack.push(child_level);
                    }
                }
                pos += 4;
            }
            while pos < end {
                if ray.intersects_box(self.box_at_soa(pos)) {
                    let index = self.indices[pos];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
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

impl SimdIndex3D {
    /// Visit items in nondecreasing entry-`t` order along the ray segment.
    ///
    /// The visitor receives `(item index, entry t)`. Return
    /// [`ControlFlow::Break`] to stop early - for example after the first N
    /// occluders. `t` is `0.0` when the ray origin starts inside a box.
    pub fn visit_raycast<B, F>(&self, ray: Ray3D, mut visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.min_xs.len() - 1;
        loop {
            let upper = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos)) {
                    queue.push(NeighborState::new(self.indices[pos], is_leaf, t));
                }
            }

            let mut continue_search = false;
            while let Some(state) = queue.pop() {
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
}

impl SimdIndex3DView<'_> {
    /// Return the indices of all items whose boxes the ray segment touches.
    pub fn raycast(&self, ray: Ray3D) -> Vec<usize> {
        let mut results = Vec::new();
        self.raycast_into(ray, &mut results);
        results
    }

    /// Raycast with a reusable result buffer.
    pub fn raycast_into(&self, ray: Ray3D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.raycast_into_stack(ray, results, &mut stack);
    }

    /// Raycast with reusable result and traversal buffers.
    pub fn raycast_with<'na>(
        &self,
        ray: Ray3D,
        workspace: &'na mut SearchWorkspace,
    ) -> &'na [usize] {
        self.raycast_into_stack(ray, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Buffer-explicit raycast (mirrors `search_into_stack`).
    #[doc(hidden)]
    pub fn raycast_into_stack(&self, ray: Ray3D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        results.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }

        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                if !ray.intersects_box(self.box_at(pos)) {
                    continue;
                }
                let index = self.index_at(pos);
                if is_leaf {
                    results.push(index);
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

    /// Return the nearest item whose box the ray segment enters, as
    /// `(item index, entry t)`, or `None` when the segment hits nothing.
    /// See [`Index3D::raycast_closest`](crate::Index3D::raycast_closest).
    pub fn raycast_closest(&self, ray: Ray3D) -> Option<(usize, f64)> {
        let mut workspace = NeighborWorkspace::new();
        self.raycast_closest_with(ray, &mut workspace)
    }

    /// Closest-hit raycast with a reusable priority-queue workspace.
    pub fn raycast_closest_with(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        let queue = &mut workspace.node_queue;
        queue.clear();
        if self.num_items == 0 {
            return None;
        }
        let root = self.num_nodes - 1;
        let root_t = ray.enter_t(self.box_at(root))?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        while let Some(node) = queue.pop() {
            // The heap yields nodes by ascending entry t, and a node's entry t is a
            // lower bound on every descendant's, so once it reaches the best hit we stop.
            if node.dist >= best_t {
                break;
            }
            let node_index = node.index;
            let upper = self.upper_bound_level(node_index);
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(upper));
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                let Some(t) = ray.enter_t(self.box_at(pos)) else {
                    continue;
                };
                if t >= best_t {
                    continue;
                }
                if is_leaf {
                    best_t = t;
                    best_index = Some(self.index_at(pos));
                } else {
                    queue.push(NeighborNodeState::new(self.index_at(pos), t));
                }
            }
        }

        best_index.map(|index| (index, best_t))
    }

    /// Visit items in nondecreasing entry-`t` order along the ray segment.
    ///
    /// The visitor receives `(item index, entry t)`. Return
    /// [`ControlFlow::Break`] to stop early - for example after the first N
    /// occluders. `t` is `0.0` when the ray origin starts inside a box.
    pub fn visit_raycast<B, F>(&self, ray: Ray3D, mut visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.num_nodes - 1;
        loop {
            let upper = self.upper_bound_level(node_index);
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(upper));
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                if let Some(t) = ray.enter_t(self.box_at(pos)) {
                    queue.push(NeighborState::new(self.index_at(pos), is_leaf, t));
                }
            }

            let mut continue_search = false;
            while let Some(state) = queue.pop() {
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
}
