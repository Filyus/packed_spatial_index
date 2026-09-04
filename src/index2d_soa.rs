//! SoA index variant with SIMD searches (available with the `simd` feature).
//!
//! items are stored as four separate arrays (`min_x[]`, `min_y[]`, `max_x[]`,
//! `max_y[]`). The tree is built exactly like the AoS version; only the layout
//! and search implementation differ.

use std::{collections::BinaryHeap, ops::ControlFlow};

use wide::f64x4;

mod raycast;
mod serialization;

use crate::config::GATHER_PREFETCH_DISTANCE;
use crate::estimate::{Estimate, box_fraction_2d, estimate_core};
#[cfg(target_arch = "x86_64")]
use crate::leftpack::leftpack4;
use crate::{
    build::BuildError,
    builder2d::BuildConfig,
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::{Box2D, Overlaps2D, Point2D, fold_max, fold_min, query_covers_tree_2d},
    join::{
        DistanceTest, OverlapTest, anti_join_core, any_within_core, closest_pair_core, join_core,
        self_closest_pair_core, self_join_components_core, self_join_core, within_core,
    },
    neighbors::{NeighborNodeState, NeighborQuery2D, NeighborState, NeighborWorkspace, best_first},
    ordered::{collect_ordered, visit_ordered},
    persistence::{LoadError, parse_index, read_f64_le_unchecked, read_u64_le_unchecked},
    range::visit_region,
    ray::Ray2D,
    sort2d::{SortKeyContext, encode_sort_by_key},
    traversal::{SearchWorkspace, prefetch_read, upper_bound_level},
    tree::{TreeLayout, try_compute_tree_layout},
    tree_access::TreeAccess,
};

type Num = f64;

pub(crate) fn build_simd_index(
    config: BuildConfig,
    items: Vec<Box2D>,
) -> Result<SimdIndex2D, BuildError> {
    let node_size = config.node_size;
    let num_items = config.num_items;

    if num_items == 0 {
        return Ok(SimdIndex2D {
            node_size,
            num_items,
            level_bounds: vec![0],
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
            vec![num_items, num_items + 1],
            items,
        ));
    }

    let TreeLayout {
        level_bounds,
        num_nodes,
    } = try_compute_tree_layout(num_items, node_size)?;

    let mut min_xs = vec![0.0f64; num_nodes];
    let mut min_ys = vec![0.0f64; num_nodes];
    let mut max_xs = vec![0.0f64; num_nodes];
    let mut max_ys = vec![0.0f64; num_nodes];
    let mut indices = vec![0usize; num_nodes];

    let (mut e_min_x, mut e_min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut e_max_x, mut e_max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for b in &items {
        e_min_x = fold_min(e_min_x, b.min_x);
        e_min_y = fold_min(e_min_y, b.min_y);
        e_max_x = fold_max(e_max_x, b.max_x);
        e_max_y = fold_max(e_max_y, b.max_y);
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
            if let Some(&(_, ahead)) = order.get(slot + GATHER_PREFETCH_DISTANCE) {
                prefetch_read(items.as_ptr().wrapping_add(ahead as usize));
            }
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
                nmnx = fold_min(nmnx, min_xs[read_pos]);
                nmny = fold_min(nmny, min_ys[read_pos]);
                nmxx = fold_max(nmxx, max_xs[read_pos]);
                nmxy = fold_max(nmxy, max_ys[read_pos]);
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

        root_min_x = fold_min(root_min_x, b.min_x);
        root_min_y = fold_min(root_min_y, b.min_y);
        root_max_x = fold_max(root_max_x, b.max_x);
        root_max_y = fold_max(root_max_y, b.max_y);
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
    /// Up to `max_results` items overlapping `region`, in nondecreasing `key`
    /// order.
    ///
    /// See [`Index2D::search_ordered`](crate::Index2D::search_ordered) for the
    /// admissible-lower-bound contract the `key` must satisfy and for when to
    /// prefer this over [`search_region`](Self::search_region);
    /// [`view_depth_2d`](crate::view_depth_2d) is the ready-made key for
    /// front-to-back order.
    ///
    /// The descent is scalar here, as on every frontend: a heap yields one node
    /// at a time, so there is nothing for the SIMD kernel to widen. This exists
    /// so the ordered query is available where your index already lives, not
    /// because it is faster than the owned f64 one.
    pub fn search_ordered<Q, K>(
        &self,
        region: Q,
        key: K,
        max_results: usize,
        max_key: f64,
    ) -> Vec<usize>
    where
        Q: Overlaps2D,
        K: Fn(Box2D) -> f64,
    {
        let mut results = Vec::new();
        self.search_ordered_into(region, key, max_results, max_key, &mut results);
        results
    }

    /// [`search_ordered`](Self::search_ordered) into a reused buffer (cleared first).
    pub fn search_ordered_into<Q, K>(
        &self,
        region: Q,
        key: K,
        max_results: usize,
        max_key: f64,
        results: &mut Vec<usize>,
    ) where
        Q: Overlaps2D,
        K: Fn(Box2D) -> f64,
    {
        collect_ordered(
            self,
            |b| region.overlaps_box(b),
            key,
            max_results,
            max_key,
            results,
        );
    }

    /// Visit items of `region` in nondecreasing `key` order; the visitor receives
    /// the key and may return [`ControlFlow::Break`] to stop early.
    pub fn visit_ordered<Q, K, B, F>(
        &self,
        region: Q,
        key: K,
        max_key: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        Q: Overlaps2D,
        K: Fn(Box2D) -> f64,
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        visit_ordered(self, |b| region.overlaps_box(b), key, max_key, &mut visitor)
    }
    /// Items overlapping the region `region` — any [`Overlaps2D`] shape, such as
    /// [`Triangle2D`](crate::Triangle2D), [`ConvexPolygon2D`](crate::ConvexPolygon2D)
    /// or a [`Box2D`].
    ///
    /// The `Box2D` [`search`](Self::search) stays the fast path; this walks the
    /// tree one node at a time against your shape, so reach for it when the shape
    /// is what you mean.
    ///
    /// Allocates a fresh `Vec` per call — see [`search_region_into`](Self::search_region_into),
    /// [`count_region`](Self::count_region), [`any_region`](Self::any_region).
    pub fn search_region<Q: Overlaps2D>(&self, region: Q) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_region_into(region, &mut out);
        out
    }

    /// [`search_region`](Self::search_region) into a reused buffer (cleared first).
    pub fn search_region_into<Q: Overlaps2D>(&self, region: Q, out: &mut Vec<usize>) {
        out.clear();
        let _: ControlFlow<()> = self.visit_region(region, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Visit every item overlapping `region`; the visitor may return
    /// [`ControlFlow::Break`] to stop early.
    pub fn visit_region<B, Q, F>(&self, region: Q, visitor: F) -> ControlFlow<B>
    where
        Q: Overlaps2D,
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        visit_region(
            self,
            &mut stack,
            |b| region.overlaps_box(b),
            |b| region.contains_box(b),
            visitor,
        )
    }

    /// Return `true` if at least one item overlaps `region`.
    pub fn any_region<Q: Overlaps2D>(&self, region: Q) -> bool {
        self.visit_region(region, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Return one item overlapping `region`, if any.
    pub fn first_region<Q: Overlaps2D>(&self, region: Q) -> Option<usize> {
        match self.visit_region(region, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Count the items overlapping `region` without collecting them.
    pub fn count_region<Q: Overlaps2D>(&self, region: Q) -> usize {
        let mut count = 0usize;
        let _: ControlFlow<()> = self.visit_region(region, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
    }
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
    ///
    /// Allocates a fresh `Vec` per call. For a boolean test use [`any`](Self::any)
    /// rather than `search(..).is_empty()`; in a hot loop write into a buffer you
    /// own with [`search_into`](Self::search_into) or
    /// [`search_with`](Self::search_with); to count the hits use
    /// [`count`](Self::count) and to fold over them use [`visit`](Self::visit).
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(query, &mut out);
        out
    }

    /// Search with a reusable result buffer.
    ///
    /// This automatically dispatches to the widest available kernel at runtime:
    /// AVX-512 (`VPCOMPRESSQ` collection), then an explicit AVX2 tier (left-pack
    /// collection), then the SSE2 `wide` fallback.
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

    /// Return the number of items overlapping `query`.
    ///
    /// Counts during the traversal, so nothing is collected — prefer it to
    /// `search(query).len()`, which allocates a `Vec` to throw away.
    pub fn count(&self, query: Box2D) -> usize {
        let mut count = 0usize;
        let _: ControlFlow<()> = self.visit(query, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
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
            if let Some(index) =
                self.nearest_one_with_queue(NeighborQuery2D::Point(point), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Point(point),
            max_results,
            max_distance,
            results,
            &mut queue,
        );
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
            if let Some(index) = self.nearest_one_with_queue(
                NeighborQuery2D::Point(point),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Point(point),
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
        self.visit_neighbors_with_queue(
            NeighborQuery2D::Point(point),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    /// Return up to `max_results` item indices nearest to the box `query`.
    /// See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box(&self, query: Box2D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box_within(
        &self,
        query: Box2D,
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
        query: Box2D,
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
                self.nearest_one_with_queue(NeighborQuery2D::Box(query), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Box(query),
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
        query: Box2D,
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
                NeighborQuery2D::Box(query),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Box(query),
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
        query: Box2D,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.visit_neighbors_with_queue(
            NeighborQuery2D::Box(query),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_avx512(query, &mut stack, visitor)
    }

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`. See [`Index2D::join`](crate::Index2D::join).
    pub fn join(&self, other: &SimdIndex2D) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other`. See
    /// [`Index2D::join_with`](crate::Index2D::join_with).
    pub fn join_with<B, F>(&self, other: &SimdIndex2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, OverlapTest, visitor)
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
        self_join_core(self, OverlapTest, visitor)
    }

    /// Return every pair `(i, j)` where item `i` of `self` and item `j` of
    /// `other` lie within `max_distance` of each other. See
    /// [`Index2D::join_within`](crate::Index2D::join_within).
    pub fn join_within(&self, other: &SimdIndex2D, max_distance: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_within_with(other, max_distance, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every pair within `max_distance` between `self` and `other`. See
    /// [`Index2D::join_within_with`](crate::Index2D::join_within_with).
    pub fn join_within_with<B, F>(
        &self,
        other: &SimdIndex2D,
        max_distance: f64,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, DistanceTest::new(max_distance), visitor)
    }

    /// Return the ids of every item whose box lies within `max_distance` of
    /// `query`: the Euclidean distance between the two boxes is at most
    /// `max_distance`, zero when they overlap (edges are inclusive).
    ///
    /// This is the radius query — "everything within 500 m of here" — the
    /// single-index sibling of [`SimdIndex2D::join_within`]. Like every query here it
    /// is a broad phase: the box distance is a lower bound on the true
    /// distance between the underlying geometries, so hits are candidates and
    /// an exact predicate stays with the caller.
    ///
    /// A negative or NaN `max_distance` matches nothing, and `max_distance = 0.0`
    /// answers exactly [`SimdIndex2D::search`]. Result order is traversal order and is
    /// not part of the API.
    ///
    /// Allocates a fresh `Vec` per call — see
    /// [`search_within_into`](SimdIndex2D::search_within_into),
    /// [`any_within`](SimdIndex2D::any_within).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder};
    ///
    /// let mut builder = Index2DBuilder::new(3);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(3.0, 0.0, 4.0, 1.0));
    /// builder.add(Box2D::new(50.0, 50.0, 51.0, 51.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut hits = index.search_within(Box2D::new(0.5, 0.5, 1.0, 1.0), 2.0);
    /// hits.sort_unstable();
    /// // item 1 is exactly 2.0 away, and the bound is inclusive.
    /// assert_eq!(hits, vec![0, 1]);
    /// ```
    pub fn search_within(&self, query: Box2D, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_within_into(query, max_distance, &mut out);
        out
    }

    /// [`search_within`](Self::search_within) into a reused buffer (cleared
    /// first).
    pub fn search_within_into(&self, query: Box2D, max_distance: f64, out: &mut Vec<usize>) {
        out.clear();
        let _: ControlFlow<()> = self.visit_within(query, max_distance, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Visit every item within `max_distance` of `query` without collecting a
    /// result `Vec`. See [`search_within`](Self::search_within).
    ///
    /// Return [`ControlFlow::Break`] for early exit.
    pub fn visit_within<B, F>(&self, query: Box2D, max_distance: f64, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        within_core(
            self,
            query,
            DistanceTest::new(max_distance),
            &mut stack,
            visitor,
        )
    }

    /// Return `true` when at least one item lies within `max_distance` of `query`.
    ///
    /// Stops at the first hit, so it takes the prune-only descent: no
    /// whole-subtree accept is computed. See
    /// [`search_within`](Self::search_within).
    pub fn any_within(&self, query: Box2D, max_distance: f64) -> bool {
        any_within_core(self, query, DistanceTest::new(max_distance))
    }

    /// Count the items within `max_distance` of `query`.
    ///
    /// The same traversal as [`visit_within`](Self::visit_within) with a
    /// counter in place of a buffer, so nothing is allocated. Mirrors
    /// [`count`](Self::count) for the overlap query.
    pub fn count_within(&self, query: Box2D, max_distance: f64) -> usize {
        let mut count = 0usize;
        let _: ControlFlow<()> = self.visit_within(query, max_distance, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
    }

    /// Bracket and estimate how many items `query` would hit, from node boxes
    /// alone. See [`Estimate`] and the crate's `estimate` module docs.
    ///
    /// Nodes the window contains count whole, nodes it misses are dropped,
    /// and nodes it cuts are expanded while their level is above `stop_level`
    /// and scored by the fraction of their box inside the window once it is
    /// not. `stop_level = 0` examines leaf boxes, so `lower == upper == count`;
    /// `stop_level = 1` never touches a leaf and is the cheapest bracket that
    /// still resolves single nodes. Levels count up from the leaves.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(64).node_size(4);
    /// for i in 0..64 {
    ///     let v = i as f64;
    ///     builder.add(Box2D::new(v, v, v + 0.5, v + 0.5));
    /// }
    /// let index = builder.finish_simd().unwrap();
    ///
    /// let window = Box2D::new(10.0, 10.0, 30.2, 30.2);
    /// let exact = index.count(window);
    /// let est = index.estimate_count(window, 1);
    /// assert!(est.lower <= exact && exact <= est.upper);
    /// assert!(est.lower as f64 <= est.estimate && est.estimate <= est.upper as f64);
    /// assert_eq!(index.estimate_count(window, 0).lower, exact);
    /// ```
    pub fn estimate_count(&self, query: Box2D, stop_level: usize) -> Estimate {
        estimate_core(
            self,
            stop_level,
            |node| node.overlaps(query),
            |node| query.contains(node),
            |node| box_fraction_2d(node, query),
        )
    }

    /// Return every unordered pair of distinct items within this index whose
    /// boxes lie within `max_distance` of each other, each pair exactly once. See
    /// [`Index2D::self_join_within`](crate::Index2D::self_join_within).
    pub fn self_join_within(&self, max_distance: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.self_join_within_with(max_distance, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every unordered pair of distinct items within this index whose
    /// boxes lie within `max_distance` of each other. See
    /// [`Index2D::self_join_within_with`](crate::Index2D::self_join_within_with).
    pub fn self_join_within_with<B, F>(&self, max_distance: f64, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, DistanceTest::new(max_distance), visitor)
    }

    /// Return the ids of items of `self` with no item of `other` within
    /// `max_distance`. See [`Index2D::anti_join_within`](crate::Index2D::anti_join_within).
    pub fn anti_join_within(&self, other: &SimdIndex2D, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.anti_join_within_with(other, max_distance, |i| {
            out.push(i);
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every item of `self` with no item of `other` within `max_distance`.
    /// See [`Index2D::anti_join_within_with`](crate::Index2D::anti_join_within_with).
    pub fn anti_join_within_with<B, F>(
        &self,
        other: &SimdIndex2D,
        max_distance: f64,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        anti_join_core(self, other, DistanceTest::new(max_distance), visitor)
    }

    /// Label every item with the smallest item id in its component of the
    /// `max_distance`-proximity graph. See
    /// [`Index2D::self_join_within_components`](crate::Index2D::self_join_within_components).
    pub fn self_join_within_components(&self, max_distance: f64) -> Vec<usize> {
        self_join_components_core(self, DistanceTest::new(max_distance))
    }

    /// Return the closest pair of items between `self` and `other` as
    /// `(item_of_self, item_of_other, distance)`, or `None` when either index
    /// is empty.
    ///
    /// The one-answer end of the distance family: where
    /// [`SimdIndex2D::join_within`] needs an `max_distance` and reports every pair inside
    /// it, this reports the single nearest pair with no bound to guess. The
    /// traversal is best-first over node pairs and stops as soon as nothing
    /// left on the frontier can beat the pair already found.
    ///
    /// The distance is between boxes (`sqrt` of
    /// [`Box2D::distance_squared_to_box`](crate::Box2D::distance_squared_to_box)),
    /// zero when they overlap — a broad phase, like every query here, so it is
    /// a lower bound on the distance between the underlying geometries. Which
    /// pair is reported among several at the same distance is traversal order
    /// and is not part of the API.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder};
    ///
    /// let mut a = Index2DBuilder::new(1);
    /// a.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// let a = a.finish().unwrap();
    ///
    /// let mut b = Index2DBuilder::new(2);
    /// b.add(Box2D::new(90.0, 0.0, 91.0, 1.0));
    /// b.add(Box2D::new(3.0, 0.0, 4.0, 1.0));
    /// let b = b.finish().unwrap();
    ///
    /// assert_eq!(a.closest_pair(&b), Some((0, 1, 2.0)));
    /// ```
    pub fn closest_pair(&self, other: &SimdIndex2D) -> Option<(usize, usize, f64)> {
        closest_pair_core(self, other)
    }

    /// Return the closest pair of *distinct* items within this index as
    /// `(i, j, distance)`, or `None` for fewer than two items.
    ///
    /// See [`SimdIndex2D::closest_pair`] for the distance semantics. An item is never
    /// paired with itself; the order of the two ids, and which of several
    /// equally close pairs is reported, are traversal order and not part of
    /// the API.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder};
    ///
    /// let mut builder = Index2DBuilder::new(3);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(3.0, 0.0, 4.0, 1.0));
    /// builder.add(Box2D::new(3.5, 0.0, 4.5, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// // Items 1 and 2 overlap, so they are zero apart.
    /// let (i, j, distance) = index.self_closest_pair().unwrap();
    /// assert_eq!((i.min(j), i.max(j)), (1, 2));
    /// assert_eq!(distance, 0.0);
    /// ```
    pub fn self_closest_pair(&self) -> Option<(usize, usize, f64)> {
        self_closest_pair_core(self)
    }

    fn collect_neighbors_with_queue(
        &self,
        query: NeighborQuery2D,
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
        query: NeighborQuery2D,
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
        query: NeighborQuery2D,
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
    fn distance_squared_at(&self, pos: usize, query: NeighborQuery2D) -> f64 {
        query.distance_squared_to(Box2D::new(
            self.min_xs[pos],
            self.min_ys[pos],
            self.max_xs[pos],
            self.max_ys[pos],
        ))
    }

    /// Total extent box, stored as the last node. Callers must ensure the index is
    /// non-empty.
    #[inline]
    fn root_box(&self) -> Box2D {
        let last = self.min_xs.len() - 1;
        Box2D::new(
            self.min_xs[last],
            self.min_ys[last],
            self.max_xs[last],
            self.max_ys[last],
        )
    }

    /// True when `query` fully contains the box stored at `pos`.
    ///
    /// The AVX2 and AVX-512 kernels below fold this into a second lane mask off the
    /// vectors the overlap test already loaded, and the portable `wide::f64x4`
    /// kernels deliberately do **not**. Porting the mask here was tried and
    /// measured worse: callgrind, 1000 queries over 100k boxes, `search_simd`,
    /// +9.9% instructions and +8.6% data refs in 2D, +13.7% / +8.3% in 3D. Four
    /// (2D) or six (3D) `f64x4` column vectors plus the query splats do not fit
    /// baseline SSE2's sixteen `xmm` registers once they have to stay live across
    /// the lane-extraction loop, and the extra data refs are the spills. The
    /// scalar test below is only reached for a hit *internal* child and
    /// short-circuits on the first failing axis, so it is the cheaper shape here.
    /// The intrinsic kernels do not have the problem because they run 256/512-bit
    /// lanes, one register per column.
    #[inline]
    fn query_contains_node(&self, query: Box2D, pos: usize) -> bool {
        query.min_x <= self.min_xs[pos]
            && self.max_xs[pos] <= query.max_x
            && query.min_y <= self.min_ys[pos]
            && self.max_ys[pos] <= query.max_y
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
        query: Box2D,
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
        if query_covers_tree_2d(query, self.root_box()) {
            for &index in &self.indices[..self.num_items] {
                visitor(index)?;
            }
            return ControlFlow::Continue(());
        }
        let qmxx_v = _mm256_set1_pd(query.max_x);
        let qmnx_v = _mm256_set1_pd(query.min_x);
        let qmxy_v = _mm256_set1_pd(query.max_y);
        let qmny_v = _mm256_set1_pd(query.min_y);

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
                    let (mnx, mxx, mny, mxy) = unsafe {
                        (
                            _mm256_loadu_pd(self.min_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.min_ys.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        )
                    };
                    let overlap = _mm256_and_pd(
                        _mm256_and_pd(
                            _mm256_cmp_pd::<_CMP_LE_OQ>(mnx, qmxx_v),
                            _mm256_cmp_pd::<_CMP_GE_OQ>(mxx, qmnx_v),
                        ),
                        _mm256_and_pd(
                            _mm256_cmp_pd::<_CMP_LE_OQ>(mny, qmxy_v),
                            _mm256_cmp_pd::<_CMP_GE_OQ>(mxy, qmny_v),
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
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mnx, qmnx_v),
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mxx, qmxx_v),
                            ),
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mny, qmny_v),
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mxy, qmxy_v),
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
    pub fn search_scalar(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        if query_covers_tree_2d(query, self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
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
        if query_covers_tree_2d(query, self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }
        let qmxx_v = f64x4::splat(query.max_x);
        let qmnx_v = f64x4::splat(query.min_x);
        let qmxy_v = f64x4::splat(query.max_y);
        let qmny_v = f64x4::splat(query.min_y);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                self.extend_contained_leaf_indices(node_index, end, level, out);
            } else {
                // Guarded against underflow for a single leaf-level node (`level == 0`);
                // `child_level` is only read on the internal-node push paths.
                let child_level = if is_leaf { 0 } else { level - 1 };
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
                    // Drain the set bits instead of testing all four lanes. This kernel
                    // always processes every hit, so the `tzcnt` never lands on a critical
                    // path — unlike `visit_simd_impl`, whose visitor may break, where the
                    // same rewrite measured 11% SLOWER on `index3d_simd_search/
                    // uniform_simd_any_wide4` despite fewer instructions, branches and
                    // mispredicts. Leave that one on the four-lane form.
                    let mut rest = bits;
                    while rest != 0 {
                        let k = rest.trailing_zeros() as usize;
                        rest &= rest - 1;
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
                if PREFETCH {
                    self.prefetch_node(stack[stack.len() - 2]);
                }
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
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
        if query_covers_tree_2d(query, self.root_box()) {
            for &index in &self.indices[..self.num_items] {
                visitor(index)?;
            }
            return ControlFlow::Continue(());
        }
        let qmxx_v = f64x4::splat(query.max_x);
        let qmnx_v = f64x4::splat(query.min_x);
        let qmxy_v = f64x4::splat(query.max_y);
        let qmny_v = f64x4::splat(query.min_y);

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
                if PREFETCH {
                    self.prefetch_node(stack[stack.len() - 2]);
                }
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
        if query_covers_tree_2d(query, self.root_box()) {
            for &index in &self.indices[..self.num_items] {
                visitor(index)?;
            }
            return ControlFlow::Continue(());
        }
        let qmxx_v = _mm512_set1_pd(query.max_x);
        let qmnx_v = _mm512_set1_pd(query.min_x);
        let qmxy_v = _mm512_set1_pd(query.max_y);
        let qmny_v = _mm512_set1_pd(query.min_y);

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
                    if is_leaf {
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            visitor(self.indices[pos + k])?;
                            bits &= bits - 1;
                        }
                    } else {
                        // query contains child: qmin <= cmin && cmax <= qmax on both axes.
                        let c1 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mnx, qmnx_v);
                        let c2 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxx, qmxx_v);
                        let c3 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mny, qmny_v);
                        let c4 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxy, qmxy_v);
                        let cbits: u8 = c1 & c2 & c3 & c4;
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
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: selected only after checking avx2 availability.
                unsafe { self.search_avx2_impl(query, out, stack) };
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
        if query_covers_tree_2d(query, self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }
        let qmxx_v = _mm512_set1_pd(query.max_x);
        let qmnx_v = _mm512_set1_pd(query.min_x);
        let qmxy_v = _mm512_set1_pd(query.max_y);
        let qmny_v = _mm512_set1_pd(query.min_y);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if contained {
                self.extend_contained_leaf_indices(node_index, end, level, out);
            } else {
                // Guarded against underflow for a single leaf-level node (`level == 0`);
                // `child_level` is only read on the internal-node push paths.
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
                        // query contains child: qmin <= cmin && cmax <= qmax on both axes.
                        let c1 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mnx, qmnx_v);
                        let c2 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxx, qmxx_v);
                        let c3 = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(mny, qmny_v);
                        let c4 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mxy, qmxy_v);
                        let cbits: u8 = c1 & c2 & c3 & c4;
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

    /// AVX2 (256-bit, 4 boxes/chunk) range search — the runtime tier between the
    /// `wide` fallback and AVX-512. AVX2 has no `VPCOMPRESSQ`, so leaf results are
    /// collected with an AVX2 **left-pack** (`VPERMD` over a 16-entry shuffle LUT)
    /// that packs the matching `u64` indices in one permute. Doc-hidden; reached
    /// through the dispatch on AVX2-but-not-AVX-512 CPUs.
    #[doc(hidden)]
    pub fn search_avx2(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
    unsafe fn search_avx2_impl(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        use std::arch::x86_64::*;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        if query_covers_tree_2d(query, self.root_box()) {
            out.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }
        let qmxx_v = _mm256_set1_pd(query.max_x);
        let qmnx_v = _mm256_set1_pd(query.min_x);
        let qmxy_v = _mm256_set1_pd(query.max_y);
        let qmny_v = _mm256_set1_pd(query.min_y);

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
                // Slack for the unconditional 4-wide left-pack store (writes 4
                // u64 past `len` regardless of popcount).
                if is_leaf {
                    out.reserve(end - node_index + 4);
                }
                let mut pos = node_index;
                while pos + 4 <= end {
                    // SAFETY: `pos + 4 <= end`, and `end` is bounded by the array length.
                    let (mnx, mxx, mny, mxy) = unsafe {
                        (
                            _mm256_loadu_pd(self.min_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_xs.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.min_ys.as_ptr().add(pos)),
                            _mm256_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        )
                    };
                    let overlap = _mm256_and_pd(
                        _mm256_and_pd(
                            _mm256_cmp_pd::<_CMP_LE_OQ>(mnx, qmxx_v),
                            _mm256_cmp_pd::<_CMP_GE_OQ>(mxx, qmnx_v),
                        ),
                        _mm256_and_pd(
                            _mm256_cmp_pd::<_CMP_LE_OQ>(mny, qmxy_v),
                            _mm256_cmp_pd::<_CMP_GE_OQ>(mxy, qmny_v),
                        ),
                    );
                    let mut bits = _mm256_movemask_pd(overlap) as usize;
                    if is_leaf {
                        if bits != 0 {
                            // AVX2 left-pack the matching index lanes (capacity
                            // reserved above). SAFETY: `pos + 4 <= end <=
                            // indices.len()`; `out` has `end - node_index + 4` slack.
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
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mnx, qmnx_v),
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mxx, qmxx_v),
                            ),
                            _mm256_and_pd(
                                _mm256_cmp_pd::<_CMP_GE_OQ>(mny, qmny_v),
                                _mm256_cmp_pd::<_CMP_LE_OQ>(mxy, qmxy_v),
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
    // One range check, not four: `a[p..p + 4]` gives LLVM a length it can see, and
    // the array conversion below cannot fail because that length is exactly four.
    let four: [f64; 4] = a[p..p + 4].try_into().unwrap();
    f64x4::from(four)
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
    /// Derived at load (not stored), so owned rather than borrowed.
    level_bounds: Vec<usize>,
    entries: &'a [u8],
    indices: &'a [u8],
}

impl<'a> SimdIndex2DView<'a> {
    /// Up to `max_results` items overlapping `region`, in nondecreasing `key`
    /// order.
    ///
    /// See [`Index2D::search_ordered`](crate::Index2D::search_ordered) for the
    /// admissible-lower-bound contract the `key` must satisfy and for when to
    /// prefer this over [`search_region`](Self::search_region);
    /// [`view_depth_2d`](crate::view_depth_2d) is the ready-made key for
    /// front-to-back order.
    ///
    /// The descent is scalar here, as on every frontend: a heap yields one node
    /// at a time, so there is nothing for the SIMD kernel to widen. This exists
    /// so the ordered query is available where your index already lives, not
    /// because it is faster than the owned f64 one.
    pub fn search_ordered<Q, K>(
        &self,
        region: Q,
        key: K,
        max_results: usize,
        max_key: f64,
    ) -> Vec<usize>
    where
        Q: Overlaps2D,
        K: Fn(Box2D) -> f64,
    {
        let mut results = Vec::new();
        self.search_ordered_into(region, key, max_results, max_key, &mut results);
        results
    }

    /// [`search_ordered`](Self::search_ordered) into a reused buffer (cleared first).
    pub fn search_ordered_into<Q, K>(
        &self,
        region: Q,
        key: K,
        max_results: usize,
        max_key: f64,
        results: &mut Vec<usize>,
    ) where
        Q: Overlaps2D,
        K: Fn(Box2D) -> f64,
    {
        collect_ordered(
            self,
            |b| region.overlaps_box(b),
            key,
            max_results,
            max_key,
            results,
        );
    }

    /// Visit items of `region` in nondecreasing `key` order; the visitor receives
    /// the key and may return [`ControlFlow::Break`] to stop early.
    pub fn visit_ordered<Q, K, B, F>(
        &self,
        region: Q,
        key: K,
        max_key: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        Q: Overlaps2D,
        K: Fn(Box2D) -> f64,
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        visit_ordered(self, |b| region.overlaps_box(b), key, max_key, &mut visitor)
    }
    /// Items overlapping the region `region` — any [`Overlaps2D`] shape, such as
    /// [`Triangle2D`](crate::Triangle2D), [`ConvexPolygon2D`](crate::ConvexPolygon2D)
    /// or a [`Box2D`].
    ///
    /// The `Box2D` [`search`](Self::search) stays the fast path; this walks the
    /// tree one node at a time against your shape, so reach for it when the shape
    /// is what you mean.
    ///
    /// Allocates a fresh `Vec` per call — see [`search_region_into`](Self::search_region_into),
    /// [`count_region`](Self::count_region), [`any_region`](Self::any_region).
    pub fn search_region<Q: Overlaps2D>(&self, region: Q) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_region_into(region, &mut out);
        out
    }

    /// [`search_region`](Self::search_region) into a reused buffer (cleared first).
    pub fn search_region_into<Q: Overlaps2D>(&self, region: Q, out: &mut Vec<usize>) {
        out.clear();
        let _: ControlFlow<()> = self.visit_region(region, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Visit every item overlapping `region`; the visitor may return
    /// [`ControlFlow::Break`] to stop early.
    pub fn visit_region<B, Q, F>(&self, region: Q, visitor: F) -> ControlFlow<B>
    where
        Q: Overlaps2D,
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        visit_region(
            self,
            &mut stack,
            |b| region.overlaps_box(b),
            |b| region.contains_box(b),
            visitor,
        )
    }

    /// Return `true` if at least one item overlaps `region`.
    pub fn any_region<Q: Overlaps2D>(&self, region: Q) -> bool {
        self.visit_region(region, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Return one item overlapping `region`, if any.
    pub fn first_region<Q: Overlaps2D>(&self, region: Q) -> Option<usize> {
        match self.visit_region(region, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Count the items overlapping `region` without collecting them.
    pub fn count_region<Q: Overlaps2D>(&self, region: Q) -> usize {
        let mut count = 0usize;
        let _: ControlFlow<()> = self.visit_region(region, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
    }
    /// Borrow a zero-copy view over the canonical `PSINDEX` 2D bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 2, 8)?;
        if payload.is_some() {
            return Err(LoadError::PayloadNotSupported);
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
    ///
    /// Allocates a fresh `Vec` per call. For a boolean test use [`any`](Self::any)
    /// rather than `search(..).is_empty()`; in a hot loop write into a buffer you
    /// own with [`search_into`](Self::search_into) or
    /// [`search_with`](Self::search_with); to count the hits use
    /// [`count`](Self::count) and to fold over them use [`visit`](Self::visit).
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

    /// Return the number of items overlapping `query`.
    ///
    /// Counts during the traversal, so nothing is collected — prefer it to
    /// `search(query).len()`, which allocates a `Vec` to throw away.
    pub fn count(&self, query: Box2D) -> usize {
        let mut count = 0usize;
        let _: ControlFlow<()> = self.visit(query, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
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
        if query_covers_tree_2d(query, self.box_at(self.num_nodes - 1)) {
            for pos in 0..self.num_items {
                visitor(self.index_at(pos))?;
            }
            return ControlFlow::Continue(());
        }
        let qmxx = f64x4::splat(query.max_x);
        let qmnx = f64x4::splat(query.min_x);
        let qmxy = f64x4::splat(query.max_y);
        let qmny = f64x4::splat(query.min_y);

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
                    let base = pos * RECORD_2D;
                    let mnx = lane4_2d(self.entries, base, 0);
                    let mxx = lane4_2d(self.entries, base, 16);
                    let mny = lane4_2d(self.entries, base, 8);
                    let mxy = lane4_2d(self.entries, base, 24);
                    let mask = mnx.simd_le(qmxx)
                        & mxx.simd_ge(qmnx)
                        & mny.simd_le(qmxy)
                        & mxy.simd_ge(qmny);
                    let bits = mask.to_bitmask();
                    if bits != 0 {
                        // query contains child: qmin <= cmin && cmax <= qmax on both axes.
                        let cmask = mnx.simd_ge(qmnx)
                            & mxx.simd_le(qmxx)
                            & mny.simd_ge(qmny)
                            & mxy.simd_le(qmxy);
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
    fn distance_squared_at(&self, pos: usize, query: NeighborQuery2D) -> f64 {
        query.distance_squared_to(self.box_at(pos))
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
            if let Some(index) =
                self.nearest_one_with_queue(NeighborQuery2D::Point(point), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Point(point),
            max_results,
            max_distance,
            results,
            &mut queue,
        );
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
            if let Some(index) = self.nearest_one_with_queue(
                NeighborQuery2D::Point(point),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }
        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Point(point),
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
        self.visit_neighbors_with_queue(
            NeighborQuery2D::Point(point),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    /// Return up to `max_results` item indices nearest to the box `query`.
    /// See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box(&self, query: Box2D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`Index2D::neighbors_of_box`](crate::Index2D::neighbors_of_box).
    pub fn neighbors_of_box_within(
        &self,
        query: Box2D,
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
        query: Box2D,
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
                self.nearest_one_with_queue(NeighborQuery2D::Box(query), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Box(query),
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
        query: Box2D,
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
                NeighborQuery2D::Box(query),
                max_distance,
                &mut workspace.node_queue,
            ) {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        workspace.node_queue.clear();
        self.collect_neighbors_with_queue(
            NeighborQuery2D::Box(query),
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
        query: Box2D,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.visit_neighbors_with_queue(
            NeighborQuery2D::Box(query),
            max_distance,
            &mut queue,
            &mut visitor,
        )
    }

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`. See [`Index2D::join`](crate::Index2D::join).
    pub fn join(&self, other: &SimdIndex2DView<'_>) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other`. See
    /// [`Index2D::join_with`](crate::Index2D::join_with).
    pub fn join_with<B, F>(&self, other: &SimdIndex2DView<'_>, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, OverlapTest, visitor)
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
        self_join_core(self, OverlapTest, visitor)
    }

    /// Return every pair `(i, j)` where item `i` of `self` and item `j` of
    /// `other` lie within `max_distance` of each other. See
    /// [`Index2D::join_within`](crate::Index2D::join_within).
    pub fn join_within(
        &self,
        other: &SimdIndex2DView<'_>,
        max_distance: f64,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_within_with(other, max_distance, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every pair within `max_distance` between `self` and `other`. See
    /// [`Index2D::join_within_with`](crate::Index2D::join_within_with).
    pub fn join_within_with<B, F>(
        &self,
        other: &SimdIndex2DView<'_>,
        max_distance: f64,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, DistanceTest::new(max_distance), visitor)
    }

    /// Return the ids of every item of the view whose box lies within
    /// `max_distance` of `query`. See [`SimdIndex2D::search_within`].
    pub fn search_within(&self, query: Box2D, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_within_into(query, max_distance, &mut out);
        out
    }

    /// [`search_within`](Self::search_within) into a reused buffer (cleared
    /// first).
    pub fn search_within_into(&self, query: Box2D, max_distance: f64, out: &mut Vec<usize>) {
        out.clear();
        let _: ControlFlow<()> = self.visit_within(query, max_distance, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Visit every item within `max_distance` of `query` without collecting a
    /// result `Vec`. See [`search_within`](Self::search_within).
    ///
    /// Return [`ControlFlow::Break`] for early exit.
    pub fn visit_within<B, F>(&self, query: Box2D, max_distance: f64, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        within_core(
            self,
            query,
            DistanceTest::new(max_distance),
            &mut stack,
            visitor,
        )
    }

    /// Return `true` when at least one item lies within `max_distance` of `query`.
    ///
    /// Stops at the first hit, so it takes the prune-only descent: no
    /// whole-subtree accept is computed. See
    /// [`search_within`](Self::search_within).
    pub fn any_within(&self, query: Box2D, max_distance: f64) -> bool {
        any_within_core(self, query, DistanceTest::new(max_distance))
    }

    /// Count the items within `max_distance` of `query`.
    ///
    /// The same traversal as [`visit_within`](Self::visit_within) with a
    /// counter in place of a buffer, so nothing is allocated. Mirrors
    /// [`count`](Self::count) for the overlap query.
    pub fn count_within(&self, query: Box2D, max_distance: f64) -> usize {
        let mut count = 0usize;
        let _: ControlFlow<()> = self.visit_within(query, max_distance, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
    }

    /// Bracket and estimate how many items `query` would hit, from node boxes
    /// alone. See [`Estimate`] and the crate's `estimate` module docs.
    ///
    /// Nodes the window contains count whole, nodes it misses are dropped,
    /// and nodes it cuts are expanded while their level is above `stop_level`
    /// and scored by the fraction of their box inside the window once it is
    /// not. `stop_level = 0` examines leaf boxes, so `lower == upper == count`;
    /// `stop_level = 1` never touches a leaf and is the cheapest bracket that
    /// still resolves single nodes. Levels count up from the leaves.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(64).node_size(4);
    /// for i in 0..64 {
    ///     let v = i as f64;
    ///     builder.add(Box2D::new(v, v, v + 0.5, v + 0.5));
    /// }
    /// let index = builder.finish_simd().unwrap();
    ///
    /// let window = Box2D::new(10.0, 10.0, 30.2, 30.2);
    /// let exact = index.count(window);
    /// let est = index.estimate_count(window, 1);
    /// assert!(est.lower <= exact && exact <= est.upper);
    /// assert!(est.lower as f64 <= est.estimate && est.estimate <= est.upper as f64);
    /// assert_eq!(index.estimate_count(window, 0).lower, exact);
    /// ```
    pub fn estimate_count(&self, query: Box2D, stop_level: usize) -> Estimate {
        estimate_core(
            self,
            stop_level,
            |node| node.overlaps(query),
            |node| query.contains(node),
            |node| box_fraction_2d(node, query),
        )
    }

    /// Return every unordered pair of distinct items within this view whose
    /// boxes lie within `max_distance` of each other, each pair exactly once. See
    /// [`Index2D::self_join_within`](crate::Index2D::self_join_within).
    pub fn self_join_within(&self, max_distance: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.self_join_within_with(max_distance, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every unordered pair of distinct items within this view whose
    /// boxes lie within `max_distance` of each other. See
    /// [`Index2D::self_join_within_with`](crate::Index2D::self_join_within_with).
    pub fn self_join_within_with<B, F>(&self, max_distance: f64, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, DistanceTest::new(max_distance), visitor)
    }

    /// Return the ids of items of `self` with no item of `other` within
    /// `max_distance`. See [`Index2D::anti_join_within`](crate::Index2D::anti_join_within).
    pub fn anti_join_within(&self, other: &SimdIndex2DView<'_>, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.anti_join_within_with(other, max_distance, |i| {
            out.push(i);
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every item of `self` with no item of `other` within `max_distance`.
    /// See [`Index2D::anti_join_within_with`](crate::Index2D::anti_join_within_with).
    pub fn anti_join_within_with<B, F>(
        &self,
        other: &SimdIndex2DView<'_>,
        max_distance: f64,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        anti_join_core(self, other, DistanceTest::new(max_distance), visitor)
    }

    /// Label every item with the smallest item id in its component of the
    /// `max_distance`-proximity graph. See
    /// [`Index2D::self_join_within_components`](crate::Index2D::self_join_within_components).
    pub fn self_join_within_components(&self, max_distance: f64) -> Vec<usize> {
        self_join_components_core(self, DistanceTest::new(max_distance))
    }

    /// Return the closest pair of items between this view and `other`. See
    /// [`SimdIndex2D::closest_pair`].
    pub fn closest_pair(&self, other: &SimdIndex2DView<'_>) -> Option<(usize, usize, f64)> {
        closest_pair_core(self, other)
    }

    /// Return the closest pair of distinct items within this view. See
    /// [`SimdIndex2D::self_closest_pair`].
    pub fn self_closest_pair(&self) -> Option<(usize, usize, f64)> {
        self_closest_pair_core(self)
    }

    fn collect_neighbors_with_queue(
        &self,
        query: NeighborQuery2D,
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
        query: NeighborQuery2D,
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
        query: NeighborQuery2D,
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

impl TreeAccess for SimdIndex2D {
    type Bounds = Box2D;

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
    fn tree_bounds(&self, pos: usize) -> Box2D {
        Box2D::new(
            self.min_xs[pos],
            self.min_ys[pos],
            self.max_xs[pos],
            self.max_ys[pos],
        )
    }
    #[inline]
    fn tree_index(&self, pos: usize) -> usize {
        self.indices[pos]
    }
    #[inline]
    fn bounds_overlap(a: Box2D, b: Box2D) -> bool {
        a.overlaps(b)
    }
}

impl TreeAccess for SimdIndex2DView<'_> {
    type Bounds = Box2D;

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
    fn tree_bounds(&self, pos: usize) -> Box2D {
        self.box_at(pos)
    }
    #[inline]
    fn tree_index(&self, pos: usize) -> usize {
        self.index_at(pos)
    }
    #[inline]
    fn bounds_overlap(a: Box2D, b: Box2D) -> bool {
        a.overlaps(b)
    }
}

impl SimdIndex2DView<'_> {
    /// Return the indices of all items whose boxes the ray segment touches.
    pub fn raycast(&self, ray: Ray2D) -> Vec<usize> {
        let mut results = Vec::new();
        self.raycast_into(ray, &mut results);
        results
    }

    /// Raycast with a reusable result buffer.
    pub fn raycast_into(&self, ray: Ray2D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.raycast_into_stack(ray, results, &mut stack);
    }

    /// Raycast with reusable result and traversal buffers.
    pub fn raycast_with<'na>(
        &self,
        ray: Ray2D,
        workspace: &'na mut SearchWorkspace,
    ) -> &'na [usize] {
        self.raycast_into_stack(ray, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Buffer-explicit raycast (mirrors `search_into_stack`).
    #[doc(hidden)]
    pub fn raycast_into_stack(&self, ray: Ray2D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
    /// See [`Index2D::raycast_closest`](crate::Index2D::raycast_closest).
    pub fn raycast_closest(&self, ray: Ray2D) -> Option<(usize, f64)> {
        let mut workspace = NeighborWorkspace::new();
        self.raycast_closest_with(ray, &mut workspace)
    }

    /// Closest-hit raycast with a reusable priority-queue workspace.
    pub fn raycast_closest_with(
        &self,
        ray: Ray2D,
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
    pub fn visit_raycast<B, F>(&self, ray: Ray2D, mut visitor: F) -> ControlFlow<B>
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
