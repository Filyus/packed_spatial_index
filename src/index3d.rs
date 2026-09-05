//! Static spatial index implementation for 3D AABBs.
//!
//! `Index3D` mirrors the scalar `Index2D` API: build with
//! [`crate::Index3DBuilder`], then run overlap searches or exact nearest-neighbor
//! queries against the finished read-only tree.

use std::{collections::BinaryHeap, ops::ControlFlow};

use crate::estimate::{Estimate, box_fraction_3d, estimate_core};
use crate::{
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::{Box3D, Overlaps3D, Point3D},
    join::{
        DistanceTest, OverlapTest, anti_join_core, any_within_core, closest_pair_core, join_core,
        self_closest_pair_core, self_join_components_core, self_join_core, within_core,
    },
    neighbors::{
        NeighborNodeState, NeighborQuery3D, NeighborState, NeighborWorkspace, best_first,
        metric_knn,
    },
    ordered::{PickHit3D, collect_ordered, collect_pick, pick_key, visit_ordered, visit_pick},
    persistence::{
        LoadError, ParsedPayload, PayloadError, build_id_to_leaf, parse_index, parse_index_owned,
        payload_slice, read_f64_le_unchecked, read_u64_le_unchecked,
    },
    range::{visit_overlaps, visit_region},
    ray::Ray3D,
    traversal::{SearchWorkspace, prefetch_read, upper_bound_level},
    tree_access::{TreeAccess, leaf_group_range},
    triangle::{Triangle3, blobs_as_records},
};

mod raycast;
mod region;
mod serializer;
#[doc(hidden)]
pub use region::SearchQuery3D;
pub use serializer::Serializer3D;

#[inline]
fn prefetch_aos_node3d(entries: &[Box3D], indices: &[usize], node_index: usize, node_size: usize) {
    if node_index < entries.len() {
        prefetch_read(entries.as_ptr().wrapping_add(node_index));
        prefetch_read(indices.as_ptr().wrapping_add(node_index));
    }
    let next_line = node_index.saturating_add((64 / std::mem::size_of::<Box3D>()).max(1));
    if node_size > 1 && next_line < entries.len() {
        prefetch_read(entries.as_ptr().wrapping_add(next_line));
        prefetch_read(indices.as_ptr().wrapping_add(next_line));
    }
}

/// Finished static read-only 3D index.
///
/// Search methods return item positions in the original insertion order. The
/// order of returned search results is traversal order and is not part of the
/// API.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, Index3DBuilder};
///
/// let mut builder = Index3DBuilder::new(2);
/// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
/// builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
/// let index = builder.finish().unwrap();
///
/// assert_eq!(index.num_items(), 2);
/// assert_eq!(
///     index.search(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)),
///     vec![0]
/// );
/// ```
pub struct Index3D {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) entries: Vec<Box3D>,
    pub(crate) indices: Vec<usize>,
}

impl Index3D {
    /// Return the number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total extent of indexed items, or `None` for an empty index.
    pub fn extent(&self) -> Option<Box3D> {
        self.entries.last().copied()
    }

    /// Return the packed node size used by this index.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// The item ids in leaf order: the order the packed tree stores them, which
    /// is the order along the Hilbert curve the builder sorted by.
    ///
    /// Positions along this slice are the *leaf ranks* the streaming readers
    /// report; the values are the insertion ids every query returns. The
    /// order is a property of the built index and a resource in its own
    /// right — every m-th entry is a spatially stratified sample, a prefix
    /// covers the whole extent coarsely, and equal slices are compact
    /// pieces — see the guide's "The leaf order as a resource". Stable for
    /// one built index; a rebuild may order differently.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index3DBuilder, Box3D};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(9.0, 9.0, 9.0, 10.0, 10.0, 10.0));
    /// builder.add(Box3D::new(0.0, 9.0, 0.0, 1.0, 10.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let order = index.leaf_order();
    /// assert_eq!(order.len(), 3);
    /// let mut ids = order.to_vec();
    /// ids.sort_unstable();
    /// assert_eq!(ids, vec![0, 1, 2]); // a permutation of the insertion ids
    /// ```
    pub fn leaf_order(&self) -> &[usize] {
        &self.indices[..self.num_items]
    }

    /// Serialize this index into the stable little-endian `PSINDEX` 3D format.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView};
    ///
    /// let mut builder = Index3DBuilder::new(1);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// let index = builder.finish()?;
    ///
    /// let bytes = index.to_bytes();
    /// let owned = Index3D::from_bytes(&bytes)?;
    /// let view = Index3DView::from_bytes(&bytes)?;
    ///
    /// let query = Box3D::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
    /// assert_eq!(owned.search(query), view.search(query));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.to_bytes_into(&mut out);
        out
    }

    /// Serialize into a caller-provided buffer, reusing its allocation.
    ///
    /// Equivalent to [`to_bytes`](Self::to_bytes) but writes into `out` (cleared
    /// first). Reusing one buffer across many serializations avoids repeated
    /// multi-megabyte allocation and page-faulting.
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        self.serialize()
            .to_bytes_into(out)
            .expect("serialization without payloads cannot fail");
    }

    /// Serialize this index together with one opaque payload per item. The 3D
    /// counterpart of
    /// [`Index2D::to_bytes_with_payloads`](crate::Index2D::to_bytes_with_payloads).
    /// Shorthand for [`serialize().payloads(..)`](Self::serialize).
    pub fn to_bytes_with_payloads<P: AsRef<[u8]>>(
        &self,
        payloads: &[P],
    ) -> Result<Vec<u8>, PayloadError> {
        self.serialize().payloads(payloads).to_bytes()
    }

    /// [`to_bytes_with_payloads`](Self::to_bytes_with_payloads) into a reused
    /// buffer (cleared first).
    pub fn to_bytes_with_payloads_into<P: AsRef<[u8]>>(
        &self,
        payloads: &[P],
        out: &mut Vec<u8>,
    ) -> Result<(), PayloadError> {
        self.serialize().payloads(payloads).to_bytes_into(out)
    }

    /// Serialize in the **interleaved** layout (each node's box followed by its
    /// index), which halves a streaming descent's round-trip depth: a level costs
    /// one fetch instead of two dependent ones, since a node's index arrives with
    /// its box instead of being requested after the box tests pick the survivors.
    /// See [`Index2D::to_bytes_interleaved`](crate::Index2D::to_bytes_interleaved)
    /// for the measured effect. Shorthand for
    /// [`serialize().interleaved()`](Self::serialize); available with `stream`.
    #[cfg(feature = "stream")]
    pub fn to_bytes_interleaved(&self) -> Vec<u8> {
        self.serialize()
            .interleaved()
            .to_bytes()
            .expect("serialization without payloads cannot fail")
    }

    /// Interleaved layout plus one payload per item. Shorthand for
    /// [`serialize().interleaved().payloads(..)`](Self::serialize); available with
    /// `stream`.
    #[cfg(feature = "stream")]
    pub fn to_bytes_interleaved_with_payloads<P: AsRef<[u8]>>(
        &self,
        payloads: &[P],
    ) -> Result<Vec<u8>, PayloadError> {
        self.serialize().interleaved().payloads(payloads).to_bytes()
    }

    /// Start a serialization builder for fine-grained control: optional per-item
    /// payloads, the streaming-tuned interleaved layout, and descriptive metadata
    /// (CRS / content type / attribution). See [`Serializer3D`].
    pub fn serialize(&self) -> Serializer3D<'_> {
        Serializer3D::new(self)
    }

    /// Build an index over the bounding box of each triangle, in slice order
    /// (item `i` is `triangles[i]`). A convenience over looping
    /// [`Index3DBuilder::add`](crate::Index3DBuilder::add) with
    /// [`Triangle3::aabb`](crate::Triangle3::aabb); the index is queryable in memory, and
    /// `index.serialize().triangles(triangles)` stores the geometry alongside it
    /// (a streamable mesh BVH). Use the builder directly for custom boxes or build
    /// options like `node_size`.
    pub fn from_triangles<T: Triangle3>(triangles: &[T]) -> Result<Self, crate::BuildError> {
        let mut builder = crate::Index3DBuilder::new(triangles.len());
        for t in triangles {
            builder.add(t.aabb());
        }
        builder.finish()
    }

    /// Load an owned 3D index from bytes previously produced by
    /// [`Index3D::to_bytes`] or an interleaved serialization such as
    /// `to_bytes_interleaved` (stream feature).
    ///
    /// Both layouts load into the same in-memory columns; the interleaved one
    /// is transposed on the way in. An index-only writer's bytes and files
    /// carrying a payload both load — the payload is ignored, use a view or
    /// the streaming reader to read it back.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let tree = parse_index_owned(bytes, 3, 8)?;

        Ok(Self {
            node_size: tree.node_size,
            num_items: tree.num_items,
            level_bounds: tree.level_bounds,
            entries: copy_box3d_entries(&tree.entries, tree.num_nodes),
            indices: copy_u64_indices(&tree.indices, tree.num_nodes),
        })
    }

    /// Return the indices of all items whose boxes overlap `query`.
    ///
    /// Accepts a [`Box3D`] by value or borrowed region geometry implementing
    /// [`Overlaps3D`], such as [`Frustum3D`](crate::Frustum3D).
    ///
    /// Allocates a fresh `Vec` per call. For a boolean test use [`any`](Self::any)
    /// rather than `search(..).is_empty()`; in a hot loop write into a buffer you
    /// own with [`search_into`](Self::search_into) or
    /// [`search_with`](Self::search_with); to count the hits use
    /// [`count`](Self::count) and to fold over them use [`visit`](Self::visit); to
    /// stop part-way through them use the lazy [`search_iter`](Self::search_iter).
    ///
    /// # Example
    ///
    /// ```
    /// # use packed_spatial_index::{Index3DBuilder, Box3D};
    /// # let mut builder = Index3DBuilder::new(2);
    /// # builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// # builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    /// # let index = builder.finish().unwrap();
    /// assert_eq!(index.search(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)), vec![0]);
    /// ```
    pub fn search<Q: SearchQuery3D>(&self, query: Q) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(query, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into<Q: SearchQuery3D>(&self, query: Q, results: &mut Vec<usize>) {
        query.search_into_index(self, results);
    }

    /// Search with reusable result and traversal buffers.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder, SearchWorkspace};
    ///
    /// let mut builder = Index3DBuilder::new(1);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut workspace = SearchWorkspace::new();
    /// let hits = index.search_with(
    ///     Box3D::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    ///     &mut workspace,
    /// );
    /// assert_eq!(hits, &[0]);
    /// ```
    pub fn search_with<'a, Q: SearchQuery3D>(
        &self,
        query: Q,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize] {
        query.search_with_index(self, workspace)
    }

    /// Return `true` if at least one item overlaps `query`.
    ///
    /// This is an early-exit path: traversal stops at the first hit and does not
    /// allocate a result `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// # use packed_spatial_index::{Index3DBuilder, Box3D};
    /// # let mut builder = Index3DBuilder::new(2);
    /// # builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// # builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    /// # let index = builder.finish().unwrap();
    /// assert!(index.any(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)));
    /// assert!(!index.any(Box3D::new(20.0, 20.0, 20.0, 21.0, 21.0, 21.0)));
    /// ```
    pub fn any<Q: SearchQuery3D>(&self, query: Q) -> bool {
        query.any_index(self)
    }

    /// Return the number of items overlapping `query`.
    ///
    /// Counts during the traversal, so nothing is collected — prefer it to
    /// `search(query).len()`, which allocates a `Vec` to throw away.
    ///
    /// # Example
    ///
    /// ```
    /// # use packed_spatial_index::{Index3DBuilder, Box3D};
    /// # let mut builder = Index3DBuilder::new(2);
    /// # builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// # builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    /// # let index = builder.finish().unwrap();
    /// assert_eq!(index.count(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)), 1);
    /// assert_eq!(index.count(Box3D::new(20.0, 20.0, 20.0, 21.0, 21.0, 21.0)), 0);
    /// ```
    pub fn count<Q: SearchQuery3D>(&self, query: Q) -> usize {
        query.count_index(self)
    }

    /// Return one overlapping item, if any.
    ///
    /// Tree traversal order is not part of the API, so this returns just some
    /// first found item, not the minimum insertion index.
    ///
    /// # Example
    ///
    /// ```
    /// # use packed_spatial_index::{Index3DBuilder, Box3D};
    /// # let mut builder = Index3DBuilder::new(2);
    /// # builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// # builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    /// # let index = builder.finish().unwrap();
    /// assert_eq!(index.first(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)), Some(0));
    /// assert_eq!(index.first(Box3D::new(20.0, 20.0, 20.0, 21.0, 21.0, 21.0)), None);
    /// ```
    pub fn first<Q: SearchQuery3D>(&self, query: Q) -> Option<usize> {
        query.first_index(self)
    }

    /// Return up to `max_results` item indices nearest to `point`.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder, Point3D};
    ///
    /// let mut builder = Index3DBuilder::new(2);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(10.0, 10.0, 10.0, 11.0, 11.0, 11.0));
    /// let index = builder.finish().unwrap();
    ///
    /// assert_eq!(index.neighbors(Point3D::new(10.25, 10.25, 10.25), 1), vec![1]);
    /// ```
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery3D::Point(point),
            max_results,
            max_distance,
            results,
            &mut item_queue,
            &mut node_queue,
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

        self.collect_neighbors_with_queues(
            NeighborQuery3D::Point(point),
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
            &mut workspace.node_queue,
        );
        &workspace.results
    }

    /// Visit items in nondecreasing squared-distance order from `point`.
    ///
    /// The visitor receives squared distances. Return [`ControlFlow::Break`] to
    /// stop early.
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

    /// Up to `max_results` item indices nearest to your query under a custom
    /// distance `metric`, nearest first.
    ///
    /// `metric(b)` returns the distance from your query to box `b` (the query is
    /// captured by the closure). It must be an **admissible lower bound** — the
    /// distance to a box never exceeds the distance to any item inside it, which
    /// holds for any "distance to the closest point of the box" metric.
    /// `max_distance` is a cutoff in the metric's own units (not squared);
    /// `f64::INFINITY` for unbounded. The default [`neighbors`](Self::neighbors)
    /// (squared Euclidean) is faster; reach for this for another metric.
    pub fn neighbors_metric<M: Fn(Box3D) -> f64>(
        &self,
        metric: M,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        self.neighbors_metric_into(metric, max_results, max_distance, &mut results);
        results
    }

    /// [`neighbors_metric`](Self::neighbors_metric) into a reused buffer (cleared first).
    pub fn neighbors_metric_into<M: Fn(Box3D) -> f64>(
        &self,
        metric: M,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
    ) {
        results.clear();
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        metric_knn::collect_neighbors(
            self.entries.len(),
            self.num_items,
            self.node_size,
            |node| self.level_bounds[upper_bound_level(&self.level_bounds, node)],
            |pos| self.indices[pos],
            max_results,
            max_distance,
            |pos| Some(metric(self.entries[pos])),
            results,
            &mut queue,
        );
    }

    /// Visit items in nondecreasing custom-`metric` distance; the visitor receives
    /// the metric distance and may return [`ControlFlow::Break`] to stop early.
    /// See [`neighbors_metric`](Self::neighbors_metric) for the metric contract.
    pub fn visit_neighbors_metric<B, M, F>(
        &self,
        metric: M,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        M: Fn(Box3D) -> f64,
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        metric_knn::visit_neighbors(
            self.entries.len(),
            self.num_items,
            self.node_size,
            |node| self.level_bounds[upper_bound_level(&self.level_bounds, node)],
            |pos| self.indices[pos],
            max_distance,
            |pos| Some(metric(self.entries[pos])),
            &mut queue,
            &mut visitor,
        )
    }

    /// Up to `max_results` item indices overlapping `region`, in nondecreasing
    /// `key` order.
    ///
    /// `region` is any [`Overlaps3D`] — a [`Box3D`], a
    /// [`Frustum3D`](crate::Frustum3D) — and `key(b)` scores the box `b`. The key
    /// must be an **admissible lower bound**: the key of a box never exceeds the
    /// key of any item inside it (a child box is contained in its parent).
    /// [`view_depth_3d`](crate::view_depth_3d) is the canonical one, and pairing
    /// it with a frustum is what makes this a front-to-back visibility query.
    /// `max_key` is a cutoff in the key's own units; `f64::INFINITY` for
    /// unbounded.
    ///
    /// [`search`](Self::search) answers the same *set* and is faster, because it
    /// is an unordered depth-first sweep that can emit a fully contained subtree
    /// wholesale. The ordered form earns its keep only when the order lets you
    /// stop early — a render budget, an occlusion loop, a "first few" probe —
    /// since `max_results` and `max_key` end the traversal without touching the
    /// rest of the tree. To order *everything*, call `search` and sort.
    ///
    /// Items with equal keys come out in an unspecified order. Like every query
    /// here this is broad phase: an overlapping box does not mean the item's
    /// exact geometry overlaps.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder, view_depth_3d};
    ///
    /// let mut b = Index3DBuilder::new(3);
    /// b.add(Box3D::new(20.0, 0.0, 0.0, 21.0, 1.0, 1.0)); // farthest
    /// b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)); // nearest
    /// b.add(Box3D::new(10.0, 0.0, 0.0, 11.0, 1.0, 1.0));
    /// let index = b.finish().unwrap();
    ///
    /// let eye = [-5.0, 0.0, 0.0];
    /// let forward = [1.0, 0.0, 0.0];
    /// let visible = Box3D::new(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0);
    ///
    /// // The two nearest along the view axis, near-to-far.
    /// let hits = index.search_ordered(
    ///     visible,
    ///     |bx| view_depth_3d(eye, forward, bx),
    ///     2,
    ///     f64::INFINITY,
    /// );
    /// assert_eq!(hits, vec![1, 2]);
    /// ```
    pub fn search_ordered<Q, K>(
        &self,
        region: Q,
        key: K,
        max_results: usize,
        max_key: f64,
    ) -> Vec<usize>
    where
        Q: Overlaps3D,
        K: Fn(Box3D) -> f64,
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
        Q: Overlaps3D,
        K: Fn(Box3D) -> f64,
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
    /// the key and may return [`ControlFlow::Break`] to stop early. See
    /// [`search_ordered`](Self::search_ordered) for the key contract.
    pub fn visit_ordered<Q, K, B, F>(
        &self,
        region: Q,
        key: K,
        max_key: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        Q: Overlaps3D,
        K: Fn(Box3D) -> f64,
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        visit_ordered(self, |b| region.overlaps_box(b), key, max_key, &mut visitor)
    }
    /// Up to `max_results` items overlapping `region`, ordered for picking: by
    /// squared perpendicular distance from `ray` to the item's box first, then
    /// by the ray's entry parameter `t` — so boxes the ray passes through come
    /// before boxes it only grazes, and along the ray near-to-far.
    ///
    /// This is the click-in-a-viewport query: `region` is the pixel's frustum
    /// (a narrow [`Frustum3D`](crate::Frustum3D) through the camera apex), `ray`
    /// is the pixel's central ray, and the first hit is the box the user most
    /// likely meant. It is a broad phase: both key components are lower bounds
    /// on the same quantities of the geometry inside the box, so exact
    /// ray-geometry tests belong to the caller (see
    /// [`Ray3D::closest_triangle`](crate::Ray3D::closest_triangle)).
    ///
    /// Each [`PickHit3D`] carries its key. The order is deterministic; boxes
    /// with equal keys keep Hilbert leaf order. Prefer this over
    /// [`search_ordered`](Self::search_ordered) when the natural order is "distance from the ray,
    /// ties by depth": a single scalar key cannot express that lexicographic
    /// order.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder, Point3D, Ray3D};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// // Two boxes on the ray at different depths, and one the ray misses.
    /// builder.add(Box3D::new(10.0, -0.5, -0.5, 11.0, 0.5, 0.5));
    /// builder.add(Box3D::new(2.0, -0.5, -0.5, 3.0, 0.5, 0.5));
    /// builder.add(Box3D::new(4.0, 0.3, -0.5, 5.0, 0.9, 0.5));
    /// let index = builder.finish().unwrap();
    ///
    /// let ray = Ray3D::new(Point3D { x: -1.0, y: 0.0, z: 0.0 }, 1.0, 0.0, 0.0, 1.0e4);
    /// // Pixel frustum wide enough to admit all three (not shown; any
    /// // `Overlaps3D` region works — a `Box3D` here keeps the example short).
    /// let region = Box3D::new(0.0, -1.0, -1.0, 12.0, 1.0, 1.0);
    ///
    /// let hits = index.search_pick(region, ray, 2);
    /// assert_eq!(hits[0].index, 1); // the near on-ray box first
    /// assert_eq!(hits[1].index, 0); // then the far on-ray box
    /// ```
    pub fn search_pick<Q>(&self, region: Q, ray: Ray3D, max_results: usize) -> Vec<PickHit3D>
    where
        Q: Overlaps3D,
    {
        let mut results = Vec::new();
        self.search_pick_into(region, ray, max_results, &mut results);
        results
    }

    /// [`search_pick`](Self::search_pick) into a reused buffer (cleared first).
    pub fn search_pick_into<Q>(
        &self,
        region: Q,
        ray: Ray3D,
        max_results: usize,
        results: &mut Vec<PickHit3D>,
    ) where
        Q: Overlaps3D,
    {
        collect_pick(
            self,
            max_results,
            |b| pick_key(&region, ray, b),
            results,
            &mut BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY),
        );
    }

    /// Visit the items [`search_pick`](Self::search_pick) would return, in the
    /// same order; the visitor may return [`ControlFlow::Break`] to stop early.
    pub fn visit_pick<Q, B, F>(&self, region: Q, ray: Ray3D, mut visitor: F) -> ControlFlow<B>
    where
        Q: Overlaps3D,
        F: FnMut(PickHit3D) -> ControlFlow<B>,
    {
        visit_pick(
            self,
            usize::MAX,
            |b| pick_key(&region, ray, b),
            &mut visitor,
        )
    }

    /// Return up to `max_results` item indices nearest to the box `query`.
    ///
    /// Distance is the box-to-box gap: items overlapping or touching `query`
    /// have distance `0.0` and come first (their mutual order is unspecified).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut builder = Index3DBuilder::new(2);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(10.0, 0.0, 0.0, 11.0, 1.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let query = Box3D::new(7.0, 0.0, 0.0, 8.0, 1.0, 1.0);
    /// assert_eq!(index.neighbors_of_box(query, 1), vec![1]);
    /// ```
    pub fn neighbors_of_box(&self, query: Box3D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`neighbors_of_box`](Self::neighbors_of_box).
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            results,
            &mut item_queue,
            &mut node_queue,
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

        self.collect_neighbors_with_queues(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
            &mut workspace.node_queue,
        );
        &workspace.results
    }

    /// Visit items in nondecreasing box-to-box distance order from `query`.
    ///
    /// The visitor receives squared gap distances (`0.0` for items overlapping
    /// the query box). Return [`ControlFlow::Break`] to stop early.
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

    /// Visit overlapping items without collecting a result `Vec`.
    ///
    /// The visitor receives item positions in the original insertion order.
    /// Return [`ControlFlow::Continue`] to continue traversal or
    /// [`ControlFlow::Break`] for early exit with a user-provided value.
    pub fn visit<B, Q, F>(&self, query: Q, visitor: F) -> ControlFlow<B>
    where
        Q: SearchQuery3D,
        F: FnMut(usize) -> ControlFlow<B>,
    {
        query.visit_index(self, visitor)
    }

    /// Return a lazy iterator over the items overlapping `query`.
    ///
    /// The tree is descended on demand, so consuming only a prefix
    /// (`.next()`, `.take(k)`, `.find(..)`) stops the traversal early and never
    /// allocates a result `Vec`. Yielded values are original insertion indices,
    /// in tree-traversal order (not part of the API). See
    /// [`Index2D::search_iter`](crate::Index2D::search_iter).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index3DBuilder, Box3D};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0));
    /// builder.add(Box3D::new(9.0, 9.0, 9.0, 10.0, 10.0, 10.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut hits: Vec<_> = index
    ///     .search_iter(Box3D::new(0.0, 0.0, 0.0, 4.0, 4.0, 4.0))
    ///     .collect();
    /// hits.sort_unstable();
    /// assert_eq!(hits, vec![0, 1]);
    /// ```
    pub fn search_iter<Q: SearchQuery3D>(&self, query: Q) -> Q::Iter<'_> {
        query.search_iter_index(self)
    }

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`. See [`Index2D::join`](crate::Index2D::join).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut a = Index3DBuilder::new(2);
    /// a.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// a.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    /// let a = a.finish().unwrap();
    ///
    /// let mut b = Index3DBuilder::new(1);
    /// b.add(Box3D::new(0.5, 0.5, 0.5, 5.5, 5.5, 5.5));
    /// let b = b.finish().unwrap();
    ///
    /// let mut pairs = a.join(&b);
    /// pairs.sort_unstable();
    /// assert_eq!(pairs, vec![(0, 0), (1, 0)]);
    /// ```
    pub fn join(&self, other: &Index3D) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other`. See
    /// [`Index2D::join_with`](crate::Index2D::join_with).
    pub fn join_with<B, F>(&self, other: &Index3D, visitor: F) -> ControlFlow<B>
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
    /// `other` lie within `max_distance` of each other: the Euclidean distance
    /// between their boxes is at most `max_distance`, zero when the boxes overlap
    /// (edges are inclusive).
    ///
    /// This is the distance join — "everything within 500 m", not
    /// "intersecting". Like every query here it is a broad phase: the box
    /// distance is a lower bound on the true distance between the underlying
    /// geometries, so hits are candidates and an exact predicate stays with
    /// the caller. A negative or NaN `max_distance` matches nothing, and
    /// `max_distance = 0.0` answers exactly [`Index3D::join`].
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut a = Index3DBuilder::new(2);
    /// a.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// a.add(Box3D::new(10.0, 0.0, 0.0, 11.0, 1.0, 1.0));
    /// let a = a.finish().unwrap();
    ///
    /// let mut b = Index3DBuilder::new(2);
    /// b.add(Box3D::new(2.5, 0.0, 0.0, 3.5, 1.0, 1.0));
    /// b.add(Box3D::new(13.0, 0.0, 0.0, 14.0, 1.0, 1.0));
    /// let b = b.finish().unwrap();
    ///
    /// let mut pairs = a.join_within(&b, 2.0);
    /// pairs.sort_unstable();
    /// // (1, 1) is exactly 2.0 apart, and the bound is inclusive.
    /// assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    /// ```
    pub fn join_within(&self, other: &Index3D, max_distance: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_within_with(other, max_distance, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every pair within `max_distance` between `self` and `other` without
    /// collecting a result `Vec`. See [`Index3D::join_within`].
    ///
    /// Return [`ControlFlow::Break`] for early exit.
    pub fn join_within_with<B, F>(
        &self,
        other: &Index3D,
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
    /// single-index sibling of [`Index3D::join_within`]. Like every query here it
    /// is a broad phase: the box distance is a lower bound on the true
    /// distance between the underlying geometries, so hits are candidates and
    /// an exact predicate stays with the caller.
    ///
    /// A negative or NaN `max_distance` matches nothing, and `max_distance = 0.0`
    /// answers exactly [`Index3D::search`]. Result order is traversal order and is
    /// not part of the API.
    ///
    /// Allocates a fresh `Vec` per call — see
    /// [`search_within_into`](Index3D::search_within_into),
    /// [`any_within`](Index3D::any_within).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(3.0, 0.0, 0.0, 4.0, 1.0, 1.0));
    /// builder.add(Box3D::new(50.0, 50.0, 50.0, 51.0, 51.0, 51.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut hits = index.search_within(Box3D::new(0.5, 0.5, 0.5, 1.0, 1.0, 1.0), 2.0);
    /// hits.sort_unstable();
    /// // item 1 is exactly 2.0 away, and the bound is inclusive.
    /// assert_eq!(hits, vec![0, 1]);
    /// ```
    pub fn search_within(&self, query: Box3D, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_within_into(query, max_distance, &mut out);
        out
    }

    /// [`search_within`](Self::search_within) into a reused buffer (cleared
    /// first).
    pub fn search_within_into(&self, query: Box3D, max_distance: f64, out: &mut Vec<usize>) {
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
    pub fn visit_within<B, F>(&self, query: Box3D, max_distance: f64, visitor: F) -> ControlFlow<B>
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
    pub fn any_within(&self, query: Box3D, max_distance: f64) -> bool {
        any_within_core(self, query, DistanceTest::new(max_distance))
    }

    /// Count the items within `max_distance` of `query`.
    ///
    /// The same traversal as [`visit_within`](Self::visit_within) with a
    /// counter in place of a buffer, so nothing is allocated. Mirrors
    /// [`count`](Self::count) for the overlap query.
    pub fn count_within(&self, query: Box3D, max_distance: f64) -> usize {
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
    /// use packed_spatial_index::{Index3DBuilder, Box3D};
    ///
    /// let mut builder = Index3DBuilder::new(64).node_size(4);
    /// for i in 0..64 {
    ///     let v = i as f64;
    ///     builder.add(Box3D::new(v, v, v, v + 0.5, v + 0.5, v + 0.5));
    /// }
    /// let index = builder.finish().unwrap();
    ///
    /// let window = Box3D::new(10.0, 10.0, 10.0, 30.2, 30.2, 30.2);
    /// let exact = index.count(window);
    /// let est = index.estimate_count(window, 1);
    /// assert!(est.lower <= exact && exact <= est.upper);
    /// assert!(est.lower as f64 <= est.estimate && est.estimate <= est.upper as f64);
    /// assert_eq!(index.estimate_count(window, 0).lower, exact);
    /// ```
    pub fn estimate_count(&self, query: Box3D, stop_level: usize) -> Estimate {
        estimate_core(
            self,
            stop_level,
            |node| node.overlaps(query),
            |node| query.contains(node),
            |node| box_fraction_3d(node, query),
        )
    }

    /// Return every unordered pair of distinct items within this index whose
    /// boxes lie within `max_distance` of each other, each pair exactly once. See
    /// [`Index3D::join_within`] for the distance semantics and
    /// [`Index2D::self_join_within`](crate::Index2D::self_join_within) for
    /// the pair shape.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0));
    /// builder.add(Box3D::new(3.5, 0.0, 0.0, 5.0, 2.0, 2.0));
    /// builder.add(Box3D::new(20.0, 20.0, 20.0, 21.0, 21.0, 21.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut pairs = index.self_join_within(2.0);
    /// pairs.sort_unstable();
    /// assert_eq!(pairs, vec![(0, 1)]);
    /// ```
    pub fn self_join_within(&self, max_distance: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.self_join_within_with(max_distance, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every unordered pair of distinct items within this index whose
    /// boxes lie within `max_distance` of each other, without collecting a result
    /// `Vec`. See [`Index3D::self_join_within`].
    ///
    /// Return [`ControlFlow::Break`] for early exit.
    pub fn self_join_within_with<B, F>(&self, max_distance: f64, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, DistanceTest::new(max_distance), visitor)
    }

    /// Return the ids of items of `self` that have no item of `other` within
    /// `max_distance` — the anti-join, the "noise" side of
    /// [`Index3D::join_within`]. One pruned search into `other` per item of
    /// `self`. (An index queried against itself pairs with itself at distance
    /// zero; isolated items of one index are a
    /// [`self_join_within_components`](Index3D::self_join_within_components)
    /// question.)
    ///
    /// A negative or NaN `max_distance` reports every item of `self`: nothing is
    /// within an empty bound.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut a = Index3DBuilder::new(2);
    /// a.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// a.add(Box3D::new(50.0, 50.0, 50.0, 51.0, 51.0, 51.0));
    /// let a = a.finish().unwrap();
    ///
    /// let mut b = Index3DBuilder::new(1);
    /// b.add(Box3D::new(2.5, 0.0, 0.0, 3.0, 1.0, 1.0));
    /// let b = b.finish().unwrap();
    ///
    /// assert_eq!(a.anti_join_within(&b, 2.0), vec![1]);
    /// ```
    pub fn anti_join_within(&self, other: &Index3D, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.anti_join_within_with(other, max_distance, |i| {
            out.push(i);
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every item of `self` with no item of `other` within `max_distance`,
    /// without collecting a result `Vec`. See [`Index3D::anti_join_within`].
    ///
    /// Return [`ControlFlow::Break`] for early exit.
    pub fn anti_join_within_with<B, F>(
        &self,
        other: &Index3D,
        max_distance: f64,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        anti_join_core(self, other, DistanceTest::new(max_distance), visitor)
    }

    /// Label every item with the smallest item id in its component of the
    /// `max_distance`-proximity graph: items are connected when their boxes lie
    /// within `max_distance` of each other. An item with no neighbour is its own
    /// label.
    ///
    /// The labels identify components; they are not clusters. Distance
    /// proximity is not transitive — a chain of items each within `max_distance`
    /// of the next forms one component no matter how far its ends lie apart —
    /// so this reports exactly what the graph of
    /// [`Index3D::self_join_within`] pairs defines, and the collapse policy
    /// stays with the caller.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(2.0, 0.0, 0.0, 3.0, 1.0, 1.0)); // 1.0 from item 0
    /// builder.add(Box3D::new(50.0, 50.0, 50.0, 51.0, 51.0, 51.0));
    /// let index = builder.finish().unwrap();
    ///
    /// assert_eq!(index.self_join_within_components(1.0), vec![0, 0, 2]);
    /// ```
    pub fn self_join_within_components(&self, max_distance: f64) -> Vec<usize> {
        self_join_components_core(self, DistanceTest::new(max_distance))
    }

    /// Return the closest pair of items between `self` and `other` as
    /// `(item_of_self, item_of_other, distance)`, or `None` when either index
    /// is empty.
    ///
    /// The one-answer end of the distance family: where
    /// [`Index3D::join_within`] needs an `max_distance` and reports every pair inside
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
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut a = Index3DBuilder::new(1);
    /// a.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// let a = a.finish().unwrap();
    ///
    /// let mut b = Index3DBuilder::new(2);
    /// b.add(Box3D::new(90.0, 0.0, 0.0, 91.0, 1.0, 1.0));
    /// b.add(Box3D::new(3.0, 0.0, 0.0, 4.0, 1.0, 1.0));
    /// let b = b.finish().unwrap();
    ///
    /// assert_eq!(a.closest_pair(&b), Some((0, 1, 2.0)));
    /// ```
    pub fn closest_pair(&self, other: &Index3D) -> Option<(usize, usize, f64)> {
        closest_pair_core(self, other)
    }

    /// Return the closest pair of *distinct* items within this index as
    /// `(i, j, distance)`, or `None` for fewer than two items.
    ///
    /// See [`Index3D::closest_pair`] for the distance semantics. An item is never
    /// paired with itself; the order of the two ids, and which of several
    /// equally close pairs is reported, are traversal order and not part of
    /// the API.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(3.0, 0.0, 0.0, 4.0, 1.0, 1.0));
    /// builder.add(Box3D::new(3.5, 0.0, 0.0, 4.5, 1.0, 1.0));
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

    fn collect_neighbors_with_queues(
        &self,
        query: NeighborQuery3D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        item_queue: &mut BinaryHeap<NeighborState>,
        node_queue: &mut BinaryHeap<NeighborNodeState>,
    ) {
        if !query.is_valid() {
            results.clear();
            item_queue.clear();
            node_queue.clear();
            return;
        }
        best_first::collect_neighbors_two_queue(
            self.entries.len(),
            self.num_items,
            self.node_size,
            |n| self.level_bounds[upper_bound_level(&self.level_bounds, n)],
            |pos| self.indices[pos],
            max_results,
            max_distance,
            |pos| query.distance_squared_to(self.entries[pos]),
            results,
            item_queue,
            node_queue,
        );
    }

    fn nearest_one_with_queue(
        &self,
        query: NeighborQuery3D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        if !query.is_valid() {
            queue.clear();
            return None;
        }
        best_first::nearest_one(
            self.entries.len(),
            self.num_items,
            self.node_size,
            |n| self.level_bounds[upper_bound_level(&self.level_bounds, n)],
            |pos| self.indices[pos],
            max_distance,
            |pos| query.distance_squared_to(self.entries[pos]),
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
        if !query.is_valid() {
            queue.clear();
            return ControlFlow::Continue(());
        }
        best_first::visit_neighbors(
            self.entries.len(),
            self.num_items,
            self.node_size,
            |n| self.level_bounds[upper_bound_level(&self.level_bounds, n)],
            |pos| self.indices[pos],
            max_distance,
            |pos| query.distance_squared_to(self.entries[pos]),
            queue,
            visitor,
        )
    }

    /// Same as [`visit`](Index3D::visit), but the traversal stack is reused by the caller.
    #[doc(hidden)]
    pub fn visit_with_stack<B, F>(
        &self,
        query: Box3D,
        stack: &mut Vec<usize>,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        // Local slice-based traversal (not the shared `visit_overlaps`): iterating
        // `&entries[node..end]` lets LLVM autovectorize the overlap test, which a
        // per-element `TreeAccess` kernel cannot. Measured ~1.5x faster than the
        // generic kernel on owned visit (the views, whose byte storage has no
        // slice to vectorize, keep using `visit_overlaps`).
        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_entries = &self.entries[node_index..end];
            let node_indices = &self.indices[node_index..end];

            if is_leaf {
                for (b, &index) in node_entries.iter().zip(node_indices) {
                    if !b.overlaps(query) {
                        continue;
                    }
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for (b, &index) in node_entries.iter().zip(node_indices).rev() {
                    if !b.overlaps(query) {
                        continue;
                    }
                    stack.push(index);
                    stack.push(child_level);
                }
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return ControlFlow::Continue(());
            }
        }
    }

    /// Same as [`search`](Index3D::search), but the traversal stack is reused by the caller.
    #[doc(hidden)]
    pub fn search_into_stack(
        &self,
        query: Box3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        results.clear();
        if self.num_items == 0 {
            stack.clear();
            return;
        }

        let root = self.entries[self.entries.len() - 1];
        // See the same guard in `range.rs`: a box with `min > max`, which the unchecked
        // `Box3D::new` allows, is contained by queries it does not overlap.
        if root.overlaps(query) && query.contains(root) {
            stack.clear();
            results.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }

        self.search_into_stack_impl(query, results, stack);
    }

    /// Count items overlapping `query`, answering a fully contained subtree from
    /// its leaf range instead of testing the items inside it.
    ///
    /// This is the counting twin of `search_into_stack_contained_impl`: same
    /// traversal, same `CONTAINED_FLAG` encoding, but a contained node contributes
    /// `end - start` and never touches an entry. `count` used to run the plain
    /// visitor traversal, which tests every item under a contained subtree and
    /// calls a closure for each hit, so a window covering a fraction `f` of an
    /// `N`-item index cost `O(f * N)` where it now costs `O(f * N / node_size)`
    /// node pops.
    fn count_overlaps(&self, query: Box3D) -> usize {
        if self.num_items == 0 {
            return 0;
        }

        const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
        const LEVEL_MASK: usize = !CONTAINED_FLAG;

        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let mut total = 0usize;
        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_entries = &self.entries[node_index..end];

            if contained {
                let (start, leaf_end) = leaf_group_range(self, node_index, end, level);
                total += leaf_end - start;
            } else if is_leaf {
                // Branch-free over the slice: LLVM vectorizes the per-node test, and
                // a count has nothing to do per hit but add one.
                let mut hits = 0usize;
                for b in node_entries {
                    hits += usize::from(b.overlaps(query));
                }
                total += hits;
            } else {
                let child_level = level - 1;
                let node_indices = &self.indices[node_index..end];
                for (b, &index) in node_entries.iter().zip(node_indices).rev() {
                    if !b.overlaps(query) {
                        continue;
                    }
                    stack.push(index);
                    let encoded_level = if query.contains(*b) {
                        child_level | CONTAINED_FLAG
                    } else {
                        child_level
                    };
                    stack.push(encoded_level);
                }
            }

            if stack.len() > 1 {
                let encoded_level = stack.pop().unwrap();
                level = encoded_level & LEVEL_MASK;
                contained = (encoded_level & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return total;
            }
        }
    }

    fn search_into_stack_impl(
        &self,
        query: Box3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        stack.clear();

        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_entries = &self.entries[node_index..end];
            let node_indices = &self.indices[node_index..end];

            if is_leaf {
                for (b, &index) in node_entries.iter().zip(node_indices) {
                    if !b.overlaps(query) {
                        continue;
                    }
                    results.push(index);
                }
            } else {
                let child_level = level - 1;
                for (b, &index) in node_entries.iter().zip(node_indices).rev() {
                    if !b.overlaps(query) {
                        continue;
                    }
                    stack.push(index);
                    stack.push(child_level);
                }
            }

            if stack.len() > 1 {
                prefetch_aos_node3d(
                    &self.entries,
                    &self.indices,
                    stack[stack.len() - 2],
                    self.node_size,
                );
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }

    /// Diagnostics: returns `(result_count, intersection_check_count)`.
    #[doc(hidden)]
    pub fn search_visited(&self, query: Box3D) -> (usize, usize) {
        let mut results = 0usize;
        let mut visited = 0usize;
        if self.num_items == 0 {
            return (0, 0);
        }

        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                visited += 1;
                if !self.entries[pos].overlaps(query) {
                    continue;
                }
                if is_leaf {
                    results += 1;
                } else {
                    stack.push(self.indices[pos]);
                    stack.push(level - 1);
                }
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return (results, visited);
            }
        }
    }
}

/// Zero-copy read-only view over bytes produced by [`Index3D::to_bytes`].
///
/// Loading validates the buffer but does not copy the tree into owned vectors.
/// Search and nearest-neighbor methods read little-endian values directly from
/// the borrowed byte slice.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, Index3DBuilder, Index3DView};
///
/// let mut builder = Index3DBuilder::new(1);
/// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
/// let bytes = builder.finish().unwrap().to_bytes();
///
/// let view = Index3DView::from_bytes(&bytes).unwrap();
/// assert_eq!(view.search(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)), vec![0]);
/// ```
pub struct Index3DView<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    /// Derived at load (not stored), so owned rather than borrowed.
    level_bounds: Vec<usize>,
    entries: &'a [u8],
    indices: &'a [u8],
    payload: Option<ParsedPayload<'a>>,
    /// `insertion id -> leaf rank` for random `payload(id)` over leaf-ordered
    /// payloads; built only when a payload is present.
    id_to_leaf: Option<Vec<u32>>,
}

impl<'a> Index3DView<'a> {
    /// Load a zero-copy 3D index view from bytes previously produced by [`Index3D::to_bytes`].
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder, Index3DView};
    ///
    /// let mut builder = Index3DBuilder::new(1);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// let bytes = builder.finish()?.to_bytes();
    ///
    /// let view = Index3DView::from_bytes(&bytes)?;
    /// assert_eq!(view.num_items(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 3, 8)?;
        let id_to_leaf = payload
            .is_some()
            .then(|| build_id_to_leaf(parsed.indices, parsed.num_items));
        Ok(Self {
            node_size: parsed.node_size,
            num_items: parsed.num_items,
            num_nodes: parsed.num_nodes,
            level_count: parsed.level_count,
            level_bounds: parsed.level_bounds,
            entries: parsed.entries,
            indices: parsed.indices,
            payload,
            id_to_leaf,
        })
    }

    /// Whether this view's bytes carry a payload section.
    pub fn has_payload(&self) -> bool {
        self.payload.is_some()
    }

    /// Borrow item `id`'s payload blob (zero-copy), or `None` if absent or out of
    /// range. See [`Index2DView::payload`](crate::Index2DView::payload).
    pub fn payload(&self, id: usize) -> Option<&'a [u8]> {
        let payload = self.payload.as_ref()?;
        let id_to_leaf = self.id_to_leaf.as_ref()?;
        let leaf_rank = *id_to_leaf.get(id)? as usize;
        Some(payload_slice(payload, leaf_rank))
    }

    /// Borrow every triangle record as a zero-copy `&[T]` (with `T` =
    /// [`Triangle3D`](crate::Triangle3D) / [`Triangle3DF32`](crate::Triangle3DF32)),
    /// in leaf (storage) order, when the payload is a fixed-width section of that
    /// record type and the underlying bytes are aligned (an mmap or an aligned
    /// buffer). Returns `None` otherwise; [`triangle`](Self::triangle) reads one
    /// record by item id regardless of alignment.
    pub fn triangles<T: Triangle3>(&self) -> Option<&'a [T]> {
        let payload = self.payload.as_ref()?;
        if payload.stride != T::STRIDE {
            return None;
        }
        blobs_as_records::<T>(payload.blobs)
    }

    /// The triangle stored for item `id`, by value (works at any alignment).
    /// `None` if there is no triangle payload of the requested type, or `id` is
    /// out of range. The type parameter chooses the record format
    /// ([`Triangle3D`](crate::Triangle3D) for `f64`,
    /// [`Triangle3DF32`](crate::Triangle3DF32) for `f32`).
    pub fn triangle<T: Triangle3>(&self, id: usize) -> Option<T> {
        let payload = self.payload.as_ref()?;
        if payload.stride != T::STRIDE {
            return None;
        }
        let id_to_leaf = self.id_to_leaf.as_ref()?;
        let leaf_rank = *id_to_leaf.get(id)? as usize;
        Some(T::read_le(payload_slice(payload, leaf_rank)))
    }

    /// Return `(item index, payload blob)` for every item intersecting `query`.
    /// See [`Index2DView::search_payloads`](crate::Index2DView::search_payloads).
    pub fn search_payloads(&self, query: Box3D) -> Vec<(usize, &'a [u8])> {
        let mut out = Vec::new();
        if self.payload.is_none() {
            return out;
        }
        for id in self.search(query) {
            if let Some(blob) = self.payload(id) {
                out.push((id, blob));
            }
        }
        out
    }

    /// Return the number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total extent of indexed items, or `None` for an empty view.
    pub fn extent(&self) -> Option<Box3D> {
        if self.num_items == 0 {
            None
        } else {
            Some(self.entry_at_unchecked(self.num_nodes - 1))
        }
    }

    /// Return the packed node size.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Return the indices of all items whose boxes overlap `query`.
    ///
    /// Allocates a fresh `Vec` per call. For a boolean test use [`any`](Self::any)
    /// rather than `search(..).is_empty()`; in a hot loop write into a buffer you
    /// own with [`search_into`](Self::search_into) or
    /// [`search_with`](Self::search_with); to count the hits use
    /// [`count`](Self::count) and to fold over them use [`visit`](Self::visit).
    pub fn search<Q: SearchQuery3D>(&self, query: Q) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(query, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into<Q: SearchQuery3D>(&self, query: Q, results: &mut Vec<usize>) {
        query.search_into_view(self, results);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'b, Q: SearchQuery3D>(
        &self,
        query: Q,
        workspace: &'b mut SearchWorkspace,
    ) -> &'b [usize] {
        query.search_with_view(self, workspace)
    }

    /// Return `true` if at least one item overlaps `query`.
    pub fn any<Q: SearchQuery3D>(&self, query: Q) -> bool {
        query.any_view(self)
    }

    /// Return the number of items overlapping `query`.
    ///
    /// Counts during the traversal, so nothing is collected — prefer it to
    /// `search(query).len()`, which allocates a `Vec` to throw away.
    pub fn count<Q: SearchQuery3D>(&self, query: Q) -> usize {
        let mut count = 0usize;
        let _: ControlFlow<()> = self.visit(query, |_| {
            count += 1;
            ControlFlow::Continue(())
        });
        count
    }

    /// Return one overlapping item, if any.
    pub fn first<Q: SearchQuery3D>(&self, query: Q) -> Option<usize> {
        query.first_view(self)
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery3D::Point(point),
            max_results,
            max_distance,
            results,
            &mut item_queue,
            &mut node_queue,
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

        self.collect_neighbors_with_queues(
            NeighborQuery3D::Point(point),
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
            &mut workspace.node_queue,
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

    /// Up to `max_results` item indices nearest to your query under a custom
    /// distance `metric`, nearest first. The zero-copy view counterpart of
    /// [`Index3D::neighbors_metric`](crate::Index3D::neighbors_metric); see it for
    /// the metric contract.
    pub fn neighbors_metric<M: Fn(Box3D) -> f64>(
        &self,
        metric: M,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        self.neighbors_metric_into(metric, max_results, max_distance, &mut results);
        results
    }

    /// [`neighbors_metric`](Self::neighbors_metric) into a reused buffer (cleared first).
    pub fn neighbors_metric_into<M: Fn(Box3D) -> f64>(
        &self,
        metric: M,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
    ) {
        results.clear();
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        metric_knn::collect_neighbors(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |node| self.level_bound_unchecked(self.upper_bound_level(node)),
            |pos| self.index_at_unchecked(pos),
            max_results,
            max_distance,
            |pos| Some(metric(self.entry_at_unchecked(pos))),
            results,
            &mut queue,
        );
    }

    /// Visit items in nondecreasing custom-`metric` distance; the visitor receives
    /// the metric distance and may return [`ControlFlow::Break`] to stop early.
    /// See [`Index3D::neighbors_metric`](crate::Index3D::neighbors_metric) for the
    /// metric contract.
    pub fn visit_neighbors_metric<B, M, F>(
        &self,
        metric: M,
        max_distance: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        M: Fn(Box3D) -> f64,
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        metric_knn::visit_neighbors(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |node| self.level_bound_unchecked(self.upper_bound_level(node)),
            |pos| self.index_at_unchecked(pos),
            max_distance,
            |pos| Some(metric(self.entry_at_unchecked(pos))),
            &mut queue,
            &mut visitor,
        )
    }

    /// Up to `max_results` item indices overlapping `region`, in nondecreasing
    /// `key` order. See [`Index3D::search_ordered`](crate::Index3D::search_ordered)
    /// for the key contract and when to prefer it over [`search`](Self::search).
    pub fn search_ordered<Q, K>(
        &self,
        region: Q,
        key: K,
        max_results: usize,
        max_key: f64,
    ) -> Vec<usize>
    where
        Q: Overlaps3D,
        K: Fn(Box3D) -> f64,
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
        Q: Overlaps3D,
        K: Fn(Box3D) -> f64,
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
    /// the key and may return [`ControlFlow::Break`] to stop early. See
    /// [`Index3D::search_ordered`](crate::Index3D::search_ordered).
    pub fn visit_ordered<Q, K, B, F>(
        &self,
        region: Q,
        key: K,
        max_key: f64,
        mut visitor: F,
    ) -> ControlFlow<B>
    where
        Q: Overlaps3D,
        K: Fn(Box3D) -> f64,
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        visit_ordered(self, |b| region.overlaps_box(b), key, max_key, &mut visitor)
    }
    /// Up to `max_results` items overlapping `region`, ordered for picking: by
    /// squared perpendicular distance from `ray` to the item's box first, then
    /// by the ray's entry parameter `t` — so boxes the ray passes through come
    /// before boxes it only grazes, and along the ray near-to-far.
    ///
    /// This is the click-in-a-viewport query: `region` is the pixel's frustum
    /// (a narrow [`Frustum3D`](crate::Frustum3D) through the camera apex), `ray`
    /// is the pixel's central ray, and the first hit is the box the user most
    /// likely meant. It is a broad phase: both key components are lower bounds
    /// on the same quantities of the geometry inside the box, so exact
    /// ray-geometry tests belong to the caller (see
    /// [`Ray3D::closest_triangle`](crate::Ray3D::closest_triangle)).
    ///
    /// Each [`PickHit3D`] carries its key. The order is deterministic; boxes
    /// with equal keys keep Hilbert leaf order. Prefer this over
    /// [`Index3D::search_pick`](crate::Index3D::search_pick) when the natural order is "distance from the ray,
    /// ties by depth": a single scalar key cannot express that lexicographic
    /// order.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder, Point3D, Ray3D};
    ///
    /// let mut builder = Index3DBuilder::new(3);
    /// // Two boxes on the ray at different depths, and one the ray misses.
    /// builder.add(Box3D::new(10.0, -0.5, -0.5, 11.0, 0.5, 0.5));
    /// builder.add(Box3D::new(2.0, -0.5, -0.5, 3.0, 0.5, 0.5));
    /// builder.add(Box3D::new(4.0, 0.3, -0.5, 5.0, 0.9, 0.5));
    /// let index = builder.finish().unwrap();
    ///
    /// let ray = Ray3D::new(Point3D { x: -1.0, y: 0.0, z: 0.0 }, 1.0, 0.0, 0.0, 1.0e4);
    /// // Pixel frustum wide enough to admit all three (not shown; any
    /// // `Overlaps3D` region works — a `Box3D` here keeps the example short).
    /// let region = Box3D::new(0.0, -1.0, -1.0, 12.0, 1.0, 1.0);
    ///
    /// let hits = index.search_pick(region, ray, 2);
    /// assert_eq!(hits[0].index, 1); // the near on-ray box first
    /// assert_eq!(hits[1].index, 0); // then the far on-ray box
    /// ```
    pub fn search_pick<Q>(&self, region: Q, ray: Ray3D, max_results: usize) -> Vec<PickHit3D>
    where
        Q: Overlaps3D,
    {
        let mut results = Vec::new();
        self.search_pick_into(region, ray, max_results, &mut results);
        results
    }

    /// [`search_pick`](Self::search_pick) into a reused buffer (cleared first).
    pub fn search_pick_into<Q>(
        &self,
        region: Q,
        ray: Ray3D,
        max_results: usize,
        results: &mut Vec<PickHit3D>,
    ) where
        Q: Overlaps3D,
    {
        collect_pick(
            self,
            max_results,
            |b| pick_key(&region, ray, b),
            results,
            &mut BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY),
        );
    }

    /// Visit the items [`search_pick`](Self::search_pick) would return, in the
    /// same order; the visitor may return [`ControlFlow::Break`] to stop early.
    pub fn visit_pick<Q, B, F>(&self, region: Q, ray: Ray3D, mut visitor: F) -> ControlFlow<B>
    where
        Q: Overlaps3D,
        F: FnMut(PickHit3D) -> ControlFlow<B>,
    {
        visit_pick(
            self,
            usize::MAX,
            |b| pick_key(&region, ray, b),
            &mut visitor,
        )
    }

    /// Return up to `max_results` item indices nearest to the box `query`.
    ///
    /// Distance is the box-to-box gap: items overlapping or touching `query`
    /// have distance `0.0` and come first (their mutual order is unspecified).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder};
    ///
    /// let mut builder = Index3DBuilder::new(2);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(10.0, 0.0, 0.0, 11.0, 1.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let query = Box3D::new(7.0, 0.0, 0.0, 8.0, 1.0, 1.0);
    /// assert_eq!(index.neighbors_of_box(query, 1), vec![1]);
    /// ```
    pub fn neighbors_of_box(&self, query: Box3D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`neighbors_of_box`](Self::neighbors_of_box).
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            results,
            &mut item_queue,
            &mut node_queue,
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

        self.collect_neighbors_with_queues(
            NeighborQuery3D::Box(query),
            max_results,
            max_distance,
            &mut workspace.results,
            &mut workspace.queue,
            &mut workspace.node_queue,
        );
        &workspace.results
    }

    /// Visit items in nondecreasing box-to-box distance order from `query`.
    ///
    /// The visitor receives squared gap distances (`0.0` for items overlapping
    /// the query box). Return [`ControlFlow::Break`] to stop early.
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

    /// Visit overlapping items without collecting a result `Vec`.
    pub fn visit<B, Q, F>(&self, query: Q, visitor: F) -> ControlFlow<B>
    where
        Q: SearchQuery3D,
        F: FnMut(usize) -> ControlFlow<B>,
    {
        query.visit_view(self, visitor)
    }

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`. See [`Index2D::join`](crate::Index2D::join).
    pub fn join(&self, other: &Index3DView<'_>) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other`. See
    /// [`Index2D::join_with`](crate::Index2D::join_with).
    pub fn join_with<B, F>(&self, other: &Index3DView<'_>, visitor: F) -> ControlFlow<B>
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
    /// [`Index3D::join_within`].
    pub fn join_within(&self, other: &Index3DView<'_>, max_distance: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_within_with(other, max_distance, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every pair within `max_distance` between `self` and `other`. See
    /// [`Index3D::join_within_with`].
    pub fn join_within_with<B, F>(
        &self,
        other: &Index3DView<'_>,
        max_distance: f64,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, DistanceTest::new(max_distance), visitor)
    }

    /// Return the ids of every item of the view whose box lies within
    /// `max_distance` of `query`. See [`Index3D::search_within`].
    pub fn search_within(&self, query: Box3D, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_within_into(query, max_distance, &mut out);
        out
    }

    /// [`search_within`](Self::search_within) into a reused buffer (cleared
    /// first).
    pub fn search_within_into(&self, query: Box3D, max_distance: f64, out: &mut Vec<usize>) {
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
    pub fn visit_within<B, F>(&self, query: Box3D, max_distance: f64, visitor: F) -> ControlFlow<B>
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
    pub fn any_within(&self, query: Box3D, max_distance: f64) -> bool {
        any_within_core(self, query, DistanceTest::new(max_distance))
    }

    /// Count the items within `max_distance` of `query`.
    ///
    /// The same traversal as [`visit_within`](Self::visit_within) with a
    /// counter in place of a buffer, so nothing is allocated. Mirrors
    /// [`count`](Self::count) for the overlap query.
    pub fn count_within(&self, query: Box3D, max_distance: f64) -> usize {
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
    /// use packed_spatial_index::{Index3DBuilder, Box3D};
    ///
    /// let mut builder = Index3DBuilder::new(64).node_size(4);
    /// for i in 0..64 {
    ///     let v = i as f64;
    ///     builder.add(Box3D::new(v, v, v, v + 0.5, v + 0.5, v + 0.5));
    /// }
    /// let index = builder.finish().unwrap();
    ///
    /// let window = Box3D::new(10.0, 10.0, 10.0, 30.2, 30.2, 30.2);
    /// let exact = index.count(window);
    /// let est = index.estimate_count(window, 1);
    /// assert!(est.lower <= exact && exact <= est.upper);
    /// assert!(est.lower as f64 <= est.estimate && est.estimate <= est.upper as f64);
    /// assert_eq!(index.estimate_count(window, 0).lower, exact);
    /// ```
    pub fn estimate_count(&self, query: Box3D, stop_level: usize) -> Estimate {
        estimate_core(
            self,
            stop_level,
            |node| node.overlaps(query),
            |node| query.contains(node),
            |node| box_fraction_3d(node, query),
        )
    }

    /// Return every unordered pair of distinct items within this view whose
    /// boxes lie within `max_distance` of each other, each pair exactly once. See
    /// [`Index3D::self_join_within`].
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
    /// [`Index3D::self_join_within_with`].
    pub fn self_join_within_with<B, F>(&self, max_distance: f64, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, DistanceTest::new(max_distance), visitor)
    }

    /// Return the ids of items of `self` with no item of `other` within
    /// `max_distance`. See [`Index3D::anti_join_within`].
    pub fn anti_join_within(&self, other: &Index3DView<'_>, max_distance: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.anti_join_within_with(other, max_distance, |i| {
            out.push(i);
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every item of `self` with no item of `other` within `max_distance`.
    /// See [`Index3D::anti_join_within_with`].
    pub fn anti_join_within_with<B, F>(
        &self,
        other: &Index3DView<'_>,
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
    /// [`Index3D::self_join_within_components`].
    pub fn self_join_within_components(&self, max_distance: f64) -> Vec<usize> {
        self_join_components_core(self, DistanceTest::new(max_distance))
    }

    /// Return the closest pair of items between this view and `other`. See
    /// [`Index3D::closest_pair`].
    pub fn closest_pair(&self, other: &Index3DView<'_>) -> Option<(usize, usize, f64)> {
        closest_pair_core(self, other)
    }

    /// Return the closest pair of distinct items within this view. See
    /// [`Index3D::self_closest_pair`].
    pub fn self_closest_pair(&self) -> Option<(usize, usize, f64)> {
        self_closest_pair_core(self)
    }

    fn collect_neighbors_with_queues(
        &self,
        query: NeighborQuery3D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        item_queue: &mut BinaryHeap<NeighborState>,
        node_queue: &mut BinaryHeap<NeighborNodeState>,
    ) {
        if !query.is_valid() {
            results.clear();
            item_queue.clear();
            node_queue.clear();
            return;
        }
        best_first::collect_neighbors_two_queue(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |n| self.level_bound_unchecked(self.upper_bound_level(n)),
            |pos| self.index_at_unchecked(pos),
            max_results,
            max_distance,
            |pos| query.distance_squared_to(self.entry_at_unchecked(pos)),
            results,
            item_queue,
            node_queue,
        );
    }

    fn nearest_one_with_queue(
        &self,
        query: NeighborQuery3D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        if !query.is_valid() {
            queue.clear();
            return None;
        }
        best_first::nearest_one(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |n| self.level_bound_unchecked(self.upper_bound_level(n)),
            |pos| self.index_at_unchecked(pos),
            max_distance,
            |pos| query.distance_squared_to(self.entry_at_unchecked(pos)),
            queue,
        )
    }

    /// Range search over the byte layout, with the contained-subtree fast path.
    ///
    /// The shared region traversal, not the overlaps-only one: a subtree the
    /// query fully contains is emitted whole, so its items are never parsed out
    /// of the buffer or tested one by one. `any` / `first` deliberately keep the
    /// overlaps-only path -- they stop at the first hit, so a containment test
    /// per node could only add work.
    #[doc(hidden)]
    pub fn search_into_stack(
        &self,
        query: Box3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        results.clear();
        let _: ControlFlow<()> = visit_region(
            self,
            stack,
            |bounds: Box3D| bounds.overlaps(query),
            |bounds: Box3D| query.contains(bounds),
            |index| {
                results.push(index);
                ControlFlow::Continue(())
            },
        );
    }

    /// Range search over the byte layout, with the contained-subtree fast path.
    ///
    /// The shared region traversal, not the overlaps-only one: a subtree the
    /// query fully contains is emitted whole, so its items are never parsed out
    /// of the buffer or tested one by one. `any` / `first` deliberately keep the
    /// overlaps-only path -- they stop at the first hit, so a containment test
    /// per node could only add work.
    #[doc(hidden)]
    pub fn visit_with_stack<B, F>(
        &self,
        query: Box3D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        visit_region(
            self,
            stack,
            |bounds: Box3D| bounds.overlaps(query),
            |bounds: Box3D| query.contains(bounds),
            visitor,
        )
    }

    /// Overlaps-only traversal, for the short-circuiting entry points.
    #[doc(hidden)]
    pub fn visit_overlaps_with_stack<B, F>(
        &self,
        query: Box3D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        visit_overlaps(self, query, stack, visitor)
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
        if !query.is_valid() {
            queue.clear();
            return ControlFlow::Continue(());
        }
        best_first::visit_neighbors(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |n| self.level_bound_unchecked(self.upper_bound_level(n)),
            |pos| self.index_at_unchecked(pos),
            max_distance,
            |pos| query.distance_squared_to(self.entry_at_unchecked(pos)),
            queue,
            visitor,
        )
    }

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

    #[inline]
    fn level_bound_unchecked(&self, index: usize) -> usize {
        self.level_bounds[index]
    }

    #[inline]
    fn entry_at_unchecked(&self, index: usize) -> Box3D {
        let offset = index * 48;
        Box3D::new(
            read_f64_le_unchecked(self.entries, offset),
            read_f64_le_unchecked(self.entries, offset + 8),
            read_f64_le_unchecked(self.entries, offset + 16),
            read_f64_le_unchecked(self.entries, offset + 24),
            read_f64_le_unchecked(self.entries, offset + 32),
            read_f64_le_unchecked(self.entries, offset + 40),
        )
    }

    #[inline]
    fn index_at_unchecked(&self, index: usize) -> usize {
        read_u64_le_unchecked(self.indices, index * 8) as usize
    }
}

impl TreeAccess for Index3D {
    type Bounds = Box3D;

    #[inline]
    fn tree_num_items(&self) -> usize {
        self.num_items
    }
    #[inline]
    fn tree_num_nodes(&self) -> usize {
        self.entries.len()
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
        self.entries[pos]
    }
    #[inline]
    fn tree_index(&self, pos: usize) -> usize {
        self.indices[pos]
    }
    #[inline]
    fn bounds_overlap(a: Box3D, b: Box3D) -> bool {
        a.overlaps(b)
    }
}

impl TreeAccess for Index3DView<'_> {
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
        self.entry_at_unchecked(pos)
    }
    #[inline]
    fn tree_index(&self, pos: usize) -> usize {
        self.index_at_unchecked(pos)
    }
    #[inline]
    fn bounds_overlap(a: Box3D, b: Box3D) -> bool {
        a.overlaps(b)
    }
}

/// Lazy iterator over the items overlapping a query, returned by
/// [`Index3D::search_iter`].
///
/// Yields original insertion indices in tree-traversal order, descending the
/// tree only as far as the consumer pulls. Holds a small traversal stack
/// (`O(depth)`); it allocates no result `Vec`.
pub struct Search3DIter<'a> {
    index: &'a Index3D,
    query: Box3D,
    stack: Vec<Search3DFrame>,
    // Half-open entry range of the leaf node currently being scanned.
    leaf_pos: usize,
    leaf_end: usize,
}

#[cold]
#[inline(never)]
fn copy_box3d_entries(bytes: &[u8], num_nodes: usize) -> Vec<Box3D> {
    if cfg!(target_endian = "little") {
        // SAFETY: Box3D is repr(C) over six f64 fields; every f64 bit pattern is valid.
        let (prefix, entries, suffix) = unsafe { bytes.align_to::<Box3D>() };
        if prefix.is_empty() && suffix.is_empty() && entries.len() == num_nodes {
            return entries.to_vec();
        }
    }
    let mut entries = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let offset = i * 48;
        entries.push(Box3D::new(
            read_f64_le_unchecked(bytes, offset),
            read_f64_le_unchecked(bytes, offset + 8),
            read_f64_le_unchecked(bytes, offset + 16),
            read_f64_le_unchecked(bytes, offset + 24),
            read_f64_le_unchecked(bytes, offset + 32),
            read_f64_le_unchecked(bytes, offset + 40),
        ));
    }
    entries
}

#[cold]
#[inline(never)]
fn copy_u64_indices(bytes: &[u8], num_nodes: usize) -> Vec<usize> {
    if cfg!(target_endian = "little") && core::mem::size_of::<usize>() == 8 {
        // SAFETY: on this path usize is a little-endian u64-width integer.
        let (prefix, indices, suffix) = unsafe { bytes.align_to::<usize>() };
        if prefix.is_empty() && suffix.is_empty() && indices.len() == num_nodes {
            return indices.to_vec();
        }
    }
    let mut indices = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        indices.push(read_u64_le_unchecked(bytes, i * 8) as usize);
    }
    indices
}

#[derive(Clone, Copy)]
struct Search3DFrame {
    node_index: usize,
    level: usize,
}

impl<'a> Search3DIter<'a> {
    fn new(index: &'a Index3D, query: Box3D) -> Self {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY / 2);
        if index.num_items != 0 {
            // Seed with the root so `next` drives the descent uniformly.
            stack.push(Search3DFrame {
                node_index: index.entries.len() - 1,
                level: index.level_bounds.len() - 1,
            });
        }
        Self {
            index,
            query,
            stack,
            leaf_pos: 0,
            leaf_end: 0,
        }
    }
}

impl Iterator for Search3DIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let index = self.index;
        loop {
            // Drain remaining hits in the leaf node currently being scanned.
            while self.leaf_pos < self.leaf_end {
                let at = self.leaf_pos;
                self.leaf_pos += 1;
                if self.query.overlaps_box(index.entries[at]) {
                    return Some(index.indices[at]);
                }
            }

            let frame = self.stack.pop()?;
            let end = (frame.node_index + index.node_size).min(index.level_bounds[frame.level]);

            if frame.node_index < index.num_items {
                // Leaf node: scan its entries on the next loop turns.
                self.leaf_pos = frame.node_index;
                self.leaf_end = end;
            } else {
                // Internal node: push overlapping children reversed so they pop
                // in forward order (matching `visit`).
                let child_level = frame.level - 1;
                for (b, &child) in index.entries[frame.node_index..end]
                    .iter()
                    .zip(&index.indices[frame.node_index..end])
                    .rev()
                {
                    if self.query.overlaps_box(*b) {
                        self.stack.push(Search3DFrame {
                            node_index: child,
                            level: child_level,
                        });
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Exact count is unknown without traversing; at most every item matches.
        (0, Some(self.index.num_items))
    }
}

impl std::iter::FusedIterator for Search3DIter<'_> {}

#[doc(hidden)]
pub struct RegionSearch3DIter<'a, Q> {
    index: &'a Index3D,
    query: Q,
    stack: Vec<RegionSearch3DFrame>,
    leaf_pos: usize,
    leaf_end: usize,
    leaf_contained: bool,
}

#[derive(Clone, Copy)]
struct RegionSearch3DFrame {
    node_index: usize,
    level: usize,
    contained: bool,
}

impl<'a, Q: Overlaps3D> RegionSearch3DIter<'a, Q> {
    fn new(index: &'a Index3D, query: Q) -> Self {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY / 2);
        if index.num_items != 0 {
            let root = index.entries.len() - 1;
            stack.push(RegionSearch3DFrame {
                node_index: root,
                level: index.level_bounds.len() - 1,
                contained: query.contains_box(index.entries[root]),
            });
        }
        Self {
            index,
            query,
            stack,
            leaf_pos: 0,
            leaf_end: 0,
            leaf_contained: false,
        }
    }
}

impl<Q: Overlaps3D> Iterator for RegionSearch3DIter<'_, Q> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let index = self.index;
        loop {
            while self.leaf_pos < self.leaf_end {
                let at = self.leaf_pos;
                self.leaf_pos += 1;
                if self.leaf_contained || self.query.overlaps_box(index.entries[at]) {
                    return Some(index.indices[at]);
                }
            }

            let frame = self.stack.pop()?;
            let end = (frame.node_index + index.node_size).min(index.level_bounds[frame.level]);

            if frame.node_index < index.num_items {
                self.leaf_pos = frame.node_index;
                self.leaf_end = end;
                self.leaf_contained = frame.contained;
            } else if frame.contained {
                let child_level = frame.level - 1;
                for &child in index.indices[frame.node_index..end].iter().rev() {
                    self.stack.push(RegionSearch3DFrame {
                        node_index: child,
                        level: child_level,
                        contained: true,
                    });
                }
            } else {
                let child_level = frame.level - 1;
                for (b, &child) in index.entries[frame.node_index..end]
                    .iter()
                    .zip(&index.indices[frame.node_index..end])
                    .rev()
                {
                    if self.query.overlaps_box(*b) {
                        self.stack.push(RegionSearch3DFrame {
                            node_index: child,
                            level: child_level,
                            contained: self.query.contains_box(*b),
                        });
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.index.num_items))
    }
}

impl<Q: Overlaps3D> std::iter::FusedIterator for RegionSearch3DIter<'_, Q> {}
