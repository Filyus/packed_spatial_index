//! Static spatial index implementation for 3D AABBs.
//!
//! `Index3D` mirrors the scalar `Index2D` API: build with
//! [`crate::Index3DBuilder`], then run overlap searches or exact nearest-neighbor
//! queries against the finished read-only tree.

use std::{collections::BinaryHeap, ops::ControlFlow};

use crate::{
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    frustum::Frustum3D,
    geometry::{Box3D, Point3D},
    join::{JoinTree, join_core, self_join_core},
    neighbors::{
        NeighborNodeState, NeighborQuery3D, NeighborState, NeighborWorkspace, max_distance_squared,
    },
    persistence::{
        LoadError, MetaFields, ParsedPayload, PayloadError, build_id_to_leaf, parse_index,
        payload_slice, read_f64_le_unchecked, read_u64_le_unchecked, write_index_container,
    },
    ray::Ray3D,
    traversal::{SearchWorkspace, prefetch_read, upper_bound_level},
    triangle::{Triangle3, blobs_as_records, records_as_bytes},
};

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

    /// Serialize in the **interleaved** layout (streaming-tuned). Shorthand for
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

    /// Load an owned 3D index from bytes previously produced by [`Index3D::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let view = Index3DView::from_bytes(bytes)?;

        let mut level_bounds = Vec::with_capacity(view.level_count);
        for i in 0..view.level_count {
            level_bounds.push(view.level_bound_unchecked(i));
        }

        let mut entries = Vec::with_capacity(view.num_nodes);
        for i in 0..view.num_nodes {
            entries.push(view.entry_at_unchecked(i));
        }

        let mut indices = Vec::with_capacity(view.num_nodes);
        for i in 0..view.num_nodes {
            indices.push(view.index_at_unchecked(i));
        }

        Ok(Self {
            node_size: view.node_size,
            num_items: view.num_items,
            level_bounds,
            entries,
            indices,
        })
    }

    /// Return the indices of all items whose boxes intersect `query`.
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
    pub fn search(&self, query: Box3D) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(query, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, query: Box3D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(query, results, &mut stack);
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
    pub fn search_with<'a>(&self, query: Box3D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.search_into_stack(query, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `query`.
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
    pub fn any(&self, query: Box3D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
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
    pub fn first(&self, query: Box3D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
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

    /// Visit intersecting items without collecting a result `Vec`.
    ///
    /// The visitor receives item positions in the original insertion order.
    /// Return [`ControlFlow::Continue`] to continue traversal or
    /// [`ControlFlow::Break`] for early exit with a user-provided value.
    pub fn visit<B, F>(&self, query: Box3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(query, &mut stack, visitor)
    }

    /// Return a lazy iterator over the items intersecting `query`.
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
    pub fn search_iter(&self, query: Box3D) -> Search3DIter<'_> {
        Search3DIter::new(self, query)
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

    fn collect_neighbors_with_queues(
        &self,
        query: NeighborQuery3D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        item_queue: &mut BinaryHeap<NeighborState>,
        node_queue: &mut BinaryHeap<NeighborNodeState>,
    ) {
        item_queue.clear();
        node_queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if self.num_items == 0 {
            return;
        }

        let root_index = self.entries.len() - 1;
        let root_dist = query.distance_squared_to(self.entries[root_index]);
        if root_dist > max_dist_sq {
            return;
        }
        node_queue.push(NeighborNodeState::new(root_index, root_dist));

        while results.len() < max_results {
            while let Some(&node) = node_queue.peek() {
                if node.dist > max_dist_sq {
                    node_queue.clear();
                    break;
                }
                if item_queue.peek().is_some_and(|item| item.dist < node.dist) {
                    break;
                }

                let node = node_queue.pop().unwrap();
                let upper_bound_level = upper_bound_level(&self.level_bounds, node.index);
                let end = (node.index + self.node_size).min(self.level_bounds[upper_bound_level]);
                let is_leaf = node.index < self.num_items;

                if is_leaf {
                    for pos in node.index..end {
                        let dist = query.distance_squared_to(self.entries[pos]);
                        if dist <= max_dist_sq {
                            item_queue.push(NeighborState::new(self.indices[pos], true, dist));
                        }
                    }
                } else {
                    for pos in node.index..end {
                        let dist = query.distance_squared_to(self.entries[pos]);
                        if dist <= max_dist_sq {
                            node_queue.push(NeighborNodeState::new(self.indices[pos], dist));
                        }
                    }
                }
            }

            match item_queue.pop() {
                Some(state) if state.dist <= max_dist_sq => results.push(state.index),
                _ => return,
            }
        }
    }

    fn nearest_one_with_queue(
        &self,
        query: NeighborQuery3D,
        max_distance: f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        queue.clear();
        let mut best_dist = max_distance_squared(max_distance)?;
        if self.num_items == 0 {
            return None;
        }

        let mut best_index = None;
        let mut node_index = self.entries.len() - 1;
        loop {
            let upper_bound_level = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let dist = query.distance_squared_to(self.entries[pos]);
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
        query: NeighborQuery3D,
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

        let mut node_index = self.entries.len() - 1;
        loop {
            let upper_bound_level = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let dist = query.distance_squared_to(self.entries[pos]);
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

    /// Item indices whose box overlaps the view frustum `frustum`.
    ///
    /// A **conservative** culling query: it returns every item whose box is inside
    /// or crosses the frustum, and may include a few boxes that lie just outside a
    /// frustum edge or corner (the standard p-vertex test). It never drops a
    /// visible box. Far tighter than `search` over the frustum's bounding box,
    /// which pulls in the whole corner volume the frustum's slanted sides miss.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index3DBuilder, Box3D, Frustum3D};
    ///
    /// let mut b = Index3DBuilder::new(2);
    /// b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)); // inside the unit cube
    /// b.add(Box3D::new(9.0, 9.0, 9.0, 9.5, 9.5, 9.5)); // far outside
    /// let index = b.finish()?;
    ///
    /// // Six axis-aligned planes bounding the unit cube [0,2]^3.
    /// let frustum = Frustum3D::from_planes([
    ///     [1.0, 0.0, 0.0, 0.0],   // x >= 0
    ///     [-1.0, 0.0, 0.0, 2.0],  // x <= 2
    ///     [0.0, 1.0, 0.0, 0.0],   // y >= 0
    ///     [0.0, -1.0, 0.0, 2.0],  // y <= 2
    ///     [0.0, 0.0, 1.0, 0.0],   // z >= 0
    ///     [0.0, 0.0, -1.0, 2.0],  // z <= 2
    /// ]);
    /// assert_eq!(index.search_frustum(frustum), vec![0]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn search_frustum(&self, frustum: Frustum3D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_frustum_into(frustum, &mut out);
        out
    }

    /// [`search_frustum`](Self::search_frustum) into a reused buffer (cleared first).
    pub fn search_frustum_into(&self, frustum: Frustum3D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_frustum_with_stack(frustum, &mut stack, |i| {
            out.push(i);
            ControlFlow::<()>::Continue(())
        });
    }

    /// Whether any item's box overlaps `frustum`, short-circuiting on the first
    /// hit (conservative, like [`search_frustum`](Self::search_frustum)).
    pub fn any_frustum(&self, frustum: Frustum3D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_frustum_with_stack(frustum, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Visit each item whose box overlaps `frustum`; return [`ControlFlow::Break`]
    /// from `visitor` to stop early (conservative).
    pub fn visit_frustum<B, F>(&self, frustum: Frustum3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_frustum_with_stack(frustum, &mut stack, visitor)
    }

    fn visit_frustum_with_stack<B, F>(
        &self,
        frustum: Frustum3D,
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

        const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
        const LEVEL_MASK: usize = !CONTAINED_FLAG;

        let root = self.entries[self.entries.len() - 1];
        if frustum.contains_box(root) {
            for &index in &self.indices[..self.num_items] {
                visitor(index)?;
            }
            return ControlFlow::Continue(());
        }

        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_entries = &self.entries[node_index..end];
            let node_indices = &self.indices[node_index..end];

            if contained {
                let start = self.leaf_start_for_entry(node_index, level);
                let leaf_end = if end < self.level_bounds[level] {
                    self.leaf_start_for_entry(end, level)
                } else {
                    self.num_items
                };
                for &index in &self.indices[start..leaf_end] {
                    visitor(index)?;
                }
            } else if is_leaf {
                for (b, &index) in node_entries.iter().zip(node_indices) {
                    if !frustum.overlaps_box(*b) {
                        continue;
                    }
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for (b, &index) in node_entries.iter().zip(node_indices).rev() {
                    if !frustum.overlaps_box(*b) {
                        continue;
                    }
                    stack.push(index);
                    let encoded_level = if frustum.contains_box(*b) {
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
                return ControlFlow::Continue(());
            }
        }
    }

    /// Diagnostics for the frustum query: `(results, nodes_visited, plane_tests,
    /// contained_subtrees)`. `plane_tests` counts `overlaps_box` calls (the cost
    /// the bounding-box query avoids), `contained_subtrees` the whole subtrees
    /// accepted without per-item tests.
    #[doc(hidden)]
    pub fn search_frustum_visited(&self, frustum: Frustum3D) -> (usize, usize, usize, usize) {
        let (mut results, mut visited, mut planes, mut contained_subtrees) = (0, 0, 0, 0);
        if self.num_items == 0 {
            return (0, 0, 0, 0);
        }
        const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
        const LEVEL_MASK: usize = !CONTAINED_FLAG;
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            if contained {
                let start = self.leaf_start_for_entry(node_index, level);
                let leaf_end = if end < self.level_bounds[level] {
                    self.leaf_start_for_entry(end, level)
                } else {
                    self.num_items
                };
                results += leaf_end - start;
            } else {
                for pos in node_index..end {
                    visited += 1;
                    planes += 1;
                    let b = self.entries[pos];
                    if !frustum.overlaps_box(b) {
                        continue;
                    }
                    if is_leaf {
                        results += 1;
                    } else {
                        stack.push(self.indices[pos]);
                        if frustum.contains_box(b) {
                            contained_subtrees += 1;
                            stack.push((level - 1) | CONTAINED_FLAG);
                        } else {
                            stack.push(level - 1);
                        }
                    }
                }
            }
            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return (results, visited, planes, contained_subtrees);
            }
        }
    }

    #[inline]
    fn leaf_start_for_entry(&self, mut index: usize, mut level: usize) -> usize {
        while level > 0 {
            index = self.indices[index];
            level -= 1;
        }
        index
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
        if query.contains(root) {
            stack.clear();
            results.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }

        self.search_into_stack_impl(query, results, stack);
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

    /// Return the indices of all items whose boxes intersect `query`.
    pub fn search(&self, query: Box3D) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(query, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, query: Box3D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(query, results, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'b>(&self, query: Box3D, workspace: &'b mut SearchWorkspace) -> &'b [usize] {
        self.search_into_stack(query, &mut workspace.results, &mut workspace.stack);
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

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(query, &mut stack, visitor)
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

    fn collect_neighbors_with_queues(
        &self,
        query: NeighborQuery3D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
        item_queue: &mut BinaryHeap<NeighborState>,
        node_queue: &mut BinaryHeap<NeighborNodeState>,
    ) {
        item_queue.clear();
        node_queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if self.num_items == 0 {
            return;
        }

        let root_index = self.num_nodes - 1;
        let root_dist = query.distance_squared_to(self.entry_at_unchecked(root_index));
        if root_dist > max_dist_sq {
            return;
        }
        node_queue.push(NeighborNodeState::new(root_index, root_dist));

        while results.len() < max_results {
            while let Some(&node) = node_queue.peek() {
                if node.dist > max_dist_sq {
                    node_queue.clear();
                    break;
                }
                if item_queue.peek().is_some_and(|item| item.dist < node.dist) {
                    break;
                }

                let node = node_queue.pop().unwrap();
                let upper_bound_level = self.upper_bound_level(node.index);
                let end = (node.index + self.node_size)
                    .min(self.level_bound_unchecked(upper_bound_level));
                let is_leaf = node.index < self.num_items;

                if is_leaf {
                    for pos in node.index..end {
                        let b = self.entry_at_unchecked(pos);
                        let dist = query.distance_squared_to(b);
                        if dist <= max_dist_sq {
                            item_queue.push(NeighborState::new(
                                self.index_at_unchecked(pos),
                                true,
                                dist,
                            ));
                        }
                    }
                } else {
                    for pos in node.index..end {
                        let b = self.entry_at_unchecked(pos);
                        let dist = query.distance_squared_to(b);
                        if dist <= max_dist_sq {
                            node_queue
                                .push(NeighborNodeState::new(self.index_at_unchecked(pos), dist));
                        }
                    }
                }
            }

            match item_queue.pop() {
                Some(state) if state.dist <= max_dist_sq => results.push(state.index),
                _ => return,
            }
        }
    }

    fn nearest_one_with_queue(
        &self,
        query: NeighborQuery3D,
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
            let upper_bound_level = self.upper_bound_level(node_index);
            let end =
                (node_index + self.node_size).min(self.level_bound_unchecked(upper_bound_level));
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let b = self.entry_at_unchecked(pos);
                let dist = query.distance_squared_to(b);
                if dist > best_dist {
                    continue;
                }
                if is_leaf {
                    if dist == 0.0 {
                        return Some(self.index_at_unchecked(pos));
                    }
                    best_dist = dist;
                    best_index = Some(self.index_at_unchecked(pos));
                } else {
                    queue.push(NeighborNodeState::new(self.index_at_unchecked(pos), dist));
                }
            }

            match queue.pop() {
                Some(state) if state.dist <= best_dist => node_index = state.index,
                _ => return best_index,
            }
        }
    }

    #[doc(hidden)]
    pub fn search_into_stack(
        &self,
        query: Box3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        results.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }

        let root = self.entry_at_unchecked(self.num_nodes - 1);
        if query.contains(root) {
            for pos in 0..self.num_items {
                results.push(self.index_at_unchecked(pos));
            }
            return;
        }

        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;

            if is_leaf {
                for pos in node_index..end {
                    let b = self.entry_at_unchecked(pos);
                    if !b.overlaps(query) {
                        continue;
                    }
                    results.push(self.index_at_unchecked(pos));
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.entry_at_unchecked(pos);
                    if !b.overlaps(query) {
                        continue;
                    }
                    stack.push(self.index_at_unchecked(pos));
                    stack.push(child_level);
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
        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;

            if is_leaf {
                for pos in node_index..end {
                    let b = self.entry_at_unchecked(pos);
                    if !b.overlaps(query) {
                        continue;
                    }
                    visitor(self.index_at_unchecked(pos))?;
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.entry_at_unchecked(pos);
                    if !b.overlaps(query) {
                        continue;
                    }
                    stack.push(self.index_at_unchecked(pos));
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

    /// Item indices whose box overlaps the view frustum `frustum`. The zero-copy
    /// view counterpart of [`Index3D::search_frustum`] (conservative culling).
    pub fn search_frustum(&self, frustum: Frustum3D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_frustum_into(frustum, &mut out);
        out
    }

    /// [`search_frustum`](Self::search_frustum) into a reused buffer (cleared first).
    pub fn search_frustum_into(&self, frustum: Frustum3D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_region_with_stack(
            &mut stack,
            |b| frustum.overlaps_box(b),
            |b| frustum.contains_box(b),
            |i| {
                out.push(i);
                ControlFlow::<()>::Continue(())
            },
        );
    }

    /// Whether any item's box overlaps `frustum`, short-circuiting (conservative).
    pub fn any_frustum(&self, frustum: Frustum3D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| frustum.overlaps_box(b),
            |b| frustum.contains_box(b),
            |_| ControlFlow::Break(()),
        )
        .is_break()
    }

    /// Visit each item whose box overlaps `frustum`; return [`ControlFlow::Break`]
    /// to stop early (conservative).
    pub fn visit_frustum<B, F>(&self, frustum: Frustum3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| frustum.overlaps_box(b),
            |b| frustum.contains_box(b),
            visitor,
        )
    }

    /// Shared region traversal over the byte-backed tree, with the contained
    /// fast path: `overlaps` prunes/leaf-tests, `contains` accepts whole subtrees.
    fn visit_region_with_stack<B>(
        &self,
        stack: &mut Vec<usize>,
        overlaps: impl Fn(Box3D) -> bool,
        contains: impl Fn(Box3D) -> bool,
        mut visitor: impl FnMut(usize) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
        const LEVEL_MASK: usize = !CONTAINED_FLAG;

        let root = self.entry_at_unchecked(self.num_nodes - 1);
        if contains(root) {
            for pos in 0..self.num_items {
                visitor(self.index_at_unchecked(pos))?;
            }
            return ControlFlow::Continue(());
        }

        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        let mut contained = false;

        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;

            if contained {
                let start = self.leaf_start_for_entry(node_index, level);
                let leaf_end = if end < self.level_bound_unchecked(level) {
                    self.leaf_start_for_entry(end, level)
                } else {
                    self.num_items
                };
                for pos in start..leaf_end {
                    visitor(self.index_at_unchecked(pos))?;
                }
            } else if is_leaf {
                for pos in node_index..end {
                    let b = self.entry_at_unchecked(pos);
                    if !overlaps(b) {
                        continue;
                    }
                    visitor(self.index_at_unchecked(pos))?;
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.entry_at_unchecked(pos);
                    if !overlaps(b) {
                        continue;
                    }
                    stack.push(self.index_at_unchecked(pos));
                    let encoded_level = if contains(b) {
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
                return ControlFlow::Continue(());
            }
        }
    }

    /// Leaf-section start of the subtree rooted at entry `index` on `level`.
    #[inline]
    fn leaf_start_for_entry(&self, mut index: usize, mut level: usize) -> usize {
        while level > 0 {
            index = self.index_at_unchecked(index);
            level -= 1;
        }
        index
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
        queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return ControlFlow::Continue(());
        };
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.num_nodes - 1;
        loop {
            let upper_bound_level = self.upper_bound_level(node_index);
            let end =
                (node_index + self.node_size).min(self.level_bound_unchecked(upper_bound_level));
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let b = self.entry_at_unchecked(pos);
                let dist = query.distance_squared_to(b);
                if dist > max_dist_sq {
                    continue;
                }
                queue.push(NeighborState::new(
                    self.index_at_unchecked(pos),
                    is_leaf,
                    dist,
                ));
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

impl JoinTree for Index3D {
    type Bounds = Box3D;

    #[inline]
    fn join_num_items(&self) -> usize {
        self.num_items
    }
    #[inline]
    fn join_num_nodes(&self) -> usize {
        self.entries.len()
    }
    #[inline]
    fn join_node_size(&self) -> usize {
        self.node_size
    }
    #[inline]
    fn join_level_count(&self) -> usize {
        self.level_bounds.len()
    }
    #[inline]
    fn join_level_bound(&self, level: usize) -> usize {
        self.level_bounds[level]
    }
    #[inline]
    fn join_bounds(&self, pos: usize) -> Box3D {
        self.entries[pos]
    }
    #[inline]
    fn join_index(&self, pos: usize) -> usize {
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

impl JoinTree for Index3DView<'_> {
    type Bounds = Box3D;

    #[inline]
    fn join_num_items(&self) -> usize {
        self.num_items
    }
    #[inline]
    fn join_num_nodes(&self) -> usize {
        self.num_nodes
    }
    #[inline]
    fn join_node_size(&self) -> usize {
        self.node_size
    }
    #[inline]
    fn join_level_count(&self) -> usize {
        self.level_count
    }
    #[inline]
    fn join_level_bound(&self, level: usize) -> usize {
        self.level_bound_unchecked(level)
    }
    #[inline]
    fn join_bounds(&self, pos: usize) -> Box3D {
        self.entry_at_unchecked(pos)
    }
    #[inline]
    fn join_index(&self, pos: usize) -> usize {
        self.index_at_unchecked(pos)
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

impl Index3D {
    /// Return the indices of all items whose boxes the ray segment touches.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Index3DBuilder, Point3D, Ray3D};
    ///
    /// let mut builder = Index3DBuilder::new(2);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    /// builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let ray = Ray3D::new(Point3D::new(-1.0, 0.5, 0.5), 1.0, 0.0, 0.0, 10.0);
    /// assert_eq!(index.raycast(ray), vec![0]);
    /// assert_eq!(index.raycast_closest(ray), Some((0, 1.0)));
    /// ```
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

    /// Buffer-explicit raycast (mirrors `search_into_stack`).
    #[doc(hidden)]
    pub fn raycast_into_stack(&self, ray: Ray3D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        results.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }

        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_entries = &self.entries[node_index..end];
            let node_indices = &self.indices[node_index..end];

            if is_leaf {
                for (&bounds, &index) in node_entries.iter().zip(node_indices) {
                    if ray.intersects_box(bounds) {
                        results.push(index);
                    }
                }
            } else {
                let child_level = level - 1;
                for (&bounds, &index) in node_entries.iter().zip(node_indices).rev() {
                    if ray.intersects_box(bounds) {
                        stack.push(index);
                        stack.push(child_level);
                    }
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
    ///
    /// Nodes are visited front-to-back by entry distance and pruned once a
    /// closer hit is known, so the cost is roughly independent of
    /// `max_distance` after the first hit. `t` is `0.0` when the ray origin
    /// starts inside the item's box, and is measured in units of the ray
    /// direction length (see [`Ray3D::new`]).
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
        let root = self.entries.len() - 1;
        let root_t = ray.enter_t(self.entries[root])?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        while let Some(node) = queue.pop() {
            // The heap yields nodes by ascending entry t, and a node's entry t is a
            // lower bound on every descendant's, so once it reaches the best hit we stop.
            if node.dist >= best_t {
                break;
            }
            let upper = upper_bound_level(&self.level_bounds, node.index);
            let end = (node.index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node.index < self.num_items;
            for pos in node.index..end {
                let Some(t) = ray.enter_t(self.entries[pos]) else {
                    continue;
                };
                if t >= best_t {
                    continue;
                }
                if is_leaf {
                    best_t = t;
                    best_index = Some(self.indices[pos]);
                } else {
                    queue.push(NeighborNodeState::new(self.indices[pos], t));
                }
            }
        }

        best_index.map(|index| (index, best_t))
    }
}

impl Index3D {
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

        let mut node_index = self.entries.len() - 1;
        loop {
            let upper = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                if let Some(t) = ray.enter_t(self.entries[pos]) {
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

impl Index3DView<'_> {
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
                if !ray.intersects_box(self.entry_at_unchecked(pos)) {
                    continue;
                }
                let index = self.index_at_unchecked(pos);
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
        let root_t = ray.enter_t(self.entry_at_unchecked(root))?;
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
                let Some(t) = ray.enter_t(self.entry_at_unchecked(pos)) else {
                    continue;
                };
                if t >= best_t {
                    continue;
                }
                if is_leaf {
                    best_t = t;
                    best_index = Some(self.index_at_unchecked(pos));
                } else {
                    queue.push(NeighborNodeState::new(self.index_at_unchecked(pos), t));
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
                if let Some(t) = ray.enter_t(self.entry_at_unchecked(pos)) {
                    queue.push(NeighborState::new(self.index_at_unchecked(pos), is_leaf, t));
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

/// Lazy iterator over the items intersecting a query box, returned by
/// [`Index3D::search_iter`].
///
/// Yields original insertion indices in tree-traversal order, descending the
/// tree only as far as the consumer pulls. Holds a small traversal stack
/// (`O(depth)`); it allocates no result `Vec`.
pub struct Search3DIter<'a> {
    index: &'a Index3D,
    query: Box3D,
    // (node_index, level) pairs still to visit, same encoding as the search stack.
    stack: Vec<usize>,
    // Half-open entry range of the leaf node currently being scanned.
    leaf_pos: usize,
    leaf_end: usize,
}

impl<'a> Search3DIter<'a> {
    fn new(index: &'a Index3D, query: Box3D) -> Self {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        if index.num_items != 0 {
            // Seed with the root so `next` drives the descent uniformly.
            stack.push(index.entries.len() - 1);
            stack.push(index.level_bounds.len() - 1);
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
                if index.entries[at].overlaps(self.query) {
                    return Some(index.indices[at]);
                }
            }

            // Pop the next node. The stack holds (node_index, level) pairs.
            if self.stack.len() < 2 {
                return None;
            }
            let level = self.stack.pop().unwrap();
            let node_index = self.stack.pop().unwrap();
            let end = (node_index + index.node_size).min(index.level_bounds[level]);

            if node_index < index.num_items {
                // Leaf node: scan its entries on the next loop turns.
                self.leaf_pos = node_index;
                self.leaf_end = end;
            } else {
                // Internal node: push overlapping children reversed so they pop
                // in forward order (matching `visit`).
                let child_level = level - 1;
                for (b, &child) in index.entries[node_index..end]
                    .iter()
                    .zip(&index.indices[node_index..end])
                    .rev()
                {
                    if b.overlaps(self.query) {
                        self.stack.push(child);
                        self.stack.push(child_level);
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

/// Builder for [`Index3D`] serialization, created by [`Index3D::serialize`]. The
/// 3D counterpart of [`Serializer2D`](crate::Serializer2D): optional per-item
/// payloads, the streaming-tuned interleaved layout, and descriptive metadata
/// (CRS / content type / attribution), read back with
/// [`read_metadata`](crate::read_metadata).
pub struct Serializer3D<'a> {
    index: &'a Index3D,
    interleaved: bool,
    payloads: Option<Vec<&'a [u8]>>,
    /// `Some(stride)` selects the fixed-width (table-less) payload layout.
    record_stride: Option<u32>,
    meta: MetaFields<'a>,
}

impl<'a> Serializer3D<'a> {
    fn new(index: &'a Index3D) -> Self {
        Self {
            index,
            interleaved: false,
            payloads: None,
            record_stride: None,
            meta: MetaFields::default(),
        }
    }

    /// Attach one opaque payload blob per item, in item order.
    pub fn payloads<P: AsRef<[u8]>>(mut self, payloads: &'a [P]) -> Self {
        self.payloads = Some(payloads.iter().map(|p| p.as_ref()).collect());
        self
    }

    /// Attach a **fixed-width** payload: `flat` is the concatenation of one
    /// `stride`-byte record per item, in item order (item `i` is
    /// `flat[i * stride ..][.. stride]`). Because every record is the same size,
    /// the offset table is dropped (the reader addresses record `r` by
    /// arithmetic), which shrinks the file and lets a view borrow the records as
    /// a zero-copy typed slice. `flat.len()` must be `num_items * stride`.
    pub fn records(mut self, stride: usize, flat: &'a [u8]) -> Self {
        self.record_stride = Some(stride as u32);
        self.payloads = Some(if stride == 0 {
            Vec::new()
        } else {
            flat.chunks_exact(stride).collect()
        });
        self
    }

    /// Attach a fixed-width triangle payload (`T` =
    /// [`Triangle3D`](crate::Triangle3D) for `f64` or
    /// [`Triangle3DF32`](crate::Triangle3DF32) for `f32`): one triangle per item,
    /// in item order. A convenience over [`records`](Self::records); pair it with
    /// [`Index3D::from_triangles`](crate::Index3D::from_triangles) to get a
    /// streamable bounding-volume hierarchy over a mesh.
    pub fn triangles<T: Triangle3>(self, triangles: &'a [T]) -> Self {
        let bytes = records_as_bytes(triangles);
        self.records(T::STRIDE, bytes)
    }

    /// Use the streaming-tuned interleaved node layout.
    #[cfg(feature = "stream")]
    pub fn interleaved(mut self) -> Self {
        self.interleaved = true;
        self
    }

    /// Set the coordinate reference system identifier (opaque, e.g. `"EPSG:4979"`).
    pub fn crs(mut self, crs: &'a str) -> Self {
        self.meta.crs = Some(crs);
        self
    }

    /// Set the payload content type / media type.
    pub fn content_type(mut self, content_type: &'a str) -> Self {
        self.meta.content_type = Some(content_type);
        self
    }

    /// Set an attribution / license string.
    pub fn attribution(mut self, attribution: &'a str) -> Self {
        self.meta.attribution = Some(attribution);
        self
    }

    /// Serialize into a new buffer.
    pub fn to_bytes(self) -> Result<Vec<u8>, PayloadError> {
        let mut out = Vec::new();
        self.to_bytes_into(&mut out)?;
        Ok(out)
    }

    /// Serialize into a reused buffer (cleared first).
    pub fn to_bytes_into(self, out: &mut Vec<u8>) -> Result<(), PayloadError> {
        let idx = self.index;
        let interleaved = self.interleaved;
        let record_stride = self.record_stride;
        write_index_container(
            out,
            3,
            8,
            interleaved,
            idx.num_items,
            idx.entries.len(),
            idx.node_size,
            |bytes| {
                #[cfg(feature = "stream")]
                if interleaved {
                    bytes.write_interleaved_3d(&idx.entries, &idx.indices);
                    return;
                }
                bytes.write_box3d_slice(&idx.entries);
                bytes.write_usize_slice_as_u64(&idx.indices);
            },
            self.payloads.as_deref(),
            record_stride,
            &idx.indices[..idx.num_items],
            &self.meta,
        )
    }
}
