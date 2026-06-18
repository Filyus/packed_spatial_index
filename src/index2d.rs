//! Static spatial index implementation for 2D AABBs:
//! a packed Hilbert R-tree in the style of flatbush / `static_aabb2d_index`.
//!
//! The public API is intentionally small: add boxes with [`crate::Index2DBuilder`],
//! call [`crate::Index2DBuilder::finish`], then search the finished [`Index2D`].
//!
//! # Example
//! ```
//! use packed_spatial_index::{Index2DBuilder, Box2D};
//!
//! let mut builder = Index2DBuilder::new(3);
//! builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
//! builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
//! builder.add(Box2D::new(0.5, 0.5, 2.0, 2.0));
//! let index = builder.finish().unwrap();
//!
//! let hits = index.search(Box2D::new(0.0, 0.0, 1.5, 1.5));
//! assert!(hits.contains(&0) && hits.contains(&2));
//! assert!(!hits.contains(&1));
//! ```

use std::{collections::BinaryHeap, ops::ControlFlow};

use crate::config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY};
use crate::geometry::{Box2D, Point2D};
use crate::join::{JoinTree, join_core, self_join_core};
use crate::neighbors::{
    NeighborNodeState, NeighborQuery2D, NeighborState, NeighborWorkspace, max_distance_squared,
};
use crate::persistence::{
    LoadError, MetaFields, ParsedPayload, PayloadError, build_id_to_leaf, parse_index,
    payload_slice, read_f64_le_unchecked, read_u64_le_unchecked, write_index_container,
};
use crate::ray::Ray2D;
use crate::traversal::{SearchWorkspace, prefetch_read, upper_bound_level};
use crate::triangle::{Triangle2, Triangle2D, blobs_as_records, records_as_bytes};

#[inline]
fn prefetch_aos_node(entries: &[Box2D], indices: &[usize], node_index: usize, node_size: usize) {
    if node_index < entries.len() {
        prefetch_read(entries.as_ptr().wrapping_add(node_index));
        prefetch_read(indices.as_ptr().wrapping_add(node_index));
    }
    let next_line = node_index.saturating_add((64 / std::mem::size_of::<Box2D>()).max(1));
    if node_size > 1 && next_line < entries.len() {
        prefetch_read(entries.as_ptr().wrapping_add(next_line));
        prefetch_read(indices.as_ptr().wrapping_add(next_line));
    }
}

/// Finished static read-only index.
///
/// Search methods return item positions in the original insertion order. The order
/// of returned results is traversal order and is not part of the API.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, Box2D};
///
/// let mut builder = Index2DBuilder::new(2);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
/// let index = builder.finish().unwrap();
///
/// assert_eq!(index.num_items(), 2);
/// assert_eq!(index.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
/// ```
pub struct Index2D {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) entries: Vec<Box2D>,
    pub(crate) indices: Vec<usize>,
}

impl Index2D {
    /// Return the number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total extent of indexed items, or `None` for an empty index.
    pub fn extent(&self) -> Option<Box2D> {
        self.entries.last().copied()
    }

    /// Return the packed node size used by this index.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Serialize this index into the stable little-endian `PSINDEX` format.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2D, Index2DBuilder, Index2DView, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// let index = builder.finish()?;
    ///
    /// let bytes = index.to_bytes();
    /// let owned = Index2D::from_bytes(&bytes)?;
    /// let view = Index2DView::from_bytes(&bytes)?;
    ///
    /// let query = Box2D::new(0.5, 0.5, 0.5, 0.5);
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
    /// multi-megabyte allocation and page-faulting, which dominates the cost for large
    /// indexes.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2D, Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// let index = builder.finish()?;
    ///
    /// let mut buffer = Vec::new();
    /// index.to_bytes_into(&mut buffer);
    /// assert_eq!(buffer, index.to_bytes());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        self.serialize()
            .to_bytes_into(out)
            .expect("serialization without payloads cannot fail");
    }

    /// Serialize this index together with one opaque payload per item, producing
    /// a self-contained file (the spatial index plus the data it indexes).
    ///
    /// `payloads` is in item order: `payloads[i]` is the blob for the item added
    /// `i`-th. Read them back via `StreamIndex2D::search_payloads` (`stream`
    /// feature) or [`Index2DView::search_payloads`] / [`Index2DView::payload`].
    /// Returns [`PayloadError::CountMismatch`] unless `payloads.len()` equals
    /// [`num_items`](Self::num_items). Shorthand for
    /// [`serialize().payloads(..)`](Self::serialize).
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder};
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// let index = builder.finish()?;
    /// let bytes = index.to_bytes_with_payloads(&[b"first".as_slice(), b"second"])?;
    /// assert!(bytes.len() > index.to_bytes().len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    /// index), a streaming-tuned layout a [`StreamIndex2D`] fetches in one read
    /// per level instead of two. The in-memory loaders and SIMD views read the
    /// default layout only. Shorthand for
    /// [`serialize().interleaved()`](Self::serialize); available with `stream`.
    ///
    /// [`StreamIndex2D`]: crate::StreamIndex2D
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
    /// (CRS / content type / attribution). See [`Serializer2D`].
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder};
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// let index = builder.finish()?;
    /// let bytes = index
    ///     .serialize()
    ///     .crs("EPSG:4326")
    ///     .payloads(&[b"feature-0".as_slice()])
    ///     .to_bytes()?;
    /// assert_eq!(packed_spatial_index::read_metadata(&bytes)?.crs.as_deref(), Some("EPSG:4326"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn serialize(&self) -> Serializer2D<'_> {
        Serializer2D::new(self)
    }

    /// Build an index over the bounding box of each triangle, in slice order
    /// (item `i` is `triangles[i]`). A convenience over looping
    /// [`Index2DBuilder::add`](crate::Index2DBuilder::add) with
    /// [`Triangle2::aabb`](crate::Triangle2::aabb); the index is queryable in memory, and
    /// `index.serialize().triangles(triangles)` stores the geometry alongside it.
    /// Use the builder directly for custom boxes or build options like `node_size`.
    pub fn from_triangles<T: Triangle2>(triangles: &[T]) -> Result<Self, crate::BuildError> {
        let mut builder = crate::Index2DBuilder::new(triangles.len());
        for t in triangles {
            builder.add(t.aabb());
        }
        builder.finish()
    }

    /// Load an owned index from bytes previously produced by [`Index2D::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let view = Index2DView::from_bytes(bytes)?;

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
    /// # use packed_spatial_index::{Index2DBuilder, Box2D};
    /// # let mut builder = Index2DBuilder::new(2);
    /// # builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// # builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// # let index = builder.finish().unwrap();
    /// assert_eq!(index.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
    /// ```
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(query, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, query: Box2D, results: &mut Vec<usize>) {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(query, results, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D, SearchWorkspace};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut workspace = SearchWorkspace::with_capacity(8, 8);
    /// let hits = index.search_with(Box2D::new(0.0, 0.0, 2.0, 2.0), &mut workspace);
    /// assert_eq!(hits, &[0]);
    /// assert_eq!(workspace.results(), &[0]);
    /// ```
    pub fn search_with<'a>(&self, query: Box2D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
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
    /// # use packed_spatial_index::{Index2DBuilder, Box2D};
    /// # let mut builder = Index2DBuilder::new(2);
    /// # builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// # builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// # let index = builder.finish().unwrap();
    /// assert!(index.any(Box2D::new(0.0, 0.0, 2.0, 2.0)));
    /// assert!(!index.any(Box2D::new(20.0, 20.0, 21.0, 21.0)));
    /// ```
    pub fn any(&self, query: Box2D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    ///
    /// Tree traversal order is not part of the API, so this returns just some first
    /// found item, not the minimum insertion index.
    ///
    /// # Example
    ///
    /// ```
    /// # use packed_spatial_index::{Index2DBuilder, Box2D};
    /// # let mut builder = Index2DBuilder::new(2);
    /// # builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// # builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// # let index = builder.finish().unwrap();
    /// assert_eq!(index.first(Box2D::new(0.0, 0.0, 2.0, 2.0)), Some(0));
    /// assert_eq!(index.first(Box2D::new(20.0, 20.0, 21.0, 21.0)), None);
    /// ```
    pub fn first(&self, query: Box2D) -> Option<usize> {
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
    /// use packed_spatial_index::{Index2DBuilder, Point2D, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(10.0, 10.0, 11.0, 11.0));
    /// let index = builder.finish().unwrap();
    ///
    /// assert_eq!(index.neighbors(Point2D::new(10.25, 10.25), 1), vec![1]);
    /// ```
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery2D::Point(point),
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

        self.collect_neighbors_with_queues(
            NeighborQuery2D::Point(point),
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
    ///
    /// Distance is the box-to-box gap: items overlapping or touching `query`
    /// have distance `0.0` and come first (their mutual order is unspecified).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(10.0, 0.0, 11.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// // The query box's nearest edge is closer to item 1 than to item 0.
    /// let query = Box2D::new(7.0, 0.0, 8.0, 1.0);
    /// assert_eq!(index.neighbors_of_box(query, 1), vec![1]);
    /// ```
    pub fn neighbors_of_box(&self, query: Box2D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`neighbors_of_box`](Self::neighbors_of_box).
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery2D::Box(query),
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

        self.collect_neighbors_with_queues(
            NeighborQuery2D::Box(query),
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
    ///
    /// The visitor receives item positions in the original insertion order. Return
    /// [`ControlFlow::Continue`] to continue traversal or [`ControlFlow::Break`] for
    /// early exit with a user-provided value.
    ///
    /// # Example
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let found = index.visit(Box2D::new(5.0, 5.0, 6.0, 6.0), ControlFlow::Break);
    /// assert_eq!(found, ControlFlow::Break(1));
    /// ```
    pub fn visit<B, F>(&self, query: Box2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(query, &mut stack, visitor)
    }

    /// Item indices whose box overlaps the 2D triangle `tri`.
    ///
    /// A tight region query: like `search(tri.aabb())` but with the bounding-box
    /// corners that the triangle misses rejected during the traversal, so the
    /// result is exactly the items the triangle's filled area overlaps. Subtrees
    /// fully inside the triangle are accepted without per-item tests, so the cost
    /// stays close to the bounding-box query while the result set is tighter.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D, Triangle2D};
    ///
    /// let mut b = Index2DBuilder::new(2);
    /// b.add(Box2D::new(0.2, 0.2, 0.3, 0.3)); // inside the triangle
    /// b.add(Box2D::new(9.0, 9.0, 9.5, 9.5)); // in the bbox corner, outside the triangle
    /// let index = b.finish()?;
    ///
    /// let tri = Triangle2D::new([0.0, 0.0], [10.0, 0.0], [0.0, 10.0]);
    /// assert_eq!(index.search_triangle(tri), vec![0]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn search_triangle(&self, tri: Triangle2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_triangle_into(tri, &mut out);
        out
    }

    /// [`search_triangle`](Self::search_triangle) into a reused buffer (cleared first).
    pub fn search_triangle_into(&self, tri: Triangle2D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_triangle_with_stack(tri, &mut stack, |i| {
            out.push(i);
            ControlFlow::<()>::Continue(())
        });
    }

    /// Whether any item's box overlaps `tri`, short-circuiting on the first real
    /// hit. The triangle-tight analogue of `any(tri.aabb())`, which over-reports
    /// items that only touch the bounding box.
    pub fn any_triangle(&self, tri: Triangle2D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_triangle_with_stack(tri, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Visit each item whose box overlaps `tri`; return [`ControlFlow::Break`]
    /// from `visitor` to stop early.
    pub fn visit_triangle<B, F>(&self, tri: Triangle2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_triangle_with_stack(tri, &mut stack, visitor)
    }

    fn visit_triangle_with_stack<B, F>(
        &self,
        tri: Triangle2D,
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
        if tri.contains_box(root) {
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
                    if !tri.overlaps_box(*b) {
                        continue;
                    }
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for (b, &index) in node_entries.iter().zip(node_indices).rev() {
                    if !tri.overlaps_box(*b) {
                        continue;
                    }
                    stack.push(index);
                    let encoded_level = if tri.contains_box(*b) {
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

    /// Diagnostics for the triangle query: `(results, nodes_visited, sat_tests,
    /// contained_subtrees)`. `sat_tests` counts `overlaps_box` calls (the cost
    /// the bounding-box query avoids), `contained_subtrees` the whole subtrees
    /// accepted without per-item tests.
    #[doc(hidden)]
    pub fn search_triangle_visited(&self, tri: Triangle2D) -> (usize, usize, usize, usize) {
        let (mut results, mut visited, mut sat, mut contained_subtrees) = (0, 0, 0, 0);
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
                    sat += 1;
                    let b = self.entries[pos];
                    if !tri.overlaps_box(b) {
                        continue;
                    }
                    if is_leaf {
                        results += 1;
                    } else {
                        stack.push(self.indices[pos]);
                        if tri.contains_box(b) {
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
                return (results, visited, sat, contained_subtrees);
            }
        }
    }

    /// Return a lazy iterator over the items intersecting `query`.
    ///
    /// The tree is descended on demand, so consuming only a prefix
    /// (`.next()`, `.take(k)`, `.find(..)`) stops the traversal early and never
    /// allocates a result `Vec`. Yielded values are original insertion indices,
    /// in tree-traversal order (not part of the API). For the whole result set
    /// [`search`](Index2D::search) is more direct; reach for the iterator to
    /// compose with adapters or to bail out partway.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(3);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(2.0, 2.0, 3.0, 3.0));
    /// builder.add(Box2D::new(9.0, 9.0, 10.0, 10.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut hits: Vec<_> = index.search_iter(Box2D::new(0.0, 0.0, 4.0, 4.0)).collect();
    /// hits.sort_unstable();
    /// assert_eq!(hits, vec![0, 1]);
    /// ```
    pub fn search_iter(&self, query: Box2D) -> Search2DIter<'_> {
        Search2DIter::new(self, query)
    }

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`.
    ///
    /// A single synchronized descent over both trees replaces one full search
    /// per item, so large joins run far faster than a search loop. Pair order
    /// is traversal order and is not part of the API.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut a = Index2DBuilder::new(2);
    /// a.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// a.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// let a = a.finish().unwrap();
    ///
    /// let mut b = Index2DBuilder::new(1);
    /// b.add(Box2D::new(0.5, 0.5, 5.5, 5.5));
    /// let b = b.finish().unwrap();
    ///
    /// let mut pairs = a.join(&b);
    /// pairs.sort_unstable();
    /// assert_eq!(pairs, vec![(0, 0), (1, 0)]);
    /// ```
    pub fn join(&self, other: &Index2D) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other` without
    /// collecting a result `Vec`.
    ///
    /// The visitor receives `(item_in_self, item_in_other)` positions in the
    /// original insertion order of each index. Return [`ControlFlow::Break`]
    /// for early exit.
    pub fn join_with<B, F>(&self, other: &Index2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, visitor)
    }

    /// Return every unordered pair of distinct intersecting items within this
    /// index, each pair exactly once.
    ///
    /// This is the broad-phase primitive: pairs of items whose boxes overlap.
    /// The order of ids within a pair and the pair order are traversal order
    /// and are not part of the API.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(3);
    /// builder.add(Box2D::new(0.0, 0.0, 2.0, 2.0));
    /// builder.add(Box2D::new(1.0, 1.0, 3.0, 3.0));
    /// builder.add(Box2D::new(9.0, 9.0, 10.0, 10.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let pairs: Vec<_> = index
    ///     .self_join()
    ///     .into_iter()
    ///     .map(|(i, j)| (i.min(j), i.max(j)))
    ///     .collect();
    /// assert_eq!(pairs, vec![(0, 1)]);
    /// ```
    pub fn self_join(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.self_join_with(|i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every unordered pair of distinct intersecting items within this
    /// index without collecting a result `Vec`.
    ///
    /// Return [`ControlFlow::Break`] for early exit.
    pub fn self_join_with<B, F>(&self, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, visitor)
    }

    fn collect_neighbors_with_queues(
        &self,
        query: NeighborQuery2D,
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
                        let b = self.entries[pos];
                        let dist = query.distance_squared_to(b);
                        if dist <= max_dist_sq {
                            item_queue.push(NeighborState::new(self.indices[pos], true, dist));
                        }
                    }
                } else {
                    for pos in node.index..end {
                        let b = self.entries[pos];
                        let dist = query.distance_squared_to(b);
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
        query: NeighborQuery2D,
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
                let b = self.entries[pos];
                let dist = query.distance_squared_to(b);
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
        query: NeighborQuery2D,
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
                let b = self.entries[pos];
                let dist = query.distance_squared_to(b);
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

    /// Same as [`visit`](Index2D::visit), but the traversal stack is reused by the caller.
    #[doc(hidden)]
    pub fn visit_with_stack<B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_with_stack_impl::<false, B, F>(query, stack, visitor)
    }

    /// Hidden prefetch variant of [`visit_with_stack`](Index2D::visit_with_stack).
    #[doc(hidden)]
    pub fn visit_with_stack_prefetch<B, F>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_with_stack_impl::<true, B, F>(query, stack, visitor)
    }

    /// Hottest path: both result buffer and traversal stack are reused by the caller.
    #[doc(hidden)]
    pub fn search_into_stack(
        &self,
        query: Box2D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        self.search_into_stack_contained_impl(query, results, stack);
    }

    /// Traversal variant that prefetches the next node from the stack.
    #[doc(hidden)]
    pub fn search_into_stack_prefetch(
        &self,
        query: Box2D,
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

        self.search_into_stack_overlaps_impl::<true>(query, results, stack);
    }

    fn search_into_stack_overlaps_impl<const PREFETCH: bool>(
        &self,
        query: Box2D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
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
                if PREFETCH {
                    prefetch_aos_node(
                        &self.entries,
                        &self.indices,
                        stack[stack.len() - 2],
                        self.node_size,
                    );
                }
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }

    fn search_into_stack_contained_impl(
        &self,
        query: Box2D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        results.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }

        const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
        const LEVEL_MASK: usize = !CONTAINED_FLAG;

        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_entries = &self.entries[node_index..end];
            let node_indices = &self.indices[node_index..end];

            if contained {
                self.extend_contained_leaf_indices(node_index, end, level, results);
            } else if is_leaf {
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
                    let encoded_level = if query.contains(*b) {
                        child_level | CONTAINED_FLAG
                    } else {
                        child_level
                    };
                    stack.push(encoded_level);
                }
            }

            if stack.len() > 1 {
                prefetch_aos_node(
                    &self.entries,
                    &self.indices,
                    stack[stack.len() - 2],
                    self.node_size,
                );
                let encoded_level = stack.pop().unwrap();
                level = encoded_level & LEVEL_MASK;
                contained = (encoded_level & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }

    #[inline]
    fn extend_contained_leaf_indices(
        &self,
        node_index: usize,
        end: usize,
        level: usize,
        results: &mut Vec<usize>,
    ) {
        let start = self.leaf_start_for_entry(node_index, level);
        let end = if end < self.level_bounds[level] {
            self.leaf_start_for_entry(end, level)
        } else {
            self.num_items
        };
        results.extend_from_slice(&self.indices[start..end]);
    }

    #[inline]
    fn leaf_start_for_entry(&self, mut index: usize, mut level: usize) -> usize {
        while level > 0 {
            index = self.indices[index];
            level -= 1;
        }
        index
    }

    fn visit_with_stack_impl<const PREFETCH: bool, B, F>(
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
                if PREFETCH {
                    prefetch_aos_node(
                        &self.entries,
                        &self.indices,
                        stack[stack.len() - 2],
                        self.node_size,
                    );
                }
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return ControlFlow::Continue(());
            }
        }
    }

    /// Diagnostics: returns `(result_count, intersection_check_count)`.
    #[doc(hidden)]
    pub fn search_visited(&self, query: Box2D) -> (usize, usize) {
        let mut results = 0usize;
        let mut visited = 0usize;
        if self.num_items == 0 {
            return (0, 0);
        }

        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                visited += 1;
                let b = &self.entries[pos];
                if !b.overlaps(query) {
                    continue;
                }
                let index = self.indices[pos];
                if is_leaf {
                    results += 1;
                } else {
                    stack.push(index);
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

/// Zero-copy read-only view over bytes produced by [`Index2D::to_bytes`].
///
/// Loading validates the buffer but does not copy the tree into owned vectors.
/// Search and nearest-neighbor methods read little-endian values directly from
/// the borrowed byte slice.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, Index2DView, Box2D};
///
/// let mut builder = Index2DBuilder::new(1);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// let bytes = builder.finish().unwrap().to_bytes();
///
/// let view = Index2DView::from_bytes(&bytes).unwrap();
/// assert_eq!(view.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
/// ```
pub struct Index2DView<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    /// Derived at load (not stored), so owned rather than borrowed.
    level_bounds: Vec<usize>,
    entries: &'a [u8],
    indices: &'a [u8],
    payload: Option<ParsedPayload<'a>>,
    /// `insertion id -> leaf rank`, built when a (leaf-ordered) payload is
    /// present, to serve random `payload(id)` lookups.
    id_to_leaf: Option<Vec<u32>>,
}

impl<'a> Index2DView<'a> {
    /// Load a zero-copy index view from bytes previously produced by [`Index2D::to_bytes`].
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Index2DView, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// let bytes = builder.finish()?.to_bytes();
    ///
    /// let view = Index2DView::from_bytes(&bytes)?;
    /// assert_eq!(view.num_items(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 2, 8)?;
        // The payload is leaf-ordered; build the id -> leaf-rank map so random
        // `payload(id)` lookups work. Only allocated when a payload is present.
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

    /// Borrow item `id`'s payload blob (zero-copy), or `None` if the bytes have
    /// no payload section or `id` is out of range.
    pub fn payload(&self, id: usize) -> Option<&'a [u8]> {
        let payload = self.payload.as_ref()?;
        let id_to_leaf = self.id_to_leaf.as_ref()?;
        let leaf_rank = *id_to_leaf.get(id)? as usize;
        Some(payload_slice(payload, leaf_rank))
    }

    /// Borrow every triangle record as a zero-copy `&[T]` (with `T` =
    /// [`Triangle2D`](crate::Triangle2D) / [`Triangle2DF32`](crate::Triangle2DF32)),
    /// in leaf (storage) order, when the payload is a fixed-width section of that
    /// record type and the underlying bytes are aligned (an mmap or an aligned
    /// buffer). Returns `None` otherwise; [`triangle`](Self::triangle) reads one
    /// record by item id regardless of alignment.
    pub fn triangles<T: Triangle2>(&self) -> Option<&'a [T]> {
        let payload = self.payload.as_ref()?;
        if payload.stride != T::STRIDE {
            return None;
        }
        blobs_as_records::<T>(payload.blobs)
    }

    /// The triangle stored for item `id`, by value (works at any alignment).
    /// `None` if there is no triangle payload of the requested type, or `id` is
    /// out of range. The type parameter chooses the record format
    /// ([`Triangle2D`](crate::Triangle2D) for `f64`,
    /// [`Triangle2DF32`](crate::Triangle2DF32) for `f32`).
    pub fn triangle<T: Triangle2>(&self, id: usize) -> Option<T> {
        let payload = self.payload.as_ref()?;
        if payload.stride != T::STRIDE {
            return None;
        }
        let id_to_leaf = self.id_to_leaf.as_ref()?;
        let leaf_rank = *id_to_leaf.get(id)? as usize;
        Some(T::read_le(payload_slice(payload, leaf_rank)))
    }

    /// Return `(item index, payload blob)` for every item intersecting `query`.
    ///
    /// The blobs are borrowed zero-copy. This is the local/in-memory counterpart
    /// of the streaming `search_payloads`; both pair query results with their
    /// stored data. Returns an empty vec if the view has no payload section.
    pub fn search_payloads(&self, query: Box2D) -> Vec<(usize, &'a [u8])> {
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
    pub fn extent(&self) -> Option<Box2D> {
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
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(query, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, query: Box2D, results: &mut Vec<usize>) {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(query, results, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'b>(&self, query: Box2D, workspace: &'b mut SearchWorkspace) -> &'b [usize] {
        self.search_into_stack(query, &mut workspace.results, &mut workspace.stack);
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
            if let Some(index) =
                self.nearest_one_with_queue(NeighborQuery2D::Point(point), max_distance, &mut queue)
            {
                results.push(index);
            }
            return;
        }

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery2D::Point(point),
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

        self.collect_neighbors_with_queues(
            NeighborQuery2D::Point(point),
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
    ///
    /// Distance is the box-to-box gap: items overlapping or touching `query`
    /// have distance `0.0` and come first (their mutual order is unspecified).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(10.0, 0.0, 11.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// // The query box's nearest edge is closer to item 1 than to item 0.
    /// let query = Box2D::new(7.0, 0.0, 8.0, 1.0);
    /// assert_eq!(index.neighbors_of_box(query, 1), vec![1]);
    /// ```
    pub fn neighbors_of_box(&self, query: Box2D, max_results: usize) -> Vec<usize> {
        self.neighbors_of_box_within(query, max_results, f64::INFINITY)
    }

    /// Return up to `max_results` item indices within `max_distance` of the
    /// box `query`. See [`neighbors_of_box`](Self::neighbors_of_box).
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            NeighborQuery2D::Box(query),
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

        self.collect_neighbors_with_queues(
            NeighborQuery2D::Box(query),
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
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(query, &mut stack, visitor)
    }

    /// Return every pair `(i, j)` where item `i` of `self` intersects item `j`
    /// of `other`. See [`Index2D::join`].
    pub fn join(&self, other: &Index2DView<'_>) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.join_with(other, |i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every intersecting pair between `self` and `other`. See
    /// [`Index2D::join_with`].
    pub fn join_with<B, F>(&self, other: &Index2DView<'_>, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        join_core(self, other, visitor)
    }

    /// Return every unordered pair of distinct intersecting items within this
    /// view, each pair exactly once. See [`Index2D::self_join`].
    pub fn self_join(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let _: ControlFlow<()> = self.self_join_with(|i, j| {
            out.push((i, j));
            ControlFlow::Continue(())
        });
        out
    }

    /// Visit every unordered pair of distinct intersecting items within this
    /// view. See [`Index2D::self_join_with`].
    pub fn self_join_with<B, F>(&self, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, usize) -> ControlFlow<B>,
    {
        self_join_core(self, visitor)
    }

    fn collect_neighbors_with_queues(
        &self,
        query: NeighborQuery2D,
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
        query: NeighborQuery2D,
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
        query: Box2D,
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
                    let index = self.index_at_unchecked(pos);
                    results.push(index);
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.entry_at_unchecked(pos);
                    if !b.overlaps(query) {
                        continue;
                    }
                    let index = self.index_at_unchecked(pos);
                    stack.push(index);
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
                    let index = self.index_at_unchecked(pos);
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.entry_at_unchecked(pos);
                    if !b.overlaps(query) {
                        continue;
                    }
                    let index = self.index_at_unchecked(pos);
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
    fn entry_at_unchecked(&self, index: usize) -> Box2D {
        let offset = index * 32;
        Box2D::new(
            read_f64_le_unchecked(self.entries, offset),
            read_f64_le_unchecked(self.entries, offset + 8),
            read_f64_le_unchecked(self.entries, offset + 16),
            read_f64_le_unchecked(self.entries, offset + 24),
        )
    }

    #[inline]
    fn index_at_unchecked(&self, index: usize) -> usize {
        read_u64_le_unchecked(self.indices, index * 8) as usize
    }
}

impl JoinTree for Index2D {
    type Bounds = Box2D;

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
    fn join_bounds(&self, pos: usize) -> Box2D {
        self.entries[pos]
    }
    #[inline]
    fn join_index(&self, pos: usize) -> usize {
        self.indices[pos]
    }
    #[inline]
    fn bounds_overlap(a: Box2D, b: Box2D) -> bool {
        a.overlaps(b)
    }
    #[inline]
    fn bounds_contain(outer: Box2D, inner: Box2D) -> bool {
        outer.contains(inner)
    }
}

impl JoinTree for Index2DView<'_> {
    type Bounds = Box2D;

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
    fn join_bounds(&self, pos: usize) -> Box2D {
        self.entry_at_unchecked(pos)
    }
    #[inline]
    fn join_index(&self, pos: usize) -> usize {
        self.index_at_unchecked(pos)
    }
    #[inline]
    fn bounds_overlap(a: Box2D, b: Box2D) -> bool {
        a.overlaps(b)
    }
    #[inline]
    fn bounds_contain(outer: Box2D, inner: Box2D) -> bool {
        outer.contains(inner)
    }
}

impl Index2D {
    /// Return the indices of all items whose boxes the ray segment touches.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder, Point2D, Ray2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let ray = Ray2D::new(Point2D::new(-1.0, 0.5), 1.0, 0.0, 10.0);
    /// assert_eq!(index.raycast(ray), vec![0]);
    /// assert_eq!(index.raycast_closest(ray), Some((0, 1.0)));
    /// ```
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
    pub fn raycast_with<'a>(&self, ray: Ray2D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
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
    /// direction length (see [`Ray2D::new`]).
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

impl Index2D {
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

impl Index2DView<'_> {
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
/// [`Index2D::search_iter`].
///
/// Yields original insertion indices in tree-traversal order, descending the
/// tree only as far as the consumer pulls. Holds a small traversal stack
/// (`O(depth)`); it allocates no result `Vec`.
pub struct Search2DIter<'a> {
    index: &'a Index2D,
    query: Box2D,
    // (node_index, level) pairs still to visit, same encoding as the search stack.
    stack: Vec<usize>,
    // Half-open entry range of the leaf node currently being scanned.
    leaf_pos: usize,
    leaf_end: usize,
}

impl<'a> Search2DIter<'a> {
    fn new(index: &'a Index2D, query: Box2D) -> Self {
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

impl Iterator for Search2DIter<'_> {
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

impl std::iter::FusedIterator for Search2DIter<'_> {}

/// Builder for [`Index2D`] serialization, created by [`Index2D::serialize`].
///
/// Set optional per-item payloads, the streaming-tuned interleaved layout, and
/// descriptive metadata (CRS / content type / attribution), then call
/// [`to_bytes`](Self::to_bytes) or [`to_bytes_into`](Self::to_bytes_into). The
/// metadata strings are stored opaquely and read back with
/// [`read_metadata`](crate::read_metadata).
pub struct Serializer2D<'a> {
    index: &'a Index2D,
    interleaved: bool,
    payloads: Option<Vec<&'a [u8]>>,
    /// `Some(stride)` selects the fixed-width (table-less) payload layout.
    record_stride: Option<u32>,
    meta: MetaFields<'a>,
}

impl<'a> Serializer2D<'a> {
    fn new(index: &'a Index2D) -> Self {
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
    /// [`Triangle2D`](crate::Triangle2D) for `f64` or
    /// [`Triangle2DF32`](crate::Triangle2DF32) for `f32`): one triangle per item,
    /// in item order. A convenience over [`records`](Self::records); pair it with
    /// [`Index2D::from_triangles`](crate::Index2D::from_triangles).
    pub fn triangles<T: Triangle2>(self, triangles: &'a [T]) -> Self {
        let bytes = records_as_bytes(triangles);
        self.records(T::STRIDE, bytes)
    }

    /// Use the streaming-tuned interleaved node layout (see
    /// [`Index2D::to_bytes_interleaved`]).
    #[cfg(feature = "stream")]
    pub fn interleaved(mut self) -> Self {
        self.interleaved = true;
        self
    }

    /// Set the coordinate reference system identifier (opaque, e.g. `"EPSG:4326"`).
    pub fn crs(mut self, crs: &'a str) -> Self {
        self.meta.crs = Some(crs);
        self
    }

    /// Set the payload content type / media type (e.g. `"application/geo+json"`).
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
            2,
            8,
            interleaved,
            idx.num_items,
            idx.entries.len(),
            idx.node_size,
            |bytes| {
                #[cfg(feature = "stream")]
                if interleaved {
                    bytes.write_interleaved_2d(&idx.entries, &idx.indices);
                    return;
                }
                bytes.write_box2d_slice(&idx.entries);
                bytes.write_usize_slice_as_u64(&idx.indices);
            },
            self.payloads.as_deref(),
            record_stride,
            &idx.indices[..idx.num_items],
            &self.meta,
        )
    }
}
