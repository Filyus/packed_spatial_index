//! Static spatial index implementation for 2D AABBs:
//! a packed Hilbert R-tree in the style of flatbush / `static_aabb2d_index`.
//!
//! The public API is intentionally small: collect bounds with [`crate::Index2DBuilder`],
//! call [`crate::Index2DBuilder::finish`], then search the finished [`Index2D`].
//!
//! # Example
//! ```
//! use packed_spatial_index::{Index2DBuilder, Bounds2D};
//!
//! let mut builder = Index2DBuilder::new(3);
//! builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
//! builder.add(Bounds2D::new(5.0, 5.0, 6.0, 6.0));
//! builder.add(Bounds2D::new(0.5, 0.5, 2.0, 2.0));
//! let index = builder.finish().unwrap();
//!
//! let hits = index.search(Bounds2D::new(0.0, 0.0, 1.5, 1.5));
//! assert!(hits.contains(&0) && hits.contains(&2));
//! assert!(!hits.contains(&1));
//! ```

use std::{collections::BinaryHeap, ops::ControlFlow};

use crate::config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY};
use crate::geometry::{Bounds2D, Point2D};
use crate::neighbors::{NeighborNodeState, NeighborState, NeighborWorkspace, max_distance_squared};
use crate::persistence::{
    ByteWriter, LoadError, parse_index_bytes, read_f64_le_unchecked, read_u64_le_unchecked,
    serialized_len,
};
use crate::traversal::{SearchWorkspace, prefetch_read, upper_bound_level};

#[inline]
fn prefetch_aos_node(boxes: &[Bounds2D], indices: &[usize], node_index: usize, node_size: usize) {
    if node_index < boxes.len() {
        prefetch_read(boxes.as_ptr().wrapping_add(node_index));
        prefetch_read(indices.as_ptr().wrapping_add(node_index));
    }
    let next_line = node_index.saturating_add((64 / std::mem::size_of::<Bounds2D>()).max(1));
    if node_size > 1 && next_line < boxes.len() {
        prefetch_read(boxes.as_ptr().wrapping_add(next_line));
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
/// use packed_spatial_index::{Index2DBuilder, Bounds2D};
///
/// let mut builder = Index2DBuilder::new(2);
/// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
/// builder.add(Bounds2D::new(5.0, 5.0, 6.0, 6.0));
/// let index = builder.finish().unwrap();
///
/// assert_eq!(index.num_items(), 2);
/// assert_eq!(index.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
/// ```
pub struct Index2D {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) boxes: Vec<Bounds2D>,
    pub(crate) indices: Vec<usize>,
}

impl Index2D {
    /// Return the number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total extent of indexed items, or `None` for an empty index.
    pub fn extent(&self) -> Option<Bounds2D> {
        self.boxes.last().copied()
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
    /// use packed_spatial_index::{Index2D, Index2DBuilder, Index2DView, Bounds2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
    /// let index = builder.finish()?;
    ///
    /// let bytes = index.to_bytes();
    /// let owned = Index2D::from_bytes(&bytes)?;
    /// let view = Index2DView::from_bytes(&bytes)?;
    ///
    /// let query = Bounds2D::new(0.5, 0.5, 0.5, 0.5);
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
    /// use packed_spatial_index::{Index2D, Index2DBuilder, Bounds2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
    /// let index = builder.finish()?;
    ///
    /// let mut buffer = Vec::new();
    /// index.to_bytes_into(&mut buffer);
    /// assert_eq!(buffer, index.to_bytes());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        let level_count = self.level_bounds.len();
        let num_nodes = self.boxes.len();
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
        bytes.write_bounds2d_slice(&self.boxes);
        bytes.write_usize_slice_as_u64(&self.indices);
        bytes.finish();
    }

    /// Load an owned index from bytes previously produced by [`Index2D::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let view = Index2DView::from_bytes(bytes)?;

        let mut level_bounds = Vec::with_capacity(view.level_count);
        for i in 0..view.level_count {
            level_bounds.push(view.level_bound_unchecked(i));
        }

        let mut boxes = Vec::with_capacity(view.num_nodes);
        for i in 0..view.num_nodes {
            boxes.push(view.box_at_unchecked(i));
        }

        let mut indices = Vec::with_capacity(view.num_nodes);
        for i in 0..view.num_nodes {
            indices.push(view.index_at_unchecked(i));
        }

        Ok(Self {
            node_size: view.node_size,
            num_items: view.num_items,
            level_bounds,
            boxes,
            indices,
        })
    }

    /// Return the indices of all items whose bounds intersect `bounds`.
    pub fn search(&self, bounds: Bounds2D) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(bounds, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, bounds: Bounds2D, results: &mut Vec<usize>) {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(bounds, results, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Bounds2D, SearchWorkspace};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let mut workspace = SearchWorkspace::with_capacity(8, 8);
    /// let hits = index.search_with(Bounds2D::new(0.0, 0.0, 2.0, 2.0), &mut workspace);
    /// assert_eq!(hits, &[0]);
    /// assert_eq!(workspace.results(), &[0]);
    /// ```
    pub fn search_with<'a>(
        &self,
        bounds: Bounds2D,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize] {
        self.search_into_stack(bounds, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `bounds`.
    ///
    /// This is an early-exit path: traversal stops at the first hit and does not
    /// allocate a result `Vec`.
    pub fn any(&self, bounds: Bounds2D) -> bool {
        self.visit(bounds, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    ///
    /// Tree traversal order is not part of the API, so this returns just some first
    /// found item, not the minimum insertion index.
    pub fn first(&self, bounds: Bounds2D) -> Option<usize> {
        match self.visit(bounds, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Return up to `max_results` item indices nearest to `point`.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Point2D, Bounds2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Bounds2D::new(10.0, 10.0, 11.0, 11.0));
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
            if let Some(index) = self.nearest_one_with_queue(point, max_distance, &mut queue) {
                results.push(index);
            }
            return;
        }

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            point,
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
            if let Some(index) =
                self.nearest_one_with_queue(point, max_distance, &mut workspace.node_queue)
            {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        self.collect_neighbors_with_queues(
            point,
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
        self.visit_neighbors_with_queue(point, max_distance, &mut queue, &mut visitor)
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
    /// use packed_spatial_index::{Index2DBuilder, Bounds2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Bounds2D::new(5.0, 5.0, 6.0, 6.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let found = index.visit(Bounds2D::new(5.0, 5.0, 6.0, 6.0), ControlFlow::Break);
    /// assert_eq!(found, ControlFlow::Break(1));
    /// ```
    pub fn visit<B, F>(&self, bounds: Bounds2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(bounds, &mut stack, visitor)
    }

    fn collect_neighbors_with_queues(
        &self,
        point: Point2D,
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

        let root_index = self.boxes.len() - 1;
        let root_dist = self.boxes[root_index].distance_squared_to(point);
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
                        let b = self.boxes[pos];
                        let dist = b.distance_squared_to(point);
                        if dist <= max_dist_sq {
                            item_queue.push(NeighborState::new(self.indices[pos], true, dist));
                        }
                    }
                } else {
                    for pos in node.index..end {
                        let b = self.boxes[pos];
                        let dist = b.distance_squared_to(point);
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
        let mut node_index = self.boxes.len() - 1;
        loop {
            let upper_bound_level = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let b = self.boxes[pos];
                let dist = b.distance_squared_to(point);
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

        let mut node_index = self.boxes.len() - 1;
        loop {
            let upper_bound_level = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let b = self.boxes[pos];
                let dist = b.distance_squared_to(point);
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
        bounds: Bounds2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_with_stack_impl::<false, B, F>(bounds, stack, visitor)
    }

    /// Experimental prefetch variant of [`visit_with_stack`](Index2D::visit_with_stack).
    #[doc(hidden)]
    pub fn visit_with_stack_prefetch<B, F>(
        &self,
        bounds: Bounds2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_with_stack_impl::<true, B, F>(bounds, stack, visitor)
    }

    /// Hottest path: both result buffer and traversal stack are reused by the caller.
    #[doc(hidden)]
    pub fn search_into_stack(
        &self,
        bounds: Bounds2D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        self.search_into_stack_impl::<false>(bounds, results, stack);
    }

    /// Traversal variant that prefetches the next node from the stack.
    #[doc(hidden)]
    pub fn search_into_stack_prefetch(
        &self,
        bounds: Bounds2D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        self.search_into_stack_impl::<true>(bounds, results, stack);
    }

    fn search_into_stack_impl<const PREFETCH: bool>(
        &self,
        bounds: Bounds2D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        results.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }

        let mut node_index = self.boxes.len() - 1;
        let mut level = self.level_bounds.len() - 1;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_boxes = &self.boxes[node_index..end];
            let node_indices = &self.indices[node_index..end];

            if is_leaf {
                for (b, &index) in node_boxes.iter().zip(node_indices) {
                    if !b.overlaps(bounds) {
                        continue;
                    }
                    results.push(index);
                }
            } else {
                let child_level = level - 1;
                for (b, &index) in node_boxes.iter().zip(node_indices).rev() {
                    if !b.overlaps(bounds) {
                        continue;
                    }
                    stack.push(index);
                    stack.push(child_level);
                }
            }

            if stack.len() > 1 {
                if PREFETCH {
                    prefetch_aos_node(
                        &self.boxes,
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

    fn visit_with_stack_impl<const PREFETCH: bool, B, F>(
        &self,
        bounds: Bounds2D,
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

        let mut node_index = self.boxes.len() - 1;
        let mut level = self.level_bounds.len() - 1;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let node_boxes = &self.boxes[node_index..end];
            let node_indices = &self.indices[node_index..end];

            if is_leaf {
                for (b, &index) in node_boxes.iter().zip(node_indices) {
                    if !b.overlaps(bounds) {
                        continue;
                    }
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for (b, &index) in node_boxes.iter().zip(node_indices).rev() {
                    if !b.overlaps(bounds) {
                        continue;
                    }
                    stack.push(index);
                    stack.push(child_level);
                }
            }

            if stack.len() > 1 {
                if PREFETCH {
                    prefetch_aos_node(
                        &self.boxes,
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
    pub fn search_visited(&self, bounds: Bounds2D) -> (usize, usize) {
        let mut results = 0usize;
        let mut visited = 0usize;
        if self.num_items == 0 {
            return (0, 0);
        }

        let mut node_index = self.boxes.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                visited += 1;
                let b = &self.boxes[pos];
                if !b.overlaps(bounds) {
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
/// use packed_spatial_index::{Index2DBuilder, Index2DView, Bounds2D};
///
/// let mut builder = Index2DBuilder::new(1);
/// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
/// let bytes = builder.finish().unwrap().to_bytes();
///
/// let view = Index2DView::from_bytes(&bytes).unwrap();
/// assert_eq!(view.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
/// ```
pub struct Index2DView<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    level_bounds: &'a [u8],
    boxes: &'a [u8],
    indices: &'a [u8],
}

impl<'a> Index2DView<'a> {
    /// Load a zero-copy index view from bytes previously produced by [`Index2D::to_bytes`].
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Index2DView, Bounds2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
    /// let bytes = builder.finish()?.to_bytes();
    ///
    /// let view = Index2DView::from_bytes(&bytes)?;
    /// assert_eq!(view.num_items(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let parsed = parse_index_bytes(bytes)?;
        Ok(Self {
            node_size: parsed.node_size,
            num_items: parsed.num_items,
            num_nodes: parsed.num_nodes,
            level_count: parsed.level_count,
            level_bounds: parsed.level_bounds,
            boxes: parsed.boxes,
            indices: parsed.indices,
        })
    }

    /// Return the number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the total extent of indexed items, or `None` for an empty view.
    pub fn extent(&self) -> Option<Bounds2D> {
        if self.num_items == 0 {
            None
        } else {
            Some(self.box_at_unchecked(self.num_nodes - 1))
        }
    }

    /// Return the packed node size.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Return the indices of all items whose bounds intersect `bounds`.
    pub fn search(&self, bounds: Bounds2D) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(bounds, &mut results);
        results
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, bounds: Bounds2D, results: &mut Vec<usize>) {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(bounds, results, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'b>(
        &self,
        bounds: Bounds2D,
        workspace: &'b mut SearchWorkspace,
    ) -> &'b [usize] {
        self.search_into_stack(bounds, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `bounds`.
    pub fn any(&self, bounds: Bounds2D) -> bool {
        self.visit(bounds, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    pub fn first(&self, bounds: Bounds2D) -> Option<usize> {
        match self.visit(bounds, ControlFlow::Break) {
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

        let mut item_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut node_queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queues(
            point,
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
            if let Some(index) =
                self.nearest_one_with_queue(point, max_distance, &mut workspace.node_queue)
            {
                workspace.results.push(index);
            }
            return &workspace.results;
        }

        self.collect_neighbors_with_queues(
            point,
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
        self.visit_neighbors_with_queue(point, max_distance, &mut queue, &mut visitor)
    }

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, bounds: Bounds2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(bounds, &mut stack, visitor)
    }

    fn collect_neighbors_with_queues(
        &self,
        point: Point2D,
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
        let root_dist = self.box_at_unchecked(root_index).distance_squared_to(point);
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
                        let b = self.box_at_unchecked(pos);
                        let dist = b.distance_squared_to(point);
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
                        let b = self.box_at_unchecked(pos);
                        let dist = b.distance_squared_to(point);
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
            let upper_bound_level = self.upper_bound_level(node_index);
            let end =
                (node_index + self.node_size).min(self.level_bound_unchecked(upper_bound_level));
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let b = self.box_at_unchecked(pos);
                let dist = b.distance_squared_to(point);
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
        bounds: Bounds2D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
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

            if is_leaf {
                for pos in node_index..end {
                    let b = self.box_at_unchecked(pos);
                    if !b.overlaps(bounds) {
                        continue;
                    }
                    let index = self.index_at_unchecked(pos);
                    results.push(index);
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.box_at_unchecked(pos);
                    if !b.overlaps(bounds) {
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
        bounds: Bounds2D,
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
                    let b = self.box_at_unchecked(pos);
                    if !b.overlaps(bounds) {
                        continue;
                    }
                    let index = self.index_at_unchecked(pos);
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.box_at_unchecked(pos);
                    if !b.overlaps(bounds) {
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
            let upper_bound_level = self.upper_bound_level(node_index);
            let end =
                (node_index + self.node_size).min(self.level_bound_unchecked(upper_bound_level));
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                let b = self.box_at_unchecked(pos);
                let dist = b.distance_squared_to(point);
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
        read_u64_le_unchecked(self.level_bounds, index * 8) as usize
    }

    #[inline]
    fn box_at_unchecked(&self, index: usize) -> Bounds2D {
        let offset = index * 32;
        Bounds2D::new(
            read_f64_le_unchecked(self.boxes, offset),
            read_f64_le_unchecked(self.boxes, offset + 8),
            read_f64_le_unchecked(self.boxes, offset + 16),
            read_f64_le_unchecked(self.boxes, offset + 24),
        )
    }

    #[inline]
    fn index_at_unchecked(&self, index: usize) -> usize {
        read_u64_le_unchecked(self.indices, index * 8) as usize
    }
}
