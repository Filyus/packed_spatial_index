//! Static spatial index implementation for 2D AABBs:
//! a packed Hilbert R-tree in the style of flatbush / `static_aabb2d_index`.
//!
//! The public API is intentionally small: collect rectangles with [`crate::IndexBuilder`],
//! call [`crate::IndexBuilder::finish`], then search the finished [`Index`].
//!
//! # Example
//! ```
//! use packed_spatial_index::{IndexBuilder, Rect};
//!
//! let mut builder = IndexBuilder::new(3);
//! builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
//! builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));
//! builder.add(Rect::new(0.5, 0.5, 2.0, 2.0));
//! let index = builder.finish().unwrap();
//!
//! let hits = index.search(Rect::new(0.0, 0.0, 1.5, 1.5));
//! assert!(hits.contains(&0) && hits.contains(&2));
//! assert!(!hits.contains(&1));
//! ```

use std::{collections::BinaryHeap, ops::ControlFlow};

use crate::config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY};
use crate::geometry::{Num, Point, Rect};
use crate::neighbors::{NeighborNodeState, NeighborState, NeighborWorkspace, max_distance_squared};
use crate::persistence::{
    LoadError, parse_index_bytes, push_f64, push_flags, push_format_version, push_header_len,
    push_magic, push_u64, read_f64_le_unchecked, read_u64_le_unchecked, serialized_len,
};

/// Reusable buffers for allocation-free repeated searches.
#[derive(Debug, Default)]
pub struct SearchWorkspace {
    pub(crate) results: Vec<usize>,
    pub(crate) stack: Vec<usize>,
}

impl SearchWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a workspace with preallocated result and traversal-stack capacity.
    pub fn with_capacity(results: usize, stack: usize) -> Self {
        Self {
            results: Vec::with_capacity(results),
            stack: Vec::with_capacity(stack),
        }
    }

    /// Results from the latest `search_with` call.
    pub fn results(&self) -> &[usize] {
        &self.results
    }
}

#[inline]
pub(crate) fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }

    #[cfg(target_arch = "x86")]
    unsafe {
        use std::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

#[inline]
fn prefetch_aos_node(boxes: &[Rect], indices: &[usize], node_index: usize, node_size: usize) {
    if node_index < boxes.len() {
        prefetch_read(boxes.as_ptr().wrapping_add(node_index));
        prefetch_read(indices.as_ptr().wrapping_add(node_index));
    }
    let next_line = node_index.saturating_add((64 / std::mem::size_of::<Rect>()).max(1));
    if node_size > 1 && next_line < boxes.len() {
        prefetch_read(boxes.as_ptr().wrapping_add(next_line));
        prefetch_read(indices.as_ptr().wrapping_add(next_line));
    }
}

/// Finished static read-only index.
///
/// Search methods return item positions in the original insertion order. The order
/// of returned results is traversal order and is not part of the API.
pub struct Index {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) boxes: Vec<Rect>,
    pub(crate) indices: Vec<usize>,
}

impl Index {
    /// Return the number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Return the packed node size.
    #[doc(hidden)]
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Serialize this index into the stable little-endian `PSINDEX` format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let level_count = self.level_bounds.len();
        let num_nodes = self.boxes.len();
        let len = serialized_len(level_count, num_nodes).expect("serialized index is too large");
        let mut bytes = Vec::with_capacity(len);
        push_magic(&mut bytes);
        push_format_version(&mut bytes);
        push_header_len(&mut bytes);
        push_flags(&mut bytes);
        push_u64(&mut bytes, self.node_size as u64);
        push_u64(&mut bytes, self.num_items as u64);
        push_u64(&mut bytes, num_nodes as u64);
        push_u64(&mut bytes, level_count as u64);
        for &bound in &self.level_bounds {
            push_u64(&mut bytes, bound as u64);
        }
        for b in &self.boxes {
            push_f64(&mut bytes, b.min_x);
            push_f64(&mut bytes, b.min_y);
            push_f64(&mut bytes, b.max_x);
            push_f64(&mut bytes, b.max_y);
        }
        for &index in &self.indices {
            push_u64(&mut bytes, index as u64);
        }
        bytes
    }

    /// Load an owned index from bytes previously produced by [`Index::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let view = IndexView::from_bytes(bytes)?;

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

    /// Return the indices of all items whose rectangles intersect `rect`.
    pub fn search(&self, rect: Rect) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(rect, &mut results);
        results
    }

    /// Return the indices of all items intersecting raw bounds.
    pub fn search_bounds(&self, min_x: Num, min_y: Num, max_x: Num, max_y: Num) -> Vec<usize> {
        self.search(Rect::new(min_x, min_y, max_x, max_y))
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, rect: Rect, results: &mut Vec<usize>) {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(rect, results, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'a>(&self, rect: Rect, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.search_into_stack(rect, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one item intersects `rect`.
    ///
    /// This is an early-exit path: traversal stops at the first hit and does not
    /// allocate a result `Vec`.
    pub fn any(&self, rect: Rect) -> bool {
        self.visit(rect, |_| ControlFlow::Break(())).is_break()
    }

    /// Return one intersecting item, if any.
    ///
    /// Tree traversal order is not part of the API, so this returns just some first
    /// found item, not the minimum insertion index.
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
    ///
    /// The visitor receives item positions in the original insertion order. Return
    /// [`ControlFlow::Continue`] to continue traversal or [`ControlFlow::Break`] for
    /// early exit with a user-provided value.
    pub fn visit<B, F>(&self, rect: Rect, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(rect, &mut stack, visitor)
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

    /// Same as [`visit`](Index::visit), but the traversal stack is reused by the caller.
    #[doc(hidden)]
    pub fn visit_with_stack<B, F>(
        &self,
        rect: Rect,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_with_stack_impl::<false, B, F>(rect, stack, visitor)
    }

    /// Experimental prefetch variant of [`visit_with_stack`](Index::visit_with_stack).
    #[doc(hidden)]
    pub fn visit_with_stack_prefetch<B, F>(
        &self,
        rect: Rect,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        self.visit_with_stack_impl::<true, B, F>(rect, stack, visitor)
    }

    /// Hottest path: both result buffer and traversal stack are reused by the caller.
    #[doc(hidden)]
    pub fn search_into_stack(&self, rect: Rect, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        self.search_into_stack_impl::<false>(rect, results, stack);
    }

    /// Traversal variant that prefetches the next node from the stack.
    #[doc(hidden)]
    pub fn search_into_stack_prefetch(
        &self,
        rect: Rect,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        self.search_into_stack_impl::<true>(rect, results, stack);
    }

    fn search_into_stack_impl<const PREFETCH: bool>(
        &self,
        rect: Rect,
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

            if is_leaf {
                for pos in node_index..end {
                    // SAFETY: pos < end <= level_bounds[level] <= boxes.len() == indices.len().
                    let b = unsafe { self.boxes.get_unchecked(pos) };
                    if !b.overlaps(rect) {
                        continue;
                    }
                    let index = unsafe { *self.indices.get_unchecked(pos) };
                    results.push(index);
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    // SAFETY: pos < end <= level_bounds[level] <= boxes.len() == indices.len().
                    let b = unsafe { self.boxes.get_unchecked(pos) };
                    if !b.overlaps(rect) {
                        continue;
                    }
                    let index = unsafe { *self.indices.get_unchecked(pos) };
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

        let mut node_index = self.boxes.len() - 1;
        let mut level = self.level_bounds.len() - 1;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            if is_leaf {
                for pos in node_index..end {
                    // SAFETY: pos < end <= level_bounds[level] <= boxes.len() == indices.len().
                    let b = unsafe { self.boxes.get_unchecked(pos) };
                    if !b.overlaps(rect) {
                        continue;
                    }
                    let index = unsafe { *self.indices.get_unchecked(pos) };
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    // SAFETY: pos < end <= level_bounds[level] <= boxes.len() == indices.len().
                    let b = unsafe { self.boxes.get_unchecked(pos) };
                    if !b.overlaps(rect) {
                        continue;
                    }
                    let index = unsafe { *self.indices.get_unchecked(pos) };
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
    pub fn search_visited(&self, rect: Rect) -> (usize, usize) {
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
                if !b.overlaps(rect) {
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

/// Zero-copy read-only view over bytes produced by [`Index::to_bytes`].
pub struct IndexView<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    level_bounds: &'a [u8],
    boxes: &'a [u8],
    indices: &'a [u8],
}

impl<'a> IndexView<'a> {
    /// Load a zero-copy index view from bytes previously produced by [`Index::to_bytes`].
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

    /// Return the packed node size.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Return the indices of all items whose rectangles intersect `rect`.
    pub fn search(&self, rect: Rect) -> Vec<usize> {
        let mut results = Vec::new();
        self.search_into(rect, &mut results);
        results
    }

    /// Return the indices of all items intersecting raw bounds.
    pub fn search_bounds(&self, min_x: Num, min_y: Num, max_x: Num, max_y: Num) -> Vec<usize> {
        self.search(Rect::new(min_x, min_y, max_x, max_y))
    }

    /// Search with a reusable result buffer.
    pub fn search_into(&self, rect: Rect, results: &mut Vec<usize>) {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(rect, results, &mut stack);
    }

    /// Search with reusable result and traversal buffers.
    pub fn search_with<'b>(&self, rect: Rect, workspace: &'b mut SearchWorkspace) -> &'b [usize] {
        self.search_into_stack(rect, &mut workspace.results, &mut workspace.stack);
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
    pub fn neighbors_with<'b>(
        &self,
        point: Point,
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
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(rect, &mut stack, visitor)
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
    pub fn search_into_stack(&self, rect: Rect, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
                    if !b.overlaps(rect) {
                        continue;
                    }
                    let index = self.index_at_unchecked(pos);
                    results.push(index);
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.box_at_unchecked(pos);
                    if !b.overlaps(rect) {
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

        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;

            if is_leaf {
                for pos in node_index..end {
                    let b = self.box_at_unchecked(pos);
                    if !b.overlaps(rect) {
                        continue;
                    }
                    let index = self.index_at_unchecked(pos);
                    visitor(index)?;
                }
            } else {
                let child_level = level - 1;
                for pos in (node_index..end).rev() {
                    let b = self.box_at_unchecked(pos);
                    if !b.overlaps(rect) {
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
    fn box_at_unchecked(&self, index: usize) -> Rect {
        let offset = index * 32;
        Rect::new(
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

pub(crate) fn upper_bound_level(level_bounds: &[usize], node_index: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = level_bounds.len() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if level_bounds[mid] > node_index {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
