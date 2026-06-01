//! Static spatial index implementation for 2D AABBs:
//! a packed Hilbert R-tree in the style of flatbush / `static_aabb2d_index`.
//!
//! The public API is intentionally small: collect rectangles with [`IndexBuilder`],
//! call [`IndexBuilder::finish`], then search the finished [`Index`].
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

use std::{error::Error, fmt, ops::ControlFlow};

use crate::hilbert;

/// Index coordinates are `f64`, matching the reference default.
type Num = f64;

const DEFAULT_NODE_SIZE: usize = 16;

/// Minimum index size at which `parallel(true)` enables rayon.
#[cfg(feature = "parallel")]
pub(crate) const DEFAULT_PARALLEL_MIN_ITEMS: usize = 50_000;

/// Axis-aligned rectangle bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Minimum x coordinate.
    pub min_x: f64,
    /// Minimum y coordinate.
    pub min_y: f64,
    /// Maximum x coordinate.
    pub max_x: f64,
    /// Maximum y coordinate.
    pub max_y: f64,
}

impl Rect {
    /// Create a rectangle from `[min_x, min_y, max_x, max_y]` bounds.
    #[inline]
    pub const fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    #[inline]
    fn overlaps(&self, query: Rect) -> bool {
        // Branchless: compute all four comparisons and combine them with bitwise `&`
        // to remove hard-to-predict floating-point branches from the traversal loop.
        (self.min_x <= query.max_x)
            & (self.max_x >= query.min_x)
            & (self.min_y <= query.max_y)
            & (self.max_y >= query.min_y)
    }
}

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
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }

    #[cfg(target_arch = "x86")]
    unsafe {
        use std::arch::x86::{_mm_prefetch, _MM_HINT_T0};
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

/// Which key to use when sorting boxes before packing the tree.
///
/// [`SortKey::Hilbert`] is the default. [`SortKey::Morton`] is cheaper to compute, but
/// usually has slightly worse spatial locality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    /// Hilbert curve order.
    Hilbert,
    /// Morton curve (Z-order).
    Morton,
}

/// Experimental sort-key implementations used by benchmarks.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentalSortKey {
    /// Hilbert curve, "magic bits" (rawrunprotected): the reference crate algorithm.
    HilbertMagicBits,
    /// Hilbert curve, classic iterative algorithm with quadrant rotations.
    HilbertLoopRotation,
    /// Hilbert curve, table-driven finite-state machine.
    HilbertLut,
    /// Morton curve (Z-order).
    Morton,
}

impl From<SortKey> for ExperimentalSortKey {
    fn from(key: SortKey) -> Self {
        match key {
            SortKey::Hilbert => ExperimentalSortKey::HilbertMagicBits,
            SortKey::Morton => ExperimentalSortKey::Morton,
        }
    }
}

impl ExperimentalSortKey {
    /// Compute the sort key for normalized coordinates `x, y in [0, 65535]`.
    #[inline]
    pub(crate) fn encode(self, x: u16, y: u16) -> u32 {
        match self {
            ExperimentalSortKey::HilbertMagicBits => hilbert::magic_bits(x, y),
            ExperimentalSortKey::HilbertLoopRotation => hilbert::loop_rotation(x, y),
            ExperimentalSortKey::HilbertLut => hilbert::lut(x, y),
            ExperimentalSortKey::Morton => hilbert::morton(x, y),
        }
    }
}

/// Build error for finishing an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The builder received the wrong number of items.
    ItemCount {
        /// Number actually added through `add`.
        added: usize,
        /// Expected by `IndexBuilder::new(count)`.
        expected: usize,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::ItemCount { added, expected } => write!(
                f,
                "added item count must match declared count (added {added}, expected {expected})"
            ),
        }
    }
}

impl Error for BuildError {}

/// Builder for [`Index`] and, with the `simd` feature, `SimdIndex`.
pub struct IndexBuilder {
    node_size: usize,
    num_items: usize,
    sort_key: ExperimentalSortKey,
    radix: bool,
    #[cfg(feature = "parallel")]
    parallel: bool,
    #[cfg(feature = "parallel")]
    parallel_min_items: usize,
    boxes: Vec<Rect>,
}

impl IndexBuilder {
    /// Create a builder for exactly `count` items with the default node size (`16`).
    pub fn new(count: usize) -> Self {
        IndexBuilder {
            node_size: DEFAULT_NODE_SIZE,
            num_items: count,
            sort_key: SortKey::Hilbert.into(),
            radix: true,
            #[cfg(feature = "parallel")]
            parallel: false,
            #[cfg(feature = "parallel")]
            parallel_min_items: DEFAULT_PARALLEL_MIN_ITEMS,
            boxes: Vec::with_capacity(count.saturating_add(1)),
        }
    }

    /// Set the maximum number of children per tree node (clamped to `[2, 65535]`).
    pub fn node_size(mut self, node_size: usize) -> Self {
        self.node_size = node_size.clamp(2, 65535);
        self
    }

    /// Choose the sort key (default: [`SortKey::Hilbert`]).
    pub fn sort_key(mut self, key: SortKey) -> Self {
        self.sort_key = key.into();
        self
    }

    /// Choose an experimental sort-key implementation.
    #[doc(hidden)]
    pub fn experimental_sort_key(mut self, key: ExperimentalSortKey) -> Self {
        self.sort_key = key;
        self
    }

    /// Use LSD radix sort on the u32 Hilbert key instead of comparison-based sorting.
    #[doc(hidden)]
    pub fn radix(mut self, radix: bool) -> Self {
        self.radix = radix;
        self
    }

    /// Allow adaptive parallel builds through rayon.
    ///
    /// Default is `false`. When set to `true`, rayon is used only when the item
    /// count is at least the current `parallel_min_items` threshold.
    #[cfg(feature = "parallel")]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the minimum `count` at which [`parallel(true)`](Self::parallel)
    /// actually enables rayon.
    #[cfg(feature = "parallel")]
    pub fn parallel_min_items(mut self, min_items: usize) -> Self {
        self.parallel_min_items = min_items;
        self
    }

    /// Add a rectangle.
    #[inline]
    pub fn add(&mut self, rect: Rect) {
        self.boxes.push(rect);
    }

    /// Add a rectangle from raw bounds.
    #[inline]
    pub fn add_bounds(&mut self, min_x: Num, min_y: Num, max_x: Num, max_y: Num) {
        self.add(Rect::new(min_x, min_y, max_x, max_y));
    }

    /// Pack the tree and return the finished index.
    pub fn finish(self) -> Result<Index, BuildError> {
        if self.boxes.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.boxes.len(),
                expected: self.num_items,
            });
        }
        Ok(self.build_unchecked())
    }

    /// Pack the tree into the SIMD-accelerated SoA index.
    #[cfg(feature = "simd")]
    pub fn finish_simd(self) -> Result<crate::SimdIndex, BuildError> {
        if self.boxes.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.boxes.len(),
                expected: self.num_items,
            });
        }
        Ok(crate::index_soa::build_simd_index(
            self.node_size,
            self.num_items,
            self.sort_key,
            self.boxes,
        ))
    }

    fn build_unchecked(self) -> Index {
        let node_size = self.node_size;
        let num_items = self.num_items;

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
            return Index {
                node_size,
                num_items,
                level_bounds,
                boxes: Vec::new(),
                indices: Vec::new(),
            };
        }

        if num_items <= node_size {
            return build_single_node_index(node_size, num_items, level_bounds, self.boxes);
        }

        let mut boxes: Vec<Rect> = vec![Rect::new(0.0, 0.0, 0.0, 0.0); num_nodes];
        let mut indices: Vec<usize> = vec![0usize; num_nodes];
        let items = &self.boxes;

        #[cfg(feature = "parallel")]
        let use_parallel = self.parallel && num_items >= self.parallel_min_items;

        let mut min_x = Num::INFINITY;
        let mut min_y = Num::INFINITY;
        let mut max_x = Num::NEG_INFINITY;
        let mut max_y = Num::NEG_INFINITY;
        for b in items {
            min_x = min_x.min(b.min_x);
            min_y = min_y.min(b.min_y);
            max_x = max_x.max(b.max_x);
            max_y = max_y.max(b.max_y);
        }

        let scaled_width = u16::MAX as f64 / (max_x - min_x);
        let scaled_height = u16::MAX as f64 / (max_y - min_y);
        let sort_key = self.sort_key;

        let encode = |i: usize, b: &Rect| -> (u32, u32) {
            let hx = hilbert_coord(scaled_width, b.min_x, b.max_x, min_x);
            let hy = hilbert_coord(scaled_height, b.min_y, b.max_y, min_y);
            (sort_key.encode(hx, hy), i as u32)
        };

        #[cfg(feature = "parallel")]
        let order: Vec<(u32, u32)> = if use_parallel {
            encode_sort_parallel(items, &encode)
        } else {
            encode_sort_serial(items, &encode, self.radix)
        };
        #[cfg(not(feature = "parallel"))]
        let order: Vec<(u32, u32)> = encode_sort_serial(items, &encode, self.radix);

        #[cfg(feature = "parallel")]
        if use_parallel {
            reorder_parallel(&mut boxes, &mut indices, &order, items, num_items);
        } else {
            reorder_serial(&mut boxes, &mut indices, &order, items);
        }
        #[cfg(not(feature = "parallel"))]
        reorder_serial(&mut boxes, &mut indices, &order, items);

        let mut read_pos = 0usize;
        let mut write_pos = num_items;
        for &level_end in &level_bounds[0..level_bounds.len() - 1] {
            while read_pos < level_end {
                let node_index = read_pos;
                let mut node_bounds = Rect::new(
                    Num::INFINITY,
                    Num::INFINITY,
                    Num::NEG_INFINITY,
                    Num::NEG_INFINITY,
                );
                let mut j = 0;
                while j < node_size && read_pos < level_end {
                    let b = boxes[read_pos];
                    read_pos += 1;
                    node_bounds.min_x = node_bounds.min_x.min(b.min_x);
                    node_bounds.min_y = node_bounds.min_y.min(b.min_y);
                    node_bounds.max_x = node_bounds.max_x.max(b.max_x);
                    node_bounds.max_y = node_bounds.max_y.max(b.max_y);
                    j += 1;
                }
                boxes[write_pos] = node_bounds;
                indices[write_pos] = node_index;
                write_pos += 1;
            }
        }

        Index {
            node_size,
            num_items,
            level_bounds,
            boxes,
            indices,
        }
    }
}

/// Finished static read-only index.
///
/// Search methods return item positions in the original insertion order. The order
/// of returned results is traversal order and is not part of the API.
pub struct Index {
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    boxes: Vec<Rect>,
    indices: Vec<usize>,
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
        let mut stack: Vec<usize> = Vec::with_capacity(16);
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

    /// Visit intersecting items without collecting a result `Vec`.
    ///
    /// The visitor receives item positions in the original insertion order. Return
    /// [`ControlFlow::Continue`] to continue traversal or [`ControlFlow::Break`] for
    /// early exit with a user-provided value.
    pub fn visit<B, F>(&self, rect: Rect, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(16);
        self.visit_with_stack(rect, &mut stack, visitor)
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
            for pos in node_index..end {
                // SAFETY: pos < end <= level_bounds[level] <= boxes.len() == indices.len().
                let b = unsafe { self.boxes.get_unchecked(pos) };
                if !b.overlaps(rect) {
                    continue;
                }
                let index = unsafe { *self.indices.get_unchecked(pos) };
                if is_leaf {
                    results.push(index);
                } else {
                    stack.push(index);
                    stack.push(level - 1);
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
            for pos in node_index..end {
                // SAFETY: pos < end <= level_bounds[level] <= boxes.len() == indices.len().
                let b = unsafe { self.boxes.get_unchecked(pos) };
                if !b.overlaps(rect) {
                    continue;
                }
                let index = unsafe { *self.indices.get_unchecked(pos) };
                if is_leaf {
                    visitor(index)?;
                } else {
                    stack.push(index);
                    stack.push(level - 1);
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
        let mut stack: Vec<usize> = Vec::with_capacity(16);

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

fn encode_sort_serial<F>(items: &[Rect], encode: &F, radix: bool) -> Vec<(u32, u32)>
where
    F: Fn(usize, &Rect) -> (u32, u32),
{
    let mut order: Vec<(u32, u32)> = Vec::with_capacity(items.len());
    for (i, b) in items.iter().enumerate() {
        order.push(encode(i, b));
    }
    if radix {
        radix_sort_u32(&mut order);
    } else {
        order.sort_unstable_by_key(|&(h, _)| h);
    }
    order
}

#[cfg(feature = "parallel")]
fn encode_sort_parallel<F>(items: &[Rect], encode: &F) -> Vec<(u32, u32)>
where
    F: Fn(usize, &Rect) -> (u32, u32) + Sync,
{
    use rayon::prelude::*;

    let mut order: Vec<(u32, u32)> = items
        .par_iter()
        .enumerate()
        .map(|(i, b)| encode(i, b))
        .collect();
    order.par_sort_unstable_by_key(|&(h, _)| h);
    order
}

fn reorder_serial(boxes: &mut [Rect], indices: &mut [usize], order: &[(u32, u32)], items: &[Rect]) {
    for (slot, &(_, orig)) in order.iter().enumerate() {
        boxes[slot] = items[orig as usize];
        indices[slot] = orig as usize;
    }
}

fn build_single_node_index(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    mut boxes: Vec<Rect>,
) -> Index {
    let mut root = Rect::new(
        Num::INFINITY,
        Num::INFINITY,
        Num::NEG_INFINITY,
        Num::NEG_INFINITY,
    );
    for b in &boxes {
        root.min_x = root.min_x.min(b.min_x);
        root.min_y = root.min_y.min(b.min_y);
        root.max_x = root.max_x.max(b.max_x);
        root.max_y = root.max_y.max(b.max_y);
    }
    boxes.push(root);

    let mut indices = Vec::with_capacity(num_items + 1);
    indices.extend(0..num_items);
    indices.push(0);

    Index {
        node_size,
        num_items,
        level_bounds,
        boxes,
        indices,
    }
}

#[cfg(feature = "parallel")]
fn reorder_parallel(
    boxes: &mut [Rect],
    indices: &mut [usize],
    order: &[(u32, u32)],
    items: &[Rect],
    num_items: usize,
) {
    use rayon::prelude::*;

    boxes[..num_items]
        .par_iter_mut()
        .zip(indices[..num_items].par_iter_mut())
        .zip(order.par_iter())
        .for_each(|((slot_box, slot_idx), &(_, orig))| {
            *slot_box = items[orig as usize];
            *slot_idx = orig as usize;
        });
}

/// Normalize the bbox center into `[0, 65535]` (Hilbert encoder input), with saturation.
#[inline]
pub(crate) fn hilbert_coord(scaled: f64, lo: f64, hi: f64, extent_min: f64) -> u16 {
    let value = scaled * (0.5 * (lo + hi) - extent_min);
    if value.is_nan() {
        0
    } else if value > u16::MAX as f64 {
        u16::MAX
    } else if value < 0.0 {
        0
    } else {
        value as u16
    }
}

/// LSD radix sort for `(key_u32, index)` pairs, with configurable digit width.
#[doc(hidden)]
pub fn radix_sort_pairs(a: &mut [(u32, u32)], bits: u32) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let buckets = 1usize << bits;
    let mask = (buckets as u32) - 1;
    let passes = 32u32.div_ceil(bits);

    let mut tmp: Vec<(u32, u32)> = vec![(0, 0); n];
    let mut counts = vec![0usize; buckets];

    fn pass(
        src: &[(u32, u32)],
        dst: &mut [(u32, u32)],
        shift: u32,
        mask: u32,
        counts: &mut [usize],
    ) {
        counts.iter_mut().for_each(|c| *c = 0);
        for &(k, _) in src {
            counts[((k >> shift) & mask) as usize] += 1;
        }
        let mut sum = 0usize;
        for c in counts.iter_mut() {
            let cnt = *c;
            *c = sum;
            sum += cnt;
        }
        for &pair in src {
            let b = ((pair.0 >> shift) & mask) as usize;
            dst[counts[b]] = pair;
            counts[b] += 1;
        }
    }

    for p in 0..passes {
        let shift = p * bits;
        if p % 2 == 0 {
            pass(a, &mut tmp, shift, mask, &mut counts);
        } else {
            pass(&tmp, a, shift, mask, &mut counts);
        }
    }
    if passes % 2 == 1 {
        a.copy_from_slice(&tmp);
    }
}

fn radix_sort_u32(a: &mut [(u32, u32)]) {
    radix_sort_pairs(a, 8);
}
