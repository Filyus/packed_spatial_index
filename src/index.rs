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

use std::{collections::BinaryHeap, error::Error, fmt, ops::ControlFlow};

use crate::hilbert;

/// Index coordinates are `f64`, matching the reference default.
type Num = f64;

const DEFAULT_NODE_SIZE: usize = 16;
const FORMAT_MAGIC: &[u8; 8] = b"PSIDX001";
const FORMAT_HEADER_LEN: usize = 40;

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

    #[inline]
    fn distance_squared_to(&self, point: Point) -> f64 {
        let dx = axis_distance(point.x, self.min_x, self.max_x);
        let dy = axis_distance(point.y, self.min_y, self.max_y);
        dx * dx + dy * dy
    }
}

/// 2D point used by nearest-neighbor searches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point {
    /// Create a point from `x, y`.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
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

/// Reusable buffers for allocation-free repeated nearest-neighbor searches.
#[derive(Debug, Default)]
pub struct NeighborWorkspace {
    pub(crate) results: Vec<usize>,
    pub(crate) queue: BinaryHeap<NeighborState>,
}

impl NeighborWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a workspace with preallocated result and priority-queue capacity.
    pub fn with_capacity(results: usize, queue: usize) -> Self {
        Self {
            results: Vec::with_capacity(results),
            queue: BinaryHeap::with_capacity(queue),
        }
    }

    /// Results from the latest `neighbors_with` call.
    pub fn results(&self) -> &[usize] {
        &self.results
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NeighborState {
    pub(crate) index: usize,
    pub(crate) is_leaf: bool,
    pub(crate) dist: f64,
}

impl NeighborState {
    #[inline]
    pub(crate) fn new(index: usize, is_leaf: bool, dist: f64) -> Self {
        Self {
            index,
            is_leaf,
            dist,
        }
    }
}

impl Eq for NeighborState {}

impl Ord for NeighborState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| self.is_leaf.cmp(&other.is_leaf))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for NeighborState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
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

/// Which key to use when sorting boxes before packing the tree.
///
/// [`SortKey::Hilbert`] is the default and currently the only stable public
/// ordering. Additional sort keys are kept in the hidden experimental API for
/// benchmarking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    /// Hilbert curve order.
    Hilbert,
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

/// Error returned when loading an index from bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// The buffer does not start with the expected `PSIDX001` magic/version marker.
    BadMagic,
    /// The buffer uses a newer or otherwise unsupported format version.
    UnsupportedVersion,
    /// The buffer ended before a complete header or section could be read.
    Truncated,
    /// The buffer length does not match the length declared by the header.
    LengthMismatch {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// The stored node size is outside the supported range.
    InvalidNodeSize {
        /// Stored node size.
        node_size: usize,
    },
    /// A stored integer does not fit this platform or a byte-size calculation overflowed.
    IntegerOverflow,
    /// The level bounds or child pointers do not describe a valid packed tree.
    InvalidTree,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::BadMagic => write!(f, "buffer is not a packed_spatial_index index"),
            LoadError::UnsupportedVersion => write!(f, "unsupported packed_spatial_index format"),
            LoadError::Truncated => write!(f, "buffer is truncated"),
            LoadError::LengthMismatch { expected, actual } => write!(
                f,
                "buffer length mismatch (expected {expected} bytes, got {actual})"
            ),
            LoadError::InvalidNodeSize { node_size } => {
                write!(f, "invalid node size in buffer ({node_size})")
            }
            LoadError::IntegerOverflow => write!(f, "buffer integer value is too large"),
            LoadError::InvalidTree => write!(f, "buffer does not contain a valid packed tree"),
        }
    }
}

impl Error for LoadError {}

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

    /// Serialize this index into the stable little-endian `PSIDX001` format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let level_count = self.level_bounds.len();
        let num_nodes = self.boxes.len();
        let len = serialized_len(level_count, num_nodes).expect("serialized index is too large");
        let mut bytes = Vec::with_capacity(len);
        bytes.extend_from_slice(FORMAT_MAGIC);
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
        let mut queue = BinaryHeap::with_capacity(16);
        results.clear();
        if max_results == 0 {
            return;
        }
        let mut visitor = |index, _dist| {
            results.push(index);
            if results.len() == max_results {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let _ = self.visit_neighbors_with_queue(point, max_distance, &mut queue, &mut visitor);
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
            return &workspace.results;
        }
        let mut visitor = |index, _dist| {
            workspace.results.push(index);
            if workspace.results.len() == max_results {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let _ = self.visit_neighbors_with_queue(
            point,
            max_distance,
            &mut workspace.queue,
            &mut visitor,
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
        let mut queue = BinaryHeap::with_capacity(16);
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
        let mut stack: Vec<usize> = Vec::with_capacity(16);
        self.visit_with_stack(rect, &mut stack, visitor)
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
        let mut stack: Vec<usize> = Vec::with_capacity(16);
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
        let mut queue = BinaryHeap::with_capacity(16);
        results.clear();
        if max_results == 0 {
            return;
        }
        let mut visitor = |index, _dist| {
            results.push(index);
            if results.len() == max_results {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let _ = self.visit_neighbors_with_queue(point, max_distance, &mut queue, &mut visitor);
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
            return &workspace.results;
        }
        let mut visitor = |index, _dist| {
            workspace.results.push(index);
            if workspace.results.len() == max_results {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let _ = self.visit_neighbors_with_queue(
            point,
            max_distance,
            &mut workspace.queue,
            &mut visitor,
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
        let mut queue = BinaryHeap::with_capacity(16);
        self.visit_neighbors_with_queue(point, max_distance, &mut queue, &mut visitor)
    }

    /// Visit intersecting items without collecting a result `Vec`.
    pub fn visit<B, F>(&self, rect: Rect, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(16);
        self.visit_with_stack(rect, &mut stack, visitor)
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
            for pos in node_index..end {
                let b = self.box_at_unchecked(pos);
                if !b.overlaps(rect) {
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
            for pos in node_index..end {
                let b = self.box_at_unchecked(pos);
                if !b.overlaps(rect) {
                    continue;
                }
                let index = self.index_at_unchecked(pos);
                if is_leaf {
                    visitor(index)?;
                } else {
                    stack.push(index);
                    stack.push(level - 1);
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

struct ParsedIndexBytes<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    level_bounds: &'a [u8],
    boxes: &'a [u8],
    indices: &'a [u8],
}

fn parse_index_bytes(bytes: &[u8]) -> Result<ParsedIndexBytes<'_>, LoadError> {
    if bytes.len() < FORMAT_MAGIC.len() {
        return Err(LoadError::Truncated);
    }
    if &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return if bytes.starts_with(b"PSIDX") {
            Err(LoadError::UnsupportedVersion)
        } else {
            Err(LoadError::BadMagic)
        };
    }
    if bytes.len() < FORMAT_HEADER_LEN {
        return Err(LoadError::Truncated);
    }

    let node_size = read_u64_at(bytes, 8).and_then(usize_from_u64)?;
    let num_items = read_u64_at(bytes, 16).and_then(usize_from_u64)?;
    let num_nodes = read_u64_at(bytes, 24).and_then(usize_from_u64)?;
    let level_count = read_u64_at(bytes, 32).and_then(usize_from_u64)?;

    if !(2..=65535).contains(&node_size) {
        return Err(LoadError::InvalidNodeSize { node_size });
    }

    let (expected_nodes, expected_levels) = expected_tree_shape(num_items, node_size)?;
    if num_nodes != expected_nodes || level_count != expected_levels {
        return Err(LoadError::InvalidTree);
    }

    let expected_len = serialized_len(level_count, num_nodes)?;
    if bytes.len() < expected_len {
        return Err(LoadError::Truncated);
    }
    if bytes.len() != expected_len {
        return Err(LoadError::LengthMismatch {
            expected: expected_len,
            actual: bytes.len(),
        });
    }

    let level_bounds_len = level_count
        .checked_mul(8)
        .ok_or(LoadError::IntegerOverflow)?;
    let boxes_len = num_nodes
        .checked_mul(32)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_len = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;

    let level_start = FORMAT_HEADER_LEN;
    let boxes_start = level_start
        .checked_add(level_bounds_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_start = boxes_start
        .checked_add(boxes_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let end = indices_start
        .checked_add(indices_len)
        .ok_or(LoadError::IntegerOverflow)?;

    let parsed = ParsedIndexBytes {
        node_size,
        num_items,
        num_nodes,
        level_count,
        level_bounds: &bytes[level_start..boxes_start],
        boxes: &bytes[boxes_start..indices_start],
        indices: &bytes[indices_start..end],
    };
    validate_level_bounds(&parsed)?;
    validate_indices(&parsed)?;
    Ok(parsed)
}

fn validate_level_bounds(parsed: &ParsedIndexBytes<'_>) -> Result<(), LoadError> {
    let mut n = parsed.num_items;
    let mut running_total = n;
    for level in 0..parsed.level_count {
        let actual = read_u64_at(parsed.level_bounds, level * 8).and_then(usize_from_u64)?;
        if actual != running_total {
            return Err(LoadError::InvalidTree);
        }
        if level + 1 == parsed.level_count {
            break;
        }
        if n == 0 {
            return Err(LoadError::InvalidTree);
        }
        n = n.div_ceil(parsed.node_size);
        running_total = running_total
            .checked_add(n)
            .ok_or(LoadError::IntegerOverflow)?;
    }
    if read_u64_at(parsed.level_bounds, (parsed.level_count - 1) * 8).and_then(usize_from_u64)?
        != parsed.num_nodes
    {
        return Err(LoadError::InvalidTree);
    }
    Ok(())
}

fn validate_indices(parsed: &ParsedIndexBytes<'_>) -> Result<(), LoadError> {
    for pos in 0..parsed.num_items {
        let index = read_u64_at(parsed.indices, pos * 8).and_then(usize_from_u64)?;
        if index >= parsed.num_items {
            return Err(LoadError::InvalidTree);
        }
    }

    for level in 1..parsed.level_count {
        let level_start =
            read_u64_at(parsed.level_bounds, (level - 1) * 8).and_then(usize_from_u64)?;
        let level_end = read_u64_at(parsed.level_bounds, level * 8).and_then(usize_from_u64)?;
        let child_level_start = if level == 1 {
            0
        } else {
            read_u64_at(parsed.level_bounds, (level - 2) * 8).and_then(usize_from_u64)?
        };
        let child_level_end = level_start;

        for pos in level_start..level_end {
            let index = read_u64_at(parsed.indices, pos * 8).and_then(usize_from_u64)?;
            if index < child_level_start || index >= child_level_end {
                return Err(LoadError::InvalidTree);
            }
            if (index - child_level_start) % parsed.node_size != 0 {
                return Err(LoadError::InvalidTree);
            }
        }
    }

    Ok(())
}

fn expected_tree_shape(num_items: usize, node_size: usize) -> Result<(usize, usize), LoadError> {
    let mut num_nodes = num_items;
    let mut levels = 1usize;
    let mut n = num_items;
    if num_items > 0 {
        loop {
            n = n.div_ceil(node_size);
            num_nodes = num_nodes.checked_add(n).ok_or(LoadError::IntegerOverflow)?;
            levels = levels.checked_add(1).ok_or(LoadError::IntegerOverflow)?;
            if n == 1 {
                break;
            }
        }
    }
    Ok((num_nodes, levels))
}

fn serialized_len(level_count: usize, num_nodes: usize) -> Result<usize, LoadError> {
    let levels = level_count
        .checked_mul(8)
        .ok_or(LoadError::IntegerOverflow)?;
    let boxes = num_nodes
        .checked_mul(32)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;
    FORMAT_HEADER_LEN
        .checked_add(levels)
        .and_then(|len| len.checked_add(boxes))
        .and_then(|len| len.checked_add(indices))
        .ok_or(LoadError::IntegerOverflow)
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, LoadError> {
    let end = offset.checked_add(8).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

#[inline]
fn read_u64_le_unchecked(bytes: &[u8], offset: usize) -> u64 {
    debug_assert!(offset <= bytes.len());
    debug_assert!(bytes.len() - offset >= 8);

    let mut value = 0u64;
    // SAFETY: callers only use this for slices and offsets validated by
    // `parse_index_bytes`; unaligned byte buffers are copied into an aligned u64.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().add(offset),
            (&mut value as *mut u64).cast::<u8>(),
            8,
        );
    }
    u64::from_le(value)
}

#[inline]
fn read_f64_le_unchecked(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64_le_unchecked(bytes, offset))
}

fn usize_from_u64(value: u64) -> Result<usize, LoadError> {
    usize::try_from(value).map_err(|_| LoadError::IntegerOverflow)
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

pub(crate) fn max_distance_squared(max_distance: f64) -> Option<f64> {
    if max_distance.is_nan() || max_distance.is_sign_negative() {
        None
    } else {
        Some(max_distance * max_distance)
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
