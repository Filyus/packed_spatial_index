//! f32-storage SoA index variant with SIMD searches (`f32-storage` feature).
//!
//! Built from f64 input through [`Index2DBuilder::finish_simd_f32`]. Coordinates
//! are stored as `f32` rounded outward, so stored boxes can be larger than the
//! original f64 boxes. Rounded range search returns every exact hit, and may
//! also include extra near-boundary hits.
//! `search_exact*` and `visit_exact` use caller-owned f64 boxes for exact
//! range hits. Exact KNN is available through `neighbors_exact*`.
//!
//! Exact methods trade some speed for compact storage. Prefer f64 indexes for
//! exact range queries with many hits and fastest exact KNN.

use crate::{
    build::BuildError,
    builder2d::BuildConfig,
    geometry::Box2D,
    persistence::{
        LoadError, MetaFields, ParsedTree, PayloadError, parse_index, read_f32_le_unchecked,
        read_u64_le_unchecked, write_index_container,
    },
    ray::Ray2D,
    sort2d::{SortKeyContext, encode_sort_by_key},
    tree::{TreeLayout, try_compute_tree_layout},
    triangle::{Triangle2, records_as_bytes},
};

// Imports used only by the SIMD query frontend (SimdIndex2DF32 + its view).
#[cfg(feature = "simd")]
use crate::{
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::Point2D,
    neighbors::{
        ExactNeighborState, NeighborNodeState, NeighborState, NeighborWorkspace,
        max_distance_squared,
    },
    traversal::{SearchWorkspace, upper_bound_level},
};
#[cfg(feature = "simd")]
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Round `x` down to the nearest `f32` that is `<= x`.
#[inline]
fn round_down(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) > x { r.next_down() } else { r }
}

/// Round `x` up to the nearest `f32` that is `>= x`.
#[inline]
fn round_up(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) < x { r.next_up() } else { r }
}

/// Materialize the SoA f32 columns of a parsed `f32`-box tree into the canonical
/// [`Index2DF32`] storage. Shared by the scalar and SIMD `from_bytes` loaders.
fn f32_columns_from_parsed_2(parsed: &ParsedTree) -> Index2DF32 {
    let num_nodes = parsed.num_nodes;
    let mut min_xs = Vec::with_capacity(num_nodes);
    let mut min_ys = Vec::with_capacity(num_nodes);
    let mut max_xs = Vec::with_capacity(num_nodes);
    let mut max_ys = Vec::with_capacity(num_nodes);
    let mut indices = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let off = i * 16; // four f32 per 2D box record
        min_xs.push(read_f32_le_unchecked(parsed.entries, off));
        min_ys.push(read_f32_le_unchecked(parsed.entries, off + 4));
        max_xs.push(read_f32_le_unchecked(parsed.entries, off + 8));
        max_ys.push(read_f32_le_unchecked(parsed.entries, off + 12));
        indices.push(read_u64_le_unchecked(parsed.indices, i * 8) as usize);
    }
    Index2DF32 {
        node_size: parsed.node_size,
        num_items: parsed.num_items,
        level_bounds: parsed.level_bounds.clone(),
        min_xs,
        min_ys,
        max_xs,
        max_ys,
        indices,
    }
}

/// High bit of the stacked level word, set when the query fully contains a node so
/// its whole subtree can be collected without further overlap tests.
#[cfg(feature = "simd")]
const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
#[cfg(feature = "simd")]
const LEVEL_MASK: usize = !CONTAINED_FLAG;

#[cfg(feature = "simd")]
#[inline]
fn encode_level(level: usize, contained: bool) -> usize {
    if contained {
        level | CONTAINED_FLAG
    } else {
        level
    }
}

/// 2D box stored as four `f32` (`min_x, min_y, max_x, max_y`).
#[derive(Clone, Copy)]
struct Box2DF32 {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl Box2DF32 {
    /// Superset of `b` with bounds rounded outward (min down, max up).
    #[inline]
    fn from_box2d_outward(b: Box2D) -> Self {
        Self {
            min_x: round_down(b.min_x),
            min_y: round_down(b.min_y),
            max_x: round_up(b.max_x),
            max_y: round_up(b.max_y),
        }
    }

    /// `b` rounded *inward* (min up, max down) onto the f32 grid. Testing a stored
    /// f32 box against this with [`overlaps`](Self::overlaps) is bit-identical to
    /// widening that stored box to f64 and testing it against `b` -- so a scalar
    /// f32 query compares f32-vs-f32 with no per-node widen yet returns exactly the
    /// same hits as the f64 comparison.
    #[inline]
    fn from_box2d_inward(b: Box2D) -> Self {
        Self {
            min_x: round_up(b.min_x),
            min_y: round_up(b.min_y),
            max_x: round_down(b.max_x),
            max_y: round_down(b.max_y),
        }
    }

    /// Widen losslessly to an f64 box.
    #[inline]
    fn widen(self) -> Box2D {
        Box2D::new(
            self.min_x as f64,
            self.min_y as f64,
            self.max_x as f64,
            self.max_y as f64,
        )
    }

    #[inline]
    fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    #[inline]
    fn definitely_overlaps_exact(self, query: Box2D) -> bool {
        (self.min_x.next_up() as f64 <= query.max_x)
            && (self.max_x.next_down() as f64 >= query.min_x)
            && (self.min_y.next_up() as f64 <= query.max_y)
            && (self.max_y.next_down() as f64 >= query.min_y)
    }
}

/// Containment test is used only by the SIMD query frontend.
#[cfg(feature = "simd")]
impl Box2DF32 {
    /// True when `self` fully contains `other` (both already rounded). Used only on
    /// the conservative (non-refined) path, where the leaf MBR property guarantees a
    /// contained node's whole subtree overlaps the rounded query.
    #[inline]
    fn contains(self, other: Self) -> bool {
        self.min_x <= other.min_x
            && other.max_x <= self.max_x
            && self.min_y <= other.min_y
            && other.max_y <= self.max_y
    }
}

/// Build the native `f32` SoA tree directly from f64 input (no transient f64
/// tree). Returns the neutral [`Index2DF32`] storage; `SimdIndex2DF32` is built
/// by moving its columns.
pub(crate) fn build_f32_2(
    config: BuildConfig,
    items: Vec<Box2D>,
) -> Result<Index2DF32, BuildError> {
    let node_size = config.node_size;
    let num_items = config.num_items;
    let TreeLayout {
        level_bounds,
        num_nodes,
    } = try_compute_tree_layout(num_items, node_size)?;

    if num_items == 0 {
        return Ok(Index2DF32 {
            node_size,
            num_items,
            level_bounds,
            min_xs: Vec::new(),
            min_ys: Vec::new(),
            max_xs: Vec::new(),
            max_ys: Vec::new(),
            indices: Vec::new(),
        });
    }

    if num_items <= node_size {
        return Ok(build_single_node_soa_f32(
            node_size,
            num_items,
            level_bounds,
            items,
        ));
    }

    let mut min_xs = vec![0.0f32; num_nodes];
    let mut min_ys = vec![0.0f32; num_nodes];
    let mut max_xs = vec![0.0f32; num_nodes];
    let mut max_ys = vec![0.0f32; num_nodes];
    let mut indices = vec![0usize; num_nodes];

    let (mut e_min_x, mut e_min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut e_max_x, mut e_max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for b in &items {
        e_min_x = e_min_x.min(b.min_x);
        e_min_y = e_min_y.min(b.min_y);
        e_max_x = e_max_x.max(b.max_x);
        e_max_y = e_max_y.max(b.max_y);
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

    for (slot, &(_, orig)) in order.iter().enumerate() {
        let b = Box2DF32::from_box2d_outward(items[orig as usize]);
        min_xs[slot] = b.min_x;
        min_ys[slot] = b.min_y;
        max_xs[slot] = b.max_x;
        max_ys[slot] = b.max_y;
        indices[slot] = orig as usize;
    }

    let mut read_pos = 0usize;
    let mut write_pos = num_items;
    for &level_end in &level_bounds[0..level_bounds.len() - 1] {
        while read_pos < level_end {
            let node_index = read_pos;
            let (mut nmnx, mut nmny) = (f32::INFINITY, f32::INFINITY);
            let (mut nmxx, mut nmxy) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            let mut j = 0;
            while j < node_size && read_pos < level_end {
                nmnx = nmnx.min(min_xs[read_pos]);
                nmny = nmny.min(min_ys[read_pos]);
                nmxx = nmxx.max(max_xs[read_pos]);
                nmxy = nmxy.max(max_ys[read_pos]);
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

    Ok(Index2DF32 {
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

fn build_single_node_soa_f32(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    items: Vec<Box2D>,
) -> Index2DF32 {
    let mut min_xs = Vec::with_capacity(num_items + 1);
    let mut min_ys = Vec::with_capacity(num_items + 1);
    let mut max_xs = Vec::with_capacity(num_items + 1);
    let mut max_ys = Vec::with_capacity(num_items + 1);
    let mut indices = Vec::with_capacity(num_items + 1);

    let (mut rmnx, mut rmny) = (f32::INFINITY, f32::INFINITY);
    let (mut rmxx, mut rmxy) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (idx, b) in items.into_iter().enumerate() {
        let b = Box2DF32::from_box2d_outward(b);
        min_xs.push(b.min_x);
        min_ys.push(b.min_y);
        max_xs.push(b.max_x);
        max_ys.push(b.max_y);
        indices.push(idx);

        rmnx = rmnx.min(b.min_x);
        rmny = rmny.min(b.min_y);
        rmxx = rmxx.max(b.max_x);
        rmxy = rmxy.max(b.max_y);
    }

    min_xs.push(rmnx);
    min_ys.push(rmny);
    max_xs.push(rmxx);
    max_ys.push(rmxy);
    indices.push(0);

    Index2DF32 {
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

/// Finished read-only f32-storage SIMD index.
///
/// Created through [`Index2DBuilder::finish_simd_f32`](crate::Index2DBuilder::finish_simd_f32).
/// Half the box storage of [`SimdIndex2D`](crate::SimdIndex2D). Rounded range
/// search may include extra near-boundary hits. Use `search_exact` for exact
/// range hits when the original f64 boxes are available.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, Box2D};
///
/// let mut builder = Index2DBuilder::new(1);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
///
/// let index = builder.finish_simd_f32().unwrap();
/// assert!(index.search(Box2D::new(0.5, 0.5, 0.5, 0.5)).contains(&0));
/// ```
#[cfg(feature = "simd")]
pub struct SimdIndex2DF32 {
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    min_xs: Vec<f32>,
    min_ys: Vec<f32>,
    max_xs: Vec<f32>,
    max_ys: Vec<f32>,
    indices: Vec<usize>,
}

#[cfg(feature = "simd")]
impl SimdIndex2DF32 {
    /// Build the SIMD frontend by moving the columns out of the native f32 build.
    pub(crate) fn from_scalar(s: Index2DF32) -> Self {
        Self {
            node_size: s.node_size,
            num_items: s.num_items,
            level_bounds: s.level_bounds,
            min_xs: s.min_xs,
            min_ys: s.min_ys,
            max_xs: s.max_xs,
            max_ys: s.max_ys,
            indices: s.indices,
        }
    }

    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Total extent of indexed items, or `None` for an empty index.
    ///
    /// The returned box covers the exact f64 extent.
    pub fn extent(&self) -> Option<Box2D> {
        if self.num_items == 0 {
            None
        } else {
            let last = self.min_xs.len() - 1;
            Some(Box2D::new(
                self.min_xs[last] as f64,
                self.min_ys[last] as f64,
                self.max_xs[last] as f64,
                self.max_ys[last] as f64,
            ))
        }
    }

    /// Packed node size used by this index.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Serialize into the little-endian `PSINDEX` format (f32 box records).
    ///
    /// This is a distinct format from [`SimdIndex2D::to_bytes`](crate::SimdIndex2D::to_bytes)
    /// (half the box bytes) and is loaded back only through
    /// [`from_bytes`](Self::from_bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.to_bytes_into(&mut out);
        out
    }

    /// Serialize into a caller-provided buffer, reusing its allocation.
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        write_index_container(
            out,
            2,
            4,
            false,
            self.num_items,
            self.min_xs.len(),
            self.node_size,
            |bytes| {
                bytes.write_soa_boxes_f32_2d(
                    &self.min_xs,
                    &self.min_ys,
                    &self.max_xs,
                    &self.max_ys,
                );
                bytes.write_usize_slice_as_u64(&self.indices);
            },
            None,
            None,
            &self.indices[..self.num_items],
            &MetaFields::default(),
        )
        .expect("index-only serialization cannot fail");
    }

    /// Load from bytes produced by [`to_bytes`](Self::to_bytes). A payload section
    /// is rejected (this SIMD index carries boxes only).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 2, 4)?;
        if payload.is_some() {
            return Err(LoadError::UnsupportedVersion);
        }
        Ok(Self::from_scalar(f32_columns_from_parsed_2(&parsed)))
    }

    /// Item indices whose rounded f32 box intersects `query`.
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(query, &mut out);
        out
    }

    /// Rounded-box range search with a reusable result buffer.
    pub fn search_into(&self, query: Box2D, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_into_stack(query, out, &mut stack);
    }

    /// Rounded-box range search with reusable result and traversal buffers.
    pub fn search_with<'a>(&self, query: Box2D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.search_into_stack(query, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Exact item indices whose caller-owned f64 box intersects `query`.
    ///
    /// Best suited for compact indexes when exact range queries return few hits.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder};
    ///
    /// let boxes = [
    ///     Box2D::new(1.0 + 1e-8, 0.0, 1.0 + 1e-8, 0.0),
    ///     Box2D::new(1.0, 0.0, 1.0, 0.0),
    /// ];
    /// let mut builder = Index2DBuilder::new(boxes.len());
    /// for &b in &boxes {
    ///     builder.add(b);
    /// }
    /// let index = builder.finish_simd_f32()?;
    ///
    /// let query = Box2D::new(1.0, 0.0, 1.0, 0.0);
    /// assert_eq!(index.search_exact(query, |i| boxes[i]), vec![1]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn search_exact<F>(&self, query: Box2D, box_at: F) -> Vec<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        let mut out = Vec::new();
        self.search_exact_into(query, box_at, &mut out);
        out
    }

    /// Exact search with a reusable result buffer.
    #[inline]
    pub fn search_exact_into<F>(&self, query: Box2D, box_at: F, out: &mut Vec<usize>)
    where
        F: FnMut(usize) -> Box2D,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.search_refined_into_stack(query, box_at, out, &mut stack);
    }

    /// Exact search with reusable result and traversal buffers.
    #[inline]
    pub fn search_exact_with<'a, F>(
        &self,
        query: Box2D,
        box_at: F,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize]
    where
        F: FnMut(usize) -> Box2D,
    {
        self.search_refined_into_stack(query, box_at, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Return `true` if at least one rounded f32 box intersects `query`.
    pub fn any(&self, query: Box2D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return `true` if at least one caller-owned f64 box intersects `query`.
    pub fn any_exact<F>(&self, query: Box2D, box_at: F) -> bool
    where
        F: FnMut(usize) -> Box2D,
    {
        self.visit_exact(query, box_at, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Return one rounded-box hit, if any.
    pub fn first(&self, query: Box2D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Return one caller-owned f64 box intersecting `query`, if any.
    pub fn first_exact<F>(&self, query: Box2D, box_at: F) -> Option<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        match self.visit_exact(query, box_at, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Up to `max_results` item indices nearest to `point` by rounded f32 boxes.
    ///
    /// Distances use the outward-rounded f32 boxes, so they are lower bounds of
    /// the exact f64 distances. Use [`neighbors_exact`](Self::neighbors_exact)
    /// when you need exact nearest neighbors over your original f64 boxes.
    pub fn neighbors(&self, point: Point2D, max_results: usize) -> Vec<usize> {
        self.neighbors_within(point, max_results, f64::INFINITY)
    }

    /// Up to `max_results` rounded-box nearest items within `max_distance`.
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

    /// Rounded-box nearest-neighbor search with a reusable result buffer.
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

        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(point, max_results, max_distance, results, &mut queue);
    }

    /// Rounded-box nearest-neighbor search with reusable result and queue buffers.
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

    /// Exact nearest neighbors over caller-owned f64 boxes.
    ///
    /// The f32 tree is used as a lower-bound traversal index. `box_at` must
    /// return the original box for the item index passed to it. Prefer f64
    /// indexes for fastest exact KNN.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};
    ///
    /// let boxes = [
    ///     Box2D::new(1.0 + 1e-8, 0.0, 1.0 + 1e-8, 0.0),
    ///     Box2D::new(1.0, 0.0, 1.0, 0.0),
    /// ];
    /// let mut builder = Index2DBuilder::new(boxes.len());
    /// for &b in &boxes {
    ///     builder.add(b);
    /// }
    /// let index = builder.finish_simd_f32()?;
    ///
    /// let nearest = index.neighbors_exact(Point2D::new(1.0, 0.0), 1, |i| boxes[i]);
    /// assert_eq!(nearest, vec![1]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn neighbors_exact<F>(&self, point: Point2D, max_results: usize, box_at: F) -> Vec<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        self.neighbors_exact_within(point, max_results, f64::INFINITY, box_at)
    }

    /// Exact nearest neighbors within `max_distance` over caller-owned f64 boxes.
    pub fn neighbors_exact_within<F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        box_at: F,
    ) -> Vec<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        let mut results = Vec::new();
        self.neighbors_exact_into(point, max_results, max_distance, box_at, &mut results);
        results
    }

    /// Exact nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_exact_into<F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        box_at: F,
        results: &mut Vec<usize>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        let mut frontier = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut best = BinaryHeap::with_capacity(max_results);
        self.collect_neighbors_refined_with_queue(
            point,
            max_results,
            max_distance,
            box_at,
            results,
            &mut frontier,
            &mut best,
        );
    }

    /// Exact nearest-neighbor search with reusable result and queue buffers.
    pub fn neighbors_exact_with<'a, F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        box_at: F,
        workspace: &'a mut NeighborWorkspace,
    ) -> &'a [usize]
    where
        F: FnMut(usize) -> Box2D,
    {
        self.collect_neighbors_refined_with_queue(
            point,
            max_results,
            max_distance,
            box_at,
            &mut workspace.results,
            &mut workspace.queue,
            &mut workspace.exact_queue,
        );
        &workspace.results
    }

    /// Visit rounded-box nearest items in nondecreasing squared-distance order.
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

    /// Visit rounded-box hits without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_with_stack(query, &mut stack, visitor)
    }

    /// Visit exact range hits after checking rounded-box hits against f64 boxes.
    pub fn visit_exact<B, BF, VF>(&self, query: Box2D, box_at: BF, visitor: VF) -> ControlFlow<B>
    where
        BF: FnMut(usize) -> Box2D,
        VF: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_refined_with_stack(query, &mut stack, box_at, visitor)
    }

    fn search_into_stack(&self, query: Box2D, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        let q = Box2DF32::from_box2d_outward(query);
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: selected only after checking avx512f availability.
                unsafe { self.search_avx512(q, out, stack) };
                return;
            }
        }
        self.search_wide(q, out, stack);
    }

    fn search_refined_into_stack<F>(
        &self,
        query: Box2D,
        mut box_at: F,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        let q = Box2DF32::from_box2d_outward(query);
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: selected only after checking avx512f availability.
                unsafe { self.search_refined_avx512(q, query, &mut box_at, out, stack) };
                return;
            }
        }
        self.search_refined_wide(q, query, &mut box_at, out, stack);
    }

    /// AVX2/SSE path through `wide::f32x8` (8 boxes per step).
    fn search_wide(&self, q: Box2DF32, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        use wide::f32x8;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        if q.contains(self.box_f32_at(self.min_xs.len() - 1)) {
            out.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }
        let qmxx_v = f32x8::splat(q.max_x);
        let qmnx_v = f32x8::splat(q.min_x);
        let qmxy_v = f32x8::splat(q.max_y);
        let qmny_v = f32x8::splat(q.min_y);

        let load8 = |a: &[f32], p: usize| -> f32x8 {
            f32x8::from([
                a[p],
                a[p + 1],
                a[p + 2],
                a[p + 3],
                a[p + 4],
                a[p + 5],
                a[p + 6],
                a[p + 7],
            ])
        };

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
                while pos + 8 <= end {
                    let mnx = load8(&self.min_xs, pos);
                    let mxx = load8(&self.max_xs, pos);
                    let mny = load8(&self.min_ys, pos);
                    let mxy = load8(&self.max_ys, pos);
                    let mask = mnx.simd_le(qmxx_v)
                        & mxx.simd_ge(qmnx_v)
                        & mny.simd_le(qmxy_v)
                        & mxy.simd_ge(qmny_v);
                    let bits = mask.to_bitmask();
                    if bits != 0 {
                        let cbits = if is_leaf {
                            0
                        } else {
                            (mnx.simd_ge(qmnx_v)
                                & mxx.simd_le(qmxx_v)
                                & mny.simd_ge(qmny_v)
                                & mxy.simd_le(qmxy_v))
                            .to_bitmask()
                        };
                        for k in 0..8 {
                            if bits & (1 << k) != 0 {
                                let p = pos + k;
                                if is_leaf {
                                    out.push(self.indices[p]);
                                } else {
                                    stack.push(self.indices[p]);
                                    stack.push(encode_level(child_level, cbits & (1 << k) != 0));
                                }
                            }
                        }
                    }
                    pos += 8;
                }

                while pos < end {
                    if self.scalar_hit(pos, q) {
                        if is_leaf {
                            out.push(self.indices[pos]);
                        } else {
                            stack.push(self.indices[pos]);
                            stack.push(encode_level(child_level, self.q_contains_node(q, pos)));
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

    fn search_refined_wide<F>(
        &self,
        q: Box2DF32,
        query: Box2D,
        box_at: &mut F,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        use wide::f32x8;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = f32x8::splat(q.max_x);
        let qmnx_v = f32x8::splat(q.min_x);
        let qmxy_v = f32x8::splat(q.max_y);
        let qmny_v = f32x8::splat(q.min_y);

        let load8 = |a: &[f32], p: usize| -> f32x8 {
            f32x8::from([
                a[p],
                a[p + 1],
                a[p + 2],
                a[p + 3],
                a[p + 4],
                a[p + 5],
                a[p + 6],
                a[p + 7],
            ])
        };

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 8 <= end {
                let mnx = load8(&self.min_xs, pos);
                let mxx = load8(&self.max_xs, pos);
                let mny = load8(&self.min_ys, pos);
                let mxy = load8(&self.max_ys, pos);
                let mask = mnx.simd_le(qmxx_v)
                    & mxx.simd_ge(qmnx_v)
                    & mny.simd_le(qmxy_v)
                    & mxy.simd_ge(qmny_v);
                let bits = mask.to_bitmask();
                if bits != 0 {
                    for k in 0..8 {
                        if bits & (1 << k) != 0 {
                            self.emit_refined(pos + k, level, is_leaf, query, box_at, out, stack);
                        }
                    }
                }
                pos += 8;
            }

            while pos < end {
                if self.scalar_hit(pos, q) {
                    self.emit_refined(pos, level, is_leaf, query, box_at, out, stack);
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

    /// AVX-512 path: 16 boxes per step (a full `node_size = 16` node at once).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn search_avx512(&self, q: Box2DF32, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        use std::arch::x86_64::*;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        if q.contains(self.box_f32_at(self.min_xs.len() - 1)) {
            out.extend_from_slice(&self.indices[..self.num_items]);
            return;
        }
        let qmxx_v = _mm512_set1_ps(q.max_x);
        let qmnx_v = _mm512_set1_ps(q.min_x);
        let qmxy_v = _mm512_set1_ps(q.max_y);
        let qmny_v = _mm512_set1_ps(q.min_y);

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
                while pos + 16 <= end {
                    // SAFETY: `pos + 16 <= end`, and `end` is bounded by the array length.
                    let (mnx, mxx, mny, mxy) = unsafe {
                        (
                            _mm512_loadu_ps(self.min_xs.as_ptr().add(pos)),
                            _mm512_loadu_ps(self.max_xs.as_ptr().add(pos)),
                            _mm512_loadu_ps(self.min_ys.as_ptr().add(pos)),
                            _mm512_loadu_ps(self.max_ys.as_ptr().add(pos)),
                        )
                    };
                    let m1 = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(mnx, qmxx_v);
                    let m2 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(mxx, qmnx_v);
                    let m3 = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(mny, qmxy_v);
                    let m4 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(mxy, qmny_v);
                    let mut bits: u16 = m1 & m2 & m3 & m4;
                    if is_leaf {
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            out.push(self.indices[pos + k]);
                            bits &= bits - 1;
                        }
                    } else {
                        // query contains child: qmin <= cmin && cmax <= qmax on both axes.
                        let c1 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(mnx, qmnx_v);
                        let c2 = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(mxx, qmxx_v);
                        let c3 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(mny, qmny_v);
                        let c4 = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(mxy, qmxy_v);
                        let cbits: u16 = c1 & c2 & c3 & c4;
                        while bits != 0 {
                            let k = bits.trailing_zeros() as usize;
                            stack.push(self.indices[pos + k]);
                            stack.push(encode_level(child_level, cbits & (1 << k) != 0));
                            bits &= bits - 1;
                        }
                    }
                    pos += 16;
                }

                while pos < end {
                    if self.scalar_hit(pos, q) {
                        if is_leaf {
                            out.push(self.indices[pos]);
                        } else {
                            stack.push(self.indices[pos]);
                            stack.push(encode_level(child_level, self.q_contains_node(q, pos)));
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

    /// AVX-512 refined path: 16 boxes per step.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn search_refined_avx512<F>(
        &self,
        q: Box2DF32,
        query: Box2D,
        box_at: &mut F,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        use std::arch::x86_64::*;

        out.clear();
        stack.clear();
        if self.num_items == 0 {
            return;
        }
        let qmxx_v = _mm512_set1_ps(q.max_x);
        let qmnx_v = _mm512_set1_ps(q.min_x);
        let qmxy_v = _mm512_set1_ps(q.max_y);
        let qmny_v = _mm512_set1_ps(q.min_y);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;

            let mut pos = node_index;
            while pos + 16 <= end {
                // SAFETY: `pos + 16 <= end`, and `end` is bounded by the array length.
                let (mnx, mxx, mny, mxy) = unsafe {
                    (
                        _mm512_loadu_ps(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_ps(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_ps(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_ps(self.max_ys.as_ptr().add(pos)),
                    )
                };
                let m1 = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(mnx, qmxx_v);
                let m2 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(mxx, qmnx_v);
                let m3 = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(mny, qmxy_v);
                let m4 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(mxy, qmny_v);
                let mut bits: u16 = m1 & m2 & m3 & m4;
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    self.emit_refined(pos + k, level, is_leaf, query, box_at, out, stack);
                    bits &= bits - 1;
                }
                pos += 16;
            }

            while pos < end {
                if self.scalar_hit(pos, q) {
                    self.emit_refined(pos, level, is_leaf, query, box_at, out, stack);
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

    #[inline]
    fn scalar_hit(&self, pos: usize, q: Box2DF32) -> bool {
        (self.min_xs[pos] <= q.max_x)
            & (self.max_xs[pos] >= q.min_x)
            & (self.min_ys[pos] <= q.max_y)
            & (self.max_ys[pos] >= q.min_y)
    }

    #[inline]
    fn box_f32_at(&self, pos: usize) -> Box2DF32 {
        Box2DF32 {
            min_x: self.min_xs[pos],
            min_y: self.min_ys[pos],
            max_x: self.max_xs[pos],
            max_y: self.max_ys[pos],
        }
    }

    /// True when the rounded query `q` fully contains the stored box at `pos`.
    #[inline]
    fn q_contains_node(&self, q: Box2DF32, pos: usize) -> bool {
        q.contains(self.box_f32_at(pos))
    }

    /// Append every leaf index under the entry at `node_index` (a node at `level`)
    /// without per-item overlap tests, used on the conservative path when the query
    /// fully contains the node.
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

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn emit_refined<F>(
        &self,
        pos: usize,
        level: usize,
        is_leaf: bool,
        query: Box2D,
        box_at: &mut F,
        out: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        let index = self.indices[pos];
        if is_leaf {
            let stored = Box2DF32 {
                min_x: self.min_xs[pos],
                min_y: self.min_ys[pos],
                max_x: self.max_xs[pos],
                max_y: self.max_ys[pos],
            };
            if stored.definitely_overlaps_exact(query) || box_at(index).overlaps(query) {
                out.push(index);
            }
        } else {
            stack.push(index);
            stack.push(level - 1);
        }
    }

    /// Scalar visitor traversal over the f32 columns.
    fn visit_with_stack<B, F>(
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
        let q = Box2DF32::from_box2d_outward(query);
        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                if !self.scalar_hit(pos, q) {
                    continue;
                }
                let index = self.indices[pos];
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

    fn visit_refined_with_stack<B, BF, VF>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        mut box_at: BF,
        mut visitor: VF,
    ) -> ControlFlow<B>
    where
        BF: FnMut(usize) -> Box2D,
        VF: FnMut(usize) -> ControlFlow<B>,
    {
        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }
        let q = Box2DF32::from_box2d_outward(query);
        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                if !self.scalar_hit(pos, q) {
                    continue;
                }
                let index = self.indices[pos];
                if is_leaf {
                    let stored = Box2DF32 {
                        min_x: self.min_xs[pos],
                        min_y: self.min_ys[pos],
                        max_x: self.max_xs[pos],
                        max_y: self.max_ys[pos],
                    };
                    if stored.definitely_overlaps_exact(query) || box_at(index).overlaps(query) {
                        visitor(index)?;
                    }
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

    fn collect_neighbors_with_queue(
        &self,
        point: Point2D,
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

    #[allow(clippy::too_many_arguments)]
    fn collect_neighbors_refined_with_queue<F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        mut box_at: F,
        results: &mut Vec<usize>,
        frontier: &mut BinaryHeap<NeighborState>,
        best: &mut BinaryHeap<ExactNeighborState>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        results.clear();
        frontier.clear();
        best.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if self.num_items == 0 || max_results == 0 {
            return;
        }

        let root = self.min_xs.len() - 1;
        let root_dist = self.distance_squared_to(root, point);
        if root_dist > max_dist_sq {
            return;
        }
        frontier.push(NeighborState::new(root, false, root_dist));

        let mut cutoff = max_dist_sq;
        while let Some(state) = frontier.pop() {
            if state.dist > cutoff {
                break;
            }
            if state.is_leaf {
                let exact_dist = distance_squared_to_box(point, box_at(state.index));
                if exact_dist <= max_dist_sq {
                    push_exact_neighbor(best, max_results, state.index, exact_dist);
                    if best.len() == max_results {
                        cutoff = best.peek().map_or(max_dist_sq, |worst| worst.dist);
                    }
                }
                continue;
            }

            let upper_bound_level = upper_bound_level(&self.level_bounds, state.index);
            let end = (state.index + self.node_size).min(self.level_bounds[upper_bound_level]);
            let is_leaf = state.index < self.num_items;
            for pos in state.index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist <= cutoff {
                    frontier.push(NeighborState::new(self.indices[pos], is_leaf, dist));
                }
            }
        }

        write_exact_results(results, best);
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
    fn distance_squared_to(&self, pos: usize, point: Point2D) -> f64 {
        let dx = axis_distance(point.x, self.min_xs[pos] as f64, self.max_xs[pos] as f64);
        let dy = axis_distance(point.y, self.min_ys[pos] as f64, self.max_ys[pos] as f64);
        dx * dx + dy * dy
    }
}

#[cfg(feature = "simd")]
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

#[cfg(feature = "simd")]
#[inline]
fn distance_squared_to_box(point: Point2D, b: Box2D) -> f64 {
    let dx = axis_distance(point.x, b.min_x, b.max_x);
    let dy = axis_distance(point.y, b.min_y, b.max_y);
    dx * dx + dy * dy
}

#[cfg(feature = "simd")]
#[inline]
fn push_exact_neighbor(
    best: &mut BinaryHeap<ExactNeighborState>,
    max_results: usize,
    index: usize,
    dist: f64,
) {
    let state = ExactNeighborState::new(index, dist);
    if best.len() < max_results {
        best.push(state);
    } else if best.peek().is_some_and(|worst| state < *worst) {
        *best.peek_mut().unwrap() = state;
    }
}

#[cfg(feature = "simd")]
fn write_exact_results(results: &mut Vec<usize>, best: &mut BinaryHeap<ExactNeighborState>) {
    let mut ordered: Vec<_> = best.drain().collect();
    ordered.sort_by(|a, b| {
        a.dist
            .total_cmp(&b.dist)
            .then_with(|| a.index.cmp(&b.index))
    });
    results.extend(ordered.into_iter().map(|state| state.index));
}

/// Zero-copy read-only view over bytes produced by [`SimdIndex2DF32::to_bytes`].
///
/// Loading validates the buffer but does not copy the tree. Rounded range search
/// returns every exact hit, and may also include extra near-boundary hits. Use
/// `search_exact` for exact range hits when the original f64 boxes are
/// available.
#[cfg(feature = "simd")]
pub struct SimdIndex2DF32View<'a> {
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    /// Derived at load (not stored), so owned rather than borrowed.
    level_bounds: Vec<usize>,
    entries: &'a [u8],
    indices: &'a [u8],
}

#[cfg(feature = "simd")]
impl<'a> SimdIndex2DF32View<'a> {
    /// Borrow a zero-copy view over f32-format `PSINDEX` bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 2, 4)?;
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

    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Packed node size.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Total extent of indexed items, or `None` for an empty view.
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

    /// Decode an f32 box record, widened to `f64`.
    #[inline]
    fn box_at(&self, pos: usize) -> Box2D {
        let b = self.box_f32_at(pos);
        Box2D::new(
            b.min_x as f64,
            b.min_y as f64,
            b.max_x as f64,
            b.max_y as f64,
        )
    }

    #[inline]
    fn box_f32_at(&self, pos: usize) -> Box2DF32 {
        let b = pos * 16;
        Box2DF32 {
            min_x: read_f32_le_unchecked(self.entries, b),
            min_y: read_f32_le_unchecked(self.entries, b + 4),
            max_x: read_f32_le_unchecked(self.entries, b + 8),
            max_y: read_f32_le_unchecked(self.entries, b + 12),
        }
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
    /// (a node at `level`), used when the rounded query fully contains that node.
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

    /// Candidate item indices whose stored box intersects `query`.
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_into(query, &mut out);
        out
    }

    /// Candidate search with a reusable result buffer.
    pub fn search_into(&self, query: Box2D, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        out.clear();
        let _: ControlFlow<()> = self.try_visit(query, &mut stack, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Candidate search with reusable result and traversal buffers.
    pub fn search_with<'b>(&self, query: Box2D, workspace: &'b mut SearchWorkspace) -> &'b [usize] {
        workspace.results.clear();
        let results = &mut workspace.results;
        let _: ControlFlow<()> = self.try_visit(query, &mut workspace.stack, |index| {
            results.push(index);
            ControlFlow::Continue(())
        });
        &workspace.results
    }

    /// Exact item indices whose caller-owned f64 box intersects `query`.
    ///
    /// Best suited for compact indexes when exact range queries return few hits.
    pub fn search_exact<F>(&self, query: Box2D, box_at: F) -> Vec<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        let mut out = Vec::new();
        self.search_exact_into(query, box_at, &mut out);
        out
    }

    /// Exact search with a reusable result buffer.
    pub fn search_exact_into<F>(&self, query: Box2D, box_at: F, out: &mut Vec<usize>)
    where
        F: FnMut(usize) -> Box2D,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        out.clear();
        let _: ControlFlow<()> = self.try_visit_refined(query, &mut stack, box_at, |index| {
            out.push(index);
            ControlFlow::Continue(())
        });
    }

    /// Exact search with reusable result and traversal buffers.
    pub fn search_exact_with<'b, F>(
        &self,
        query: Box2D,
        box_at: F,
        workspace: &'b mut SearchWorkspace,
    ) -> &'b [usize]
    where
        F: FnMut(usize) -> Box2D,
    {
        workspace.results.clear();
        let results = &mut workspace.results;
        let _: ControlFlow<()> =
            self.try_visit_refined(query, &mut workspace.stack, box_at, |index| {
                results.push(index);
                ControlFlow::Continue(())
            });
        &workspace.results
    }

    /// Return `true` if at least one rounded f32 box intersects `query`.
    pub fn any(&self, query: Box2D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Return `true` if at least one caller-owned f64 box intersects `query`.
    pub fn any_exact<F>(&self, query: Box2D, box_at: F) -> bool
    where
        F: FnMut(usize) -> Box2D,
    {
        self.visit_exact(query, box_at, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Return one rounded-box hit, if any.
    pub fn first(&self, query: Box2D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Return one caller-owned f64 box intersecting `query`, if any.
    pub fn first_exact<F>(&self, query: Box2D, box_at: F) -> Option<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        match self.visit_exact(query, box_at, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Visit rounded-box hits without collecting a result `Vec`.
    pub fn visit<B, F>(&self, query: Box2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.try_visit(query, &mut stack, visitor)
    }

    /// Visit exact range hits after checking rounded-box hits against f64 boxes.
    pub fn visit_exact<B, BF, VF>(&self, query: Box2D, box_at: BF, visitor: VF) -> ControlFlow<B>
    where
        BF: FnMut(usize) -> Box2D,
        VF: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.try_visit_refined(query, &mut stack, box_at, visitor)
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
        let query = Box2DF32::from_box2d_outward(query);
        if query.contains(self.box_f32_at(self.num_nodes - 1)) {
            for pos in 0..self.num_items {
                visitor(self.index_at(pos))?;
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
                let (start, end) = self.contained_leaf_range(node_index, end, level);
                for pos in start..end {
                    visitor(self.index_at(pos))?;
                }
            } else {
                let child_level = if is_leaf { 0 } else { level - 1 };
                for pos in node_index..end {
                    let stored = self.box_f32_at(pos);
                    if !stored.overlaps(query) {
                        continue;
                    }
                    let index = self.index_at(pos);
                    if is_leaf {
                        visitor(index)?;
                    } else {
                        stack.push(index);
                        stack.push(encode_level(child_level, query.contains(stored)));
                    }
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

    fn try_visit_refined<B, BF, VF>(
        &self,
        query: Box2D,
        stack: &mut Vec<usize>,
        mut box_at: BF,
        mut visitor: VF,
    ) -> ControlFlow<B>
    where
        BF: FnMut(usize) -> Box2D,
        VF: FnMut(usize) -> ControlFlow<B>,
    {
        stack.clear();
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }
        let rounded_query = Box2DF32::from_box2d_outward(query);
        let mut node_index = self.num_nodes - 1;
        let mut level = self.level_count - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bound_unchecked(level));
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                let stored = self.box_f32_at(pos);
                if !stored.overlaps(rounded_query) {
                    continue;
                }
                let index = self.index_at(pos);
                if is_leaf {
                    if stored.definitely_overlaps_exact(query) || box_at(index).overlaps(query) {
                        visitor(index)?;
                    }
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

    #[inline]
    fn distance_squared_to(&self, pos: usize, point: Point2D) -> f64 {
        let b = self.box_at(pos);
        let dx = axis_distance(point.x, b.min_x, b.max_x);
        let dy = axis_distance(point.y, b.min_y, b.max_y);
        dx * dx + dy * dy
    }

    /// Up to `max_results` item indices nearest to `point` by rounded f32 boxes.
    pub fn neighbors(&self, point: Point2D, max_results: usize) -> Vec<usize> {
        self.neighbors_within(point, max_results, f64::INFINITY)
    }

    /// Up to `max_results` rounded-box nearest items within `max_distance`.
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

    /// Rounded-box nearest-neighbor search with a reusable result buffer.
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
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        self.collect_neighbors_with_queue(point, max_results, max_distance, results, &mut queue);
    }

    /// Rounded-box nearest-neighbor search with reusable result and queue buffers.
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

    /// Exact nearest neighbors over caller-owned f64 boxes.
    ///
    /// The f32 tree is used as a lower-bound traversal index. `box_at` must
    /// return the original box for the item index passed to it. Prefer f64
    /// indexes for fastest exact KNN.
    pub fn neighbors_exact<F>(&self, point: Point2D, max_results: usize, box_at: F) -> Vec<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        self.neighbors_exact_within(point, max_results, f64::INFINITY, box_at)
    }

    /// Exact nearest neighbors within `max_distance` over caller-owned f64 boxes.
    pub fn neighbors_exact_within<F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        box_at: F,
    ) -> Vec<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        let mut results = Vec::new();
        self.neighbors_exact_into(point, max_results, max_distance, box_at, &mut results);
        results
    }

    /// Exact nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_exact_into<F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        box_at: F,
        results: &mut Vec<usize>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        let mut frontier = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        let mut best = BinaryHeap::with_capacity(max_results);
        self.collect_neighbors_refined_with_queue(
            point,
            max_results,
            max_distance,
            box_at,
            results,
            &mut frontier,
            &mut best,
        );
    }

    /// Exact nearest-neighbor search with reusable result and queue buffers.
    pub fn neighbors_exact_with<'b, F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        box_at: F,
        workspace: &'b mut NeighborWorkspace,
    ) -> &'b [usize]
    where
        F: FnMut(usize) -> Box2D,
    {
        self.collect_neighbors_refined_with_queue(
            point,
            max_results,
            max_distance,
            box_at,
            &mut workspace.results,
            &mut workspace.queue,
            &mut workspace.exact_queue,
        );
        &workspace.results
    }

    /// Visit rounded-box nearest items in nondecreasing squared-distance order.
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

    fn collect_neighbors_with_queue(
        &self,
        point: Point2D,
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

    #[allow(clippy::too_many_arguments)]
    fn collect_neighbors_refined_with_queue<F>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        mut box_at: F,
        results: &mut Vec<usize>,
        frontier: &mut BinaryHeap<NeighborState>,
        best: &mut BinaryHeap<ExactNeighborState>,
    ) where
        F: FnMut(usize) -> Box2D,
    {
        results.clear();
        frontier.clear();
        best.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if self.num_items == 0 || max_results == 0 {
            return;
        }

        let root = self.num_nodes - 1;
        let root_dist = self.distance_squared_to(root, point);
        if root_dist > max_dist_sq {
            return;
        }
        frontier.push(NeighborState::new(root, false, root_dist));

        let mut cutoff = max_dist_sq;
        while let Some(state) = frontier.pop() {
            if state.dist > cutoff {
                break;
            }
            if state.is_leaf {
                let exact_dist = distance_squared_to_box(point, box_at(state.index));
                if exact_dist <= max_dist_sq {
                    push_exact_neighbor(best, max_results, state.index, exact_dist);
                    if best.len() == max_results {
                        cutoff = best.peek().map_or(max_dist_sq, |worst| worst.dist);
                    }
                }
                continue;
            }

            let upper = self.upper_bound_level(state.index);
            let end = (state.index + self.node_size).min(self.level_bound_unchecked(upper));
            let is_leaf = state.index < self.num_items;
            for pos in state.index..end {
                let dist = self.distance_squared_to(pos, point);
                if dist <= cutoff {
                    frontier.push(NeighborState::new(self.index_at(pos), is_leaf, dist));
                }
            }
        }

        write_exact_results(results, best);
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
}

/// Compact **scalar** f32-storage 2D index: the same outward-rounded `f32` boxes
/// as [`SimdIndex2DF32`] (half the box memory of [`Index2D`](crate::Index2D)),
/// queried without SIMD. Built natively in `f32` (no transient `f64` tree).
/// Range and ray results are a conservative **superset** of the f64 index (every
/// exact hit, plus a few near-boundary false positives from outward rounding).
pub struct Index2DF32 {
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    min_xs: Vec<f32>,
    min_ys: Vec<f32>,
    max_xs: Vec<f32>,
    max_ys: Vec<f32>,
    indices: Vec<usize>,
}

impl Index2DF32 {
    /// Build a compact index over each triangle's bounding box.
    pub fn from_triangles<T: Triangle2>(triangles: &[T]) -> Result<Self, BuildError> {
        let mut builder = crate::Index2DBuilder::new(triangles.len());
        for t in triangles {
            builder.add(t.aabb());
        }
        builder.finish_f32()
    }

    /// Serialize the index (f32 box records) into a new buffer. To attach a
    /// payload (e.g. triangles), use [`serialize`](Self::serialize).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.serialize()
            .to_bytes_into(&mut out)
            .expect("index-only serialization cannot fail");
        out
    }

    /// Start a serialization builder: attach a per-item payload (opaque blobs,
    /// fixed-width [`records`](Serializer2DF32::records), or
    /// [`triangles`](Serializer2DF32::triangles)) and descriptive metadata. The
    /// f32-box counterpart of [`Index2D::serialize`](crate::Index2D::serialize);
    /// the bytes stream through [`StreamIndex2DF32`](crate::StreamIndex2DF32).
    pub fn serialize(&self) -> Serializer2DF32<'_> {
        Serializer2DF32::new(self)
    }

    /// Load a compact f32 index from bytes produced by [`to_bytes`](Self::to_bytes)
    /// or [`serialize`](Self::serialize). Any payload section is ignored (the
    /// index loads box-only); stream the payload with
    /// [`StreamIndex2DF32`](crate::StreamIndex2DF32).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let (parsed, _payload) = parse_index(bytes, 2, 4)?;
        Ok(f32_columns_from_parsed_2(&parsed))
    }

    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.num_items
    }

    /// Whether the index has no items.
    pub fn is_empty(&self) -> bool {
        self.num_items == 0
    }

    /// Packed node size of the index.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Total extent of all indexed items (widened f32 root box), or `None` when
    /// empty.
    pub fn extent(&self) -> Option<Box2D> {
        if self.num_items == 0 {
            None
        } else {
            Some(self.box_at(self.indices.len() - 1))
        }
    }

    #[inline]
    fn box_at(&self, pos: usize) -> Box2D {
        Box2D::new(
            self.min_xs[pos] as f64,
            self.min_ys[pos] as f64,
            self.max_xs[pos] as f64,
            self.max_ys[pos] as f64,
        )
    }

    #[inline]
    fn box_f32_at(&self, pos: usize) -> Box2DF32 {
        Box2DF32 {
            min_x: self.min_xs[pos],
            min_y: self.min_ys[pos],
            max_x: self.max_xs[pos],
            max_y: self.max_ys[pos],
        }
    }

    /// Items whose (outward-rounded) box overlaps `query`. A conservative superset
    /// of [`Index2D::search`](crate::Index2D::search).
    pub fn search(&self, query: Box2D) -> Vec<usize> {
        let q = Box2DF32::from_box2d_inward(query);
        let mut out = Vec::new();
        let _ = self.visit_hits(
            |b| b.overlaps(q),
            |i, _| {
                out.push(i);
                ControlFlow::<()>::Continue(())
            },
        );
        out
    }

    /// Items whose (outward-rounded) box the ray segment crosses. A conservative
    /// superset of [`Index2D::raycast`](crate::Index2D::raycast).
    pub fn raycast(&self, ray: Ray2D) -> Vec<usize> {
        let mut out = Vec::new();
        let _ = self.visit_hits(
            |b| ray.intersects_box(b.widen()),
            |i, _| {
                out.push(i);
                ControlFlow::<()>::Continue(())
            },
        );
        out
    }

    /// Visit each item whose (rounded) box overlaps `query`; return
    /// [`ControlFlow::Break`] from `visitor` to stop early.
    pub fn visit<B>(
        &self,
        query: Box2D,
        mut visitor: impl FnMut(usize) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let q = Box2DF32::from_box2d_inward(query);
        self.visit_hits(|b| b.overlaps(q), |i, _| visitor(i))
    }

    /// Whether any item's (rounded) box overlaps `query` (early-exit).
    pub fn any(&self, query: Box2D) -> bool {
        self.visit(query, |_| ControlFlow::Break(())).is_break()
    }

    /// Some item whose (rounded) box overlaps `query`, or `None`. Traversal order
    /// is unspecified.
    pub fn first(&self, query: Box2D) -> Option<usize> {
        match self.visit(query, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// **Exact** item indices whose caller-owned `f64` box (from `box_at`)
    /// intersects `query`: the conservative f32 boxes prune the tree, then each
    /// candidate is refined against its true box (no near-boundary false
    /// positives, unlike [`search`](Self::search)).
    pub fn search_exact<F: FnMut(usize) -> Box2D>(&self, query: Box2D, box_at: F) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_exact_into(query, box_at, &mut out);
        out
    }

    /// Exact search into a reused buffer (cleared first).
    pub fn search_exact_into<F: FnMut(usize) -> Box2D>(
        &self,
        query: Box2D,
        box_at: F,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let _ = self.visit_exact(query, box_at, |id| {
            out.push(id);
            ControlFlow::<()>::Continue(())
        });
    }

    /// Whether any caller-owned `f64` box intersects `query` (exact, early-exit).
    pub fn any_exact<F: FnMut(usize) -> Box2D>(&self, query: Box2D, box_at: F) -> bool {
        self.visit_exact(query, box_at, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Some item whose caller-owned `f64` box intersects `query` (exact), or `None`.
    pub fn first_exact<F: FnMut(usize) -> Box2D>(&self, query: Box2D, box_at: F) -> Option<usize> {
        match self.visit_exact(query, box_at, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    /// Visit each item whose caller-owned `f64` box (from `box_at`) intersects
    /// `query` (exact); the f32 boxes prune the descent, `box_at` refines each
    /// candidate. Return [`ControlFlow::Break`] from `visitor` to stop early.
    pub fn visit_exact<B, BF, VF>(
        &self,
        query: Box2D,
        mut box_at: BF,
        mut visitor: VF,
    ) -> ControlFlow<B>
    where
        BF: FnMut(usize) -> Box2D,
        VF: FnMut(usize) -> ControlFlow<B>,
    {
        let q = Box2DF32::from_box2d_inward(query);
        self.visit_hits(
            |b| b.overlaps(q),
            |id, stored| {
                if stored.definitely_overlaps_exact(query) || box_at(id).overlaps(query) {
                    visitor(id)
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
    }

    /// Shared stack descent: call `visitor` for each leaf item whose stored f32 box
    /// passes `hit`, recursing into internal nodes that pass. Stops early when
    /// `visitor` returns [`ControlFlow::Break`].
    fn visit_hits<B>(
        &self,
        hit: impl Fn(Box2DF32) -> bool,
        mut visitor: impl FnMut(usize, Box2DF32) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        if self.indices.is_empty() {
            return ControlFlow::Continue(());
        }
        let mut stack: Vec<usize> = Vec::new();
        let mut node_index = self.indices.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                let b = self.box_f32_at(pos);
                if !hit(b) {
                    continue;
                }
                let index = self.indices[pos];
                if is_leaf {
                    visitor(index, b)?;
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
}

/// Serialization builder for [`Index2DF32`], created by
/// [`Index2DF32::serialize`]. Writes f32 box records plus an optional per-item
/// payload and descriptive metadata; the f32-box counterpart of
/// [`Serializer2D`](crate::Serializer2D).
pub struct Serializer2DF32<'a> {
    index: &'a Index2DF32,
    payloads: Option<Vec<&'a [u8]>>,
    record_stride: Option<u32>,
    interleaved: bool,
    meta: MetaFields<'a>,
}

impl<'a> Serializer2DF32<'a> {
    fn new(index: &'a Index2DF32) -> Self {
        Self {
            index,
            payloads: None,
            record_stride: None,
            interleaved: false,
            meta: MetaFields::default(),
        }
    }

    /// Attach one opaque payload blob per item, in item order.
    pub fn payloads<P: AsRef<[u8]>>(mut self, payloads: &'a [P]) -> Self {
        self.payloads = Some(payloads.iter().map(|p| p.as_ref()).collect());
        self
    }

    /// Use the streaming-tuned interleaved node layout, so
    /// [`StreamIndex2DF32`](crate::StreamIndex2DF32) fetches each level in one
    /// coalesced read instead of two. Same file size.
    #[cfg(feature = "stream")]
    pub fn interleaved(mut self) -> Self {
        self.interleaved = true;
        self
    }

    /// Attach a fixed-width payload: `flat` is `num_items * stride` bytes (one
    /// `stride`-byte record per item). See
    /// [`Serializer2D::records`](crate::Serializer2D::records).
    pub fn records(mut self, stride: usize, flat: &'a [u8]) -> Self {
        self.record_stride = Some(stride as u32);
        self.payloads = Some(if stride == 0 {
            Vec::new()
        } else {
            flat.chunks_exact(stride).collect()
        });
        self
    }

    /// Attach a fixed-width triangle payload, one per item. A compact mesh that
    /// streams through [`StreamIndex2DF32`](crate::StreamIndex2DF32).
    pub fn triangles<T: Triangle2>(self, triangles: &'a [T]) -> Self {
        let bytes = records_as_bytes(triangles);
        self.records(T::STRIDE, bytes)
    }

    /// Set the coordinate reference system identifier (opaque, e.g. `"EPSG:4326"`).
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
        let record_stride = self.record_stride;
        let interleaved = self.interleaved;
        write_index_container(
            out,
            2,
            4,
            interleaved,
            idx.num_items,
            idx.min_xs.len(),
            idx.node_size,
            |bytes| {
                #[cfg(feature = "stream")]
                if interleaved {
                    bytes.write_interleaved_f32_2d(
                        &idx.min_xs,
                        &idx.min_ys,
                        &idx.max_xs,
                        &idx.max_ys,
                        &idx.indices,
                    );
                    return;
                }
                bytes.write_soa_boxes_f32_2d(&idx.min_xs, &idx.min_ys, &idx.max_xs, &idx.max_ys);
                bytes.write_usize_slice_as_u64(&idx.indices);
            },
            self.payloads.as_deref(),
            record_stride,
            &idx.indices[..idx.num_items],
            &self.meta,
        )
    }
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use std::ops::ControlFlow;

    use crate::{Box2D, Index2DBuilder, Point2D, SimdIndex2DF32, SimdIndex2DF32View};

    fn build(boxes: &[Box2D]) -> SimdIndex2DF32 {
        let mut b = Index2DBuilder::new(boxes.len()).node_size(4);
        for &x in boxes {
            b.add(x);
        }
        b.finish_simd_f32().unwrap()
    }

    #[test]
    fn empty_index_returns_nothing() {
        let index = Index2DBuilder::new(0).finish_simd_f32().unwrap();
        assert_eq!(index.num_items(), 0);
        assert!(index.search(Box2D::new(0.0, 0.0, 1.0, 1.0)).is_empty());
    }

    #[test]
    fn single_box_is_found() {
        let index = build(&[Box2D::new(0.0, 0.0, 1.0, 1.0)]);
        assert!(index.search(Box2D::new(0.5, 0.5, 0.5, 0.5)).contains(&0));
    }

    #[test]
    fn rounded_search_keeps_exact_hits() {
        let boxes: Vec<Box2D> = (0..200)
            .map(|i| {
                let x = (i as f64 * 7.0) % 1000.0;
                let y = (i as f64 * 13.0) % 1000.0;
                Box2D::new(x, y, x + 3.0, y + 3.0)
            })
            .collect();

        let f32_index = build(&boxes);
        let mut f64_builder = Index2DBuilder::new(boxes.len()).node_size(4);
        for &x in &boxes {
            f64_builder.add(x);
        }
        let f64_index = f64_builder.finish_simd().unwrap();

        for qi in 0..50 {
            let qx = (qi as f64 * 19.0) % 1000.0;
            let qy = (qi as f64 * 23.0) % 1000.0;
            let query = Box2D::new(qx, qy, qx + 30.0, qy + 30.0);

            let rounded_hits = f32_index.search(query);
            let truth = f64_index.search(query);
            for hit in &truth {
                assert!(rounded_hits.contains(hit));
            }
            let mut exact = f32_index.search_exact(query, |i| boxes[i]);
            exact.sort_unstable();
            let mut truth = truth;
            truth.sort_unstable();
            assert_eq!(exact, truth);
        }
    }

    #[test]
    fn large_window_search_keeps_exact_hits() {
        // Large windows that fully contain whole subtrees exercise the conservative
        // covered-range fast path on the owned index and the view.
        let boxes: Vec<Box2D> = (0..400)
            .map(|i| {
                let x = (i as f64 * 7.0) % 1000.0;
                let y = (i as f64 * 13.0) % 1000.0;
                Box2D::new(x, y, x + 3.0, y + 3.0)
            })
            .collect();

        let f32_index = build(&boxes);
        let bytes = f32_index.to_bytes();
        let view = SimdIndex2DF32View::from_bytes(&bytes).unwrap();
        let mut f64_builder = Index2DBuilder::new(boxes.len()).node_size(4);
        for &x in &boxes {
            f64_builder.add(x);
        }
        let f64_index = f64_builder.finish_simd().unwrap();

        for size in [40.0, 200.0, 600.0, 2_000.0] {
            for qi in 0..30 {
                let qx = (qi as f64 * 19.0) % 1000.0;
                let qy = (qi as f64 * 23.0) % 1000.0;
                let query = Box2D::new(qx, qy, qx + size, qy + size);

                let mut truth = f64_index.search(query);
                truth.sort_unstable();

                // Conservative paths must never drop a true overlap.
                let rounded = f32_index.search(query);
                let view_rounded = view.search(query);
                for hit in &truth {
                    assert!(rounded.contains(hit), "owned dropped {hit}");
                    assert!(view_rounded.contains(hit), "view dropped {hit}");
                }

                // Exact refinement must reproduce the f64 truth exactly.
                let mut exact = f32_index.search_exact(query, |i| boxes[i]);
                exact.sort_unstable();
                assert_eq!(exact, truth);
            }
        }

        // Full-extent query returns every item through the contains shortcut.
        let extent = f32_index.extent().unwrap();
        assert_eq!(f32_index.search(extent).len(), boxes.len());
        assert_eq!(view.search(extent).len(), boxes.len());
    }

    #[test]
    fn exact_search_filters_extra_f32_boundary_hit() {
        let query = Box2D::new(1.0, 0.0, 1.0, 0.0);
        let boxes = [
            Box2D::new(1.0 + 1e-8, 0.0, 1.0 + 1e-8, 0.0),
            Box2D::new(1.0, 0.0, 1.0, 0.0),
        ];
        let index = build(&boxes);
        let bytes = index.to_bytes();
        let view = SimdIndex2DF32View::from_bytes(&bytes).unwrap();

        let mut rounded_hits = index.search(query);
        rounded_hits.sort_unstable();
        assert_eq!(rounded_hits, vec![0, 1]);

        assert_eq!(index.search_exact(query, |i| boxes[i]), vec![1]);
        let mut out = vec![usize::MAX];
        index.search_exact_into(query, |i| boxes[i], &mut out);
        assert_eq!(out, vec![1]);

        let mut workspace = crate::SearchWorkspace::new();
        assert_eq!(
            index.search_exact_with(query, |i| boxes[i], &mut workspace),
            &[1][..]
        );
        assert!(index.any_exact(query, |i| boxes[i]));
        assert_eq!(index.first_exact(query, |i| boxes[i]), Some(1));

        let mut visited = Vec::new();
        let _: ControlFlow<()> = index.visit_exact(
            query,
            |i| boxes[i],
            |i| {
                visited.push(i);
                ControlFlow::Continue(())
            },
        );
        assert_eq!(visited, vec![1]);

        assert_eq!(view.search_exact(query, |i| boxes[i]), vec![1]);
        assert_eq!(
            view.search_exact_with(query, |i| boxes[i], &mut workspace),
            &[1][..]
        );
        assert!(view.any_exact(query, |i| boxes[i]));
        assert_eq!(view.first_exact(query, |i| boxes[i]), Some(1));
    }

    #[test]
    fn exact_search_uses_certified_hits_without_source_lookup() {
        let boxes = [Box2D::new(0.0, 0.0, 10.0, 10.0)];
        let index = build(&boxes);
        let bytes = index.to_bytes();
        let view = SimdIndex2DF32View::from_bytes(&bytes).unwrap();
        let query = Box2D::new(5.0, 5.0, 6.0, 6.0);
        let panic_box_at = |_| panic!("source box lookup should not be needed");

        assert_eq!(index.search_exact(query, panic_box_at), vec![0]);
        assert!(index.any_exact(query, panic_box_at));
        assert_eq!(index.first_exact(query, panic_box_at), Some(0));

        let mut visited = Vec::new();
        let _: ControlFlow<()> = index.visit_exact(query, panic_box_at, |i| {
            visited.push(i);
            ControlFlow::Continue(())
        });
        assert_eq!(visited, vec![0]);

        assert_eq!(view.search_exact(query, panic_box_at), vec![0]);
        assert!(view.any_exact(query, panic_box_at));
        assert_eq!(view.first_exact(query, panic_box_at), Some(0));
    }

    #[test]
    fn persistence_round_trip() {
        let boxes: Vec<Box2D> = (0..100)
            .map(|i| {
                let x = (i as f64 * 11.0) % 500.0;
                let y = (i as f64 * 17.0) % 500.0;
                Box2D::new(x, y, x + 2.0, y + 2.0)
            })
            .collect();
        let index = build(&boxes);
        let bytes = index.to_bytes();
        let loaded = SimdIndex2DF32::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.num_items(), index.num_items());
        let query = Box2D::new(40.0, 40.0, 120.0, 120.0);
        let mut a = index.search(query);
        let mut b = loaded.search(query);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn rounded_neighbors_find_nearest_box() {
        let boxes = [
            Box2D::new(0.0, 0.0, 1.0, 1.0),
            Box2D::new(100.0, 100.0, 101.0, 101.0),
        ];
        let index = build(&boxes);
        assert_eq!(index.neighbors(Point2D::new(100.5, 100.5), 1), vec![1]);
    }

    #[test]
    fn exact_neighbors_match_f64_index() {
        let boxes: Vec<Box2D> = (0..200)
            .map(|i| {
                let x = (i as f64 * 11.0) % 500.0;
                let y = (i as f64 * 17.0) % 500.0;
                Box2D::new(x, y, x + 2.0, y + 2.0)
            })
            .collect();
        let index = build(&boxes);
        let mut f64_builder = Index2DBuilder::new(boxes.len()).node_size(4);
        for &x in &boxes {
            f64_builder.add(x);
        }
        let f64_index = f64_builder.finish_simd().unwrap();

        for qi in 0..20 {
            let point = Point2D::new((qi as f64 * 13.0) % 500.0, (qi as f64 * 19.0) % 500.0);
            assert_eq!(
                index.neighbors_exact(point, 5, |i| boxes[i]),
                f64_index.neighbors(point, 5)
            );

            let mut workspace = crate::NeighborWorkspace::new();
            assert_eq!(
                index.neighbors_exact_with(point, 5, f64::INFINITY, |i| boxes[i], &mut workspace),
                f64_index.neighbors(point, 5)
            );
        }
    }

    #[test]
    fn exact_neighbors_use_exact_boxes_at_f32_boundaries() {
        let point = Point2D::new(1.0, 0.0);
        let boxes = [
            Box2D::new(1.0 + 1e-8, 0.0, 1.0 + 1e-8, 0.0),
            Box2D::new(1.0, 0.0, 1.0, 0.0),
        ];
        let index = build(&boxes);
        let bytes = index.to_bytes();
        let view = SimdIndex2DF32View::from_bytes(&bytes).unwrap();

        assert_eq!(index.neighbors_exact(point, 1, |i| boxes[i]), vec![1]);
        assert_eq!(view.neighbors_exact(point, 1, |i| boxes[i]), vec![1]);
    }

    #[test]
    fn exact_neighbors_clear_results_on_empty_or_invalid_limits() {
        let boxes = [Box2D::new(0.0, 0.0, 1.0, 1.0)];
        let index = build(&boxes);
        let point = Point2D::new(0.5, 0.5);

        let mut out = vec![usize::MAX];
        index.neighbors_exact_into(
            point,
            0,
            f64::INFINITY,
            |_| panic!("source box lookup should not be needed"),
            &mut out,
        );
        assert!(out.is_empty());

        out.push(usize::MAX);
        index.neighbors_exact_into(
            point,
            1,
            f64::NAN,
            |_| panic!("source box lookup should not be needed"),
            &mut out,
        );
        assert!(out.is_empty());

        let mut workspace = crate::NeighborWorkspace::with_capacity(1, 1);
        assert!(
            index
                .neighbors_exact_with(
                    point,
                    1,
                    -1.0,
                    |_| panic!("source box lookup should not be needed"),
                    &mut workspace,
                )
                .is_empty()
        );
        assert!(workspace.results().is_empty());
    }

    #[test]
    fn view_matches_owned() {
        let boxes: Vec<Box2D> = (0..120)
            .map(|i| {
                let x = (i as f64 * 11.0) % 500.0;
                let y = (i as f64 * 17.0) % 500.0;
                Box2D::new(x, y, x + 2.0, y + 2.0)
            })
            .collect();
        let index = build(&boxes);
        let bytes = index.to_bytes();
        let view = SimdIndex2DF32View::from_bytes(&bytes).unwrap();

        let query = Box2D::new(40.0, 40.0, 120.0, 120.0);
        let mut a = index.search(query);
        let mut b = view.search(query);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);

        let mut a = index.search_exact(query, |i| boxes[i]);
        let mut b = view.search_exact(query, |i| boxes[i]);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(
            index.neighbors_exact(Point2D::new(60.0, 60.0), 5, |i| boxes[i]),
            view.neighbors_exact(Point2D::new(60.0, 60.0), 5, |i| boxes[i])
        );
    }
}
