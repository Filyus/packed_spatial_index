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
    f32_storage::{Box2DF32, F32Columns2D, columns2d_from_parsed},
    geometry::Box2D,
    persistence::{LoadError, parse_index},
    ray::Ray2D,
    sort2d::{SortKeyContext, encode_sort_by_key},
    tree::{TreeLayout, try_compute_tree_layout},
    triangle::Triangle2,
};

mod serializer;
#[cfg(feature = "simd")]
mod simd_serialization;
pub use serializer::Serializer2DF32;

// Point kNN over the owned f32 indexes (scalar `Index2DF32` + SIMD
// `SimdIndex2DF32`) needs these whenever `f32-storage` is enabled.
#[cfg(feature = "simd")]
use crate::f32_storage::{CONTAINED_FLAG, LEVEL_MASK, encode_level};
#[cfg(feature = "simd")]
use crate::persistence::read_u64_le_unchecked;
use crate::{geometry::Point2D, neighbors::NeighborWorkspace};

// Imports used only by the SIMD query frontend (SimdIndex2DF32 + its view).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::leftpack::leftpack4;
#[cfg(feature = "simd")]
use crate::{config::DEFAULT_SEARCH_STACK_CAPACITY, traversal::SearchWorkspace};
use std::ops::ControlFlow;

/// Build the canonical scalar f32 storage from already-decoded SoA columns.
fn index2d_from_columns(columns: F32Columns2D) -> Index2DF32 {
    Index2DF32 {
        node_size: columns.node_size,
        num_items: columns.num_items,
        level_bounds: columns.level_bounds,
        min_xs: columns.min_xs,
        min_ys: columns.min_ys,
        max_xs: columns.max_xs,
        max_ys: columns.max_ys,
        indices: columns.indices,
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

    if num_items == 0 {
        return Ok(Index2DF32 {
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
        return Ok(build_single_node_soa_f32(
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

// Point kNN is a scalar priority-queue descent (not SIMD); the heavy traversal
// lives in `crate::neighbors`. The scalar `Index2DF32` and SIMD `SimdIndex2DF32`
// are field-identical, so each carries the same thin adapter into it.
impl Index2DF32 {
    /// Up to `max_results` item indices nearest to `point` by rounded f32 boxes.
    ///
    /// Distances use the outward-rounded f32 boxes, so they are lower bounds of
    /// the exact f64 distances. Use [`neighbors_exact`](Self::neighbors_exact)
    /// when you need exact nearest neighbors over your original f64 boxes.
    pub fn neighbors(&self, point: Point2D, max_results: usize) -> Vec<usize> {
        crate::neighbors::point_neighbors(self, point, max_results)
    }

    /// Up to `max_results` rounded-box nearest items within `max_distance`.
    pub fn neighbors_within(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        crate::neighbors::point_neighbors_within(self, point, max_results, max_distance)
    }

    /// Rounded-box nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_into(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
    ) {
        crate::neighbors::point_neighbors_into(self, point, max_results, max_distance, results);
    }

    /// Rounded-box nearest-neighbor search with reusable result and queue buffers.
    pub fn neighbors_with<'a>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        workspace: &'a mut NeighborWorkspace,
    ) -> &'a [usize] {
        crate::neighbors::point_neighbors_with(self, point, max_results, max_distance, workspace)
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
    /// let index = builder.finish_f32()?;
    ///
    /// let nearest = index.neighbors_exact(Point2D::new(1.0, 0.0), 1, |i| boxes[i]);
    /// assert_eq!(nearest, vec![1]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn neighbors_exact<F>(&self, point: Point2D, max_results: usize, box_at: F) -> Vec<usize>
    where
        F: FnMut(usize) -> Box2D,
    {
        crate::neighbors::point_neighbors_exact(self, point, max_results, box_at)
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
        crate::neighbors::point_neighbors_exact_within(
            self,
            point,
            max_results,
            max_distance,
            box_at,
        )
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
        crate::neighbors::point_neighbors_exact_into(
            self,
            point,
            max_results,
            max_distance,
            box_at,
            results,
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
        crate::neighbors::point_neighbors_exact_with(
            self,
            point,
            max_results,
            max_distance,
            box_at,
            workspace,
        )
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
        crate::neighbors::visit_point_neighbors(self, point, max_distance, &mut visitor)
    }
}

impl crate::neighbors::PointKnn for Index2DF32 {
    type Point = Point2D;
    type ExactBox = Box2D;

    #[inline]
    fn knn_num_nodes(&self) -> usize {
        self.min_xs.len()
    }

    #[inline]
    fn knn_num_items(&self) -> usize {
        self.num_items
    }

    #[inline]
    fn knn_node_size(&self) -> usize {
        self.node_size
    }

    #[inline]
    fn knn_point_is_valid(point: Point2D) -> bool {
        point.x.is_finite() && point.y.is_finite()
    }

    #[inline]
    fn knn_level_end(&self, node: usize) -> usize {
        crate::neighbors::level_end_of(&self.level_bounds, node)
    }

    #[inline]
    fn knn_index_at(&self, pos: usize) -> usize {
        self.indices[pos]
    }

    #[inline]
    fn knn_distance_squared_to(&self, pos: usize, point: Point2D) -> f64 {
        let dx = axis_distance(point.x, self.min_xs[pos] as f64, self.max_xs[pos] as f64);
        let dy = axis_distance(point.y, self.min_ys[pos] as f64, self.max_ys[pos] as f64);
        dx * dx + dy * dy
    }

    #[inline]
    fn exact_distance_squared(point: Point2D, bbox: Box2D) -> f64 {
        bbox.distance_squared_to(point)
    }
}

// Same thin point-kNN adapter on the field-identical SIMD index (see above).
#[cfg(feature = "simd")]
impl SimdIndex2DF32 {
    /// Up to `max_results` item indices nearest to `point` by rounded f32 boxes.
    ///
    /// Distances use the outward-rounded f32 boxes, so they are lower bounds of
    /// the exact f64 distances. Use [`neighbors_exact`](Self::neighbors_exact)
    /// when you need exact nearest neighbors over your original f64 boxes.
    pub fn neighbors(&self, point: Point2D, max_results: usize) -> Vec<usize> {
        crate::neighbors::point_neighbors(self, point, max_results)
    }

    /// Up to `max_results` rounded-box nearest items within `max_distance`.
    pub fn neighbors_within(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        crate::neighbors::point_neighbors_within(self, point, max_results, max_distance)
    }

    /// Rounded-box nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_into(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
    ) {
        crate::neighbors::point_neighbors_into(self, point, max_results, max_distance, results);
    }

    /// Rounded-box nearest-neighbor search with reusable result and queue buffers.
    pub fn neighbors_with<'a>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        workspace: &'a mut NeighborWorkspace,
    ) -> &'a [usize] {
        crate::neighbors::point_neighbors_with(self, point, max_results, max_distance, workspace)
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
        crate::neighbors::point_neighbors_exact(self, point, max_results, box_at)
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
        crate::neighbors::point_neighbors_exact_within(
            self,
            point,
            max_results,
            max_distance,
            box_at,
        )
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
        crate::neighbors::point_neighbors_exact_into(
            self,
            point,
            max_results,
            max_distance,
            box_at,
            results,
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
        crate::neighbors::point_neighbors_exact_with(
            self,
            point,
            max_results,
            max_distance,
            box_at,
            workspace,
        )
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
        crate::neighbors::visit_point_neighbors(self, point, max_distance, &mut visitor)
    }
}

#[cfg(feature = "simd")]
impl crate::neighbors::PointKnn for SimdIndex2DF32 {
    type Point = Point2D;
    type ExactBox = Box2D;

    #[inline]
    fn knn_num_nodes(&self) -> usize {
        self.min_xs.len()
    }

    #[inline]
    fn knn_num_items(&self) -> usize {
        self.num_items
    }

    #[inline]
    fn knn_node_size(&self) -> usize {
        self.node_size
    }

    #[inline]
    fn knn_point_is_valid(point: Point2D) -> bool {
        point.x.is_finite() && point.y.is_finite()
    }

    #[inline]
    fn knn_level_end(&self, node: usize) -> usize {
        crate::neighbors::level_end_of(&self.level_bounds, node)
    }

    #[inline]
    fn knn_index_at(&self, pos: usize) -> usize {
        self.indices[pos]
    }

    #[inline]
    fn knn_distance_squared_to(&self, pos: usize, point: Point2D) -> f64 {
        let dx = axis_distance(point.x, self.min_xs[pos] as f64, self.max_xs[pos] as f64);
        let dy = axis_distance(point.y, self.min_ys[pos] as f64, self.max_ys[pos] as f64);
        dx * dx + dy * dy
    }

    #[inline]
    fn exact_distance_squared(point: Point2D, bbox: Box2D) -> f64 {
        bbox.distance_squared_to(point)
    }
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
        let q = Box2DF32::from_box2d_inward(query);
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: selected only after checking avx512f availability.
                unsafe { self.search_avx512(q, out, stack) };
                return;
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: selected only after checking avx2 availability.
                unsafe { self.search_avx2(q, out, stack) };
                return;
            }
        }
        self.search_wide(q, out, stack);
    }

    /// Force the AVX2 search path (doc-hidden; for benchmarks/tests — the public
    /// `search` auto-dispatches AVX-512 → AVX2 → wide).
    #[doc(hidden)]
    pub fn search_avx2_into(&self, query: Box2D, out: &mut Vec<usize>) {
        let q = Box2DF32::from_box2d_inward(query);
        let mut stack = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the avx2 feature check.
                unsafe { self.search_avx2(q, out, &mut stack) };
                return;
            }
        }
        self.search_wide(q, out, &mut stack);
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
        let q = Box2DF32::from_box2d_inward(query);
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

    /// AVX2 path: 8 boxes per step. Leaf results use the AVX2 left-pack (applied
    /// to each 4-lane half of the 8-bit mask), since AVX2 lacks `VPCOMPRESSQ`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_avx2(&self, q: Box2DF32, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
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
        let qmxx_v = _mm256_set1_ps(q.max_x);
        let qmnx_v = _mm256_set1_ps(q.min_x);
        let qmxy_v = _mm256_set1_ps(q.max_y);
        let qmny_v = _mm256_set1_ps(q.min_y);

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
                    out.reserve(end - node_index + 8);
                }
                let mut pos = node_index;
                while pos + 8 <= end {
                    // SAFETY: `pos + 8 <= end`, and `end` is bounded by the array length.
                    let (mnx, mxx, mny, mxy) = unsafe {
                        (
                            _mm256_loadu_ps(self.min_xs.as_ptr().add(pos)),
                            _mm256_loadu_ps(self.max_xs.as_ptr().add(pos)),
                            _mm256_loadu_ps(self.min_ys.as_ptr().add(pos)),
                            _mm256_loadu_ps(self.max_ys.as_ptr().add(pos)),
                        )
                    };
                    let overlap = _mm256_and_ps(
                        _mm256_and_ps(
                            _mm256_cmp_ps::<_CMP_LE_OQ>(mnx, qmxx_v),
                            _mm256_cmp_ps::<_CMP_GE_OQ>(mxx, qmnx_v),
                        ),
                        _mm256_and_ps(
                            _mm256_cmp_ps::<_CMP_LE_OQ>(mny, qmxy_v),
                            _mm256_cmp_ps::<_CMP_GE_OQ>(mxy, qmny_v),
                        ),
                    );
                    let mut bits = _mm256_movemask_ps(overlap) as usize;
                    if is_leaf {
                        if bits != 0 {
                            // Left-pack each 4-lane half of the 8-bit mask. SAFETY:
                            // `pos + 8 <= end <= indices.len()`; `out` reserved
                            // `end - node_index + 8` slack.
                            unsafe {
                                let lo = leftpack4(
                                    self.indices.as_ptr().add(pos),
                                    bits as u32,
                                    out.as_mut_ptr().add(out.len()),
                                );
                                out.set_len(out.len() + lo);
                                let hi = leftpack4(
                                    self.indices.as_ptr().add(pos + 4),
                                    (bits >> 4) as u32,
                                    out.as_mut_ptr().add(out.len()),
                                );
                                out.set_len(out.len() + hi);
                            }
                        }
                    } else {
                        let contains = _mm256_and_ps(
                            _mm256_and_ps(
                                _mm256_cmp_ps::<_CMP_GE_OQ>(mnx, qmnx_v),
                                _mm256_cmp_ps::<_CMP_LE_OQ>(mxx, qmxx_v),
                            ),
                            _mm256_and_ps(
                                _mm256_cmp_ps::<_CMP_GE_OQ>(mny, qmny_v),
                                _mm256_cmp_ps::<_CMP_LE_OQ>(mxy, qmxy_v),
                            ),
                        );
                        let cbits = _mm256_movemask_ps(contains) as usize;
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
                // Reserve the whole node's worth of results up front so the
                // compress-store below writes through a stable base pointer.
                if is_leaf {
                    out.reserve(end - node_index);
                }
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
                        // VPCOMPRESSQ over both 8-lane halves of the u16 mask
                        // (indices are u64), guarded so an empty chunk pays
                        // nothing — the two-op cost otherwise loses on sparse
                        // leaves. Capacity reserved above.
                        if bits != 0 {
                            // SAFETY: `pos + 16 <= end <= indices.len()`; `out` has
                            // `end - node_index` slack reserved.
                            unsafe {
                                let base = out.as_mut_ptr();
                                let mut len = out.len();
                                let lo = bits as u8;
                                let hi = (bits >> 8) as u8;
                                let vlo = _mm512_loadu_epi64(
                                    self.indices.as_ptr().add(pos) as *const i64
                                );
                                _mm512_mask_compressstoreu_epi64(
                                    base.add(len) as *mut i64,
                                    lo,
                                    vlo,
                                );
                                len += lo.count_ones() as usize;
                                let vhi = _mm512_loadu_epi64(
                                    self.indices.as_ptr().add(pos + 8) as *const i64
                                );
                                _mm512_mask_compressstoreu_epi64(
                                    base.add(len) as *mut i64,
                                    hi,
                                    vhi,
                                );
                                len += hi.count_ones() as usize;
                                out.set_len(len);
                            }
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
        Box2DF32::from_soa(&self.min_xs, &self.min_ys, &self.max_xs, &self.max_ys, pos)
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
            let stored = self.box_f32_at(pos);
            if stored.overlaps_exact_or_refined(query, || box_at(index)) {
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
        let q = Box2DF32::from_box2d_inward(query);
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
        let q = Box2DF32::from_box2d_inward(query);
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
                    let stored = self.box_f32_at(pos);
                    if stored.overlaps_exact_or_refined(query, || box_at(index)) {
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
        self.box_f32_at(pos).widen()
    }

    #[inline]
    fn box_f32_at(&self, pos: usize) -> Box2DF32 {
        Box2DF32::read_tree(self.entries, pos)
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
        let query = Box2DF32::from_box2d_inward(query);
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
        let rounded_query = Box2DF32::from_box2d_inward(query);
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
                    if stored.overlaps_exact_or_refined(query, || box_at(index)) {
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
        crate::neighbors::point_neighbors(self, point, max_results)
    }

    /// Up to `max_results` rounded-box nearest items within `max_distance`.
    pub fn neighbors_within(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
    ) -> Vec<usize> {
        crate::neighbors::point_neighbors_within(self, point, max_results, max_distance)
    }

    /// Rounded-box nearest-neighbor search with a reusable result buffer.
    pub fn neighbors_into(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        results: &mut Vec<usize>,
    ) {
        crate::neighbors::point_neighbors_into(self, point, max_results, max_distance, results);
    }

    /// Rounded-box nearest-neighbor search with reusable result and queue buffers.
    pub fn neighbors_with<'b>(
        &self,
        point: Point2D,
        max_results: usize,
        max_distance: f64,
        workspace: &'b mut NeighborWorkspace,
    ) -> &'b [usize] {
        crate::neighbors::point_neighbors_with(self, point, max_results, max_distance, workspace)
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
        crate::neighbors::point_neighbors_exact(self, point, max_results, box_at)
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
        crate::neighbors::point_neighbors_exact_within(
            self,
            point,
            max_results,
            max_distance,
            box_at,
        )
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
        crate::neighbors::point_neighbors_exact_into(
            self,
            point,
            max_results,
            max_distance,
            box_at,
            results,
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
        crate::neighbors::point_neighbors_exact_with(
            self,
            point,
            max_results,
            max_distance,
            box_at,
            workspace,
        )
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
        crate::neighbors::visit_point_neighbors(self, point, max_distance, &mut visitor)
    }
}

#[cfg(feature = "simd")]
impl crate::neighbors::PointKnn for SimdIndex2DF32View<'_> {
    type Point = Point2D;
    type ExactBox = Box2D;

    #[inline]
    fn knn_num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    fn knn_num_items(&self) -> usize {
        self.num_items
    }

    #[inline]
    fn knn_node_size(&self) -> usize {
        self.node_size
    }

    #[inline]
    fn knn_point_is_valid(point: Point2D) -> bool {
        point.x.is_finite() && point.y.is_finite()
    }

    #[inline]
    fn knn_level_end(&self, node: usize) -> usize {
        self.level_bound_unchecked(self.upper_bound_level(node))
    }

    #[inline]
    fn knn_index_at(&self, pos: usize) -> usize {
        self.index_at(pos)
    }

    #[inline]
    fn knn_distance_squared_to(&self, pos: usize, point: Point2D) -> f64 {
        self.distance_squared_to(pos, point)
    }

    #[inline]
    fn exact_distance_squared(point: Point2D, bbox: Box2D) -> f64 {
        bbox.distance_squared_to(point)
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
        Ok(index2d_from_columns(columns2d_from_parsed(&parsed)))
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
        self.box_f32_at(pos).widen()
    }

    #[inline]
    fn box_f32_at(&self, pos: usize) -> Box2DF32 {
        Box2DF32::from_soa(&self.min_xs, &self.min_ys, &self.max_xs, &self.max_ys, pos)
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
                if stored.overlaps_exact_or_refined(query, || box_at(id)) {
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
