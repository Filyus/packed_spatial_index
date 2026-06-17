use std::collections::BinaryHeap;

use crate::geometry::{Box2D, Box3D, Point2D, Point3D};

/// What a 2D nearest-neighbor traversal measures distance from: a point or a
/// query box (box queries use box-to-box gap distance, `0.0` on overlap).
#[derive(Clone, Copy)]
pub(crate) enum NeighborQuery2D {
    Point(Point2D),
    Box(Box2D),
}

impl NeighborQuery2D {
    #[inline]
    pub(crate) fn distance_squared_to(self, bounds: Box2D) -> f64 {
        match self {
            NeighborQuery2D::Point(point) => bounds.distance_squared_to(point),
            NeighborQuery2D::Box(query) => bounds.distance_squared_to_box(query),
        }
    }
}

/// 3D counterpart of [`NeighborQuery2D`].
#[derive(Clone, Copy)]
pub(crate) enum NeighborQuery3D {
    Point(Point3D),
    Box(Box3D),
}

impl NeighborQuery3D {
    #[inline]
    pub(crate) fn distance_squared_to(self, bounds: Box3D) -> f64 {
        match self {
            NeighborQuery3D::Point(point) => bounds.distance_squared_to(point),
            NeighborQuery3D::Box(query) => bounds.distance_squared_to_box(query),
        }
    }
}

/// Reusable buffers for allocation-free repeated nearest-neighbor searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, NeighborWorkspace, Point2D, Box2D};
///
/// let mut builder = Index2DBuilder::new(2);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// builder.add(Box2D::new(10.0, 10.0, 11.0, 11.0));
/// let index = builder.finish().unwrap();
///
/// let mut workspace = NeighborWorkspace::new();
/// let hits = index.neighbors_with(Point2D::new(0.5, 0.5), 1, f64::INFINITY, &mut workspace);
/// assert_eq!(hits, &[0]);
/// assert_eq!(workspace.results(), &[0]);
/// ```
#[derive(Debug, Default)]
pub struct NeighborWorkspace {
    pub(crate) results: Vec<usize>,
    pub(crate) queue: BinaryHeap<NeighborState>,
    pub(crate) node_queue: BinaryHeap<NeighborNodeState>,
    #[cfg(feature = "f32-storage")]
    pub(crate) exact_queue: BinaryHeap<ExactNeighborState>,
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
            node_queue: BinaryHeap::with_capacity(queue),
            #[cfg(feature = "f32-storage")]
            exact_queue: BinaryHeap::with_capacity(results),
        }
    }

    /// Results from the latest nearest-neighbor search that used this workspace.
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NeighborNodeState {
    pub(crate) index: usize,
    pub(crate) dist: f64,
}

impl NeighborNodeState {
    #[inline]
    pub(crate) fn new(index: usize, dist: f64) -> Self {
        Self { index, dist }
    }
}

impl Eq for NeighborNodeState {}

impl Ord for NeighborNodeState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for NeighborNodeState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "f32-storage")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExactNeighborState {
    pub(crate) index: usize,
    pub(crate) dist: f64,
}

#[cfg(feature = "f32-storage")]
impl ExactNeighborState {
    #[inline]
    pub(crate) fn new(index: usize, dist: f64) -> Self {
        Self { index, dist }
    }
}

#[cfg(feature = "f32-storage")]
impl Eq for ExactNeighborState {}

#[cfg(feature = "f32-storage")]
impl Ord for ExactNeighborState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.index.cmp(&other.index))
    }
}

#[cfg(feature = "f32-storage")]
impl PartialOrd for ExactNeighborState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn max_distance_squared(max_distance: f64) -> Option<f64> {
    if max_distance.is_nan() || max_distance.is_sign_negative() {
        None
    } else {
        Some(max_distance * max_distance)
    }
}

/// f32-only point kNN traversal, shared by the owned `Index2DF32` /
/// `SimdIndex2DF32` macro and the f32 views. Dimension/layout-agnostic: callers
/// pass `dist`/`exact_dist` closures that read their own box storage.
#[cfg(feature = "f32-storage")]
mod f32_knn {
    // The traversal loops index `indices`/box storage by node position, which is
    // also passed to the `dist` closure, so a range loop is the clear form.
    #![allow(clippy::needless_range_loop)]
    use super::{ExactNeighborState, NeighborNodeState, NeighborState, max_distance_squared};
    use std::collections::BinaryHeap;
    use std::ops::ControlFlow;

    /// First tree level whose exclusive end is past `node_index` (the level the node
    /// lives on). `level_bounds` is ascending, so this is a partition point.
    #[inline]
    fn node_level(level_bounds: &[usize], node_index: usize) -> usize {
        level_bounds.partition_point(|&end| end <= node_index)
    }

    /// Keep `best` to the `max_results` smallest exact distances (a bounded max-heap).
    pub(crate) fn push_exact_neighbor(
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

    /// Drain `best` into `results` in nondecreasing (distance, index) order.
    pub(crate) fn write_exact_results(
        results: &mut Vec<usize>,
        best: &mut BinaryHeap<ExactNeighborState>,
    ) {
        let mut ordered: Vec<_> = best.drain().collect();
        ordered.sort_by(|a, b| {
            a.dist
                .total_cmp(&b.dist)
                .then_with(|| a.index.cmp(&b.index))
        });
        results.extend(ordered.into_iter().map(|state| state.index));
    }

    /// Best-first descent collecting up to `max_results` nearest items. `dist(pos)`
    /// is the squared distance from the query to the box at node `pos`. Layout- and
    /// dimension-agnostic: callers supply `dist` reading their own box storage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collect_neighbors(
        num_nodes: usize,
        num_items: usize,
        node_size: usize,
        level_bounds: &[usize],
        indices: &[usize],
        max_results: usize,
        max_distance: f64,
        dist: impl Fn(usize) -> f64,
        results: &mut Vec<usize>,
        queue: &mut BinaryHeap<NeighborState>,
    ) {
        queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if num_items == 0 {
            return;
        }
        let mut node_index = num_nodes - 1;
        loop {
            let end =
                (node_index + node_size).min(level_bounds[node_level(level_bounds, node_index)]);
            let is_leaf = node_index < num_items;
            for pos in node_index..end {
                let d = dist(pos);
                if d > max_dist_sq {
                    continue;
                }
                queue.push(NeighborState::new(indices[pos], is_leaf, d));
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

    /// Best-first search for the single nearest item (`max_results == 1` fast path).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn nearest_one(
        num_nodes: usize,
        num_items: usize,
        node_size: usize,
        level_bounds: &[usize],
        indices: &[usize],
        max_distance: f64,
        dist: impl Fn(usize) -> f64,
        queue: &mut BinaryHeap<NeighborNodeState>,
    ) -> Option<usize> {
        queue.clear();
        let mut best_dist = max_distance_squared(max_distance)?;
        if num_items == 0 {
            return None;
        }
        let mut best_index = None;
        let mut node_index = num_nodes - 1;
        loop {
            let end =
                (node_index + node_size).min(level_bounds[node_level(level_bounds, node_index)]);
            let is_leaf = node_index < num_items;
            for pos in node_index..end {
                let d = dist(pos);
                if d > best_dist {
                    continue;
                }
                if is_leaf {
                    if d == 0.0 {
                        return Some(indices[pos]);
                    }
                    best_dist = d;
                    best_index = Some(indices[pos]);
                } else {
                    queue.push(NeighborNodeState::new(indices[pos], d));
                }
            }
            match queue.pop() {
                Some(state) if state.dist <= best_dist => node_index = state.index,
                _ => return best_index,
            }
        }
    }

    /// Best-first exact kNN: prune the tree by `dist` (lower-bound box distance),
    /// refine each candidate leaf by `exact_dist` (true caller-box distance).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collect_neighbors_refined(
        num_nodes: usize,
        num_items: usize,
        node_size: usize,
        level_bounds: &[usize],
        indices: &[usize],
        max_results: usize,
        max_distance: f64,
        dist: impl Fn(usize) -> f64,
        mut exact_dist: impl FnMut(usize) -> f64,
        results: &mut Vec<usize>,
        frontier: &mut BinaryHeap<NeighborState>,
        best: &mut BinaryHeap<ExactNeighborState>,
    ) {
        results.clear();
        frontier.clear();
        best.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return;
        };
        if num_items == 0 || max_results == 0 {
            return;
        }
        let root = num_nodes - 1;
        let root_dist = dist(root);
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
                let exact = exact_dist(state.index);
                if exact <= max_dist_sq {
                    push_exact_neighbor(best, max_results, state.index, exact);
                    if best.len() == max_results {
                        cutoff = best.peek().map_or(max_dist_sq, |worst| worst.dist);
                    }
                }
                continue;
            }
            let end =
                (state.index + node_size).min(level_bounds[node_level(level_bounds, state.index)]);
            let is_leaf = state.index < num_items;
            for pos in state.index..end {
                let d = dist(pos);
                if d <= cutoff {
                    frontier.push(NeighborState::new(indices[pos], is_leaf, d));
                }
            }
        }
        write_exact_results(results, best);
    }

    /// Visit items in nondecreasing squared-distance order; `visitor` may break early.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn visit_neighbors<B>(
        num_nodes: usize,
        num_items: usize,
        node_size: usize,
        level_bounds: &[usize],
        indices: &[usize],
        max_distance: f64,
        dist: impl Fn(usize) -> f64,
        queue: &mut BinaryHeap<NeighborState>,
        visitor: &mut impl FnMut(usize, f64) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        queue.clear();
        let Some(max_dist_sq) = max_distance_squared(max_distance) else {
            return ControlFlow::Continue(());
        };
        if num_items == 0 {
            return ControlFlow::Continue(());
        }
        let mut node_index = num_nodes - 1;
        loop {
            let end =
                (node_index + node_size).min(level_bounds[node_level(level_bounds, node_index)]);
            let is_leaf = node_index < num_items;
            for pos in node_index..end {
                let d = dist(pos);
                if d > max_dist_sq {
                    continue;
                }
                queue.push(NeighborState::new(indices[pos], is_leaf, d));
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

#[cfg(feature = "f32-storage")]
pub(crate) use f32_knn::*;
