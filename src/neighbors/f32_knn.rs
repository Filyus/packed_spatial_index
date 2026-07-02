//! f32-only point kNN traversal.
//!
//! Shared by the owned `Index*F32`, SIMD `SimdIndex*F32`, and f32 views.
//! Dimension/layout-agnostic: frontends implement `PointKnn`, and the traversal
//! layer still receives plain closures for distances and tree access.

#![allow(clippy::needless_range_loop)]

use super::{
    ExactNeighborState, NeighborNodeState, NeighborState, NeighborWorkspace, max_distance_squared,
};
use crate::config::DEFAULT_NEIGHBOR_QUEUE_CAPACITY;
use crate::neighbors::best_first::{collect_neighbors, nearest_one, visit_neighbors};
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Minimal tree access needed by shared f32 point-kNN adapters.
pub(crate) trait PointKnn {
    type Point: Copy;
    type ExactBox;

    fn knn_num_nodes(&self) -> usize;
    fn knn_num_items(&self) -> usize;
    fn knn_node_size(&self) -> usize;
    fn knn_level_end(&self, node: usize) -> usize;
    fn knn_index_at(&self, pos: usize) -> usize;
    fn knn_point_is_valid(point: Self::Point) -> bool;
    fn knn_distance_squared_to(&self, pos: usize, point: Self::Point) -> f64;
    fn exact_distance_squared(point: Self::Point, bbox: Self::ExactBox) -> f64;
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

pub(crate) fn point_neighbors<T: PointKnn + ?Sized>(
    tree: &T,
    point: T::Point,
    max_results: usize,
) -> Vec<usize> {
    point_neighbors_within(tree, point, max_results, f64::INFINITY)
}

pub(crate) fn point_neighbors_within<T: PointKnn + ?Sized>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
) -> Vec<usize> {
    let mut results = Vec::new();
    point_neighbors_into(tree, point, max_results, max_distance, &mut results);
    results
}

pub(crate) fn point_neighbors_into<T: PointKnn + ?Sized>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
    results: &mut Vec<usize>,
) {
    results.clear();
    if max_results == 0 || !T::knn_point_is_valid(point) {
        return;
    }
    if max_results == 1 {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        if let Some(index) = point_nearest_one(tree, point, max_distance, &mut queue) {
            results.push(index);
        }
        return;
    }

    let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    collect_point_neighbors_with_queue(tree, point, max_results, max_distance, results, &mut queue);
}

pub(crate) fn point_neighbors_with<'a, T: PointKnn + ?Sized>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
    workspace: &'a mut NeighborWorkspace,
) -> &'a [usize] {
    workspace.results.clear();
    if max_results == 0 || !T::knn_point_is_valid(point) {
        workspace.queue.clear();
        workspace.node_queue.clear();
        return &workspace.results;
    }
    if max_results == 1 {
        workspace.queue.clear();
        if let Some(index) = point_nearest_one(tree, point, max_distance, &mut workspace.node_queue)
        {
            workspace.results.push(index);
        }
        return &workspace.results;
    }

    workspace.node_queue.clear();
    collect_point_neighbors_with_queue(
        tree,
        point,
        max_results,
        max_distance,
        &mut workspace.results,
        &mut workspace.queue,
    );
    &workspace.results
}

pub(crate) fn point_neighbors_exact<T, F>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    box_at: F,
) -> Vec<usize>
where
    T: PointKnn + ?Sized,
    F: FnMut(usize) -> T::ExactBox,
{
    point_neighbors_exact_within(tree, point, max_results, f64::INFINITY, box_at)
}

pub(crate) fn point_neighbors_exact_within<T, F>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
    box_at: F,
) -> Vec<usize>
where
    T: PointKnn + ?Sized,
    F: FnMut(usize) -> T::ExactBox,
{
    let mut results = Vec::new();
    point_neighbors_exact_into(tree, point, max_results, max_distance, box_at, &mut results);
    results
}

pub(crate) fn point_neighbors_exact_into<T, F>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
    box_at: F,
    results: &mut Vec<usize>,
) where
    T: PointKnn + ?Sized,
    F: FnMut(usize) -> T::ExactBox,
{
    results.clear();
    if !T::knn_point_is_valid(point) {
        return;
    }
    let mut frontier = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    let mut best = BinaryHeap::with_capacity(max_results);
    collect_point_neighbors_refined_with_queue(
        tree,
        point,
        max_results,
        max_distance,
        box_at,
        results,
        &mut frontier,
        &mut best,
    );
}

pub(crate) fn point_neighbors_exact_with<'a, T, F>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
    box_at: F,
    workspace: &'a mut NeighborWorkspace,
) -> &'a [usize]
where
    T: PointKnn + ?Sized,
    F: FnMut(usize) -> T::ExactBox,
{
    if !T::knn_point_is_valid(point) {
        workspace.results.clear();
        workspace.queue.clear();
        workspace.exact_queue.clear();
        return &workspace.results;
    }
    collect_point_neighbors_refined_with_queue(
        tree,
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

pub(crate) fn visit_point_neighbors<T, B>(
    tree: &T,
    point: T::Point,
    max_distance: f64,
    visitor: &mut impl FnMut(usize, f64) -> ControlFlow<B>,
) -> ControlFlow<B>
where
    T: PointKnn + ?Sized,
{
    if !T::knn_point_is_valid(point) {
        return ControlFlow::Continue(());
    }
    let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    visit_point_neighbors_with_queue(tree, point, max_distance, &mut queue, visitor)
}

fn collect_point_neighbors_with_queue<T: PointKnn + ?Sized>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
    results: &mut Vec<usize>,
    queue: &mut BinaryHeap<NeighborState>,
) {
    collect_neighbors(
        tree.knn_num_nodes(),
        tree.knn_num_items(),
        tree.knn_node_size(),
        |n| tree.knn_level_end(n),
        |p| tree.knn_index_at(p),
        max_results,
        max_distance,
        |pos| tree.knn_distance_squared_to(pos, point),
        results,
        queue,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_point_neighbors_refined_with_queue<T, F>(
    tree: &T,
    point: T::Point,
    max_results: usize,
    max_distance: f64,
    mut box_at: F,
    results: &mut Vec<usize>,
    frontier: &mut BinaryHeap<NeighborState>,
    best: &mut BinaryHeap<ExactNeighborState>,
) where
    T: PointKnn + ?Sized,
    F: FnMut(usize) -> T::ExactBox,
{
    collect_neighbors_refined(
        tree.knn_num_nodes(),
        tree.knn_num_items(),
        tree.knn_node_size(),
        |n| tree.knn_level_end(n),
        |p| tree.knn_index_at(p),
        max_results,
        max_distance,
        |pos| tree.knn_distance_squared_to(pos, point),
        |index| T::exact_distance_squared(point, box_at(index)),
        results,
        frontier,
        best,
    );
}

fn point_nearest_one<T: PointKnn + ?Sized>(
    tree: &T,
    point: T::Point,
    max_distance: f64,
    queue: &mut BinaryHeap<NeighborNodeState>,
) -> Option<usize> {
    nearest_one(
        tree.knn_num_nodes(),
        tree.knn_num_items(),
        tree.knn_node_size(),
        |n| tree.knn_level_end(n),
        |p| tree.knn_index_at(p),
        max_distance,
        |pos| tree.knn_distance_squared_to(pos, point),
        queue,
    )
}

fn visit_point_neighbors_with_queue<T, B>(
    tree: &T,
    point: T::Point,
    max_distance: f64,
    queue: &mut BinaryHeap<NeighborState>,
    visitor: &mut impl FnMut(usize, f64) -> ControlFlow<B>,
) -> ControlFlow<B>
where
    T: PointKnn + ?Sized,
{
    visit_neighbors(
        tree.knn_num_nodes(),
        tree.knn_num_items(),
        tree.knn_node_size(),
        |n| tree.knn_level_end(n),
        |p| tree.knn_index_at(p),
        max_distance,
        |pos| tree.knn_distance_squared_to(pos, point),
        queue,
        visitor,
    )
}

/// Best-first exact kNN: prune the tree by `dist` (lower-bound box distance),
/// refine each candidate leaf by `exact_dist` (true caller-box distance).
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_neighbors_refined(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
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
        let end = (state.index + node_size).min(level_end(state.index));
        let is_leaf = state.index < num_items;
        for pos in state.index..end {
            let d = dist(pos);
            if d <= cutoff {
                frontier.push(NeighborState::new(index_at(pos), is_leaf, d));
            }
        }
    }
    write_exact_results(results, best);
}
