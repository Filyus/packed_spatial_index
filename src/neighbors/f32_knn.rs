//! f32-only point kNN traversal.
//!
//! Shared by the owned `Index*F32`, SIMD `SimdIndex*F32`, and f32 views.
//! Dimension/layout-agnostic: callers pass `dist`/`exact_dist` closures that
//! read their own box storage.

#![allow(clippy::needless_range_loop)]

use super::{ExactNeighborState, NeighborNodeState, NeighborState, max_distance_squared};
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Exclusive node-position end of the level containing `node`, for the
/// slice-backed owned indexes. (Byte-backed views supply their own closure that
/// reads the level bounds from the file header.)
#[inline]
pub(crate) fn level_end_of(level_bounds: &[usize], node: usize) -> usize {
    level_bounds[level_bounds.partition_point(|&end| end <= node)]
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
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
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
        let end = (node_index + node_size).min(level_end(node_index));
        let is_leaf = node_index < num_items;
        for pos in node_index..end {
            let d = dist(pos);
            if d > max_dist_sq {
                continue;
            }
            queue.push(NeighborState::new(index_at(pos), is_leaf, d));
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
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
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
        let end = (node_index + node_size).min(level_end(node_index));
        let is_leaf = node_index < num_items;
        for pos in node_index..end {
            let d = dist(pos);
            if d > best_dist {
                continue;
            }
            if is_leaf {
                if d == 0.0 {
                    return Some(index_at(pos));
                }
                best_dist = d;
                best_index = Some(index_at(pos));
            } else {
                queue.push(NeighborNodeState::new(index_at(pos), d));
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

/// Visit items in nondecreasing squared-distance order; `visitor` may break early.
#[allow(clippy::too_many_arguments)]
pub(crate) fn visit_neighbors<B>(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
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
        let end = (node_index + node_size).min(level_end(node_index));
        let is_leaf = node_index < num_items;
        for pos in node_index..end {
            let d = dist(pos);
            if d > max_dist_sq {
                continue;
            }
            queue.push(NeighborState::new(index_at(pos), is_leaf, d));
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
