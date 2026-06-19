//! Layout-agnostic best-first nearest-neighbor traversal.

use super::{NeighborNodeState, NeighborState, max_distance_squared};
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Exclusive node-position end of the level containing `node`, for slice-backed
/// level-bound tables.
#[cfg(feature = "f32-storage")]
#[inline]
pub(crate) fn level_end_of(level_bounds: &[usize], node: usize) -> usize {
    level_bounds[level_bounds.partition_point(|&end| end <= node)]
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

/// Visit items in nondecreasing squared-distance order; `visitor` may break
/// early.
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
