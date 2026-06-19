//! Custom-metric point kNN traversal.
//!
//! Shared by the owned `Index2D` / `Index3D` and their f64 byte views.
//! Layout/dimension-agnostic: callers pass a `dist` closure that returns the
//! metric distance from the query to the box at a node position. Distances are
//! in the metric's own units (NOT squared), so the caller's `max_distance` is
//! compared directly. `dist` must be an admissible lower bound for internal
//! nodes: the distance to the node's bounding box never exceeds the distance to
//! any item inside it.

#![allow(clippy::needless_range_loop)]

use super::{NeighborState, valid_max_distance};
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Best-first descent collecting up to `max_results` nearest items in
/// nondecreasing metric distance. Single priority queue holding both pending
/// nodes and candidate leaves (Hjaltason-Samet).
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
    let Some(max_dist) = valid_max_distance(max_distance) else {
        return;
    };
    if num_items == 0 || max_results == 0 {
        return;
    }
    let mut node_index = num_nodes - 1;
    loop {
        let end = (node_index + node_size).min(level_end(node_index));
        let is_leaf = node_index < num_items;
        for pos in node_index..end {
            let d = dist(pos);
            if d > max_dist {
                continue;
            }
            queue.push(NeighborState::new(index_at(pos), is_leaf, d));
        }
        let mut continue_search = false;
        while let Some(state) = queue.pop() {
            if state.dist > max_dist {
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

/// Visit items in nondecreasing metric distance; `visitor` may break early.
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
    let Some(max_dist) = valid_max_distance(max_distance) else {
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
            if d > max_dist {
                continue;
            }
            queue.push(NeighborState::new(index_at(pos), is_leaf, d));
        }
        let mut continue_search = false;
        while let Some(state) = queue.pop() {
            if state.dist > max_dist {
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
