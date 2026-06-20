//! Layout-agnostic best-first nearest-neighbor traversal.

use super::{NeighborNodeState, NeighborState, max_distance_squared};
use crate::config::DEFAULT_NEIGHBOR_QUEUE_CAPACITY;
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Exclusive node-position end of the level containing `node`, for slice-backed
/// level-bound tables.
#[cfg(feature = "f32-storage")]
#[inline]
pub(crate) fn level_end_of(level_bounds: &[usize], node: usize) -> usize {
    level_bounds[level_bounds.partition_point(|&end| end <= node)]
}

/// Best-first descent collecting up to `max_results` nearest items, for callers
/// that hold only the item queue (the SIMD and f32 frontends). A convenience over
/// [`collect_neighbors_two_queue`]: it allocates a small pre-sized scratch node
/// queue and delegates. The scalar f64 indexes call the two-queue kernel directly
/// with their workspace's node queue.
///
/// The two-queue distance-browsing algorithm measured faster than a single-queue
/// collect across every frontend — ~5% scalar f64, ~6% `SimdIndex2D`, ~11%
/// `Index2DF32`, ~7% `Index3DF32`, flat on `SimdIndex3D` (k=10, 200k boxes,
/// pinned) — so it is the one kNN collect kernel everywhere. The per-call scratch
/// alloc is one pre-sized `BinaryHeap` (the same cost as the item queue) and
/// measured no worse than a reused buffer.
#[cfg(any(feature = "simd", feature = "f32-storage"))]
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
    let mut node_queue: BinaryHeap<NeighborNodeState> =
        BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    collect_neighbors_two_queue(
        num_nodes,
        num_items,
        node_size,
        level_end,
        index_at,
        max_results,
        max_distance,
        dist,
        results,
        queue,
        &mut node_queue,
    );
}

/// Best-first descent with separate node and item priority queues — the
/// distance-browsing variant the scalar f64 indexes use. It expands nodes in
/// distance order and only emits an item once it is closer than the next pending
/// node, so it avoids pushing every candidate leaf into one heap. `dist(pos)` is
/// the squared distance to the box at node `pos`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_neighbors_two_queue(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
    max_results: usize,
    max_distance: f64,
    dist: impl Fn(usize) -> f64,
    results: &mut Vec<usize>,
    item_queue: &mut BinaryHeap<NeighborState>,
    node_queue: &mut BinaryHeap<NeighborNodeState>,
) {
    item_queue.clear();
    node_queue.clear();
    let Some(max_dist_sq) = max_distance_squared(max_distance) else {
        return;
    };
    if num_items == 0 {
        return;
    }
    let root_index = num_nodes - 1;
    let root_dist = dist(root_index);
    if root_dist > max_dist_sq {
        return;
    }
    node_queue.push(NeighborNodeState::new(root_index, root_dist));

    while results.len() < max_results {
        while let Some(&node) = node_queue.peek() {
            if node.dist > max_dist_sq {
                node_queue.clear();
                break;
            }
            if item_queue.peek().is_some_and(|item| item.dist < node.dist) {
                break;
            }

            let node = node_queue.pop().unwrap();
            let end = (node.index + node_size).min(level_end(node.index));
            let is_leaf = node.index < num_items;

            if is_leaf {
                for pos in node.index..end {
                    let d = dist(pos);
                    if d <= max_dist_sq {
                        item_queue.push(NeighborState::new(index_at(pos), true, d));
                    }
                }
            } else {
                for pos in node.index..end {
                    let d = dist(pos);
                    if d <= max_dist_sq {
                        node_queue.push(NeighborNodeState::new(index_at(pos), d));
                    }
                }
            }
        }

        match item_queue.pop() {
            Some(state) if state.dist <= max_dist_sq => results.push(state.index),
            _ => return,
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
/// early. A convenience over [`visit_neighbors_two_queue`] for callers holding
/// only the item queue: allocates a small pre-sized scratch node queue and
/// delegates, so `visit` emits in the SAME order as the two-queue `neighbors`
/// collect (they must agree on ties — a contract the tests check).
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
    let mut node_queue: BinaryHeap<NeighborNodeState> =
        BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    visit_neighbors_two_queue(
        num_nodes,
        num_items,
        node_size,
        level_end,
        index_at,
        max_distance,
        dist,
        queue,
        &mut node_queue,
        visitor,
    )
}

/// Two-queue distance-browsing visit: emits items in nondecreasing squared
/// distance via `visitor` (which may break early), in the same order the
/// two-queue [`collect_neighbors_two_queue`] produces.
#[allow(clippy::too_many_arguments)]
pub(crate) fn visit_neighbors_two_queue<B>(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
    max_distance: f64,
    dist: impl Fn(usize) -> f64,
    item_queue: &mut BinaryHeap<NeighborState>,
    node_queue: &mut BinaryHeap<NeighborNodeState>,
    visitor: &mut impl FnMut(usize, f64) -> ControlFlow<B>,
) -> ControlFlow<B> {
    item_queue.clear();
    node_queue.clear();
    let Some(max_dist_sq) = max_distance_squared(max_distance) else {
        return ControlFlow::Continue(());
    };
    if num_items == 0 {
        return ControlFlow::Continue(());
    }
    let root_index = num_nodes - 1;
    let root_dist = dist(root_index);
    if root_dist > max_dist_sq {
        return ControlFlow::Continue(());
    }
    node_queue.push(NeighborNodeState::new(root_index, root_dist));

    loop {
        while let Some(&node) = node_queue.peek() {
            if node.dist > max_dist_sq {
                node_queue.clear();
                break;
            }
            if item_queue.peek().is_some_and(|item| item.dist < node.dist) {
                break;
            }

            let node = node_queue.pop().unwrap();
            let end = (node.index + node_size).min(level_end(node.index));
            let is_leaf = node.index < num_items;

            if is_leaf {
                for pos in node.index..end {
                    let d = dist(pos);
                    if d <= max_dist_sq {
                        item_queue.push(NeighborState::new(index_at(pos), true, d));
                    }
                }
            } else {
                for pos in node.index..end {
                    let d = dist(pos);
                    if d <= max_dist_sq {
                        node_queue.push(NeighborNodeState::new(index_at(pos), d));
                    }
                }
            }
        }

        match item_queue.pop() {
            Some(state) if state.dist <= max_dist_sq => visitor(state.index, state.dist)?,
            _ => return ControlFlow::Continue(()),
        }
    }
}
