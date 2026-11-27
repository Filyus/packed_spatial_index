//! Layout-agnostic best-first nearest-neighbor traversal.

use super::{
    ExactNeighborState, NeighborNodeState, NeighborState, max_distance_squared, push_top_k,
};
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Exclusive node-position end of the level containing `node`, for slice-backed
/// level-bound tables.
#[cfg(feature = "f32-storage")]
#[inline]
pub(crate) fn level_end_of(level_bounds: &[usize], node: usize) -> usize {
    level_bounds[level_bounds.partition_point(|&end| end <= node)]
}

/// Best-first branch-and-bound kNN with a min-heap of pending nodes and a
/// bounded max-heap of the `max_results` best items found so far. Once `k`
/// candidates have been seen, the `k`-th worst distance tightens `cutoff`
/// dynamically so farther children and leaves are pruned before heap pushes.
/// Ties break by smallest item index, matching [`visit_neighbors_two_queue`].
/// `dist(pos)` is the squared distance to the box at node `pos`.
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
    top_k: &mut BinaryHeap<ExactNeighborState>,
    node_queue: &mut BinaryHeap<NeighborNodeState>,
) {
    top_k.clear();
    node_queue.clear();
    let Some(mut cutoff) = max_distance_squared(max_distance) else {
        return;
    };
    if num_items == 0 || max_results == 0 {
        return;
    }
    let root_index = num_nodes - 1;
    if cutoff.is_finite() && dist(root_index) > cutoff {
        return;
    }

    let mut node_index = root_index;
    loop {
        let end = (node_index + node_size).min(level_end(node_index));
        let is_leaf = node_index < num_items;

        if is_leaf {
            for pos in node_index..end {
                let d = dist(pos);
                if d <= cutoff {
                    push_top_k(top_k, max_results, index_at(pos), d, &mut cutoff);
                }
            }
        } else {
            for pos in node_index..end {
                let d = dist(pos);
                if d <= cutoff {
                    node_queue.push(NeighborNodeState::new(index_at(pos), d));
                }
            }
        }

        match node_queue.pop() {
            Some(node) if node.dist <= cutoff => node_index = node.index,
            _ => break,
        }
    }

    let n = top_k.len();
    results.resize(n, 0);
    for slot in results.iter_mut().rev() {
        *slot = top_k.pop().unwrap().index;
    }
}

/// Best-first search for the single nearest item (`max_results == 1` fast path).
///
/// Ties go to the smallest item index, which is the rule
/// [`collect_neighbors_two_queue`] already follows — its item heap orders equal
/// distances by index (see `NeighborState::cmp`). Both kernels answer the same
/// question, so they have to answer it the same way: that is what keeps
/// `neighbors(q, 1)` the prefix of `neighbors(q, k)`.
///
/// The rule is why there is no early return once a zero-distance item is found.
/// Stopping there would answer with whichever tied item the traversal reached
/// first, and a smaller index can still be sitting in another subtree that is
/// also at distance zero. The pruning is unaffected: the node loop below already
/// descends into every node whose distance ties `best_dist`.
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
                let item = index_at(pos);
                // `d > best_dist` was already skipped, so `d <= best_dist` here
                // and the tie case is exactly `d == best_dist`.
                if best_index.is_none_or(|best| d < best_dist || item < best) {
                    best_dist = d;
                    best_index = Some(item);
                }
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
