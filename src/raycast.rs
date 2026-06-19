//! Layout-agnostic scalar raycast traversal.

use crate::neighbors::{NeighborNodeState, NeighborState};
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Depth-first raycast collection over a packed tree. Callers provide storage
/// accessors for hit testing and item/node indices.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn collect_hits(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_count: usize,
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
    hit_at: impl Fn(usize) -> bool,
    reverse_internal_push: bool,
    results: &mut Vec<usize>,
    stack: &mut Vec<usize>,
) {
    results.clear();
    stack.clear();
    if num_items == 0 {
        return;
    }

    let mut node_index = num_nodes - 1;
    let mut level = level_count - 1;

    loop {
        let end = (node_index + node_size).min(level_end(level));
        let is_leaf = node_index < num_items;

        if is_leaf {
            for pos in node_index..end {
                if hit_at(pos) {
                    results.push(index_at(pos));
                }
            }
        } else {
            let child_level = level - 1;
            if reverse_internal_push {
                for pos in (node_index..end).rev() {
                    if hit_at(pos) {
                        stack.push(index_at(pos));
                        stack.push(child_level);
                    }
                }
            } else {
                for pos in node_index..end {
                    if hit_at(pos) {
                        stack.push(index_at(pos));
                        stack.push(child_level);
                    }
                }
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

/// Best-first closest-hit traversal. `enter_at(pos)` returns the ray entry
/// parameter for the box at `pos`, or `None` for a miss.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn closest_hit(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    max_distance: f64,
    level_end_of_node: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
    enter_at: impl Fn(usize) -> Option<f64>,
    queue: &mut BinaryHeap<NeighborNodeState>,
) -> Option<(usize, f64)> {
    queue.clear();
    if num_items == 0 {
        return None;
    }

    let root = num_nodes - 1;
    let root_t = enter_at(root)?;
    let mut best_t = max_distance;
    let mut best_index = None;
    queue.push(NeighborNodeState::new(root, root_t));

    while let Some(node) = queue.pop() {
        // The heap yields nodes by ascending entry t, and a node's entry t is a
        // lower bound on every descendant's, so once it reaches the best hit we stop.
        if node.dist >= best_t {
            break;
        }
        let end = (node.index + node_size).min(level_end_of_node(node.index));
        let is_leaf = node.index < num_items;
        for pos in node.index..end {
            let Some(t) = enter_at(pos) else {
                continue;
            };
            if t >= best_t {
                continue;
            }
            if is_leaf {
                best_t = t;
                best_index = Some(index_at(pos));
            } else {
                queue.push(NeighborNodeState::new(index_at(pos), t));
            }
        }
    }

    best_index.map(|index| (index, best_t))
}

/// Visit hits in nondecreasing entry-`t` order.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn visit_hits<B>(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_end_of_node: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
    enter_at: impl Fn(usize) -> Option<f64>,
    queue: &mut BinaryHeap<NeighborState>,
    visitor: &mut impl FnMut(usize, f64) -> ControlFlow<B>,
) -> ControlFlow<B> {
    queue.clear();
    if num_items == 0 {
        return ControlFlow::Continue(());
    }

    let mut node_index = num_nodes - 1;
    loop {
        let end = (node_index + node_size).min(level_end_of_node(node_index));
        let is_leaf = node_index < num_items;

        for pos in node_index..end {
            if let Some(t) = enter_at(pos) {
                queue.push(NeighborState::new(index_at(pos), is_leaf, t));
            }
        }

        let mut continue_search = false;
        while let Some(state) = queue.pop() {
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
