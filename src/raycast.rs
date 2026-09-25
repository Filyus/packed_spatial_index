//! Layout-agnostic scalar raycast traversal.

use crate::index2d::{MASK_CHUNK, for_each_hit, for_each_hit_rev, frame};
use crate::neighbors::{NeighborNodeState, NeighborState};
use crate::ray::inclusive_ray_cutoff;
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

/// Hit tests of the positions `start..end` (at most `MASK_CHUNK`) as a bitmask,
/// bit `i` for position `start + i`.
#[inline(always)]
fn hit_mask(start: usize, end: usize, hit_at: &impl Fn(usize) -> bool) -> u64 {
    debug_assert!(end - start <= MASK_CHUNK);
    let mut mask = 0u64;
    for (i, pos) in (start..end).enumerate() {
        mask |= u64::from(hit_at(pos)) << i;
    }
    mask
}

/// Visit, low to high, the positions in `start..stop` that `hit_at` accepts:
/// through [`hit_mask`] when `MASKED`, one branch per position when not. The
/// shipping callers pass `true`; `false` is the per-child branch the mask
/// replaced, kept so both can be timed in one binary
/// (`benches/paired_mask_forms.rs`).
#[inline(always)]
fn each_hit<const MASKED: bool>(
    start: usize,
    stop: usize,
    hit_at: &impl Fn(usize) -> bool,
    mut f: impl FnMut(usize),
) {
    if MASKED {
        for_each_hit(hit_mask(start, stop, hit_at), |i| f(start + i));
    } else {
        for pos in start..stop {
            if hit_at(pos) {
                f(pos);
            }
        }
    }
}

/// [`each_hit`] high to low.
#[inline(always)]
fn each_hit_rev<const MASKED: bool>(
    start: usize,
    stop: usize,
    hit_at: &impl Fn(usize) -> bool,
    mut f: impl FnMut(usize),
) {
    if MASKED {
        for_each_hit_rev(hit_mask(start, stop, hit_at), |i| f(start + i));
    } else {
        for pos in (start..stop).rev() {
            if hit_at(pos) {
                f(pos);
            }
        }
    }
}

/// Depth-first raycast collection over a packed tree. Callers provide storage
/// accessors for hit testing and item/node indices.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn collect_hits<const MASKED: bool>(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_count: usize,
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
    hit_at: impl Fn(usize) -> bool,
    reverse_internal_push: bool,
    prefetch_at: impl Fn(usize),
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

        // No early exit here, so each node's hit tests fold into a bitmask and
        // the loop branches once per hit instead of once per child. See
        // `index2d::overlap_mask`.
        if is_leaf {
            let mut start = node_index;
            while start < end {
                let stop = (start + MASK_CHUNK).min(end);
                each_hit::<MASKED>(start, stop, &hit_at, |pos| results.push(index_at(pos)));
                start = stop;
            }
        } else {
            let child_level = level - 1;
            if reverse_internal_push {
                // Chunks from the back, bits from the top: children pop in order.
                let mut stop = end;
                while stop > node_index {
                    let start = stop.saturating_sub(MASK_CHUNK).max(node_index);
                    each_hit_rev::<MASKED>(start, stop, &hit_at, |pos| {
                        stack.push(frame::pack(index_at(pos), child_level));
                    });
                    stop = start;
                }
            } else {
                let mut start = node_index;
                while start < end {
                    let stop = (start + MASK_CHUNK).min(end);
                    each_hit::<MASKED>(start, stop, &hit_at, |pos| {
                        stack.push(frame::pack(index_at(pos), child_level));
                    });
                    start = stop;
                }
            }
        }

        match stack.pop() {
            Some(f) => {
                // Prefetch the next node to be popped so its box loads while
                // this node is hit-tested.
                if let Some(&next) = stack.last() {
                    prefetch_at(frame::node(next));
                }
                node_index = frame::node(f);
                level = frame::level(f);
            }
            None => return,
        }
    }
}

/// Depth-first test for any hit: `true` at the first leaf entry the ray
/// segment enters. No order is promised, so it needs no priority queue, only
/// the stack [`collect_hits`] uses. `MASKED` picks the child test as there.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn any_hit<const MASKED: bool>(
    num_nodes: usize,
    num_items: usize,
    node_size: usize,
    level_count: usize,
    level_end: impl Fn(usize) -> usize,
    index_at: impl Fn(usize) -> usize,
    hit_at: impl Fn(usize) -> bool,
    stack: &mut Vec<usize>,
) -> bool {
    stack.clear();
    if num_items == 0 {
        return false;
    }

    let mut node_index = num_nodes - 1;
    let mut level = level_count - 1;

    loop {
        let end = (node_index + node_size).min(level_end(level));
        if node_index < num_items {
            let mut start = node_index;
            while start < end {
                let stop = (start + MASK_CHUNK).min(end);
                let hit = if MASKED {
                    hit_mask(start, stop, &hit_at) != 0
                } else {
                    (start..stop).any(&hit_at)
                };
                if hit {
                    return true;
                }
                start = stop;
            }
        } else {
            let child_level = level - 1;
            let mut stop = end;
            while stop > node_index {
                let start = stop.saturating_sub(MASK_CHUNK).max(node_index);
                each_hit_rev::<MASKED>(start, stop, &hit_at, |pos| {
                    stack.push(frame::pack(index_at(pos), child_level));
                });
                stop = start;
            }
        }

        match stack.pop() {
            Some(f) => {
                node_index = frame::node(f);
                level = frame::level(f);
            }
            None => return false,
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
    let mut best_t = inclusive_ray_cutoff(max_distance);
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
