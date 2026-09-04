//! Selectivity estimation: how many items a window would hit, answered from
//! node boxes without running the query to the leaves.
//!
//! A packed tree is its own multi-level histogram. The number of items under
//! a node is fixed by construction — a node at level `L` covers `node_size^L`
//! leaves, clipped at the end of the array — so nothing is stored and nothing
//! is read to know it. Walking the upper levels therefore brackets a window's
//! hit count exactly:
//!
//! - `lower`: items under nodes whose box lies entirely inside the window;
//! - `upper`: items under nodes whose box overlaps the window at all;
//! - `estimate`: `lower` plus, for each node cut by the window's edge, its
//!   size scaled by the fraction of its box the window covers — the one
//!   assumption here, that items are spread uniformly inside a node box.
//!
//! The bracket is exact whatever the data does; only `estimate` trusts the
//! uniformity assumption, and it gets better the deeper the walk stops. Both
//! are box counts, like every answer in this crate: a lower bound on how many
//! geometries the window really touches.

use crate::tree_access::TreeAccess;
use crate::{Box2D, Box3D};

/// A bracketed hit-count estimate for a window: `lower <= hits <= upper` is
/// exact from node boxes alone, `estimate` assumes items spread uniformly
/// inside each node box the window cuts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Items guaranteed to be hit: those under nodes the window contains.
    pub lower: usize,
    /// Items that can possibly be hit: those under nodes the window overlaps.
    pub upper: usize,
    /// The point estimate, in `[lower, upper]`.
    pub estimate: f64,
    /// Node boxes tested, the whole cost of the estimate.
    pub nodes_tested: usize,
}

impl Estimate {
    /// `upper - lower`: how much the bracket could still tighten.
    pub fn spread(&self) -> usize {
        self.upper - self.lower
    }
}

/// Leaf-array `[start, end)` under the node at `pos` on `level`, by rank
/// arithmetic: a packed level `L` node with rank `r` covers leaves
/// `[r * node_size^L, (r + 1) * node_size^L)`, clipped to `num_items`.
#[inline]
pub(crate) fn subtree_leaf_range(
    pos: usize,
    level: usize,
    level_start: usize,
    node_size: usize,
    num_items: usize,
) -> (usize, usize) {
    let rank = pos - level_start;
    let span = node_size.checked_pow(level as u32).unwrap_or(usize::MAX);
    let start = rank.saturating_mul(span).min(num_items);
    let end = start.saturating_add(span).min(num_items);
    (start, end)
}

/// Fraction of `node`'s box that `query` covers, per axis multiplied. An axis
/// along which the node is flat contributes 1: overlapping on it means being
/// inside on it.
#[inline]
fn axis_fraction(node_min: f64, node_max: f64, query_min: f64, query_max: f64) -> f64 {
    let node_len = node_max - node_min;
    if node_len.is_nan() || node_len <= 0.0 {
        return 1.0;
    }
    let overlap = query_max.min(node_max) - query_min.max(node_min);
    (overlap / node_len).clamp(0.0, 1.0)
}

/// Fraction of a 2D node box inside `query`.
#[inline]
pub(crate) fn box_fraction_2d(node: Box2D, query: Box2D) -> f64 {
    axis_fraction(node.min_x, node.max_x, query.min_x, query.max_x)
        * axis_fraction(node.min_y, node.max_y, query.min_y, query.max_y)
}

/// Fraction of a 3D node box inside `query`.
#[inline]
pub(crate) fn box_fraction_3d(node: Box3D, query: Box3D) -> f64 {
    axis_fraction(node.min_x, node.max_x, query.min_x, query.max_x)
        * axis_fraction(node.min_y, node.max_y, query.min_y, query.max_y)
        * axis_fraction(node.min_z, node.max_z, query.min_z, query.max_z)
}

/// The estimate over any tree the join family can walk.
///
/// Descends from the root; a node the window contains is counted whole, a
/// node the window misses is dropped, and a node the window cuts is expanded
/// while its level is above `stop_level` and scored by `fraction` once it is
/// not. Leaves (level 0) are exact whichever way they are reached: a leaf box
/// that overlaps the window is a hit.
pub(crate) fn estimate_core<T, O, C, Fr>(
    tree: &T,
    stop_level: usize,
    overlaps: O,
    contains: C,
    fraction: Fr,
) -> Estimate
where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    C: Fn(T::Bounds) -> bool,
    Fr: Fn(T::Bounds) -> f64,
{
    let num_items = tree.tree_num_items();
    let mut out = Estimate {
        lower: 0,
        upper: 0,
        estimate: 0.0,
        nodes_tested: 0,
    };
    if num_items == 0 {
        return out;
    }
    let node_size = tree.tree_node_size();
    let level_start = |level: usize| {
        if level == 0 {
            0
        } else {
            tree.tree_level_bound(level - 1)
        }
    };

    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(64);
    stack.push((tree.tree_num_nodes() - 1, tree.tree_level_count() - 1));
    while let Some((pos, level)) = stack.pop() {
        let bounds = tree.tree_bounds(pos);
        out.nodes_tested += 1;
        if !overlaps(bounds) {
            continue;
        }
        let (start, end) = subtree_leaf_range(pos, level, level_start(level), node_size, num_items);
        let size = end - start;
        if level == 0 || contains(bounds) {
            out.lower += size;
            out.upper += size;
            out.estimate += size as f64;
            continue;
        }
        if level <= stop_level {
            out.upper += size;
            out.estimate += size as f64 * fraction(bounds);
            continue;
        }
        let child_level = level - 1;
        let first = tree.tree_index(pos);
        let last = (first + node_size).min(tree.tree_level_bound(child_level));
        for child in first..last {
            stack.push((child, child_level));
        }
    }
    out.estimate = out.estimate.clamp(out.lower as f64, out.upper as f64);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtree_ranges_follow_rank_arithmetic() {
        // 10 items, node_size 4: level 0 has 10 leaves, level 1 has 3 nodes
        // (starting at position 10), level 2 has 1 node (position 13).
        assert_eq!(subtree_leaf_range(10, 1, 10, 4, 10), (0, 4));
        assert_eq!(subtree_leaf_range(11, 1, 10, 4, 10), (4, 8));
        assert_eq!(subtree_leaf_range(12, 1, 10, 4, 10), (8, 10));
        assert_eq!(subtree_leaf_range(13, 2, 13, 4, 10), (0, 10));
        assert_eq!(subtree_leaf_range(7, 0, 0, 4, 10), (7, 8));
    }

    #[test]
    fn fraction_is_area_ratio_and_one_on_flat_axes() {
        let node = Box2D::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(box_fraction_2d(node, Box2D::new(0.0, 0.0, 5.0, 10.0)), 0.5);
        assert_eq!(
            box_fraction_2d(node, Box2D::new(5.0, 5.0, 20.0, 20.0)),
            0.25
        );
        assert_eq!(
            box_fraction_2d(node, Box2D::new(-1.0, -1.0, 11.0, 11.0)),
            1.0
        );
        assert_eq!(
            box_fraction_2d(node, Box2D::new(20.0, 20.0, 30.0, 30.0)),
            0.0
        );
        // A point node overlapping the query is inside it.
        let point = Box2D::new(3.0, 3.0, 3.0, 3.0);
        assert_eq!(box_fraction_2d(point, Box2D::new(0.0, 0.0, 5.0, 5.0)), 1.0);
        // A horizontal segment: only x contributes.
        let seg = Box2D::new(0.0, 3.0, 10.0, 3.0);
        assert_eq!(box_fraction_2d(seg, Box2D::new(0.0, 0.0, 2.5, 5.0)), 0.25);
    }
}
