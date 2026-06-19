//! Pairwise spatial joins: report every intersecting pair of items between two
//! packed trees, or within one tree (`self_join`).
//!
//! The traversal descends both trees simultaneously from the pair of roots. One
//! bounds test between two internal entries prunes their whole subtree pair, so
//! the cost scales with the output size instead of running one full search per
//! item. The generic core works over [`TreeAccess`], a minimal accessor view of
//! the packed layout shared by every f64 index and byte-view type.

use std::ops::ControlFlow;

use crate::tree_access::{TreeAccess, leaf_range};

/// One traversal step: expand the higher-level side of the entry pair, emit
/// leaf/leaf pairs inline, and push surviving pairs onto the stack.
///
/// Invariants: the two entry bounds overlap, and `max(a_level, b_level) >= 1`
/// (both-leaf pairs are emitted by the caller and never reach the stack).
#[inline]
#[allow(clippy::too_many_arguments)]
fn expand_pair<R, T, U, F>(
    a: &T,
    b: &U,
    a_pos: usize,
    a_level: usize,
    b_pos: usize,
    b_level: usize,
    stack: &mut Vec<(usize, usize, usize, usize)>,
    visitor: &mut F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    U: TreeAccess<Bounds = T::Bounds>,
    F: FnMut(usize, usize) -> ControlFlow<R>,
{
    if a_level >= b_level {
        debug_assert!(a_level > 0);
        let child_level = a_level - 1;
        let start = a.tree_index(a_pos);
        let end = (start + a.tree_node_size()).min(a.tree_level_bound(child_level));
        let b_bounds = b.tree_bounds(b_pos);
        for pos in start..end {
            let bounds = a.tree_bounds(pos);
            if !T::bounds_overlap(bounds, b_bounds) {
                continue;
            }
            if child_level == 0 {
                if b_level == 0 {
                    visitor(a.tree_index(pos), b.tree_index(b_pos))?;
                } else if T::bounds_contain(bounds, b_bounds) {
                    // The leaf box covers B's whole subtree: every item under
                    // `b_pos` intersects it, so emit the range without tests.
                    let item_a = a.tree_index(pos);
                    let (s, e) = leaf_range(b, b_pos, b_level);
                    for b_leaf in s..e {
                        visitor(item_a, b.tree_index(b_leaf))?;
                    }
                } else {
                    stack.push((pos, 0, b_pos, b_level));
                }
            } else if b_level == 0 && T::bounds_contain(b_bounds, bounds) {
                // The B leaf box covers this whole A subtree: mirror fast path.
                let item_b = b.tree_index(b_pos);
                let (s, e) = leaf_range(a, pos, child_level);
                for a_leaf in s..e {
                    visitor(a.tree_index(a_leaf), item_b)?;
                }
            } else {
                stack.push((pos, child_level, b_pos, b_level));
            }
        }
    } else {
        let child_level = b_level - 1;
        let start = b.tree_index(b_pos);
        let end = (start + b.tree_node_size()).min(b.tree_level_bound(child_level));
        let a_bounds = a.tree_bounds(a_pos);
        for pos in start..end {
            let bounds = b.tree_bounds(pos);
            if !T::bounds_overlap(a_bounds, bounds) {
                continue;
            }
            if child_level == 0 {
                if a_level == 0 {
                    visitor(a.tree_index(a_pos), b.tree_index(pos))?;
                } else if T::bounds_contain(bounds, a_bounds) {
                    let item_b = b.tree_index(pos);
                    let (s, e) = leaf_range(a, a_pos, a_level);
                    for a_leaf in s..e {
                        visitor(a.tree_index(a_leaf), item_b)?;
                    }
                } else {
                    stack.push((a_pos, a_level, pos, 0));
                }
            } else if a_level == 0 && T::bounds_contain(a_bounds, bounds) {
                let item_a = a.tree_index(a_pos);
                let (s, e) = leaf_range(b, pos, child_level);
                for b_leaf in s..e {
                    visitor(item_a, b.tree_index(b_leaf))?;
                }
            } else {
                stack.push((a_pos, a_level, pos, child_level));
            }
        }
    }
    ControlFlow::Continue(())
}

/// Visit every pair `(i, j)` where item `i` of `a` intersects item `j` of `b`.
/// Pair order is traversal order and is not part of the API.
pub(crate) fn join_core<R, T, U, F>(a: &T, b: &U, mut visitor: F) -> ControlFlow<R>
where
    T: TreeAccess,
    U: TreeAccess<Bounds = T::Bounds>,
    F: FnMut(usize, usize) -> ControlFlow<R>,
{
    if a.tree_num_items() == 0 || b.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }

    // Roots are always internal entries (a non-empty tree has >= 2 levels).
    let mut a_pos = a.tree_num_nodes() - 1;
    let mut a_level = a.tree_level_count() - 1;
    let mut b_pos = b.tree_num_nodes() - 1;
    let mut b_level = b.tree_level_count() - 1;
    if !T::bounds_overlap(a.tree_bounds(a_pos), b.tree_bounds(b_pos)) {
        return ControlFlow::Continue(());
    }

    let mut stack: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(64);
    loop {
        expand_pair(
            a,
            b,
            a_pos,
            a_level,
            b_pos,
            b_level,
            &mut stack,
            &mut visitor,
        )?;
        match stack.pop() {
            Some((ap, al, bp, bl)) => {
                a_pos = ap;
                a_level = al;
                b_pos = bp;
                b_level = bl;
            }
            None => return ControlFlow::Continue(()),
        }
    }
}

/// Visit every unordered pair of distinct intersecting items within `tree`,
/// each pair exactly once. The order of the two ids within a pair and the pair
/// order are traversal order and are not part of the API.
pub(crate) fn self_join_core<R, T, F>(tree: &T, mut visitor: F) -> ControlFlow<R>
where
    T: TreeAccess,
    F: FnMut(usize, usize) -> ControlFlow<R>,
{
    if tree.tree_num_items() < 2 {
        return ControlFlow::Continue(());
    }

    let mut a_pos = tree.tree_num_nodes() - 1;
    let mut a_level = tree.tree_level_count() - 1;
    let mut b_pos = a_pos;
    let mut b_level = a_level;

    let mut stack: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(64);
    loop {
        if a_pos == b_pos && a_level == b_level {
            // Identical subtrees: expand into ordered child pairs `i <= j` so
            // each unordered pair of distinct items is reached exactly once.
            debug_assert!(a_level > 0);
            let child_level = a_level - 1;
            let start = tree.tree_index(a_pos);
            let end = (start + tree.tree_node_size()).min(tree.tree_level_bound(child_level));
            for i in start..end {
                let bounds_i = tree.tree_bounds(i);
                if child_level > 0 {
                    stack.push((i, child_level, i, child_level));
                }
                for j in (i + 1)..end {
                    if !T::bounds_overlap(bounds_i, tree.tree_bounds(j)) {
                        continue;
                    }
                    if child_level == 0 {
                        visitor(tree.tree_index(i), tree.tree_index(j))?;
                    } else {
                        stack.push((i, child_level, j, child_level));
                    }
                }
            }
        } else {
            expand_pair(
                tree,
                tree,
                a_pos,
                a_level,
                b_pos,
                b_level,
                &mut stack,
                &mut visitor,
            )?;
        }
        match stack.pop() {
            Some((ap, al, bp, bl)) => {
                a_pos = ap;
                a_level = al;
                b_pos = bp;
                b_level = bl;
            }
            None => return ControlFlow::Continue(()),
        }
    }
}
