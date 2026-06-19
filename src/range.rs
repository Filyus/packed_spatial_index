use std::ops::ControlFlow;

use crate::tree_access::{TreeAccess, leaf_group_range};

/// Visit every leaf item whose bounds overlap `query`.
///
/// This is the dimension-independent overlap traversal shared by scalar f64
/// owned indexes and zero-copy views. Hotter specialized search paths may still
/// layer prefetching or contained-subtree shortcuts on top of the same
/// [`TreeAccess`] contract.
#[inline]
pub(crate) fn visit_overlaps<R, T, F>(
    tree: &T,
    query: T::Bounds,
    stack: &mut Vec<usize>,
    mut visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    F: FnMut(usize) -> ControlFlow<R>,
{
    stack.clear();
    if tree.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }

    let mut node_index = tree.tree_num_nodes() - 1;
    let mut level = tree.tree_level_count() - 1;

    loop {
        let end = (node_index + tree.tree_node_size()).min(tree.tree_level_bound(level));
        let is_leaf = node_index < tree.tree_num_items();

        if is_leaf {
            for pos in node_index..end {
                if !T::bounds_overlap(tree.tree_bounds(pos), query) {
                    continue;
                }
                visitor(tree.tree_index(pos))?;
            }
        } else {
            let child_level = level - 1;
            for pos in (node_index..end).rev() {
                if !T::bounds_overlap(tree.tree_bounds(pos), query) {
                    continue;
                }
                stack.push(tree.tree_index(pos));
                stack.push(child_level);
            }
        }

        if stack.len() > 1 {
            level = stack.pop().unwrap();
            node_index = stack.pop().unwrap();
        } else {
            return ControlFlow::Continue(());
        }
    }
}

/// Collect every leaf item whose bounds overlap `query`.
///
/// The root-contained fast path mirrors existing view traversal: if the query
/// contains the whole tree extent, every leaf id is emitted without descending.
#[inline]
pub(crate) fn collect_overlaps<T>(
    tree: &T,
    query: T::Bounds,
    results: &mut Vec<usize>,
    stack: &mut Vec<usize>,
) where
    T: TreeAccess,
{
    results.clear();
    stack.clear();
    if tree.tree_num_items() == 0 {
        return;
    }

    let root = tree.tree_bounds(tree.tree_num_nodes() - 1);
    if T::bounds_contain(query, root) {
        for pos in 0..tree.tree_num_items() {
            results.push(tree.tree_index(pos));
        }
        return;
    }

    let _: ControlFlow<()> = visit_overlaps(tree, query, stack, |index| {
        results.push(index);
        ControlFlow::Continue(())
    });
}

/// Visit every item whose bounds overlap an arbitrary region predicate.
///
/// `overlaps` decides whether a node must be descended or a leaf item emitted;
/// `contains` accepts a whole subtree without per-leaf region tests.
#[inline]
pub(crate) fn visit_region<R, T, O, C, F>(
    tree: &T,
    stack: &mut Vec<usize>,
    overlaps: O,
    contains: C,
    mut visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    C: Fn(T::Bounds) -> bool,
    F: FnMut(usize) -> ControlFlow<R>,
{
    stack.clear();
    if tree.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }

    const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
    const LEVEL_MASK: usize = !CONTAINED_FLAG;

    let root = tree.tree_bounds(tree.tree_num_nodes() - 1);
    if contains(root) {
        for pos in 0..tree.tree_num_items() {
            visitor(tree.tree_index(pos))?;
        }
        return ControlFlow::Continue(());
    }

    let mut node_index = tree.tree_num_nodes() - 1;
    let mut level = tree.tree_level_count() - 1;
    let mut contained = false;

    loop {
        let end = (node_index + tree.tree_node_size()).min(tree.tree_level_bound(level));
        let is_leaf = node_index < tree.tree_num_items();

        if contained {
            let (start, leaf_end) = leaf_group_range(tree, node_index, end, level);
            for pos in start..leaf_end {
                visitor(tree.tree_index(pos))?;
            }
        } else if is_leaf {
            for pos in node_index..end {
                let bounds = tree.tree_bounds(pos);
                if !overlaps(bounds) {
                    continue;
                }
                visitor(tree.tree_index(pos))?;
            }
        } else {
            let child_level = level - 1;
            for pos in (node_index..end).rev() {
                let bounds = tree.tree_bounds(pos);
                if !overlaps(bounds) {
                    continue;
                }
                stack.push(tree.tree_index(pos));
                let encoded_level = if contains(bounds) {
                    child_level | CONTAINED_FLAG
                } else {
                    child_level
                };
                stack.push(encoded_level);
            }
        }

        if stack.len() > 1 {
            let encoded_level = stack.pop().unwrap();
            level = encoded_level & LEVEL_MASK;
            contained = (encoded_level & CONTAINED_FLAG) != 0;
            node_index = stack.pop().unwrap();
        } else {
            return ControlFlow::Continue(());
        }
    }
}
