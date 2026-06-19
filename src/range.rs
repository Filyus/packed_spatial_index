use std::ops::ControlFlow;

use crate::tree_access::TreeAccess;

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
