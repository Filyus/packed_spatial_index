use std::ops::ControlFlow;

use crate::index2d::{MASK_CHUNK, for_each_hit, for_each_hit_rev, frame};
use crate::tree_access::{TreeAccess, leaf_group_range};

/// Overlap tests of the positions `start..end` (at most `MASK_CHUNK` of them)
/// folded into a bitmask, bit `i` for position `start + i`. The generic twin of
/// `index2d::overlap_mask` for byte-backed views, whose bounds are decoded per
/// position rather than read from a slice.
#[inline(always)]
pub(crate) fn overlap_mask_at<T: TreeAccess>(
    tree: &T,
    start: usize,
    end: usize,
    overlaps: &impl Fn(T::Bounds) -> bool,
) -> u64 {
    debug_assert!(end - start <= MASK_CHUNK);
    let mut mask = 0u64;
    for (i, pos) in (start..end).enumerate() {
        mask |= u64::from(overlaps(tree.tree_bounds(pos))) << i;
    }
    mask
}

/// Collect every leaf item whose bounds overlap an arbitrary region predicate,
/// with the contained-subtree shortcut of [`search_region_each`].
///
/// The collect twin of `search_region_each`: it has no early exit, so each node's
/// overlap tests run branch-free into a bitmask and the loop branches once per
/// hit instead of once per child — the same trade the owned indexes make in
/// their `search_into_stack` paths.
///
/// `MASKED = false` swaps the mask for the per-child branch it replaced, so the
/// two forms can be timed in one binary (`benches/paired_mask_forms.rs`);
/// every shipping caller passes `true`.
#[inline]
pub(crate) fn collect_region<const MASKED: bool, T, O, C, F>(
    tree: &T,
    stack: &mut Vec<usize>,
    overlaps: O,
    contains: C,
    mut emit: F,
) where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    C: Fn(T::Bounds) -> bool,
    F: FnMut(usize),
{
    stack.clear();
    if tree.tree_num_items() == 0 {
        return;
    }

    let root = tree.tree_bounds(tree.tree_num_nodes() - 1);
    // See `search_region_each` for why `overlaps` is tested before `contains`.
    if overlaps(root) && contains(root) {
        for pos in 0..tree.tree_num_items() {
            emit(tree.tree_index(pos));
        }
        return;
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
                emit(tree.tree_index(pos));
            }
        } else if is_leaf {
            let mut start = node_index;
            while start < end {
                let stop = (start + MASK_CHUNK).min(end);
                if MASKED {
                    for_each_hit(overlap_mask_at(tree, start, stop, &overlaps), |i| {
                        emit(tree.tree_index(start + i));
                    });
                } else {
                    for pos in start..stop {
                        if overlaps(tree.tree_bounds(pos)) {
                            emit(tree.tree_index(pos));
                        }
                    }
                }
                start = stop;
            }
        } else {
            let child_level = level - 1;
            // Chunks from the back, bits from the top: children pop in forward order.
            let mut stop = end;
            while stop > node_index {
                let start = stop.saturating_sub(MASK_CHUNK).max(node_index);
                let mut push = |pos: usize| {
                    let flag = usize::from(contains(tree.tree_bounds(pos))) * frame::CONTAINED;
                    stack.push(frame::pack(tree.tree_index(pos), child_level) | flag);
                };
                if MASKED {
                    for_each_hit_rev(overlap_mask_at(tree, start, stop, &overlaps), |i| {
                        push(start + i);
                    });
                } else {
                    for pos in (start..stop).rev() {
                        if overlaps(tree.tree_bounds(pos)) {
                            push(pos);
                        }
                    }
                }
                stop = start;
            }
        }

        match stack.pop() {
            Some(f) => {
                node_index = frame::node(f);
                level = frame::level(f);
                contained = frame::contained(f);
            }
            None => return,
        }
    }
}

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
                stack.push(frame::pack(tree.tree_index(pos), child_level));
            }
        }

        match stack.pop() {
            Some(f) => {
                node_index = frame::node(f);
                level = frame::level(f);
            }
            None => return ControlFlow::Continue(()),
        }
    }
}

/// Visit every item whose bounds overlap an arbitrary region predicate.
///
/// `overlaps` decides whether a node must be descended or a leaf item emitted;
/// `contains` accepts a whole subtree without per-leaf region tests.
#[inline]
pub(crate) fn search_region_each<R, T, O, C, F>(
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

    let root = tree.tree_bounds(tree.tree_num_nodes() - 1);
    // `overlaps` first, even though containment implies it for any well-formed box:
    // a box built through the unchecked `Box2D::new` may have `min > max`, and such a
    // box is contained by a query it does not overlap. Without this the shortcut
    // answers "every item" where the per-item test answers "none".
    if overlaps(root) && contains(root) {
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
                let flag = usize::from(contains(bounds)) * frame::CONTAINED;
                stack.push(frame::pack(tree.tree_index(pos), child_level) | flag);
            }
        }

        match stack.pop() {
            Some(f) => {
                node_index = frame::node(f);
                level = frame::level(f);
                contained = frame::contained(f);
            }
            None => return ControlFlow::Continue(()),
        }
    }
}
