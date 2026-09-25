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
/// `MASKED = false` swaps the mask for the per-child branch it replaced. The
/// 2D view passes `index2d::MASK_PAYS_IN_2D` (so branching on aarch64), the 3D
/// view and the radius mask pass `true`, and the timing hooks reach both
/// (`benches/paired_mask_forms.rs`).
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
    visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    C: Fn(T::Bounds) -> bool,
    F: FnMut(usize) -> ControlFlow<R>,
{
    visit_region::<false, true, R, T, O, C, F>(tree, stack, overlaps, contains, visitor)
}

/// The callback traversal both of the above run, in the forms two const
/// parameters pick.
///
/// `COVERED` adds the contained-subtree shortcut: each pushed child is also
/// tested with `contains`, and a covered one hands its leaf range to the
/// visitor whole. A visitor that stops at its first item never gets that far,
/// so `any` and `first` leave it off.
///
/// `MASKED` folds each node's overlap tests into a bitmask, as
/// [`collect_region`] does, and branches once per hit instead of once per
/// child. The views pass it for box queries, whose test is cheap; the region
/// predicates keep the branch (see the boundary in `docs/performance.md`).
#[inline]
pub(crate) fn visit_region<const MASKED: bool, const COVERED: bool, R, T, O, C, F>(
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

    if COVERED {
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
    }

    let mut node_index = tree.tree_num_nodes() - 1;
    let mut level = tree.tree_level_count() - 1;
    let mut contained = false;

    loop {
        let end = (node_index + tree.tree_node_size()).min(tree.tree_level_bound(level));
        let is_leaf = node_index < tree.tree_num_items();

        if COVERED && contained {
            let (start, leaf_end) = leaf_group_range(tree, node_index, end, level);
            for pos in start..leaf_end {
                visitor(tree.tree_index(pos))?;
            }
        } else if MASKED && is_leaf {
            let mut start = node_index;
            while start < end {
                let stop = (start + MASK_CHUNK).min(end);
                let mut mask = overlap_mask_at(tree, start, stop, &overlaps);
                while mask != 0 {
                    visitor(tree.tree_index(start + mask.trailing_zeros() as usize))?;
                    mask &= mask - 1;
                }
                start = stop;
            }
        } else if MASKED {
            let child_level = level - 1;
            // Chunks from the back, bits from the top: children pop in forward order.
            let mut stop = end;
            while stop > node_index {
                let start = stop.saturating_sub(MASK_CHUNK).max(node_index);
                for_each_hit_rev(overlap_mask_at(tree, start, stop, &overlaps), |i| {
                    let pos = start + i;
                    let flag = if COVERED {
                        usize::from(contains(tree.tree_bounds(pos))) * frame::CONTAINED
                    } else {
                        0
                    };
                    stack.push(frame::pack(tree.tree_index(pos), child_level) | flag);
                });
                stop = start;
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
                let flag = if COVERED {
                    usize::from(contains(bounds)) * frame::CONTAINED
                } else {
                    0
                };
                stack.push(frame::pack(tree.tree_index(pos), child_level) | flag);
            }
        }

        match stack.pop() {
            Some(f) => {
                node_index = frame::node(f);
                level = frame::level(f);
                contained = COVERED && frame::contained(f);
            }
            None => return ControlFlow::Continue(()),
        }
    }
}

/// Tree levels whose resume points [`find_region`] keeps on the call stack. A
/// deeper tree needs node size 2 and more than 2^31 items, so the heap
/// fallback never runs on a tree that fits in memory at that node size.
const FIND_LEVELS: usize = 32;

/// Depth-first search for the items `overlaps` accepts, in the order
/// [`visit_region`] visits them, built for a visitor that stops early (`any`,
/// `first`).
///
/// It descends into a node's first overlapping child as soon as it finds it
/// and keeps the node's rest as the resume point of its level, where
/// [`visit_region`] pushes every overlapping child of a node before descending
/// into the first. One resume point per level, in a fixed array, also takes
/// the scratch stack off the call (kb:task/202).
///
/// `MASKED` picks the child test. Branching tests children one by one and
/// stops at the first hit, so a traversal that stops early skips the siblings
/// after each child on its path: on 100 000 boxes and node size 16, 37 child
/// tests per `first` against 63 on large windows and 50 against 72 on small
/// ones. Masked folds a node's tests into a bitmask, as [`visit_region`] does,
/// and keeps the untaken bits as the resume point: every sibling is tested,
/// but without a data-dependent branch per child, which is what a query that
/// finds nothing (and so skips nothing) pays for. [`find_region_switched`]
/// picks one per query.
///
/// Out of line off aarch64: inlined next to the other form behind that switch,
/// the form it picked ran 2-10% slower than alone on a Xeon
/// (`benches/paired_find_switch.rs`); one call per query costs less.
/// aarch64 runs the branching form alone and inlines the whole chain into the
/// caller: out of line, or under a wrapper that did not inline, the 3D `first`
/// on small windows ran about 20% slower on a Neoverse N2
/// (`benches/paired_mask_forms.rs`).
#[cfg_attr(not(target_arch = "aarch64"), inline(never))]
#[cfg_attr(target_arch = "aarch64", inline(always))]
pub(crate) fn find_region<const MASKED: bool, R, T, O, F>(
    tree: &T,
    overlaps: O,
    visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    F: FnMut(usize) -> ControlFlow<R>,
{
    if MASKED {
        find_region_masked::<FIND_LEVELS, R, T, O, F>(tree, overlaps, visitor)
    } else {
        find_region_branching::<FIND_LEVELS, R, T, O, F>(tree, overlaps, visitor)
    }
}

/// Whether a window over `root` expects fewer than `below` of `num_items`
/// hits, spread uniformly: the clipped overlap product against `below` root
/// areas (volumes in 3D), without a division. A flat root has zero area, so it
/// reads as "not fewer". Only the form of the traversal depends on the answer,
/// never its items, so a NaN coordinate costs at most the better form.
#[inline(always)]
pub(crate) fn expects_fewer_hits<const D: usize>(
    root_min: [f64; D],
    root_max: [f64; D],
    query_min: [f64; D],
    query_max: [f64; D],
    num_items: usize,
    below: f64,
) -> bool {
    let (mut overlap, mut size) = (num_items as f64, below);
    for d in 0..D {
        overlap *= (query_max[d].min(root_max[d]) - query_min[d].max(root_min[d])).max(0.0);
        size *= root_max[d] - root_min[d];
    }
    overlap < size
}

/// [`find_region`] in the form `masked` picks from the root box: the
/// frontends' switch on expected hits. Without `SWITCH` (a target where the
/// mask never pays) it is the branching form alone, with no root read in
/// front.
#[inline(always)]
pub(crate) fn find_region_switched<const SWITCH: bool, R, T, O, M, F>(
    tree: &T,
    overlaps: O,
    masked: M,
    visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    M: FnOnce(T::Bounds) -> bool,
    F: FnMut(usize) -> ControlFlow<R>,
{
    if !SWITCH {
        return find_region::<false, R, T, O, F>(tree, overlaps, visitor);
    }
    if tree.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }
    if masked(tree.tree_bounds(tree.tree_num_nodes() - 1)) {
        find_region::<true, R, T, O, F>(tree, overlaps, visitor)
    } else {
        find_region::<false, R, T, O, F>(tree, overlaps, visitor)
    }
}

/// The branching [`find_region`] with `LEVELS` resume points on the call
/// stack, so a test can reach the heap fallback on a tree that fits in memory.
#[inline(always)]
fn find_region_branching<const LEVELS: usize, R, T, O, F>(
    tree: &T,
    overlaps: O,
    mut visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    F: FnMut(usize) -> ControlFlow<R>,
{
    if tree.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }
    let top = tree.tree_level_count() - 1;
    let mut local = [(0usize, 0usize); LEVELS];
    let mut heap = Vec::new();
    // `resume[level]`: the untested `[pos, end)` rest of the node last entered
    // at `level`, written on the way down before any read on the way up.
    let resume: &mut [(usize, usize)] = if top < LEVELS {
        &mut local
    } else {
        heap.resize(top + 1, (0, 0));
        &mut heap
    };
    let node_size = tree.tree_node_size();
    let mut pos = tree.tree_num_nodes() - 1;
    let mut end = pos + 1;
    let mut level = top;
    'node: loop {
        if level == 0 {
            for p in pos..end {
                if overlaps(tree.tree_bounds(p)) {
                    visitor(tree.tree_index(p))?;
                }
            }
        } else {
            while pos < end {
                let p = pos;
                pos += 1;
                if overlaps(tree.tree_bounds(p)) {
                    resume[level] = (pos, end);
                    level -= 1;
                    pos = tree.tree_index(p);
                    end = (pos + node_size).min(tree.tree_level_bound(level));
                    continue 'node;
                }
            }
        }
        // This node is done: resume the nearest level above with children left.
        loop {
            level += 1;
            if level > top {
                return ControlFlow::Continue(());
            }
            (pos, end) = resume[level];
            if pos < end {
                continue 'node;
            }
        }
    }
}

/// The masked [`find_region`]; `LEVELS` as in [`find_region_branching`].
#[inline(always)]
fn find_region_masked<const LEVELS: usize, R, T, O, F>(
    tree: &T,
    overlaps: O,
    mut visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    O: Fn(T::Bounds) -> bool,
    F: FnMut(usize) -> ControlFlow<R>,
{
    if tree.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }
    let top = tree.tree_level_count() - 1;
    let mut local = [(0usize, 0u64, 0usize); LEVELS];
    let mut heap = Vec::new();
    // `resume[level]`: `(base, mask, end)` of the node last entered at
    // `level`: the untaken hits of its chunk at `base` and the chunks after
    // it up to `end`, written on the way down before any read on the way up.
    let resume: &mut [(usize, u64, usize)] = if top < LEVELS {
        &mut local
    } else {
        heap.resize(top + 1, (0, 0, 0));
        &mut heap
    };
    let node_size = tree.tree_node_size();
    let mut pos = tree.tree_num_nodes() - 1;
    let mut end = pos + 1;
    let mut level = top;
    loop {
        if level == 0 {
            let mut start = pos;
            while start < end {
                let stop = (start + MASK_CHUNK).min(end);
                let mut mask = tree.tree_mask(start, stop, &overlaps);
                while mask != 0 {
                    visitor(tree.tree_index(start + mask.trailing_zeros() as usize))?;
                    mask &= mask - 1;
                }
                start = stop;
            }
            level = 1;
        } else {
            let stop = (pos + MASK_CHUNK).min(end);
            resume[level] = (pos, tree.tree_mask(pos, stop, &overlaps), end);
        }
        // Take the next hit at `level`, or climb to the nearest level with one.
        loop {
            if level > top {
                return ControlFlow::Continue(());
            }
            let (base, mask, node_end) = resume[level];
            if mask != 0 {
                resume[level].1 = mask & (mask - 1);
                pos = tree.tree_index(base + mask.trailing_zeros() as usize);
                level -= 1;
                end = (pos + node_size).min(tree.tree_level_bound(level));
                break;
            }
            let next = base + MASK_CHUNK;
            if next < node_end {
                let stop = (next + MASK_CHUNK).min(node_end);
                resume[level] = (next, tree.tree_mask(next, stop, &overlaps), node_end);
            } else {
                level += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use crate::{Box2D, Index2DBuilder};

    use super::{find_region_branching, find_region_masked};

    /// The heap fallback for trees deeper than the local array and nodes
    /// wider than one mask chunk (node size 100, 7000 items: 70 children
    /// under the root), visit what `visit` does, in its order.
    #[test]
    fn find_region_forms_match_visit() {
        for node_size in [2, 100] {
            let mut b = Index2DBuilder::new(7000).node_size(node_size);
            for i in 0..7000 {
                let x = f64::from((i * 37) % 1001);
                let y = f64::from((i * 53) % 997);
                b.add(Box2D::new(x, y, x + 3.0, y + 2.0));
            }
            let index = b.finish().unwrap();
            for q in [
                Box2D::new(10.0, 10.0, 30.0, 40.0),
                Box2D::new(0.0, 0.0, 2000.0, 2000.0),
                Box2D::new(900.0, 900.0, 1000.0, 1000.0),
                Box2D::new(5000.0, 5000.0, 5001.0, 5001.0),
            ] {
                let mut want = Vec::new();
                let _ = index.visit(q, |i| {
                    want.push(i);
                    ControlFlow::<()>::Continue(())
                });
                let overlaps = |b: Box2D| b.overlaps(q);
                for got in [
                    collect(|f| find_region_branching::<2, (), _, _, _>(&index, overlaps, f)),
                    collect(|f| find_region_branching::<32, (), _, _, _>(&index, overlaps, f)),
                    collect(|f| find_region_masked::<2, (), _, _, _>(&index, overlaps, f)),
                    collect(|f| find_region_masked::<32, (), _, _, _>(&index, overlaps, f)),
                ] {
                    assert_eq!(got, want, "{node_size} {q:?}");
                }
            }
        }
    }

    #[test]
    fn expected_hits_switch_reads_the_covered_share() {
        let root = ([0.0, 0.0], [100.0, 100.0]);
        // A 10 x 10 window over 10 000 items spread on 100 x 100 expects 100.
        let fewer = |q: ([f64; 2], [f64; 2]), below| {
            super::expects_fewer_hits(root.0, root.1, q.0, q.1, 10_000, below)
        };
        assert!(fewer(([0.0, 0.0], [10.0, 10.0]), 101.0));
        assert!(!fewer(([0.0, 0.0], [10.0, 10.0]), 99.0));
        // Clipped to the root; a miss expects nothing.
        assert!(fewer(([-50.0, 0.0], [10.0, 10.0]), 101.0));
        assert!(fewer(([200.0, 200.0], [300.0, 300.0]), 0.5));
        // A flat root reads as "not fewer".
        assert!(!super::expects_fewer_hits(
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            10,
            1e9
        ));
    }

    fn collect(
        run: impl FnOnce(&mut dyn FnMut(usize) -> ControlFlow<()>) -> ControlFlow<()>,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        let _ = run(&mut |i| {
            out.push(i);
            ControlFlow::Continue(())
        });
        out
    }
}
