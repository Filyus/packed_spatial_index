//! Pairwise spatial joins: report every intersecting pair of items between two
//! packed trees, or within one tree (`self_join`), or every pair within a
//! distance bound (`join_within`).
//!
//! The traversal descends both trees simultaneously from the pair of roots. One
//! bounds test between two internal entries prunes their whole subtree pair, so
//! the cost scales with the output size instead of running one full search per
//! item. The generic core works over [`TreeAccess`], a minimal accessor view of
//! the packed layout shared by every f64 index and byte-view type, and over a
//! [`PairTest`] that decides which entry pairs can hold output.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

use crate::geometry::{Box2D, Box3D};
use crate::range::visit_region;
use crate::tree_access::{TreeAccess, leaf_range};

/// Which entry pairs of a dual-tree descent can hold output pairs.
///
/// `keeps` is the prune test: a pair failing it holds no output and is dropped.
/// It must be a lower bound on the distance (or the exact test) between any
/// item pair under the two entries — items lie inside their node boxes, and
/// shrinking a box can only push it farther from another box.
///
/// `covers(leaf, subtree)` is the whole-subtree fast path: every item under
/// `subtree` pairs with the single item `leaf`. For overlap that is the leaf
/// containing the subtree box; for a distance bound it is the *farthest-corner*
/// distance being within `max_distance`, because items inside the subtree box can
/// sit anywhere in it — the plain box distance (a lower bound) is not enough.
pub(crate) trait PairTest<B: Copy> {
    fn keeps(&self, a: B, b: B) -> bool;
    fn covers(&self, leaf: B, subtree: B) -> bool;
}

/// Plain box intersection: the `join` / `self_join` semantics.
pub(crate) struct OverlapTest;

impl PairTest<Box2D> for OverlapTest {
    #[inline]
    fn keeps(&self, a: Box2D, b: Box2D) -> bool {
        a.overlaps(b)
    }
    #[inline]
    fn covers(&self, leaf: Box2D, subtree: Box2D) -> bool {
        leaf.contains(subtree)
    }
}

impl PairTest<Box3D> for OverlapTest {
    #[inline]
    fn keeps(&self, a: Box3D, b: Box3D) -> bool {
        a.overlaps(b)
    }
    #[inline]
    fn covers(&self, leaf: Box3D, subtree: Box3D) -> bool {
        leaf.contains(subtree)
    }
}

/// Box-to-box distance at most `max_distance`: the `join_within` semantics.
#[derive(Clone, Copy)]
pub(crate) struct DistanceTest {
    eps_squared: f64,
}

impl DistanceTest {
    /// A negative or NaN `max_distance` matches nothing (distances are never
    /// negative), which falls out of comparing against `-1.0`.
    #[inline]
    pub(crate) fn new(max_distance: f64) -> Self {
        Self {
            eps_squared: if max_distance >= 0.0 {
                max_distance * max_distance
            } else {
                -1.0
            },
        }
    }
}

/// Square of the farthest-corner distance between two boxes: an upper bound on
/// the distance between any point of one and any point of the other.
///
/// Join-specific math, kept local: only the leaf fast path needs the far
/// corner, and `geometry` carries no upper-bound primitive.
#[inline]
fn far_distance_squared_2d(a: Box2D, b: Box2D) -> f64 {
    let dx = (b.max_x - a.min_x).max(a.max_x - b.min_x);
    let dy = (b.max_y - a.min_y).max(a.max_y - b.min_y);
    dx * dx + dy * dy
}

#[inline]
fn far_distance_squared_3d(a: Box3D, b: Box3D) -> f64 {
    let dx = (b.max_x - a.min_x).max(a.max_x - b.min_x);
    let dy = (b.max_y - a.min_y).max(a.max_y - b.min_y);
    let dz = (b.max_z - a.min_z).max(a.max_z - b.min_z);
    dx * dx + dy * dy + dz * dz
}

impl PairTest<Box2D> for DistanceTest {
    #[inline]
    fn keeps(&self, a: Box2D, b: Box2D) -> bool {
        a.distance_squared_to_box(b) <= self.eps_squared
    }
    #[inline]
    fn covers(&self, leaf: Box2D, subtree: Box2D) -> bool {
        far_distance_squared_2d(leaf, subtree) <= self.eps_squared
    }
}

impl PairTest<Box3D> for DistanceTest {
    #[inline]
    fn keeps(&self, a: Box3D, b: Box3D) -> bool {
        a.distance_squared_to_box(b) <= self.eps_squared
    }
    #[inline]
    fn covers(&self, leaf: Box3D, subtree: Box3D) -> bool {
        far_distance_squared_3d(leaf, subtree) <= self.eps_squared
    }
}

/// One traversal step: expand the higher-level side of the entry pair, emit
/// leaf/leaf pairs inline, and push surviving pairs onto the stack.
///
/// Invariants: the two entry bounds pass `test.keeps`, and
/// `max(a_level, b_level) >= 1` (both-leaf pairs are emitted by the caller and
/// never reach the stack).
#[inline]
#[allow(clippy::too_many_arguments)]
fn expand_pair<R, T, U, P, F>(
    a: &T,
    b: &U,
    test: &P,
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
    P: PairTest<T::Bounds>,
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
            if !test.keeps(bounds, b_bounds) {
                continue;
            }
            if child_level == 0 {
                if b_level == 0 {
                    visitor(a.tree_index(pos), b.tree_index(b_pos))?;
                } else if test.covers(bounds, b_bounds) {
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
            } else if b_level == 0 && test.covers(b_bounds, bounds) {
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
            if !test.keeps(a_bounds, bounds) {
                continue;
            }
            if child_level == 0 {
                if a_level == 0 {
                    visitor(a.tree_index(a_pos), b.tree_index(pos))?;
                } else if test.covers(bounds, a_bounds) {
                    let item_b = b.tree_index(pos);
                    let (s, e) = leaf_range(a, a_pos, a_level);
                    for a_leaf in s..e {
                        visitor(a.tree_index(a_leaf), item_b)?;
                    }
                } else {
                    stack.push((a_pos, a_level, pos, 0));
                }
            } else if a_level == 0 && test.covers(a_bounds, bounds) {
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

/// Visit every pair `(i, j)` where item `i` of `a` pairs with item `j` of `b`
/// under `test`. Pair order is traversal order and is not part of the API.
pub(crate) fn join_core<R, T, U, P, F>(a: &T, b: &U, test: P, mut visitor: F) -> ControlFlow<R>
where
    T: TreeAccess,
    U: TreeAccess<Bounds = T::Bounds>,
    P: PairTest<T::Bounds>,
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
    if !test.keeps(a.tree_bounds(a_pos), b.tree_bounds(b_pos)) {
        return ControlFlow::Continue(());
    }

    let mut stack: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(64);
    loop {
        expand_pair(
            a,
            b,
            &test,
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

/// Visit every unordered pair of distinct items within `tree` that pair under
/// `test`, each pair exactly once. The order of the two ids within a pair and
/// the pair order are traversal order and are not part of the API.
pub(crate) fn self_join_core<R, T, P, F>(tree: &T, test: P, mut visitor: F) -> ControlFlow<R>
where
    T: TreeAccess,
    P: PairTest<T::Bounds>,
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
                    if !test.keeps(bounds_i, tree.tree_bounds(j)) {
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
                &test,
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

/// Visit every item of `tree` whose box lies within `max_distance` of `query`: the
/// radius query, single-tree sibling of the `join_within` family.
///
/// Node prune and whole-subtree accept are deliberately different tests. A node
/// is descended when its box is within `max_distance` — items sit inside their node
/// box, and shrinking a box only pushes it farther from an external query, so
/// the node distance is a lower bound and prunes soundly. It never *accepts*
/// for the same reason: the sufficient condition is the node's *farthest*
/// corner being within `max_distance`, which is what `covers` tests.
///
/// A cheaper node prune was tried and lost: overlap against the query grown by
/// `max_distance` is a valid necessary condition (the L-infinity ball contains the
/// L2 one) and costs four compares where the exact distance costs two axis
/// gaps and two multiplies, but the extra subtrees it descends cost 1.2x-2.2x
/// more than the predicate saves across uniform and clustered data at every
/// radius measured.
///
/// Item order is traversal order and is not part of the API.
#[inline]
pub(crate) fn within_core<R, T, P, F>(
    tree: &T,
    query: T::Bounds,
    test: P,
    stack: &mut Vec<usize>,
    visitor: F,
) -> ControlFlow<R>
where
    T: TreeAccess,
    P: PairTest<T::Bounds>,
    F: FnMut(usize) -> ControlFlow<R>,
{
    visit_region(
        tree,
        stack,
        |node| test.keeps(node, query),
        |node| test.covers(query, node),
        visitor,
    )
}

/// Box bounds a closest-pair descent can measure between.
///
/// Squared, because the descent only ever compares distances against each
/// other and against the running best — the ordering is the same and the
/// square root is paid once, on the one answer that comes back.
pub(crate) trait PairDistance: Copy {
    fn distance_squared_between(self, other: Self) -> f64;
}

impl PairDistance for Box2D {
    #[inline]
    fn distance_squared_between(self, other: Self) -> f64 {
        self.distance_squared_to_box(other)
    }
}

impl PairDistance for Box3D {
    #[inline]
    fn distance_squared_between(self, other: Self) -> f64 {
        self.distance_squared_to_box(other)
    }
}

/// One entry pair on the closest-pair frontier, ordered so [`BinaryHeap`]
/// (a max-heap) pops the *smallest* lower bound first.
#[derive(Clone, Copy)]
struct PairState {
    dist_squared: f64,
    a_pos: usize,
    a_level: usize,
    b_pos: usize,
    b_level: usize,
}

impl PartialEq for PairState {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PairState {}

impl Ord for PairState {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed on distance to make the max-heap a min-heap. The position
        // tie-break only keeps the order total and deterministic; which of two
        // equally distant pairs is expanded first does not affect the answer's
        // distance.
        other
            .dist_squared
            .total_cmp(&self.dist_squared)
            .then_with(|| other.a_pos.cmp(&self.a_pos))
            .then_with(|| other.b_pos.cmp(&self.b_pos))
    }
}

impl PartialOrd for PairState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The running best pair of a closest-pair descent.
struct Best {
    dist_squared: f64,
    pair: Option<(usize, usize)>,
}

impl Best {
    #[inline]
    fn offer(&mut self, dist_squared: f64, i: usize, j: usize) {
        if dist_squared < self.dist_squared {
            self.dist_squared = dist_squared;
            self.pair = Some((i, j));
        }
    }

    #[inline]
    fn finish(self) -> Option<(usize, usize, f64)> {
        let (i, j) = self.pair?;
        Some((i, j, self.dist_squared.sqrt()))
    }
}

/// Seed `best` with a real pair before the frontier opens, by walking a sample
/// of `a`'s items down `b` greedily — at each node taking the child closest to
/// the item's box.
///
/// Costs one root-to-leaf walk per sample and changes no answer: every offer
/// is an actual item pair, so it can only start `best` lower than infinity.
/// That matters because the descent prunes against `best`, which otherwise
/// stays infinite until the first leaf-leaf pair is popped — on dense data
/// with no overlapping pair that is deep into the traversal, and until then
/// nothing is pruned at all.
fn seed_best<T, U>(a: &T, b: &U, best: &mut Best)
where
    T: TreeAccess,
    U: TreeAccess<Bounds = T::Bounds>,
    T::Bounds: PairDistance,
{
    const SAMPLES: usize = 16;
    let n = a.tree_num_items();
    // Spread the samples across the leaf array rather than taking a prefix:
    // the leaves are in spatial-sort order, so a prefix is one corner of `a`.
    let step = (n / SAMPLES).max(1);
    for a_pos in (0..n).step_by(step) {
        let bounds = a.tree_bounds(a_pos);
        let mut pos = b.tree_num_nodes() - 1;
        let mut level = b.tree_level_count() - 1;
        while level > 0 {
            let child_level = level - 1;
            let start = b.tree_index(pos);
            let end = (start + b.tree_node_size()).min(b.tree_level_bound(child_level));
            let mut nearest = start;
            let mut nearest_dist = f64::INFINITY;
            for child in start..end {
                let dist = bounds.distance_squared_between(b.tree_bounds(child));
                if dist < nearest_dist {
                    nearest_dist = dist;
                    nearest = child;
                }
            }
            pos = nearest;
            level = child_level;
        }
        best.offer(
            bounds.distance_squared_between(b.tree_bounds(pos)),
            a.tree_index(a_pos),
            b.tree_index(pos),
        );
        if best.dist_squared == 0.0 {
            // Nothing can beat zero, so the remaining samples cannot tighten
            // anything and the descent will exit on its first pop.
            return;
        }
    }
}

/// Seed `best` for the self case from neighbours in the leaf array.
///
/// The leaves are in spatial-sort order, so consecutive entries are usually
/// close; a sweep of adjacent pairs is one pass over the leaf bounds and
/// typically lands within a small factor of the answer. Same guarantee as
/// [`seed_best`]: every offer is a real pair of distinct items, so it can only
/// tighten the bound the descent prunes against.
fn seed_self_best<T>(tree: &T, best: &mut Best)
where
    T: TreeAccess,
    T::Bounds: PairDistance,
{
    let n = tree.tree_num_items();
    let mut previous = tree.tree_bounds(0);
    for pos in 1..n {
        let bounds = tree.tree_bounds(pos);
        best.offer(
            previous.distance_squared_between(bounds),
            tree.tree_index(pos - 1),
            tree.tree_index(pos),
        );
        if best.dist_squared == 0.0 {
            // Two items overlap: nothing can beat zero, so stop sweeping.
            return;
        }
        previous = bounds;
    }
}

/// The closest pair of items between `a` and `b`, as `(item_a, item_b,
/// distance)`, or `None` when either tree is empty.
///
/// Best-first over *entry pairs* rather than the stack descent the joins use:
/// the frontier is a heap keyed by the pair's box-to-box distance, which is a
/// lower bound on any item pair beneath it, so the first time the heap's head
/// is no closer than the best pair found the answer is settled and everything
/// still queued can be dropped unexamined. That early exit is the whole point —
/// there is one answer, not a stream, and a `join_within` would have to guess
/// an `max_distance` that contains it.
///
/// Ties: the pair reported among several at the same distance is traversal
/// order and is not part of the API.
pub(crate) fn closest_pair_core<T, U>(a: &T, b: &U) -> Option<(usize, usize, f64)>
where
    T: TreeAccess,
    U: TreeAccess<Bounds = T::Bounds>,
    T::Bounds: PairDistance,
{
    if a.tree_num_items() == 0 || b.tree_num_items() == 0 {
        return None;
    }

    let a_root = a.tree_num_nodes() - 1;
    let b_root = b.tree_num_nodes() - 1;
    let mut best = Best {
        dist_squared: f64::INFINITY,
        pair: None,
    };
    seed_best(a, b, &mut best);
    let mut heap: BinaryHeap<PairState> = BinaryHeap::with_capacity(64);
    heap.push(PairState {
        dist_squared: a
            .tree_bounds(a_root)
            .distance_squared_between(b.tree_bounds(b_root)),
        a_pos: a_root,
        a_level: a.tree_level_count() - 1,
        b_pos: b_root,
        b_level: b.tree_level_count() - 1,
    });

    while let Some(state) = heap.pop() {
        // The head is the smallest lower bound left, so nothing queued can
        // beat the best already found.
        if state.dist_squared >= best.dist_squared {
            break;
        }
        if state.a_level == 0 && state.b_level == 0 {
            best.offer(
                state.dist_squared,
                a.tree_index(state.a_pos),
                b.tree_index(state.b_pos),
            );
            continue;
        }
        expand_closest_pair(a, b, state, &best, &mut heap);
    }
    best.finish()
}

/// Expand the higher-level side of `state` onto the frontier, dropping child
/// pairs that already cannot beat `best`.
#[inline]
fn expand_closest_pair<T, U>(
    a: &T,
    b: &U,
    state: PairState,
    best: &Best,
    heap: &mut BinaryHeap<PairState>,
) where
    T: TreeAccess,
    U: TreeAccess<Bounds = T::Bounds>,
    T::Bounds: PairDistance,
{
    if state.a_level >= state.b_level {
        let child_level = state.a_level - 1;
        let start = a.tree_index(state.a_pos);
        let end = (start + a.tree_node_size()).min(a.tree_level_bound(child_level));
        let b_bounds = b.tree_bounds(state.b_pos);
        for pos in start..end {
            let dist_squared = a.tree_bounds(pos).distance_squared_between(b_bounds);
            if dist_squared >= best.dist_squared {
                continue;
            }
            heap.push(PairState {
                dist_squared,
                a_pos: pos,
                a_level: child_level,
                b_pos: state.b_pos,
                b_level: state.b_level,
            });
        }
    } else {
        let child_level = state.b_level - 1;
        let start = b.tree_index(state.b_pos);
        let end = (start + b.tree_node_size()).min(b.tree_level_bound(child_level));
        let a_bounds = a.tree_bounds(state.a_pos);
        for pos in start..end {
            let dist_squared = a_bounds.distance_squared_between(b.tree_bounds(pos));
            if dist_squared >= best.dist_squared {
                continue;
            }
            heap.push(PairState {
                dist_squared,
                a_pos: state.a_pos,
                a_level: state.a_level,
                b_pos: pos,
                b_level: child_level,
            });
        }
    }
}

/// The closest pair of *distinct* items within one tree, as `(i, j, distance)`,
/// or `None` for a tree with fewer than two items.
///
/// Same frontier as [`closest_pair_core`], with the diagonal handled the way
/// [`self_join_core`] handles it: an entry paired with itself expands into
/// child pairs `i <= j`, so each unordered pair is reached once and an item is
/// never paired with itself. The order of the two ids within the pair, and
/// which of several equally close pairs is reported, are traversal order and
/// not part of the API.
pub(crate) fn self_closest_pair_core<T>(tree: &T) -> Option<(usize, usize, f64)>
where
    T: TreeAccess,
    T::Bounds: PairDistance,
{
    if tree.tree_num_items() < 2 {
        return None;
    }

    let root = tree.tree_num_nodes() - 1;
    let root_level = tree.tree_level_count() - 1;
    let mut best = Best {
        dist_squared: f64::INFINITY,
        pair: None,
    };
    seed_self_best(tree, &mut best);
    let mut heap: BinaryHeap<PairState> = BinaryHeap::with_capacity(64);
    heap.push(PairState {
        dist_squared: 0.0,
        a_pos: root,
        a_level: root_level,
        b_pos: root,
        b_level: root_level,
    });

    while let Some(state) = heap.pop() {
        if state.dist_squared >= best.dist_squared {
            break;
        }
        let diagonal = state.a_pos == state.b_pos && state.a_level == state.b_level;
        if state.a_level == 0 && state.b_level == 0 {
            // The diagonal at leaf level is one item against itself: not a pair.
            if !diagonal {
                best.offer(
                    state.dist_squared,
                    tree.tree_index(state.a_pos),
                    tree.tree_index(state.b_pos),
                );
            }
            continue;
        }
        if diagonal {
            let child_level = state.a_level - 1;
            let start = tree.tree_index(state.a_pos);
            let end = (start + tree.tree_node_size()).min(tree.tree_level_bound(child_level));
            for i in start..end {
                let bounds_i = tree.tree_bounds(i);
                // `(i, i)` carries the pairs *within* that child; skip it at
                // leaf level, where it would be an item against itself.
                if child_level > 0 {
                    heap.push(PairState {
                        dist_squared: 0.0,
                        a_pos: i,
                        a_level: child_level,
                        b_pos: i,
                        b_level: child_level,
                    });
                }
                for j in (i + 1)..end {
                    let dist_squared = bounds_i.distance_squared_between(tree.tree_bounds(j));
                    if dist_squared >= best.dist_squared {
                        continue;
                    }
                    heap.push(PairState {
                        dist_squared,
                        a_pos: i,
                        a_level: child_level,
                        b_pos: j,
                        b_level: child_level,
                    });
                }
            }
        } else {
            expand_closest_pair(tree, tree, state, &best, &mut heap);
        }
    }
    best.finish()
}

/// Is there an item of `tree` pairing with `bounds` under `test`? One pruned
/// descent: a node box failing the test drops its whole subtree (items inside
/// it are farther still), and leaves are tested exactly. There is no fast
/// accept — items inside a passing node box may be farther than the node box
/// itself is.
pub(crate) fn any_within_core<T: TreeAccess, P>(tree: &T, bounds: T::Bounds, test: P) -> bool
where
    P: PairTest<T::Bounds> + Copy,
{
    let n = tree.tree_num_items();
    if n == 0 {
        return false;
    }

    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(64);
    stack.push((tree.tree_num_nodes() - 1, tree.tree_level_count() - 1));
    while let Some((pos, level)) = stack.pop() {
        if level == 0 {
            // A leaf holds one item, so the bounds are the item's own box.
            if test.keeps(tree.tree_bounds(pos), bounds) {
                return true;
            }
            continue;
        }
        let child_level = level - 1;
        let start = tree.tree_index(pos);
        let end = (start + tree.tree_node_size()).min(tree.tree_level_bound(child_level));
        for child in start..end {
            if !test.keeps(tree.tree_bounds(child), bounds) {
                continue;
            }
            if child_level == 0 {
                return true;
            }
            stack.push((child, child_level));
        }
    }
    false
}

/// Visit every item of `a` that pairs with no item of `b` under `test`: the
/// anti-join. One pruned descent into `b` per item of `a`.
pub(crate) fn anti_join_core<R, T, U, P, F>(a: &T, b: &U, test: P, mut visitor: F) -> ControlFlow<R>
where
    T: TreeAccess,
    U: TreeAccess<Bounds = T::Bounds>,
    P: PairTest<T::Bounds> + Copy,
    F: FnMut(usize) -> ControlFlow<R>,
{
    for pos in 0..a.tree_num_items() {
        if !any_within_core(b, a.tree_bounds(pos), test) {
            visitor(a.tree_index(pos))?;
        }
    }
    ControlFlow::Continue(())
}

/// Label every item of `tree` with the smallest item id in its component of
/// the proximity graph `test` defines. An item with no pair is its own label.
/// Deterministic: the label of a component does not depend on the order the
/// pairs arrive in.
pub(crate) fn self_join_components_core<T: TreeAccess, P: PairTest<T::Bounds>>(
    tree: &T,
    test: P,
) -> Vec<usize> {
    let n = tree.tree_num_items();
    let mut parent: Vec<usize> = (0..n).collect();
    if n < 2 {
        return parent;
    }

    fn find(parent: &[usize], mut x: usize) -> usize {
        while parent[x] != x {
            x = parent[x];
        }
        x
    }
    // Attach under the smaller root, so a component's root stays its minimum
    // id no matter which order the pairs arrive in.
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra < rb {
            parent[rb] = ra;
        } else if rb < ra {
            parent[ra] = rb;
        }
    }

    let _ = self_join_core::<(), T, P, _>(tree, test, |i, j| {
        union(&mut parent, i, j);
        ControlFlow::<()>::Continue(())
    });
    // Path-halving final pass; the roots are already the minimum ids.
    for x in 0..n {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut y = x;
        while parent[y] != root {
            let next = parent[y];
            parent[y] = root;
            y = next;
        }
    }
    parent
}
