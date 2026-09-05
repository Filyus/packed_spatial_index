//! Ordered region traversal: emit the items of a region in nondecreasing order
//! of a caller-supplied key.
//!
//! The sibling of [`crate::range::visit_region`] — same `TreeAccess` view of the
//! tree, same region predicate — but a best-first descent
//! ([`crate::neighbors::metric_knn`]) instead of a depth-first stack, so the
//! caller can stop on a budget instead of filtering the output. The region test
//! rides in as the kernel's prune channel: a node that fails it yields `None`
//! and neither it nor its subtree is visited, which is sound because a child box
//! lies inside its parent.
//!
//! The descent is scalar on every frontend: a heap yields one node at a time, so
//! there is nothing for a SIMD kernel to widen.

use std::collections::BinaryHeap;
use std::ops::ControlFlow;

use crate::config::DEFAULT_NEIGHBOR_QUEUE_CAPACITY;
use crate::neighbors::metric_knn;
use crate::tree_access::TreeAccess;

/// End of the level containing `node`, by binary search over the level bounds.
/// Mirrors [`crate::traversal::upper_bound_level`] without needing the slice, so
/// that byte views (which compute their bounds) work too.
#[inline]
fn level_end_of<T: TreeAccess>(tree: &T, node: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = tree.tree_level_count() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if tree.tree_level_bound(mid) > node {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    tree.tree_level_bound(lo)
}

/// Collect up to `max_results` items of the region in nondecreasing `key` order
/// into `out` (cleared first).
pub(crate) fn collect_ordered<T: TreeAccess>(
    tree: &T,
    overlaps: impl Fn(T::Bounds) -> bool,
    key: impl Fn(T::Bounds) -> f64,
    max_results: usize,
    max_key: f64,
    out: &mut Vec<usize>,
) {
    out.clear();
    if tree.tree_num_items() == 0 {
        return;
    }
    let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    metric_knn::collect_neighbors(
        tree.tree_num_nodes(),
        tree.tree_num_items(),
        tree.tree_node_size(),
        |node| level_end_of(tree, node),
        |pos| tree.tree_index(pos),
        max_results,
        max_key,
        |pos| {
            let bounds = tree.tree_bounds(pos);
            overlaps(bounds).then(|| key(bounds))
        },
        out,
        &mut queue,
    );
}

/// Visit the items of the region in nondecreasing `key` order; the visitor
/// receives the key and may break early.
pub(crate) fn visit_ordered<T: TreeAccess, B>(
    tree: &T,
    overlaps: impl Fn(T::Bounds) -> bool,
    key: impl Fn(T::Bounds) -> f64,
    max_key: f64,
    visitor: &mut impl FnMut(usize, f64) -> ControlFlow<B>,
) -> ControlFlow<B> {
    if tree.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }
    let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    metric_knn::visit_neighbors(
        tree.tree_num_nodes(),
        tree.tree_num_items(),
        tree.tree_node_size(),
        |node| level_end_of(tree, node),
        |pos| tree.tree_index(pos),
        max_key,
        |pos| {
            let bounds = tree.tree_bounds(pos);
            overlaps(bounds).then(|| key(bounds))
        },
        &mut queue,
        visitor,
    )
}

// ---------------------------------------------------------------- pick

/// Pick key for one box: `None` on a region miss; `(0.0, entry_t)` when the
/// ray segment passes through the box; otherwise the exact squared
/// ray-to-box distance (a lower bound on the distance to anything inside)
/// and an infinite `t`.
#[inline]
pub(crate) fn pick_key<Q: crate::geometry::Overlaps3D>(
    region: &Q,
    ray: crate::ray::Ray3D,
    b: crate::geometry::Box3D,
) -> Option<(f64, f64)> {
    if !region.overlaps_box(b) {
        return None;
    }
    Some(match ray.enter_t(b) {
        Some(t) => (0.0, t),
        None => (ray.distance_squared_to_box(b), f64::INFINITY),
    })
}

/// A pick candidate: the item's index plus the two components of its ordering
/// key — the squared perpendicular distance from the pick ray to the item's
/// box (0.0 when the ray passes through the box) and the ray's entry parameter
/// `t` ([`f64::INFINITY`] when the ray misses the box). Both are in the ray's
/// direction-length units and are lower bounds on the same quantities of the
/// geometry inside the box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickHit3D {
    /// The item's index (insertion id).
    pub index: usize,
    /// Squared distance from the pick ray to the item's box.
    pub distance_squared: f64,
    /// Ray entry parameter of the item's box, in direction-length units.
    pub entry_t: f64,
}

/// Best-first heap state for pick traversal: lexicographic
/// (perpendicular distance², entry t), then leaves before internal entries,
/// then lower position (Hilbert leaf order) — the same tie discipline as
/// [`crate::neighbors::NeighborState`].
#[derive(PartialEq)]
pub(crate) struct PickState {
    index: usize,
    is_leaf: bool,
    perp2: f64,
    entry_t: f64,
}

impl Eq for PickState {}

impl Ord for PickState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .perp2
            .total_cmp(&self.perp2)
            .then_with(|| other.entry_t.total_cmp(&self.entry_t))
            .then_with(|| self.is_leaf.cmp(&other.is_leaf))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for PickState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Best-first pick descent over the shared tree layout, mirroring
/// [`metric_knn::collect_neighbors`] but with a lexicographic two-component
/// key. The key closure returns `None` for a pruned entry (region miss) and
/// `(perp2, entry_t)` for a kept one.
pub(crate) fn collect_pick<T: TreeAccess<Bounds = crate::geometry::Box3D>>(
    tree: &T,
    max_results: usize,
    key: impl Fn(T::Bounds) -> Option<(f64, f64)>,
    results: &mut Vec<PickHit3D>,
    queue: &mut BinaryHeap<PickState>,
) {
    queue.clear();
    results.clear();
    if tree.tree_num_items() == 0 || max_results == 0 {
        return;
    }
    let num_nodes = tree.tree_num_nodes();
    let node_size = tree.tree_node_size();
    let mut node_index = num_nodes - 1;
    loop {
        let end = (node_index + node_size).min(level_end_of(tree, node_index));
        let is_leaf = node_index < tree.tree_num_items();
        for pos in node_index..end {
            if let Some((perp2, entry_t)) = key(tree.tree_bounds(pos)) {
                queue.push(PickState {
                    index: tree.tree_index(pos),
                    is_leaf,
                    perp2,
                    entry_t,
                });
            }
        }
        let mut continue_search = false;
        while let Some(state) = queue.pop() {
            if state.is_leaf {
                results.push(PickHit3D {
                    index: state.index,
                    distance_squared: state.perp2,
                    entry_t: state.entry_t,
                });
                if results.len() == max_results {
                    return;
                }
            } else {
                node_index = state.index;
                continue_search = true;
                break;
            }
        }
        if !continue_search {
            return;
        }
    }
}

/// [`collect_pick`] for a visitor; the visitor may break early. Streaming:
/// hits are visited as the heap pops them, so a break stops the traversal.
pub(crate) fn visit_pick<T, B>(
    tree: &T,
    max_results: usize,
    key: impl Fn(crate::geometry::Box3D) -> Option<(f64, f64)>,
    visitor: &mut impl FnMut(PickHit3D) -> ControlFlow<B>,
) -> ControlFlow<B>
where
    T: TreeAccess<Bounds = crate::geometry::Box3D>,
{
    if tree.tree_num_items() == 0 || max_results == 0 {
        return ControlFlow::Continue(());
    }
    let num_nodes = tree.tree_num_nodes();
    let node_size = tree.tree_node_size();
    let mut queue: BinaryHeap<PickState> =
        BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    let mut node_index = num_nodes - 1;
    let mut visited = 0usize;
    loop {
        let end = (node_index + node_size).min(level_end_of(tree, node_index));
        let is_leaf = node_index < tree.tree_num_items();
        for pos in node_index..end {
            if let Some((perp2, entry_t)) = key(tree.tree_bounds(pos)) {
                queue.push(PickState {
                    index: tree.tree_index(pos),
                    is_leaf,
                    perp2,
                    entry_t,
                });
            }
        }
        let mut continue_search = false;
        while let Some(state) = queue.pop() {
            if state.is_leaf {
                visited += 1;
                let hit = PickHit3D {
                    index: state.index,
                    distance_squared: state.perp2,
                    entry_t: state.entry_t,
                };
                if let ControlFlow::Break(b) = visitor(hit) {
                    return ControlFlow::Break(b);
                }
                if visited == max_results {
                    return ControlFlow::Continue(());
                }
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
