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
