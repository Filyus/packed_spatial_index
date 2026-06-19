/// Minimal read-only view of a packed tree, implemented by f64 index and view
/// types that share the same flat layout.
///
/// Positions (`pos`) address the flat entry array: `[0, num_items)` are leaf
/// items, higher positions are internal entries, and the root entry is last.
/// `tree_index` returns the original item id for a leaf position and the first
/// child position for an internal entry.
pub(crate) trait TreeAccess {
    type Bounds: Copy;

    fn tree_num_items(&self) -> usize;
    fn tree_num_nodes(&self) -> usize;
    fn tree_node_size(&self) -> usize;
    fn tree_level_count(&self) -> usize;
    fn tree_level_bound(&self, level: usize) -> usize;
    fn tree_bounds(&self, pos: usize) -> Self::Bounds;
    fn tree_index(&self, pos: usize) -> usize;
    fn bounds_overlap(a: Self::Bounds, b: Self::Bounds) -> bool;
    fn bounds_contain(outer: Self::Bounds, inner: Self::Bounds) -> bool;
}

/// Leaf-array `[start, end)` range covered by the subtree of the entry at
/// `pos` (a node entry at `level`).
#[inline]
pub(crate) fn leaf_range<T: TreeAccess>(tree: &T, pos: usize, level: usize) -> (usize, usize) {
    let start = leaf_start(tree, pos, level);
    let end = if pos + 1 < tree.tree_level_bound(level) {
        leaf_start(tree, pos + 1, level)
    } else {
        tree.tree_num_items()
    };
    (start, end)
}

/// Leaf-array `[start, end)` range covered by a packed node group
/// `[node_start, node_end)` at `level`.
#[inline]
pub(crate) fn leaf_group_range<T: TreeAccess>(
    tree: &T,
    node_start: usize,
    node_end: usize,
    level: usize,
) -> (usize, usize) {
    let start = leaf_start(tree, node_start, level);
    let end = if node_end < tree.tree_level_bound(level) {
        leaf_start(tree, node_end, level)
    } else {
        tree.tree_num_items()
    };
    (start, end)
}

#[inline]
fn leaf_start<T: TreeAccess>(tree: &T, mut pos: usize, level: usize) -> usize {
    for _ in 0..level {
        pos = tree.tree_index(pos);
    }
    pos
}
