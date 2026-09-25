//! Top-k by the scalar column: the items of a region, heaviest first.
//!
//! The per-node max the `AGGR` chunk already stores bounds every item below the
//! node from above, so it is an admissible best-first key as it stands — no
//! format change and no caller closure. `search_ordered` cannot express this:
//! its key sees a node's box, not its position, so never its summary.

use std::collections::BinaryHeap;
use std::ops::ControlFlow;

use super::AggregateSource;
use crate::config::DEFAULT_NEIGHBOR_QUEUE_CAPACITY;
use crate::geometry::{Overlaps2D, Overlaps3D};
use crate::tree_access::TreeAccess;
use crate::{Index2D, Index2DView, Index3D, Index3DView};
#[cfg(feature = "simd")]
use crate::{SimdIndex2D, SimdIndex2DView, SimdIndex3D, SimdIndex3DView};

/// A pending entry of [`heaviest_each`]: a node position ranked by the max its
/// subtree stores, or an item ranked by its own scalar.
///
/// Heavier pops first; equal weights pop in ascending `id`. Weights are never
/// NaN here (NaN items are skipped, a NaN node max is read as `+inf`), so the
/// partial order is total and `-0.0` ties `0.0`, as `f64::max` folded them.
#[derive(Clone, Copy)]
struct Heaviest {
    weight: f64,
    /// An item id, or a node position.
    id: usize,
    /// The node's level (unused for items).
    level: usize,
}

impl PartialEq for Heaviest {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Heaviest {}

impl Ord for Heaviest {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight
            .partial_cmp(&other.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Heaviest {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Visit the items whose box passes `overlaps` in nonincreasing order of their
/// scalar, ties in ascending item id; `visitor` may break early.
///
/// Two heaps, as in the kNN distance browsing: nodes open while the heaviest
/// pending node is at least as heavy as the heaviest pending item, so an item
/// is emitted only once nothing unopened can beat or tie it. That is what makes
/// the tie order deterministic and the first `k` a prefix of the first `k + 1`.
/// Items with a NaN scalar are never emitted.
///
/// The caller checks the scalar column is present.
pub(crate) fn heaviest_each<T, S, B>(
    tree: &T,
    agg: &S,
    overlaps: impl Fn(T::Bounds) -> bool,
    visitor: &mut impl FnMut(usize, f64) -> ControlFlow<B>,
) -> ControlFlow<B>
where
    T: TreeAccess,
    S: AggregateSource,
{
    if tree.tree_num_items() == 0 {
        return ControlFlow::Continue(());
    }
    let mut nodes = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    let mut items = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
    let node_size = tree.tree_node_size();
    // Push the entries of the group starting at `start` on `level` that pass
    // the region test.
    let open = |start: usize,
                level: usize,
                nodes: &mut BinaryHeap<Heaviest>,
                items: &mut BinaryHeap<Heaviest>| {
        let end = (start + node_size).min(tree.tree_level_bound(level));
        for pos in start..end {
            if !overlaps(tree.tree_bounds(pos)) {
                continue;
            }
            if level == 0 {
                let weight = agg.item_scalar(pos);
                if !weight.is_nan() {
                    let id = tree.tree_index(pos);
                    items.push(Heaviest { weight, id, level });
                }
            } else {
                let max = agg.node_scalar(pos).2;
                let weight = if max.is_nan() { f64::INFINITY } else { max };
                nodes.push(Heaviest {
                    weight,
                    id: pos,
                    level,
                });
            }
        }
    };
    let root = tree.tree_num_nodes() - 1;
    open(root, tree.tree_level_count() - 1, &mut nodes, &mut items);
    loop {
        while let Some(&node) = nodes.peek() {
            if items.peek().is_some_and(|item| item.weight > node.weight) {
                break;
            }
            nodes.pop();
            open(
                tree.tree_index(node.id),
                node.level - 1,
                &mut nodes,
                &mut items,
            );
        }
        match items.pop() {
            Some(item) => visitor(item.id, item.weight)?,
            None => return ControlFlow::Continue(()),
        }
    }
}

/// The `search_heaviest` pair on one frontend; `$doc` is the prose above the
/// signature of the collecting form.
macro_rules! search_heaviest {
    ($ty:ty, $overlaps:ident, $doc:literal) => {
        impl $ty {
            #[doc = $doc]
            pub fn search_heaviest<Q: $overlaps>(&self, region: Q, k: usize) -> Option<Vec<usize>> {
                let mut out = Vec::new();
                if k == 0 {
                    return self.aggregates().filter(|a| a.has_scalar()).map(|_| out);
                }
                let _ = self.search_heaviest_each(region, |id, _| {
                    out.push(id);
                    if out.len() == k {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })?;
                Some(out)
            }

            /// Visit the items of `region` heaviest first — nonincreasing
            /// scalar, equal scalars in ascending item index — with the scalar
            /// alongside the id; return [`ControlFlow::Break`] to stop, for
            /// example once the scalar drops below a threshold. `None` when
            /// the index carries no scalar column. See `search_heaviest`.
            pub fn search_heaviest_each<Q, B, F>(
                &self,
                region: Q,
                mut visitor: F,
            ) -> Option<ControlFlow<B>>
            where
                Q: $overlaps,
                F: FnMut(usize, f64) -> ControlFlow<B>,
            {
                let agg = self.aggregates().filter(|a| a.has_scalar())?;
                Some(heaviest_each(
                    self,
                    agg,
                    |b| region.overlaps_box(b),
                    &mut visitor,
                ))
            }
        }
    };
}

search_heaviest!(
    Index2D,
    Overlaps2D,
    r#"The `k` items of `region` with the largest
[`aggregate_scalar`](crate::Index2DBuilder::aggregate_scalar) value, heaviest
first — "the ten largest objects in view".

A best-first descent over the per-node max the aggregates already store: a
subtree whose max cannot beat the `k`-th item found is never opened, so the cost
follows `k` and the tree height, not the number of items in `region`. Equal
scalars come out in ascending item index, so `search_heaviest(region, k)` is a
prefix of `search_heaviest(region, k + 1)`. Items whose scalar is NaN are never
returned.

Returns `None` when the index carries no scalar column. The region test is
broad phase, like every query here: an overlapping box does not mean the item's
exact geometry overlaps.

# Example

```
use packed_spatial_index::{Box2D, Index2DBuilder};

let mut b = Index2DBuilder::new(4);
b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
b.add(Box2D::new(2.0, 0.0, 3.0, 1.0));
b.add(Box2D::new(4.0, 0.0, 5.0, 1.0));
b.add(Box2D::new(50.0, 0.0, 51.0, 1.0)); // heaviest, but out of view
let index = b.aggregate_scalar(&[3.0, 7.0, 3.0, 99.0]).finish().unwrap();

let view = Box2D::new(0.0, 0.0, 10.0, 10.0);
assert_eq!(index.search_heaviest(view, 2), Some(vec![1, 0]));
```"#
);
search_heaviest!(
    Index3D,
    Overlaps3D,
    r#"The `k` items of `region` with the largest
[`aggregate_scalar`](crate::Index3DBuilder::aggregate_scalar) value, heaviest
first; equal scalars in ascending item index. `None` without a scalar column.
See [`Index2D::search_heaviest`](crate::Index2D::search_heaviest).

# Example

```
use packed_spatial_index::{Box3D, Index3DBuilder};

let mut b = Index3DBuilder::new(3);
b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
b.add(Box3D::new(2.0, 0.0, 0.0, 3.0, 1.0, 1.0));
b.add(Box3D::new(4.0, 0.0, 0.0, 5.0, 1.0, 1.0));
let index = b.aggregate_scalar(&[5.0, 1.0, 5.0]).finish().unwrap();

let all = Box3D::new(-1.0, -1.0, -1.0, 9.0, 9.0, 9.0);
assert_eq!(index.search_heaviest(all, 2), Some(vec![0, 2]));
```"#
);
search_heaviest!(
    Index2DView<'_>,
    Overlaps2D,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` when the file carries no scalar \
     column. See [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);
search_heaviest!(
    Index3DView<'_>,
    Overlaps3D,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` when the file carries no scalar \
     column. See [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex2D,
    Overlaps2D,
    "The `k` items of `region` with the largest scalar, heaviest first. `None` \
     without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest); the descent \
     is scalar here too (a heap pops one node at a time)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex3D,
    Overlaps3D,
    "The `k` items of `region` with the largest scalar, heaviest first. `None` \
     without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest); the descent \
     is scalar here too (a heap pops one node at a time)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex2DView<'_>,
    Overlaps2D,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex3DView<'_>,
    Overlaps3D,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);
