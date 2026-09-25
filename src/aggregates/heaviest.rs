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
use crate::geometry::{Box2D, Box3D, Overlaps2D, Overlaps3D};
use crate::index2d::MASK_PAYS_IN_2D;
use crate::range::{collect_region, expects_fewer_hits};
use crate::traversal::ScratchStack;
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

/// The tree with its leaf entries reporting their own position instead of the
/// item id, so the shared collecting traversal hands out positions — the key
/// the item scalars are stored under. Node entries still point at children.
struct LeafPositions<'a, T>(&'a T);

impl<T: TreeAccess> TreeAccess for LeafPositions<'_, T> {
    type Bounds = T::Bounds;

    #[inline(always)]
    fn tree_num_items(&self) -> usize {
        self.0.tree_num_items()
    }
    #[inline(always)]
    fn tree_num_nodes(&self) -> usize {
        self.0.tree_num_nodes()
    }
    #[inline(always)]
    fn tree_node_size(&self) -> usize {
        self.0.tree_node_size()
    }
    #[inline(always)]
    fn tree_level_count(&self) -> usize {
        self.0.tree_level_count()
    }
    #[inline(always)]
    fn tree_level_bound(&self, level: usize) -> usize {
        self.0.tree_level_bound(level)
    }
    #[inline(always)]
    fn tree_bounds(&self, pos: usize) -> Self::Bounds {
        self.0.tree_bounds(pos)
    }
    #[inline(always)]
    fn tree_index(&self, pos: usize) -> usize {
        if pos < self.0.tree_num_items() {
            pos
        } else {
            self.0.tree_index(pos)
        }
    }
    #[inline(always)]
    fn tree_mask(&self, start: usize, end: usize, overlaps: &impl Fn(Self::Bounds) -> bool) -> u64 {
        self.0.tree_mask(start, end, overlaps)
    }
}

/// The items of [`heaviest_each`] in its order, gathered by a plain region
/// traversal and ranked afterwards: every hit with a scalar collected, the
/// heaviest `k` split off by `select_nth_unstable` and only those sorted. A
/// `k` of `usize::MAX` sorts every hit.
///
/// Same items, same order as the best-first descent — the order of
/// [`Heaviest`] is total, so the ranking has one answer — but the cost follows
/// the hits in the region instead of `k` and the tree height: cheaper while
/// the region holds few items, hopeless once it holds many.
///
/// A node the region contains hands over its leaves untested. The descent
/// tests each of them instead, which answers the same: the builders reject an
/// item box with `min > max`, the one box a containing region can miss.
pub(crate) fn heaviest_collect<const MASKED: bool, T, S>(
    tree: &T,
    agg: &S,
    overlaps: impl Fn(T::Bounds) -> bool,
    contains: impl Fn(T::Bounds) -> bool,
    k: usize,
) -> Vec<Ranked>
where
    T: TreeAccess,
    S: AggregateSource,
{
    let mut hits: Vec<Ranked> = Vec::with_capacity(COLLECT_CAPACITY);
    if k == 0 {
        return hits;
    }
    let mut stack = ScratchStack::take();
    collect_region::<MASKED, _, _, _, _>(
        &LeafPositions(tree),
        &mut stack,
        overlaps,
        contains,
        |pos| {
            let weight = agg.item_scalar(pos);
            if !weight.is_nan() {
                hits.push(Ranked {
                    key: descending_key(weight),
                    id: tree.tree_index(pos),
                    weight,
                });
            }
        },
    );
    let rank = |a: &Ranked, b: &Ranked| (a.key, a.id).cmp(&(b.key, b.id));
    if hits.len() > k {
        hits.select_nth_unstable_by(k, rank);
        hits.truncate(k);
    }
    hits.sort_unstable_by(rank);
    hits
}

/// The hits [`heaviest_collect`] makes room for up front.
const COLLECT_CAPACITY: usize = 64;

/// A hit of [`heaviest_collect`]: ascending `(key, id)` is heaviest first,
/// equal weights by ascending id — the order of [`Heaviest`] — on integers.
pub(crate) struct Ranked {
    key: u64,
    id: usize,
    weight: f64,
}

/// An integer that sorts non-NaN weights heaviest first; `-0.0` and `0.0`
/// share one key, as they tie under `f64` comparison.
#[inline(always)]
fn descending_key(weight: f64) -> u64 {
    let bits = (weight + 0.0).to_bits();
    // Ascending in the weight: flip every bit of a negative, the sign of the rest.
    let ascending = if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    };
    !ascending
}

/// From how many expected hits `search_heaviest` collects the region and
/// selects instead of descending best-first; see [`prefers_collect`].
pub(crate) const COLLECT_FROM_HITS: f64 = 2.0;
/// The expected hits, for `k = 1`, below which collecting still wins; see
/// [`prefers_collect`].
pub(crate) const COLLECT_BELOW_HITS_K1: f64 = 10.0;
/// How much further the collecting form wins for every item asked for beyond
/// the first; see [`prefers_collect`].
pub(crate) const COLLECT_HITS_PER_K: f64 = 50.0;

/// Whether a `search_heaviest` for `k` items expects from
/// [`COLLECT_FROM_HITS`] up to `10 + 50 * (k - 1)` hits in the region, by its
/// bounding box, as if items spread uniformly under the root: the band where
/// collecting the region and selecting beats the best-first descent. A region
/// with no bounding box never collects, nor does `k = 0`.
///
/// Both forms answer the same, so this only picks the cheaper one. The descent
/// costs about the same whatever the window holds once it holds well over `k`
/// items, and grows with `k`; the collection costs the same per hit, whatever
/// `k`. Measured on 1M boxes (`benches/paired_heaviest.rs`, query sets the
/// predictor cannot learn) on Zen 3 (EPYC 7763), Zen 4 (EPYC 9V74), Neoverse
/// N2 and a Xeon (Emerald Rapids), the lowest crossing over the four machines
/// was about 10 hits at `k = 1`, 400 at `k = 10` (Zen 3 and 4: 600) and 5000
/// at `k = 100` (Zen 4: 9500); at `k = 1000` the collection still won at
/// 30 000 hits everywhere, so the line stops short of its crossing there and
/// leaves the rest to the descent, as before the switch. Below about two hits
/// no form won everywhere: the collection by 5% on the Zens, the descent by
/// 6% on N2 and 15% on the Xeon. The descent keeps that range.
#[inline(always)]
fn prefers_collect<const D: usize>(
    root: ([f64; D], [f64; D]),
    region: Option<([f64; D], [f64; D])>,
    num_items: usize,
    k: usize,
) -> bool {
    let Some((min, max)) = region else {
        return false;
    };
    let below = COLLECT_BELOW_HITS_K1 + COLLECT_HITS_PER_K * (k as f64 - 1.0);
    k > 0
        && !expects_fewer_hits(root.0, root.1, min, max, num_items, COLLECT_FROM_HITS)
        && expects_fewer_hits(root.0, root.1, min, max, num_items, below)
}

#[inline(always)]
fn corners_2d(b: Box2D) -> ([f64; 2], [f64; 2]) {
    ([b.min_x, b.min_y], [b.max_x, b.max_y])
}

#[inline(always)]
fn corners_3d(b: Box3D) -> ([f64; 3], [f64; 3]) {
    ([b.min_x, b.min_y, b.min_z], [b.max_x, b.max_y, b.max_z])
}

/// The `search_heaviest` pair on one frontend; `$doc` is the prose above the
/// signature of the collecting form.
macro_rules! search_heaviest {
    ($ty:ty, $overlaps:ident, $corners:ident, $masked:expr, $doc:literal) => {
        impl $ty {
            #[doc = $doc]
            pub fn search_heaviest<Q: $overlaps>(&self, region: Q, k: usize) -> Option<Vec<usize>> {
                let collect = self.tree_num_items() > 0
                    && prefers_collect(
                        $corners(self.tree_bounds(self.tree_num_nodes() - 1)),
                        region.bounding_box_hint().map($corners),
                        self.tree_num_items(),
                        k,
                    );
                if collect {
                    self.search_heaviest_forced::<true, Q>(region, k)
                } else {
                    self.search_heaviest_forced::<false, Q>(region, k)
                }
            }

            /// `search_heaviest` in the form `COLLECT` names: the region
            /// collected and the heaviest `k` selected, or the best-first
            /// descent. Same items, same order; for the equality tests and
            /// for timing both in one binary.
            #[doc(hidden)]
            pub fn search_heaviest_forced<const COLLECT: bool, Q: $overlaps>(
                &self,
                region: Q,
                k: usize,
            ) -> Option<Vec<usize>> {
                let agg = self.aggregates().filter(|a| a.has_scalar())?;
                if COLLECT {
                    let contains = |b| region.contains_box(b);
                    let hits = heaviest_collect::<{ $masked }, _, _>(
                        self,
                        agg,
                        |b| region.overlaps_box(b),
                        contains,
                        k,
                    );
                    return Some(hits.iter().map(|h| h.id).collect());
                }
                let mut out = Vec::new();
                if k == 0 {
                    return Some(out);
                }
                let _ = heaviest_each(self, agg, |b| region.overlaps_box(b), &mut |id, _| {
                    out.push(id);
                    if out.len() == k {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                });
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
                visitor: F,
            ) -> Option<ControlFlow<B>>
            where
                Q: $overlaps,
                F: FnMut(usize, f64) -> ControlFlow<B>,
            {
                self.search_heaviest_each_forced::<false, Q, B, F>(region, visitor)
            }

            /// `search_heaviest_each` in the form `COLLECT` names: every hit
            /// collected and sorted before the first visit, or the best-first
            /// descent. Same items, same order; for the equality tests and for
            /// timing both in one binary.
            #[doc(hidden)]
            pub fn search_heaviest_each_forced<const COLLECT: bool, Q, B, F>(
                &self,
                region: Q,
                mut visitor: F,
            ) -> Option<ControlFlow<B>>
            where
                Q: $overlaps,
                F: FnMut(usize, f64) -> ControlFlow<B>,
            {
                let agg = self.aggregates().filter(|a| a.has_scalar())?;
                let overlaps = |b| region.overlaps_box(b);
                if COLLECT {
                    let contains = |b| region.contains_box(b);
                    let hits = heaviest_collect::<{ $masked }, _, _>(
                        self,
                        agg,
                        overlaps,
                        contains,
                        usize::MAX,
                    );
                    for h in hits {
                        if let ControlFlow::Break(b) = visitor(h.id, h.weight) {
                            return Some(ControlFlow::Break(b));
                        }
                    }
                    return Some(ControlFlow::Continue(()));
                }
                Some(heaviest_each(self, agg, overlaps, &mut visitor))
            }
        }
    };
}

search_heaviest!(
    Index2D,
    Overlaps2D,
    corners_2d,
    MASK_PAYS_IN_2D,
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
    corners_3d,
    true,
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
    corners_2d,
    MASK_PAYS_IN_2D,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` when the file carries no scalar \
     column. See [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);
search_heaviest!(
    Index3DView<'_>,
    Overlaps3D,
    corners_3d,
    true,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` when the file carries no scalar \
     column. See [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex2D,
    Overlaps2D,
    corners_2d,
    MASK_PAYS_IN_2D,
    "The `k` items of `region` with the largest scalar, heaviest first. `None` \
     without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest); the descent \
     is scalar here too (a heap pops one node at a time)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex3D,
    Overlaps3D,
    corners_3d,
    true,
    "The `k` items of `region` with the largest scalar, heaviest first. `None` \
     without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest); the descent \
     is scalar here too (a heap pops one node at a time)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex2DView<'_>,
    Overlaps2D,
    corners_2d,
    MASK_PAYS_IN_2D,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);
#[cfg(feature = "simd")]
search_heaviest!(
    SimdIndex3DView<'_>,
    Overlaps3D,
    corners_3d,
    true,
    "The `k` items of `region` with the largest scalar, heaviest first, read \
     zero-copy from the `AGGR` chunk. `None` without a scalar column. See \
     [`Index2D::search_heaviest`](crate::Index2D::search_heaviest)."
);

#[cfg(test)]
mod tests {
    use super::prefers_collect;

    /// The band the switch collects in, read off the covered share times the
    /// item count: at least two expected hits, fewer than `10 + 50 * (k - 1)`.
    #[test]
    fn collects_between_two_hits_and_the_k_line() {
        let root = ([0.0, 0.0], [1000.0, 1000.0]);
        // A window of side `s` covers (s / 1000)^2 of the root: at 1M items,
        // s^2 expected hits.
        let window = |s: f64| Some(([100.0, 100.0], [100.0 + s, 100.0 + s]));
        let collects = |s: f64, k| prefers_collect(root, window(s), 1_000_000, k);
        // One expected hit: too few, for any k.
        assert!(!collects(1.0, 10));
        // Four hits collect at k = 1, sixteen are over its line of 10.
        assert!(collects(2.0, 1));
        assert!(!collects(4.0, 1));
        // The k = 10 line is 460: 400 hits under it, 484 over.
        assert!(collects(20.0, 10));
        assert!(!collects(22.0, 10));
        // 4900 hits under the k = 100 line (4960); 90 000 hits over k = 1000.
        assert!(collects(70.0, 100));
        assert!(!collects(300.0, 1000));
        // No box, no k, a window off the root: the descent.
        assert!(!prefers_collect(root, None, 1_000_000, 10));
        assert!(!collects(20.0, 0));
        assert!(!prefers_collect(
            root,
            Some(([2000.0, 0.0], [2010.0, 10.0])),
            1_000_000,
            10
        ));
    }
}
