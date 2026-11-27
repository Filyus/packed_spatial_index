# Nearest-neighbor traversal: bounded top-k cutoff & two-queue distance browsing

How `neighbors` finds the *k* nearest boxes to a query point (or box) over the
packed Hilbert R-tree — comparing unbounded **two-queue distance browsing**
(Hjaltason & Samet, 1999) used by `neighbors_each`, the **bounded `top-k`
branch-and-bound** used by fixed-`k` `neighbors`, and the **k = 1** fast path.

The running example: `index.neighbors(point, k)` returns the `k` item boxes whose
distance to `point` is smallest, nearest first.

## The shape of the problem

Each tree node stores a bounding box that encloses its whole subtree, so the
distance from the query to a node's box is a **lower bound** on the distance to
any item beneath it. That single fact is what makes kNN cheap: if a node's box is
already farther than the `k`-th result you have, nothing inside it can matter, so
you never open it. The job is to visit nodes in just the right order to emit the
nearest items early and prune the rest.

A plain DFS with a running `k`-th-best cutoff works but wastes time descending
subtrees that a better ordering would skip. **Best-first** traversal fixes the
order: always expand the *closest* unexplored node next. That needs a priority
queue keyed by distance.

Nothing in that paragraph is about *distance*. Any per-node score that is a lower
bound for its subtree gives the same guarantee, which is what
`neighbors_metric` (your own metric) and `search_ordered` (a region query
emitted in key order — depth along a view axis, say) both trade on. Those two use
the **single**-queue kernel described next; only the default squared-Euclidean
`neighbors` uses the two-queue refinement this page argues for.

## One queue is correct but not ideal

The obvious best-first keeps a single distance-ordered queue holding both pending
nodes and candidate items. Pop the closest entry: if it is an item, emit it; if
it is a node, open it and push its children. It returns the exact `k` nearest in
order.

The cost is that **every candidate item from every opened leaf lands in the same
heap**, mixed with the pending interior nodes. The heap grows to the size of all
the items you have touched plus all the frontier nodes; every push/pop pays
`log` of that combined size. For a dense query (many items near the point) the
item candidates dominate and the heap churns.

## Two queues: browse items and nodes separately

Distance browsing splits the frontier in two:

- a **node queue** ordered by each node box's distance (a lower bound on its
  subtree);
- an **item queue** ordered by each candidate item's exact box distance.

```text
node_queue: nodes not yet opened, by lower-bound distance
item_queue: items already materialized, by exact distance
```

The loop interleaves them with one invariant — *an item is the global next
nearest exactly when it is closer than the closest unopened node*:

```text
push the root node
loop:
    while the closest node is nearer than the closest pending item:
        pop that node
        if it is a leaf:     push its items   onto item_queue
        else (internal):     push its children onto node_queue
    pop the closest item  ->  it is the next nearest; emit it
    stop once k items are emitted
```

(`src/neighbors/best_first.rs::visit_neighbors_two_queue` is the unbounded
distance-browsing kernel behind `neighbors_each` and `neighbors_of_box_each`,
while `collect_neighbors_two_queue` — plus the 4-wide `f64x4` SIMD paths on
`SimdIndex2D`/`SimdIndex3D` and `SimdIndex2DF32`/`SimdIndex3DF32` — uses the
bounded `top-k` branch-and-bound refinement described below.)

## Fixed-k branch-and-bound: tightening the cutoff dynamically

When `k` (`max_results`) is known upfront (`neighbors`, `neighbors_into`,
`neighbors_with`, and their `*_of_box` twins), keeping every touched candidate
item within the initial `max_distance` in an unbounded min-heap wastes pushes and
never tightens the pruning radius before items are popped.

Instead, fixed-`k` collection keeps:

- `node_queue`: a min-heap of unopened internal nodes ordered by lower-bound box
  distance (popping when the closest node already exceeds `cutoff`);
- `top_k`: a bounded **max-heap** of capacity `k` (`ExactNeighborState`, storing
  up to `k` best `(dist, index)` candidates seen so far).

Once `top_k.len() == k`, its root holds the `k`-th worst distance, which
dynamically replaces `cutoff` (and refreshes the `f64x4` SIMD `cutoff` vector on
SoA/`f32` frontends). Every subsequent child node and leaf box farther than
`cutoff` is rejected *before* a heap push; closer items overwrite the root in
place via `peek_mut()`. At the end, popping `top_k` in reverse fills `results` in
ascending `(d², index)` order, matching `neighbors_each().take(k)` identically.

Measured on `surface_cloud3d_bench` (`n = 1,000,000`, `k = 24..40`, `BENCH_PIN_CORE=0`):
replacing the unbounded item min-heap with the bounded `top-k` cutoff sped up
`SimdIndex3D::neighbors_with` by 15–21% against the untouched
`neighbors_each_with` control (`0.87–0.94×` of `Index3D`), and combined with
4-wide `f64x4` widened box distances made `SimdIndex3DF32::neighbors_with`
`0.76–0.82×` of `Index3D` (`7.51–10.38 ms` vs `9.80–12.60 ms`). See
[performance.md](../performance.md) for full tables.

## The k = 1 fast path

For a single nearest neighbor there is nothing to browse: keep one node queue and
a single `best` slot, descend the closest node, and shrink `best` whenever a
leaf box is closer — breaking distance ties in favor of the smaller item index.
Because of that tie-break, finding a box at distance `0` does **not** return
immediately: another subtree at distance `0` may hold an item with a smaller
index, so the loop finishes descending every node whose distance ties
`best_dist`. That is `best_first::nearest_one`, which `neighbors(p, 1)` calls
directly so `neighbors(q, 1)` is always the prefix of `neighbors(q, k)`.

## Correctness, ties, cutoffs

- **Exact.** The node lower bound is admissible (a node's box distance never
  exceeds the distance to any item inside it, because the box encloses the
  subtree), so the browse never emits an item before a closer one. The result is
  the true `k` nearest by box distance.
- **Order on ties.** Items at equal distance are emitted in ascending item-index
  order `(d², index)` for any `k` (`NeighborState` and `ExactNeighborState`
  order equal distances by smallest index). The bounded `top-k` collect,
  the two-queue `neighbors_each`, and the `k = 1` `nearest_one` fast path all
  enforce the exact same order, so `neighbors(q, 1)` agrees with the first
  element of `neighbors(q, k)`.
- **`max_distance`.** The caller's `max_distance²` seeds the initial `cutoff`
  (and for fixed `k`, tightens further once `k` candidates are held), dropping
  nodes and items beyond it before heap insertion.

## What layers on top

The kernel pattern adapts across all index frontends:

- **SIMD f64 and widened f32.** `SimdIndex2D`/`SimdIndex3D` evaluate four SoA
  child or leaf boxes per step using `wide::f64x4` and extract hits via
  bitmask. `SimdIndex2DF32`/`SimdIndex3DF32` load four `f32` column entries per
  step and widen each lane to `f64` inside `f64x4` — arithmetic is bit-identical
  to scalar `PointKnn` widening at half the coordinate memory bandwidth.
- **Custom metrics.** [`neighbors_metric`](../../README.md) passes a closure that
  returns any admissible lower-bound distance — Euclidean, Manhattan, weighted, or
  great-circle via `haversine_distance_2d`. The same best-first idea drives
  geographic kNN; the only rule is that the closure must never over-estimate (see
  [guide.md](../guide.md)).
- **Compact f32 exact.** The `f32` `neighbors_exact*` methods browse on the
  rounded `f32` lower-bound distance, then refine candidates against the
  caller's original `f64` boxes for an exact result — the browse finds the
  candidate set, the refinement re-ranks it.
- **Box queries.** `neighbors_of_box` swaps the point-to-box distance for a
  box-to-box gap distance (`0` on overlap); the traversal is unchanged.

## Workspace reuse

`NeighborWorkspace` holds reusable buffers for allocation-free calls across all
owned (`Index*`, `SimdIndex*`, `Index*F32`, `SimdIndex*F32`) and zero-copy view
frontends: `node_queue` (reused by fixed-`k`, `k = 1`, and `neighbors_each_with`),
`exact_queue` (reused as the bounded `top-k` heap for fixed-`k` queries and as
the exact refinement heap), and `queue` (reused as the unbounded item min-heap in
`neighbors_each_with`).
