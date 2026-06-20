# Nearest-neighbor traversal: two-queue distance browsing

How `neighbors` finds the *k* nearest boxes to a query point (or box) over the
packed Hilbert R-tree — and why it uses two priority queues, not one. The
technique is Hjaltason & Samet's *distance browsing* (1999); this is how it maps
onto a static packed tree, what the alternatives cost, plus the measured win.

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

(`src/neighbors/best_first.rs::collect_neighbors_two_queue` is the kernel;
`visit_neighbors_two_queue` is the same browse calling a visitor instead of
collecting, so `neighbors` and `visit_neighbors` emit in the *same* order.)

Why it is faster than the single queue: a leaf's items are materialized **only
when that leaf is the closest pending node**; node expansion stops the moment
a held item beats the nearest node. So you push fewer items, the two heaps
each stay smaller than the combined one — and `log(a) + log(b)` beats
`log(a + b)`. It is the same `O(visited · log)` work in the worst case, just with
a smaller constant on the realistic case where the answer is found before the
tree is fully opened.

Measured, switching the collect from one queue to two was ~5% faster for scalar
f64 `neighbors` and ~6–11% for the f32 / SIMD frontends (k=10, 200k boxes, on a
quiet pinned core); it never regressed, so it is the one collect kernel for every
index type. See [performance.md](performance.md) for how to reproduce.

## The k = 1 fast path

For a single nearest neighbor there is nothing to browse: keep one node queue and
a single `best` slot, descend the closest node, shrink `best` whenever a
leaf box is closer. A box distance of `0` (the query is inside the box) is the
global minimum, so it returns immediately. That is
`best_first::nearest_one`, which `neighbors(p, 1)` calls directly.

## Correctness, ties, cutoffs

- **Exact.** The node lower bound is admissible (a node's box distance never
  exceeds the distance to any item inside it, because the box encloses the
  subtree), so the browse never emits an item before a closer one. The result is
  the true `k` nearest by box distance.
- **Order on ties.** Items at equal distance may come out in any order — the API
  does not promise a tie-break. The two-queue collect and the two-queue visit use
  the *same* browse, so they agree with each other (a contract the tests check);
  do not assume a particular order across distinct query methods otherwise.
- **`max_distance`.** A finite cutoff drops nodes and items beyond it as they are
  reached, so a bounded query touches only the local neighborhood.

## What layers on top

The kernel is generic over a `dist(pos)` closure that reads the caller's own box
storage, so every frontend reuses it:

- **Custom metrics.** [`neighbors_metric`](../README.md) passes a closure that
  returns any admissible lower-bound distance — Euclidean, Manhattan, weighted, or
  great-circle via `haversine_distance_2d`. The same browse drives geographic kNN;
  the only rule is that the closure must never over-estimate (see
  [guide.md](guide.md)).
- **Compact f32.** The f32 indexes browse on the rounded f32 box distance, then
  refine the candidates against the original boxes for an exact result — the
  browse finds the candidate set, the refinement re-ranks it.
- **Box queries.** `neighbors_of_box` swaps the point-to-box distance for a
  box-to-box gap distance (`0` on overlap); the traversal is unchanged.

## A note on the scratch queue

The browse needs both queues. The reusable-workspace methods
(`neighbors_with`) hand it the workspace's two heaps; the allocation-free callers
that hold only the item heap get a small pre-sized scratch node queue per call —
measured no worse than a reused one, since a pre-sized `BinaryHeap` is a single
allocation, the same cost as the item queue already paid.
