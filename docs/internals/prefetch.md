# Hiding tree-walk cache misses: prefetch the next node

How range search and all-hits raycast shave time for free by telling the CPU
which tree node they will touch next, so its box is already in cache when they
get there. No change to results — a pure latency hint.

## The problem: every node is a cold miss

A range or raycast query walks the packed tree from the root, descending into the
children whose boxes the query touches. Two facts make that walk memory-bound:

- The boxes live in one contiguous array, but the tree is stored **level-order,
  leaves-first** — a node and the child it sends you to are in *different* level
  blocks, far apart in memory. Consecutive steps of the walk jump around.
- The jumps are **data-dependent**: which child you visit next depends on the
  query-vs-box test you just ran. A hardware prefetcher keys off regular strides,
  so it cannot guess the next address — every node you open is a **cold
  last-level-cache miss**, roughly one per node visited.

So the walk spends much of its time stalled on the load of the next node's box,
not on the cheap overlap test itself. (This is exactly the cold-miss problem
characterised for vectorised R-trees in Rayhan & Sahu, *SIMD-ified R-tree Query
Processing*, 2023, which motivates a software-prefetch remedy.)

## The fix: one node of look-ahead

The traversal keeps an explicit stack (range) or works through `collect_hits`'s
stack (raycast). The node it will process **next** is already known — it sits one
entry below the top of the stack. So right after popping the current node, issue
a software prefetch for that next node's box:

```text
pop current node off the stack
prefetch the box of stack[next]      // its load starts now, in the background
hit-test the current node's children // ~the compute that hides the load
loop: pop the (now-cached) next node
```

The prefetch and the current node's hit-testing run concurrently: by the time the
loop comes back around and pops the next node, its box has had a full node's worth
of compute to arrive. The memory latency is overlapped with work that had to
happen anyway, so it costs nothing and is removed from the critical path.

It is a `_mm_prefetch`-style hint (`prefetch_read` in `src/traversal.rs`), wired
through a small per-layout helper — `prefetch_aos_node` / `prefetch_aos_node3d`
for the owned indexes (one box and its index slot), `SimdIndex2D::prefetch_node`
for the SoA columns. The raycast kernel takes it as a `prefetch_at` closure so the
owned indexes pass the real hint and the byte views pass a no-op.

## What it buys, and when

The win tracks **how much of the tree the query walks**:

- **Range search** (`search_into_stack`): ~3–5% across window sizes, 2D and 3D,
  at 1M boxes.
- **All-hits raycast** (`collect_hits`): ~5–12% when a ray crosses a lot of the
  scene (2D uniform +11.6%, 3D uniform +4.7% at 1M boxes), down to neutral when
  little is visited (a ray that hits a few clustered boxes opens few nodes, so
  there are few misses to hide). It never regressed.

Both were confirmed with a **same-binary A/B** — a prefetching and a
non-prefetching entry compiled into one binary, alternated on a quiet pinned core
— so the small deltas are real and not inter-binary layout noise. (See
[performance.md](../performance.md) for the measurement method.)

## Edges and what is not (yet) covered

- It is a hint: when there is no compute to hide behind, or no cold miss to hide
  (a query that touches a handful of cached nodes), it does nothing — at worst a
  negligible issued-but-unneeded prefetch. The win is concentrated on the
  heavy-traversal queries, which are the ones that cost in the first place.
- **Best-first traversals are a different shape.** kNN ([knn.md](knn.md)) and
  closest-hit raycast pop from a distance-ordered *heap*, not a stack, so "the
  next node" is the heap minimum rather than `stack[next]`. Prefetching there
  means peeking the heap root. That is a separate change.
- **Byte views** read nodes out of a borrowed byte buffer. They currently pass the
  no-op, so the hint is owned-index only for now.
- Only **one** node of look-ahead is issued. Going deeper (`pf_distance > 1`, as
  the paper parameterises) is an open knob — more look-ahead can hide more latency
  but risks evicting useful lines, so it is a measure-then-keep question.

This sits alongside the SIMD box-test work ([simd.md](simd.md)). SIMD makes each
node's test cheaper. Prefetch makes the *next* node's data arrive sooner. The two
are orthogonal, and both are free at the call site.
