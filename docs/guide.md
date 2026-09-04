# Guide

Practical recipes and configuration. For the per-method API reference, see
[docs.rs](https://docs.rs/packed_spatial_index).

## Choosing a query method

Start from the answer you want, not from the query shape. The table below names
the method for each need; the notes after it explain the reasoning.

| I need | Use | Instead of |
| --- | --- | --- |
| A `bool` — "is anything in this box?" | `any(query)` — stops at the first hit, allocates nothing | `!search(query).is_empty()`, which collects every hit first |
| Any one hit | `first(query) -> Option<usize>` | `search(query).first().copied()` |
| Just how many hits there are | `count(query)` — counts during the traversal, allocates nothing | `search(query).len()`, which builds a `Vec` to throw away |
| Every hit, one call, simplest code | `search(query) -> Vec<usize>` | — |
| Every hit in a hot loop, no reallocation | `search_into(query, &mut vec)` (your `Vec`, cleared per call) or `search_with(query, &mut workspace)` (a reusable `SearchWorkspace`, returns a `&[usize]`) | `search`, which allocates a fresh `Vec` per query |
| To stop part-way through the hits | `search_iter(query)` on the owned `f64` indexes — a lazy iterator, so `.take(k)` / `.find(..)` end the traversal — or `visit` returning `ControlFlow::Break` | collecting everything and then breaking |
| To fold or aggregate hits (sum, min, push elsewhere) | `visit(query, ..)` | `search` followed by a loop |
| The *k* nearest to a point | `neighbors` / `neighbors_within` / `neighbors_into` / `neighbors_with` / `visit_neighbors` — same alloc-vs-buffer choice as above | sorting search results by distance |
| The *k* nearest under my own distance (lon/lat, weighted, …) | `neighbors_metric(..)` with a `\|box\| -> f64` lower bound — `haversine_distance_2d` ships for geographic data | — |
| Every hit near-to-far, or just the nearest *N* in a frustum | `search_ordered(region, key, max_results, max_key)` / `visit_ordered` with `view_depth_3d` as the key — the traversal ends at the budget | `search(region)` and then sorting the hits |
| The *k* nearest to a **box**, not a point | `neighbors_of_box` and its `_within` / `_into` / `_with` / `visit_` forms | — |
| Hits along a ray, or the closest one | `raycast` / `raycast_into` / `raycast_with` / `visit_raycast`, and `raycast_closest` when only the nearest matters | — |
| All overlapping pairs between two indexes | `join` / `join_with` (`self_join` within one index) | a query per item |
| Everything within a distance of one place | `search_within(query, max_distance)` / `search_within_into` / `visit_within` / `any_within` | `search` on an `max_distance`-inflated box and then filtering the hits |
| The single closest pair, with no distance to guess | `closest_pair(&other)` / `self_closest_pair()` | `join_within` with a guessed `max_distance`, widened until it is non-empty |
| All pairs within a distance — "within 500 m", not "intersecting" | `join_within` / `join_within_with` (`self_join_within` within one index, `anti_join_within` for the unpaired items, `self_join_within_components` for groups) | joining indexes of `max_distance`-inflated boxes and filtering |
| To query bytes I already have, with no build step | `Index2DView::from_bytes` / `Index3DView` — the same query surface, zero-copy | loading into an owned index |
| To query a file I do not want to download | `StreamIndex2D` / `StreamIndex3D` over a `RangeReader` | fetching the whole index |
| The per-item blob back, not just the id | `payload(id)` / `search_payloads(query)` on a view, or `search_payloads` on a streaming reader | a side table keyed by id |
| Exact answers from an `f32` index | `search_exact` / `neighbors_exact`, passing your own `f64` boxes | trusting the conservative superset |
| To index fewer than ~100 boxes, or query fewer than ~50 times | a plain loop over your own `Box2D`s | building an index — see the crossovers below |

`any`, `first`, `count`, `search_into`, `search_with` and `visit` all exist on the `f64`
frontends — owned, SIMD and the zero-copy views alike — so a tight loop there
never has to fall back to `search`. Two exceptions: `search_iter` is on the owned
`f64` `Index2D` / `Index3D` only, and the scalar `Index2DF32` / `Index3DF32`
carry `any` / `first` / `count` / `visit` but not the buffer-reusing
`search_into` / `search_with` (the `SimdIndex*F32` frontends have those). The
streaming readers carry `count` too, as `count(query) -> Result<usize, _>`,
alongside `count_region` for the shape queries (and the `_async` twins under the
`async` feature).

Why the distinctions matter:

- **`search` allocates; the alternatives do not.** Every `search` call returns a
  fresh `Vec`, so a query in a hot loop pays an allocation per call even when the
  result is one number or a `bool`. `search_into` reuses a `Vec` you own,
  `search_with` / `neighbors_with` reuse a `SearchWorkspace` / `NeighborWorkspace`,
  and `any` / `first` / `count` / `visit` never build a result buffer at all.
- **Short-circuiting is a traversal property, not a filter.** `any`, `first`,
  `visit`-with-`Break` and `search_iter` end the descent early, so they skip the
  subtrees a collected result would have walked. Collecting first and then
  breaking out of a loop over the results saves nothing.
- **`search_iter` is genuinely lazy.** It holds an O(depth) stack and descends on
  demand, so `.next()` / `.take(k)` / `.find(..)` stop mid-traversal with no
  result `Vec`. Unlike the others it is only on the owned `f64` indexes.
- **`count` is not `search(..).len()`.** It counts inside the traversal, so no
  result buffer is built; on a streaming reader it is fallible like every other
  query there, and it charges the same read budget as `search`.
- Use `search_exact` / `neighbors_exact` on the `f32` indexes for exact results
  from compact storage; prefer the `f64` indexes for exact queries with many
  hits.
- Scan, scalar index, or SIMD index? Measured crossovers (uniform 2D boxes, one
  machine — treat as orders of magnitude, not exact):
  - **Below ~100–130 boxes**, a plain linear scan over your own `Box2D`s beats an
    index *per query* — the traversal's fixed overhead doesn't pay off yet.
  - **Building an index amortizes after ~50–120 queries** over the same box set
    (for a few hundred boxes and up); for fewer queries than that, or under ~100
    boxes, just scan. Above the crossover the index pulls away fast — at 1M boxes
    it answers a window query ~30–50× faster than a scan.
  - **`SimdIndex*` over the scalar `Index*`**: on an **AVX-512** CPU the search
    runs ~**1.6–1.9×** faster on range queries across 100k–1M boxes (it collects
    results with a masked compress-store, so the win holds even on broad,
    high-result queries). On an **AVX2** CPU (no AVX-512) it runs ~**1.3–1.6×**
    via a runtime AVX2 tier that emulates the compress with a
    [left-pack](internals/simd.md); on older CPUs it falls back to SSE2 width. At very
    small sizes it ties the scalar index. The tier is picked automatically
    (`AVX-512 → AVX2 → SSE2`); `-C target-cpu=native` (see
    [performance.md](performance.md#build-flags)) additionally widens the scalar
    autovectorization.

## Coverage matrix

Which query each index type answers. `✓` available, `✗` not, `*` a conservative
superset over outward-rounded `f32` boxes (refine with the `*_exact` family).
"Payload" is attaching (`write`) or returning (`read`) a per-item blob;
"Streaming" is answering queries over a `RangeReader` without loading the whole
file; `search_iter` is the lazy iterator form of range search.

| Index type | Range | Region shapes | Ordered region | Point kNN | Box kNN | Raycast | Join | Payload | `search_iter` | Streaming |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Index2D` / `Index3D` (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | write | ✓ | ✗ |
| `Index2DView` / `Index3DView` (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | read | ✗ | ✗ |
| `SimdIndex2D` / `SimdIndex3D` (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| SIMD views (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `Index2DF32` / `Index3DF32` (f32) | ✓* | ✓* | ✓* | ✓* | ✗ | ✓* | ✗ | write | ✗ | ✗ |
| `SimdIndex2DF32` / `SimdIndex3DF32` (f32) | ✓* | ✓* | ✓* | ✓* | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `StreamIndex2D` / `StreamIndex3D` (and `…F32`) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | read | ✗ | ✓ |

`count` has no column because it would be a column of `✓`: every row above
answers `count(query)`, including the streaming readers, where it is fallible
like their other queries (`count(query) -> Result<usize, _>`).

"Region shapes" is the 2D triangle and convex-polygon and the 3D frustum query.
On `Index2D` / `Index3D` and their views they ride the ordinary `search` / `any`
/ `first` / `count` / `visit`, which take borrowed region geometry as well as a
box. Everywhere else they are a parallel family — `search_region` /
`search_region_into` / `visit_region` / `any_region` / `first_region` /
`count_region` — so that the `Box` entry points keep their specialized kernels:
that is how the SIMD and `f32` frontends carry them, and how the streaming
readers already did (plus `search_payloads_region`, and the matching `*_async`
methods under the `async` feature). On the `f32` frontends the shape is tested
against the stored box widened back to `f64`; it was rounded outward, so the
answer is the same conservative superset those types return everywhere else.

The empty cells are intentional, not gaps to fill:

- Streaming covers range and region search (with payloads). kNN, raycast and
  `search_ordered` use a best-first traversal, and what rules it out is **round
  trips, not bytes**. The level-order descent fetches a whole level at once and
  the async path issues that level's reads concurrently, so a query costs a
  handful of *dependent* waves however many reads it makes; a heap cannot name
  the node to open next until the previous node's boxes have arrived, so each of
  its reads is its own wave. Measured on 1M boxes with a frustum holding ~160k of
  them: the region query is 313 reads in 4 waves, while a best-first descent
  budgeted to 100 items is ~132 reads in ~132 waves — a tenth of the bytes and
  thirty times the latency. Load those with a view or an in-memory index. (The
  in-memory and `f32` indexes serialize the files that `StreamIndex*` reads.) An
  ordered *result* is still available over a stream, just not through a heap —
  see [top-k over a stream](#top-k-over-a-stream). Waves are also what
  `serialize().interleaved()` buys: it puts each node's child pointer in the box
  record, so a level costs one fetch instead of two dependent ones (4 waves -> 2
  for the query above), at the same read count and file size.
- The ordered region query is a scalar descent on every frontend, SIMD included:
  a heap yields one node at a time, so there is nothing for a wide kernel to
  test in parallel. It is on the SIMD and `f32` types so the query is available
  where your index already lives, not because it is faster there.
- The `f32` indexes answer range, point-kNN, and (scalar only) raycast as a
  conservative superset; refine with the `*_exact` family against your own `f64`
  boxes. The SIMD `f32` frontend carries no payload and no raycast; the compact
  mesh-BVH story uses the scalar `Index3DF32` (AABBs from `from_triangles`,
  triangles as the payload).
- Payload read lives on the byte views and `StreamIndex*`, not the owned or SIMD
  indexes: an owned index returns ids into your own data, so attach a per-item
  blob at serialize time and read it back through a view or streamed.

## Query by a triangle (2D)

`Index2D` answers a triangle region query through the generic overlap API:
`search(&tri)` / `search_into(&tri, ...)` (collect),
`any(&tri)` (boolean, short-circuits), and
`visit(&tri, ...)` (fold without collecting). Each returns the
items whose box overlaps the triangle's filled area — the bounding-box corners
the triangle misses are rejected during the traversal.

```rust
# use packed_spatial_index::{Index2DBuilder, Box2D, Triangle2D};
# let mut b = Index2DBuilder::new(2);
# b.add(Box2D::new(0.2, 0.2, 0.3, 0.3));
# b.add(Box2D::new(9.0, 9.0, 9.5, 9.5));
# let index = b.finish()?;
let tri = Triangle2D::new([0.0, 0.0], [10.0, 0.0], [0.0, 10.0]);
assert_eq!(index.search(&tri), vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

Prefer this to `search(tri.aabb())` filtered by hand. It is both tighter and
faster: in a 200k-box field it rejects roughly 2× (fat triangle) to 7× (sliver)
of the bounding-box hits, and runs ~2.5×–5× faster than collect-then-filter —
internal nodes are pruned with a cheap box-vs-bbox test, subtrees fully inside
the triangle are accepted whole without per-item tests, and the full
triangle-AABB separating-axis test runs only at boundary leaves.
`any(&tri)` is the exact-culling analogue of `any`. The same methods
are on the zero-copy `Index2DView`, so you can run triangle queries straight
over serialized bytes.

## Query by a convex polygon (2D)

`Index2D` also answers an arbitrary **convex polygon** region query — the N-gon
generalization of the triangle query via `search(&poly)` /
`search_into(&poly, ...)` (collect), `any(&poly)` (boolean,
short-circuits), and `visit(&poly, ...)` (fold).
A four-vertex polygon is a 2D view frustum / FOV trapezoid; any convex shape
works.

```rust
# use packed_spatial_index::{Index2DBuilder, Box2D, ConvexPolygon2D};
# let mut b = Index2DBuilder::new(2);
# b.add(Box2D::new(1.0, 1.0, 2.0, 2.0));
# b.add(Box2D::new(0.0, 5.0, 0.5, 5.5));
# let index = b.finish()?;
// A trapezoid: a 2D camera frustum seen from above.
let trapezoid = ConvexPolygon2D::new(vec![
    [0.0, 0.0], [10.0, -4.0], [10.0, 8.0], [0.0, 3.0],
]);
assert_eq!(index.search(&trapezoid), vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

The test is exact (a separating-axis test over the box's two axes and the
polygon's edge normals), so the result is precisely the boxes the polygon's
filled area overlaps. Two wins over `search` on the polygon's bounding box,
measured in a 200k-box field:

- **Tighter:** ~1.5x fewer hits for a near-round polygon (hexagon/octagon), up to
  ~4.6x for a narrow trapezoid — the win tracks how much slimmer the shape is
  than its bounding box.
- **Faster anyway:** `search(&poly)` beats collecting `search(bbox)` and
  filtering by hand by **~2.2x even for the round octagon** (its weakest
  selectivity case) and up to **~13x for a wide trapezoid** — internal nodes are
  pruned with the polygon test and subtrees fully inside are accepted whole,
  instead of materializing the whole bounding-box result and filtering every box.

For triangles, use `Triangle2D` with `search(&tri)` rather than
representing the same shape as a three-vertex polygon. The same generic overlap
methods are also on the zero-copy `Index2DView`.

## Frustum culling (3D)

`Index3D` answers a view-frustum query through the generic overlap API:
`search(&frustum)` / `search_into(&frustum, ...)` (collect),
`any(&frustum)` (boolean, short-circuits), and
`visit(&frustum, ...)` (fold without collecting). Build a
[`Frustum3D`] from six
inward-pointing planes, or from a row-major view-projection matrix via
`Frustum3D::from_view_projection` (column-major engines pass the transpose).

```rust
# use packed_spatial_index::{Index3DBuilder, Box3D, Frustum3D, ClipSpaceZ};
# let mut b = Index3DBuilder::new(1);
# b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
# let index = b.finish()?;
let identity = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
let frustum = Frustum3D::from_view_projection(identity, ClipSpaceZ::NegOneToOne); // OpenGL clip cube
assert_eq!(index.search(&frustum), vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

The query is **conservative**: it returns every box inside or crossing the
frustum and may include a few that lie just past an edge or corner (the standard
p-vertex test), but never drops a visible box. That is what culling wants — an
extra box is cheap to reject downstream; a missing one is a hole in the frame.
Prefer it to `search` over the frustum's bounding box: in a 200k-box scene it
returns ~2x-4x fewer boxes and runs ~3x-14x faster (the slanted sides prune
internal nodes, and subtrees fully inside the frustum are accepted whole). It is
also *more* correct than a hand-rolled bounding-box-plus-filter, which can miss
boxes the conservative test accepts just outside the frustum's tight bbox. The
same generic overlap methods are on the zero-copy `Index3DView`.

### Picking (3D)

The frustum need not be the camera's. Narrowed to the few pixels around the
cursor it answers "what is under the click", and widened to a dragged rectangle
it answers rubber-band selection — both by tree traversal, without scanning the
scene. Build the narrowed matrix the way `gluPickMatrix` did, by scaling the
view-projection so the clicked NDC rectangle fills the clip cube, then read the
planes off it with `Frustum3D::from_view_projection`; there is a complete,
compiled example on [`Frustum3D`](https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Frustum3D.html).

What that gives you is the **candidate set**, conservatively: an object just
outside the tolerance may be included, none that belongs is dropped. What it
does not give you is the winner. Results are boxes, in unspecified order, so
picking has a second step and the second step is where the accuracy lives:

- **Nearest along the cursor ray** — `raycast_closest` returns the first box the
  ray enters and the distance to it. For a single-pixel pick with no tolerance
  this is the whole job and the frustum is unnecessary; for a click with a
  radius, the frustum narrows the set and the ray orders it.
- **Nearest to a point in space** — `neighbors` (or `neighbors_metric` for your
  own distance) over the candidates' region, when "closest to where the user
  clicked in world space" is the question rather than "first along the ray".
- **Exact geometry** — the index stores bounding boxes, so a click inside an
  object's box but beside the object itself is a hit here and a miss against the
  real mesh. Any exact test is yours to run over the candidates; for triangle
  meshes `Ray3D::closest_triangle` does it, and the payload can carry the
  triangles (see [Keep payloads outside the index](#keep-payloads-outside-the-index)).

The practical shape is: frustum to narrow, ray or exact test to decide. The
frustum's job is to turn "test every object" into "test the handful the click
could possibly touch".

## Front-to-back region queries

`search(&frustum)` hands back an unordered bag. When you want the near objects
first — a renderer filling z, an occlusion loop, a "draw the closest 500 and
stop" budget — `search_ordered` emits the same set in nondecreasing order of a
key you supply, and `view_depth_3d` is that key for a view direction:

```rust
# use packed_spatial_index::{Box3D, Index3DBuilder, view_depth_3d};
# let mut b = Index3DBuilder::new(3);
# b.add(Box3D::new(20.0, 0.0, 0.0, 21.0, 1.0, 1.0));
# b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
# b.add(Box3D::new(10.0, 0.0, 0.0, 11.0, 1.0, 1.0));
# let index = b.finish()?;
let eye = [-5.0, 0.0, 0.0];
let forward = [1.0, 0.0, 0.0];
let visible = Box3D::new(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0);

// The two nearest visible objects, near-to-far, without touching the rest.
let closest = index.search_ordered(
    visible,
    |bx| view_depth_3d(eye, forward, bx),
    2,
    f64::INFINITY,
);
assert_eq!(closest, vec![1, 2]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

The key must be an **admissible lower bound**, the same contract
[custom-metric kNN](#geographic-and-custom-metric-knn) asks for: the key of a box
never exceeds the key of any item inside it. `view_depth_3d` satisfies it by
construction (a node box encloses its children, so its minimum depth is no
larger than theirs), and so does any "smallest value over the box" score — depth,
distance, a priority you store per item and summarize per subtree. The direction
need not be normalized: a longer vector rescales the key and any `max_key`
cutoff, never the order.

**Order everything and you lose.** The ordered traversal drives a heap, while
`search` is a depth-first sweep that can emit a wholly contained subtree without
testing it. Measured on 1M boxes with a frustum holding ~160k of them
([performance](performance.md#ordered-front-to-back-region-queries)), ordering
the whole result is ~1.8x slower than `search` plus a sort, while a budget of 100
is ~185x faster, and 10 000 still ~10x. The rule of thumb: reach for
`search_ordered` when something — a budget, a `max_key` cutoff, a
`ControlFlow::Break` — lets the traversal stop; reach for `search` and `sort`
when you genuinely need every hit ordered.

`visit_ordered` gives the same sequence through a visitor that receives the key
alongside the id, so a renderer can accumulate until its budget is spent and
break. Every f64 and `f32` in-memory frontend answers it, SIMD included, though
the descent is scalar everywhere (a heap pops one node at a time). Streaming
readers do not carry the method — a heap costs one round trip per node it opens
— but they can still answer the question it is usually asked for; see below.

### Top-k over a stream

There is no `search_ordered` on `StreamIndex2D` / `StreamIndex3D`, and adding one
would be a mistake: a best-first descent turns a query that costs four dependent
round trips into one that costs a hundred. But "the nearest k in this region" does
not need a heap. Cap the region at a key threshold and the ordinary
level-by-level `search_region` answers it, because the cap prunes internal nodes
for the same reason the region does — a child box lies inside its parent, so its
key is no smaller:

```rust
use packed_spatial_index::{Box3D, Frustum3D, Overlaps3D, view_depth_3d};

struct DepthCapped {
    frustum: Frustum3D,
    eye: [f64; 3],
    direction: [f64; 3],
    max_depth: f64,
}

impl Overlaps3D for DepthCapped {
    fn overlaps_box(&self, b: Box3D) -> bool {
        view_depth_3d(self.eye, self.direction, b) <= self.max_depth
            && self.frustum.overlaps_box(b)
    }
}
```

Pass that to `search_region` and sort the (few) results by the same key. Measured
on the same 1M-box scene, `max_depth` set to the true 100th-nearest key: **75
reads / 0.26 MB in 4 waves**, against 313 reads / 12.5 MB for the uncapped region
query — cheaper on every axis than the region query that ships, and cheaper in
latency than a heap by thirty times.

The one thing this asks of you is the threshold. Start from an estimate and
iterate: run the capped query, and if fewer than *k* items came back, raise the
cap and run it again — each round is another four waves, and one or two rounds is
typical. A useful first estimate costs no I/O at all, since `open` caches the
upper levels: a node's *far*-corner depth is an upper bound on the nearest depth
of anything inside it. Widen generously rather than tightly — an over-large cap
costs bytes, an over-small one costs a whole extra round.

## Search by distance (radius query)

`search` answers "which boxes overlap this one". `search_within` answers
"which boxes are within 500 m of it" — the same distance the ε-join uses, one
tree instead of a pair, and no `k` to invent. The distance is between boxes
(`Box2D::distance_to_box`, or the `sqrt`-free `distance_squared_to_box`): zero
when the boxes overlap, edges inclusive, so `max_distance = 0.0` reproduces `search`
exactly. A degenerate query box (`min == max`) is a point. A negative or NaN
`max_distance` matches nothing. Results come back in traversal order, as from
`search`.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut b = Index2DBuilder::new(3);
# b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# b.add(Box2D::new(3.0, 0.0, 4.0, 1.0));
# b.add(Box2D::new(200.0, 200.0, 201.0, 201.0));
# let towers = b.finish()?;
let mut near = towers.search_within(Box2D::new(0.5, 0.5, 0.5, 0.5), 3.0);
near.sort_unstable();
assert_eq!(near, vec![0, 1]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

`search_within_into` fills a buffer you own, `visit_within` folds without one
(return `ControlFlow::Break` to stop early), and `any_within` answers "is there
anything near here" without collecting. All four are on the same eight types
that carry `join_within`: the owned `f64` indexes, their views, and the SIMD
indexes and views. The `f32` and streaming frontends carry no distance
operations yet.

Two node tests, deliberately different. A node is descended when its box is
within `max_distance` — items sit inside their node box, so the node distance is a
lower bound and prunes soundly. It never *accepts*, for the same reason: an
item inside a passing node box can be farther than the node box is. The
sufficient test is the node's *farthest* corner being within `max_distance`, and a
node passing that has its whole subtree emitted without per-item tests.

Measured on 1 million boxes against the workaround it replaces — `search` on
the `max_distance`-inflated query box, then an exact distance filter over the
candidates — pinned to one core, arm order alternated per round, control arm
0.99–1.07×: at ~10 hits per query `search_within` runs 1.25–1.5× faster on
uniform data and level with the workaround on clustered point queries, at ~100
hits 2.7–5.3×, and at ~7000 hits 8.5–17.5×. The win is not selectivity — for a
point query the circle-to-square area ratio is only π/4 — it is that the
pruning happens inside the traversal, where the workaround must materialize
every candidate and gather its geometry back to filter it.

A cheaper node prune was tried and rejected: overlapping against the query
grown by `max_distance` is a valid necessary condition and costs four compares where
the exact distance costs two axis gaps and two multiplies, but the extra
subtrees it descends cost 1.2–2.2× more than the predicate saves, measured
with both arms in one binary.

## Join by distance (ε-join)

Naming: every method that takes a distance bound ends in `_within`
(`search_within`, `neighbors_within`, `join_within`, ...), and the HTTP
parameter and CLI flag are `within=` / `--within`. "ε-join" is the literature
name for the same operation; the word `epsilon` appears nowhere in the API,
and the bound parameter is `max_distance` throughout.
`radius` is reserved for the spherical lon/lat query, where it is literally a
circle on the sphere.

`join` answers "which boxes intersect". `join_within` answers the question
people actually ask — "which pairs are within 500 m of each other". The
distance is between boxes (`Box2D::distance_to_box`, `Box3D::distance_to_box`,
or the `sqrt`-free `distance_squared_to_box`): zero when the boxes overlap,
edges inclusive, so `max_distance = 0.0` reproduces `join` exactly. Like every query
here it is a broad phase — the box distance is a lower bound on the true
distance between the underlying geometries, so hits are candidates and the
exact predicate stays with the caller. A negative or NaN `max_distance` matches
nothing.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut b = Index2DBuilder::new(3);
# b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# b.add(Box2D::new(3.0, 0.0, 4.0, 1.0));
# b.add(Box2D::new(200.0, 200.0, 201.0, 201.0));
# let towers = b.finish()?;
let mut pairs = towers.join_within(&towers, 5.0);
pairs.sort_unstable();
assert_eq!(pairs, vec![(0, 1)]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

The family shares the `join` descent with the prune test swapped for the
distance, on every type that carries `join` (the owned `f64` indexes, their
views, and the SIMD indexes and views):

- `join_within` / `join_within_with`, `self_join_within` /
  `self_join_within_with` — the pair stream. A leaf whose whole subtree lies
  within `max_distance` is emitted as a range without per-item tests.
- `anti_join_within` / `anti_join_within_with` — items of `self` with *no*
  partner within `max_distance`: the noise side of the graph, one pruned search per
  item. An index queried against itself pairs with itself at distance zero, so
  isolation within one index is a components question, not an anti-join.
- `self_join_within_components` — one label per item: the smallest item id in
  its component of the `max_distance`-proximity graph, an isolated item being its
  own label. The labels identify components; they are not clusters. Distance
  proximity is *not transitive* — a chain of items each within `max_distance` of the
  next is one component no matter how far its ends lie apart — so what a
  component "is" (merge, split, keep as noise) stays with the caller. This
  reports what the graph defines, deterministically.

Measured on 100 000 × 100 000 uniform 2D boxes (extent 1000, unit size):
`join_within` at `max_distance = 2` (324 000 pairs) runs ~2.5× faster than the
workaround of joining two indexes of `max_distance`-inflated boxes and filtering
the pairs by exact distance; at `max_distance = 6` (1.6 million pairs) ~3×. The
workaround also needs a second, larger index — 6–10 ms extra build and more
memory in this setup. Against plain `join` the picture splits by data shape:
on uniform data the distance predicate costs nothing extra —
`join_within(0.0)`, the same pairs with the predicate swapped, measures
0.5–1.0× of `join` — while on clustered data plain `join` stays the cheaper
tool (1.1–2.9×, worst where almost nothing matches), so pick by the question
being asked.

## The closest pair

`closest_pair` answers "which two of these are nearest each other" — one
answer, no `max_distance` to guess. `join_within` can be walked up to it, but only
by picking a bound, finding it empty, and widening; this finds it directly.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut b = Index2DBuilder::new(3);
# b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# b.add(Box2D::new(3.0, 0.0, 4.0, 1.0));
# b.add(Box2D::new(3.5, 0.0, 4.5, 1.0));
# let towers = b.finish()?;
let (i, j, distance) = towers.self_closest_pair().unwrap();
assert_eq!((i.min(j), i.max(j)), (1, 2));
assert_eq!(distance, 0.0); // items 1 and 2 overlap
# Ok::<(), packed_spatial_index::BuildError>(())
```

`closest_pair(&other)` does the same across two indexes, returning
`(item_of_self, item_of_other, distance)`. Both return `None` when there is no
pair to report — an empty index either side, or fewer than two items for the
self form. An item is never paired with itself. The distance is between boxes,
zero when they overlap, so like everything here it is a broad phase and a lower
bound on the distance between the underlying geometries. Which pair is
reported among several at the same distance is traversal order.

This is a different traversal from the joins: a best-first frontier of *node
pairs* keyed by the pair's box distance, which is a lower bound on any item
pair beneath it. The first time the frontier's head is no closer than the best
pair already found, everything still queued can be dropped unexamined — the
early exit is the whole point of having the operation at all.

That exit only helps once `best` is finite, so the descent seeds it first with
a real pair: a handful of items walked greedily down the other tree for the
cross form, and a sweep of adjacent entries in the leaf array for the self
form, where spatial sort order already puts near things near. Both stop the
moment they find an overlapping pair, which cannot be beaten. The seed is
worth far more than it costs — on a million non-overlapping points it took the
query from 1.58 s to 0.39 s.

Measured on 1 M boxes against the workaround the shipped API otherwise forces —
one `neighbors_of_box(item, 1)` per item, keeping the minimum — pinned to one
core, arm order alternated per round, ten paired rounds, control arm
0.98-1.03×:

| data | `closest_pair` | vs the kNN loop | `self_closest_pair` | vs the loop |
| --- | --- | --- | --- | --- |
| uniform points, no overlaps | 386 ms | 9.2× | 68 ms | 62× |
| clustered boxes | 12.4 ms | 294× | ~0 | — |

The two rows are the two regimes. Where no pair overlaps, the descent has to
work for its answer and wins by a single-digit factor on the cross form. Where
some pair is zero apart — clustered or dense box data, which is most real data
— the seed usually finds it outright and the query returns in microseconds
whatever the index size; the ratio there is large enough to be meaningless to
quote, which is the honest way to read those cells rather than a headline.

## Find boxes that contain a point

Search with a zero-size query box at the point. Box overlap is inclusive, so
items touching the point are included.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 2.0, 2.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let point = Point2D::new(1.0, 1.0);
assert_eq!(index.search(Box2D::from_point(point)), vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

For 3D use `Box3D::from_point(point)` the same way.

## Keep payloads outside the index

The index returns item indices. Store your own payloads in the same order as the
boxes you add to the builder.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
let payloads = ["park", "station"];
let boxes = [
    Box2D::new(0.0, 0.0, 2.0, 2.0),
    Box2D::new(5.0, 5.0, 6.0, 6.0),
];

let mut builder = Index2DBuilder::new(boxes.len());
for bounds in boxes {
    builder.add(bounds);
}
let index = builder.finish()?;

let names: Vec<_> = index
    .search(Box2D::new(0.0, 0.0, 3.0, 3.0))
    .into_iter()
    .map(|item| payloads[item])
    .collect();
assert_eq!(names, vec!["park"]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Configuring the builder

```rust
use packed_spatial_index::{DEFAULT_NODE_SIZE, Index2DBuilder, Box2D, SortKey2D};

let mut builder = Index2DBuilder::new(10_000)
    .node_size(DEFAULT_NODE_SIZE) // children per node, clamped to [2, 65535]
    .sort_key(SortKey2D::Hilbert); // stable default ordering

builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
```

Parallel builds (with the `parallel` feature):

```rust
# use packed_spatial_index::{DEFAULT_PARALLEL_MIN_ITEMS, Index2DBuilder};
let builder = Index2DBuilder::new(100_000)
    .parallel(true)
    .parallel_min_items(DEFAULT_PARALLEL_MIN_ITEMS);
```

SIMD and f32 indexes (with `simd` / `f32-storage`):

```rust
# use packed_spatial_index::{Index2DBuilder, Box2D};
let mut builder = Index2DBuilder::new(1);
builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
let simd_index = builder.finish_simd()?;       // SimdIndex2D
# Ok::<(), packed_spatial_index::BuildError>(())
```

`finish_simd()` is also on `Index3DBuilder` (returns `SimdIndex3D`).
`finish_simd_f32()` (both builders) returns the `f32`-storage indexes: half the
box memory, with range results that may include extra near-boundary hits, and
exact range/KNN available when you pass your source boxes back. On AVX-512 the
SIMD `f32` *rounded* range query is also **faster** than the f64 `SimdIndex`
(~1.2–1.45×: half the box bytes plus a wider SIMD batch), so it is a win on speed
and memory when the extra near-boundary hits are acceptable. Prefer the `f64`
indexes when you need *exact* results with many hits (the `f32` `*_exact`
refinement pass is slower on broad queries) and for the fastest exact KNN. Note
the *scalar* `f32` indexes (no `simd`) trade speed for memory — they run a bit
slower than scalar `f64`; the speed win is the SIMD `f32` path.

## 3D

3D uses the same builder/search shape:

```rust
use packed_spatial_index::{Box3D, Index3DBuilder, Point3D};

let mut builder = Index3DBuilder::new(2);
builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));

let index = builder.finish()?;
assert_eq!(index.search(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)), vec![0]);
assert_eq!(index.neighbors(Point3D::new(5.5, 5.5, 5.5), 1), vec![1]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Geographic and custom-metric kNN

`neighbors` orders by squared Euclidean distance. When your coordinates are
longitude/latitude, or you want a different distance entirely, use
`neighbors_metric` (also `neighbors_metric_into` and `visit_neighbors_metric`,
and the same trio on `Index2DView` / `Index3DView`). It takes a closure
`|box| -> f64` returning the distance from your query to a box, and returns the
nearest items in that metric:

```rust
use packed_spatial_index::{Box2D, Index2DBuilder, Point2D, haversine_distance_2d, EARTH_RADIUS_M};

let mut b = Index2DBuilder::new(2);
b.add(Box2D::from_point(Point2D::new(13.405, 52.52)));  // Berlin (lon, lat)
b.add(Box2D::from_point(Point2D::new(2.3522, 48.8566))); // Paris
let index = b.finish()?;

let query = (13.0, 52.4);
let nearest = index.neighbors_metric(
    |bx| haversine_distance_2d(query, bx, EARTH_RADIUS_M),
    1,
    f64::INFINITY, // cutoff is in the metric's units (meters here), not squared
);
assert_eq!(nearest, vec![0]); // Berlin
# Ok::<(), packed_spatial_index::BuildError>(())
```

The closure must return an **admissible lower bound**: the distance to a box may
never exceed the distance to any item inside it. Every "distance to the closest
point of the box" metric satisfies this (a child box sits inside its parent), so
Euclidean, Manhattan, Chebyshev, weighted axes, and the provided
`haversine_distance_2d` all work. The haversine helper clamps the query onto the
box per axis — exact for small boxes, a slight over-estimate for very large or
near-polar ones. `neighbors_metric` is generic, so the default `neighbors` stays
the faster path when plain Euclidean is what you want.

All of these — default, box and custom-metric kNN — run on the same best-first
**distance-browsing** traversal; [docs/knn.md](internals/knn.md) explains the two-queue
technique and why it is the one collect kernel.

## Runnable examples

```bash
cargo run --example basic_2d
cargo run --example basic_3d
cargo run --example persistence_2d
cargo run --example persistence_3d
cargo run --example knn_2d
cargo run --example knn_3d
cargo run --example reuse_workspace_2d
cargo run --example reuse_workspace_3d
cargo run --example f32_exact_2d --no-default-features --features f32-storage
```

## WASM demo

Live demo: <https://filyus.github.io/packed_spatial_index/>

A Vite + TypeScript demo builds `SimdIndex2D` / `SimdIndex3D` WASM wrappers for
interactive 2D and 3D box and point searches:

```bash
cd wasm-demo
npm install
npm run dev      # or: npm run build
```

It uses `wasm-pack` with `RUSTFLAGS=-Ctarget-feature=+simd128` and
`packed_spatial_index` with `default-features = false, features = ["simd"]`,
supports range and nearest-neighbor modes, and is excluded from the published
crates.io package.

