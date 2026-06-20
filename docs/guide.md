# Guide

Practical recipes and configuration. For the per-method API reference, see
[docs.rs](https://docs.rs/packed_spatial_index).

## Choosing a query method

- "Is there any overlap?" use `any`. It returns a `bool`, stops at the first hit,
  and allocates nothing. Prefer it to `search(..).is_empty()`, which builds the
  whole result `Vec` first. Use `first` for one hit, `visit` to fold over hits
  without collecting.
- `search` returns an owned `Vec`. In a hot loop reuse a buffer with `search_into`
  (a caller `Vec`) or `search_with` / `neighbors_with` (a reusable `SearchWorkspace`
  / `NeighborWorkspace`), or skip the `Vec` entirely with `visit` / `any` / `first`.
- `search_iter` (owned `f64` indexes) is a lazy iterator: it descends on demand,
  so `.next()` / `.take(k)` / `.find(..)` stop the traversal early with no result
  `Vec`.
- Use `Index2DView` / `Index3DView` to query persisted bytes without allocating
  an owned index.
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

| Index type | Range | Point kNN | Box kNN | Raycast | Join | Payload | `search_iter` | Streaming |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Index2D` / `Index3D` (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | write | ✓ | ✗ |
| `Index2DView` / `Index3DView` (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | read | ✗ | ✗ |
| `SimdIndex2D` / `SimdIndex3D` (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| SIMD views (f64) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `Index2DF32` / `Index3DF32` (f32) | ✓* | ✓* | ✗ | ✓* | ✗ | write | ✗ | ✗ |
| `SimdIndex2DF32` / `SimdIndex3DF32` (f32) | ✓* | ✓* | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `StreamIndex2D` / `StreamIndex3D` (and `…F32`) | ✓ | ✗ | ✗ | ✗ | ✗ | read | ✗ | ✓ |

Region / culling queries are not a column above: 2D triangle and convex-polygon
(`Index2D` / `Index2DView`) and 3D frustum (`Index3D` / `Index3DView`) are
covered in their own sections below. They are f64-only and not on the SIMD, f32,
or streaming frontends.

The empty cells are intentional, not gaps to fill:

- Streaming covers only range search (with payloads). kNN and raycast use a
  best-first traversal that jumps around the tree, so adjacent reads do not
  coalesce; streaming them would be one read per node. Load those with a view or
  an in-memory index. (The in-memory and `f32` indexes serialize the files that
  `StreamIndex*` reads.)
- The `f32` indexes answer range, point-kNN, and (scalar only) raycast as a
  conservative superset; refine with the `*_exact` family against your own `f64`
  boxes. The SIMD `f32` frontend carries no payload and no raycast; the compact
  mesh-BVH story uses the scalar `Index3DF32` (AABBs from `from_triangles`,
  triangles as the payload).
- Payload read lives on the byte views and `StreamIndex*`, not the owned or SIMD
  indexes: an owned index returns ids into your own data, so attach a per-item
  blob at serialize time and read it back through a view or streamed.

## Query by a triangle (2D)

`Index2D` answers a triangle region query directly:
`search_triangle` / `search_triangle_into` (collect), `any_triangle` (boolean,
short-circuits), and `visit_triangle` (fold without collecting). Each returns the
items whose box overlaps the triangle's filled area — the bounding-box corners
the triangle misses are rejected during the traversal.

```rust
# use packed_spatial_index::{Index2DBuilder, Box2D, Triangle2D};
# let mut b = Index2DBuilder::new(2);
# b.add(Box2D::new(0.2, 0.2, 0.3, 0.3));
# b.add(Box2D::new(9.0, 9.0, 9.5, 9.5));
# let index = b.finish()?;
let tri = Triangle2D::new([0.0, 0.0], [10.0, 0.0], [0.0, 10.0]);
assert_eq!(index.search_triangle(tri), vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

Prefer this to `search(tri.aabb())` filtered by hand. It is both tighter and
faster: in a 200k-box field it rejects roughly 2× (fat triangle) to 7× (sliver)
of the bounding-box hits, and runs ~2.5×–5× faster than collect-then-filter —
internal nodes are pruned with a cheap box-vs-bbox test, subtrees fully inside
the triangle are accepted whole without per-item tests, and the full
triangle-AABB separating-axis test runs only at boundary leaves. `any_triangle`
is the exact-culling analogue of `any`. The same methods are on the zero-copy
`Index2DView`, so you can run triangle queries straight over serialized bytes.

## Query by a convex polygon (2D)

`Index2D` also answers an arbitrary **convex polygon** region query — the N-gon
generalization of the triangle query: `search_polygon` / `search_polygon_into`
(collect), `any_polygon` (boolean, short-circuits), and `visit_polygon` (fold).
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
assert_eq!(index.search_polygon(&trapezoid), vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

The test is exact (a separating-axis test over the box's two axes and the
polygon's edge normals), so the result is precisely the boxes the polygon's
filled area overlaps. Two wins over `search` on the polygon's bounding box,
measured in a 200k-box field:

- **Tighter:** ~1.5x fewer hits for a near-round polygon (hexagon/octagon), up to
  ~4.6x for a narrow trapezoid — the win tracks how much slimmer the shape is
  than its bounding box.
- **Faster anyway:** `search_polygon` beats collecting `search(bbox)` and
  filtering by hand by **~2.2x even for the round octagon** (its weakest
  selectivity case) and up to **~13x for a wide trapezoid** — internal nodes are
  pruned with the polygon test and subtrees fully inside are accepted whole,
  instead of materializing the whole bounding-box result and filtering every box.

For a triangle, `Triangle2D` + `search_triangle` is ~1.4x faster than a
three-vertex polygon (fixed vertices, no per-edge loop) and returns the same set.
`search_polygon` and friends are also on the zero-copy `Index2DView`.

## Frustum culling (3D)

`Index3D` answers a view-frustum query directly: `search_frustum` /
`search_frustum_into` (collect), `any_frustum` (boolean, short-circuits), and
`visit_frustum` (fold without collecting). Build a [`Frustum3D`] from six
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
assert_eq!(index.search_frustum(frustum), vec![0]);
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
same methods are on the zero-copy `Index3DView`.

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
(~1.3–1.5×: half the box bytes plus a wider SIMD batch), so it is a win on speed
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
