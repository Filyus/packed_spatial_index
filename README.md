# packed_spatial_index

[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/packed_spatial_index.svg)](https://crates.io/crates/packed_spatial_index)
[![docs.rs](https://docs.rs/packed_spatial_index/badge.svg)](https://docs.rs/packed_spatial_index)

[Live WASM demo](https://filyus.github.io/packed_spatial_index/)

`packed_spatial_index` is a packed static spatial index for 2D and 3D
axis-aligned bounding boxes.

It is built for read-heavy workloads where the full set of boxes is known up
front: build once, then run many range/intersection searches. The default
`Index2D` and `Index3D` use packed Hilbert R-tree layouts. With the `simd`
feature, `SimdIndex2D` and `SimdIndex3D` store boxes in structure-of-arrays form
and use SIMD intersection checks. Scalar and SIMD indexes share the same
canonical byte format for owned persistence.

```rust
use packed_spatial_index::{Index2DBuilder, Box2D};

let mut builder = Index2DBuilder::new(2);
builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));

let index = builder.finish()?;
let hits = index.search(Box2D::new(0.0, 0.0, 2.0, 2.0));

assert_eq!(hits, vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Installation

Requires Rust 1.89 or newer.

```toml
[dependencies]
packed_spatial_index = "0.4"
```

## When To Use It

Use this crate when:

- your geometry is static or rebuilt in batches;
- search results can be returned as insertion-order indices into your own payload array;
- you want a compact in-memory index with reusable buffers for repeated searches;
- batch search throughput matters.

It is not a dynamic R-tree: there are no insert/delete operations after build.

## Limitations

These are the main API contracts and trade-offs to keep in mind.

- The index is static: rebuild it when the dataset changes.
- 2D and 3D axis-aligned bounding boxes are supported.
- Search results are item indices, not stored payloads or geometries.
- Result ordering is not a stable API guarantee.
- Persistence uses canonical `Index2D` and `Index3D` byte layouts.
  `SimdIndex2D` and `SimdIndex3D` can save and load those same bytes, but there
  is no separate persisted SoA byte format. With `f32-storage`, the `f32`
  indexes use their own `f32` box layout (a distinct format flag).
- With `f32-storage`, `SimdIndex2DF32` and `SimdIndex3DF32` store
  outward-rounded boxes. Plain range search returns every exact hit, but can
  also return extra near-edge hits.
- `search_exact` and `visit_exact` use your original `f64` boxes for exact
  range hits. They are useful with compact indexes when exact queries return
  few hits. For exact range queries with many hits, prefer the `f64` indexes.
- `f64` nearest-neighbor search is exact over indexed boxes. `f32` KNN can use
  `neighbors_exact` for exact results, but fastest exact KNN is usually the
  `f64` indexes. Dynamic spatial joins are out of scope for now.

## Main Types

### Geometry

- `Box2D` and `Box3D` are public AABB types, with inclusive `overlaps`,
  `contains`, `contains_point`, and `from_point` helpers.
- `Point2D` and `Point3D` are public point types for KNN and point queries.

### Builders

- `Index2DBuilder` and `Index3DBuilder` build scalar indexes, or SIMD indexes
  with the `simd` feature.

### Indexes

- `Index2D` and `Index3D` are scalar read-only indexes.
- `SimdIndex2D` and `SimdIndex3D` are available with the `simd` feature and have
  the same search API and owned persistence API as their scalar counterparts.
- `SimdIndex2DF32` and `SimdIndex3DF32` (with `f32-storage`, built via
  `finish_simd_f32`) store coordinates as `f32` rounded outward, halving box
  memory. Plain range queries may include extra near-boundary hits; exact range
  and KNN are available through `*_exact` callbacks to your source boxes (see
  Limitations).

### Views

- `Index2DView` and `Index3DView` are zero-copy read-only views over bytes
  produced by scalar or SIMD `to_bytes` methods.
- `SimdIndex2DView` and `SimdIndex3DView` are zero-copy SIMD views over the
  canonical byte format.
- `SimdIndex2DF32View` and `SimdIndex3DF32View` are zero-copy views over f32
  index bytes.

### Workspaces

- `SearchWorkspace` reuses result and traversal buffers.
- `NeighborWorkspace` reuses result and priority-queue buffers for KNN.

### Sorting

- `SortKey2D` selects the public build ordering curve. `Hilbert` is the stable
  default.
- `SortKey3D` does the same for 3D. `Hilbert` is the stable default.

### Errors

- `BoundsError` is returned by checked box constructors.
- `BuildError` is returned when build inputs are invalid.
- `LoadError` is returned when loading invalid or unsupported bytes.

## Querying

Searches take a `Box2D` or `Box3D` and return indices into the item list you
added to the builder. Result order is intentionally unspecified.

### `search`

Allocates a fresh `Vec<usize>` and returns all overlaps. This is the simplest
choice for one-off queries.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let hits = index.search(Box2D::new(0.0, 0.0, 2.0, 2.0));
assert_eq!(hits, vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `search_into`

Reuses your result `Vec`, clearing it before writing new hits.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let mut results = Vec::new();
index.search_into(Box2D::new(0.0, 0.0, 2.0, 2.0), &mut results);
assert_eq!(results, vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `search_with`

Reuses both the result buffer and the internal traversal stack through a
`SearchWorkspace`. This is the best fit for hot query loops.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, SearchWorkspace};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;

let mut workspace = SearchWorkspace::new();
let hits = index.search_with(Box2D::new(0.0, 0.0, 2.0, 2.0), &mut workspace);
assert_eq!(hits, &[0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `any`

Checks whether at least one item overlaps the query.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let query = Box2D::new(0.0, 0.0, 2.0, 2.0);

assert!(index.any(query));
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `first`

Returns one matching item index found by traversal.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let query = Box2D::new(0.0, 0.0, 2.0, 2.0);

assert_eq!(index.first(query), Some(0));
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `visit`

Calls your visitor for each match and lets it stop early with
`ControlFlow::Break`.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
use std::ops::ControlFlow;
#
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let query = Box2D::new(0.0, 0.0, 2.0, 2.0);
let first_even = index.visit(query, |item| {
    if item % 2 == 0 {
        ControlFlow::Break(item)
    } else {
        ControlFlow::Continue(())
    }
});
assert_eq!(first_even, ControlFlow::Break(0));
# Ok::<(), packed_spatial_index::BuildError>(())
```

### f32 exact search

`finish_simd_f32()` stores outward-rounded `f32` boxes. Plain range search
may include extra near-boundary hits. `search_exact` checks your original `f64`
boxes.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
let boxes = [
    Box2D::new(1.0 + 1e-8, 0.0, 1.0 + 1e-8, 0.0),
    Box2D::new(1.0, 0.0, 1.0, 0.0),
];

let mut builder = Index2DBuilder::new(boxes.len());
for &b in &boxes {
    builder.add(b);
}
let index = builder.finish_simd_f32()?;

let query = Box2D::new(1.0, 0.0, 1.0, 0.0);
let mut rounded_hits = index.search(query);
rounded_hits.sort_unstable();
assert_eq!(rounded_hits, vec![0, 1]);

let exact = index.search_exact(query, |i| boxes[i]);
assert_eq!(exact, vec![1]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

The same pattern applies to exact KNN through `neighbors_exact`. See
`examples/f32_exact_2d.rs`.

### `extent`

`extent()` returns the total box covering every item, or `None` for an empty
index.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
assert_eq!(index.extent(), Some(Box2D::new(0.0, 0.0, 6.0, 6.0)));

let empty = Index2DBuilder::new(0).finish()?;
assert_eq!(empty.extent(), None);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### Nearest Neighbors

Nearest-neighbor queries are exact over boxes for the `f64` indexes. Distance is
zero when the point is inside a box, otherwise it is the Euclidean distance to
the nearest point on the box. The `f32` indexes also expose rounded-box KNN.
For exact KNN on source `f64` boxes, use their `neighbors_exact` methods.
For fastest exact KNN, prefer the `f64` indexes.

### `neighbors`

Returns the nearest item indices with no distance limit.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let point = Point2D::new(5.5, 5.5);

let nearest = index.neighbors(point, 1);
assert_eq!(nearest, vec![1]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `neighbors_within`

Returns nearest item indices within a maximum distance.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let point = Point2D::new(5.5, 5.5);

let nearby = index.neighbors_within(point, 8, 2.0);
assert_eq!(nearby, vec![1]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `neighbors_into`

Reuses your result `Vec` for repeated KNN queries.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let mut results = Vec::new();
index.neighbors_into(Point2D::new(5.5, 5.5), 4, f64::INFINITY, &mut results);
assert_eq!(results, vec![1, 0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `neighbors_with`

Reuses both result and queue buffers through a `NeighborWorkspace`.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, NeighborWorkspace, Point2D};
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let mut workspace = NeighborWorkspace::new();
let hits = index.neighbors_with(Point2D::new(5.5, 5.5), 4, f64::INFINITY, &mut workspace);
assert_eq!(hits, &[1, 0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

### `visit_neighbors`

Visits `(index, distance_squared)` pairs in nearest-first order and can stop
early with `ControlFlow::Break`.

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};
use std::ops::ControlFlow;
#
# let mut builder = Index2DBuilder::new(2);
# builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = builder.finish()?;
let close = index.visit_neighbors(Point2D::new(5.5, 5.5), 10.0, |item, distance_squared| {
    if distance_squared <= 1.0 {
        ControlFlow::Break(item)
    } else {
        ControlFlow::Continue(())
    }
});
assert_eq!(close, ControlFlow::Break(1));
# Ok::<(), packed_spatial_index::BuildError>(())
```

Results are returned in nondecreasing distance order. Ties between equal-distance
items are not stable across index layouts.

## Common Tasks

### Find boxes that contain a point

Search with a zero-size query box at that point. Box overlap is inclusive, so
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

For 3D, use `Box3D::from_point(point)` in the same way.

### Keep payloads outside the index

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

### Choose a query method

- Use `search` for simple one-off queries.
- Use `search_with` or `neighbors_with` inside tight loops.
- Use `any`, `first`, `visit`, or `visit_neighbors` when you can stop early.
- Use `Index2DView` or `Index3DView` when loading persisted bytes without
  allocating an owned index.

## Builder

```rust
use packed_spatial_index::{DEFAULT_NODE_SIZE, Index2DBuilder, Box2D, SortKey2D};

let mut builder = Index2DBuilder::new(10_000)
    .node_size(DEFAULT_NODE_SIZE)
    .sort_key(SortKey2D::Hilbert);

builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
```

With `parallel` enabled:

```rust
# use packed_spatial_index::{DEFAULT_PARALLEL_MIN_ITEMS, Index2DBuilder};
let builder = Index2DBuilder::new(100_000)
    .parallel(true)
    .parallel_min_items(DEFAULT_PARALLEL_MIN_ITEMS);
```

With `simd` enabled:

```rust
# use packed_spatial_index::{Index2DBuilder, Box2D};
let mut builder = Index2DBuilder::new(1);
builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
let simd_index = builder.finish_simd()?;
# Ok::<(), packed_spatial_index::BuildError>(())
```

The same `finish_simd()` method is available on `Index3DBuilder` and returns
`SimdIndex3D`. `finish_simd_f32()` (on both builders) returns the `f32`-storage
`SimdIndex2DF32` / `SimdIndex3DF32`: half the box memory, with range results
that may include extra near-boundary hits. Exact range/KNN is available when you
pass back your source boxes.
Prefer `f64` indexes for exact range queries with many hits and fastest exact
KNN.

3D uses the same builder/search shape:

```rust
use packed_spatial_index::{Box3D, Index3DBuilder, Point3D};

let mut builder = Index3DBuilder::new(2);
builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));

let index = builder.finish()?;
assert_eq!(
    index.search(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)),
    vec![0]
);
assert_eq!(index.neighbors(Point3D::new(5.5, 5.5, 5.5), 1), vec![1]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Persistence

`Index2D` and `Index3D` can be serialized to stable little-endian byte formats
and loaded back either as owned indexes or as zero-copy views:

```rust
use packed_spatial_index::{Index2D, Index2DBuilder, Index2DView, Box2D};

let mut builder = Index2DBuilder::new(1);
builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
let index = builder.finish()?;

let bytes = index.to_bytes();
let mut reusable = Vec::new();
index.to_bytes_into(&mut reusable);
assert_eq!(reusable, bytes);

let owned = Index2D::from_bytes(&bytes)?;
let view = Index2DView::from_bytes(&bytes)?;

assert_eq!(owned.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
assert_eq!(view.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

3D persistence uses the same header and sections, with a dimension flag and
six `f64` coordinates per stored box. With the `simd` feature,
`SimdIndex2D` and `SimdIndex3D` read and write the same canonical bytes as the
scalar indexes. Loading a SIMD index is an owned load that scatters canonical
box records into SoA columns. `SimdIndex2DView` and `SimdIndex3DView` borrow the
same canonical bytes for zero-copy SIMD-over-AoS queries.

The `f32` indexes persist to their own `f32` box layout (distinct format flags),
with matching `from_bytes` loaders and zero-copy views.

The binary layout is documented in [`FORMAT.md`](FORMAT.md).

## Examples

Runnable examples cover the public paths:

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

## WASM Demo

Live demo: <https://filyus.github.io/packed_spatial_index/>

The repository includes a Vite + TypeScript demo that builds `SimdIndex2D` and
`SimdIndex3D` WASM wrappers for interactive 2D and 3D box and point searches:

```bash
cd wasm-demo
npm install
npm run dev
```

Production build:

```bash
cd wasm-demo
npm run build
```

The demo uses `wasm-pack` with `RUSTFLAGS=-Ctarget-feature=+simd128` and
`packed_spatial_index` with `default-features = false, features = ["simd"]`.
It supports range search and nearest-neighbor modes, renders with WebGL2, and
is excluded from the published crates.io package.

## Benchmarking Layout

Performance-related code lives under `benches`:

- `benches/*.rs` are Criterion benchmark suites run with `cargo bench`.
- `benches/tools` is a local developer package for quick comparisons of encoder
  variants, sort strategies, node sizes, parallel builds, and SoA layouts.

The local tools use the hidden `bench-internals` feature and are excluded from
the published crate.

```bash
cargo run --release --manifest-path benches/tools/Cargo.toml --bin sortkey_quality_2d
cargo run --release --manifest-path benches/tools/Cargo.toml --bin node_size_3d
```

## Features

Runtime acceleration features:

- `parallel`: adaptive rayon-based index builds through `Index2DBuilder::parallel`
  and `Index3DBuilder::parallel`. Enabled by default.
- `simd`: SoA index and SIMD search paths through `SimdIndex2D` and
  `SimdIndex3D`, plus owned and zero-copy SIMD persistence through the canonical
  byte format. Enabled by default.
- `f32-storage`: compact f32-storage SIMD indexes. It enables `simd` and is not
  enabled by default.
- `bench-internals`: hidden support API for this crate's own benchmarks and
  local performance tools. It is not enabled by default and is not part of the
  stable user-facing API.

Minimal build:

```bash
cargo build --no-default-features
```

Feature-specific builds:

```bash
cargo build --no-default-features --features simd
cargo build --no-default-features --features parallel
cargo build --no-default-features --features f32-storage
```

## Safety

The public API is safe Rust; users do not need `unsafe` to build, load, search,
or query neighbors.

Internally, the crate keeps `unsafe` limited to narrow performance-sensitive
paths:

- unaligned little-endian reads for validated `Index2DView` and `Index3DView`
  byte buffers;
- bulk byte copies for `repr(C)` boxes and index sections when serializing on
  compatible little-endian targets;
- x86/x86_64 prefetch intrinsics used only by hidden benchmark/performance-tool paths;
- AVX-512 loads in the `simd` feature, guarded by runtime CPU feature detection.

Loaded buffers are validated before they can be searched, so malformed input is
reported as `LoadError` instead of relying on caller-side invariants.

## Prior Art

This crate builds on ideas from existing packed spatial index work.

- [`flatbush`](https://github.com/mourner/flatbush) by Vladimir Agafonkin is a
  static packed Hilbert R-tree for 2D rectangles in JavaScript.
- [`static_aabb2d_index`](https://crates.io/crates/static_aabb2d_index) by
  Jedidiah McCready is the Rust Flatbush port.
- [FlatGeobuf](https://flatgeobuf.org/) by Pirmin Kalberer and Björn Harrtell
  is a geospatial format inspired by Flatbush.

## Performance Notes

### 2D Competitors

Lower is better. The 2D competitor workload uses 100,000 random AABBs and
1,000 random query boxes; build and search competitors are measured in the
same benchmark suite on the same generated inputs. Persistence rows use the
canonical byte format for 100,000 boxes.

| Benchmark | FlatGeobuf | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: | ---: |
| Full build | 70.18 ms | 8.95 ms | 3.18 ms serial / 2.20 ms parallel | - |
| Search batch | 555.83 us | 341.56 us | 416.15 us | 128.64 us |
| Serialize built tree (fresh buffer) | - | - | 407.64 us | 689.42 us |
| Serialize built tree (reused buffer) | 131.93 us | - | 68.78 us | 140.21 us |
| Load owned tree | 740.23 us | - | 607.62 us | 935.51 us |
| Load zero-copy view | - | - | 37.59 us | n/a |

Scalar `Index2D` search versus `static_aabb2d_index` is dataset-sensitive.
Two search runs with the same item/query counts but different generated inputs
showed opposite scalar ordering:

| Search batch | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: |
| `flatgeobuf2d_bench`, seed `0xF6B` | 341.56 us | 416.15 us | 128.64 us |
| `index2d_bench`, seed `0xB0B` | 643.85 us | 311.72 us | 126.21 us |

### 2D vs 3D

Lower latency is better. The `3D speed` column is `2D latency / 3D latency`, so
values above `1.00x` mean 3D is faster. The build workload uses 100,000 boxes
with `node_size = 16`; search and KNN use 1,000 query boxes or points.

| Stage | Dataset / mode | `Index2D` | `Index3D` | 3D speed |
| --- | --- | ---: | ---: | ---: |
| Hilbert encode | production 2D LUT vs 3D nibble LUT | 739.90 us | 983.42 us | 0.75x |
| Build | planar XY | 2.7433 ms | 4.5660 ms | 0.60x |
| Build | uniform XYZ | 3.4270 ms | 5.1252 ms | 0.67x |
| Search batch | planar XY | 466.41 us | 636.37 us | 0.73x |
| Search batch | uniform XYZ | 495.82 us | 391.32 us | 1.27x |

| KNN batch | Dataset / mode | `Index2D` | `Index3D` | 3D speed |
| --- | --- | ---: | ---: | ---: |
| Top-1 | planar XY | 973.85 us | 1.2445 ms | 0.78x |
| Top-10 | planar XY | 1.9378 ms | 2.5625 ms | 0.76x |
| Top-1 | uniform XYZ | 1.0028 ms | 1.8530 ms | 0.54x |
| Top-10 | uniform XYZ | 2.0221 ms | 4.1143 ms | 0.49x |

| Persistence | `Index2D` | `Index3D` | 3D speed |
| --- | ---: | ---: | ---: |
| Serialize built tree (fresh buffer) | 407.64 us | 562.55 us | 0.72x |
| Serialize built tree (reused buffer) | 68.78 us | 96.51 us | 0.71x |
| Load owned tree | 607.62 us | 819.04 us | 0.74x |
| Load zero-copy view | 37.59 us | 37.66 us | 1.00x |

| SIMD persistence | `SimdIndex2D` | `SimdIndex3D` | 3D speed |
| --- | ---: | ---: | ---: |
| Serialize built tree (fresh buffer) | 689.42 us | 973.17 us | 0.71x |
| Serialize built tree (reused buffer) | 140.21 us | 240.51 us | 0.58x |
| Load owned tree | 935.51 us | 1.3083 ms | 0.72x |

### 3D SIMD

The speed column is scalar/serial latency divided by SIMD/parallel latency, so
values above `1.00x` mean the SIMD or parallel path is faster.

| Stage | Dataset / mode | Baseline | SIMD / parallel | Speed |
| --- | --- | ---: | ---: | ---: |
| Search batch | uniform XYZ | `Index3D` 389.13 us | `SimdIndex3D` 129.08 us | 3.01x |
| Search batch | flat Z | `Index3D` 1.8443 ms | `SimdIndex3D` 1.1514 ms | 1.60x |
| Build `finish_simd` | uniform XYZ, 200k boxes | serial 10.632 ms | parallel 6.5412 ms | 1.63x |

### f32 Storage vs f64

The `coord_precision` suite compares compact f32 storage with f64 storage.
Lower is better. Range rows run `search(Box2D)` for 1,000 random query boxes.
Small query boxes cover 0.1% of the coordinate extent per axis; large query
boxes cover 5%. KNN rows use 200 query points with top-8 results.

Quick selector:

- `SimdIndex2D`: 32-byte f64 boxes. Use for exact range queries with many hits
  and fastest exact KNN.
- `SimdIndex2DF32::search`: 16-byte rounded f32 boxes. Use for compact
  first-pass filtering, or when near-boundary false positives are OK.
- `SimdIndex2DF32::*_exact`: 16-byte f32 index plus source f64 boxes. Use when
  exact range queries return few hits and compact storage matters. Exact KNN is
  available, but f64 is faster in these runs.

| Range query | Items | `f64` exact | `f32` rounded | `f32` exact |
| --- | ---: | ---: | ---: | ---: |
| small query boxes | 10k | 84 us | 72 us | 73 us |
| small query boxes | 100k | 115 us | 87 us | 96 us |
| small query boxes | 1M | 156 us | 125 us | 134 us |
| large query boxes | 10k | 149 us | 117 us | 292 us |
| large query boxes | 100k | 1.06 ms | 704 us | 1.79 ms |
| large query boxes | 1M | 9.61 ms | 6.09 ms | 16.59 ms |

| KNN workload | `f64` exact | `f32` rounded | `f32` exact |
| --- | ---: | ---: | ---: |
| 10k items | 243 us | 282 us | 302 us |
| 100k items | 357 us | 376 us | 410 us |

### Summary

- `Index2D` is the general-purpose path;
- `SimdIndex2D` and `SimdIndex3D` are best for heavier query batches where SIMD
  work amortizes well;
- scalar `Index2D` search versus `static_aabb2d_index` depends on the generated
  data and query distribution, while `Index2D` build is faster in these runs;
- `Index3D` build and KNN are still slower than `Index2D`, but uniform 3D search
  can be faster when Z meaningfully prunes the tree;
- f32 storage halves box memory; exact callbacks trade source-box lookup for
  exact results;
- SIMD persistence uses the same canonical bytes as scalar persistence; it pays
  an SoA gather/scatter cost but avoids a second file format;
- `any` is often much faster than collecting full result sets when all you need is existence;
- AVX-512 is not always the fastest path in parallel workloads because CPU frequency behavior matters.
- `flatgeobuf2d_bench` compares against FlatGeobuf's packed Hilbert R-tree;
- `index2d_bench` compares build/search paths against `static_aabb2d_index`;
- `index3d_bench` covers 3D build/search/KNN, SIMD search/build, dimension
  comparisons, node sizes, and hidden Morton baseline;
- `persistence_knn2d_bench` covers 2D scalar/SIMD persistence, loaded views, and KNN;
- `persistence_knn3d_bench` covers 3D scalar/SIMD persistence, loaded views, and KNN.

### Reproducing

Run the focused benchmark suites with:

```bash
cargo bench --bench index2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench index3d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench persistence_knn2d_bench --no-default-features --features simd,bench-internals
cargo bench --bench persistence_knn3d_bench --no-default-features --features simd,bench-internals
cargo bench --bench flatgeobuf2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench coord_precision --no-default-features --features f32-storage
```

## Status

Major API changes are not planned, but remain possible before a `1.0` release.

## AI Usage Note

AI assistance is part of my development process for this project. I guide the
architecture, review generated output carefully and take responsibility for the
crate as published.

## License

Licensed under the Apache License, Version 2.0.
