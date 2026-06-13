# Guide

Practical recipes and configuration. For the per-method API reference, see
[docs.rs](https://docs.rs/packed_spatial_index).

## Choosing a query method

- Use `search` for simple one-off range queries.
- Use `search_with` / `neighbors_with` inside tight loops (reuses buffers).
- Use `any`, `first`, `visit`, `visit_neighbors`, or `visit_raycast` when you can
  stop early.
- Use `Index2DView` / `Index3DView` to query persisted bytes without allocating
  an owned index.
- Use `search_exact` / `neighbors_exact` on the `f32` indexes for exact results
  from compact storage; prefer the `f64` indexes for exact queries with many
  hits.

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
exact range/KNN available when you pass your source boxes back. Prefer `f64`
indexes for exact range queries with many hits and fastest exact KNN.

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
