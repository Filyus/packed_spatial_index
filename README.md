# packed_spatial_index

[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/packed_spatial_index.svg)](https://crates.io/crates/packed_spatial_index)
[![docs.rs](https://docs.rs/packed_spatial_index/badge.svg)](https://docs.rs/packed_spatial_index)

A fast, packed **static spatial index** for 2D and 3D axis-aligned bounding
boxes (AABBs). It builds a packed **Hilbert R-tree** (in the style of
[flatbush](https://github.com/mourner/flatbush) /
[`static_aabb2d_index`](https://crates.io/crates/static_aabb2d_index)) once, then
answers many queries: **range / intersection search**, **nearest-neighbor (kNN)**
from a point or a box, **ray casts**, and **spatial joins**. With the `simd`
feature the SoA indexes use AVX2 / AVX-512 intersection tests; indexes also
serialize to a stable byte format with **zero-copy views** (mmap-friendly).

[Live WASM demo](https://filyus.github.io/packed_spatial_index/)

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
packed_spatial_index = "0.5"
```

## When to use it

Use this crate when your geometry is static or rebuilt in batches, you can key
results by insertion-order index into your own payload array, and you want a
compact in-memory (or mmap'd) index with reusable buffers for high query
throughput. It is **not** a dynamic R-tree — there are no insert/delete
operations after `finish()`.

## Queries at a glance

Every query exists on `Index2D` / `Index3D`, the `simd`-feature `SimdIndex2D` /
`SimdIndex3D`, and the zero-copy views. Range/ray results are item indices in
insertion order; result order is unspecified. See
[docs.rs](https://docs.rs/packed_spatial_index) for full per-method docs.

| Query | Methods |
| --- | --- |
| Range / intersection | [`search`][search], [`search_into`][search_into], [`search_with`][search_with], [`any`][any], [`first`][first], [`visit`][visit] |
| Nearest neighbors (point) | [`neighbors`][neighbors], [`neighbors_within`][neighbors_within], [`neighbors_into`][neighbors_into], [`neighbors_with`][neighbors_with], [`visit_neighbors`][visit_neighbors] |
| Nearest neighbors (box) | [`neighbors_of_box`][neighbors_of_box], [`neighbors_of_box_within`][neighbors_of_box_within], [`neighbors_of_box_into`][neighbors_of_box_into], [`neighbors_of_box_with`][neighbors_of_box_with], [`visit_neighbors_of_box`][visit_neighbors_of_box] |
| Ray segment | [`raycast`][raycast], [`raycast_into`][raycast_into], [`raycast_with`][raycast_with], [`raycast_closest`][raycast_closest], [`raycast_closest_with`][raycast_closest_with], [`visit_raycast`][visit_raycast] |
| Spatial join | [`join`][join], [`join_with`][join_with], [`self_join`][self_join], [`self_join_with`][self_join_with] |
| Extent / exact | [`extent`][extent], and [`search_exact`][search_exact] / [`neighbors_exact`][neighbors_exact] on the `f32` indexes |

```rust
# use packed_spatial_index::{Index2DBuilder, Box2D, Point2D, Ray2D};
# let mut b = Index2DBuilder::new(2);
# b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
# let index = b.finish()?;
let overlaps = index.search(Box2D::new(0.0, 0.0, 2.0, 2.0)); // range query
let nearest = index.neighbors(Point2D::new(5.5, 5.5), 1);    // kNN
let hit = index.raycast_closest(Ray2D::new(Point2D::new(-1.0, 0.5), 1.0, 0.0, 10.0));
assert_eq!(overlaps, vec![0]);
assert_eq!(nearest, vec![1]);
assert_eq!(hit, Some((0, 1.0)));
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Types at a glance

- **Geometry**: `Box2D`, `Box3D` (inclusive `overlaps` / `contains` /
  `contains_point` / `from_point`), `Point2D`, `Point3D`, `Ray2D`, `Ray3D`.
- **Builders**: `Index2DBuilder`, `Index3DBuilder` — `finish()` (scalar),
  `finish_simd()` (SoA + SIMD), `finish_simd_f32()` (compact f32 boxes).
- **Indexes**: `Index2D` / `Index3D` (scalar), `SimdIndex2D` / `SimdIndex3D`
  (SIMD), `SimdIndex2DF32` / `SimdIndex3DF32` (half-memory f32 boxes).
- **Views**: zero-copy `*View` types over serialized bytes for every index.
- **Workspaces**: `SearchWorkspace`, `NeighborWorkspace` reuse buffers in loops.
- **Sorting / errors**: `SortKey2D` / `SortKey3D` (default `Hilbert`),
  `BoundsError`, `BuildError`, `LoadError`.

## Features

- `parallel` *(default)* — adaptive rayon-based parallel builds.
- `simd` *(default)* — SoA indexes and SIMD search/raycast (`wide` + AVX-512).
- `f32-storage` — compact f32-storage SIMD indexes (implies `simd`).
- `bench-internals` — hidden support API for this crate's benchmarks.

```bash
cargo build --no-default-features                      # minimal
cargo build --no-default-features --features simd      # SIMD only
```

## Documentation

- **[Guide](docs/guide.md)** — recipes, choosing a query method, builder configuration, examples, WASM demo.
- **[Persistence](docs/persistence.md)** — serialize / load / zero-copy views, and querying large or on-disk indexes via mmap.
- **[Performance](docs/performance.md)** — benchmarks vs `static_aabb2d_index`, FlatGeobuf, and the `bvh` crate.
- **[Binary format](FORMAT.md)** — the `PSINDEX` on-disk layout.
- **API reference** — [docs.rs/packed_spatial_index](https://docs.rs/packed_spatial_index).

## Limitations

- Static: rebuild when the dataset changes; no insert/delete.
- Results are item indices, not stored payloads; result order is unspecified.
- `f32-storage` indexes store outward-rounded boxes — plain range search may
  return extra near-boundary hits; use `search_exact` / `neighbors_exact` (with
  your source `f64` boxes) for exact results, and prefer `f64` indexes for exact
  queries with many hits.

## Safety

The public API is safe Rust. Internally `unsafe` is confined to narrow, audited
paths: validated unaligned little-endian reads in the byte views, bulk
`repr(C)` byte copies during serialization on little-endian targets, and
runtime-feature-gated x86-64 SIMD (AVX-512) loads/prefetch. Loaded buffers are
validated before use, so malformed input returns `LoadError` rather than relying
on caller invariants.

## Status

Major API changes are not planned, but remain possible before a `1.0` release.

## AI usage note

AI assistance is part of my development process for this project. I guide the
architecture, review generated output carefully, and take responsibility for the
crate as published.

## License

Licensed under the Apache License, Version 2.0.

<!-- docs.rs method links -->
[search]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search
[search_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_into
[search_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_with
[any]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.any
[first]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.first
[visit]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.visit
[neighbors]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors
[neighbors_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_within
[neighbors_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_into
[neighbors_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_with
[visit_neighbors]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.visit_neighbors
[neighbors_of_box]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box
[neighbors_of_box_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box_within
[neighbors_of_box_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box_into
[neighbors_of_box_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box_with
[visit_neighbors_of_box]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.visit_neighbors_of_box
[raycast]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast
[raycast_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_into
[raycast_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_with
[raycast_closest]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_closest
[raycast_closest_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_closest_with
[visit_raycast]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.visit_raycast
[join]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.join
[join_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.join_with
[self_join]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.self_join
[self_join_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.self_join_with
[extent]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.extent
[search_exact]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2DF32.html#method.search_exact
[neighbors_exact]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2DF32.html#method.neighbors_exact
