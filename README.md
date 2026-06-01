# packed_spatial_index

[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/packed_spatial_index.svg)](https://crates.io/crates/packed_spatial_index)
[![docs.rs](https://docs.rs/packed_spatial_index/badge.svg)](https://docs.rs/packed_spatial_index)

`packed_spatial_index` is a packed static spatial index for 2D and 3D
axis-aligned bounding boxes.

It is built for read-heavy workloads where the full set of boxes is known up
front: build once, then run many window/intersection searches. The default
`Index2D` and `Index3D` use packed Hilbert R-tree layouts. With the `simd`
feature, `SimdIndex2D` stores 2D boxes in structure-of-arrays form and uses SIMD
intersection checks.

```rust
use packed_spatial_index::{Index2DBuilder, Bounds2D};

let mut builder = Index2DBuilder::new(2);
builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
builder.add(Bounds2D::new(5.0, 5.0, 6.0, 6.0));

let index = builder.finish()?;
let hits = index.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0));

assert_eq!(hits, vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Installation

Requires Rust 1.89 or newer.

```toml
[dependencies]
packed_spatial_index = "0.3"
```

## When To Use It

Use this crate when:

- your geometry is static or rebuilt in batches;
- search results can be returned as insertion-order indices into your own payload array;
- you want a compact in-memory index with reusable buffers for repeated searches;
- batch search throughput matters.

It is not a dynamic R-tree: there are no insert/delete operations after build.

## Limitations

- The index is static: rebuild it when the dataset changes.
- 2D and scalar 3D axis-aligned bounding boxes are supported.
- Search results are item indices, not stored payloads or geometries.
- Result ordering is not a stable API guarantee.
- Persistence is defined for the canonical `Index2D` format. `Index3D` and
  `SimdIndex2D` can be rebuilt from source boxes but do not have stable file
  formats yet.
- Nearest-neighbor search is exact over indexed bounds; approximate KNN and dynamic
  spatial joins are out of scope for now.

## Main Types

- `Bounds2D` is the public AABB type, with inclusive `overlaps`, `contains`, and
  `contains_point` helpers. `Bounds2D::new` is unchecked; use `Bounds2D::try_new` for
  untrusted bounds.
- `Bounds3D` and `Point3D` are the equivalent scalar 3D geometry types.
- `Index2DBuilder` builds either `Index2D` or, with `simd`, `SimdIndex2D`.
- `Index3DBuilder` builds scalar `Index3D`.
- `Index2D` is the default read-only index.
- `Index3D` is the scalar read-only 3D index.
- `Index2DView` is a zero-copy read-only view over bytes produced by `Index2D::to_bytes`.
- `SimdIndex2D` is available with the `simd` feature and has the same search API.
- `SearchWorkspace` reuses result and traversal buffers.
- `Point2D`, `Point3D`, and `NeighborWorkspace` support nearest-neighbor searches.
- `SortKey2D` selects the public build ordering curve. `Hilbert` is the stable default.
- `SortKey3D` does the same for 3D. `Hilbert` is the stable default.

Search APIs:

- `extent()` returns the total item bounds, or `None` for an empty index.
- `search(bounds)` allocates and returns a `Vec<usize>`.
- `search_into(bounds, &mut results)` reuses a result buffer.
- `search_with(bounds, &mut workspace)` reuses result and traversal buffers.
- `any(bounds)`, `first(bounds)`, and `visit(bounds, visitor)` support early exit.

Nearest-neighbor APIs:

- `neighbors(point, max_results)` returns nearest item indices.
- `neighbors_within(point, max_results, max_distance)` caps the search radius.
- `neighbors_into(...)` and `neighbors_with(...)` reuse buffers.
- `visit_neighbors(point, max_distance, visitor)` visits `(index, distance_squared)` pairs.

## Builder

```rust
use packed_spatial_index::{DEFAULT_NODE_SIZE, Index2DBuilder, Bounds2D, SortKey2D};

let mut builder = Index2DBuilder::new(10_000)
    .node_size(DEFAULT_NODE_SIZE)
    .sort_key(SortKey2D::Hilbert);

builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
builder.add(Bounds2D::new(5.0, 5.0, 6.0, 6.0));
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
# use packed_spatial_index::{Index2DBuilder, Bounds2D};
let mut builder = Index2DBuilder::new(1);
builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
let simd_index = builder.finish_simd()?;
# Ok::<(), packed_spatial_index::BuildError>(())
```

3D uses the same builder/search shape:

```rust
use packed_spatial_index::{Bounds3D, Index3DBuilder, Point3D};

let mut builder = Index3DBuilder::new(2);
builder.add(Bounds3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
builder.add(Bounds3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));

let index = builder.finish()?;
assert_eq!(
    index.search(Bounds3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)),
    vec![0]
);
assert_eq!(index.neighbors(Point3D::new(5.5, 5.5, 5.5), 1), vec![1]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Persistence

`Index2D` can be serialized to a stable little-endian byte format and loaded back
either as an owned index or as a zero-copy view:

```rust
use packed_spatial_index::{Index2D, Index2DBuilder, Index2DView, Bounds2D};

let mut builder = Index2DBuilder::new(1);
builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
let index = builder.finish()?;

let bytes = index.to_bytes();
let owned = Index2D::from_bytes(&bytes)?;
let view = Index2DView::from_bytes(&bytes)?;

assert_eq!(owned.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
assert_eq!(view.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Index3D` and `SimdIndex2D` are not persisted as separate formats yet.

The binary layout is documented in [`FORMAT.md`](FORMAT.md).

## Examples

Runnable examples cover the public paths:

```bash
cargo run --example basic
cargo run --example basic_3d
cargo run --example persistence
cargo run --example knn
cargo run --example reuse_workspace
```

## Features

Both features are enabled by default:

- `parallel`: adaptive rayon-based index builds through `Index2DBuilder::parallel`
  and `Index3DBuilder::parallel`.
- `simd`: SoA index and SIMD search paths through `SimdIndex2D`.

Minimal build:

```bash
cargo build --no-default-features
```

SIMD-only or parallel-only builds:

```bash
cargo build --no-default-features --features simd
cargo build --no-default-features --features parallel
```

## Safety

The public API is safe Rust; users do not need `unsafe` to build, load, search,
or query neighbors.

Internally, the crate keeps `unsafe` limited to narrow performance-sensitive
paths:

- unaligned little-endian reads for validated `Index2DView` byte buffers;
- x86/x86_64 prefetch intrinsics used only by hidden benchmark/experimental paths;
- AVX-512 loads in the `simd` feature, guarded by runtime CPU feature detection.

Loaded buffers are validated before they can be searched, so malformed input is
reported as `LoadError` instead of relying on caller-side invariants.

## Performance Notes

Recent local Criterion run, lower is better. The workload uses 100,000 random
AABBs and 1,000 random search windows.

| Benchmark | FlatGeobuf packed R-tree | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: |
| Full build | 48.03 ms | 2.64 ms serial / 2.06 ms parallel | - |
| Search batch | 568.23 us | 418.81 us | 136.27 us |
| Serialize built tree | 132.73 us | 523.98 us | - |
| Load owned tree | 733.06 us | 548.26 us | - |
| Load zero-copy view | - | 35.59 us | - |

The short version:

- `Index2D` is the general-purpose path;
- `SimdIndex2D` is best for heavier query batches where SIMD work amortizes well;
- `any` is often much faster than collecting full result sets when all you need is existence;
- AVX-512 is not always the fastest path in parallel workloads because CPU frequency behavior matters.
- `flatgeobuf_bench` compares against FlatGeobuf's packed Hilbert R-tree;
- `index_bench` compares build/search paths against `static_aabb2d_index`;
- `persistence_knn_bench` covers persistence, loaded views, and KNN.

Run the focused benchmark suites with:

```bash
cargo bench --bench index_bench --no-default-features --features parallel,simd
cargo bench --bench persistence_knn_bench --no-default-features --features simd
cargo bench --bench flatgeobuf_bench --no-default-features --features parallel,simd
```

## Status

The core API is intentionally small and breaking API cleanup happens before a
`1.0` release.

## License

Licensed under the Apache License, Version 2.0.
