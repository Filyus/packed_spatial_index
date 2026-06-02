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
- Persistence is defined for canonical `Index2D` and `Index3D` formats.
  `SimdIndex2D` can be rebuilt from source boxes but does not have a stable SoA
  file format yet.
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
- `Index2DView` and `Index3DView` are zero-copy read-only views over bytes
  produced by `Index2D::to_bytes` and `Index3D::to_bytes`.
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

`Index2D` and `Index3D` can be serialized to stable little-endian byte formats
and loaded back either as owned indexes or as zero-copy views:

```rust
use packed_spatial_index::{Index2D, Index2DBuilder, Index2DView, Bounds2D};

let mut builder = Index2DBuilder::new(1);
builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
let index = builder.finish()?;

let bytes = index.to_bytes();
let mut reusable = Vec::new();
index.to_bytes_into(&mut reusable);
assert_eq!(reusable, bytes);

let owned = Index2D::from_bytes(&bytes)?;
let view = Index2DView::from_bytes(&bytes)?;

assert_eq!(owned.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
assert_eq!(view.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

3D persistence uses the same header and sections, with a dimension flag and
six `f64` coordinates per stored bounds. `SimdIndex2D` is not persisted as a
separate SoA format yet.

The binary layout is documented in [`FORMAT.md`](FORMAT.md).

## Examples

Runnable examples cover the public paths:

```bash
cargo run --example basic_2d
cargo run --example basic_3d
cargo run --example persistence_2d
cargo run --example knn_2d
cargo run --example reuse_workspace_2d
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

- unaligned little-endian reads for validated `Index2DView` and `Index3DView`
  byte buffers;
- bulk byte copies for `repr(C)` bounds and index sections when serializing on
  compatible little-endian targets;
- x86/x86_64 prefetch intrinsics used only by hidden benchmark/experimental paths;
- AVX-512 loads in the `simd` feature, guarded by runtime CPU feature detection.

Loaded buffers are validated before they can be searched, so malformed input is
reported as `LoadError` instead of relying on caller-side invariants.

## Performance Notes

Recent local Criterion runs, lower is better. The workload uses 100,000 random
AABBs and 1,000 random search windows; build and search competitors are measured
in the same benchmark suite on the same generated inputs.

| Benchmark | FlatGeobuf | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: | ---: |
| Full build | 70.18 ms | 8.95 ms | 3.18 ms serial / 2.20 ms parallel | - |
| Search batch | 555.83 us | 341.56 us | 416.15 us | 128.64 us |
| Serialize built tree (fresh buffer) | - | - | 413.09 us | - |
| Serialize built tree (reused buffer) | 131.93 us | - | 68.35 us | - |
| Load owned tree | 740.23 us | - | 620.03 us | - |
| Load zero-copy view | - | - | 37.01 us | - |

Scalar `Index2D` search versus `static_aabb2d_index` is dataset-sensitive.
Two local search runs with the same item/query counts but different generated
inputs showed opposite scalar ordering:

| Search batch | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: |
| `flatgeobuf2d_bench`, seed `0xF6B` | 341.56 us | 416.15 us | 128.64 us |
| `index2d_bench`, seed `0xB0B` | 643.85 us | 311.72 us | 126.21 us |

Recent local 2D-vs-3D Criterion run. Lower latency is better. The `3D speed`
column is `2D latency / 3D latency`, so values above `1.00x` mean 3D is faster.
The build workload uses 100,000 boxes with `node_size = 16`; search and KNN use
1,000 query windows or points.

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
| Serialize built tree | 556.79 us | 909.29 us | 0.61x |
| Load owned tree | 585.02 us | 875.21 us | 0.67x |
| Load zero-copy view | 35.636 us | 35.743 us | 1.00x |

The short version:

- `Index2D` is the general-purpose path;
- `SimdIndex2D` is best for heavier query batches where SIMD work amortizes well;
- scalar `Index2D` search versus `static_aabb2d_index` depends on the generated
  data and query distribution, while `Index2D` build is faster in these runs;
- `Index3D` build and KNN are still slower than `Index2D`, but uniform 3D search
  can be faster when Z meaningfully prunes the tree;
- `any` is often much faster than collecting full result sets when all you need is existence;
- AVX-512 is not always the fastest path in parallel workloads because CPU frequency behavior matters.
- `flatgeobuf2d_bench` compares against FlatGeobuf's packed Hilbert R-tree;
- `index2d_bench` compares build/search paths against `static_aabb2d_index`;
- `index3d_bench` covers scalar 3D build/search/KNN, persistence, loaded views,
  node sizes, and hidden Morton baseline;
- `persistence_knn2d_bench` covers persistence, loaded views, and KNN.

Run the focused benchmark suites with:

```bash
cargo bench --bench index2d_bench --no-default-features --features parallel,simd
cargo bench --bench index3d_bench --no-default-features
cargo bench --bench persistence_knn2d_bench --no-default-features --features simd
cargo bench --bench flatgeobuf2d_bench --no-default-features --features parallel,simd
```

## Status

The core API is intentionally small and breaking API cleanup happens before a
`1.0` release.

## License

Licensed under the Apache License, Version 2.0.
