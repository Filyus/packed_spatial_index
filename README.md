# packed_spatial_index

`packed_spatial_index` is a packed static spatial index for 2D axis-aligned
bounding boxes.

It is built for read-heavy workloads where the full set of boxes is known up
front: build once, then run many window/intersection searches. The default
`Index` uses a packed Hilbert R-tree layout. With the `simd` feature,
`SimdIndex` stores boxes in structure-of-arrays form and uses SIMD intersection
checks.

```rust
use packed_spatial_index::{IndexBuilder, Rect};

let mut builder = IndexBuilder::new(2);
builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));

let index = builder.finish()?;
let hits = index.search(Rect::new(0.0, 0.0, 2.0, 2.0));

assert_eq!(hits, vec![0]);
# Ok::<(), packed_spatial_index::BuildError>(())
```

## When To Use It

Use this crate when:

- your geometry is static or rebuilt in batches;
- search results can be returned as insertion-order indices into your own payload array;
- you want a compact in-memory index with reusable buffers for repeated searches;
- batch search throughput matters.

It is not a dynamic R-tree: there are no insert/delete operations after build.

## Main Types

- `Rect` is the public AABB type.
- `IndexBuilder` builds either `Index` or, with `simd`, `SimdIndex`.
- `Index` is the default read-only index.
- `IndexView` is a zero-copy read-only view over bytes produced by `Index::to_bytes`.
- `SimdIndex` is available with the `simd` feature and has the same search API.
- `SearchWorkspace` reuses result and traversal buffers.
- `Point` and `NeighborWorkspace` support nearest-neighbor searches.
- `SortKey` selects the public build ordering curve: `Hilbert` or `Morton`.

Search APIs:

- `search(rect)` allocates and returns a `Vec<usize>`.
- `search_into(rect, &mut results)` reuses a result buffer.
- `search_with(rect, &mut workspace)` reuses result and traversal buffers.
- `any(rect)`, `first(rect)`, and `visit(rect, visitor)` support early exit.

Nearest-neighbor APIs:

- `neighbors(point, max_results)` returns nearest item indices.
- `neighbors_within(point, max_results, max_distance)` caps the search radius.
- `neighbors_into(...)` and `neighbors_with(...)` reuse buffers.
- `visit_neighbors(point, max_distance, visitor)` visits `(index, distance_squared)` pairs.

## Builder

```rust
use packed_spatial_index::{IndexBuilder, Rect, SortKey};

let mut builder = IndexBuilder::new(10_000)
    .node_size(16)
    .sort_key(SortKey::Hilbert);

builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
builder.add_bounds(5.0, 5.0, 6.0, 6.0);
```

With `parallel` enabled:

```rust
# use packed_spatial_index::IndexBuilder;
let builder = IndexBuilder::new(100_000)
    .parallel(true)
    .parallel_min_items(50_000);
```

With `simd` enabled:

```rust
# use packed_spatial_index::{IndexBuilder, Rect};
let mut builder = IndexBuilder::new(1);
builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
let simd_index = builder.finish_simd()?;
# Ok::<(), packed_spatial_index::BuildError>(())
```

## Persistence

`Index` can be serialized to a stable little-endian byte format and loaded back
either as an owned index or as a zero-copy view:

```rust
use packed_spatial_index::{Index, IndexBuilder, IndexView, Rect};

let mut builder = IndexBuilder::new(1);
builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
let index = builder.finish()?;

let bytes = index.to_bytes();
let owned = Index::from_bytes(&bytes)?;
let view = IndexView::from_bytes(&bytes)?;

assert_eq!(owned.search(Rect::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
assert_eq!(view.search(Rect::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`SimdIndex` is not persisted as a separate SoA format yet.

## Features

Both features are enabled by default:

- `parallel`: adaptive rayon-based index builds through `IndexBuilder::parallel`.
- `simd`: SoA index and SIMD search paths through `SimdIndex`.

Minimal build:

```bash
cargo build --no-default-features
```

SIMD-only or parallel-only builds:

```bash
cargo build --no-default-features --features simd
cargo build --no-default-features --features parallel
```

## Migration From 0.1

- `PackedIndexBuilder::new(count, node_size)` -> `IndexBuilder::new(count).node_size(node_size)`
- `PackedIndexBuilder::build()` / `try_build()` -> `IndexBuilder::finish()`
- `PackedIndex` -> `Index`
- `PackedSoaIndexBuilder` -> `IndexBuilder::finish_simd()`
- `PackedSoaIndex` -> `SimdIndex`
- `PackedIndexBuildError` -> `BuildError`
- `query*` -> `search*`
- `query_any`, `query_first`, `visit_query` -> `any`, `first`, `visit`
- encoder-specific `SortKey` variants moved to the hidden experimental API used by benches

## Performance Notes

The short version from the local benchmarks:

- build is faster than `static_aabb2d_index` for the measured random AABB workloads;
- `Index` is the general-purpose path;
- `SimdIndex` is best for heavier query batches where SIMD work amortizes well;
- `any` is often much faster than collecting full result sets when all you need is existence;
- AVX-512 is not always the fastest path in parallel workloads because CPU frequency behavior matters.

See `REPORT.md` for the detailed research notes and benchmark tables.

## Status

This crate is still marked `publish = false` while publishing metadata and the remaining API surface settle.
