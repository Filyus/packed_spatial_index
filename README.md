# packed_spatial_index

[![crates.io](https://img.shields.io/crates/v/packed_spatial_index.svg)](https://crates.io/crates/packed_spatial_index)
[![docs.rs](https://docs.rs/packed_spatial_index/badge.svg)](https://docs.rs/packed_spatial_index)
[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/packed_spatial_index.svg)](https://crates.io/crates/packed_spatial_index)
[![License](https://img.shields.io/crates/l/packed_spatial_index.svg)](LICENSE)

A packed **static spatial index** for 2D and 3D axis-aligned bounding boxes
(AABBs), in the style of [flatbush](https://github.com/mourner/flatbush) /
[`static_aabb2d_index`](https://crates.io/crates/static_aabb2d_index). Pack the
boxes into a Hilbert R-tree once, then run many queries over it with SIMD:

- **range / intersection** search
- **nearest neighbors** (kNN) from a point or a box
- **ray casts** (all hits or the closest)
- **spatial joins** between two indexes

Builds and SIMD searches run **faster** than comparable Rust indexes
([benchmarks](docs/performance.md)), and the same bytes load back as
**zero-copy**, mmap-friendly views. A file can also carry an optional
per-item **payload** and file-level **metadata**, and a **streaming reader** can
query a large or remote index without loading the whole file.

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
packed_spatial_index = "0.6"
```

## When to use it

Use this crate when your geometry is static or rebuilt in batches, you can key
results by insertion-order index into your own payload array, and you want a
compact in-memory (or mmap'd) index with reusable buffers for high query
throughput. It is **not** a dynamic R-tree — there are no insert/delete
operations after `finish()`.

## Performance

Built for throughput on static geometry: fast builds, SIMD range / kNN / raycast
(AVX2 / AVX-512), and reusable query buffers for tight loops. The
[Performance](docs/performance.md) page has the full benchmarks against
`static_aabb2d_index`, FlatGeobuf and the `bvh` crate, showing where it leads and
where it doesn't.

## Queries at a glance

Every query exists on `Index2D` / `Index3D`, the `simd`-feature `SimdIndex2D` /
`SimdIndex3D`, and the zero-copy views. Range/ray results are item indices in
insertion order; result order is unspecified. See
[docs.rs](https://docs.rs/packed_spatial_index) for full per-method docs.

| Query | Methods |
| --- | --- |
| Range / intersection | [`search`][search], [`search_iter`][search_iter], [`search_into`][search_into], [`search_with`][search_with], [`any`][any], [`first`][first], [`visit`][visit] |
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

- **Geometry**: [`Box2D`][Box2D], [`Box3D`][Box3D] (inclusive `overlaps` /
  `contains` / `contains_point` / `from_point`), [`Point2D`][Point2D],
  [`Point3D`][Point3D], [`Ray2D`][Ray2D], [`Ray3D`][Ray3D].
- **Builders**: [`Index2DBuilder`][Index2DBuilder],
  [`Index3DBuilder`][Index3DBuilder] — [`finish`][finish] (scalar),
  [`finish_simd`][finish_simd] (SoA + SIMD),
  [`finish_simd_f32`][finish_simd_f32] (compact f32 boxes).
- **Indexes**: [`Index2D`][Index2D] / [`Index3D`][Index3D] (scalar),
  [`SimdIndex2D`][SimdIndex2D] / [`SimdIndex3D`][SimdIndex3D] (SIMD),
  [`SimdIndex2DF32`][SimdIndex2DF32] / [`SimdIndex3DF32`][SimdIndex3DF32]
  (half-memory f32 boxes).
- **Views**: zero-copy [`Index2DView`][Index2DView] /
  [`Index3DView`][Index3DView] (and SIMD / f32 view variants) over serialized
  bytes.
- **Workspaces**: [`SearchWorkspace`][SearchWorkspace] /
  [`NeighborWorkspace`][NeighborWorkspace] reuse buffers in loops.
- **Sorting / errors**: [`SortKey2D`][SortKey2D] / [`SortKey3D`][SortKey3D]
  (default `Hilbert`), [`BoundsError`][BoundsError], [`BuildError`][BuildError],
  [`LoadError`][LoadError].

## Serialization & metadata

`to_bytes` / `from_bytes` round-trip an index; the `serialize()` builder adds the
optional pieces — one opaque payload blob per item and descriptive metadata
(coordinate reference system, payload content type, attribution):

```rust
# use packed_spatial_index::{Box2D, Index2DBuilder, read_metadata};
# let mut b = Index2DBuilder::new(1);
# b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
# let index = b.finish()?;
let bytes = index
    .serialize()
    .crs("EPSG:4326")
    .payloads(&[b"feature-0".as_slice()])
    .to_bytes()?;

// Read the metadata back without loading the index.
assert_eq!(read_metadata(&bytes)?.crs.as_deref(), Some("EPSG:4326"));
# Ok::<(), Box<dyn std::error::Error>>(())
```

The metadata is opaque (the crate stores the strings you give it, verbatim).
Pair query results with their payloads via the zero-copy views or the streaming
reader — see [Persistence](docs/persistence.md) and the [binary format](FORMAT.md).

## Features

| Feature | Pulls in | Adds |
| --- | --- | --- |
| `parallel` *(default)* | `rayon` | adaptive parallel index builds |
| `simd` *(default)* | `wide` | SoA indexes + SIMD search / raycast (AVX2 / AVX-512) |
| `f32-storage` | *(implies `simd`)* | compact f32-storage SIMD indexes |
| `stream` | — | query a serialized index over a `RangeReader` (local file or remote object) without loading it whole |
| `async` | `futures-util` *(implies `stream`)* | query over an `AsyncRangeReader` (browser / edge worker, HTTP range or object storage) |
| `bench-internals` | — | hidden support API for this crate's benchmarks |

Serialization, metadata, and the scalar indexes are always available — no feature
required.

```bash
cargo build --no-default-features                      # minimal: scalar + serialize + metadata
cargo build --no-default-features --features simd      # SIMD only
```

## Documentation

- **[Guide](docs/guide.md)** — recipes, choosing a query method, builder configuration, examples, WASM demo.
- **[Persistence](docs/persistence.md)** — serialize / load / zero-copy views, querying large or on-disk indexes via mmap, and streaming queries over a `RangeReader` (local file or remote object).
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

The public API is safe Rust; `unsafe` is confined to narrow, audited paths
(validated unaligned reads, `repr(C)` bulk copies, gated x86-64 SIMD). Serialized
input is treated as untrusted: the in-memory loaders validate the whole buffer
before use, and the streaming reader validates pointers and payload offsets as it
follows them, with per-query cost limits to bound broad queries. See
[SAFETY.md](SAFETY.md) for the memory-safety and untrusted-input hardening
details.

## Status

Pre-`1.0`: the API and on-disk format may still change between minor releases.
The crate is covered by unit, property and fuzz tests across the feature matrix,
but it has not yet been proven in production. Validate it for your workload
before you depend on it.

## Feedback

Built something with it? I'd love to hear about it! Start a
[discussion](https://github.com/Filyus/packed_spatial_index/discussions) with your
use case, your numbers or any rough edges, and file an
[issue](https://github.com/Filyus/packed_spatial_index/issues) for bugs.
Real-world reports are what push it toward `1.0`.

## AI usage note

AI assistance is part of my development process for this project. I guide the
architecture, review generated output carefully, and take responsibility for the
crate as published.

## License

Licensed under the Apache License, Version 2.0.

<!-- docs.rs method links -->
[search]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search
[search_iter]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_iter
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
[Box2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Box2D.html
[Box3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Box3D.html
[Point2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Point2D.html
[Point3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Point3D.html
[Ray2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Ray2D.html
[Ray3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Ray3D.html
[Index2DBuilder]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html
[Index3DBuilder]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index3DBuilder.html
[Index2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html
[Index3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index3D.html
[SimdIndex2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2D.html
[SimdIndex3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex3D.html
[SimdIndex2DF32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2DF32.html
[SimdIndex3DF32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex3DF32.html
[Index2DView]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DView.html
[Index3DView]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index3DView.html
[SearchWorkspace]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SearchWorkspace.html
[NeighborWorkspace]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.NeighborWorkspace.html
[finish]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html#method.finish
[finish_simd]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html#method.finish_simd
[finish_simd_f32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html#method.finish_simd_f32
[SortKey2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.SortKey2D.html
[SortKey3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.SortKey3D.html
[BoundsError]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.BoundsError.html
[BuildError]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.BuildError.html
[LoadError]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.LoadError.html
