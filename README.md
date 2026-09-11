# Packed Spatial Index

[![crates.io](https://img.shields.io/crates/v/packed_spatial_index.svg)](https://crates.io/crates/packed_spatial_index)
[![docs.rs](https://docs.rs/packed_spatial_index/badge.svg)](https://docs.rs/packed_spatial_index)
[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/packed_spatial_index.svg)](https://crates.io/crates/packed_spatial_index)
[![License](https://img.shields.io/crates/l/packed_spatial_index.svg)](LICENSE)

A fast, packed **static spatial index** for 2D and 3D axis-aligned bounding boxes
(AABBs). Pack the boxes into a Hilbert R-tree once, then run millions of queries
of every kind:

- **range / intersection** search
- **nearest neighbors** (kNN) from a point or a box, under Euclidean or any custom
  metric — including **great-circle distance** for lon/lat data
- **ray casts** (all hits or the closest)
- **spatial joins** between two indexes — intersecting or within a distance
  (ε-join), with the anti-join and the connected components of the distance graph
- **region / culling / picking** — 2D triangle / convex-polygon and 3D
  view-frustum queries that prune to the true shape: **~1.5–7× fewer hits and
  ~2–14× faster** than the bounding-box workaround (synthetic 200k-box bench).
  A frustum narrowed to the pixels around the cursor turns the same query into
  3D picking or rubber-band selection

Queries run on **runtime-dispatched SIMD** — the widest kernel your CPU offers is
chosen at load time (`AVX-512 → AVX2 → SSE2`), no special build flags. The same
bytes load back as **zero-copy**, mmap-friendly views; a file can carry an
optional per-item **payload** and file-level **metadata**; and a **streaming
reader** answers a windowed query over a 100 MB index on object storage in a
handful of range reads, without loading the whole file.

[Live WASM demo](https://filyus.github.io/packed_spatial_index/)

## Quick start

```toml
[dependencies]
packed_spatial_index = "0.30"
```

Requires Rust 1.89 or newer.

```rust
use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};

// Your data stays yours. The index only ever hands back positions into it.
let parks = ["Riverside", "Hilltop"];

// Build once. One bounding box per item, in the same order as your data.
let mut builder = Index2DBuilder::new(parks.len());
builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0)); // Riverside: min_x, min_y, max_x, max_y
builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0)); // Hilltop
let index = builder.finish().unwrap();

// "Which parks fall inside this window?"
let visible = index.search(Box2D::new(0.0, 0.0, 2.0, 2.0));
assert_eq!(visible, vec![0]);               // position 0 ...
assert_eq!(parks[visible[0]], "Riverside"); // ... is the box you added first

// "Which park is nearest to me?"
let me = Point2D::new(5.5, 5.5);
let nearest = index.neighbors(me, 1);
assert_eq!(parks[nearest[0]], "Hilltop");
```

That is the whole shape of it: describe each item with one box, build once, then
ask. Every answer is a position in the order you added things, so you look the
real record up yourself. That is also why the index stays small no matter what
your records weigh.

Ray casts, nearest pairs, joins and the region queries all follow this pattern.
The [API map](docs/api.md) lists every family and type on one page; the
[guide](docs/guide.md) starts from what you want rather than from a method name.

## Where to go next

- **[API map](docs/api.md)** — every query family and every type, one page.
- **[Guide](docs/guide.md)** — an *I need … → use …* table, builder configuration, recipes, the naming rule the method names follow.
- **[When to use it](docs/when-to-use.md)** — where this fits, and where a spatial database fits better.
- **[Persistence](docs/persistence.md)** — serialize, load, zero-copy views, mmap, payloads and metadata, streaming over a `RangeReader`.
- **[Performance](docs/performance.md)** — benchmarks against `static_aabb2d_index`, FlatGeobuf and the `bvh` crate, plus the build flags that matter.
- **[Internals](docs/internals/)** — SIMD kernels, two-queue kNN, traversal prefetch.
- **[Binary format](FORMAT.md)** — the `PSINDEX` on-disk layout.
- **[API reference](https://docs.rs/packed_spatial_index)** — per-method docs.

## When to use it

Use this crate when your geometry is static or rebuilt in batches, you can key
results by insertion-order index into your own payload array, and you want a
compact in-memory (or mmap'd) index with reusable buffers for high query
throughput. It is **not** a dynamic R-tree — there are no insert/delete
operations after `finish()`.

It also serializes to a single file you can put on object storage and
range-query from the edge or a browser, with no backend. For the longer answer,
including where a spatial database wins, see
[When to use it](docs/when-to-use.md).

## Features

| Feature | Pulls in | Adds |
| --- | --- | --- |
| `parallel` *(default)* | `rayon` | adaptive parallel index builds |
| `simd` *(default)* | `wide` | SoA indexes + SIMD search / raycast (AVX2 / AVX-512) |
| `f32-storage` | — | compact f32-box indexes (scalar `Index2DF32` / `Index3DF32`; the `SimdIndex*F32` variants also need `simd`) |
| `stream` | — | query a serialized index over a `RangeReader` (local file or remote object) without loading it whole |
| `async` | `futures-util` *(implies `stream`)* | query over an `AsyncRangeReader` (browser / edge worker, HTTP range or object storage) |
| `bench-internals` | — | hidden support API for this crate's benchmarks |

Serialization, metadata, and the scalar indexes are always available — no feature
required.

```bash
cargo build --no-default-features                      # minimal: scalar + serialize + metadata
cargo build --no-default-features --features simd      # SIMD only
```

## Limitations

- Static: rebuild when the dataset changes; no insert/delete.
- Results are item indices, not stored payloads. Result order is unspecified
  unless you asked for one with `search_ordered`, which costs a heap: ordering an
  entire result set is slower than `search` plus a sort, and pays off when a
  budget or a cutoff lets the traversal stop early.
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

## Development

AI assistance is part of building this project. The architecture is human-directed
and the generated output is reviewed carefully before it ships, also a broad test
suite catches mistakes.

## License

Licensed under the Apache License, Version 2.0.
