# Changelog

All notable changes to this crate are documented here.

## [Unreleased]


## [0.9.0](https://github.com/Filyus/packed_spatial_index/compare/v0.8.0...v0.9.0) - 2026-06-17

### Persistence
- Add `StreamDirectory` and `into_directory` / `from_directory`
  (`from_directory_with_limits`) on every streaming index (`StreamIndex2D` /
  `StreamIndex3D` and the compact `StreamIndex2DF32` / `StreamIndex3DF32`). Open
  an index once, split off the reader-independent directory, then rebuild a fresh
  index from it with a new reader and no I/O. A handler that uses one reader per
  request (e.g. an edge worker over object storage) caches the directory and pays
  the upper-level reads once instead of on every query. A directory rejects a
  reattach to a mismatched dimension or precision instead of misreading.
- Add `StreamLimits::directory_budget_bytes`: cache more (or all) of the internal
  tree levels at open, so a query descends through fewer round-trips. Trade a
  little memory for latency where memory is plentiful. The cached directory bytes
  are reference-counted, so reattaching across queries is a refcount bump, not a
  copy. **Breaking:** `StreamLimits` gained a field, so a struct literal that set
  every field without `..StreamLimits::default()` now needs it.

## [0.8.0](https://github.com/Filyus/packed_spatial_index/compare/v0.7.0...v0.8.0) - 2026-06-16

### Geometry
- Add triangle primitives: `Triangle2D` / `Triangle3D` (f64) and
  `Triangle2DF32` / `Triangle3DF32` (f32), the sealed `Triangle2` / `Triangle3`
  traits, and `TriangleHit`. Build an index straight from a mesh with
  `Index2D` / `Index3D::from_triangles(..)` (and the new f32 indexes).
- Add `Ray3D::closest_triangle(&[T])` for the nearest ray-triangle hit (f64
  scalar, f32 through a wide SIMD kernel) for mesh-BVH closest-hit queries.

### Indexes
- Add scalar `Index2DF32` / `Index3DF32`: half-memory f32-box indexes (16 / 24
  byte boxes) built with `Index*Builder::finish_f32()` or `from_triangles(..)`.
  They cover `search` / `raycast` / `visit` / `any` / `first`, the
  exact-refining `search_exact` / `any_exact` / `first_exact` / `visit_exact`
  family (filter the conservative f32 hits against your own f64 boxes for no
  false positives), and `serialize()` / `to_bytes()` / `from_bytes()`. No `simd`
  dependency.
- **Breaking:** `f32-storage` no longer enables `simd`. The scalar `Index*F32`
  types build under `f32-storage` alone; the `SimdIndex*F32` frontends now need
  both `f32-storage` and `simd`.

### Persistence
- Add a fixed-width (table-less) payload layout: `serialize().records(stride,
  flat)` and `.triangles(&[T])`, read back zero-copy with `triangles::<T>()` /
  `triangle::<T>(id)`. Files are smaller than the variable-payload table when
  every record is the same size. The variable-payload bytes are byte-identical
  to 0.7.0.
- Add `Serializer2DF32` / `Serializer3DF32` (via `Index*F32::serialize()`) that
  write f32 boxes plus an optional payload, fixed-width records or triangles,
  metadata, and the interleaved node layout.
- Add `StreamIndex2DF32` / `StreamIndex3DF32` (sync and async) to range-query
  and stream payloads from a serialized f32 index at half the box bytes over the
  wire.

### Performance
- Scalar and SIMD f32 range queries round the query once onto the f32 grid (min
  up, max down) and compare f32-vs-f32 with no per-node widen. `Index*F32::search`
  and `SimdIndex*F32::search` now return the identical conservative superset, and
  scalar f32 `search` / `search_exact` are faster. `SimdIndex*F32::search`
  returns slightly fewer near-boundary false positives than 0.7.0.

## [0.7.0](https://github.com/Filyus/packed_spatial_index/compare/v0.6.0...v0.7.0) - 2026-06-14

### API
- Add a `serialize()` builder (`Serializer2D` / `Serializer3D`) that replaces the
  growing family of `to_bytes_*` methods. Chain `.payloads(..)`, `.interleaved()`,
  `.crs(..)`, `.content_type(..)`, `.attribution(..)`, then finish with
  `.to_bytes()` or `.to_bytes_into(..)`.
- Add `FileMetadata` and `read_metadata()` to read file-level metadata (CRS,
  content type, attribution) from a serialized index without loading the tree.

### Safety
- Harden the streaming reader and the payload path against untrusted or remote
  input. Chunk ranges, tree pointers, and payload offsets are bounds-checked as
  they are followed, and broad queries are bounded by per-query cost limits. The
  new `SAFETY.md` documents the memory-safety and untrusted-input guarantees.

### Persistence
- **Breaking:** new on-disk format (`format_version` 2), a chunk container with a
  superblock and a typed chunk directory (TREE / PYLD / META). v1 files no longer
  load. The container is forward-compatible: readers skip unknown optional chunks
  and reject unknown critical ones, and descriptors can grow without breaking
  older readers.
- Add a streaming reader. `StreamIndex2D` / `StreamIndex3D` query a serialized
  index over a `RangeReader` (sync) or `AsyncRangeReader` (async) without loading
  the whole file, with coalesced per-level range reads. An optional interleaved
  layout fetches each level in a single read.
- Add an optional per-item payload (the `PYLD` chunk): attach one opaque blob per
  item to make a file self-contained. Blobs are stored in leaf (Hilbert) order so
  a spatial query reads them in coalesced runs, and they are served by both the
  zero-copy views and the streaming reader, in 2D and 3D.


## [0.6.0](https://github.com/Filyus/packed_spatial_index/compare/v0.5.1...v0.6.0) - 2026-06-14

### Search
- Add `search_iter`, a lazy iterator over the items intersecting a query box, on
  `Index2D` and `Index3D`. It descends the tree on demand, so consuming only a
  prefix (`.next()`, `.take(k)`, `.find(..)`) stops the traversal early and never
  allocates a result `Vec`. Reach for it to compose with iterator adapters or to
  bail out partway, where `search` (a whole owned `Vec`) and `visit` (a
  push-based callback) are awkward.


## [0.5.1](https://github.com/Filyus/packed_spatial_index/compare/v0.5.0...v0.5.1) - 2026-06-13

### Documentation
- Restructure the README into a concise reference and move the long-form guide,
  persistence, and performance docs into `docs/`. Link every query method and
  type to docs.rs, add examples to `search` / `any` / `first`, document querying
  large or on-disk indexes via memory mapping, and add a clickable queries
  overview to the crate landing page.


## [0.5.0](https://github.com/Filyus/packed_spatial_index/compare/v0.4.3...v0.5.0) - 2026-06-13

### 2D
- Reject 2D builds with more than `u32::MAX` items (returns
  `BuildError::TreeTooLarge`) instead of silently truncating the `u32` item
  indices and producing a corrupt index.

### Search
- Add spatial joins. `join`/`join_with` report every intersecting pair of items
  between two indexes, and `self_join`/`self_join_with` report every unordered
  pair of distinct intersecting items within one index. A single synchronized
  descent over both trees replaces one search per item (about 7x faster than a
  search loop for 1M-by-1M joins, about 19x for 1M self-joins). Available on
  `Index2D`, `Index3D`, the SIMD indexes, and all zero-copy f64 views.
- Add ray-segment queries. New `Ray2D` and `Ray3D` types, plus `raycast` /
  `raycast_into` / `raycast_with` (all hits), `raycast_closest` /
  `raycast_closest_with` (nearest box the segment enters), and `visit_raycast`
  (visit hits in nondecreasing entry-`t` order with early exit). Available on
  every f64 index and zero-copy view. The SIMD indexes evaluate the slab test
  four (`wide`) or eight (AVX-512) children at a time, with a masked path that
  keeps axis-parallel rays exact on box faces.

### Nearest Neighbors
- Add box-query nearest-neighbor search: `neighbors_of_box`,
  `neighbors_of_box_within`, `neighbors_of_box_into`, `neighbors_of_box_with`,
  and `visit_neighbors_of_box`. Distance is the box-to-box gap, so items
  overlapping or touching the query box rank first at distance zero. Available
  on all f64 indexes and views.

### Performance
- Extend the covered-range fast path to the owned SIMD `visit` traversals (2D
  and 3D), matching the search paths and byte-view visitors.
- Prefetch the next stacked node in the default scalar range search (`Index2D`
  and `Index3D`), a consistent ~3-5% range-query speedup.


## [0.4.3](https://github.com/Filyus/packed_spatial_index/compare/v0.4.2...v0.4.3) - 2026-06-09

### Performance
- Speed up covered range queries by collecting fully contained subtrees directly
  instead of testing every item.
- Apply the covered-range fast path across scalar indexes, SIMD indexes,
  zero-copy SIMD views, and `f32-storage` variants.
- Add full-extent shortcuts for 2D views and SIMD scalar search paths.
- Keep conservative `f32-storage` searches semantically unchanged; exact f32
  searches still re-check candidates.

### Documentation
- Add large-window range search benchmark results to the README.


## [0.4.2](https://github.com/Filyus/packed_spatial_index/compare/v0.4.1...v0.4.2) - 2026-06-08

### SIMD
- Update SIMD comparisons for `wide` 1.5.

### Documentation
- Add README notes for AI usage and prior art.
- Clarify the live WASM demo link.

### WASM Demo
- Publish the interactive demo through GitHub Pages.

## [0.4.1](https://github.com/Filyus/packed_spatial_index/compare/v0.4.0...v0.4.1) - 2026-06-05

### API
- Add opt-in `f32-storage` SIMD indexes for compact coordinate storage.
- Add exact range and KNN callbacks for `f32-storage` indexes using
  caller-owned `f64` boxes.

### Binary Format
- Document the packed spatial index binary format.
- Add distinct f32 box layout flags for `f32-storage` indexes.

### WASM
- Add the interactive WASM demo for 2D and 3D searches.
- Add 3D depth slicing, depth coloring, and an interactive depth legend.
- Tighten demo controls, query overlays, status bar, and wrapper helpers.

### Benchmarks
- Add f32-vs-f64 storage benchmarks for range queries and KNN.

### Documentation
- Document f32 storage trade-offs, exact query APIs, and benchmark guidance.

### Examples
- Add an f32 exact-query example.

### Tests
- Add f32 storage coverage for range search, exact range search, KNN,
  persistence, and views.
- Add proptest search and persistence robustness checks
- Rustfmt proptest files

## [0.4.0](https://github.com/Filyus/packed_spatial_index/compare/v0.3.3...v0.4.0) - 2026-06-03

### API
- Return `BuildError::TreeTooLarge` instead of panicking when a requested tree
  layout cannot fit in memory.

### Benchmarks
- Move internal performance tools out of the published examples and into a
  local benchmark tools package.


## [0.3.3](https://github.com/Filyus/packed_spatial_index/compare/v0.3.2...v0.3.3) - 2026-06-03

### Geometry
- Add point box constructors
- Share box accumulator helpers

### Documentation
- Add docs.rs feature badges and verify the docs.rs build
- Clarify query API guidance

### Lint
- Require SAFETY comments on all unsafe blocks

## [0.3.2](https://github.com/Filyus/packed_spatial_index/compare/v0.3.1...v0.3.2) - 2026-06-02

### SIMD

- Add zero-copy SIMD views

### Documentation

- Clarify release-plz release flow
- Document environment approval setup
- Fold tag fallback into first release
- Reorder release guide sections

### Build, CI, and Packaging

- Add safe release-plz draft workflow
- Make release-plz dry run preview only
- Run semver checks in release-plz workflow
- Simplify release workflows
- Clarify workflow names
- Use action-oriented workflow names
- Rename prepare workflow file
- Use lowercase manual run names
