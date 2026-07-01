# Changelog

All notable changes to this crate are documented here.

## [Unreleased]

### Documentation

- Clarified streaming-query documentation: sync streaming supports range and
  region queries, while async streaming currently mirrors box queries; kNN and
  raycast remain in-memory/view use cases.

## [0.21.1](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.21.0...psi-v0.21.1) - 2026-07-01

### 3D

- `Frustum3D::bounding_box()` now uses a scale-invariant degeneracy test.
  Previously an absolute determinant epsilon scaled with the product of the
  three plane-normal magnitudes, so a valid frustum whose (non-normalized)
  planes were uniformly scaled down could be wrongly reported degenerate
  (`None`). The check now compares the normalized triple product of the
  normals.

## [0.21.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.20.0...psi-v0.21.0) - 2026-07-01

### 3D

- Added `Frustum3D::bounding_box()`, an axis-aligned bounding box computed
  from the frustum's eight corner points. Lets downstream code (e.g. the geo
  companion crate) narrow a frustum query to a coarse box before a streaming
  or non-generic search. Returns `None` for a degenerate/near-parallel plane
  arrangement rather than a silently-wrong box.

## [0.20.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.19.0...psi-v0.20.0) - 2026-06-30

### Search

- Added region queries to the streaming indexes. `StreamIndex2D` / `StreamIndex3D`
  (and the `f32` variants) gained `search_region` / `visit_region` /
  `search_payloads_region` / `visit_payloads_region`, taking any `Overlaps2D` /
  `Overlaps3D` shape (polygon, triangle, frustum, …) instead of only a box.
  Subtrees outside the query shape are pruned during the streamed descent, so a
  region fetches only the leaves it overlaps — less data than its bounding box,
  the key win for out-of-core / remote region queries. (Pruning can fragment the
  coalesced runs, so the range-request count is shape-dependent; the bytes always
  shrink.)

## [0.19.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.18.1...psi-v0.19.0) - 2026-06-28

### API
- **BREAKING:** overlap region queries now use the same short method family as
  box range queries. Use `index.search(&triangle)`, `index.search(&polygon)`,
  or `index.search(&frustum)` plus the matching `search_into`, `search_with`,
  `search_iter`, `any`, `first`, and `visit` forms. The older shape-specific
  convenience methods such as `search_triangle`, `search_polygon`,
  `search_frustum`, and their `*_into` / `any_*` / `visit_*` variants have been
  removed.
- Added `Overlaps2D` and `Overlaps3D` as the shared predicate traits behind
  borrowed region queries. `Box2D` / `Box3D`, `Triangle2D`,
  `ConvexPolygon2D`, and `Frustum3D` implement these traits, including
  `contains_box` for contained-subtree pruning.

### Search
- **BREAKING:** `Index2D::search_iter` and `Index3D::search_iter` now dispatch
  through the same query API as `search` / `search_into` / `any` / `first` /
  `visit`: box queries return the lightweight box iterator, while borrowed
  geometry queries such as `&Triangle2D`, `&ConvexPolygon2D`, and `&Frustum3D`
  return a region iterator with contained-subtree fast paths. Existing call
  sites that simply iterate the result keep the same `index.search_iter(query)`
  shape; code that names the concrete iterator type may need to use
  `impl Iterator` or the new region iterator type.
- Lazy region iteration is much faster for broad shape queries because fully
  contained subtrees skip per-item geometry predicates; box iteration keeps a
  separate lightweight traversal path.

## [0.18.1](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.18.0...psi-v0.18.1) - 2026-06-20

### Raycast
- All-hits raycast on the scalar `Index2D` / `Index3D` is ~5–12% faster on heavy
  traversal: it prefetches the next tree node while the current one is hit-tested
  (a free cache hint, neutral when little is visited). No API change.

## [0.18.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.17.0...psi-v0.18.0) - 2026-06-20

### Geometry
- **BREAKING:** `Frustum3D::from_view_projection` now takes a second argument, the
  new `ClipSpaceZ` enum, naming the NDC depth range your projection targets. The
  old call assumed the OpenGL/WebGL `[-1, 1]` clip cube and silently produced a
  wrong near plane for D3D12 / Vulkan / Metal / WebGPU, which clip `z` to `[0, 1]`
  (only the near plane differs). Pass `ClipSpaceZ::ZeroToOne` (the modern
  majority, also the `Default`) or `ClipSpaceZ::NegOneToOne` (OpenGL/WebGL); the
  convention is not recoverable from the matrix, so there is no silent default.
  Migration: existing OpenGL callers add `ClipSpaceZ::NegOneToOne`.

### Nearest Neighbors
- kNN on the compact f32 indexes (`Index2DF32` / `Index3DF32`) and `SimdIndex2D`
  is ~7–11% faster: the SIMD and f32 frontends now use the same two-queue
  distance-browsing collect the scalar `Index2D` already used, so it is the one
  kNN collect kernel everywhere. No API change. The technique is written up in
  [docs/knn.md](docs/internals/knn.md).

## [0.17.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.16.0...psi-v0.17.0) - 2026-06-19

### Nearest Neighbors
- **Custom-metric nearest-neighbor queries.** `neighbors_metric` /
  `neighbors_metric_into` / `visit_neighbors_metric` on `Index2D`, `Index3D` and
  their zero-copy views (`Index2DView` / `Index3DView`) take a closure
  `|box| -> f64` returning the distance from your query to a box, so kNN can run
  under any admissible metric — Euclidean, Manhattan, Chebyshev, weighted axes, or
  **great-circle distance** for lon/lat data. A `haversine_distance_2d(query, box,
  earth_radius)` helper and an `EARTH_RADIUS_M` constant are provided for the
  geographic case. The default squared-Euclidean `neighbors` path is unchanged.

## [0.16.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.15.0...psi-v0.16.0) - 2026-06-19

### SIMD
- Add a runtime **AVX2 tier** to the SIMD search / visit / all-hits raycast
  kernels, so a generic binary on an AVX2-but-not-AVX-512 CPU (the large
  Haswell–Ice Lake / Zen 1–3 installed base) no longer falls back to SSE2 width.
  AVX2 has no `VPCOMPRESSQ`, so result collection uses an AVX2 *left-pack*
  (`VPERMD` over a 16-entry shuffle LUT) that emulates the compress. Range search
  runs ~1.3–1.65× and all-hits raycast ~1.3–1.6× over the SSE2 `wide` fallback,
  across `SimdIndex2D` / `SimdIndex3D` and the compact `SimdIndex2DF32` /
  `SimdIndex3DF32`. The kernels now dispatch `AVX-512 → AVX2 → SSE2` at runtime.
  No API change. See [docs/simd.md](docs/internals/simd.md).
- Collect the **AVX-512 all-hits raycast** results with `VPCOMPRESSQ` instead of a
  scalar loop (it was the one collection path still left scalar): a dense 1M-box
  ray drops ~29.5 µs to ~17.1 µs (~1.73×).

## [0.15.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.14.0...psi-v0.15.0) - 2026-06-19

### SIMD
- Collect AVX-512 range-search results with a masked compress-store
  (`VPCOMPRESSQ`) instead of a scalar bit-loop, on `SimdIndex2D` / `SimdIndex3D`
  and the compact `SimdIndex2DF32` / `SimdIndex3DF32`. This removes the
  large-result collection bottleneck: SIMD range search is now ~1.6–1.9× faster
  than the scalar index across 100k–1M boxes (it previously trailed the scalar
  index on full-extent queries), and the rounded `SimdIndex*F32` range search is
  now ~1.3–1.5× faster than the f64 `SimdIndex*` — a win on speed as well as
  memory. No API or result change; the win applies on AVX-512 CPUs.

### Configuration
- Lower `DEFAULT_PARALLEL_MIN_ITEMS` from 50,000 to 32,000, just above the
  measured serial/parallel build crossover (~30k items), so parallel builds kick
  in across the 30k–50k range where they are already faster. Override with
  `Index2DBuilder::parallel_min_items`.

### Documentation
- Document `RUSTFLAGS="-C target-cpu=native"` (or `x86-64-v3`) to enable AVX2 /
  AVX-512 codegen for the SIMD fallback and scalar autovectorization, and add
  measured scan / scalar-index / SIMD-index crossovers to the guide (a linear
  scan wins below ~100–130 boxes; an index amortizes after ~50–120 queries).

## [0.14.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.13.0...psi-v0.14.0) - 2026-06-19

### 2D
- Add the 2D region queries to the zero-copy `Index2DView`: `search_triangle` /
  `search_polygon` (plus `_into` / `any_*` / `visit_*`), so triangle and
  convex-polygon culling run straight over serialized bytes without an owned
  index.

### 3D
- Add `search_frustum` (plus `_into` / `any_frustum` / `visit_frustum`) to the
  zero-copy `Index3DView`, for frustum culling directly over serialized bytes.

## [0.13.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.12.0...psi-v0.13.0) - 2026-06-18

### 2D
- Add 2D convex-polygon region queries to `Index2D`: `search_polygon` /
  `search_polygon_into` (collect), `any_polygon` (boolean, short-circuits), and
  `visit_polygon` (fold without collecting). Build a `ConvexPolygon2D` from
  vertices in boundary order; a four-vertex polygon is a 2D view frustum / FOV
  trapezoid, and any convex shape works. The N-gon generalization of the triangle
  query, using the same exact separating-axis test (the box's two axes and the
  polygon's edge normals), so the result is precisely the boxes the polygon's
  filled area overlaps. Tighter than `search` over the polygon's bounding box —
  roughly 1.5x fewer hits for a near-round polygon, up to ~4.6x for a narrow
  trapezoid — and faster anyway (~2x for a round octagon, up to ~13x for a wide
  trapezoid), since internal nodes are pruned and subtrees fully inside are
  accepted whole instead of materializing the bounding-box result and filtering.
  For a triangle, `Triangle2D` + `search_triangle` returns the same set and is a
  touch faster. The predicates are public on `ConvexPolygon2D`: `overlaps_box`
  and `contains_box`.

## [0.12.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.11.0...psi-v0.12.0) - 2026-06-18

### 3D
- Add 3D frustum culling queries to `Index3D`: `search_frustum` /
  `search_frustum_into` (collect), `any_frustum` (boolean, short-circuits), and
  `visit_frustum` (fold without collecting). Build a `Frustum3D` from six
  inward-pointing planes (`from_planes`) or from a row-major view-projection
  matrix (`from_view_projection`, Gribb-Hartmann). The query is conservative: it
  returns every box overlapping the frustum and may include a few just past an
  edge or corner, but never drops a visible box. Far tighter than `search` over
  the frustum's bounding box — roughly 2x-4x fewer boxes and 3x-14x faster in a
  200k-box scene, since the slanted sides prune internal nodes and subtrees fully
  inside the frustum are accepted whole. The predicates are public on `Frustum3D`:
  `overlaps_box` and `contains_box`.

## [0.11.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.10.0...psi-v0.11.0) - 2026-06-18

### 2D
- Add 2D triangle region queries to `Index2D`: `search_triangle` /
  `search_triangle_into` (collect), `any_triangle` (boolean, short-circuits), and
  `visit_triangle` (fold without collecting). They return the items whose box
  overlaps the triangle's filled area — tighter than `search(tri.aabb())`, which
  over-reports the bounding-box corners the triangle misses. The traversal prunes
  internal nodes with a cheap box-vs-bbox test and accepts whole subtrees that lie
  inside the triangle without per-item tests, so it is also *faster* than
  collecting the bounding-box hits and filtering them by hand (roughly 2x-5x in a
  200k-box field, with 2x-7x fewer false positives). The predicates are public on
  `Triangle2D`: `overlaps_box` (separating-axis test) and `contains_box`.

### Documentation
- Move the API coverage matrix into the guide (`docs/guide.md`), where the full
  width renders, and leave a pointer from the README.

## [0.10.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.9.0...psi-v0.10.0) - 2026-06-18

### Nearest Neighbors
- Add point nearest-neighbor queries to the scalar `Index2DF32` / `Index3DF32`:
  `neighbors` / `neighbors_within` / `neighbors_into` / `neighbors_with`, the
  exact-refining `neighbors_exact*` (refined against your own f64 boxes), and
  `visit_neighbors`. Previously only the SIMD `SimdIndex*F32` carried them, so the
  no-`simd` compact path can now answer nearest-neighbor as well as range and
  raycast.

### Persistence
- Add `StreamLimits::coalesce_gap_bytes` to tune read coalescing. Records (tree
  nodes or payload blobs) within this many bytes of each other are fetched in one
  read; raising it to ~128-256 KB over-reads the gaps to collapse round-trips, a
  strong win on a remote source and waste on a local one, bounded by
  `max_read_bytes`. **Breaking:** `StreamLimits` gained a field, so a struct
  literal that set every field without `..StreamLimits::default()` now needs it.

### Documentation
- Add an API coverage matrix (which index type answers which query) to the README,
  and make method guidance explicit: a boolean overlap check is `any` (no
  allocation, stops early) rather than `search(..).is_empty()`, `search` returns
  an owned `Vec` so hot loops should reuse a buffer (`search_into` / `search_with`)
  or fold with `visit`, and for a few boxes a scalar index or a plain linear scan
  can beat the SIMD one.

## [0.9.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.8.0...psi-v0.9.0) - 2026-06-17

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

## [0.8.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.7.0...psi-v0.8.0) - 2026-06-16

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

## [0.7.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.6.0...psi-v0.7.0) - 2026-06-14

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

## [0.6.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.5.1...psi-v0.6.0) - 2026-06-14

### Search
- Add `search_iter`, a lazy iterator over the items intersecting a query box, on
  `Index2D` and `Index3D`. It descends the tree on demand, so consuming only a
  prefix (`.next()`, `.take(k)`, `.find(..)`) stops the traversal early and never
  allocates a result `Vec`. Reach for it to compose with iterator adapters or to
  bail out partway, where `search` (a whole owned `Vec`) and `visit` (a
  push-based callback) are awkward.

## [0.5.1](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.5.0...psi-v0.5.1) - 2026-06-13

### Documentation
- Restructure the README into a concise reference and move the long-form guide,
  persistence, and performance docs into `docs/`. Link every query method and
  type to docs.rs, add examples to `search` / `any` / `first`, document querying
  large or on-disk indexes via memory mapping, and add a clickable queries
  overview to the crate landing page.

## [0.5.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.4.3...psi-v0.5.0) - 2026-06-13

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

## [0.4.3](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.4.2...psi-v0.4.3) - 2026-06-09

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

## [0.4.2](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.4.1...psi-v0.4.2) - 2026-06-08

### SIMD
- Update SIMD comparisons for `wide` 1.5.

### Documentation
- Add README notes for AI usage and prior art.
- Clarify the live WASM demo link.

### WASM Demo
- Publish the interactive demo through GitHub Pages.

## [0.4.1](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.4.0...psi-v0.4.1) - 2026-06-05

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

## [0.4.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.3.3...psi-v0.4.0) - 2026-06-03

### API
- Return `BuildError::TreeTooLarge` instead of panicking when a requested tree
  layout cannot fit in memory.

### Benchmarks
- Move internal performance tools out of the published examples and into a
  local benchmark tools package.

## [0.3.3](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.3.2...psi-v0.3.3) - 2026-06-03

### Geometry
- Add point box constructors
- Share box accumulator helpers

### Documentation
- Add docs.rs feature badges and verify the docs.rs build
- Clarify query API guidance

### Lint
- Require SAFETY comments on all unsafe blocks

## [0.3.2](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.3.1...psi-v0.3.2) - 2026-06-02

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
