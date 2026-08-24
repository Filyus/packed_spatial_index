# Changelog

All notable changes to this crate are documented here.

## [Unreleased]

### Performance

- The build's reorder gather is prefetched. Profiling a 100 000-box build put
  60.9% of all L1 read misses in one place — `finish` reading `items` in Hilbert
  order while writing its output sequentially — and every address it needs is
  already sitting in the sort's `order` array, so the loads can be started
  early. A 1 000 000-box build runs at 0.92x and 0.91x across two independently
  built binaries (spreads 0.9-2.8%, untouched baseline control at 1.004), and
  `index3d_simd_build/finish_simd_serial` at 0.967. Smaller builds do not
  resolve and change sign between builds: at a thousand items the gather fits in
  cache and there is nothing to hide. The distance (64) is swept rather than
  guessed — short distances measure *worse* than no prefetch at all, because the
  hint lands too late to help and still costs an instruction per item.

- The SoA indexes' serialize and load paths now move a whole box record at a
  time. `SimdIndex*::to_bytes` wrote one `f64` per `extend_from_slice` — a
  capacity branch and a length store-back for every eight bytes, four or six
  times per node — and `from_bytes` pushed the same way into `with_capacity`
  columns. Staging the record and appending it whole, and filling one column per
  exact-size `collect`, takes `simd_to_bytes_into` to 0.69x at 100 000 boxes and
  0.84x at 1 000 000 (2D; 0.68x / 0.84x in 3D) and `simd_from_bytes_owned` to
  0.89x at 1 000 000 2D. This is the SoA<->AoS transpose the persistence table
  in `docs/performance.md` prices as the reason the SIMD index costs more to
  persist than the scalar one. The 3D 1 000 000 load did not resolve — at 48 MB
  it is bandwidth-bound, not instruction-bound. Measured interleaved on a pinned
  performance core over 7 rounds, against the scalar `from_bytes_owned`
  benchmarks as controls (0.995 and 0.999).
- The portable `wide` SIMD kernels — the tier the runtime dispatch falls back
  to without AVX2, which is also the wasm32 `simd128` path — assembled each lane
  vector from four (or eight) separately indexed elements, so every load carried
  that many bounds checks; `end` comes from a different `Vec` than the columns,
  so LLVM could not fold them. Slicing once gives it a length it can see. The
  collecting kernels also drain the set bits rather than testing all four lanes.
  `query/simd_simd_serial` runs at 0.80x, `index3d_simd_search/
  uniform_simd_any_wide4` at 0.90x, `query/simd_any_wide4_serial` at 0.92x,
  against controls at 1.005 and 0.997. The AVX-512 and AVX2 tiers are untouched,
  so machines that reach them see no change to range search.
- Build-time bounds folds and coordinate clamps are branch-free. `f64::min` and
  `f64::max` carry a four-instruction NaN fixup on baseline x86-64 that also
  forces an AoS row to be deinterleaved before it can fold; the inputs here are
  already proven NaN-free by `check_item_bounds`, so a compare-select is exact
  and compiles to one instruction. With the saturating clamps in `hilbert_coord`
  / `normalize_center` and an exact-size `collect` for the sort-key vector, a
  100 000-box build executes 0.86x the instructions (0.88x in 3D).

  Clocked by isolating the commit and building each side twice, because a single
  pair of binaries could not separate the effect from code layout:

  | `build/index_serial` | build 1 | build 2 | spread b/h    |
  | ---                  | ---:    | ---:    | ---           |
  | 16                   | 0.905   | 0.907   | 0.7-2.6%      |
  | 1 000                | 0.878   | 0.954   | 1.6-3.7%      |
  | 100 000              | 0.922   | 1.003   | 10-16%        |

  `build/crate` — the `static_aabb2d_index` baseline, identical code in every
  binary — stayed inside 1.003 in both pairs, and two builds of the *unchanged*
  source read 0.998-1.007 at the two small sizes. So the small-size wins are
  real; the 1 000 figure is layout-sensitive and is honestly a 5-12% band rather
  than the 12% one build alone would have claimed.

  The 100 000 row does not resolve, and the size sweep is why: the instruction
  saving is in the encode, extent and pack stages, while at that size the build
  is dominated by the radix sort and the reorder gather, which are memory-bound.
  The win is largest where the working set stays in cache.
- The owned indexes' `count(query)` now answers a fully contained subtree from
  its leaf range instead of testing every item inside it, so a wide window costs
  node pops rather than item tests. The new `count_windows` benchmark group
  measures it against a `search_into_stack` baseline that reuses both buffers,
  over 1 000 queries and 100 000 boxes:

  | window        | before     | after    | ratio    |
  | ---           | ---:       | ---:     | ---:     |
  | `full_extent` | 61.84 ms   | 24.26 us | 0.0004   |
  | `large`       | 7.166 ms   | 2.955 ms | 0.41     |
  | `wide_sliver` | 1.375 ms   | 1.266 ms | 0.92     |
  | `small`       | 229.9 us   | 238.2 us | **1.04** |

  Small windows pay 3.6% more, reproduced across two runs while eleven control
  arms in the same group stayed inside 1.005: nothing is contained at that size,
  so the per-child containment test is pure overhead there. That is the same
  trade `search` already makes — its contained traversal runs the identical test
  — so `count` is now consistent with it rather than cheaper on the small case
  and dramatically worse on every larger one. `wide_sliver` is the designed
  control: a long thin window contains no whole node, and it moved least.

  Region queries (`&Triangle2D` / `&ConvexPolygon2D` / `&Frustum3D`) keep the
  traversal they had. The views and the SIMD frontends were already effectively
  contained-aware here — their counting enumeration collapses to an add because
  the closure ignores the item index — and are unchanged.

- The zero-copy views' range search now takes the contained-subtree fast path.
  `Index2DView` / `Index3DView` had a root-contains shortcut and nothing below
  it, so a query covering a whole subtree still parsed every one of its boxes
  out of the byte buffer and tested it — the one search path in the crate that
  did, since the owned indexes and the views' own region queries already share
  that traversal. Routing `search` / `search_into` / `search_with` / `visit`
  through it removes the per-item work: on 200 000 clustered boxes the bounds
  parses for a 15% window drop 5 414 → 1 070 and the query runs at 0.65× the
  time, a 40% window 35 254 → 2 358 at 0.48×, a whole-extent-ish window
  213 337 → 2 997 at 0.34×; at 1 000 000 boxes the same cells read 0.48×, 0.35×
  and 0.21×. 3D behaves the same way and goes further at depth: at 200 000 boxes
  a 40%-per-axis window drops 14 926 → 3 317 parses at 0.68×, an 80% one
  115 911 → 12 421 at 0.52×, whole-extent 213 337 → 5 942 at 0.32×; at
  1 000 000 those run 0.49×, 0.30× and 0.18×.
  Windows too small to contain a whole node pay a few percent instead — one
  containment test per visited node with nothing to skip, tens of nanoseconds on
  a sub-microsecond query — which is the price of the rest.
  `any` and `first` deliberately keep the overlaps-only traversal: they stop at
  the first hit, so a containment test per node could only add work. Measured
  with both traversals in one binary (paired, interleaved, order-alternating),
  against a sliver-window control where the mechanism cannot fire and an
  extent-covering control where both arms take the root shortcut; both read
  1.00 ± 0.03, and an arm-against-itself floor reads 0.97-1.00. The SIMD views
  were never affected: they carry their own traversal, which has had the
  contained fast path all along.

### Search

- Added `count(query)`, which returns how many items overlap a query without
  building a result `Vec`. It counts inside the traversal, so it costs a
  `search` minus the collection; the shape it replaces — `search(query).len()` —
  allocated a `Vec` only to read its length. Available on every frontend that
  answers a range query: the owned `f64` indexes and their views (where it also
  takes the region shapes, `Triangle2D` / `ConvexPolygon2D` / `Frustum3D`), the
  SIMD indexes and views, the scalar and SIMD `f32` frontends (counting the same
  conservative superset their `search` returns), and the streaming readers as
  `count(query) -> Result<usize, StreamError>`, charging the same read budget as
  `search`. The streaming readers also gained `count_region` for the shape
  queries (and `count_async` / `count_region_async` under the `async` feature),
  so counting is not the one query there that has no region form. The alternative was to keep pointing callers at `visit` with their
  own counter; that is four lines and a `ControlFlow` import for the most common
  aggregate over a query, which is how `search(..).len()` kept winning.

### Documentation

- The guide's "Choosing a query method" section now opens with an "I need … →
  use …" table keyed by the answer you want — a `bool`, one hit, a count, every
  hit, an early exit, a payload — rather than by the query shape, and the notes
  under it explain why short-circuiting is a property of the traversal and not a
  filter over collected results. A production consumer asked for methods that
  already existed (`any`, `search_into`, allocation-free counting) because
  nothing pointed at them from where they were looking.
- `Frustum3D` now documents picking, which it could always do and nothing said
  so: narrowed to the pixels around a cursor the same query answers "what is
  under the click", and widened to a dragged rectangle it answers rubber-band
  selection. The type carries a compiled pick-matrix example (scale the
  view-projection so the clicked NDC rectangle fills the clip cube), and both it
  and the guide are explicit about the half the index does not do — results are
  conservative bounding-box candidates in unspecified order, so the winner comes
  from a second step: `raycast_closest` along the cursor ray, `neighbors` for
  nearest-in-world-space, or an exact test against real geometry. For a
  single-pixel pick the ray alone remains the more direct tool.

- Every `search` method now says in its own rustdoc that it allocates a fresh
  `Vec` per call and names the cheaper sibling for a boolean test, a hot loop,
  or a fold — `search` is where a reader lands first, so the alternatives are
  documented there rather than only in the guide. No API change.

## [0.27.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.26.0...psi-v0.27.0) - 2026-08-12

### API

- **Breaking:** building an index from a box with `min > max` on any axis, or a
  `NaN` bound, now fails with the new `BuildError::InvalidItemBounds { at }` —
  naming the position the box was added at — instead of producing an index.
  `Box2D::try_new` / `Box3D::try_new` already rejected these and `new` does not,
  so they could reach a tree, where they were answered inconsistently: a box that
  covers no region is still *contained* by queries it does not overlap, and the
  whole-subtree search shortcut tested containment. The check is one pass over the
  items before any packing work — no measurable cost at 100 000 items, about 3% of
  build time at 1 000 000. Callers that fed boxes straight from an external source
  should route them through `try_new`, or normalize with `min`/`max`, before
  adding.
- **Breaking:** `BuildError` is now `#[non_exhaustive]`, so the next variant will
  not break callers again. An exhaustive `match` on it needs a `_` arm.

### Safety

- Corrected SAFETY.md's `unsafe` inventory, which had gone stale in silence: it
  named AVX-512 while AVX2 carries the SoA, f32 and raycast search paths, and it
  did not mention the SSE prefetch hint, the payload record casts, or the
  left-packing of SIMD hits into a result buffer's reserved capacity — the
  category an auditor would want to read first. The document now also states what
  loading deliberately does not validate, and why that is a correctness question
  rather than a memory-safety one.

### Search

- Every search entry point now agrees on an index that carries a crossed box,
  which a loaded file may still do: the whole-subtree shortcut in the owned
  indexes, the zero-copy views, the SIMD layouts and the shared range traversal
  test overlap before containment. Previously one query could return every item
  through `search` and none through `search_iter` on the same index.

## [0.26.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.25.0...psi-v0.26.0) - 2026-07-31

### API

- **Breaking:** `StreamLimits` is now `#[non_exhaustive]`, so it is built from
  `Default` and assigned field by field rather than with a struct literal. The
  fields stay public; this is the last time adding a limit breaks callers.

### Persistence

- Added the optional `PFIX` chunk (format revision 13, no `format_version`
  bump) and `Serializer*::payload_prefix_len`: a dense, leaf-rank-indexed copy
  of each payload blob's leading bytes, so a prefix scan reads runs instead of
  one range per match. The bytes stay in the blobs too, so the chunk is purely
  additive — an older reader skips it and answers identically, just with more
  reads. It is not written for a fixed-width payload whose whole record is
  already the prefix.
- Prefix scans now read that section where an artifact carries one, which is
  what turns the chunk above into fewer reads rather than only fewer bytes on
  disk: measured on 300 items with 512-byte bodies, 301 reads become 2 for the
  same 9 608 bytes. The section rides on `StreamCoreParts`, so a directory
  split off and reattached to a fresh reader keeps it — the warm-isolate case
  the chunk exists for. An artifact without the section still uses the strided
  blob path, and both answer byte for byte alike.
- Added `StreamLimits::prefix_coalesce_gap_bytes`, the coalescing gap for
  payload-prefix visits. It defaulted — and still defaults — to `prefix_len`,
  which never merges once bodies exceed `2 * prefix_len`, so a prefix scan
  issues one range read per match. That is the right trade for a local file and
  the wrong one over object storage, where it is a request per match; raising
  the gap buys those requests back by reading the bodies in between. Measured
  over 100 000 items with 64-byte bodies and 1 001 matches: 1 011 reads for
  339 188 bytes become 11 reads for 379 188.

### SIMD

- The `simd` feature now requires `wide` 1.6, which deprecates the `.blend`
  this crate called on `f64x4`/`f32x8` in favour of `.select`. Identical
  semantics at every call site and the same `rust-version = 1.89`; the floor is
  raised so the method resolves whatever a consumer's lockfile already pins.

## [0.25.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.24.1...psi-v0.25.0) - 2026-07-15

### Search

- Added async payload-prefix and sparse payload-body visitors to every streaming
  index variant: `visit_payload_prefixes_async`,
  `visit_payload_prefixes_region_async`, and
  `visit_payloads_at_ranks_async`. They let async range readers paginate and
  materialize only the selected payload bodies without a full payload scan.

### Persistence

- Made `StreamIndex*::has_payload()` available for every reader type, including
  async range readers, because it reads cached stream metadata and performs no
  I/O.

## [0.24.1](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.24.0...psi-v0.24.1) - 2026-07-06

### Documentation

- Added a compile-checked rustdoc example for paging payload reads with
  `visit_payload_prefixes` and `visit_payloads_at_ranks`.

## [0.24.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.23.0...psi-v0.24.0) - 2026-07-06

### Search

- Added payload-header streaming to the sync streaming indexes:
  `visit_payload_prefixes` / `visit_payload_prefixes_region` yield each
  match's insertion id, leaf rank, full payload length, and leading payload
  bytes without reading payload bodies, and `visit_payloads_at_ranks` fetches
  full payloads for an explicit rank set in coalesced ascending runs —
  together enabling paged payload access. The new `PayloadPrefix` struct is
  exported at the crate root. Breaking: `StreamError` gains an `InvalidRank`
  variant.

## [0.23.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.22.0...psi-v0.23.0) - 2026-07-02

### Safety

- Hardened streaming index container opening against hostile inputs: oversized
  chunk directories, overflowing chunk ranges, and unknown-length readers are
  rejected before large allocations or unchecked alignment.

### Search

- `raycast_closest` now includes hits exactly at `max_distance`, rejects
  non-finite ray origins/directions, and keeps SIMD raycast behavior aligned
  with scalar paths for zero or subnormal direction components.
- `Ray3D::closest_triangle` now uses scale-aware triangle intersection
  tolerances so tiny valid triangles can still be hit.

### Nearest Neighbors

- kNN point queries with `NaN` coordinates now return empty results across
  scalar, view, SIMD, and f32 paths; `-0.0` cutoffs are treated as zero.

### Persistence

- SIMD and view loaders now return `PayloadNotSupported` for payload-carrying
  files they cannot expose, while owning scalar loaders continue to load the
  validated index portion of those files.

### Performance

- `Index2D::from_bytes` and `Index3D::from_bytes` use little-endian aligned
  bulk copies for owning scalar loads, with the existing decoded fallback for
  other layouts.

## [0.22.0](https://github.com/Filyus/packed_spatial_index/compare/psi-v0.21.1...psi-v0.22.0) - 2026-07-02

### Search

- Added async region-query APIs to the streaming indexes. `StreamIndex2D` /
  `StreamIndex3D` (and the `f32` variants) now expose
  `search_region_async` / `visit_region_async` /
  `search_payloads_region_async` / `visit_payloads_region_async`, matching the
  sync streaming coverage for 2D region and 3D frustum queries.

### Documentation

- Clarified streaming-query documentation: sync and async streaming support
  range and region queries, while kNN and raycast remain in-memory/view use
  cases.

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
