# Performance

Benchmark results and how to reproduce them. See the
[README](https://github.com/Filyus/packed_spatial_index#readme) for the API
overview.

## Baselines

The benchmarks below compare against:

- [`static_aabb2d_index`](https://crates.io/crates/static_aabb2d_index) by
  Jedidiah McCready — a Rust Flatbush port (build and search).
- [FlatGeobuf](https://flatgeobuf.org/) by Pirmin Kalberer and Björn Harrtell —
  a Flatbush-inspired geospatial format (build, search, persistence).
- [`bvh`](https://crates.io/crates/bvh) — used in the closest-hit raycast
  comparison.
- [`fast_hilbert`](https://crates.io/crates/fast_hilbert) by Stephan Hügel — a
  popular standalone Hilbert curve encoder (encode throughput).

This crate's own design follows the packed Hilbert R-tree of
[flatbush](https://github.com/mourner/flatbush) (Vladimir Agafonkin) and its
Rust port `static_aabb2d_index`.

## Hilbert encoder throughput

The build step sorts items by the Hilbert index of their box centers, so encoder
throughput feeds directly into build time. The `hilbert2d_bench` suite encodes
100,000 random `(u16, u16)` points into an output buffer — independent
iterations the compiler can pipeline and vectorize, which reflects real build
usage. `black_box` wraps only the input and output buffers; wrapping every
element would collapse the measurement into single-call latency and bias it
toward the table-driven path. All Hilbert encoders produce identical indices for
the full `u16` range, so this is a like-for-like speed comparison. Lower is
better.

| Encoder | 100k encode | Throughput | vs `fast_hilbert` |
| --- | ---: | ---: | ---: |
| `magic_bits` (crate default) | 186 us | 537 Melem/s | 6.9x |
| reference `static_aabb2d_index::hilbert_xy_to_index` | 180 us | 556 Melem/s | 7.1x |
| `lut` (4-bit state machine) | 243 us | 412 Melem/s | 5.3x |
| `loop_rotation` | 1.02 ms | 98 Melem/s | 1.3x |
| `fast_hilbert::xy2h` (order 16) | 1.28 ms | 78 Melem/s | 1.0x |
| `morton` (Z-order baseline, not Hilbert) | 47 us | 2.1 Gelem/s | n/a |

The crate's default `magic_bits` encoder is branchless bit arithmetic that
auto-vectorizes, landing within a few percent of the `static_aabb2d_index`
reference and about 6.9x faster than `fast_hilbert`. `fast_hilbert` is generic
over coordinate width and curve order; that generality costs it throughput on the
fixed `u16` path, where it lands near `loop_rotation`. Note the trade-off
direction flips for single-call latency: in a dependent-accumulation loop the
table-driven `lut` wins because it has no arithmetic dependency chain, while
`magic_bits` is fastest only when iterations are independent (the build case).
The Morton row is a Z-order curve, included only as a locality/speed baseline —
it is not a Hilbert curve and is not used for ordering. Reproduce with
`cargo bench --bench hilbert2d_bench --features bench-internals`.

## 2D competitors

Lower is better. The 2D competitor workload uses 100,000 random AABBs and
1,000 random query boxes; build and search competitors are measured in the
same benchmark suite on the same generated inputs. Persistence rows use the
canonical byte format for 100,000 boxes.

| Benchmark | FlatGeobuf | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: | ---: |
| Full build | 70.18 ms | 8.95 ms | 3.18 ms serial / 2.20 ms parallel | - |
| Search batch | 555.83 us | 341.56 us | 416.15 us | 128.64 us |
| Serialize built tree (fresh buffer) | - | - | 407.64 us | 689.42 us |
| Serialize built tree (reused buffer) | 131.93 us | - | 68.78 us | 140.21 us |
| Load owned tree | 740.23 us | - | 607.62 us | 935.51 us |
| Load zero-copy view | - | - | 37.59 us | n/a |

`SimdIndex2D` searches faster than `Index2D` but **serializes and loads slower**
(roughly 1.5–2.8× in a clean re-measure) — expected, not noise. The on-disk
format is AoS (one canonical format shared by both, so the bytes are
interchangeable). `Index2D` stores AoS too, so `to_bytes` is close to a memcpy
and `from_bytes` close to zero-copy; `SimdIndex2D` stores SoA (separate min/max
columns, what makes its queries fast), so it gathers SoA→AoS to serialize and
scatters AoS→SoA to load. The `reused buffer` row isolates this best:
`Index2D` 68.78 us (≈memcpy) vs `SimdIndex2D` 140.21 us (the transpose). So
`SimdIndex2D` pays at serialize/load to win at query time — prefer it when you
query far more than you persist, and load read-mostly bytes through the zero-copy
`Index2DView` (37.59 us) rather than rebuilding an owned SoA index.

Scalar `Index2D` search versus `static_aabb2d_index` is dataset-sensitive.
Two search runs with the same item/query counts but different generated inputs
showed opposite scalar ordering:

| Search batch | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: |
| `flatgeobuf2d_bench`, seed `0xF6B` | 341.56 us | 416.15 us | 128.64 us |
| `index2d_bench`, seed `0xB0B` | 643.85 us | 311.72 us | 126.21 us |

## 2D vs 3D

Lower latency is better. The `3D speed` column is `2D latency / 3D latency`, so
values above `1.00x` mean 3D is faster. The build workload uses 100,000 boxes
with `node_size = 16`; search and KNN use 1,000 query boxes or points.

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
| Serialize built tree (fresh buffer) | 407.64 us | 562.55 us | 0.72x |
| Serialize built tree (reused buffer) | 68.78 us | 96.51 us | 0.71x |
| Load owned tree | 607.62 us | 819.04 us | 0.74x |
| Load zero-copy view | 37.59 us | 37.66 us | 1.00x |

| SIMD persistence | `SimdIndex2D` | `SimdIndex3D` | 3D speed |
| --- | ---: | ---: | ---: |
| Serialize built tree (fresh buffer) | 689.42 us | 973.17 us | 0.71x |
| Serialize built tree (reused buffer) | 140.21 us | 240.51 us | 0.58x |
| Load owned tree | 935.51 us | 1.3083 ms | 0.72x |

## 3D SIMD

The speed column is scalar/serial latency divided by SIMD/parallel latency, so
values above `1.00x` mean the SIMD or parallel path is faster.

| Stage | Dataset / mode | Baseline | SIMD / parallel | Speed |
| --- | --- | ---: | ---: | ---: |
| Search batch | uniform XYZ | `Index3D` 432.61 us | `SimdIndex3D` 164.68 us | 2.63x |
| Search batch | flat Z | `Index3D` 2.01 ms | `SimdIndex3D` 0.85 ms | 2.36x |
| Build `finish_simd` | uniform XYZ, 200k boxes | serial 9.99 ms | parallel 7.03 ms | 1.42x |

## Large-window range search

When a query fully contains a tree node, the covered-range fast path collects the
whole subtree by copying its contiguous leaf-index range instead of running
per-item overlap tests. This keeps the SIMD indexes from regressing against the
scalar indexes as the window grows: full-extent windows reach parity (both paths
just copy the contiguous index range) and everything smaller stays ahead. On
AVX-512 a masked compress-store collects the matching leaf indices in one
instruction, widening the SIMD lead on dense mid-to-large windows (e.g. the 3D
flat-Z batch above, and the `large` / `thin slab` rows here). Workload: 100,000
boxes over a 10,000-wide space, 1,000 query boxes per window class. Lower is
better.

| Window (2D) | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: |
| small (10–200) | 362.51 us | 124.87 us |
| large (2,000–5,000) | 6.51 ms | 4.11 ms |
| wide sliver | 2.47 ms | 0.87 ms |
| full extent | 11.99 ms | 11.97 ms |

| Window (3D) | `Index3D` | `SimdIndex3D` |
| --- | ---: | ---: |
| small (50–300) | 427.57 us | 166.42 us |
| large (2,000–5,000) | 10.70 ms | 4.31 ms |
| thin slab | 3.64 ms | 1.29 ms |
| full extent | 11.73 ms | 11.21 ms |

## Closest-hit raycast vs the `bvh` crate

Closest-hit raycast over the packed index against the
[`bvh`](https://crates.io/crates/bvh) crate (100k boxes, 1,000 rays of length
4,000). For closest hit, "BVH" is a fair hand-rolled ordered traversal over its
SAH tree; for all hits, its broad-phase `traverse_iterator`.

| metric | packed SoA/SIMD | BVH |
|---|---:|---:|
| build (uniform) | **5 ms** | 58 ms |
| closest hit, uniform | **1.2 ms** | 1.8 ms |
| closest hit, clustered | 111 µs | **54 µs** |
| all hits, uniform | **0.85 ms** | 3.2 ms |
| all hits, clustered | **100 µs** | 104 µs |

The packed Hilbert tree builds ~11x faster. All-hits has no early-exit, so the
SIMD slab test wins on uniform scenes and ties on clustered; for closest hit a
SAH BVH builds a structurally better tree and wins on heavily clustered scenes.
Reproduce with `cargo bench --bench raycast3d_bench --features simd`.

## Ray-triangle closest hit (mesh payload)

A triangle payload plus the index over each triangle's bounding box is a
streamable mesh BVH: `raycast` returns candidate boxes, then
`Ray3D::closest_triangle` runs the exact Moller-Trumbore test only on those. The
records are fixed-width, so the payload drops its offset table (smaller file, one
fewer streamed read) and a view borrows them as a zero-copy typed slice. The
`f32` records (`Triangle3DF32`, 36 bytes) are half the size of `f64`
(`Triangle3D`, 72 bytes) and test 8 at a time through `wide::f32x8`; the `f64`
path is scalar. Workload: 4,096 rays against 4,096 candidate triangles (the
narrow-phase test). Lower is better.

| `closest_triangle` | per batch | per ray x triangle |
| --- | ---: | ---: |
| `f64` `Triangle3D` (scalar) | 202.8 ms | 12.1 ns |
| `f32` `Triangle3DF32` (SIMD) | 90.9 ms | 5.4 ns |

The `f32` SIMD kernel runs ~2.2x faster than scalar `f64` here. Most of that is
the kernel: in pure scalar (no `simd` feature) `f32` is only modestly ahead of
`f64`, since both autovectorize and the win is mainly the 8-wide test. f32's
other benefit is size — half the payload bytes on disk and over the wire.
Reproduce with `cargo bench --bench raytriangle3d_bench --features simd`.

## f32 storage vs f64

The `coord_precision` suite compares compact f32 storage with f64 storage.
Lower is better. Range rows run `search(Box2D)` for 1,000 random query boxes.
Small query boxes cover 0.1% of the coordinate extent per axis; large query
boxes cover 5%. KNN rows use 200 query points with top-8 results.

Quick selector:

- `SimdIndex2D`: 32-byte f64 boxes. Use for exact range queries with many hits
  and fastest exact KNN.
- `SimdIndex2DF32::search`: 16-byte rounded f32 boxes, SIMD-batched. In the
  AVX-512 runs above it is also the **fastest** range path — ~1.3–1.5× over the
  f64 `SimdIndex2D` (half the box bytes plus 16 boxes per SIMD chunk to f64's 8),
  so it wins on speed *and* memory, not just memory — at the cost of a few
  near-boundary false positives from the outward-rounded boxes. Returns the same
  hits as the scalar `Index2DF32` (both round the query inward onto the f32 grid).
  Use it when those extra hits are OK, or as a compact first-pass filter.
- `SimdIndex2DF32::*_exact`: 16-byte f32 index plus source f64 boxes. Use when
  exact range queries return few hits and compact storage matters. Exact KNN is
  available, but f64 is faster in these runs.
- `Index2DF32` / `Index3DF32`: the same 16/24-byte f32 boxes, scalar (no `simd`).
  Identical range hits to `SimdIndex2DF32` (a conservative superset from the
  outward-rounded boxes) plus `search_exact`; pick it for the half memory
  without the SIMD dependency, or to stream a compact file with
  `StreamIndex2DF32` / `StreamIndex3DF32` (half the box bytes over the wire).
  Scalar f32 trades speed for memory: a 1M-box spot check ran range queries
  about 30% slower than `Index3D` and `search_exact` about 45% slower. The query
  is rounded once onto the f32 grid so each node compares f32-vs-f32 with no
  per-node widen (and bit-identical hits to the f64 test); the residual gap is
  the few extra conservative candidates from the outward-rounded boxes, and a
  build about 1.7x slower from that rounding. Reach for it when you want half the
  memory without a SIMD dependency, not for raw query speed (use
  `SimdIndex2DF32` for that).

| Range query | Items | `f64` exact | `f32` rounded | `f32` exact |
| --- | ---: | ---: | ---: | ---: |
| small query boxes | 10k | 107 us | 70 us | 78 us |
| small query boxes | 100k | 125 us | 93 us | 102 us |
| small query boxes | 1M | 177 us | 137 us | 145 us |
| large query boxes | 10k | 135 us | 113 us | 304 us |
| large query boxes | 100k | 594 us | 465 us | 1.86 ms |
| large query boxes | 1M | 5.01 ms | 3.28 ms | 16.78 ms |

The `f64 exact` and `f32 rounded` columns use the compress-store collection on
AVX-512, which roughly halves the large-window rows versus the scalar collection;
`f32 exact` runs the per-item refinement callback (no compress) and is unchanged.

| KNN workload | `f64` exact | `f32` rounded | `f32` exact |
| --- | ---: | ---: | ---: |
| 10k items | 243 us | 282 us | 302 us |
| 100k items | 357 us | 376 us | 410 us |

## Summary

- `Index2D` is the general-purpose path;
- `SimdIndex2D` and `SimdIndex3D` are best for heavier query batches where SIMD
  work amortizes well;
- scalar `Index2D` search versus `static_aabb2d_index` depends on the generated
  data and query distribution, while `Index2D` build is faster in these runs;
- `Index3D` build and KNN are still slower than `Index2D`, but uniform 3D search
  can be faster when Z meaningfully prunes the tree;
- f32 storage halves box memory; exact callbacks trade source-box lookup for
  exact results;
- SIMD persistence uses the same canonical bytes as scalar persistence; it pays
  an SoA gather/scatter cost but avoids a second file format;
- `any` is often much faster than collecting full result sets when all you need
  is existence;
- AVX-512 is not always the fastest path in parallel workloads because CPU
  frequency behavior matters.

## Benchmark layout

Performance-related code lives under `benches`:

- `benches/*.rs` are Criterion benchmark suites run with `cargo bench`.
- `benches/tools` is a local developer package for quick comparisons of encoder
  variants, sort strategies, node sizes, parallel builds, and SoA layouts.

The local tools use the hidden `bench-internals` feature and are excluded from
the published crate.

```bash
cargo run --release --manifest-path benches/tools/Cargo.toml --bin sortkey_quality_2d
cargo run --release --manifest-path benches/tools/Cargo.toml --bin node_size_3d
```

Benchmark coverage:

- `hilbert2d_bench` compares the crate's Hilbert encoders against the
  `static_aabb2d_index` reference and the `fast_hilbert` crate;
- `flatgeobuf2d_bench` compares against FlatGeobuf's packed Hilbert R-tree;
- `index2d_bench` compares build/search paths against `static_aabb2d_index`;
- `index3d_bench` covers 3D build/search/KNN, SIMD search/build, dimension
  comparisons, node sizes, and a hidden Morton baseline;
- `persistence_knn2d_bench` / `persistence_knn3d_bench` cover scalar/SIMD
  persistence, loaded views, and KNN;
- `raycast3d_bench` compares closest-hit raycast against the `bvh` crate;
- `raytriangle3d_bench` compares `closest_triangle` over `f64` vs compact `f32`
  triangle records.

## Build flags

The default `x86-64` target compiles SIMD at SSE2 width (128-bit). To get AVX2 /
AVX-512 codegen, build with one of:

```bash
RUSTFLAGS="-C target-cpu=native"     # best for a binary you run on the build machine
RUSTFLAGS="-C target-cpu=x86-64-v3"  # portable AVX2 baseline (all v3 CPUs)
```

`native` enables every feature of the building CPU but produces a **non-portable**
binary (an older CPU can fault on a missing instruction); use the `x86-64-v3`
microarchitecture level for binaries you distribute.

The explicit SIMD search / visit / raycast kernels are selected at runtime
(`is_x86_feature_detected!`) and dispatch **AVX-512 → AVX2 → SSE2**: AVX-512 uses
`VPCOMPRESSQ` result collection (~1.6–1.9× over scalar), the AVX2 tier uses a
[left-pack](left-pack.md) emulation (~1.3–1.6× over the SSE2 fallback on
AVX2-only CPUs), and SSE2 is the floor. So these kernels do **not** need
`target-cpu` to pick the right width. The flag's remaining benefit is widening
the **scalar** autovectorized loops (~1.1–1.3×). (The WASM demo passes
`-Ctarget-feature=+simd128` for the same reason.)

## Reproducing

```bash
cargo bench --bench hilbert2d_bench --features bench-internals
cargo bench --bench index2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench index3d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench persistence_knn2d_bench --no-default-features --features simd,bench-internals
cargo bench --bench persistence_knn3d_bench --no-default-features --features simd,bench-internals
cargo bench --bench flatgeobuf2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench coord_precision --no-default-features --features f32-storage
cargo bench --bench raycast3d_bench --features simd
cargo bench --bench raytriangle3d_bench --features simd
```
