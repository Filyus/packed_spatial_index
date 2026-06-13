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

This crate's own design follows the packed Hilbert R-tree of
[flatbush](https://github.com/mourner/flatbush) (Vladimir Agafonkin) and its
Rust port `static_aabb2d_index`.

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
| Search batch | uniform XYZ | `Index3D` 389.13 us | `SimdIndex3D` 129.08 us | 3.01x |
| Search batch | flat Z | `Index3D` 1.8443 ms | `SimdIndex3D` 1.1514 ms | 1.60x |
| Build `finish_simd` | uniform XYZ, 200k boxes | serial 10.632 ms | parallel 6.5412 ms | 1.63x |

## Large-window range search

When a query fully contains a tree node, the covered-range fast path collects the
whole subtree by copying its contiguous leaf-index range instead of running
per-item overlap tests. This keeps the SIMD indexes from regressing against the
scalar indexes as the window grows: without it, full-extent SIMD searches were
several times slower than `Index2D`/`Index3D`; with it they reach parity on
full-extent windows and stay ahead on everything smaller. Workload: 100,000
boxes over a 10,000-wide space, 1,000 query boxes per window class. Lower is
better.

| Window (2D) | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: |
| small (10–200) | 337.61 us | 144.21 us |
| large (2,000–5,000) | 5.94 ms | 5.69 ms |
| wide sliver | 2.33 ms | 1.58 ms |
| full extent | 10.46 ms | 10.86 ms |

| Window (3D) | `Index3D` | `SimdIndex3D` |
| --- | ---: | ---: |
| small (50–300) | 382.86 us | 146.43 us |
| large (2,000–5,000) | 9.49 ms | 7.29 ms |
| thin slab | 3.33 ms | 1.67 ms |
| full extent | 10.54 ms | 10.91 ms |

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

## f32 storage vs f64

The `coord_precision` suite compares compact f32 storage with f64 storage.
Lower is better. Range rows run `search(Box2D)` for 1,000 random query boxes.
Small query boxes cover 0.1% of the coordinate extent per axis; large query
boxes cover 5%. KNN rows use 200 query points with top-8 results.

Quick selector:

- `SimdIndex2D`: 32-byte f64 boxes. Use for exact range queries with many hits
  and fastest exact KNN.
- `SimdIndex2DF32::search`: 16-byte rounded f32 boxes. Use for compact
  first-pass filtering, or when near-boundary false positives are OK.
- `SimdIndex2DF32::*_exact`: 16-byte f32 index plus source f64 boxes. Use when
  exact range queries return few hits and compact storage matters. Exact KNN is
  available, but f64 is faster in these runs.

| Range query | Items | `f64` exact | `f32` rounded | `f32` exact |
| --- | ---: | ---: | ---: | ---: |
| small query boxes | 10k | 84 us | 72 us | 73 us |
| small query boxes | 100k | 115 us | 87 us | 96 us |
| small query boxes | 1M | 156 us | 125 us | 134 us |
| large query boxes | 10k | 149 us | 117 us | 292 us |
| large query boxes | 100k | 1.06 ms | 704 us | 1.79 ms |
| large query boxes | 1M | 9.61 ms | 6.09 ms | 16.59 ms |

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

- `flatgeobuf2d_bench` compares against FlatGeobuf's packed Hilbert R-tree;
- `index2d_bench` compares build/search paths against `static_aabb2d_index`;
- `index3d_bench` covers 3D build/search/KNN, SIMD search/build, dimension
  comparisons, node sizes, and a hidden Morton baseline;
- `persistence_knn2d_bench` / `persistence_knn3d_bench` cover scalar/SIMD
  persistence, loaded views, and KNN;
- `raycast3d_bench` compares closest-hit raycast against the `bvh` crate.

## Reproducing

```bash
cargo bench --bench index2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench index3d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench persistence_knn2d_bench --no-default-features --features simd,bench-internals
cargo bench --bench persistence_knn3d_bench --no-default-features --features simd,bench-internals
cargo bench --bench flatgeobuf2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench coord_precision --no-default-features --features f32-storage
cargo bench --bench raycast3d_bench --features simd
```
