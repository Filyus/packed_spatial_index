# Bounds3D Notes

This document records the research that led to the first scalar 3D API. The
crate now exposes `Bounds3D`, `Point3D`, `Index3DBuilder`, `Index3D`, and
`Index3DView`; the remaining 3D work is SIMD/SoA layout and publication-grade
real-world benchmarks.

## Decision

3D support starts with a scalar, AoS, Hilbert-ordered packed R-tree:

- Public shape: `Bounds3D`, `Point3D`, `Index3DBuilder`, `Index3D`, and
  `Index3DView`.
- Keep 2D and 3D as separate public APIs instead of introducing a const-generic
  index immediately.
- Use `Vec<Bounds3D>`, `Vec<usize>`, and `Vec<usize>` level bounds as the first
  internal layout, mirroring `Index2D`.
- Reuse `SearchWorkspace` and `NeighborWorkspace` if their internals remain
  dimension-agnostic.
- Defer `SimdIndex3D` until scalar `Index3D` usage and benchmarks are proven.
  3D persistence now uses the canonical `PSINDEX` format with a 3D dimension
  flag.

Hilbert3D is the recommended first serious default candidate for read-heavy 3D
indexes. The prototype shows meaningfully better traversal quality and KNN time
than Morton3D on uniform, flat-Z, and degenerate datasets. Morton3D should stay
as a hidden experimental/build-speed baseline because it is much cheaper to
encode.

## Prototype Structure

The temporary prototype in `tests/bounds3d_research.rs` was used to choose the
initial production shape. It uses:

- `Bounds3D { min_x, min_y, min_z, max_x, max_y, max_z }`;
- `Point3D { x, y, z }`;
- a packed tree with item bounds first, then internal node bounds;
- `level_bounds` semantics matching `Index2D`;
- exact overlap search and exact KNN using squared point-to-box distance.

The production scalar API now lives in `src/`; the ignored research tests remain
as a benchmark-style notebook for comparing future 3D changes.

## Algorithm Notes

Morton3D normalizes box centers into `[0, 2^21 - 1]` per axis and produces a
63-bit `u64` key. Hilbert3D intentionally uses `[0, 2^16 - 1]` per axis: the
extra 21-bit precision was an encoding bottleneck and did not improve the
smoke-test layout quality.

The first Hilbert3D prototype used Skilling-style axes-to-transpose code. After
checking existing work, the current prototype uses a compact 24-state transition
LUT over `(state, octant)` and emits 3 Hilbert bits per input level. This follows
the same direction described by rawrunprotected/threadlocalmutex for fast 3D
Morton/Hilbert conversion, but builds the tiny transition table locally instead
of copying a published table.

Node sizes `8`, `16`, and `32` are the initial comparison set. The current 2D
default is `16`, but 3D has different overlap behavior, so the default should
be revalidated with 3D datasets before exposing `Index3DBuilder::new`.

Datasets used by the temporary test cover:

- planar XY boxes embedded into 3D with `z = 0`;
- uniform random boxes;
- clustered boxes;
- nearly flat Z data;
- degenerate zero-volume boxes.

Search metrics track visited bounds as a proxy for layout quality. KNN metrics
cover top-1, top-10, and finite-radius top-10.

## Current Recommendation

Use `Hilbert3D` as the default for `Index3D`, matching the current 2D philosophy.
Keep `Morton3D` hidden as an experimental option for build-speed-sensitive
workloads and benchmarks. Public `SortKey3D` intentionally exposes only
`Hilbert` for now.

Do not generalize `Bounds2D`/`Index2D` into a single const-generic public type
yet. A shared internal helper layer may be useful later, but premature generic
abstraction would make the hot paths harder to read and benchmark.

Initial release-mode prototype numbers suggested `node_size = 8` as a possible
3D candidate, especially for KNN-heavy workloads. The first production pass keeps
the crate-wide default at `16` for API consistency; this should be revisited with
Criterion and realistic user data before a 1.0 API.

Local command:

```bash
cargo test --release --test bounds3d_research -- --ignored --nocapture
```

Production Criterion benchmarks now live in `benches/index3d_bench.rs`:

```bash
cargo bench --bench index3d_bench --no-default-features
```

For quick targeted runs, filter by group:

```bash
cargo bench --bench index3d_bench --no-default-features 100000 -- --sample-size 10
cargo bench --bench index3d_bench --no-default-features index3d_search/uniform -- --sample-size 10
cargo bench --bench index3d_bench --no-default-features index3d_knn/uniform -- --sample-size 10
cargo bench --bench index3d_bench --no-default-features index3d_persistence -- --sample-size 10
cargo bench --bench index3d_bench --no-default-features index3d_loaded_view -- --sample-size 10
```

Prototype output from this branch:

| Dataset | Sort key | Node size | Sort ms | Search ms | Avg visited | Avg hits | KNN top-1 ms | KNN top-10 ms | KNN top-10 r80 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PlanarXY | Morton | 8 | 0.172 | 0.047 | 85.31 | 2.88 | 0.279 | 0.402 | 0.131 |
| PlanarXY | Hilbert | 8 | 0.272 | 0.035 | 72.94 | 2.88 | 0.233 | 0.353 | 0.105 |
| Uniform | Morton | 8 | 0.159 | 0.042 | 87.44 | 0.05 | 0.498 | 0.860 | 0.151 |
| Uniform | Hilbert | 8 | 0.271 | 0.026 | 49.06 | 0.05 | 0.271 | 0.503 | 0.081 |
| Clustered | Morton | 8 | 0.157 | 0.005 | 4.16 | 0.00 | 0.236 | 0.536 | 0.020 |
| Clustered | Hilbert | 8 | 0.271 | 0.004 | 4.16 | 0.00 | 0.205 | 0.421 | 0.016 |
| FlatZ | Morton | 8 | 0.157 | 0.115 | 306.62 | 1.71 | 1.261 | 1.430 | 0.529 |
| FlatZ | Hilbert | 8 | 0.274 | 0.068 | 150.38 | 1.71 | 0.578 | 0.754 | 0.276 |
| Degenerate | Morton | 8 | 0.160 | 0.038 | 86.19 | 0.04 | 0.447 | 0.757 | 0.136 |
| Degenerate | Hilbert | 8 | 0.271 | 0.024 | 47.31 | 0.04 | 0.276 | 0.503 | 0.076 |

The timing columns are only prototype smoke metrics, not publication-grade
benchmarks. The visited-bound counts are the more useful signal here: Hilbert3D
cuts visited bounds by about 40-51% on the non-clustered 3D datasets in this
smoke test. With the LUT encoder, Morton3D's sort-speed advantage is now usually
around 1.7x instead of the earlier 4-7x.

## Hilbert2D vs Hilbert3D Cost

The branch has two older ignored comparison tests from the prototype phase:

```bash
cargo test --release --test bounds3d_research hilbert2d_vs_hilbert3d_encode_survey -- --ignored --nocapture --exact
cargo test --release --test bounds3d_research hilbert2d_vs_hilbert3d_index_survey -- --ignored --nocapture --exact
```

The current production comparison lives in Criterion:

```bash
cargo bench --bench index3d_bench --no-default-features dimension_compare -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Short smoke run, lower is better:

| Area | Scenario | 2D | 3D | 3D/2D |
| --- | --- | ---: | ---: | ---: |
| Raw Hilbert encode | 262144 keys | 0.757 ms | 3.509 ms | 4.64x |
| Build | PlanarXY, node 8, 100k | 2.974 ms | 6.576 ms | 2.21x |
| Build | PlanarXY, node 16, 100k | 2.567 ms | 6.557 ms | 2.55x |
| Build | Uniform, node 16, 100k | 2.248 ms | 6.360 ms | 2.83x |
| Search | PlanarXY, node 16, 1k queries | 469 us | 644 us | 1.37x |
| Search | Uniform, node 16, 1k queries | 483 us | 424 us | 0.88x |
| KNN top-1 | PlanarXY, node 16, 1k points | 957 us | 1.218 ms | 1.27x |
| KNN top-10 | PlanarXY, node 16, 1k points | 2.061 ms | 2.898 ms | 1.41x |
| KNN top-10 | Uniform, node 16, 1k points | 2.072 ms | 4.586 ms | 2.21x |
| Serialize | PlanarXY, node 16, 100k | 593 us | 894 us | 1.51x |
| Load owned | PlanarXY, node 16, 100k | 558 us | 890 us | 1.59x |
| Load view | PlanarXY, node 16, 100k | 35.9 us | 36.8 us | 1.02x |

The closest apples-to-apples runtime scenario is `PlanarXY`: the same X/Y boxes
embedded into 3D with `z = 0`, so 2D and 3D produce the same hits and nearest
neighbors. In that case, the main 3D penalties are:

- build: 3D Hilbert key generation is much more expensive, and build remains
  about 2.2-2.8x slower after sort and packing amortize it;
- search: roughly 1.37x slower when Z cannot prune anything;
- KNN: about 1.3-1.4x slower for planar data, and about 2x slower on the current
  uniform comparison;
- persistence: owned serialization/loading tracks the larger 48-byte 3D bounds
  record, while zero-copy view loading is almost unchanged because validation is
  mostly tree-shape and index-pointer work.

True 3D search can be faster than projected 2D search when the Z dimension
filters candidates, as the uniform search comparison shows. KNN does not get the
same win in this smoke run; the extra distance dimension and queue work dominate.

## External References

- John Skilling, "Programming the Hilbert curve" (2004): compact n-dimensional
  Hilbert code based on a global Gray code and a cleanup pass.
- rawrunprotected/threadlocalmutex, "LUT-based 3D Hilbert curves" and "3D Hilbert
  curves in even fewer instructions": Morton/Hilbert conversion through small
  3D transformation-state tables.
- The Rust `fast_hilbert` crate confirms the same broad lesson in 2D: compact
  lookup tables can beat straightforward Hilbert implementations by large
  margins.

## How To Run

The research tests are ignored by default:

```bash
cargo test --test bounds3d_research -- --ignored --nocapture
```

For less noisy timing numbers:

```bash
cargo test --release --test bounds3d_research -- --ignored --nocapture
```
