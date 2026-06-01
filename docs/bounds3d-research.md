# Bounds3D Research Notes

This branch is intentionally a research branch. It should leave enough evidence
to make a 3D implementation decision, without committing the crate to a full
public `Index3D` API yet.

## Decision

Start future 3D support with a scalar, AoS, Hilbert-ordered packed R-tree:

- Public shape: `Bounds3D`, `Point3D`, `Index3DBuilder`, `Index3D`, and later
  `Index3DView`.
- Keep 2D and 3D as separate public APIs instead of introducing a const-generic
  index immediately.
- Use `Vec<Bounds3D>`, `Vec<usize>`, and `Vec<usize>` level bounds as the first
  internal layout, mirroring `Index2D`.
- Reuse `SearchWorkspace` and `NeighborWorkspace` if their internals remain
  dimension-agnostic.
- Defer `SimdIndex3D`, 3D persistence, and a stable 3D file format until scalar
  `Index3D` is proven.

Hilbert3D is the recommended first serious default candidate for read-heavy 3D
indexes. The prototype shows meaningfully better traversal quality and KNN time
than Morton3D on uniform, flat-Z, and degenerate datasets. Morton3D should stay
as a hidden experimental/build-speed baseline because it is much cheaper to
encode.

## Prototype Structure

The temporary prototype in `tests/bounds3d_research.rs` uses:

- `Bounds3D { min_x, min_y, min_z, max_x, max_y, max_z }`;
- `Point3D { x, y, z }`;
- a packed tree with item bounds first, then internal node bounds;
- `level_bounds` semantics matching `Index2D`;
- exact overlap search and exact KNN using squared point-to-box distance.

This is deliberately not wired into `src/` yet. The goal is to measure layout
quality and shake out edge cases before designing the real public API.

## Algorithm Notes

Both prototype sort keys normalize box centers into `[0, 2^21 - 1]` per axis and
produce a 63-bit `u64` key. This leaves one spare bit in `u64` and avoids
multiword keys. Morton3D interleaves bits directly. Hilbert3D uses a compact
Skilling-style axes-to-transpose transform before interleaving bits. The final
3-axis interleave uses the same SWAR-style bit spreading as Morton3D, but there
is not yet a 3D equivalent of the current 2D `magic_bits` Hilbert encoder.

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

Use `Hilbert3D` as the initial default candidate for future `Index3D`, matching
the current 2D philosophy. Keep `Morton3D` hidden as an experimental option for
build-speed-sensitive workloads and benchmarks. Avoid public `SortKey3D` until
both paths have Criterion coverage and the Hilbert encoder has more audit
coverage.

Do not generalize `Bounds2D`/`Index2D` into a single const-generic public type
yet. A shared internal helper layer may be useful later, but premature generic
abstraction would make the hot paths harder to read and benchmark.

Initial release-mode prototype numbers suggest `node_size = 8` should be the
first 3D default candidate, especially for KNN-heavy workloads. Do not change
the 2D default. Before stabilizing `Index3DBuilder::new`, repeat the comparison
with Criterion and realistic user data.

Local command:

```bash
cargo test --release --test bounds3d_research -- --ignored --nocapture
```

Prototype output from this branch:

| Dataset | Sort key | Node size | Sort ms | Search ms | Avg visited | Avg hits | KNN top-1 ms | KNN top-10 ms | KNN top-10 r80 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PlanarXY | Morton | 8 | 0.177 | 0.051 | 85.31 | 2.88 | 0.299 | 0.439 | 0.135 |
| PlanarXY | Hilbert | 8 | 0.922 | 0.039 | 68.62 | 2.88 | 0.231 | 0.351 | 0.113 |
| Uniform | Morton | 8 | 0.156 | 0.040 | 87.44 | 0.05 | 0.483 | 0.781 | 0.152 |
| Uniform | Hilbert | 8 | 1.138 | 0.025 | 50.81 | 0.05 | 0.294 | 0.536 | 0.083 |
| Clustered | Morton | 8 | 0.229 | 0.006 | 4.16 | 0.00 | 0.340 | 0.657 | 0.020 |
| Clustered | Hilbert | 8 | 1.078 | 0.004 | 4.16 | 0.00 | 0.230 | 0.436 | 0.017 |
| FlatZ | Morton | 8 | 0.167 | 0.110 | 306.62 | 1.71 | 1.076 | 1.494 | 0.534 |
| FlatZ | Hilbert | 8 | 1.127 | 0.076 | 165.06 | 1.71 | 0.566 | 0.795 | 0.291 |
| Degenerate | Morton | 8 | 0.163 | 0.038 | 86.19 | 0.04 | 0.447 | 0.775 | 0.134 |
| Degenerate | Hilbert | 8 | 1.141 | 0.026 | 50.25 | 0.04 | 0.267 | 0.518 | 0.077 |

The timing columns are only prototype smoke metrics, not publication-grade
benchmarks. The visited-bound counts are the more useful signal here: Hilbert3D
cuts visited bounds by about 40-46% on the non-clustered datasets in this smoke
test, while Morton3D keeps build sorting roughly 5-7x faster.

## Hilbert2D vs Hilbert3D Cost

The branch also has two focused comparison tests:

```bash
cargo test --release --test bounds3d_research hilbert2d_vs_hilbert3d_encode_survey -- --ignored --nocapture --exact
cargo test --release --test bounds3d_research hilbert2d_vs_hilbert3d_index_survey -- --ignored --nocapture --exact
```

Raw key encoding is the largest visible gap:

| Encoder | Items | Total ms | ns/key |
| --- | ---: | ---: | ---: |
| Hilbert2D | 262144 | 1.816 | 6.93 |
| Hilbert3D | 262144 | 31.754 | 121.13 |

This is not a pure dimension-only benchmark: `Hilbert2D` is the current optimized
2D magic-bits encoder over 16-bit axes, while the temporary `Hilbert3D` path is a
compact Skilling-style 21-bit-per-axis prototype plus SWAR bit spreading for the
final interleave.

The closest apples-to-apples index scenario is `PlanarXY`: the same X/Y boxes
embedded into 3D with `z = 0`, so 2D and 3D produce the same average hit count.

| Dataset | Dimension | Node size | Build ms | Search ms | Avg visited | Avg hits | KNN top-1 ms | KNN top-10 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PlanarXY | 2D | 8 | 0.310 | 0.022 | 56.94 | 2.88 | 0.121 | 0.243 |
| PlanarXY | 3D | 8 | 1.012 | 0.037 | 68.62 | 2.88 | 0.235 | 0.343 |
| PlanarXY | 2D | 16 | 0.287 | 0.022 | 81.62 | 2.88 | 0.118 | 0.282 |
| PlanarXY | 3D | 16 | 0.987 | 0.037 | 100.50 | 2.88 | 0.285 | 0.390 |

Current read: `node_size = 8` matters more for the 3D path. In the planar
same-hit-count case, 3D build is about 3.3x slower, search is about 1.7x slower,
and KNN top-10 is about 1.4x slower than 2D at `node_size = 8`. True 3D datasets
can look better or worse depending on how much the Z dimension prunes queries.

## How To Run

The research tests are ignored by default:

```bash
cargo test --test bounds3d_research -- --ignored --nocapture
```

For less noisy timing numbers:

```bash
cargo test --release --test bounds3d_research -- --ignored --nocapture
```
