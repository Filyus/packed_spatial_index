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
Skilling-style axes-to-transpose transform before interleaving bits.

Node sizes `8`, `16`, and `32` are the initial comparison set. The current 2D
default is `16`, but 3D has different overlap behavior, so the default should
be revalidated with 3D datasets before exposing `Index3DBuilder::new`.

Datasets used by the temporary test cover:

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
| Uniform | Morton | 8 | 0.175 | 0.040 | 87.44 | 0.05 | 0.581 | 0.932 | 0.155 |
| Uniform | Hilbert | 8 | 1.298 | 0.028 | 50.81 | 0.05 | 0.380 | 0.657 | 0.090 |
| Clustered | Morton | 8 | 0.159 | 0.005 | 4.16 | 0.00 | 0.277 | 0.549 | 0.020 |
| Clustered | Hilbert | 8 | 1.231 | 0.005 | 4.16 | 0.00 | 0.270 | 0.489 | 0.018 |
| FlatZ | Morton | 8 | 0.170 | 0.114 | 306.62 | 1.71 | 1.273 | 1.577 | 0.587 |
| FlatZ | Hilbert | 8 | 1.300 | 0.076 | 165.06 | 1.71 | 0.691 | 0.845 | 0.297 |
| Degenerate | Morton | 8 | 0.167 | 0.039 | 86.19 | 0.04 | 0.526 | 0.939 | 0.151 |
| Degenerate | Hilbert | 8 | 1.281 | 0.027 | 50.25 | 0.04 | 0.349 | 0.640 | 0.085 |

The timing columns are only prototype smoke metrics, not publication-grade
benchmarks. The visited-bound counts are the more useful signal here: Hilbert3D
cuts visited bounds by about 40-46% on the non-clustered datasets in this smoke
test, while Morton3D keeps build sorting roughly 7-8x faster.

## How To Run

The research tests are ignored by default:

```bash
cargo test --test bounds3d_research -- --ignored --nocapture
```

For less noisy timing numbers:

```bash
cargo test --release --test bounds3d_research -- --ignored --nocapture
```
