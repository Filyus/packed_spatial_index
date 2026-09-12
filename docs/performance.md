# Performance

Benchmark results and how to reproduce them. See the
[README](https://github.com/Filyus/packed_spatial_index#readme) for the API
overview.

Numbers are from one machine and depend on hardware and workload, so treat them as
relative, not absolute. Single-thread rows are measured with the benchmark thread
pinned to one performance core (`BENCH_PIN_CORE`, see [Reproducing](#reproducing));
parallel rows run unpinned so the rayon workers can spread across cores.

## Baselines

The benchmarks below compare against these crate versions:

- [`static_aabb2d_index`](https://crates.io/crates/static_aabb2d_index) `2.1.0` by
  Jedidiah McCready — a Rust Flatbush port (build and search).
- [FlatGeobuf](https://flatgeobuf.org/) (`flatgeobuf` `6.0.1`) by Pirmin Kalberer
  and Björn Harrtell — a Flatbush-inspired geospatial format (build, search,
  persistence).
- [`bvh`](https://crates.io/crates/bvh) `0.12.0` — used in the closest-hit raycast
  comparison.
- [`fast_hilbert`](https://crates.io/crates/fast_hilbert) `2.1.0` by Stephan Hügel — a
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
| `magic_bits` (crate default) | 185 us | 541 Melem/s | 3.5x |
| reference `static_aabb2d_index::hilbert_xy_to_index` | 183 us | 546 Melem/s | 3.5x |
| `lut` (4-bit state machine) | 241 us | 416 Melem/s | 2.7x |
| `loop_rotation` | 998 us | 100 Melem/s | 0.65x |
| `fast_hilbert::xy2h` (order 16) | 644 us | 155 Melem/s | 1.0x |
| `morton` (Z-order baseline, not Hilbert) | 47 us | 2.1 Gelem/s | n/a |

The crate's default `magic_bits` encoder is branchless bit arithmetic that
auto-vectorizes, landing within a few percent of the `static_aabb2d_index`
reference and about 3.5x faster than `fast_hilbert`. `fast_hilbert` is generic
over coordinate width and curve order; that generality costs it throughput on the
fixed `u16` path, though it still beats the naive `loop_rotation`. Note the trade-off
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
| Full build | 46.82 ms | 6.31 ms | 2.23 ms serial / 1.73 ms parallel | - |
| Search batch | 588.75 us | 485.64 us | 205.62 us | 127.36 us |
| Serialize built tree (fresh buffer) | - | - | 399.11 us | 613.21 us |
| Serialize built tree (reused buffer) | 131.28 us | - | 68.02 us | 160.86 us |
| Load owned tree | 646.49 us | - | 372.24 us | 554.75 us |
| Load zero-copy view | - | - | 34.96 us | n/a |

`SimdIndex2D` searches faster than `Index2D` but **serializes and loads slower**
(roughly 1.5–2.4× in a clean re-measure) — expected, not noise. The on-disk
format is AoS (one canonical format shared by both, so the bytes are
interchangeable). `Index2D` stores AoS too, so `to_bytes` is close to a memcpy
and `from_bytes` close to zero-copy; `SimdIndex2D` stores SoA (separate min/max
columns, what makes its queries fast), so it gathers SoA→AoS to serialize and
scatters AoS→SoA to load. The `reused buffer` row isolates this best:
`Index2D` 68.02 us (≈memcpy) vs `SimdIndex2D` 160.86 us (the transpose). So
`SimdIndex2D` pays at serialize/load to win at query time — prefer it when you
query far more than you persist, and load read-mostly bytes through the zero-copy
`Index2DView` (34.96 us) rather than rebuilding an owned SoA index.

Scalar `Index2D` search leads `static_aabb2d_index` on both generated inputs,
by 2.4× on the `0xF6B` set and 3.0× on `0xB0B`. That used to be a
dataset-sensitive call — a few percent to 1.8× depending on the inputs, and the
other way round on `0xF6B` under `2.0.0` of the baseline crate (see below) —
until the scalar collect paths stopped branching once per child: each node's
overlap tests now fold into a bitmask and the traversal branches once per hit,
which removed the mispredicts that dominated wide queries. The margin is now
well outside the run-to-run spread on either side (about 1% here, with the
baseline crate's own column moving by 2–3% between runs), so the ordering holds
across the inputs tried, while the 2D competitor table's warning about the
baseline version still applies to that column:

| Search batch | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: |
| `flatgeobuf2d_bench`, seed `0xF6B` (`search_with`) | 485.64 us | 205.62 us | 127.36 us |
| `index2d_bench`, seed `0xB0B` (`search_into_stack` / `search_simd`) | 645.93 us | 215.56 us | 208.98 us |

The `SimdIndex2D` columns are not the same entry point: `search_with` picks a
kernel for the query, while `search_simd` is the explicit wide-4 path, so read
each row against itself rather than down the column.

### `static_aabb2d_index` 2.0.0 vs 2.1.0

`2.1.0` replaced the comparison sort in its build with an in-place MSD radix sort
that stops descending once a range falls inside a single tree node, and it skips
sorting entirely when every Hilbert key is equal. Medians on this crate's
benchmark inputs (uniform random AABBs, `node_size = 16`): the build and `0xB0B`
search rows are over three interleaved runs per version with both binaries pinned
to the same four cores; the `0xF6B` search row is a single pinned run of each
binary, back to back.

| Benchmark | `2.0.0` | `2.1.0` | Change |
| --- | ---: | ---: | ---: |
| Build 1,000 | 16.67 ms | 10.86 ms | -35% |
| Build 100,000 | 5.99 ms | 6.29 ms | +5% |
| Build 1,000,000 | 73.06 ms | 76.20 ms | +4% |
| Search batch, `index2d_bench` seed `0xB0B` | 619.71 us | 605.08 us | -2% |
| Search batch, `flatgeobuf2d_bench` seed `0xF6B` | 319.35 us | 439.54 us | **+38%** |

So the new sort wins clearly at small `n`, where the node-boundary cutoff removes
most of the work, and loses a few percent at 100k and above, where the MSD
partition passes cost more than the old comparison sort on well-spread keys.

Search moves with the dataset rather than uniformly. The `2.1.0` changelog notes
that internal item ordering may differ, and on the `0xF6B` inputs that ordering
costs the baseline 38% on the query batch, while on `0xB0B` it is unchanged
within noise. The other three search rows of that suite move by 3% or less
between the two runs, and downwards (FlatGeobuf 551.80 → 534.07 us, `Index2D`
425.04 → 421.30 us, `SimdIndex2D` 117.30 → 117.43 us), so this is the baseline
crate changing, not the machine.

Duplicate-heavy build inputs
(`build_degenerate` in `index2d_bench`, 100,000 boxes) move in both directions
too: 64 distinct keys goes 4.92 ms → 3.76 ms, all-identical boxes 5.83 ms →
6.55 ms. None of it changes the standing of this crate's build, which stays
2.4–3.4× faster serial across all four of those shapes.

## 2D vs 3D

Lower latency is better. The `3D speed` column is `2D latency / 3D latency`, so
values above `1.00x` mean 3D is faster. The build workload uses 100,000 boxes
with `node_size = 16`; search and KNN use 1,000 query boxes or points.

| Stage | Dataset / mode | `Index2D` | `Index3D` | 3D speed |
| --- | --- | ---: | ---: | ---: |
| Hilbert encode | production 2D LUT vs 3D nibble LUT | 753.19 us | 1.0053 ms | 0.75x |
| Build | planar XY | 2.2526 ms | 3.8785 ms | 0.58x |
| Build | uniform XYZ | 2.3997 ms | 4.0482 ms | 0.59x |
| Search batch | planar XY | 317.39 us | 454.14 us | 0.70x |
| Search batch | uniform XYZ | 322.08 us | 222.61 us | 1.45x |

| KNN batch | Dataset / mode | `Index2D` | `Index3D` | 3D speed |
| --- | --- | ---: | ---: | ---: |
| Top-1 | planar XY | 1.0084 ms | 1.2634 ms | 0.80x |
| Top-10 | planar XY | 1.9223 ms | 2.6466 ms | 0.73x |
| Top-1 | uniform XYZ | 1.0070 ms | 1.7472 ms | 0.58x |
| Top-10 | uniform XYZ | 1.9427 ms | 4.1653 ms | 0.47x |

| Persistence | `Index2D` | `Index3D` | 3D speed |
| --- | ---: | ---: | ---: |
| Serialize built tree (fresh buffer) | 399.11 us | 558.86 us | 0.71x |
| Serialize built tree (reused buffer) | 68.02 us | 96.21 us | 0.71x |
| Load owned tree | 372.24 us | 520.90 us | 0.71x |
| Load zero-copy view | 34.96 us | 36.26 us | 0.96x |

| SIMD persistence | `SimdIndex2D` | `SimdIndex3D` | 3D speed |
| --- | ---: | ---: | ---: |
| Serialize built tree (fresh buffer) | 613.21 us | 894.42 us | 0.69x |
| Serialize built tree (reused buffer) | 160.86 us | 263.74 us | 0.61x |
| Load owned tree | 554.75 us | 939.51 us | 0.59x |

## 3D SIMD

The speed column is scalar/serial latency divided by SIMD/parallel latency, so
values above `1.00x` mean the SIMD or parallel path is faster.

| Stage | Dataset / mode | Baseline | SIMD / parallel | Speed |
| --- | --- | ---: | ---: | ---: |
| Search batch | uniform XYZ | `Index3D` 220.23 us | `SimdIndex3D` 165.02 us | 1.33x |
| Search batch | flat Z | `Index3D` 1.24 ms | `SimdIndex3D` 842.56 us | 1.47x |
| Build `finish_simd` | uniform XYZ, 200k boxes | serial 10.03 ms | parallel 6.98 ms | 1.44x |

## Branch-free node tests

Along a query's boundary a node's per-child overlap test comes out roughly 50/50,
and the branch on its result mispredicts; deeper inside or outside the query it
predicts perfectly. The collect paths therefore fold a node's children into a
`u64` mask — up to 64 tests, no branches — and then walk the set bits, paying one
branch per *hit* instead of one per *child*. That is where the 2D and 3D search
numbers above come from: 25–37% off 2D collect paths on wide queries, 33–52% off
`Index3D`, 30–33% off the zero-copy views, 4–12% off the scalar `Index2DF32` /
`Index3DF32` collect forms, 6–12% off 2D all-hits raycast, and most of the
narrowing of the SIMD indexes' lead on range search. The ray predicate is a slab
test rather than a box overlap, so it sits between the cheap and the expensive
end: 2D gains clearly, 3D lands within drift.

The spatial join is where the mask pays most. `join` expands one node against one
box at a time, and along the other tree's boxes that per-child test is 50/50 far
more often than a range query's is, so its branch was the join's dominant cost:
on 100 000 × 100 000 uniform boxes (extent 1000, unit size, node size 16) `join`
went from 12.6–13.3 ms to 4.5–5.4 ms in 2D and from 36–40 ms to 12.8–13.8 ms in
3D — 2.6–2.8× — with the same pair sets (checked against a brute-force
`count` per item). `join_within` takes the same traversal with the distance
predicate and gains 20–25%. Both predicates come out packed in the shipped
build (the distance kernel's added instructions are `subpd`/`maxpd`/`mulpd`, the
overlap kernel's are `cmplepd`); the distance test is simply several times the
arithmetic per child — a subtraction, a clamp and a square per axis against one
compare per side — so its branch was a smaller share of the whole.

Removing the branch is only half of what happens. A loop that tests every child
into a bitmask has no `continue` in it, and that is what lets the autovectorizer
widen it: in the shipped build (`lto = true`), the collect paths' mask loop
compiles to 4-wide `vcmppd` against mask registers, while the branching loops it
replaced stayed one box at a time. So the collect paths get vector compares out
of a change that reads as a branch-prediction fix, and the callback paths, which
keep their branches for the reason below, keep the scalar loop as well.

Two boundaries on the technique are measured, and both keep it off the other
paths:

- **The callback and early-exit forms keep their branches.** `visit`, `any`,
  `first` and the search iterators leave at the first hit, so the full mask of
  every internal node on the way down is fixed overhead they never recover.
  Measured at +40–60% on `any` and a loss on narrow `visit`.
- **The per-child test has to be cheap.** The saving is one mispredicted branch,
  so a predicate that costs many times that swallows it. Routing the shape-region
  collect paths (convex polygon, frustum) through the same traversal moved
  nothing outside run-to-run drift: a polygon SAT test is six edge normals
  against four box corners, an order of magnitude more work than the branch it
  replaces.

The radius queries sit exactly on the second boundary and split by query width
rather than by form, which is what the next section is about.

## Radius queries: which traversal

`search_within_into` and `count_within` pick between the branching traversal and
the masked one per query, because neither wins everywhere. The two answer
identically — the switch changes only which code produced the answer — so the
whole question is speed.

What the switch reads is the expected hit count: the fraction of the root box
covered by the query grown by `max_distance`, times the item count. Below one
expected hit it takes the branching path, at or above it the masked one.

That threshold is an item count and **not** a covered fraction, which is the part
worth writing down because the first attempt got it wrong. At 100k items a query
covering 1e-6 of the extent lost 25% on the mask; at 1M items the *same fraction*
won 9.5%. Same geometry, opposite sign, so the fraction is not what the crossover
tracks — the hit count is, and a threshold calibrated as a fraction would have
been tuned to one corpus size. Measured on Zen5, one binary, arms behind a
runtime switch read outside the timed loop, the box collect path as a control
(`benches/paired_within.rs`, ratios of masked to branching):

| 2d, 100k items | hits/query | masked / branching |
| --- | ---: | ---: |
| r=1 | 0 | 1.31 |
| r=20 | 2 | 1.03 |
| r=60 | 14 | 0.83 |
| r=400 | 502 | 0.84 |
| r=700 | 1 478 | 0.88 |
| r=1200 | 4 110 | 0.95 |
| r=1800 | 8 691 | 1.01 |
| r=2500 | 15 635 | 1.07 |

3D behaves the same at the narrow end (1.04–1.08 at zero hits, 0.74 at 27 hits)
and, unlike 2D, keeps winning all the way up: 0.71 at 4 853 hits per query.

Two things the table says that the switch does not act on. First, `count_within`
wins with the mask at *every* width — a flat ~16% even in the rows where
`search_within_into` has given the win back — because it has no output to push
and nothing else to be limited by. Second, the 2D collect form degrades once the
output gets very large, and that upper crossover is not predictable: it sits near
8 700 hits per query at 100k items and near 760 at 1M, a tenfold disagreement, so
it is explained by neither an absolute count nor a share of the index. A second
constant fitted to it would be fitted to this machine and this corpus, so there
is none. The cost of leaving it is the bottom two rows — up to 7% on 2D radius
queries that return roughly a sixth of the index — against 12–17% won in the
middle of the range and ~16% on every `count_within`.

The callback forms (`search_within_each`, `search_within_any`) never take the
masked path at any width. They can stop early, and a mask spends its work before
the first hit is reported; the same change measured 40–60% worse on `any`.

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

The window classes also separate the two halves of the traversal. The `large`
and `full extent` rows are almost entirely covered-range copying, so the
branch-free node test above leaves them where they were; the `small` and
sliver/slab rows are all per-child testing, and there the scalar column fell
36–37%, closing the SIMD gap from 2.8× to 1.8× in 2D and from 2.4× to 1.6× in
3D.

| Window (2D) | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: |
| small (10–200) | 224.83 us | 123.01 us |
| large (2,000–5,000) | 6.56 ms | 3.82 ms |
| wide sliver | 1.62 ms | 0.80 ms |
| full extent | 11.42 ms | 11.90 ms |

| Window (3D) | `Index3D` | `SimdIndex3D` |
| --- | ---: | ---: |
| small (50–300) | 258.54 us | 159.40 us |
| large (2,000–5,000) | 10.90 ms | 4.10 ms |
| thin slab | 2.28 ms | 1.33 ms |
| full extent | 12.61 ms | 12.03 ms |

The zero-copy views take this path too: a window that covers a node collects its
leaf range out of the byte buffer instead of parsing and testing each box in it,
so a view's range search scales with window size the way the owned indexes do
rather than with the number of items covered. The gain grows with the tree's
depth, since a deeper tree holds larger fully contained subtrees, and a window
too small to contain any whole node pays a containment test that skips nothing —
the same trade the owned indexes make.

## SIMD search: one mask per node before dispatch

`count_simd_impl` won by building one 64-bit mask for a whole internal node and
only then walking it, instead of gathering, containment-testing and pushing each
hit as its four-lane test came out. `search_simd` is the same kind of path — a
collect with no early exit, so the order is free — and had never been tried on
that shape. It was, behind a const generic so both ran in one binary against the
owned `search_into` as a control (`benches/paired_simd_search.rs`), 100k boxes,
Zen5, four passes. Mask-first against per-4-lane:

| window | 2d | 3d |
| --- | ---: | ---: |
| small | 0.93-0.98 | 1.01-1.03 |
| large (2000..5000) | 0.93-0.95 | 0.93 |
| full extent | 0.95-0.96 | 0.99 |

Large windows are the clear win at -6..-7% in both dimensions, with a 2D
full-extent scan at -4.5%; the small-window and 3D full-extent cells have spreads
wide enough to span 1.0 and are unresolved rather than losses. No cell measured a
real regression, so the shape ships everywhere and `search_shape::<0>` is kept
hidden so the comparison can be re-run.

`visit_simd_impl` is deliberately excluded. Its visitor may break, so a mask
spends work that an early exit then discards, and the same rewrite measured 11%
slower there; the callback family's mask experiments are at +40-60% elsewhere on
this page. The AVX2 and AVX-512 tiers are untouched — the AVX2 tier already folds
its containment test into vector lanes (`cbits`), which is the other half of the
mechanism.

## Profiling the two count/aggregate open questions

The paired benches left two questions open: why the scalar `Index2D::count`
still counts large windows ~1.2x faster than `count_simd_impl`, and why
`aggregate` on a ~10k-hit window costs ~3x `count` when cache misses do not
explain it. Both were profiled with `callgrind` (instruction, cache and branch
simulation, WSL2 — there is no valgrind on Windows) plus live `perf` cycle
sampling, using `examples/callgrind_count_agg.rs`: one arm per process with
`--collect-atstart=no --toggle-collect` around the query loop, the index build
excluded by construction, and a parity check first (`owned` and `simd` must
print the same count). Both gaps reproduce on the profiling platform
(1.24x and 2.0x), so the profiles answer the question that was asked.

**The SIMD count kernel loses on dispatch, not on work.** For 200 windows of
side 2000–5000 (100k boxes): the SIMD kernel executes *fewer* instructions than
the scalar one (12.2M vs 14.9M Ir for the whole query loop) and takes
essentially *no* last-level misses against the scalar kernel's 31k — yet burns
55% of the alternating run's cycles against the scalar kernel's 45%, which is
the 1.2x. `llvm-mca` puts the loop bodies at parity (8.0 vs 8.2 cycles per four
boxes), so the loss is not inside any loop either: it is the per-node
dispatch structure. The scalar kernel builds one 64-bit overlap mask for the
whole 16-entry node (autovectorized, independent lanes) and only then walks the
mask to push cut children; `count_simd_impl` interleaved test → bitmask →
gather/containment/push four times per node, putting the serial
next-child-dependent chain on the critical path four times as often.

The fix — build a chunk's 64-entry mask first, then walk it (the owned
kernel's shape) — was measured as two binaries alternated A B A B, three
passes, pinned core, on a quiet machine, figure = kernel/scalar-count ratio
per pass (the `count` control's own level tracks the binary-to-binary layout
bias, which moved up to 9% between builds — the per-run ratio is the only
figure worth reading across builds):

| window | before | after |
| --- | ---: | ---: |
| 2d small | 0.91 | 0.84 |
| 2d large | 1.25 | 1.05 |
| 3d small | 0.75 | 0.79 |
| 3d large | 1.26 | 1.07 |

Cross-checked on Windows/Zen5 with the one-binary harness
(`benches/paired_simd_count.rs`, ratio to the old visit-closure
implementation, so no build-to-build term), three dispatch shapes behind a
const generic (`probe/simd-count-shapes`, two passes of 12–15 rounds):

| window | interleaved (before) | chunk loop | whole-node mask |
| --- | ---: | ---: | ---: |
| 2d small | 0.71–0.76 | 0.72–0.76 | **0.64–0.67** |
| 2d large | 0.73–0.75 | 0.66–0.68 | **0.62–0.64** |
| 3d small | 1.04–1.14 | 1.08–1.20 | **1.03–1.13** |
| 3d large | 0.80–0.84 | **0.65–0.68** | 0.69–0.72 |

The chunk loop that the WSL2 measurement landed cost 3d small ~6% over the
interleaved kernel; building the mask for the whole node when it has at most
64 children (every default configuration — the chunk loop remains for wider
nodes) keeps the large-window win, takes 2d small a further 10%, and returns
3d small to the interleaved level. Final kernel against the old visit-based
count: 2d small 0.64, 2d large 0.60 (scalar count 0.59), 3d small 1.04
[0.96..1.07], 3d large 0.64 (scalar 0.61). Two
intermediate shapes were measured and dropped: a per-batch scratch-array of
masks (large-window win halved, small-window cost the same) and merging two
four-lane batches per dispatch round (worse everywhere — the second batch's
loads in flight cost more than the saved dispatch rounds). The
remaining 3–5% on large windows is the SoA kernel's per-node bookkeeping
(column bounds checks and the packed-frame push) that the owned kernel's
contiguous entries do not pay.

**`aggregate` pays for per-item folds on the window's cut leaves.** For 200
windows of side 100k (1M boxes, ~10k hits each): `aggregate` executes only 33%
more instructions than `count` (17.1M vs 12.8M Ir) but takes 2.02x the cycles
(perf split 65%/32%), reproducing the wall-clock ratio. The +33% instructions
are the aggregate kernel testing children through the generic per-child
closure (`overlap_mask_at`) where the count kernel autovectorizes a 16-entry
mask over contiguous entries. The cycles are elsewhere: only ~840 of the ~10k
hits per window are cut-leaf items folded one by one (`Fold::item`) — the rest
come from contained-subtree summaries — but those folds form a loop-carried
FP chain (`sum += v`, a serial `addsd`) reading the scalar and mask columns,
and ~40% of the kernel's samples sit on that chain's accumulator spill and
`addsd`. The cache simulation agrees (264k vs 183k last-level misses), and
explains why the per-node record-layout experiment did not move the ratio:
the misses are per-*item* column reads on cut leaves, not per-node summary
reads.

The obvious fix for that chain — splitting the fold over several sum/min/max
lanes, collapsed at the end — was built and measured, and it lost. Two
binaries alternated on WSL2 read a 4% gain (1.62 → 1.55 at ~100 hits), which
is inside the up-to-9% build-to-build band that comparison carries; putting
the lane count behind a const generic and running 1, 2 and 4 lanes in *one*
binary on Windows/Zen5 (`probe/aggr-fold-lanes`, `aggregate`/`count` per
round, two passes of 12 rounds) gave:

| window | 1 lane | 2 lanes | 4 lanes |
| --- | ---: | ---: | ---: |
| ~0 hits | 1.03 | 1.02–1.04 | 1.07–1.08 |
| ~100 hits | 1.44–1.45 | 1.63–1.65 | 1.67–1.69 |
| ~10k hits | 2.66–2.74 | 3.11–3.16 | 3.00–3.28 |

One lane wins everywhere; four lanes cost 16% at ~100 hits. The profile's
mechanism (samples on the accumulator chain) was real and still did not
convert into time — the chain is not what the out-of-order core is waiting
on, the column reads are. The single accumulator ships. The Windows ratio is
2.9–3.1x against 2.0–2.5x on WSL2; the mechanism is the same, the host's
relative memory latency amplifies it.

So the next suspect was the column reads themselves: a cut leaf gathers one
scattered load per set bit of its hit mask, and its items are contiguous, so
reading the whole range straight through and folding branch-free under the mask
should have turned a gather into a stream. It lost too, and by more than the
lane split did — one binary, both shapes behind a const generic, two passes
(`probe/aggr-leaf-scan`, `aggregate`/`count`):

| window | gather per hit (ships) | scan the leaf |
| --- | ---: | ---: |
| ~0 hits | 1.03–1.04 | 1.20–1.21 |
| ~100 hits | 1.50–1.53 | 2.45–2.48 |
| ~10k hits | 2.46–2.47 | 2.63–2.81 |

The premise was wrong in a way worth keeping: at `node_size` 16 a cut leaf holds
at most 16 items, so there is no stream long enough to amortize anything, and
the scan only pays for the items the mask drops. Both attempts on this gap have
now failed for the same underlying reason — the profile names a mechanism, and
the mechanism is not what the core is waiting on.

What the same runs do show is that the gap is not a defect to fix. Against the
workaround `aggregate` replaces — `search` the window and fold the hits — it is
**1.4x faster** at ~10k hits (2.46 vs 3.35–3.45 times `count`) and **1.6x** at
~100 hits (1.50 vs 2.46). `count` is simply a much cheaper question: it reads no
per-item columns at all. Treat `aggregate`/`count` as the price of the two extra
columns rather than as headroom; the line is closed unless the format grows a
way to answer from summaries alone.

## Overlapping boxes

An R-tree prunes by node bounding box, so the usual worry is that data packed
densely into one space makes the node boxes overlap too, and a query then has to
open subtrees holding nothing for it. This measures whether that happens here.

The separating metric is **checks per hit**, from the `search_visited(query) ->
(hits, intersection_checks)` diagnostic. If pruning were failing, checks per hit
would rise as the field gets denser. If the extra cost is simply that the answer
is bigger, checks per hit stays flat or falls.

Workload: 100,000 boxes over a 1,000-wide square, 2,000 queries per row. "Boxes
over a point" is the expected number of boxes covering any one point, which is
what "denser" means here. The counters are exact and reproduce byte for byte
between runs; only the last column is timed.

Fixed 5-wide query window:

| box side | boxes over a point | hits/query | checks/query | checks/hit | µs/query |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.03 | 3.1 | 107 | 35.1 | 0.21 |
| 2 | 0.4 | 4.9 | 115 | 23.2 | 0.22 |
| 5 | 2.5 | 10.1 | 131 | 13.1 | 0.27 |
| 8 | 6.4 | 16.9 | 150 | 8.9 | 0.34 |
| 20 | 40 | 61.9 | 237 | 3.8 | 0.51 |
| 60 | 360 | 398.4 | 731 | 1.8 | 1.46 |

A point query is the cleaner probe, because its answer is exactly the boxes
covering the point, so the answer grows with density by definition and anything
left in checks per hit is the tree:

| box side | boxes over a point | hits/query | checks/query | checks/hit |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.03 | 0.03 | 87 | — |
| 2 | 0.4 | 0.39 | 93 | — |
| 5 | 2.5 | 2.44 | 106 | 43.4 |
| 8 | 6.4 | 6.25 | 120 | 19.1 |
| 20 | 40 | 39.1 | 196 | 5.0 |
| 60 | 360 | 338.9 | 650 | 1.9 |

**Pruning holds.** Checks per hit falls throughout, and the absolute work grows
far slower than the answer: across a 12,000× rise in density the point query goes
from 87 checks to 650, a factor of 7.4, while its answer goes from 0.03 items to
339. Subtract the hits, which any index must touch, and the overhead above the
answer grows from about 87 to about 310 — under 4× for four orders of magnitude
of density.

So a query over heavily overlapping boxes costs more because the answer is
bigger, not because the tree stopped pruning. That also explains the one kNN
figure that looks alarming in isolation: `neighbors(point, 1)` on such a field
takes about 4 µs instead of 1.3, because with hundreds of boxes covering the
point there are hundreds of equally near answers to settle between, not because
the descent got worse.

This says nothing about **clustered** data, which is a different shape: there the
boxes are small but the empty space between groups is large, and node boxes span
the gaps. That case is the [raycast comparison against `bvh`](#closest-hit-raycast-vs-the-bvh-crate)
below, where a surface-area-heuristic build wins and Hilbert packing does not.

## Ordered front-to-back region queries

`search_ordered` answers the same set as `search` but emits it in nondecreasing
order of a key — `view_depth_3d` for a view direction, making a frustum query
front-to-back. The control that matters is not `search`; it is what a caller
writes today to get that order: search, key each hit once, sort the pairs.
Workload: 1,000,000 uniform boxes over a 10,000-wide space, one slab frustum
holding ~160k of them, `node_size` 16, pinned to one core. Lower is better.

| Query | Time | vs. search + sort |
| --- | ---: | ---: |
| `search` alone, unordered (reference) | 505 us | — |
| `search` + key + sort (the workaround) | 5.87 ms | 1.0x |
| `search_ordered`, whole result | 10.30 ms | **0.57x** |
| `search_ordered`, nearest 100 | 31.6 us | **186x** |
| `search_ordered`, nearest 1,000 | 96.4 us | **61x** |
| `search_ordered`, nearest 10,000 | 571 us | **10x** |

Ordering the *whole* result loses, and by enough to state plainly: a heap over
every hit against a depth-first sweep that can emit a wholly contained subtree
untested, plus one sort. What wins is the budget — by one to two orders of
magnitude, because `max_results` (and the `max_key` cutoff, and a
`ControlFlow::Break`) end the traversal rather than filter its output. The
nearest 100 cost a sixteenth of even the unordered `search`, which has to visit
everything the frustum touches.

So the selector is not "ordered or unordered" but "does something stop this
query": a render budget, a z-prepass, an occlusion loop, a "closest few" probe
take `search_ordered`; "every hit, ordered" takes `search` and `sort`.

The descent is scalar on every frontend — a heap pops one node at a time, so the
SIMD kernels have nothing to widen. The numbers above therefore also describe
`SimdIndex3D` and the `f32` frontends, which carry the query for availability,
not for speed.

## Closest-hit raycast vs the `bvh` crate

Closest-hit raycast over the packed index against the
[`bvh`](https://crates.io/crates/bvh) crate (100k boxes, 1,000 rays of length
4,000). For closest hit, "BVH" is a fair hand-rolled ordered traversal over its
SAH tree; for all hits, its broad-phase `traverse_iterator`.

| metric | packed SoA/SIMD | BVH |
|---|---:|---:|
| build (uniform) | **4 ms** | 31 ms |
| closest hit, uniform | **0.72 ms** | 1.6 ms |
| closest hit, clustered | 58 µs | **27 µs** |
| all hits, uniform | **0.5 ms** | 1.5 ms |
| all hits, clustered | 53 µs | **41 µs** |

The packed Hilbert tree builds ~7x faster. All-hits has no early-exit, so the
SIMD slab test wins on uniform scenes but is edged out on heavily clustered ones;
for closest hit a SAH BVH builds a structurally better tree and wins on clustered
scenes. Reproduce with `cargo bench --bench raycast3d_bench --features simd`.

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
| `f64` `Triangle3D` (scalar) | 121.6 ms | 7.2 ns |
| `f32` `Triangle3DF32` (SIMD) | 47.8 ms | 2.8 ns |

The `f32` SIMD kernel runs ~2.5x faster than scalar `f64` here. Most of that is
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

- **Exact answers, most hits, fastest KNN** — `SimdIndex2D`. 32-byte `f64`
  boxes; nothing else is faster on an exact range query with many hits.
- **Fastest range search, and the smallest** — `SimdIndex2DF32::search`.
  16-byte outward-rounded `f32` boxes, 16 to a SIMD chunk against `f64`'s 8, so
  on AVX-512 it beats even `SimdIndex2D` on the range rows above. It wins on
  speed *and* memory rather than trading one for the other; the price is a few
  near-boundary false positives. Its hits match the scalar `Index2DF32`
  exactly — both round the query inward onto the same grid.
- **Compact storage, exact answers, few hits** — `SimdIndex2DF32::*_exact`, the
  `f32` index plus your own `f64` boxes. Exact KNN works here too, but plain
  `f64` is faster at it in these runs.
- **Half the memory without a SIMD dependency** — `Index2DF32` / `Index3DF32`,
  the same 16/24-byte boxes with no `simd` feature, and the file that
  `StreamIndex2DF32` / `StreamIndex3DF32` streams at half the box bytes over the
  wire. Same hits as `SimdIndex2DF32`, plus `search_exact`.

  Its collect forms take the branch-free node test too: `search` fell 4-12% and
  `count` 6-8% on 100 000 boxes, with the early-exit forms flat, so the ordering
  below is unchanged.

  It is the memory choice, not the speed one: on a 1M-box spot check range
  queries ran ~30% slower than `Index3D`, `search_exact` ~45% slower, and the
  build ~1.7x slower from the rounding. The gap is not per-node widening — the
  query is rounded onto the `f32` grid once, so each node compares `f32` to
  `f32` and the hits are bit-identical to the `f64` test — it is the extra
  conservative candidates the outward-rounded boxes admit.

| Range query | Items | `f64` exact | `f32` rounded | `f32` exact |
| --- | ---: | ---: | ---: | ---: |
| small query boxes | 10k | 89 us | 72 us | 78 us |
| small query boxes | 100k | 123 us | 92 us | 102 us |
| small query boxes | 1M | 163 us | 128 us | 146 us |
| large query boxes | 10k | 131 us | 110 us | 298 us |
| large query boxes | 100k | 561 us | 456 us | 1.84 ms |
| large query boxes | 1M | 5.14 ms | 3.52 ms | 17.35 ms |

The `f64 exact` and `f32 rounded` columns use the compress-store collection on
AVX-512, which roughly halves the large-window rows versus the scalar collection;
`f32 exact` runs the per-item refinement callback (no compress) and is unchanged.

| KNN workload | `f64` exact | `f32` rounded | `f32` exact |
| --- | ---: | ---: | ---: |
| 10k items | 218 us | 233 us | 374 us |
| 100k items | 337 us | 349 us | 493 us |

## Summary

- `Index2D` is the general-purpose path;
- `SimdIndex2D` and `SimdIndex3D` are best for heavier query batches where SIMD
  work amortizes well;
- scalar `Index2D` search leads `static_aabb2d_index` by 2.4–3.0× on both
  generated inputs, and `Index2D` build is faster as well;
- `Index3D` build and KNN are still slower than `Index2D`, but uniform 3D search
  is faster when Z meaningfully prunes the tree;
- the SIMD indexes' lead over the scalar ones on range search narrowed to
  1.3–1.5× once the scalar collect paths stopped branching per child; the
  scalar path is now the one to beat on sparse queries too;
- the branch-free node test behind those collect numbers applies only where the
  path has no early exit *and* the per-child predicate is cheap; the callback
  forms, the shape regions and the radius queries measured worse with it and
  keep their branching traversal;
- f32 storage halves box memory; exact callbacks trade source-box lookup for
  exact results;
- SIMD persistence uses the same canonical bytes as scalar persistence; it pays
  an SoA gather/scatter cost but avoids a second file format;
- `any` is often much faster than collecting full result sets when all you need
  is existence;
- `search_ordered` is for stopping early, not for ordering: a budgeted
  front-to-back frustum query beats search-then-sort by 10-186x, while ordering
  the whole result is 1.8x slower than sorting it;
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
  comparisons, node sizes, a hidden Morton baseline, and the ordered
  front-to-back frustum query against its search-then-sort control;
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
[left-pack](internals/simd.md) emulation (~1.3–1.6× over the SSE2 fallback on
AVX2-only CPUs), and SSE2 is the floor. So these kernels do **not** need
`target-cpu` to pick the right width. The flag's remaining benefit is widening
the **scalar** autovectorized loops (~1.1–1.3×). (The WASM demo passes
`-Ctarget-feature=+simd128` for the same reason.)

### Dials that were measured and left alone

A consumer picks the profile, so these are reported rather than set. All on
100 000 boxes, 1 000 `count` queries, `lto = true` and `codegen-units = 1` held
constant except where they are the subject.

- **`opt-level`: keep the release default of 3.** Native: `2` costs 1–4%, `1`
  costs 37–42%, `"s"` 30–34%, `"z"` 53–62%. The size levels turn off
  autovectorization, which is what the collect paths are built around, so they
  cost far more here than the usual rule of thumb suggests.
- **PGO makes this slower, not faster.** Instrumented build, profile collected
  on the very workload then measured — PGO's best case — and it lost by 7% on
  wide queries and 13% on narrow, consistently across interleaved runs. It is
  not undoing the vectorization (the profiled build has *more* packed compares,
  42 against 18); it unrolls an inner loop that runs at most `node_size`
  iterations and is already branch-free and vectorized, so there is nothing for
  a profile to discover and the larger loop body costs. PGO earns its keep on
  branchy code; this traversal stopped being branchy.
- **`panic = "abort"`: a size dial, not a speed one.** −6.4% binary, with the
  speed difference inside the ~7% layout noise between two separately compiled
  binaries. It is also the wrong default for a server: a panicking request
  takes the process with it instead of unwinding.
- **`strip = true` does nothing on MSVC** — byte-identical binary, because the
  debug info is in a separate `.pdb` already. It is an ELF dial.

Independently of width, range search and all-hits raycast **prefetch the next
tree node** while the current one is tested — a free latency hint worth ~3–5% on
range and ~5–12% on heavy raycast traversal. See
[internals/prefetch.md](internals/prefetch.md).

## Reproducing

```bash
cargo bench --bench hilbert2d_bench --features bench-internals
cargo bench --bench index2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench index3d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench persistence_knn2d_bench --no-default-features --features simd,bench-internals
cargo bench --bench persistence_knn3d_bench --no-default-features --features simd,bench-internals
cargo bench --bench flatgeobuf2d_bench --no-default-features --features parallel,simd,bench-internals
cargo bench --bench coord_precision --no-default-features --features f32-storage,simd
cargo bench --bench raycast3d_bench --features simd
cargo bench --bench raytriangle3d_bench --features simd
```

For low-noise numbers, set `BENCH_PIN_CORE=<n>` to pin the measuring thread to one
logical core — a fast performance core (on a hybrid CPU, avoid the efficiency
cores; check your CPU's topology for which logical IDs are performance cores). It
is read at startup and is a no-op when unset:

```bash
BENCH_PIN_CORE=8 cargo bench --bench index2d_bench --features parallel,simd,bench-internals
```

On Linux the self-pin is a no-op; pin from the OS instead, e.g.
`taskset -c 8 cargo bench …`.
