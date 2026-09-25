# Performance

Benchmark results and how to reproduce them. See the
[README](https://github.com/Filyus/packed_spatial_index#readme) for the API
overview.

Numbers are from one machine unless a section names another: a Zen 5 laptop (a
Ryzen AI 7 350). Its AVX-512 runs on 256-bit datapaths, as Zen 4's does; a
server Zen 5 with full-width ones is measured where a section names it. They
depend on hardware and workload, so treat them as relative, not absolute; how
far they move between processors is measured in
[the four frontends, by CPU](#the-four-range-search-frontends-by-cpu).
Single-thread rows are measured with the benchmark thread
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

Lower is better. The workload is 100,000 random AABBs (sides `0.1..20` in a
10,000 square) with `node_size = 16` for every participant.

Queries come in three classes by output: small windows (sides `10..200`, 13
hits a query on average), mid (`200..1000`, 380 hits) and large
(`2000..5000`, about 12,700 hits). A class gets 10,000, 2000 or 400 windows;
an early exit gets 10,000 in every class. A set that repeats every
iteration and is small enough lets the branch predictor learn each query's
path through the tree, which flatters branchy traversals on a Zen 4 or Zen 5
(see [Branch-free node tests](#branch-free-node-tests)); these sets are too
large for that. Every participant runs the same set.

Search, `visit` and `first` below are
[`paired_competitors`](#reproducing): all participants in one binary,
interleaved, 15 rounds, as the median per-round time relative to
`static_aabb2d_index` (1.00), averaged over two or three runs per machine.
Runs of one machine agree within 0.07 and mostly within 0.02. The machines:
a cloud Xeon VM (Emerald Rapids, family 6 model 207), a Zen 3 (EPYC 7763), a
Zen 4 (EPYC 9V74) with AVX-512 exposed, the same Zen 4 in VMs that hide it and
a Neoverse N2. The last four are GitHub-hosted runners.

**Search batch** (collect every hit). The reference is `static_aabb2d_index`'s
`query_with_stack`, its collecting call, which returns a fresh `Vec` per query;
FlatGeobuf's `PackedRTree::search` does the same. This crate's rows are
`search_with`, which reuses a workspace. Driving `static_aabb2d_index`'s
`visit_query_with_stack` into a reused `Vec` instead costs it 0.77–0.95 of
the reference, so a caller who reuses buffers there narrows the gap by that
much.

| Windows | Participant | Xeon (EMR) | Zen 3 | Zen 4 | Zen 4, no AVX-512 | N2 |
|---|---|---:|---:|---:|---:|---:|
| small | FlatGeobuf | 1.15 | 1.18 | 1.22 | 1.22 | 1.08 |
| small | `Index2D` | 0.67 | 0.58 | 0.56 | 0.55 | 0.70 |
| small | `SimdIndex2D` | 0.51 | 0.48 | 0.38 | 0.47 | 0.68 |
| mid | FlatGeobuf | 1.00 | 0.99 | 1.07 | 1.07 | 0.98 |
| mid | `Index2D` | 0.59 | 0.54 | 0.54 | 0.54 | 0.69 |
| mid | `SimdIndex2D` | 0.40 | 0.48 | 0.34 | 0.49 | 0.72 |
| large | FlatGeobuf | 1.12 | 0.97 | 1.17 | 1.17 | 0.92 |
| large | `Index2D` | 0.34 | 0.30 | 0.33 | 0.33 | 0.39 |
| large | `SimdIndex2D` | 0.28 | 0.27 | 0.25 | 0.30 | 0.44 |

So scalar `Index2D` collects 1.4–1.8× faster than `static_aabb2d_index` on
small windows, 1.4–1.9× on mid ones and 2.6–3.3× on large ones. The lead grows
with the output because the collect paths fold a node's child tests into a
bitmask and branch once per hit (see
[Branch-free node tests](#branch-free-node-tests)), while the baseline branches
once per child. FlatGeobuf's search sits within 0.92–1.22 of the baseline.

**Full traversal with a callback** (`visit`). The reference is
`static_aabb2d_index`'s `visit_query_with_stack` with a unit visitor, which
skips its break test; this crate's `visit` gets a closure returning
`ControlFlow::Continue`. FlatGeobuf's `PackedRTree` has no callback query.

| Windows | Participant | Xeon (EMR) | Zen 3 | Zen 4 | Zen 4, no AVX-512 | N2 |
|---|---|---:|---:|---:|---:|---:|
| small | `Index2D` | 0.83 | 0.76 | 0.69 | 0.68 | 0.88 |
| small | `SimdIndex2D` | 0.72 | 0.68 | 0.62 | 0.62 | 0.98 |
| mid | `Index2D` | 0.72 | 0.66 | 0.60 | 0.60 | 0.80 |
| mid | `SimdIndex2D` | 0.87 | 0.69 | 0.64 | 0.66 | 0.96 |
| large | `Index2D` | 0.43 | 0.40 | 0.40 | 0.40 | 0.44 |
| large | `SimdIndex2D` | 0.51 | 0.42 | 0.42 | 0.43 | 0.54 |

`visit` leads by less than collecting does on small windows (1.1–1.5×) and by
2.3–2.5× on large ones, where a covered subtree reaches the callback as a whole
leaf range without per-item tests.

**Early exit** (`first`; `any` runs the same traversal and lands within 0.02 of
it). The reference is `visit_query_with_stack` with a visitor that breaks on
the first hit.

| Windows | Participant | Xeon (EMR) | Zen 3 | Zen 4 | Zen 4, no AVX-512 | N2 |
|---|---|---:|---:|---:|---:|---:|
| small | `Index2D` | 0.66 | 0.71 | 0.67 | 0.66 | 0.67 |
| small | `SimdIndex2D` | 0.92 | 0.78 | 0.74 | 0.71 | 1.07 |
| mid | `Index2D` | 0.64 | 0.70 | 0.65 | 0.65 | 0.68 |
| mid | `SimdIndex2D` | 0.88 | 0.94 | 0.86 | 0.85 | 1.36 |
| large | `Index2D` | 0.65 | 0.72 | 0.69 | 0.69 | 0.75 |
| large | `SimdIndex2D` | 0.99 | 1.06 | 1.12 | 1.04 | 1.67 |

`Index2D::first` takes 0.64–0.75 of the baseline's time on every window class
and machine. Until 0.33 it lost on large windows (1.22–1.35) and was level on
mid ones: its early exit tested every child of each node on the path, where
the baseline also tests them all but descends into the partial tail node of
each level. The descent that replaced it stops testing at the first
overlapping child (see [Early exit: which descent](#early-exit-which-descent)).
The `Index2D` rows are from that change's runs (the Xeon three, the Zen 3 three,
the Zen 4 one with AVX-512 exposed and two without, the N2 two). The
`SimdIndex2D` rows are from the runs above: its early exit is a separate
traversal the change did not touch. It lands at
0.99–1.12 of the baseline on large windows on x86 and is slower on the N2
(1.36 mid, 1.67 large), where its NEON path has no movemask.

Build and persistence were measured with Criterion (`flatgeobuf2d_bench`) on
the Zen 5 laptop, with persistence on the canonical byte format for the same
100,000 boxes:

| Benchmark | FlatGeobuf | `static_aabb2d_index` | `Index2D` | `SimdIndex2D` |
| --- | ---: | ---: | ---: | ---: |
| Full build | 46.82 ms | 6.31 ms | 2.23 ms serial / 1.73 ms parallel | - |
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
`Index3D`, 30–33% off the zero-copy views, 6–12% off 2D all-hits raycast, and
most of the narrowing of the SIMD indexes' lead on range search. The ray
predicate is a slab test rather than a box overlap, so it sits between the cheap
and the expensive end: 2D gains clearly, 3D lands within drift.

The scalar `Index2DF32` / `Index3DF32` collect forms read a node's children
from their `f32` columns sliced once per chunk and test them with the four (six)
comparisons joined by `&`, so the mask loop vectorizes as the `f64` one does.
Timed on its own (`benches/paired_mask_forms.rs`, masked / branching, small /
mid / large windows), the mask then pays everywhere measured. That holds on
aarch64 too, where the `f64` 2D paths go without it:

| machine | 2D f32 | 3D f32 |
| --- | --- | --- |
| Zen 5 laptop (Ryzen AI 7 350) | 0.60 / 0.70 / 0.82 | 0.44 / 0.48 / 0.61 |
| Zen 5 server (EPYC 9V45) | 0.54 / 0.64 / 0.74 | 0.40 / 0.45 / 0.55 |
| Zen 4 (EPYC 9V74) | 0.51 / 0.60 / 0.73 | 0.41 / 0.43 / 0.54 |
| Zen 3 (EPYC 7763) | 0.56 / 0.66 / 0.77 | 0.45 / 0.49 / 0.59 |
| Intel Xeon (family 6 model 207), cloud VM | 0.50 / 0.75 / 0.92 | 0.40 / 0.49 / 0.70 |
| Neoverse N2 | 0.87 / 0.85 / 0.92 | 0.85 / 0.79 / 0.72 |

While the child test read each box through four bounds-checked lookups joined
by `&&` it stayed scalar; the f32 mask was then within a few percent of the
branch on x86 and lost 2–10% in 2D on the N2.
These forms skip a subtree the window covers, like the `f64` indexes: its leaf
range goes to the output as one slice, without a test per item.

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
of a change that reads as a branch-prediction fix. The callback paths, owned
and view, take the same mask; the search iterators keep their branches for
the reason below, so they keep the scalar loop as well.

Two boundaries on the technique are measured, and both keep it off the other
paths:

- **The search iterators keep their branches.** An iterator yields one item
  per call, so it cannot drain a mask in one pass: masked it loses 7–14%.
  `visit` takes the mask in 2D and 3D, owned and view. It also hands a covered
  subtree's leaf range to the callback whole, in both forms. Masked /
  branching on a Zen 3 (EPYC 7763), small, mid and large windows: 0.63, 0.77,
  0.90 owned and 0.62, 0.74, 0.84 on the view in 2D, 0.52, 0.51, 0.68 and 0.52,
  0.49, 0.64 in 3D. Those runs put both arms behind one closure and a `match`,
  which [read up to 25% off](#early-exit-which-descent) on one arm. `any` and
  `first` have their own descent, which tests children into a mask only for
  windows that expect almost nothing (see
  [Early exit: which descent](#early-exit-which-descent)).
- **The per-child test has to be cheap.** The saving is one mispredicted branch,
  so a predicate that costs many times that swallows it. Routing the shape-region
  collect paths (convex polygon, frustum) through the same traversal moved
  nothing outside run-to-run drift: a polygon SAT test is six edge normals
  against four box corners, an order of magnitude more work than the branch it
  replaces.
- **The target has to make a mask cheap.** Everything above was measured on x86.
  On aarch64 (a Neoverse N2, `benches/paired_mask_forms.rs` through the
  `bench-arm.yml` workflow) the 2D mask loses to the per-child branch on every
  window: masked / branching 1.10, 1.03, 1.02 on owned `search_into` (small,
  mid, large windows) and 1.15, 1.06, 1.04 on the view, against 0.64–0.92 on
  every x86 machine measured (Zen 3, Zen 4, both Zen 5s).
  So the `f64` 2D paths build the mask only off aarch64 (`MASK_PAYS_IN_2D`).
  NEON has no movemask; a 2D box test is cheap enough for building the bit
  mask to cost more than the mispredicts it saves. That is the likely reason
  rather than a measured one. The 3D collect paths keep the mask on aarch64:
  the N2 gives 0.88 on mid and large view windows and 0.97–0.99 on raycast.
  The 3D callback paths do not (`CALLBACK_MASK_PAYS_IN_3D`): the N2 measured
  1.06–1.17 on small-window `visit`, a win only on large-window `visit`
  (0.88–0.91); `any` / `first` branch there in both dimensions. The scalar `Index2DF32` keeps the mask as well: with its
  vectorized child test it wins on the N2 too (see above).
- **The bench has to show the predictor queries it cannot learn.** Every rep
  replays the same query set. A Zen 4 or Zen 5 predictor learns each query's
  traversal once the set's hard-to-predict branches fit its tables: about
  30 000 on Zen 5 per Lemire's measurement, against about 70 per small window
  (callgrind). Over a replayed set of 400 small windows the branching form then
  looked faster than it is; the mask seemed to lose 23–71% on Zen 4 and
  Zen 5; over 10 000 windows it wins there as everywhere, 0.53–0.75. So the
  paired benches size each query set by its output: 10 000 queries for small
  ones, fewer where one query does too much work to be learned.

The radius queries sit exactly on the second boundary and split by query width
rather than by form, which is what the next section is about.

## Radius queries: which traversal

`search_within_into` and `count_within` pick between the branching traversal and
the masked one per query, because neither wins everywhere. The two answer
identically — the switch changes only which code produced the answer — so the
whole question is speed.

What the 2D switch reads is the expected hit count: the fraction of the root box
covered by the query grown by `max_distance`, times the item count. Below one
expected hit it takes the branching path, at or above it the masked one. The 3D
switch reads it only on aarch64; elsewhere a 3D radius query always takes the
mask (below).

That threshold is an item count and **not** a covered fraction, which is the part
worth writing down because the first attempt got it wrong. At 100k items a query
covering 1e-6 of the extent lost 25% on the mask; at 1M items the *same fraction*
won 9.5%. Same geometry, opposite sign, so the fraction is not what the crossover
tracks — the hit count is, and a threshold calibrated as a fraction would have
been tuned to one corpus size. One binary, the arms behind a runtime switch read
outside the timed loop, the box collect path as a control, query sets sized so
the predictor cannot learn them (`benches/paired_within.rs`, masked /
branching on `search_within_into`, 100k items):

| 2d radius | hits/query | Zen 5 laptop | Zen 5 server | Zen 4 | Zen 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| r=1 | 0 | 1.07 | 1.09 | 1.08 | 1.08 |
| r=20 | 2 | 0.99 | 1.00 | 0.99 | 1.01 |
| r=60 | 14 | 0.86 | 0.85 | 0.83 | 0.88 |
| r=150 | 76 | 0.81 | 0.80 | 0.77 | 0.84 |
| r=400 | 502 | 0.82 | 0.84 | 0.81 | 0.87 |
| r=1000 | 2 916 | 0.85 | 0.97 | 0.93 | 0.99 |
| r=2500 | 15 635 | 0.89 | 1.19 | 1.13 | 1.16 |

The switch sits where every machine crosses: a query with no expected hit
loses 7–9% with the mask, one with two breaks even. In 3D the mask wins from
the narrowest radius on (0.90–0.95 at zero hits, 0.72–0.78 at 27, 0.73–0.81 at
4 853 hits per query), so off aarch64 the 3D switch no longer applies the
threshold: it gave those 5–10% away on empty queries.

**On aarch64 the 2D switch always takes the branching path.** The same harness on
a Neoverse N2 (GitHub's `ubuntu-24.04-arm` runner, the `bench-arm.yml` workflow)
found the 2D mask slower at every radius — masked / branching 1.40 at zero hits,
1.34 at 2, 1.25 at 14, 1.17 at 502 and still 1.06 at 15 635 — where every x86
machine wins with it from about 14 hits. NEON has no movemask; in 2D the box
test is cheap enough that building the bit mask costs more than the mispredicts
it saves; that is the likely reason, not a measured one. 3D keeps the mask
there too: on the N2 it gives 0–2% back with no hits and wins 7–10% from a few
dozen hits up. That is the one place the 3D threshold still acts.

Two things the table says that the switch does not act on. First, `count_within`
keeps winning with the mask as the output grows — 0.46–0.55 on x86 at 15 635
hits per query, where `search_within_into` has given the win back — because it
has no output to push and nothing else to be limited by. Second, the 2D collect
form degrades once the output gets very large. That upper crossover depends
on the machine: the servers lose 13–19% at 15 635 hits per query, the Zen 5
laptop still wins 11% there. An earlier laptop measurement put it near 8 700
hits at 100k items and near 760 at 1M. A second constant fitted to it would be
fitted to one machine and one corpus, so there is none. The cost of leaving it
is the bottom rows — up to 19% on 2D radius queries that return roughly a
sixth of the index — against 12–23% won in the middle of the range.

Which arm the switch picks is pinned by unit tests on the predicate itself
rather than read off this table, and the two traversals are reachable
individually (`search_within_into_forced::<MASKED>`, hidden) so the threshold
can be re-calibrated on another machine. Note that the shipping entry point is
deliberately not a fourth arm in that bench: it reaches the same two bodies
through a different function, so timing it against them would measure an
inlining difference and reads as a regression that is not there.

The callback forms (`search_within_each`, `search_within_any`) never take the
masked path at any width. They can stop early, and a mask spends its work before
the first hit is reported; the same change measured 40–60% worse on `any`.

## Early exit: which descent

`any` and `first` stop at their first item, so what they cost is the path to
it. They used to share the callback traversal of `visit`: test every child of
a node, push every overlapping one, descend into the lowest. On a large window
almost every child overlaps, so every node on the path was scanned to the end.
Counted on the [2D competitors](#2d-competitors) workload (100,000 boxes, node
size 16), child tests per `first`:

| Windows | old traversal | `static_aabb2d_index` | depth-first |
|---|---:|---:|---:|
| small | 72 | 66 | 50 |
| mid | 67 | 58 | 44 |
| large | 63 | 52 | 37 |

`static_aabb2d_index` does the same full scan but descends into the last child
it pushed, the highest. On each level the highest node is the partial tail
of the level (2, 9, 7 and 10 children on the way down here, against 16). That
alone was its 22% lead on large windows. The old traversal also took the scratch
stack from thread-local storage on every call, worth 5–10% of a `first`.

The descent that replaced it (`range::find_region`) enters a node's first
overlapping child as soon as the test finds it and keeps the node's untested
rest as the resume point of its level, one per level in a fixed array. No child
is tested twice, so a traversal that runs to the end costs what the old one
did; one that stops early skips the siblings after each child on its path, and
no stack is taken. The items come in `visit` order, so `first` returns what it
returned before.

The child test has two forms; neither wins everywhere. Branching stops at
the first hit; masked folds a node's tests into a bitmask and keeps the
untaken bits as the resume point. Masked / branching, both depth first, over
10,000 windows per row (`benches/paired_find_switch.rs`; owned, then view):

| Windows | expected hits | Xeon (EMR) | Zen 3 | Zen 4 | N2 |
|---|---:|---:|---:|---:|---:|
| 2D, sides 1..10 | 0.03 | 0.78, 0.82 | 0.75, 0.81 | 0.68, 0.73 | 1.17, 1.24 |
| 2D, sides 30..60 | 2.1 | 1.14, 1.20 | 1.01, 1.13 | 0.91, 1.05 | 1.39, 1.47 |
| 2D, sides 200..400 | 93 | 1.19, 1.24 | 1.09, 1.24 | 1.00, 1.20 | 1.47, 1.57 |
| 2D, sides 2000..5000 | 12,941 | 1.52, 1.67 | 1.32, 1.51 | 1.27, 1.44 | 1.77, 1.90 |
| 3D, sides 10..200 | 0.2 | 0.65, 0.68 | 0.64, 0.69 | 0.55, 0.59 | 1.02, 1.06 |
| 3D, sides 400..700 | 18 | 0.95, 1.01 | 0.90, 0.99 | 0.82, 0.87 | 1.23, 1.27 |
| 3D, sides 1000..1500 | 200 | 1.10, 1.12 | 1.00, 1.11 | 0.90, 0.97 | 1.30, 1.36 |
| 3D, sides 2000..5000 | 5,029 | 1.31, 1.41 | 1.17, 1.29 | 1.05, 1.14 | 1.44, 1.53 |

The mask wins on windows that find little, where branching mispredicts and has
no sibling to skip anyway; it loses once hits are likely. Where it crosses
depends on the machine and the storage, so `find` switches per query on the
same uniform estimate the radius switch reads (the window's share of the root
box times the item count): the mask below 2 expected hits in 2D
(`FIND_MASK_BELOW_HITS_2D`) and below 100 in 3D (`FIND_MASK_BELOW_HITS_3D`).
aarch64 runs the branching form alone.

Against the old traversal, both behind the shipping entry points (masked on
x86, branching on aarch64), small, mid and large windows, as a fraction of the
old time (`benches/paired_mask_forms.rs`, two or three runs per machine):

| Path | Xeon (EMR) | Zen 3 | Zen 4 | N2 |
|---|---|---|---|---|
| 2D owned | 0.85, 0.83, 0.50 | 0.86, 0.73, 0.54 | 0.93, 0.79, 0.59 | 0.87, 0.78, 0.66 |
| 2D view | 0.81, 0.77, 0.46 | 0.86, 0.70, 0.50 | 0.94, 0.77, 0.56 | 0.87, 0.78, 0.65 |
| 3D owned | 0.90, 0.82, 0.65 | 0.94, 0.91, 0.65 | 0.96, 0.95, 0.75 | 1.03, 0.85, 0.66 |
| 3D view | 0.98, 0.96, 0.63 | 1.00, 0.97, 0.67 | 0.99, 0.98, 0.77 | 1.02, 0.85, 0.67 |

The one loss is 3D windows that find nothing on the N2, 2–3%: with no sibling
to skip, the descent only adds its resume array. Zeroing it is about half
of that (an array of 8 levels read 1.02–1.03 there against 1.05 for 32).

Three measurement traps showed up on the way, each worth 20–25% on some cell:

- **A runtime `match` inside one bench closure.** The callback rows of
  `paired_mask_forms` used to put every form behind one closure and a `match`
  on the form. The arm the match reached last read up to 25% slow on a form
  that was the same code (an A/A arm showed it). Each arm is its own closure
  now.
- **Two forms inlined behind one switch.** With both kernels inlined into
  `find`, the form it picked ran 2–10% slower than when called alone. On x86
  `find_region` is out of line; one call per query costs less.
- **Out of line on aarch64.** There the same boundary, or a wrapper that did not
  inline, cost the 3D `first` on small windows about 20% on the N2, so aarch64,
  which runs one form, inlines the whole chain.

## Large-window range search

When a query fully contains a tree node, the covered-range fast path collects the
whole subtree by copying its contiguous leaf-index range instead of running
per-item overlap tests. This keeps the SIMD indexes from regressing against the
scalar indexes as the window grows: full-extent windows reach parity (both paths
just copy the contiguous index range) and everything smaller stays ahead. On
AVX-512 a masked compress-store collects the matching leaf indices in one
instruction, widening the SIMD lead on dense mid-to-large windows (e.g. the 3D
flat-Z batch above, and the `large` / `thin slab` rows here). Workload: 100,000
boxes over a 10,000-wide space, 1,000 query boxes per window class, on a Zen 5
built with `-C target-cpu=native`. Lower is better.

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
real regression, so the shape ships in that kernel and `search_shape::<0>` is
kept hidden so the comparison can be re-run.

**Which kernel that is, and what it is worth.** `search_simd` is the portable
`wide` tier, and `SimdIndex2D::search_into` reaches it only when the CPU offers
neither AVX-512 nor AVX2 — so on current x86_64 hardware the numbers above
describe a kernel that does not run. Measured in the same binary and the same
run, `search_into` (dispatching to AVX-512 on the Zen5 box) is about **half** the
`wide` tier's time on small and large windows alike, and the AVX2 tier about
two thirds of it; on a full-extent scan all three converge, because the
contained-subtree shortcut does the work and the kernel barely matters. So the
shape above is a 5% slice of the slowest tier. On aarch64, where that tier is
the whole story, it does not buy even that: on a Neoverse N2 mask-first against
per-4-lane measured 0.98–1.03 in 2D and 0.98–1.02 in 3D, a tie. It ships there
as a harmless default rather than a measured win; pre-AVX2 x86 is unmeasured.

### The same shape in the intrinsic tiers: refuted

Porting mask-first to the AVX2 and AVX-512 kernels was the obvious follow-up, and
it does not ship. Both already compute containment in lanes (`cbits`), so the
piece that made the portable kernel's version win — a scalar four-column
containment test per hit, sitting in the middle of the vector loop — was never
there to remove. What is left to reorder is an index load and two pushes.

Four sweeps at different repetition counts, each tier its own three-arm group
against its own shipping shape, 100k boxes, pinned:

| window | avx512 | avx2 |
| --- | ---: | ---: |
| small (10..200) | 1.04 / 1.12 / 1.08 / 1.06 | 0.97 / 0.93 (bands span 1.0) |
| large (2000..5000) | 0.95 / 0.96 / 0.95 / 0.945 | 0.96 / 0.98 / 0.97 / 0.97 |
| full extent | 0.955 / 0.947 / 0.960 / 0.950 | ~0.95 |

About 5% on large and full-extent windows, and a consistent ~5% **loss** on
small ones in AVX-512 — opposite signs, in the tier runtime dispatch actually
selects, with the loss falling on the commoner query shape. The gate that would
fix it is a selectivity constant fitted to one machine, which this project
declined once already for the radius switch's upper crossover.

The mechanism, after the run corrected it: **mask-first costs per chunk and pays
per hit.** The prediction from the first, wrong version of that sentence was that
an AVX-512 node of only two 8-lane chunks has too little to defer, so widening to
`node_size` 64 should flip the sign. It read 1.110 [1.060..1.141] — worse, and
the most cleanly resolved cell of the session. More chunks with few hits is
strictly more bookkeeping for the same empty drain.

The kernels, the node-size arm that falsified the prediction, and one genuine fix
found on the way (the mask form asked `query_contains_node` per *miss* in a
partial node's scalar tail, worth 1.083 → 1.055) are on
`probe/simd-tier-shapes`.

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

Raycasts over the packed index against the [`bvh`](https://crates.io/crates/bvh)
crate's SAH tree, 100,000 boxes in a 10,000 cube, rays of length 4,000 from
random origins in random directions. Three uniform scenes set by box size cover
a few to hundreds of boxes per ray: sparse (sides `1..150`, 2.5 boxes crossed
per ray on average), mid (`1..450`, 23) and dense (`1..1400`, 209). The
clustered scene puts boxes with sides `1..40` in four dense blobs; a random
ray crosses 0.2 of them. All hits run 10,000, 2000 and 400 rays by density
(2000 on the clustered scene); closest hit and occlusion run 10,000 in every
scene.

The `bvh` side of each row is the fair counterpart it offers. For closest hit
that is a hand-rolled front-to-back traversal of its tree with the same
pruning, since its API has no closest-hit query. For all hits it is the
broad-phase `traverse_iterator`. For occlusion it is that iterator's first
item, which it yields lazily.

Median per-round time relative to `bvh` (1.00), lower is better, from
[`paired_competitors`](#reproducing) on the machines of
[2D competitors](#2d-competitors):

| Scene | Query | Participant | Xeon (EMR) | Zen 3 | Zen 4 | Zen 4, no AVX-512 | N2 |
|---|---|---|---:|---:|---:|---:|---:|
| sparse | closest hit | `SimdIndex3D` | 0.38 | 0.73 | 0.46 | 0.64 | 0.71 |
| sparse | closest hit | `Index3D` | 0.90 | 1.53 | 1.34 | 1.45 | 1.13 |
| mid | closest hit | `SimdIndex3D` | 0.37 | 0.69 | 0.44 | 0.61 | 0.61 |
| mid | closest hit | `Index3D` | 0.78 | 1.35 | 1.14 | 1.30 | 0.95 |
| dense | closest hit | `SimdIndex3D` | 0.26 | 0.47 | 0.30 | 0.43 | 0.38 |
| dense | closest hit | `Index3D` | 0.48 | 0.81 | 0.67 | 0.79 | 0.56 |
| clustered | closest hit | `SimdIndex3D` | 1.09 | 1.50 | 1.21 | 1.41 | 1.77 |
| clustered | closest hit | `Index3D` | 1.92 | 2.41 | 2.39 | 2.39 | 2.39 |
| sparse | all hits | `SimdIndex3D` | 0.32 | 0.41 | 0.33 | 0.35 | 0.57 |
| sparse | all hits | `Index3D` | 0.74 | 1.31 | 1.20 | 1.25 | 1.02 |
| mid | all hits | `SimdIndex3D` | 0.24 | 0.37 | 0.25 | 0.31 | 0.52 |
| mid | all hits | `Index3D` | 0.67 | 1.21 | 1.12 | 1.14 | 0.98 |
| dense | all hits | `SimdIndex3D` | 0.17 | 0.29 | 0.15 | 0.26 | 0.40 |
| dense | all hits | `Index3D` | 0.54 | 0.92 | 0.80 | 0.90 | 0.78 |
| clustered | all hits | `SimdIndex3D` | 0.52 | 0.67 | 1.11 | 0.93 | 0.94 |
| clustered | all hits | `Index3D` | 1.24 | 1.88 | 3.25 | 3.39 | 1.60 |

On uniform scenes `SimdIndex3D` wins closest hit by 1.4–3.8× and all hits by
1.8–6.7× on every machine, more as the scene gets denser. On the clustered
scene the SAH tree is structurally better: `bvh` wins closest hit by
1.1–1.8×. All hits there come out between a 1.9× win for `SimdIndex3D` (the
Xeon) and a 1.1× loss (the Zen 4). The scalar `Index3D` needs the dense scene
or the Xeon to beat `bvh`; on AMD it trails by up to 1.5× on the sparse and mid ones.

Occlusion (`raycast_any`, any hit at all) was timed after `SimdIndex3D` moved
to a depth-first descent, on a second set of machines: a cloud Xeon VM
(family 6 model 85, the Skylake-SP line, AVX-512), a Zen 3 (EPYC 7763,
two runs), a Zen 4 (EPYC 9V74) with AVX-512 exposed (one run) and a Neoverse
N2 (one run). Same scenes and rays as above, time relative to `bvh`:

| Scene | Participant | Xeon (model 85) | Zen 3 | Zen 4 | N2 |
|---|---|---:|---:|---:|---:|
| sparse | `SimdIndex3D` | 0.40 | 0.53 | 0.42 | 0.67 |
| sparse | `Index3D` | 1.03 | 1.39 | 1.39 | 1.06 |
| mid | `SimdIndex3D` | 0.47 | 0.62 | 0.52 | 0.71 |
| mid | `Index3D` | 1.20 | 1.61 | 1.61 | 1.19 |
| dense | `SimdIndex3D` | 0.46 | 0.59 | 0.48 | 0.65 |
| dense | `Index3D` | 1.16 | 1.48 | 1.48 | 1.11 |
| clustered | `SimdIndex3D` | 1.20 | 1.30 | 1.15 | 1.71 |
| clustered | `Index3D` | 2.09 | 2.37 | 2.33 | 2.30 |

`bvh`'s lazy iterator stops at the first leaf it reaches. So do both of
ours. The scalar `Index3D::raycast_any` trails it by 1.03–1.61× on the uniform
scenes. `SimdIndex3D::raycast_any` runs the same descent with the vector slab
test of `raycast` and wins them by 1.4–2.5×. The clustered scene is the SAH
tree's again: `bvh` wins by 1.15–1.7×. Before the change `SimdIndex3D` stopped
a front-to-back `raycast_each` with a priority queue it did not need and
took 1.2–3.3× the `bvh` time; `paired_raycast_any` times the two forms against
each other. For oblique rays the depth-first one takes 0.16–0.25 of the old
time on the uniform scenes on x86 and 0.34–0.41 on the N2 (0.35–0.44 and 0.59
on the clustered scene). Axis-parallel rays always take the `wide` kernel and
gain less: 0.27–0.39 on x86 and 0.42–0.55 on the N2. Two cells did not win.
On the N2 the clustered scene's axis-parallel rays take 1.12× the scalar
`Index3D::raycast_any` time (NEON `wide` against the scalar slab, on rays that
mostly miss). On the model 85 Xeon the AVX2 kernel beats the AVX-512 one the
dispatch picks by 8–10% on the 2D and clustered rows (the two are level on
the others), while on the Zen 4 AVX-512 is ahead by 2–16%.

The packed Hilbert tree builds ~7x faster than the SAH tree (4 ms against
31 ms for 100k uniform boxes on the Zen 5 laptop, Criterion). Reproduce the
Criterion view with `cargo bench --bench raycast3d_bench --features simd`.

## Ray-triangle closest hit (mesh payload)

A triangle payload plus the index over each triangle's bounding box is a
streamable mesh BVH: `raycast` returns candidate boxes, then
`Ray3D::closest_triangle` runs the exact Moller-Trumbore test only on those. The
records are fixed-width, so the payload drops its offset table (smaller file, one
fewer streamed read) and a view borrows them as a zero-copy typed slice. Any
payload whose blobs are all the same size gets the first two of those without
asking — the serializer infers the width — but only a *declared* width is
borrowed as typed records, since a stride alone does not identify a type. The
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

## The four range-search frontends, by CPU

What the SIMD frontends and compact `f32` storage buy depends on the processor
more than anything else on this page, so every number here names its machine.
`benches/paired_precision.rs` puts `Index*`, `Index*F32`, `SimdIndex*` and
`SimdIndex*F32` on the same boxes and windows in one binary, interleaved, with
every arm divided by the scalar `f64` index. Run it where the answer matters:

```bash
BENCH_PIN_CORE=8 cargo bench --features simd,f32-storage --bench paired_precision
```

On the Zen 5 laptop (AVX-512 on 256-bit datapaths), `search` time relative to
the scalar `f64` index — lower is faster. Uniform boxes over a 10 000-unit
extent; small windows are 10–200 units wide in 2D and 10–300 in 3D, large ones
2 000–5 000; "all" covers the whole index:

| frontend, items | 2D small | 2D large | 2D all | 3D small | 3D large | 3D all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SimdIndex*`, 100k | 0.71 | 0.73 | 0.99 | 1.05 | 0.52 | 1.00 |
| `SimdIndex*`, 1M | 0.71 | 0.98 | 1.00 | 0.99 | 0.71 | 1.00 |
| `SimdIndex*F32`, 100k | 0.67 | 0.65 | 1.00 | 0.92 | 0.37 | 0.99 |
| `SimdIndex*F32`, 1M | 0.48 | 0.91 | 1.01 | 0.69 | 0.46 | 1.01 |

- The SIMD `f64` index leads on small 2D windows and on large 3D ones. It ties
  wherever copying a covered subtree does the work. On small 3D windows, which
  here return almost nothing to collect, it gains nothing.
- `SimdIndex*F32` is the fastest frontend on every small and large window: an
  AVX-512 chunk holds 16 of its boxes against 8 `f64` ones, at half the bytes.

The hosted runners at 100k boxes, small / large windows: a Zen 4 with AVX-512
(an EPYC 9V74) gives `SimdIndex*` 0.71 / 0.76 in 2D and 0.97 / 0.51 in 3D,
`SimdIndex*F32` 0.67 / 0.64 and 0.85 / 0.36. A server Zen 5 (an EPYC 9V45,
AVX-512 at full width) is the fastest measured: `SimdIndex*` 0.61 / 0.68 and
0.92 / 0.48, `SimdIndex*F32` 0.61 / 0.59 and 0.87 / 0.35.

The `f32` frontends run close to the `f64` speed or ahead of it on every
machine measured. 100k boxes, the same windows, time relative to the scalar
`f64` index on the same call, small / large / all (hosted runners through
`.github/workflows/bench-arm.yml`):

| frontend, call | Zen 5 (EPYC 9V45) | Zen 4 (EPYC 9V74) | Zen 3 (EPYC 7763) | Neoverse N2 |
| --- | --- | --- | --- | --- |
| `Index2DF32::search` | 1.01 / 0.90 / 1.00 | 1.01 / 0.91 / 0.98 | 0.98 / 0.91 / 1.00 | 0.93 / 0.89 / 1.00 |
| `Index3DF32::search` | 1.06 / 0.76 / 1.00 | 1.08 / 0.77 / 0.97 | 1.06 / 0.78 / 1.07 | 0.94 / 0.66 / 1.01 |
| `Index2DF32::count` | 1.29 / 1.44 / 1.22 | 1.23 / 1.38 / 1.10 | 1.21 / 1.34 / 1.01 | 0.94 / 0.97 / 1.13 |
| `Index3DF32::count` | 1.05 / 1.54 / 1.29 | 1.05 / 1.42 / 1.08 | 1.05 / 1.41 / 1.08 | 0.89 / 1.01 / 1.27 |
| `SimdIndex2DF32::count` | 0.69 / 0.70 / 0.52 | 0.70 / 0.68 / 0.42 | 0.68 / 0.67 / 0.36 | 0.62 / 0.63 / 0.32 |
| `SimdIndex3DF32::count` | 0.77 / 0.66 / 0.71 | 0.78 / 0.65 / 0.54 | 0.77 / 0.68 / 0.48 | 0.72 / 0.65 / 0.42 |

- The scalar `f32` index costs about what the `f64` one does on `search`,
  0.66–1.08× across these machines. It takes a covered subtree as one slice
  as the `f64` index does. Its `count` runs 0.89–1.54×, the one call here
  still behind the `f64` index.
- `SimdIndex*F32::count` is the fastest count on these machines, 0.32–0.78× the
  scalar `f64` one: it adds a covered subtree's leaf range and a leaf's
  popcount, like the `f64` `SimdIndex*::count`, with eight `f32` lanes to a
  test against four `f64` ones.
- A `search` over everything costs every frontend about the same: copying the
  whole index is the work. A `count` over everything is ~25 µs per 1 000
  queries on the `f64` index (it answers from the root), so the ratios in that
  column swing with a few microseconds.

The SIMD kernels are where machines part ways. `SimdIndex2D::search_into`
against `Index2D::search_into`, both into reused buffers, 100k boxes
(`benches/paired_simd_search.rs`; every column but the laptop's comes from
`.github/workflows/bench-arm.yml` on GitHub's hosted runners):

| 2D window | Zen 5 laptop | Zen 5 server | Zen 4 | Zen 3 (AVX2) | Neoverse N2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 0.61 | 0.56 | 0.64 | 0.80 | 0.97 |
| large | 0.69 | 0.67 | 0.74 | 0.94 | 1.09 |
| all | 1.00 | 0.98 | 0.99 | 0.96 | 1.00 |

The server Zen 5 is an EPYC 9V45 with AVX-512 at full width; it leads by the
most on small windows. With AVX-512 a Zen 4 keeps the lead on large windows as
the Zen 5s do. That rests on the kernels compressing hits in a register: Zen 4
microcodes the compress-to-memory form ([internals](internals/simd.md)), which
the server Zen 5 runs as fast as the register one. A VM may hide AVX-512
even on a Zen 4 — one EPYC 9V74 runner listed `avx512f`, another did not — and
the AVX2 tier runs then: it keeps most of the small-window lead and gives up the
large-window one, on a Zen 4 as on a Zen 3. The N2 has neither: its SIMD index
runs the portable `wide` tier on NEON and stays within 10% of the scalar index
either way. A hosted runner is a shared VM, so only these ratios, taken inside
one binary, carry over; its microseconds do not. Every number in this section
comes from query sets sized so the branch predictor cannot learn them (see
[Branch-free node tests](#branch-free-node-tests)); over the smaller sets used
before, the SIMD indexes looked 5–25 points further ahead on small windows.

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

  It saves memory without costing range-query speed: `search` takes 0.65–1.07×
  the scalar `f64` index's time and `count` 0.87–1.41× on a Xeon, a Zen 4, a
  Zen 3 and a Neoverse N2 (see
  [the four frontends](#the-four-range-search-frontends-by-cpu)). The build
  runs ~1.7× slower from the rounding (a 1M-box spot check). The query is
  rounded onto the `f32` grid once, so each node compares `f32` to `f32`; the
  extra conservative candidates the outward-rounded boxes admit come to a few
  per ten million hits.

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
- against `static_aabb2d_index` on query sets the branch predictor cannot
  learn (a cloud Xeon, a Zen 3, a Zen 4, a Neoverse N2), scalar `Index2D`
  collects 1.4–1.8× faster on small windows and 2.6–3.3× on large ones. Its
  `visit` leads by 1.1–1.5× and 2.3–2.5×; its early exit is 1.3–1.6× faster on
  every window class. `Index2D` build is faster as well;
- against the `bvh` crate, `SimdIndex3D` wins closest hit, all hits and
  occlusion on uniform scenes on every machine measured and loses closest hit
  and occlusion on a clustered one, where the SAH tree is better;
- `Index3D` build and KNN are still slower than `Index2D`, but uniform 3D search
  is faster when Z meaningfully prunes the tree;
- the SIMD indexes' lead over the scalar ones on range search depends on the
  CPU: up to 1.8× on the Zen 5 laptop and 1.6× on a Zen 4 with AVX-512, up to
  1.3× on the AVX2 tier (a Zen 3, or a Zen 4 whose VM hides AVX-512), within
  10% either way on a Neoverse N2;
- the branch-free node test behind those collect numbers applies only where the
  per-child predicate is cheap; the collect paths and `visit` take it (the 2D
  ones keep branches on aarch64), `any` / `first` only on windows that expect
  almost nothing, since their depth-first descent stops testing at the first
  hit. The search iterators and the
  shape regions keep their branching traversal; the radius queries choose per
  query;
- f32 storage halves box memory; the SIMD `f32` index is also the fastest range
  search and count; the scalar one runs `search` at 0.65–1.07× the `f64`
  index's time (a Xeon, a Zen 4, a Zen 3, a Neoverse N2); exact callbacks trade
  source-box lookup for exact results;
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
- `paired_find_switch` times the two child tests of the depth-first `any` /
  `first` descent and the shipped switch across the expected-hits range (the
  table in [Early exit: which descent](#early-exit-which-descent));
- `paired_competitors` times the `static_aabb2d_index`, FlatGeobuf and `bvh`
  comparisons interleaved in one binary, on the class-sized query sets of
  `benches/support/competitors.rs` (the numbers in
  [2D competitors](#2d-competitors) and the `bvh` section);
- `paired_raycast_any` times the SIMD indexes' depth-first `raycast_any`
  against the priority-queue form it replaced, owned and view, 2D and 3D,
  with the scalar indexes as the control;
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
`VPCOMPRESSQ` result collection (up to ~1.8× over the scalar index on the Zen 5
laptop, ~1.6× on a Zen 4; see
[the four frontends](#the-four-range-search-frontends-by-cpu)), the AVX2 tier uses a
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
taskset -c 1 cargo bench --bench paired_competitors --features simd
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
