# SIMD optimization in `packed_spatial_index` — the path to the black belt

This crate's query speed comes from a stack of techniques layered on top of each
other, each one earned by a measurement. This is the whole ladder, from "write it
so the compiler can vectorize" to "emulate an instruction the CPU doesn't have",
with the *why* and the numbers behind each rung. Every claim here was gated by a
measure-then-commit A/B; ideas that lost are recorded too, because knowing what
*doesn't* help is half the belt.

The running example is the hot loop of a range query: for each tree node, test
which child boxes overlap the query, then collect the matching item indices.

---

## White belt — branchless code the compiler can vectorize

Before any intrinsics, write the kernel so LLVM's auto-vectorizer can do the work.
The box-overlap test is four comparisons AND-ed together, with **no early-exit
branch**:

```rust
let hit = (min_x[i] <= q.max_x) & (max_x[i] >= q.min_x)
        & (min_y[i] <= q.max_y) & (max_y[i] >= q.min_y);
```

Using `&` (not `&&`) keeps it branchless, so the loop auto-vectorizes. The
ray-triangle test learned this the hard way: a *branchy* version ran 13.95
ns/(ray·tri), the *branchless* one 5.77 ns — **2.4× from deleting early-exits
alone**, because the branches were what blocked auto-vectorization.

## Yellow belt — Structure-of-Arrays (the enabler)

The owned indexes store boxes Array-of-Structs (one `Box2D` per item, the on-disk
layout). SIMD wants the opposite: `SimdIndex*` keeps **separate columns**
(`min_xs[]`, `min_ys[]`, `max_xs[]`, `max_ys[]`). Now one SIMD register holds the
same coordinate of N consecutive boxes; the overlap test is a handful of
vector ops per N boxes instead of per box. SoA is what makes everything below
possible — it is the single most important layout decision.

(It isn't free: serializing an `SimdIndex` gathers SoA→AoS and loading scatters
AoS→SoA, so its persistence is ~1.5–2.8× slower than the AoS `Index`. One file
format, paid for at load time, not query time. See the persistence note in
[performance.md](../performance.md).)

## Orange belt — explicit portable SIMD (`wide`)

The baseline SIMD kernel uses the [`wide`](https://crates.io/crates/wide) crate's
`f64x4`: load four boxes per column, compare, AND the masks, then read out a
bitmask of hits. `wide` lowers to whatever width the build's target features
allow — **SSE2 (128-bit) by default**, AVX2 (256-bit) only with
`-C target-cpu=native` / `x86-64-v3`. This is the portable floor and the wasm
(`simd128`) path.

## Green belt — runtime feature dispatch

A binary shipped to crates.io can't assume AVX-512. So the kernels are selected
**at runtime** with `is_x86_feature_detected!`, newest first:

```text
AVX-512  →  AVX2  →  SSE2 (wide)
```

Each tier is a separate `#[target_feature(enable = "…")]` function; the check is
cached, so dispatch is a predictable branch. This is the same idea as a hand-rolled
`memcpy`'s `cpu_supports()` probe — pick the widest path the actual CPU offers,
fall back gracefully.

## Blue belt — find the *real* bottleneck (it's not the box test)

The trap: widen the box test and call it a day. Measured, that **ties** the
narrower kernel at large result sets. Why? A query is two phases — *test* and
*collect* — and at scale the **collection dominates**. An AVX2 box test that only
sped the test, leaving a scalar `trailing_zeros` push-loop for collection, ran
~1.15× at 100k boxes and **0.97× (tied) at 1M**. The 1M SIMD path even ran
*slower* than scalar before this was fixed. The lesson that unlocks the rest of
the ladder: **attack the collection, not the comparison.**

## Purple belt — AVX-512 `VPCOMPRESSQ` (compress-store)

AVX-512 has the perfect instruction for the collect phase.
`_mm512_mask_compressstoreu_epi64` takes the 8-lane overlap mask and the 8 index
lanes and writes **only the matching indices, packed contiguously, in one
instruction**. No per-bit loop:

```rust
let dst  = out.as_mut_ptr().add(out.len()).cast::<i64>();
let vidx = _mm512_loadu_epi64(indices.as_ptr().add(pos).cast());
_mm512_mask_compressstoreu_epi64(dst, mask, vidx);   // pack + store
out.set_len(out.len() + mask.count_ones() as usize); // advance by popcount
```

This removed the large-N inversion: SIMD range search went from trailing the
scalar index to **~1.6–1.9× faster** across 100k–1M; a dense all-hits raycast
dropped from ~29.5 µs to ~17.1 µs at 1M (**1.73×**). It's used for AVX-512 range
search and all-hits raycast (the raycast path a later addition).

## Brown belt — AVX2 left-pack (emulating the instruction)

AVX2 has **no compress**. The brown-belt move is to emulate it with a shuffle and
a lookup table — the classic "left-pack" trick.

A 4-wide box test gives a 4-bit mask; the four `u64` indices sit in one 256-bit
register. `_mm256_permutevar8x32_epi32` (`VPERMD`) permutes the eight 32-bit lanes
by eight indices. A `u64` is two adjacent 32-bit lanes, so a control vector
listing the `2k, 2k+1` halves of each set bit moves the matching `u64`s to the
front. There are only 16 possible 4-bit masks, so the controls are a **16-entry
table built at compile time** (`const fn`):

```rust
let idx    = _mm256_loadu_si256(indices[pos..].as_ptr().cast());   // 4 u64
let ctrl   = _mm256_loadu_si256(LEFTPACK_LUT[mask].as_ptr().cast());
let packed = _mm256_permutevar8x32_epi32(idx, ctrl);               // matches to front
_mm256_storeu_si256(out[len..].as_mut_ptr().cast(), packed);       // store all 4
len += mask.count_ones();                                          // count the real ones
```

This is the shared `leftpack4` helper in [`src/leftpack.rs`](../src/leftpack.rs).
With it the AVX2 tier runs **1.3–1.65× over the SSE2 fallback across 100k–1M** —
the win *holds at large N* because the collection is no longer scalar. It is the
difference between AVX2 being worth shipping and not: with a scalar collect it
tied at 1M; with left-pack it wins everywhere.

This rung is the headline lesson of the whole ladder: when the CPU lacks the
instruction you want, an instruction you *do* have plus a small precomputed table
often recovers most of it.

## Black belt — the safety invariant

Both compress and left-pack **store a full vector regardless of how many lanes
matched** (8 for compress, 4 for left-pack); only the first `popcount` are valid.
That's the speed — no branch on the count — but it means the output buffer must
have a full vector of slack past its logical length. The invariant: **reserve
`node_size + vector_width` before each leaf node**, so the base pointer is stable
for the whole node and every store lands within capacity; advance the logical
length only by `popcount`. Get this wrong and you have a buffer overrun in
`unsafe` code, so it's pinned by `tests/avx2.rs` (24k+ queries across sizes, node
sizes and edge cases, asserted equal to the scalar index).

The f32 kernels add a twist: they test **8** boxes per step (`f32` is half the
width) but indices are still `u64` (two 256-bit registers). The 4-wide left-pack
is simply applied **twice** — to the low and high nibble of the 8-bit mask — each
guarded `if nibble != 0`, because the two-store fixed cost otherwise *regresses*
on sparse leaves (measured: 0.68× unguarded → 1.14× guarded at low density).

## Master techniques — skip the work entirely

The fastest test is the one you don't run.

- **Contained-subtree fast path.** When the query fully contains a tree node, its
  whole subtree is a guaranteed hit — collected with one `extend_from_slice`
  (a `memcpy`) instead of any per-item test. The region queries (triangle /
  polygon / frustum) lean on this hard.
- **Software prefetch.** Tree traversal is pointer-chasing, so prefetching the
  next node off the stack before processing the current one hides latency
  (~3–5%, free). Note this is the *opposite* of a streaming `memcpy`, where the
  hardware prefetcher already saturates sequential reads and software prefetch is
  a no-op — the technique fits the access pattern, not a rule of thumb.
- **Build-side SIMD.** Hilbert keys are encoded with a SWAR "magic-bits"
  multiply-shift (≈7× a generic curve library), deliberately *not* `pdep`/`pext`,
  because BMI2 is microcoded and slow on older AMD — portability beat the
  instruction. Sorting those keys uses LSD radix.

## What does *not* benefit (also part of the belt)

Measured and deliberately skipped:

- **`visit` and best-first `raycast_closest`** get only the box-test width (their
  "collection" is a user callback or a single result — no packing lever), so AVX2
  there is ~1.2× at mid-N and ties at large N. Kept where it doesn't regress;
  the closest-hit AVX2 kernel was skipped as not worth the masked-variant code.
- **A thin AVX2 wrapper around the `wide` loop does nothing** — `wide`/`safe_arch`
  pick their width from *compile-time* target features, globally, so wrapping the
  loop in `#[target_feature(enable="avx2")]` does not re-lower it. The explicit
  `_mm256` kernel is unavoidable.
- **Serialization NT-stores / a custom `memcpy`** — serialization is one-shot and
  streaming is round-trip-bound, so faster bulk copies move nothing.

## The dispatch ladder, summarized

| Tier | Box test | Collection | Range search vs next-lower |
| --- | --- | --- | --- |
| AVX-512 | 8-wide `cmp_pd_mask` | `VPCOMPRESSQ` | ~1.2–1.4× over AVX2 |
| AVX2 | 4-wide `_mm256_cmp_pd` | left-pack (`VPERMD` + LUT) | ~1.3–1.65× over SSE2 |
| SSE2 (`wide`) | `f64x4` (2×128-bit) | scalar bit-loop | the floor |

Numbers are from a Zen 5 (Ryzen AI 7 350), f64 2D, 100k–1M boxes; they depend on
hardware and workload. Reproduce with the harnesses described in
[performance.md](../performance.md). Correctness for every tier is in
`tests/avx2.rs`.

## Meta-belt — measure-then-commit

The real discipline isn't any one instruction; it's that **every rung was an A/B
measurement; the losers were reverted.** AVX2 looked dead (tied at 1M) until
the left-pack fixed the collection; the f32 left-pack looked like a regression
until the per-nibble guard; "smaller node size helps closest-hit," "a thin AVX2
wrapper," and "cluster ordering" all looked plausible and all lost on the bench.
The black belt is the habit, not the intrinsic.
