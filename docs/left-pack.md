# AVX2 left-pack (result collection without `VPCOMPRESSQ`)

The SoA SIMD kernels do two things per tree node: a **box test** (compare each
child box to the query, producing a bitmask of matches) and a **collection**
(append the matching item indices to the result vector). On AVX-512 the
collection is one instruction — `VPCOMPRESSQ` packs the masked `u64` index lanes
contiguously and stores them. On AVX2 there is no compress instruction, so a
naive port collects with a scalar `trailing_zeros` loop. That loop is the
bottleneck: with it, an AVX2 search only matched the SSE2-`wide` fallback at large
result sets (the box-test width gain was lost in the scalar collection).

The fix is an **AVX2 left-pack**, the well-known `VPERMD` + lookup-table trick. It
recovers most of the compress win on AVX2 hardware (Intel Haswell–Ice Lake, AMD
Zen 1–3 — the large installed base without AVX-512).

## How it works

A 4-wide `f64` box test (`_mm256_cmp_pd` + `_mm256_movemask_pd`) gives a 4-bit
mask of which of the four child boxes overlap. The four item indices live in one
256-bit register (4 × `u64`). We want the matching ones packed to the front so a
single store appends exactly them.

`_mm256_permutevar8x32_epi32` (`VPERMD`) permutes the eight 32-bit lanes of a
register by eight per-lane indices. A `u64` is two adjacent `u32` lanes, so a
control vector that lists the `2k, 2k+1` halves of each set bit `k` (in order)
moves the matching `u64`s to the front. There are only 16 possible 4-bit masks,
so the controls are a **16-entry lookup table**, built at compile time:

```rust
const LEFTPACK_LUT: [[i32; 8]; 16] = { /* for each mask m, list 2k,2k+1 of set bits */ };
```

The collection per chunk is then:

```rust
let idx    = _mm256_loadu_si256(indices[pos..].as_ptr().cast());   // 4 u64
let ctrl   = _mm256_loadu_si256(LEFTPACK_LUT[mask].as_ptr().cast());
let packed = _mm256_permutevar8x32_epi32(idx, ctrl);               // matches to front
_mm256_storeu_si256(out[len..].as_mut_ptr().cast(), packed);       // store all 4
len += mask.count_ones();                                          // count the real ones
```

The store always writes four `u64`; only the first `popcount(mask)` are valid, the
rest are overwritten by the next chunk or left past the logical length. This is
the shared `leftpack4` helper in [`src/leftpack.rs`](../src/leftpack.rs).

### Safety invariant

`leftpack4` stores a full 256 bits regardless of how many lanes matched, so the
caller must guarantee at least four free `u64` slots past the current length.
Each kernel reserves `end - node_index + 4` before a leaf node, so the base
pointer is stable for the whole node and every store stays within capacity; the
logical length is advanced only by `popcount`.

### f32 (8-wide)

The compact `f32` kernels test eight boxes per step (`_mm256_cmp_ps`, 8-bit mask),
but indices are still `u64` (two 256-bit registers). The same 4-wide primitive is
applied twice — to the low nibble over `indices[pos..pos+4]` and the high nibble
over `indices[pos+4..pos+8]` — each guarded `if nibble != 0`, because the
two-store fixed cost otherwise regresses on sparse leaves.

## Versus AVX-512 `VPCOMPRESSQ`

`VPCOMPRESSQ` packs and stores in one instruction over 8 lanes and needs no LUT;
left-pack needs a permute + a LUT load and works 4 lanes at a time. AVX-512 stays
faster where available — the dispatch is `avx512f → avx2 → wide`. Left-pack is the
middle tier for CPUs without AVX-512.

## Measured (Ryzen AI 7 350 / Zen 5, AVX2 forced)

AVX2 left-pack vs the SSE2-`wide` fallback, f64 2D, 100k–1M boxes:

| Path | 100k | 1M |
| --- | ---: | ---: |
| range search | ~1.5–1.6× | ~1.3× |
| all-hits raycast | 1.32× | 1.59× |

Correctness is verified against the scalar indexes in `tests/avx2.rs` (the
`*_avx2` entries are called directly, so the path is exercised even on an AVX-512
machine) across a sweep of sizes, node sizes, and edge cases.
