use crate::{geometry::Box2D, hilbert2d as hilbert};

/// LSD digit width, i.e. `ceil(32 / bits)` passes over the key.
///
/// Eight looks arbitrary next to the 32-bit Hilbert key — three passes of 11
/// bits would touch the pair array once less — so it was swept end to end
/// through [`Index2DBuilder::radix_sort_bits`], median of 7 interleaved rounds,
/// pinned, on a full build:
///
/// | bits | passes | 100k  | 1M    |
/// | ---: | ---:   | ---:  | ---:  |
/// | 8    | 4      | 1.000 | 1.000 |
/// | 10   | 4      | 1.001 | 1.016 |
/// | 11   | 3      | 0.965 | 1.053 |
/// | 12   | 3      | 0.977 | 1.062 |
/// | 16   | 2      | 1.020 | 1.051 |
///
/// Fewer passes wins only while the histogram and the scatter's working set
/// stay in cache. At a million items, spraying into 2048 buckets costs more
/// than the pass it saves, and the 100k figure sits inside that measurement's
/// own 6% spread. Eight stays.
pub(crate) const DEFAULT_RADIX_BITS: u32 = 8;
// Measured crossover: below this, comparison sort beats radix's scratch
// allocation and multi-pass fixed cost for full 2D builds.
const RADIX_SORT_MIN_ITEMS: usize = 385;
const MIN_RADIX_BITS: u32 = 1;
const MAX_RADIX_BITS: u32 = 16;

/// Which key to use when sorting boxes before packing the tree.
///
/// [`SortKey2D::Hilbert`] is the default and currently the only stable public
/// ordering. Additional sort-key implementations are available only through the
/// hidden benchmark support API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey2D {
    /// Hilbert curve order.
    Hilbert,
}

/// Sort-key implementation variants used by benchmarks.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey2DStrategy {
    /// Hilbert curve, "magic bits" (rawrunprotected): the reference crate algorithm.
    HilbertMagicBits,
    /// Hilbert curve, classic iterative algorithm with quadrant rotations.
    HilbertLoopRotation,
    /// Hilbert curve, table-driven finite-state machine.
    HilbertLut,
    /// Morton curve (Z-order).
    Morton,
}

impl From<SortKey2D> for SortKey2DStrategy {
    fn from(key: SortKey2D) -> Self {
        match key {
            // Same Hilbert values as the reference-style magic-bits encoder, but faster
            // in the real build path where keys are produced as `(key, index)` pairs.
            SortKey2D::Hilbert => SortKey2DStrategy::HilbertLut,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SortKeyContext {
    pub(crate) scaled_width: f64,
    pub(crate) scaled_height: f64,
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) radix: bool,
    pub(crate) radix_bits: u32,
    #[cfg(feature = "parallel")]
    pub(crate) use_parallel: bool,
}

pub(crate) fn encode_sort_serial<F>(
    items: &[Box2D],
    encode: &F,
    radix: bool,
    radix_bits: u32,
) -> Vec<(u32, u32)>
where
    F: Fn(usize, &Box2D) -> (u32, u32),
{
    // `collect` over an ExactSizeIterator allocates exactly `items.len()` once and
    // writes through a pointer, where the `push` loop re-read the capacity and
    // stored the new length back on every item.
    let mut order: Vec<(u32, u32)> = items
        .iter()
        .enumerate()
        .map(|(i, item)| encode(i, item))
        .collect();
    if radix && items.len() >= RADIX_SORT_MIN_ITEMS {
        radix_sort_u32(&mut order, radix_bits);
    } else {
        order.sort_unstable_by_key(|&(h, _)| h);
    }
    order
}

#[cfg(feature = "parallel")]
pub(crate) fn encode_sort_by_key(
    items: &[Box2D],
    sort_key: SortKey2DStrategy,
    context: SortKeyContext,
) -> Vec<(u32, u32)> {
    match sort_key {
        SortKey2DStrategy::HilbertMagicBits => {
            encode_sort_with_encoder(items, hilbert::magic_bits, context)
        }
        SortKey2DStrategy::HilbertLoopRotation => {
            encode_sort_with_encoder(items, hilbert::loop_rotation, context)
        }
        SortKey2DStrategy::HilbertLut => encode_sort_with_encoder(items, hilbert::lut, context),
        SortKey2DStrategy::Morton => encode_sort_with_encoder(items, hilbert::morton, context),
    }
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn encode_sort_by_key(
    items: &[Box2D],
    sort_key: SortKey2DStrategy,
    context: SortKeyContext,
) -> Vec<(u32, u32)> {
    match sort_key {
        SortKey2DStrategy::HilbertMagicBits => {
            encode_sort_with_encoder(items, hilbert::magic_bits, context)
        }
        SortKey2DStrategy::HilbertLoopRotation => {
            encode_sort_with_encoder(items, hilbert::loop_rotation, context)
        }
        SortKey2DStrategy::HilbertLut => encode_sort_with_encoder(items, hilbert::lut, context),
        SortKey2DStrategy::Morton => encode_sort_with_encoder(items, hilbert::morton, context),
    }
}

#[cfg(feature = "parallel")]
fn encode_sort_with_encoder<F>(
    items: &[Box2D],
    key_fn: F,
    context: SortKeyContext,
) -> Vec<(u32, u32)>
where
    F: Fn(u16, u16) -> u32 + Copy + Sync,
{
    let encode = |i: usize, item: &Box2D| -> (u32, u32) {
        let hx = hilbert_coord(context.scaled_width, item.min_x, item.max_x, context.min_x);
        let hy = hilbert_coord(context.scaled_height, item.min_y, item.max_y, context.min_y);
        (key_fn(hx, hy), i as u32)
    };

    if context.use_parallel {
        encode_sort_parallel(items, &encode)
    } else {
        encode_sort_serial(items, &encode, context.radix, context.radix_bits)
    }
}

#[cfg(not(feature = "parallel"))]
fn encode_sort_with_encoder<F>(
    items: &[Box2D],
    key_fn: F,
    context: SortKeyContext,
) -> Vec<(u32, u32)>
where
    F: Fn(u16, u16) -> u32 + Copy,
{
    let encode = |i: usize, item: &Box2D| -> (u32, u32) {
        let hx = hilbert_coord(context.scaled_width, item.min_x, item.max_x, context.min_x);
        let hy = hilbert_coord(context.scaled_height, item.min_y, item.max_y, context.min_y);
        (key_fn(hx, hy), i as u32)
    };

    encode_sort_serial(items, &encode, context.radix, context.radix_bits)
}

#[cfg(feature = "parallel")]
pub(crate) fn encode_sort_parallel<F>(items: &[Box2D], encode: &F) -> Vec<(u32, u32)>
where
    F: Fn(usize, &Box2D) -> (u32, u32) + Sync,
{
    use rayon::prelude::*;

    let mut order: Vec<(u32, u32)> = items
        .par_iter()
        .enumerate()
        .map(|(i, item)| encode(i, item))
        .collect();
    order.par_sort_unstable_by_key(|&(h, _)| h);
    order
}

/// Normalize the box center into `[0, 65535]` (Hilbert encoder input), with saturation.
#[inline]
pub(crate) fn hilbert_coord(scaled: f64, lo: f64, hi: f64, extent_min: f64) -> u16 {
    let value = scaled * (0.5 * (lo + hi) - extent_min);
    // Saturate with `max`/`min` against constants rather than three data-dependent
    // branch pairs. With a non-NaN constant operand LLVM emits a bare `maxsd`/
    // `minsd` and no NaN fixup, and the clamp answers every case the branches did:
    // NaN loses to `0.0` (`f64::max` returns the non-NaN operand), a value above
    // the top saturates, a negative value floors at zero, and the final cast still
    // truncates toward zero.
    value.max(0.0).min(u16::MAX as f64) as u16
}

/// LSD radix sort for `(key_u32, index)` pairs, with configurable digit width.
#[doc(hidden)]
pub fn radix_sort_pairs(a: &mut [(u32, u32)], bits: u32) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let bits = normalize_radix_bits(bits);
    let buckets = 1usize << bits;
    let mask = (buckets as u32) - 1;
    let passes = 32u32.div_ceil(bits);

    let mut tmp: Vec<(u32, u32)> = vec![(0, 0); n];
    let mut counts = vec![0usize; buckets];

    fn pass(
        src: &[(u32, u32)],
        dst: &mut [(u32, u32)],
        shift: u32,
        mask: u32,
        counts: &mut [usize],
    ) {
        counts.iter_mut().for_each(|c| *c = 0);
        for &(k, _) in src {
            counts[((k >> shift) & mask) as usize] += 1;
        }
        let mut sum = 0usize;
        for c in counts.iter_mut() {
            let cnt = *c;
            *c = sum;
            sum += cnt;
        }
        for &pair in src {
            let b = ((pair.0 >> shift) & mask) as usize;
            dst[counts[b]] = pair;
            counts[b] += 1;
        }
    }

    for p in 0..passes {
        let shift = p * bits;
        if p % 2 == 0 {
            pass(a, &mut tmp, shift, mask, &mut counts);
        } else {
            pass(&tmp, a, shift, mask, &mut counts);
        }
    }
    if passes % 2 == 1 {
        a.copy_from_slice(&tmp);
    }
}

fn radix_sort_u32(a: &mut [(u32, u32)], bits: u32) {
    radix_sort_pairs(a, bits);
}

#[inline]
pub(crate) fn normalize_radix_bits(bits: u32) -> u32 {
    bits.clamp(MIN_RADIX_BITS, MAX_RADIX_BITS)
}

#[cfg(test)]
mod tests {
    use super::radix_sort_pairs;

    /// Key shapes chosen so that whole digits come out constant, plus duplicate-heavy ones:
    /// these are where a pass degenerates into a single bucket and where stability shows.
    fn keys(kind: usize, i: u32, seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        match kind {
            0 => *seed,               // full width
            1 => 0xABCD_1234,         // every key equal
            2 => *seed & 0xFF,        // only the low digit varies
            3 => *seed & 0xFF00_00FF, // low and high digits vary, middle constant
            _ => i % 3,               // heavy duplicates, few distinct
        }
    }

    #[test]
    fn radix_sort_pairs_matches_a_stable_sort() {
        for kind in 0..5 {
            for &n in &[0usize, 1, 2, 385, 1000, 4096] {
                for &bits in &[1u32, 4, 8, 11, 16] {
                    let mut seed = 0x1234_5678u32;
                    let mut data: Vec<(u32, u32)> = (0..n as u32)
                        .map(|i| (keys(kind, i, &mut seed), i))
                        .collect();
                    let mut expected = data.clone();
                    expected.sort_by_key(|&(k, _)| k);
                    radix_sort_pairs(&mut data, bits);
                    // Compares the indices too: the sort must stay stable, since equal
                    // Hilbert keys otherwise reorder items between builds.
                    assert_eq!(data, expected, "kind={kind} n={n} bits={bits}");
                }
            }
        }
    }
}
