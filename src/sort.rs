use crate::{geometry::Bounds2D, hilbert};

pub(crate) const DEFAULT_RADIX_BITS: u32 = 8;
const MIN_RADIX_BITS: u32 = 1;
const MAX_RADIX_BITS: u32 = 16;

/// Which key to use when sorting boxes before packing the tree.
///
/// [`SortKey2D::Hilbert`] is the default and currently the only stable public
/// ordering. Additional sort keys are kept in the hidden experimental API for
/// benchmarking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey2D {
    /// Hilbert curve order.
    Hilbert,
}

/// Experimental sort-key implementations used by benchmarks.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentalSortKey2D {
    /// Hilbert curve, "magic bits" (rawrunprotected): the reference crate algorithm.
    HilbertMagicBits,
    /// Hilbert curve, classic iterative algorithm with quadrant rotations.
    HilbertLoopRotation,
    /// Hilbert curve, table-driven finite-state machine.
    HilbertLut,
    /// Morton curve (Z-order).
    Morton,
}

impl From<SortKey2D> for ExperimentalSortKey2D {
    fn from(key: SortKey2D) -> Self {
        match key {
            SortKey2D::Hilbert => ExperimentalSortKey2D::HilbertMagicBits,
        }
    }
}

impl ExperimentalSortKey2D {
    /// Compute the sort key for normalized coordinates `x, y in [0, 65535]`.
    #[inline]
    pub(crate) fn encode(self, x: u16, y: u16) -> u32 {
        match self {
            ExperimentalSortKey2D::HilbertMagicBits => hilbert::magic_bits(x, y),
            ExperimentalSortKey2D::HilbertLoopRotation => hilbert::loop_rotation(x, y),
            ExperimentalSortKey2D::HilbertLut => hilbert::lut(x, y),
            ExperimentalSortKey2D::Morton => hilbert::morton(x, y),
        }
    }
}

pub(crate) fn encode_sort_serial<F>(
    items: &[Bounds2D],
    encode: &F,
    radix: bool,
    radix_bits: u32,
) -> Vec<(u32, u32)>
where
    F: Fn(usize, &Bounds2D) -> (u32, u32),
{
    let mut order: Vec<(u32, u32)> = Vec::with_capacity(items.len());
    for (i, b) in items.iter().enumerate() {
        order.push(encode(i, b));
    }
    if radix {
        radix_sort_u32(&mut order, radix_bits);
    } else {
        order.sort_unstable_by_key(|&(h, _)| h);
    }
    order
}

#[cfg(feature = "parallel")]
pub(crate) fn encode_sort_parallel<F>(items: &[Bounds2D], encode: &F) -> Vec<(u32, u32)>
where
    F: Fn(usize, &Bounds2D) -> (u32, u32) + Sync,
{
    use rayon::prelude::*;

    let mut order: Vec<(u32, u32)> = items
        .par_iter()
        .enumerate()
        .map(|(i, b)| encode(i, b))
        .collect();
    order.par_sort_unstable_by_key(|&(h, _)| h);
    order
}

/// Normalize the bbox center into `[0, 65535]` (Hilbert encoder input), with saturation.
#[inline]
pub(crate) fn hilbert_coord(scaled: f64, lo: f64, hi: f64, extent_min: f64) -> u16 {
    let value = scaled * (0.5 * (lo + hi) - extent_min);
    if value.is_nan() {
        0
    } else if value > u16::MAX as f64 {
        u16::MAX
    } else if value < 0.0 {
        0
    } else {
        value as u16
    }
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
