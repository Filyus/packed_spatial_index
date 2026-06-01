use crate::geometry::Bounds3D;

const DEFAULT_RADIX_BITS_3D: u32 = 8;
const MIN_RADIX_BITS: u32 = 1;
const MAX_RADIX_BITS: u32 = 16;

const MORTON_BITS_PER_AXIS: u32 = 21;
const MORTON_AXIS_MAX: u32 = (1 << MORTON_BITS_PER_AXIS) - 1;
const HILBERT_BITS_PER_AXIS: u32 = 16;
const HILBERT_AXIS_MAX: u32 = (1 << HILBERT_BITS_PER_AXIS) - 1;
const HILBERT3_STEP_LUT: [u8; 192] = build_hilbert3_step_lut();
const HILBERT3_PAIR_LUT: [u16; 1536] = build_hilbert3_pair_lut();

/// Which key to use when sorting 3D boxes before packing the tree.
///
/// [`SortKey3D::Hilbert`] is the default and currently the only stable public
/// ordering. Additional sort keys are kept in the hidden experimental API for
/// benchmarking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey3D {
    /// Hilbert curve order.
    Hilbert,
}

/// Experimental 3D sort-key implementations used by benchmarks.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentalSortKey3D {
    /// 3D Hilbert curve over 16 bits per axis.
    Hilbert,
    /// 3D Morton/Z-order curve over 21 bits per axis.
    Morton,
}

impl From<SortKey3D> for ExperimentalSortKey3D {
    fn from(key: SortKey3D) -> Self {
        match key {
            SortKey3D::Hilbert => ExperimentalSortKey3D::Hilbert,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SortKey3DContext {
    pub(crate) extent: Bounds3D,
    pub(crate) radix: bool,
    pub(crate) radix_bits: u32,
    #[cfg(feature = "parallel")]
    pub(crate) use_parallel: bool,
}

impl SortKey3DContext {
    pub(crate) fn new(extent: Bounds3D, radix: bool, radix_bits: u32) -> Self {
        Self {
            extent,
            radix,
            radix_bits: normalize_radix_bits_3d(radix_bits),
            #[cfg(feature = "parallel")]
            use_parallel: false,
        }
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn parallel(mut self, use_parallel: bool) -> Self {
        self.use_parallel = use_parallel;
        self
    }
}

pub(crate) fn encode_sort_by_key_3d(
    items: &[Bounds3D],
    sort_key: ExperimentalSortKey3D,
    context: SortKey3DContext,
) -> Vec<(u64, usize)> {
    match sort_key {
        ExperimentalSortKey3D::Hilbert => {
            encode_sort_with_encoder_3d(items, encode_hilbert3, HILBERT_AXIS_MAX, context)
        }
        ExperimentalSortKey3D::Morton => {
            encode_sort_with_encoder_3d(items, encode_morton3, MORTON_AXIS_MAX, context)
        }
    }
}

fn encode_sort_with_encoder_3d<F>(
    items: &[Bounds3D],
    key_fn: F,
    axis_max: u32,
    context: SortKey3DContext,
) -> Vec<(u64, usize)>
where
    F: Fn(u32, u32, u32) -> u64 + Copy + Sync,
{
    let encode = |i: usize, bounds: &Bounds3D| -> (u64, usize) {
        let x = normalize_center(
            bounds.min_x,
            bounds.max_x,
            context.extent.min_x,
            context.extent.max_x,
            axis_max,
        );
        let y = normalize_center(
            bounds.min_y,
            bounds.max_y,
            context.extent.min_y,
            context.extent.max_y,
            axis_max,
        );
        let z = normalize_center(
            bounds.min_z,
            bounds.max_z,
            context.extent.min_z,
            context.extent.max_z,
            axis_max,
        );
        (key_fn(x, y, z), i)
    };

    #[cfg(feature = "parallel")]
    if context.use_parallel {
        return encode_sort_parallel_3d(items, &encode);
    }

    encode_sort_serial_3d(items, &encode, context.radix, context.radix_bits)
}

pub(crate) fn encode_sort_serial_3d<F>(
    items: &[Bounds3D],
    encode: &F,
    radix: bool,
    radix_bits: u32,
) -> Vec<(u64, usize)>
where
    F: Fn(usize, &Bounds3D) -> (u64, usize),
{
    let mut order = Vec::with_capacity(items.len());
    for (i, bounds) in items.iter().enumerate() {
        order.push(encode(i, bounds));
    }
    if radix {
        radix_sort_pairs_u64(&mut order, radix_bits);
    } else {
        order.sort_unstable_by_key(|&(key, _)| key);
    }
    order
}

#[cfg(feature = "parallel")]
pub(crate) fn encode_sort_parallel_3d<F>(items: &[Bounds3D], encode: &F) -> Vec<(u64, usize)>
where
    F: Fn(usize, &Bounds3D) -> (u64, usize) + Sync,
{
    use rayon::prelude::*;

    let mut order: Vec<(u64, usize)> = items
        .par_iter()
        .enumerate()
        .map(|(i, bounds)| encode(i, bounds))
        .collect();
    order.par_sort_unstable_by_key(|&(key, _)| key);
    order
}

pub(crate) fn default_radix_bits_3d() -> u32 {
    DEFAULT_RADIX_BITS_3D
}

#[inline]
pub(crate) fn normalize_radix_bits_3d(bits: u32) -> u32 {
    bits.clamp(MIN_RADIX_BITS, MAX_RADIX_BITS)
}

#[doc(hidden)]
pub fn radix_sort_pairs_u64(a: &mut [(u64, usize)], bits: u32) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let bits = normalize_radix_bits_3d(bits);
    let buckets = 1usize << bits;
    let mask = (buckets as u64) - 1;
    let passes = 64u32.div_ceil(bits);

    let mut tmp = vec![(0u64, 0usize); n];
    let mut counts = vec![0usize; buckets];

    fn pass(
        src: &[(u64, usize)],
        dst: &mut [(u64, usize)],
        shift: u32,
        mask: u64,
        counts: &mut [usize],
    ) {
        counts.iter_mut().for_each(|count| *count = 0);
        for &(key, _) in src {
            counts[((key >> shift) & mask) as usize] += 1;
        }
        let mut sum = 0usize;
        for count in counts.iter_mut() {
            let current = *count;
            *count = sum;
            sum += current;
        }
        for &pair in src {
            let bucket = ((pair.0 >> shift) & mask) as usize;
            dst[counts[bucket]] = pair;
            counts[bucket] += 1;
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

#[inline]
fn normalize_center(min: f64, max: f64, extent_min: f64, extent_max: f64, axis_max: u32) -> u32 {
    let width = extent_max - extent_min;
    if width <= 0.0 || !width.is_finite() {
        return 0;
    }

    let normalized = ((0.5 * (min + max) - extent_min) / width) * f64::from(axis_max);
    if normalized.is_nan() || normalized <= 0.0 {
        0
    } else if normalized >= f64::from(axis_max) {
        axis_max
    } else {
        normalized as u32
    }
}

#[doc(hidden)]
#[inline]
pub fn encode_morton3(x: u32, y: u32, z: u32) -> u64 {
    split_by_3(u64::from(x)) | (split_by_3(u64::from(y)) << 1) | (split_by_3(u64::from(z)) << 2)
}

#[doc(hidden)]
#[inline]
pub fn encode_hilbert3(x: u32, y: u32, z: u32) -> u64 {
    let mut index = 0u64;
    let mut state = 0usize;

    let mut shift = HILBERT_BITS_PER_AXIS;
    while shift > 0 {
        shift -= 2;
        let m = (((x >> shift) & 3) << 4) | (((y >> shift) & 3) << 2) | ((z >> shift) & 3);
        let entry = HILBERT3_PAIR_LUT[state * 64 + m as usize];
        index = (index << 6) | u64::from(entry & 0x3f);
        state = (entry >> 6) as usize;
    }

    index
}

const fn build_hilbert3_step_lut() -> [u8; 192] {
    let mut table = [0u8; 192];
    let mut state = 0usize;
    while state < 24 {
        let c = (state & 7) as u32;
        let n = (state / 8) as u32;
        let mut m = 0u32;
        while m < 8 {
            let gray = rotate_right_3(c ^ m, n);
            let i = gray_to_integer_3(gray);
            let without_high_bit = gray & 0b011;
            let next_rotation = if without_high_bit == 0 {
                1
            } else if (without_high_bit & 1) != 0 {
                2
            } else {
                3
            };
            let transform = if i == 0 {
                0
            } else {
                let low_bit = i & 0u32.wrapping_sub(i);
                gray ^ (low_bit | 1)
            };
            let next_c = c ^ rotate_left_3(transform, n);
            let next_n = (n + next_rotation) % 3;
            let next_state = next_n * 8 + next_c;
            table[state * 8 + m as usize] = ((next_state as u8) << 3) | (i as u8);
            m += 1;
        }
        state += 1;
    }
    table
}

const fn build_hilbert3_pair_lut() -> [u16; 1536] {
    let mut table = [0u16; 1536];
    let mut state = 0usize;
    while state < 24 {
        let mut m = 0u32;
        while m < 64 {
            let x = (m >> 4) & 3;
            let y = (m >> 2) & 3;
            let z = m & 3;
            let mut next_state = state;
            let mut out = 0u32;
            let mut bit = 2u32;
            while bit > 0 {
                bit -= 1;
                let step_m = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1);
                let entry = HILBERT3_STEP_LUT[next_state * 8 + step_m as usize];
                out = (out << 3) | ((entry & 7) as u32);
                next_state = (entry >> 3) as usize;
            }
            table[state * 64 + m as usize] = ((next_state as u16) << 6) | (out as u16);
            m += 1;
        }
        state += 1;
    }
    table
}

const fn rotate_left_3(value: u32, shift: u32) -> u32 {
    match shift {
        0 => value & 7,
        1 => ((value << 1) | (value >> 2)) & 7,
        _ => ((value << 2) | (value >> 1)) & 7,
    }
}

const fn rotate_right_3(value: u32, shift: u32) -> u32 {
    match shift {
        0 => value & 7,
        1 => ((value >> 1) | (value << 2)) & 7,
        _ => ((value >> 2) | (value << 1)) & 7,
    }
}

const fn gray_to_integer_3(mut gray: u32) -> u32 {
    gray ^= gray >> 1;
    gray ^= gray >> 2;
    gray & 7
}

#[inline]
fn split_by_3(mut value: u64) -> u64 {
    value &= 0x1f_ffff;
    value = (value | (value << 32)) & 0x001f_0000_0000_ffff;
    value = (value | (value << 16)) & 0x001f_0000_ff00_00ff;
    value = (value | (value << 8)) & 0x100f_00f0_0f00_f00f;
    value = (value | (value << 4)) & 0x10c3_0c30_c30c_30c3;
    value = (value | (value << 2)) & 0x1249_2492_4924_9249;
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_hilbert3_stepwise(x: u32, y: u32, z: u32) -> u64 {
        let mut index = 0u64;
        let mut state = 0usize;

        for shift in (0..HILBERT_BITS_PER_AXIS).rev() {
            let m = (((x >> shift) & 1) << 2) | (((y >> shift) & 1) << 1) | ((z >> shift) & 1);
            let entry = HILBERT3_STEP_LUT[state * 8 + m as usize];
            index = (index << 3) | u64::from(entry & 7);
            state = (entry >> 3) as usize;
        }

        index
    }

    #[test]
    fn paired_hilbert3_lut_matches_stepwise_encoder() {
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..32 {
                    assert_eq!(
                        encode_hilbert3(x, y, z),
                        encode_hilbert3_stepwise(x, y, z),
                        "dense mismatch at ({x}, {y}, {z})"
                    );
                }
            }
        }

        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..100_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (seed & HILBERT_AXIS_MAX as u64) as u32;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (seed & HILBERT_AXIS_MAX as u64) as u32;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let z = (seed & HILBERT_AXIS_MAX as u64) as u32;
            assert_eq!(
                encode_hilbert3(x, y, z),
                encode_hilbert3_stepwise(x, y, z),
                "sample mismatch at ({x}, {y}, {z})"
            );
        }
    }
}
