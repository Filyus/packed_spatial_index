//! Alternative Hilbert-curve encoders: `(x: u16, y: u16) -> u32`.
//!
//! Every function in this module must produce the exact same **bit-for-bit** result as
//! the reference `static_aabb2d_index::hilbert_xy_to_index`. This keeps performance
//! comparisons isolated from index semantics.
//!
//! Three approaches are implemented:
//!  * [`magic_bits`]    — "magic bits" (a rawrunprotected port), like the reference implementation;
//!  * [`loop_rotation`] — the classic iterative xy2d algorithm with quadrant rotations;
//!  * [`lut`]           — a table-driven finite-state machine (4 states, 2 bits per step).

/// Encoder signature: 2D coordinates (16 bits each) -> Hilbert-curve value.
pub type HilbertFn = fn(u16, u16) -> u32;

/// This crate's implementations with human-readable names (for benches/tests).
pub const ENCODERS: &[(&str, HilbertFn)] = &[
    ("magic_bits", magic_bits),
    ("loop_rotation", loop_rotation),
    ("lut", lut),
];

/// "Magic bits": a port of the rawrunprotected algorithm (threadlocalmutex.com, public domain).
///
/// The same algorithm family used inside `static_aabb2d_index`. It contains
/// no branches or memory lookups, only XOR/AND/shifts, so it is very friendly to
/// pipelining and vectorization.
#[inline]
pub fn magic_bits(x: u16, y: u16) -> u32 {
    let x = x as u32;
    let y = y as u32;

    let mut a_1 = x ^ y;
    let mut b_1 = 0xFFFF ^ a_1;
    let mut c_1 = 0xFFFF ^ (x | y);
    let mut d_1 = x & (y ^ 0xFFFF);

    let mut a_2 = a_1 | (b_1 >> 1);
    let mut b_2 = (a_1 >> 1) ^ a_1;
    let mut c_2 = ((c_1 >> 1) ^ (b_1 & (d_1 >> 1))) ^ c_1;
    let mut d_2 = ((a_1 & (c_1 >> 1)) ^ (d_1 >> 1)) ^ d_1;

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 2)) ^ (b_1 & (b_1 >> 2));
    b_2 = (a_1 & (b_1 >> 2)) ^ (b_1 & ((a_1 ^ b_1) >> 2));
    c_2 ^= (a_1 & (c_1 >> 2)) ^ (b_1 & (d_1 >> 2));
    d_2 ^= (b_1 & (c_1 >> 2)) ^ ((a_1 ^ b_1) & (d_1 >> 2));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 4)) ^ (b_1 & (b_1 >> 4));
    b_2 = (a_1 & (b_1 >> 4)) ^ (b_1 & ((a_1 ^ b_1) >> 4));
    c_2 ^= (a_1 & (c_1 >> 4)) ^ (b_1 & (d_1 >> 4));
    d_2 ^= (b_1 & (c_1 >> 4)) ^ ((a_1 ^ b_1) & (d_1 >> 4));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    c_2 ^= (a_1 & (c_1 >> 8)) ^ (b_1 & (d_1 >> 8));
    d_2 ^= (b_1 & (c_1 >> 8)) ^ ((a_1 ^ b_1) & (d_1 >> 8));

    a_1 = c_2 ^ (c_2 >> 1);
    b_1 = d_2 ^ (d_2 >> 1);

    let mut i0 = x ^ y;
    let mut i1 = b_1 | (0xFFFF ^ (i0 | a_1));

    i0 = (i0 | (i0 << 8)) & 0x00FF00FF;
    i0 = (i0 | (i0 << 4)) & 0x0F0F0F0F;
    i0 = (i0 | (i0 << 2)) & 0x33333333;
    i0 = (i0 | (i0 << 1)) & 0x55555555;

    i1 = (i1 | (i1 << 8)) & 0x00FF00FF;
    i1 = (i1 | (i1 << 4)) & 0x0F0F0F0F;
    i1 = (i1 | (i1 << 2)) & 0x33333333;
    i1 = (i1 | (i1 << 1)) & 0x55555555;

    (i1 << 1) | i0
}

/// Number of bits per coordinate (16 -> u16 input, u32 output).
const BITS: u32 = 16;

/// Batch encoder: computes `magic_bits` for a whole slice. The algorithm is branchless, so
/// the loop is a candidate for autovectorization (8 x u32 on AVX2). Slice lengths must match.
pub fn magic_bits_batch(xs: &[u16], ys: &[u16], out: &mut [u32]) {
    let n = out.len();
    assert!(xs.len() == n && ys.len() == n);
    for i in 0..n {
        out[i] = magic_bits(xs[i], ys[i]);
    }
}

/// Z-order (Morton) code: interleaves bits from x and y. This is also a space-filling
/// curve (cheaper than Hilbert), but with slightly worse locality at quadrant boundaries.
/// Used as a comparison point for index sort keys (NOT equivalent to the reference).
#[inline]
pub fn morton(x: u16, y: u16) -> u32 {
    #[inline]
    fn part1by1(mut n: u32) -> u32 {
        n &= 0x0000FFFF;
        n = (n | (n << 8)) & 0x00FF00FF;
        n = (n | (n << 4)) & 0x0F0F0F0F;
        n = (n | (n << 2)) & 0x33333333;
        n = (n | (n << 1)) & 0x55555555;
        n
    }
    part1by1(x as u32) | (part1by1(y as u32) << 1)
}

/// Classic iterative algorithm mapping (x, y) to a Hilbert-curve index.
///
/// At each of the 16 levels it extracts one x bit and one y bit, computes the quadrant
/// number `(3*rx) ^ ry` (traversal order 0,1,3,2), then rotates the lower bits with an
/// inner helper so the sub-quadrant is traversed in the correct orientation.
#[inline]
pub fn loop_rotation(x: u16, y: u16) -> u32 {
    let mut x = x as u32;
    let mut y = y as u32;
    let mut d: u32 = 0;

    let mut s: u32 = 1 << (BITS - 1);
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        // Accumulate the level digit in base-4 by shifting (MSB-first) instead of multiplying by s*s:
        // the first (highest) level ends up in the high bits of d.
        d = (d << 2) | ((3 * rx) ^ ry);
        rot(s, &mut x, &mut y, rx, ry);
        s >>= 1;
    }
    d
}

/// Quadrant rotation/reflection for [`loop_rotation`].
#[inline]
fn rot(n: u32, x: &mut u32, y: &mut u32, rx: u32, ry: u32) {
    if ry == 0 {
        if rx == 1 {
            // reflect around the center of the current n..2n-1 block
            *x = (2 * n - 1).wrapping_sub(*x);
            *y = (2 * n - 1).wrapping_sub(*y);
        }
        std::mem::swap(x, y);
    }
}

// --- Table-driven implementation (finite-state machine) -------------------------------------------

/// Automaton state is an element of the Z2 x Z2 symmetry group: bit 0 = axis swap,
/// bit 1 = flip. The base rules are the same as in [`loop_rotation`], so
/// the result is identical.
///
/// The tables are **coarsened**: one step processes a whole coordinate nibble (4 levels
/// at once), so the index is assembled in 4 iterations instead of 16. This shortens
/// the dependent `state -> state` lookup chain by 4x, the main bottleneck of the fine-grained automaton.
struct LutTables {
    /// emit[state][xnib<<4 | ynib] -> 8-bit index fragment (4 digits of 2 bits each).
    emit: [[u8; 256]; 4],
    /// next[state][xnib<<4 | ynib] -> state after 4 levels.
    next: [[u8; 256]; 4],
}

static LUT: LutTables = build_lut();

const fn build_lut() -> LutTables {
    // 1) small 1-level automaton (derived analytically from quadrant and rotation rules)
    let mut f_emit = [[0u8; 4]; 4];
    let mut f_next = [[0u8; 4]; 4];
    let mut state = 0usize;
    while state < 4 {
        let swap = (state & 1) as u8;
        let flip = ((state >> 1) & 1) as u8;
        let mut q = 0usize;
        while q < 4 {
            let mut a = ((q >> 1) & 1) as u8;
            let mut b = (q & 1) as u8;
            if flip == 1 {
                a ^= 1;
                b ^= 1;
            }
            if swap == 1 {
                let tmp = a;
                a = b;
                b = tmp;
            }
            let (rx, ry) = (a as u32, b as u32);
            f_emit[state][q] = ((3 * rx) ^ ry) as u8;

            let (rot_swap, rot_flip) = if ry == 1 {
                (0u8, 0u8) // identity
            } else if rx == 0 {
                (1u8, 0u8) // swap
            } else {
                (1u8, 1u8) // reflect + swap
            };
            f_next[state][q] = (rot_swap ^ swap) | ((rot_flip ^ flip) << 1);
            q += 1;
        }
        state += 1;
    }

    // 2) compose 4 levels -> "one nibble per step" table
    let mut emit = [[0u8; 256]; 4];
    let mut next = [[0u8; 256]; 4];
    let mut s = 0usize;
    while s < 4 {
        let mut idx = 0usize;
        while idx < 256 {
            let xnib = (idx >> 4) & 0xF;
            let ynib = idx & 0xF;
            let mut st = s;
            let mut out = 0u32;
            let mut bit = 4usize;
            while bit > 0 {
                bit -= 1;
                let a = (xnib >> bit) & 1;
                let b = (ynib >> bit) & 1;
                let q = (a << 1) | b;
                out = (out << 2) | f_emit[st][q] as u32;
                st = f_next[st][q] as usize;
            }
            emit[s][idx] = out as u8;
            next[s][idx] = st as u8;
            idx += 1;
        }
        s += 1;
    }

    LutTables { emit, next }
}

/// Table-driven encoder: a 4-state finite-state machine processing one coordinate nibble
/// (2 x 4 input bits -> 8 output bits) per step through precomputed tables (2 KiB, entirely
/// in L1). Only 4 iterations versus 16 in the fine-grained version.
#[inline]
pub fn lut(x: u16, y: u16) -> u32 {
    let x = x as u32;
    let y = y as u32;
    let mut state = 0usize;
    let mut d: u32 = 0;

    let mut shift = BITS;
    while shift > 0 {
        shift -= 4;
        let xnib = (x >> shift) & 0xF;
        let ynib = (y >> shift) & 0xF;
        let idx = ((xnib << 4) | ynib) as usize;
        d = (d << 8) | LUT.emit[state][idx] as u32;
        state = LUT.next[state][idx] as usize;
    }
    d
}
