//! AVX2 "left-pack" of `u64` index lanes — the SoA SIMD search/raycast kernels'
//! result collection on AVX2 CPUs, which lack AVX-512's `VPCOMPRESSQ`.
//!
//! A 4-wide box test yields a 4-bit overlap mask; `LEFTPACK_LUT[mask]` is a
//! `VPERMD` control that moves the matching `u64` lanes to the front of the
//! register, so a single unaligned 256-bit store writes the matches contiguously.
//! The caller advances its length by `popcount(mask)` and must reserve slack for
//! the unconditional 4-lane store. See `docs/simd.md`.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// `VPERMD` controls indexed by a 4-bit overlap mask: each entry maps the set
/// `u64` lanes (two `u32` halves each) to the front. Built at compile time.
pub(crate) const LEFTPACK_LUT: [[i32; 8]; 16] = {
    let mut lut = [[0i32; 8]; 16];
    let mut m = 0usize;
    while m < 16 {
        let mut j = 0usize;
        let mut k = 0usize;
        while k < 4 {
            if m & (1 << k) != 0 {
                lut[m][2 * j] = (2 * k) as i32;
                lut[m][2 * j + 1] = (2 * k + 1) as i32;
                j += 1;
            }
            k += 1;
        }
        m += 1;
    }
    lut
};

/// Left-pack the four `u64` at `src` selected by the low four bits of `mask` into
/// `dst`, returning how many were written. Writes a full 256-bit (four-`u64`)
/// store regardless of `popcount`, so `dst` must have room for four `u64`.
///
/// # Safety
/// AVX2 must be available; `src` must be readable for four `u64` and `dst`
/// writable for four `u64`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn leftpack4(src: *const usize, mask: u32, dst: *mut usize) -> usize {
    let lane = (mask & 0xF) as usize;
    // SAFETY: caller guarantees avx2 + 4-u64 readable `src` / writable `dst`.
    unsafe {
        let idx = _mm256_loadu_si256(src as *const __m256i);
        let ctrl = _mm256_loadu_si256(LEFTPACK_LUT[lane].as_ptr() as *const __m256i);
        let packed = _mm256_permutevar8x32_epi32(idx, ctrl);
        _mm256_storeu_si256(dst as *mut __m256i, packed);
    }
    lane.count_ones() as usize
}
