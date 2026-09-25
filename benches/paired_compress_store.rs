//! AVX-512 result collection, in ONE binary, interleaved: `VPCOMPRESSQ`
//! straight to memory — the form every AVX-512 kernel used, microcoded on Zen 4
//! at about 142 cycles an instruction — against a compress into a register and
//! a full store, the form they ship. Range search on the four SIMD indexes and
//! all-hits raycast in 2D and 3D, with the owned scalar index as the control
//! neither form touches. Both forms return the same hits, which the checksum
//! column pins.
//!
//! Without AVX-512 both forms fall back to the same dispatch; the first line of
//! the output says which it was.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --features simd,f32-storage --bench paired_compress_store

use std::hint::black_box;

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index3DBuilder, Point2D, Point3D, Ray2D, Ray3D,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const N: usize = 100_000;
const EXTENT: f64 = 10_000.0;
/// Every rep replays the same queries, and a Zen 4 or Zen 5 predictor learns
/// each one's traversal while the set is small; 1000 small windows flattered
/// the SIMD arms by 10-25 points (kb:task/191). Small outputs get 10 000
/// queries, large ones keep 1000, where a query costs too much to repeat.
const SMALL_SET: usize = 10_000;
const LARGE_SET: usize = 1000;

fn boxes_2d(seed: u64, max_side: f64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            let (w, h): (f64, f64) = (
                rng.random_range(0.1..max_side),
                rng.random_range(0.1..max_side),
            );
            let x: f64 = rng.random_range(0.0..EXTENT - w);
            let y: f64 = rng.random_range(0.0..EXTENT - h);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn boxes_3d(seed: u64, max_side: f64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            let s: f64 = rng.random_range(0.1..max_side);
            let (x, y, z): (f64, f64, f64) = (
                rng.random_range(0.0..EXTENT - s),
                rng.random_range(0.0..EXTENT - s),
                rng.random_range(0.0..EXTENT - s),
            );
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect()
}

fn windows_2d(seed: u64, side: std::ops::Range<f64>, n: usize) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let s: f64 = rng.random_range(side.clone());
            let x: f64 = rng.random_range(0.0..EXTENT - s);
            let y: f64 = rng.random_range(0.0..EXTENT - s);
            Box2D::new(x, y, x + s, y + s)
        })
        .collect()
}

fn windows_3d(seed: u64, side: std::ops::Range<f64>, n: usize) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let s: f64 = rng.random_range(side.clone());
            let (x, y, z): (f64, f64, f64) = (
                rng.random_range(0.0..EXTENT - s),
                rng.random_range(0.0..EXTENT - s),
                rng.random_range(0.0..EXTENT - s),
            );
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect()
}

/// Rays entering the square from below and crossing all of it.
fn rays_2d(seed: u64) -> Vec<Ray2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..LARGE_SET)
        .map(|_| {
            Ray2D::new(
                Point2D::new(rng.random_range(0.0..EXTENT), -10.0),
                rng.random_range(-0.15..0.15),
                1.0,
                EXTENT * 2.0,
            )
        })
        .collect()
}

/// Rays entering the cube from below and crossing all of it.
fn rays_3d(seed: u64) -> Vec<Ray3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..LARGE_SET)
        .map(|_| {
            Ray3D::new(
                Point3D::new(
                    rng.random_range(0.0..EXTENT),
                    rng.random_range(0.0..EXTENT),
                    -10.0,
                ),
                rng.random_range(-0.15..0.15),
                rng.random_range(-0.15..0.15),
                1.0,
                EXTENT * 2.0,
            )
        })
        .collect()
}

/// One paired group: the control, then the two compress forms, every ratio
/// against the old memory form. `PAIRED_ONLY` (comma-separated substrings of
/// the group names) runs just the matching groups.
fn forms<Q: Copy>(
    label: &str,
    qs: &[Q],
    control: impl Fn(Q, &mut Vec<usize>),
    forced: impl Fn(bool, Q, &mut Vec<usize>),
) {
    if let Ok(only) = std::env::var("PAIRED_ONLY")
        && !only.split(',').any(|s| label.contains(s.trim()))
    {
        return;
    }
    let (mut oc, mut om, mut or) = (Vec::new(), Vec::new(), Vec::new());
    let hits: usize = qs
        .iter()
        .map(|&q| {
            forced(true, q, &mut or);
            or.len()
        })
        .sum();
    let title = format!("{label} ({} hits/query)", hits / qs.len().max(1));
    let mut arms = vec![
        paired::arm("owned scalar (control)", || {
            let mut t = 0;
            for &q in black_box(qs) {
                control(q, &mut oc);
                t += oc.len();
            }
            t
        }),
        paired::arm("compress to memory (old)", || {
            let mut t = 0;
            for &q in black_box(qs) {
                forced(false, q, &mut om);
                t += om.len();
            }
            t
        }),
        paired::arm("compress in register (ships)", || {
            let mut t = 0;
            for &q in black_box(qs) {
                forced(true, q, &mut or);
                t += or.len();
            }
            t
        }),
    ];
    paired::run(&title, &mut arms, "compress to memory (old)");
}

fn main() {
    pin::pin_from_env();
    #[cfg(target_arch = "x86_64")]
    let avx512 = std::is_x86_feature_detected!("avx512f");
    #[cfg(not(target_arch = "x86_64"))]
    let avx512 = false;
    println!(
        "avx512f: {}",
        if avx512 {
            "yes, the forms differ"
        } else {
            "no, both forms run the same fallback"
        }
    );

    // ---- range search, f64 and f32 ----
    let b2 = boxes_2d(0xB0B, 20.0);
    let build2 = || {
        let mut b = Index2DBuilder::new(N);
        for &bx in &b2 {
            b.add(bx);
        }
        b
    };
    let (owned2, simd2, simd2f) = (
        build2().finish().unwrap(),
        build2().finish_simd().unwrap(),
        build2().finish_simd_f32().unwrap(),
    );
    let b3 = boxes_3d(0xB0B3, 20.0);
    let build3 = || {
        let mut b = Index3DBuilder::new(N);
        for &bx in &b3 {
            b.add(bx);
        }
        b
    };
    let (owned3, simd3, simd3f) = (
        build3().finish().unwrap(),
        build3().finish_simd().unwrap(),
        build3().finish_simd_f32().unwrap(),
    );

    for (name, seed, side, n) in [
        ("small", 0x51A11, 10.0..200.0, SMALL_SET),
        ("large", 0x1A96E, 2000.0..5000.0, LARGE_SET),
    ] {
        let qs = windows_2d(seed, side.clone(), n);
        let control = |q, out: &mut Vec<usize>| owned2.search_into(q, out);
        forms(
            &format!("SimdIndex2D search {name}"),
            &qs,
            control,
            |reg, q, out| {
                if reg {
                    simd2.search_avx512_compress_into::<true>(q, out)
                } else {
                    simd2.search_avx512_compress_into::<false>(q, out)
                }
            },
        );
        forms(
            &format!("SimdIndex2DF32 search {name}"),
            &qs,
            control,
            |reg, q, out| {
                if reg {
                    simd2f.search_avx512_compress_into::<true>(q, out)
                } else {
                    simd2f.search_avx512_compress_into::<false>(q, out)
                }
            },
        );
        let qs = windows_3d(seed + 3, side, n);
        let control = |q, out: &mut Vec<usize>| owned3.search_into(q, out);
        forms(
            &format!("SimdIndex3D search {name}"),
            &qs,
            control,
            |reg, q, out| {
                if reg {
                    simd3.search_avx512_compress_into::<true>(q, out)
                } else {
                    simd3.search_avx512_compress_into::<false>(q, out)
                }
            },
        );
        forms(
            &format!("SimdIndex3DF32 search {name}"),
            &qs,
            control,
            |reg, q, out| {
                if reg {
                    simd3f.search_avx512_compress_into::<true>(q, out)
                } else {
                    simd3f.search_avx512_compress_into::<false>(q, out)
                }
            },
        );
    }

    // ---- all-hits raycast, on scenes dense enough to collect something ----
    let b2 = boxes_2d(0x7B2, 60.0);
    let owned2 = {
        let mut b = Index2DBuilder::new(N);
        b2.iter().for_each(|&bx| b.add(bx));
        b.finish().unwrap()
    };
    let simd2 = {
        let mut b = Index2DBuilder::new(N);
        b2.iter().for_each(|&bx| b.add(bx));
        b.finish_simd().unwrap()
    };
    forms(
        "SimdIndex2D raycast",
        &rays_2d(0x7A2),
        |r, out: &mut Vec<usize>| owned2.raycast_into(r, out),
        |reg, r, out| {
            if reg {
                simd2.raycast_avx512_compress_into::<true>(r, out)
            } else {
                simd2.raycast_avx512_compress_into::<false>(r, out)
            }
        },
    );
    let b3 = boxes_3d(0x7B, 900.0);
    let owned3 = {
        let mut b = Index3DBuilder::new(N);
        b3.iter().for_each(|&bx| b.add(bx));
        b.finish().unwrap()
    };
    let simd3 = {
        let mut b = Index3DBuilder::new(N);
        b3.iter().for_each(|&bx| b.add(bx));
        b.finish_simd().unwrap()
    };
    forms(
        "SimdIndex3D raycast",
        &rays_3d(0x7A),
        |r, out: &mut Vec<usize>| owned3.raycast_into(r, out),
        |reg, r, out| {
            if reg {
                simd3.raycast_avx512_compress_into::<true>(r, out)
            } else {
                simd3.raycast_avx512_compress_into::<false>(r, out)
            }
        },
    );
}
