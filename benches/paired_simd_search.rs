//! The portable SIMD search kernel's internal-node dispatch: per-4-lane
//! against mask-first (the shape that ships, and the one `count_simd_impl`
//! landed on), in ONE binary, interleaved, with the owned `search_into` as the
//! control neither shape can touch. Both arms return identical results, which
//! the checksum column pins, so this only asks which is faster.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --features simd --bench paired_simd_search

use std::hint::black_box;

use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index3DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const N: usize = 100_000;
const EXTENT: f64 = 10_000.0;

fn boxes_2d(seed: u64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn windows_2d(seed: u64, side: std::ops::Range<f64>, count: usize) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let s: f64 = rng.random_range(side.clone());
            let x: f64 = rng.random_range(0.0..EXTENT - s);
            let y: f64 = rng.random_range(0.0..EXTENT - s);
            Box2D::new(x, y, x + s, y + s)
        })
        .collect()
}

fn boxes_3d(seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            let (x, y, z): (f64, f64, f64) = (
                rng.random_range(0.0..EXTENT),
                rng.random_range(0.0..EXTENT),
                rng.random_range(0.0..EXTENT),
            );
            let s: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect()
}

fn windows_3d(seed: u64, side: std::ops::Range<f64>, count: usize) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
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

fn main() {
    pin::pin_from_env();

    let items = boxes_2d(0xB0B);
    let mut b = Index2DBuilder::new(N);
    let mut s = Index2DBuilder::new(N);
    for &bx in &items {
        b.add(bx);
        s.add(bx);
    }
    let owned = b.finish().unwrap();
    let simd = s.finish_simd().unwrap();
    let extent = owned.extent().unwrap();
    let cases: Vec<(&str, Vec<Box2D>)> = vec![
        ("2d small (10..200)", windows_2d(0x51A11, 10.0..200.0, 1000)),
        (
            "2d large (2000..5000)",
            windows_2d(0x1A96E, 2000.0..5000.0, 1000),
        ),
        ("2d full extent", vec![extent; 1000]),
    ];
    for (label, qs) in &cases {
        let (mut o0, mut o1, mut oc) = (Vec::new(), Vec::new(), Vec::new());
        let (mut st0, mut st1) = (Vec::new(), Vec::new());
        let mut arms = vec![
            paired::arm("owned Index2D::search_into (control)", || {
                let mut t = 0;
                for q in black_box(qs) {
                    owned.search_into(*q, &mut oc);
                    t += oc.len();
                }
                t
            }),
            paired::arm("simd search shape 0 (per 4 lanes)", || {
                let mut t = 0;
                for q in black_box(qs) {
                    simd.search_shape::<0>(*q, &mut o0, &mut st0);
                    t += o0.len();
                }
                t
            }),
            paired::arm("simd search shape 1 (node mask, ships)", || {
                let mut t = 0;
                for q in black_box(qs) {
                    simd.search_shape::<1>(*q, &mut o1, &mut st1);
                    t += o1.len();
                }
                t
            }),
        ];
        paired::run(label, &mut arms, "simd search shape 0 (per 4 lanes)");
    }

    // ---- 3D ----
    let items = boxes_3d(0xB0B3);
    let mut b = Index3DBuilder::new(N);
    let mut s = Index3DBuilder::new(N);
    for &bx in &items {
        b.add(bx);
        s.add(bx);
    }
    let owned = b.finish().unwrap();
    let simd = s.finish_simd().unwrap();
    let extent = owned.extent().unwrap();
    let cases: Vec<(&str, Vec<Box3D>)> = vec![
        ("3d small (10..300)", windows_3d(0x51A13, 10.0..300.0, 1000)),
        (
            "3d large (2000..5000)",
            windows_3d(0x1A963, 2000.0..5000.0, 1000),
        ),
        ("3d full extent", vec![extent; 1000]),
    ];
    for (label, qs) in &cases {
        let (mut o0, mut o1, mut oc) = (Vec::new(), Vec::new(), Vec::new());
        let (mut st0, mut st1) = (Vec::new(), Vec::new());
        let mut arms = vec![
            paired::arm("owned Index3D::search_into (control)", || {
                let mut t = 0;
                for q in black_box(qs) {
                    owned.search_into(*q, &mut oc);
                    t += oc.len();
                }
                t
            }),
            paired::arm("simd search shape 0 (per 4 lanes)", || {
                let mut t = 0;
                for q in black_box(qs) {
                    simd.search_shape::<0>(*q, &mut o0, &mut st0);
                    t += o0.len();
                }
                t
            }),
            paired::arm("simd search shape 1 (node mask, ships)", || {
                let mut t = 0;
                for q in black_box(qs) {
                    simd.search_shape::<1>(*q, &mut o1, &mut st1);
                    t += o1.len();
                }
                t
            }),
        ];
        paired::run(label, &mut arms, "simd search shape 0 (per 4 lanes)");
    }
}
