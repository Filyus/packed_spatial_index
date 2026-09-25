//! The four range-search frontends on the same boxes and the same windows, in
//! ONE binary, interleaved: scalar `f64` (the reference every ratio is taken
//! against), scalar `f32`, SIMD `f64` and SIMD `f32`. What the SIMD frontends
//! and compact storage buy depends on the CPU, so this is the bench to run on
//! the machine a claim is about.
//!
//! The SIMD arms go through runtime dispatch: AVX-512, AVX2 or the portable
//! `wide` tier (NEON on aarch64), whichever the CPU offers. The `f32` arms
//! return a few more hits than the `f64` ones -- outward-rounded boxes admit
//! near-boundary candidates -- so their checksums differ by design.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --features simd,f32-storage --bench paired_precision
//! `PAIRED_N` (default 100000) sets the item count; query counts shrink with it
//! so a full-extent window stays affordable.

use std::hint::black_box;

use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index3DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const EXTENT: f64 = 10_000.0;

fn item_count() -> usize {
    std::env::var("PAIRED_N")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(100_000)
}

/// `base` queries at 100k items, fewer as the index grows.
fn queries(base: usize, n: usize) -> usize {
    (base * 100_000 / n.max(1)).clamp(10, base)
}

fn boxes_2d(seed: u64, n: usize) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
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

fn boxes_3d(seed: u64, n: usize) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
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

/// One `search` arm and one `count` arm per frontend, so the two groups differ
/// only in the call.
macro_rules! frontends {
    ($qs:expr, $f64:expr, $f32:expr, $simd:expr, $simd_f32:expr, $dim:literal) => {{
        // Shared references are `Copy`, so every `move` closure gets its own.
        let (qs, a, b, c, d) = ($qs, $f64, $f32, $simd, $simd_f32);
        let search = vec![
            paired::arm(concat!("Index", $dim, "::search (f64)"), move || {
                black_box(qs).iter().map(|q| a.search(*q).len()).sum()
            }),
            paired::arm(concat!("Index", $dim, "F32::search"), move || {
                black_box(qs).iter().map(|q| b.search(*q).len()).sum()
            }),
            paired::arm(concat!("SimdIndex", $dim, "::search"), move || {
                black_box(qs).iter().map(|q| c.search(*q).len()).sum()
            }),
            paired::arm(concat!("SimdIndex", $dim, "F32::search"), move || {
                black_box(qs).iter().map(|q| d.search(*q).len()).sum()
            }),
        ];
        let count = vec![
            paired::arm(concat!("Index", $dim, "::count (f64)"), move || {
                black_box(qs).iter().map(|q| a.count(*q)).sum()
            }),
            paired::arm(concat!("Index", $dim, "F32::count"), move || {
                black_box(qs).iter().map(|q| b.count(*q)).sum()
            }),
            paired::arm(concat!("SimdIndex", $dim, "::count"), move || {
                black_box(qs).iter().map(|q| c.count(*q)).sum()
            }),
            paired::arm(concat!("SimdIndex", $dim, "F32::count"), move || {
                black_box(qs).iter().map(|q| d.count(*q)).sum()
            }),
        ];
        (search, count)
    }};
}

fn main() {
    pin::pin_from_env();
    let n = item_count();
    println!("items: {n}");

    // ---- 2D ----
    let items = boxes_2d(0xB0B, n);
    let build = || {
        let mut b = Index2DBuilder::new(n);
        for &bx in &items {
            b.add(bx);
        }
        b
    };
    let (f64_2d, f32_2d) = (build().finish().unwrap(), build().finish_f32().unwrap());
    let (simd_2d, simd_f32_2d) = (
        build().finish_simd().unwrap(),
        build().finish_simd_f32().unwrap(),
    );
    // The `f32` root box, so every frontend can cover its root. The exact
    // `f64` extent is not `f32`-representable in general: an `f32` index
    // rounds it inward, the root and every node along the data's edge poke
    // out of it, and that band gets tested item by item.
    let extent = f32_2d.extent().unwrap();
    let cases: Vec<(&str, Vec<Box2D>)> = vec![
        (
            "2d small (10..200)",
            windows_2d(0x51A11, 10.0..200.0, queries(10_000, n)),
        ),
        (
            "2d large (2000..5000)",
            windows_2d(0x1A96E, 2000.0..5000.0, queries(1000, n)),
        ),
        ("2d full extent", vec![extent; queries(1000, n)]),
    ];
    for (label, qs) in &cases {
        let (mut search, mut count) =
            frontends!(qs, &f64_2d, &f32_2d, &simd_2d, &simd_f32_2d, "2D");
        paired::run(
            &format!("{label}, search"),
            &mut search,
            "Index2D::search (f64)",
        );
        paired::run(
            &format!("{label}, count"),
            &mut count,
            "Index2D::count (f64)",
        );
    }

    // ---- 3D ----
    let items = boxes_3d(0xB0B3, n);
    let build = || {
        let mut b = Index3DBuilder::new(n);
        for &bx in &items {
            b.add(bx);
        }
        b
    };
    let (f64_3d, f32_3d) = (build().finish().unwrap(), build().finish_f32().unwrap());
    let (simd_3d, simd_f32_3d) = (
        build().finish_simd().unwrap(),
        build().finish_simd_f32().unwrap(),
    );
    let extent = f32_3d.extent().unwrap();
    let cases: Vec<(&str, Vec<Box3D>)> = vec![
        (
            "3d small (10..300)",
            windows_3d(0x51A13, 10.0..300.0, queries(10_000, n)),
        ),
        (
            "3d large (2000..5000)",
            windows_3d(0x1A963, 2000.0..5000.0, queries(1000, n)),
        ),
        ("3d full extent", vec![extent; queries(1000, n)]),
    ];
    for (label, qs) in &cases {
        let (mut search, mut count) =
            frontends!(qs, &f64_3d, &f32_3d, &simd_3d, &simd_f32_3d, "3D");
        paired::run(
            &format!("{label}, search"),
            &mut search,
            "Index3D::search (f64)",
        );
        paired::run(
            &format!("{label}, count"),
            &mut count,
            "Index3D::count (f64)",
        );
    }
}
