//! The radius collect forms (`search_within_into`, `count_within`) on the
//! branching traversal against the masked one, in ONE binary, interleaved,
//! with the box collect path — which neither shape can touch — as the control.
//!
//! Both arms answer identically (the checksum column pins that), so this only
//! asks which is faster, and the answer depends on the query: the sweep runs
//! the radius from "hits almost nothing" to "covers most of the extent" and
//! prints the expected hit fraction beside each case, which is the quantity the
//! shipping switch reads.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --bench paired_within

use std::hint::black_box;

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index3D, Index3DBuilder, force_within_shape,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

fn n_items() -> usize {
    std::env::var("PAIRED_N")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(100_000)
}
const EXTENT: f64 = 10_000.0;
const QUERIES: usize = 1_000;

const BRANCHING: u8 = 1;
const MASKED: u8 = 2;

fn build_2d(seed: u64) -> Index2D {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = n_items();
    let mut b = Index2DBuilder::new(n);
    for _ in 0..n {
        let x: f64 = rng.random_range(0.0..EXTENT);
        let y: f64 = rng.random_range(0.0..EXTENT);
        let w: f64 = rng.random_range(0.1..20.0);
        let h: f64 = rng.random_range(0.1..20.0);
        b.add(Box2D::new(x, y, x + w, y + h));
    }
    b.finish().unwrap()
}

fn build_3d(seed: u64) -> Index3D {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = n_items();
    let mut b = Index3DBuilder::new(n);
    for _ in 0..n {
        let x: f64 = rng.random_range(0.0..EXTENT);
        let y: f64 = rng.random_range(0.0..EXTENT);
        let z: f64 = rng.random_range(0.0..EXTENT);
        let s: f64 = rng.random_range(0.1..20.0);
        b.add(Box3D::new(x, y, z, x + s, y + s, z + s));
    }
    b.finish().unwrap()
}

fn points_2d(seed: u64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..QUERIES)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            Box2D::new(x, y, x, y)
        })
        .collect()
}

fn points_3d(seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..QUERIES)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            let z: f64 = rng.random_range(0.0..EXTENT);
            Box3D::new(x, y, z, x, y, z)
        })
        .collect()
}

/// The quantity the switch reads: the share of the root box covered by the
/// query grown by `max_distance`, averaged over the query set.
fn mean_fraction_2d(root: Box2D, qs: &[Box2D], r: f64) -> f64 {
    let (rw, rh) = (root.max_x - root.min_x, root.max_y - root.min_y);
    qs.iter()
        .map(|q| {
            let w = ((q.max_x + r).min(root.max_x) - (q.min_x - r).max(root.min_x)).max(0.0);
            let h = ((q.max_y + r).min(root.max_y) - (q.min_y - r).max(root.min_y)).max(0.0);
            (w / rw) * (h / rh)
        })
        .sum::<f64>()
        / qs.len() as f64
}

fn mean_fraction_3d(root: Box3D, qs: &[Box3D], r: f64) -> f64 {
    let (rw, rh, rd) = (
        root.max_x - root.min_x,
        root.max_y - root.min_y,
        root.max_z - root.min_z,
    );
    qs.iter()
        .map(|q| {
            let w = ((q.max_x + r).min(root.max_x) - (q.min_x - r).max(root.min_x)).max(0.0);
            let h = ((q.max_y + r).min(root.max_y) - (q.min_y - r).max(root.min_y)).max(0.0);
            let d = ((q.max_z + r).min(root.max_z) - (q.min_z - r).max(root.min_z)).max(0.0);
            (w / rw) * (h / rh) * (d / rd)
        })
        .sum::<f64>()
        / qs.len() as f64
}

fn radii() -> Vec<f64> {
    match std::env::var("PAIRED_RADII") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![1.0, 5.0, 20.0, 60.0, 150.0, 400.0, 1000.0, 2500.0],
    }
}

fn main() {
    pin::pin_from_env();

    // ---- 2D ----
    let index = build_2d(0xB0B);
    let qs = points_2d(0xACE);
    let root = index.extent().unwrap();

    for r in radii() {
        let frac = mean_fraction_2d(root, &qs, r);
        let hits: usize = qs.iter().map(|q| index.count_within(*q, r)).sum();
        let label = format!(
            "2d r={r} (covered {frac:.4}, {:.0} hits/query)",
            hits as f64 / qs.len() as f64
        );
        let (mut out_c, mut out_b, mut out_m, mut out_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut arms = vec![
            // Control: the box overlap collect path, which neither shape touches.
            paired::arm("box search_into (control)", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    let grown = Box2D::new(q.min_x - r, q.min_y - r, q.max_x + r, q.max_y + r);
                    index.search_into(grown, &mut out_c);
                    t += out_c.len();
                }
                t
            }),
            paired::arm("within_into branching", || {
                force_within_shape(BRANCHING);
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into(*q, r, &mut out_b);
                    t += out_b.len();
                }
                t
            }),
            paired::arm("within_into masked", || {
                force_within_shape(MASKED);
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into(*q, r, &mut out_m);
                    t += out_m.len();
                }
                t
            }),
            paired::arm("count_within branching", || {
                force_within_shape(BRANCHING);
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within(*q, r);
                }
                t
            }),
            paired::arm("count_within masked", || {
                force_within_shape(MASKED);
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within(*q, r);
                }
                t
            }),
            // The shipping switch: its ratio should track whichever of the two
            // forced arms above is faster in this row.
            paired::arm("within_into switch (ships)", || {
                force_within_shape(0);
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into(*q, r, &mut out_s);
                    t += out_s.len();
                }
                t
            }),
        ];
        paired::run(&label, &mut arms, "within_into branching");
        force_within_shape(0);
    }

    // ---- 3D ----
    let index = build_3d(0xB0B3);
    let qs = points_3d(0xACE3);
    let root = index.extent().unwrap();

    for r in radii() {
        let frac = mean_fraction_3d(root, &qs, r);
        let hits: usize = qs.iter().map(|q| index.count_within(*q, r)).sum();
        let label = format!(
            "3d r={r} (covered {frac:.4}, {:.0} hits/query)",
            hits as f64 / qs.len() as f64
        );
        let (mut out_c, mut out_b, mut out_m, mut out_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut arms = vec![
            paired::arm("box search_into (control)", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    let grown = Box3D::new(
                        q.min_x - r,
                        q.min_y - r,
                        q.min_z - r,
                        q.max_x + r,
                        q.max_y + r,
                        q.max_z + r,
                    );
                    index.search_into(grown, &mut out_c);
                    t += out_c.len();
                }
                t
            }),
            paired::arm("within_into branching", || {
                force_within_shape(BRANCHING);
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into(*q, r, &mut out_b);
                    t += out_b.len();
                }
                t
            }),
            paired::arm("within_into masked", || {
                force_within_shape(MASKED);
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into(*q, r, &mut out_m);
                    t += out_m.len();
                }
                t
            }),
            paired::arm("count_within branching", || {
                force_within_shape(BRANCHING);
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within(*q, r);
                }
                t
            }),
            paired::arm("count_within masked", || {
                force_within_shape(MASKED);
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within(*q, r);
                }
                t
            }),
            // The shipping switch: its ratio should track whichever of the two
            // forced arms above is faster in this row.
            paired::arm("within_into switch (ships)", || {
                force_within_shape(0);
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into(*q, r, &mut out_s);
                    t += out_s.len();
                }
                t
            }),
        ];
        paired::run(&label, &mut arms, "within_into branching");
        force_within_shape(0);
    }
}
