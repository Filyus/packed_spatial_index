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
//! Only the two forced traversals are timed. The shipping `search_within_into`
//! is deliberately NOT an arm here: it reaches the same two bodies through a
//! different function, so a ratio against these arms would carry an inlining
//! difference rather than the traversal difference, and it reads as a
//! regression that is not there. Which arm the switch picks is pinned
//! separately and deterministically by the unit tests in `src/join.rs`.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --bench paired_within

use std::hint::black_box;

use packed_spatial_index::{Box2D, Box3D, Index2D, Index2DBuilder, Index3D, Index3DBuilder};
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

/// Query points per radius. Every rep replays the same set, and a Zen 4 or
/// Zen 5 predictor learns the traversal of each query in it until the set's
/// hard-to-predict branches outgrow its tables (~30 000 on Zen 5): a radius
/// with 0-2 hits per query was learned over 2000 points and not over 5000
/// (kb:task/191). So the set is sized by the output: 10 000 points up to 20
/// hits per query, 2000 up to 500, 1000 above that, where a query has too many
/// branches to learn and costs too much to repeat. `PAIRED_QUERIES` overrides
/// every radius.
const POINTS_MAX: usize = 10_000;

fn set_size(hits_per_query: f64) -> usize {
    if let Some(n) = std::env::var("PAIRED_QUERIES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
    {
        return n;
    }
    if hits_per_query <= 20.0 {
        10_000
    } else if hits_per_query <= 500.0 {
        2_000
    } else {
        1_000
    }
}

fn points_needed() -> usize {
    std::env::var("PAIRED_QUERIES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(POINTS_MAX)
}

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
    (0..points_needed())
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            Box2D::new(x, y, x, y)
        })
        .collect()
}

fn points_3d(seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..points_needed())
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
    let all = points_2d(0xACE);
    let root = index.extent().unwrap();

    for r in radii() {
        let probe = &all[..all.len().min(1000)];
        let per: usize = probe.iter().map(|q| index.count_within(*q, r)).sum();
        let qs: Vec<Box2D> =
            all[..set_size(per as f64 / probe.len() as f64).min(all.len())].to_vec();
        let frac = mean_fraction_2d(root, &qs, r);
        let hits: usize = qs.iter().map(|q| index.count_within(*q, r)).sum();
        let label = format!(
            "2d r={r} (covered {frac:.4}, {:.0} hits/query, {} points)",
            hits as f64 / qs.len() as f64,
            qs.len()
        );
        let (mut out_c, mut out_b, mut out_m) = (Vec::new(), Vec::new(), Vec::new());
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
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into_forced::<false>(*q, r, &mut out_b);
                    t += out_b.len();
                }
                t
            }),
            paired::arm("within_into masked", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into_forced::<true>(*q, r, &mut out_m);
                    t += out_m.len();
                }
                t
            }),
            paired::arm("count_within branching", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within_forced::<false>(*q, r);
                }
                t
            }),
            paired::arm("count_within masked", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within_forced::<true>(*q, r);
                }
                t
            }),
        ];
        paired::run(&label, &mut arms, "within_into branching");
    }

    // ---- 3D ----
    let index = build_3d(0xB0B3);
    let all = points_3d(0xACE3);
    let root = index.extent().unwrap();

    for r in radii() {
        let probe = &all[..all.len().min(1000)];
        let per: usize = probe.iter().map(|q| index.count_within(*q, r)).sum();
        let qs: Vec<Box3D> =
            all[..set_size(per as f64 / probe.len() as f64).min(all.len())].to_vec();
        let frac = mean_fraction_3d(root, &qs, r);
        let hits: usize = qs.iter().map(|q| index.count_within(*q, r)).sum();
        let label = format!(
            "3d r={r} (covered {frac:.4}, {:.0} hits/query, {} points)",
            hits as f64 / qs.len() as f64,
            qs.len()
        );
        let (mut out_c, mut out_b, mut out_m) = (Vec::new(), Vec::new(), Vec::new());
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
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into_forced::<false>(*q, r, &mut out_b);
                    t += out_b.len();
                }
                t
            }),
            paired::arm("within_into masked", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    index.search_within_into_forced::<true>(*q, r, &mut out_m);
                    t += out_m.len();
                }
                t
            }),
            paired::arm("count_within branching", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within_forced::<false>(*q, r);
                }
                t
            }),
            paired::arm("count_within masked", || {
                let mut t = 0;
                for q in black_box(&qs) {
                    t += index.count_within_forced::<true>(*q, r);
                }
                t
            }),
        ];
        paired::run(&label, &mut arms, "within_into branching");
    }
}
