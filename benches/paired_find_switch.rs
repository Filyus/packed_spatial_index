//! The two child tests of the depth-first `find` behind `any` and `first`,
//! across the expected-hits range the switch between them reads, in ONE
//! binary, interleaved.
//!
//! `find` descends into a node's first overlapping child as soon as it finds
//! it (`range::find_region`). Its branching form stops testing a node's
//! children at that child, so a query with hits skips the siblings after it;
//! its masked form tests every child into a bitmask, which costs no mispredict
//! on a query that finds nothing and so has nothing to skip. Which one wins
//! turns on how many hits the query expects; where it turns differs by
//! machine and dimension. The shipped `find` picks by the uniform estimate
//! (window fraction of the root box times the item count), as the radius
//! switch does (`join::prefers_mask_2d`).
//!
//! Each group is one window size class over 100 000 boxes: the branching form
//! (the reference), the masked form and the shipped `find`, owned and view,
//! 2D and 3D. The label carries the estimate the switch reads and the mean
//! hit count. 10 000 windows per class, so the predictor cannot learn the set
//! (kb:observation/534).
//!
//! Run:
//!   taskset -c 1 cargo bench --bench paired_find_switch

use std::hint::black_box;
use std::ops::ControlFlow;

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index2DView, Index3D, Index3DBuilder, Index3DView,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;

const N: usize = 100_000;
const EXTENT: f64 = 10_000.0;
const QUERIES: usize = 10_000;

/// Window side ranges: from windows that expect a fraction of a hit to the
/// large class of the other paired benches.
const SIDES_2D: [(f64, f64); 8] = [
    (1.0, 10.0),
    (10.0, 30.0),
    (30.0, 60.0),
    (60.0, 100.0),
    (100.0, 200.0),
    (200.0, 400.0),
    (400.0, 1000.0),
    (2000.0, 5000.0),
];
const SIDES_3D: [(f64, f64); 7] = [
    (10.0, 200.0),
    (200.0, 400.0),
    (400.0, 700.0),
    (700.0, 1000.0),
    (1000.0, 1500.0),
    (1500.0, 2000.0),
    (2000.0, 5000.0),
];

/// One arm: `$call` on every window of `$qs`, the visitor `$f` breaking on the
/// first hit. Each arm is its own closure, so each form compiles into its own
/// loop: behind one closure and a runtime `match`, the arm the match reached
/// last read up to 25% slow on a form that was the same code.
macro_rules! arm {
    ($name:expr, $qs:expr, |$q:ident, $f:ident| $call:expr) => {{
        let qs = $qs;
        paired::arm($name, move || {
            let mut t = 0usize;
            for &$q in black_box(qs) {
                let $f = |i: usize| {
                    t += i;
                    ControlFlow::Break(())
                };
                let _: ControlFlow<()> = $call;
            }
            t
        })
    }};
}

/// Time the forms on one index over one window set.
macro_rules! group {
    ($label:expr, $index:expr, $qs:expr) => {{
        let (index, qs) = (&$index, &$qs);
        let mut arms = vec![
            arm!("branching", qs, |q, f| index
                .find_forced::<false, (), _>(q, f)),
            arm!("masked", qs, |q, f| index.find_forced::<true, (), _>(q, f)),
            arm!("switched (ships)", qs, |q, f| index.find(q, f)),
            arm!("branching again", qs, |q, f| index
                .find_forced::<false, (), _>(q, f)),
        ];
        paired::run(&$label, &mut arms, "branching");
    }};
}

fn label(tag: &str, (lo, hi): (f64, f64), expected: f64, hits: f64) -> String {
    format!("{tag} {lo}..{hi} ({expected:.1} expected, {hits:.1} hits/query)")
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0xF1_2D);
    let boxes: Vec<Box2D> = (0..N)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect();
    let root = boxes.iter().fold(boxes[0], |a, b| {
        Box2D::new(
            a.min_x.min(b.min_x),
            a.min_y.min(b.min_y),
            a.max_x.max(b.max_x),
            a.max_y.max(b.max_y),
        )
    });
    let mut b = Index2DBuilder::new(N);
    for &bx in &boxes {
        b.add(bx);
    }
    let owned: Index2D = b.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    for sides in SIDES_2D {
        let qs: Vec<Box2D> = (0..QUERIES)
            .map(|_| {
                let s: f64 = rng.random_range(sides.0..sides.1);
                let x: f64 = rng.random_range(0.0..EXTENT - s);
                let y: f64 = rng.random_range(0.0..EXTENT - s);
                Box2D::new(x, y, x + s, y + s)
            })
            .collect();
        let n = qs.len() as f64;
        let area = (root.max_x - root.min_x) * (root.max_y - root.min_y);
        let expected = qs
            .iter()
            .map(|q| (q.max_x - q.min_x) * (q.max_y - q.min_y) / area * N as f64)
            .sum::<f64>()
            / n;
        let hits = qs.iter().map(|&q| owned.count(q) as f64).sum::<f64>() / n;
        group!(label("2d owned", sides, expected, hits), owned, qs);
        group!(label("2d view", sides, expected, hits), view, qs);
    }

    let boxes: Vec<Box3D> = (0..N)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            let z: f64 = rng.random_range(0.0..EXTENT);
            let s: f64 = rng.random_range(0.1..60.0);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect();
    let root = boxes.iter().fold(boxes[0], |a, b| {
        Box3D::new(
            a.min_x.min(b.min_x),
            a.min_y.min(b.min_y),
            a.min_z.min(b.min_z),
            a.max_x.max(b.max_x),
            a.max_y.max(b.max_y),
            a.max_z.max(b.max_z),
        )
    });
    let mut b = Index3DBuilder::new(N);
    for &bx in &boxes {
        b.add(bx);
    }
    let owned: Index3D = b.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    for sides in SIDES_3D {
        let qs: Vec<Box3D> = (0..QUERIES)
            .map(|_| {
                let s: f64 = rng.random_range(sides.0..sides.1);
                let x: f64 = rng.random_range(0.0..EXTENT - s);
                let y: f64 = rng.random_range(0.0..EXTENT - s);
                let z: f64 = rng.random_range(0.0..EXTENT - s);
                Box3D::new(x, y, z, x + s, y + s, z + s)
            })
            .collect();
        let n = qs.len() as f64;
        let volume =
            (root.max_x - root.min_x) * (root.max_y - root.min_y) * (root.max_z - root.min_z);
        let expected = qs
            .iter()
            .map(|q| {
                (q.max_x - q.min_x) * (q.max_y - q.min_y) * (q.max_z - q.min_z) / volume * N as f64
            })
            .sum::<f64>()
            / n;
        let hits = qs.iter().map(|&q| owned.count(q) as f64).sum::<f64>() / n;
        group!(label("3d owned", sides, expected, hits), owned, qs);
        group!(label("3d view", sides, expected, hits), view, qs);
    }
}
