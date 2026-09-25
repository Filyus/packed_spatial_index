//! Mask-and-iterate against the per-child branch it replaced, on every collect
//! path that uses it, in ONE binary, interleaved.
//!
//! Every collect path with no early exit folds a node's child tests into a u64
//! mask and walks its set bits (b1c7a24 and the commits after it). That was
//! calibrated on x86 only. On aarch64 the 2D radius mask turned out slower than
//! the branch at every radius (kb:observation/528), and the likely reason --
//! NEON has no movemask -- applies to every other mask site too. Each path now
//! keeps both forms behind a `const MASKED: bool`; the shipping callers pass
//! the target's choice (`true`, except the `f64` 2D paths on aarch64 after this
//! bench found them losing there). The `*_forced` hooks timed here reach
//! both.
//!
//! Each group times one path: the branching form (the reference), the masked
//! form, and a control that neither touches -- `any` on the same index, which
//! is a callback path, so it builds the same mask on x86 whichever form the
//! group times. The checksum column pins that
//! both forms return the same hits.
//!
//! Families: owned `Index2D`, `Index2DView` and `Index3DView` over the same
//! bytes, `Index3D` raycast, and the scalar f32 indexes. Owned `Index3D` range
//! search has no mask and is not here.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --features f32-storage --bench paired_mask_forms

use std::hint::black_box;
use std::ops::ControlFlow;

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index2DView, Index3D, Index3DBuilder, Index3DView,
    Point3D, Ray3D,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const N: usize = 100_000;
const EXTENT: f64 = 10_000.0;

/// Every rep replays the same query set, and a Zen 4 or Zen 5 predictor learns
/// the traversal of each query in it until the set's hard-to-predict branches
/// outgrow its tables (~30 000 on Zen 5): 400 small windows were learned, and
/// the branching form looked 1.4x faster than it is (kb:observation/534,
/// kb:task/191). So each window class gets a set sized by its output: small
/// windows 10 000, mid windows and rays 2000, large windows 400, where one
/// query has too many branches to learn and costs too much to repeat.
/// `PAIRED_QUERIES` overrides every class.
fn queries(default: usize) -> usize {
    std::env::var("PAIRED_QUERIES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// (label, side range, set size): small windows hit a handful, large ones
/// thousands.
const WINDOWS: [(&str, f64, f64, usize); 3] = [
    ("small (10..200)", 10.0, 200.0, 10_000),
    ("mid (200..1000)", 200.0, 1000.0, 2_000),
    ("large (2000..5000)", 2000.0, 5000.0, 400),
];

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

fn boxes_3d(seed: u64, max_side: f64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            let z: f64 = rng.random_range(0.0..EXTENT);
            let s: f64 = rng.random_range(0.1..max_side);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect()
}

fn windows_2d(seed: u64, lo: f64, hi: f64, n: usize) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..queries(n))
        .map(|_| {
            let s: f64 = rng.random_range(lo..hi);
            let x: f64 = rng.random_range(0.0..EXTENT - s);
            let y: f64 = rng.random_range(0.0..EXTENT - s);
            Box2D::new(x, y, x + s, y + s)
        })
        .collect()
}

fn windows_3d(seed: u64, lo: f64, hi: f64, n: usize) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..queries(n))
        .map(|_| {
            let s: f64 = rng.random_range(lo..hi);
            let x: f64 = rng.random_range(0.0..EXTENT - s);
            let y: f64 = rng.random_range(0.0..EXTENT - s);
            let z: f64 = rng.random_range(0.0..EXTENT - s);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect()
}

fn build_2d(boxes: &[Box2D]) -> Index2D {
    let mut b = Index2DBuilder::new(boxes.len());
    for &bx in boxes {
        b.add(bx);
    }
    b.finish().unwrap()
}

fn build_3d(boxes: &[Box3D]) -> Index3D {
    let mut b = Index3DBuilder::new(boxes.len());
    for &bx in boxes {
        b.add(bx);
    }
    b.finish().unwrap()
}

/// Rays entering the cube from below and crossing all of it, spread so a ray
/// meets tens to hundreds of boxes.
fn rays(seed: u64) -> Vec<Ray3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..queries(2_000))
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

macro_rules! pair {
    ($label:expr, $any:expr, $qs:expr, $forced:expr) => {{
        let qs = $qs;
        let (mut out_b, mut out_m) = (Vec::new(), Vec::new());
        let mut arms = vec![
            paired::arm("any (control)", || {
                let mut t = 0usize;
                for q in black_box(qs) {
                    t += usize::from($any(*q));
                }
                t
            }),
            paired::arm("branching", || {
                let mut t = 0usize;
                for q in black_box(qs) {
                    $forced(false, *q, &mut out_b);
                    t += out_b.len();
                }
                t
            }),
            paired::arm("masked (ships)", || {
                let mut t = 0usize;
                for q in black_box(qs) {
                    $forced(true, *q, &mut out_m);
                    t += out_m.len();
                }
                t
            }),
        ];
        paired::run(&$label, &mut arms, "branching");
    }};
}

fn hits_label<Q: Copy>(family: &str, window: &str, qs: &[Q], count: impl Fn(Q) -> usize) -> String {
    let hits: usize = qs.iter().map(|q| count(*q)).sum();
    format!(
        "{family} {window} ({:.0} hits/query)",
        hits as f64 / qs.len() as f64
    )
}

fn main() {
    pin::pin_from_env();

    // ---- 2D: owned, view ----
    let b2 = boxes_2d(0x2D);
    let owned2 = build_2d(&b2);
    let bytes2 = owned2.to_bytes();
    let view2 = Index2DView::from_bytes(&bytes2).unwrap();
    for (i, (name, lo, hi, n)) in WINDOWS.iter().enumerate() {
        let qs = windows_2d(0x51 + i as u64, *lo, *hi, *n);
        let label = hits_label("2d owned", name, &qs, |q| owned2.count(q));
        pair!(
            label,
            |q| owned2.any(q),
            &qs,
            |m: bool, q, out: &mut Vec<usize>| {
                if m {
                    owned2.search_into_forced::<true>(q, out)
                } else {
                    owned2.search_into_forced::<false>(q, out)
                }
            }
        );
        let label = hits_label("2d view", name, &qs, |q| view2.count(q));
        pair!(
            label,
            |q| view2.any(q),
            &qs,
            |m: bool, q, out: &mut Vec<usize>| {
                if m {
                    view2.search_into_forced::<true>(q, out)
                } else {
                    view2.search_into_forced::<false>(q, out)
                }
            }
        );
    }

    // ---- Callback paths: full visit, and `first` (what `any` runs) ----
    // `visit_with_stack` carries `visit`; `find_with_stack` carries `any` and
    // `first`, the same traversal without the containment test.
    // An early exit descends a few nodes whatever the window, so 400 large
    // windows are few enough branches to learn: `first` gets 10 000 in every
    // class (kb:task/192).
    macro_rules! callbacks {
        ($tag:expr, $index:expr, $windows:ident, $seed:expr) => {
            for (i, (name, lo, hi, n)) in WINDOWS.iter().enumerate() {
                for early in [false, true] {
                    let count = if early { 10_000 } else { *n };
                    let qs = $windows($seed + i as u64, *lo, *hi, count);
                    let call = if early { "first" } else { "visit" };
                    let arm =
                        |masked: bool| {
                            let (qs, index) = (&qs, &$index);
                            let mut stack = Vec::new();
                            move || {
                                let mut t = 0usize;
                                for &q in black_box(qs) {
                                    let f = |idx: usize| {
                                        t += idx;
                                        if early {
                                            ControlFlow::Break(())
                                        } else {
                                            ControlFlow::Continue(())
                                        }
                                    };
                                    let s = &mut stack;
                                    let _ =
                                        match (early, masked) {
                                            (false, true) => index
                                                .visit_with_stack_forced::<true, (), _>(q, s, f),
                                            (false, false) => index
                                                .visit_with_stack_forced::<false, (), _>(q, s, f),
                                            (true, true) => {
                                                index.find_with_stack_forced::<true, (), _>(q, s, f)
                                            }
                                            (true, false) => index
                                                .find_with_stack_forced::<false, (), _>(q, s, f),
                                        };
                                }
                                t
                            }
                        };
                    let mut arms = vec![
                        paired::arm("branching", arm(false)),
                        paired::arm("masked (ships)", arm(true)),
                    ];
                    let label =
                        hits_label(&format!("{} {call}", $tag), name, &qs, |q| $index.count(q));
                    paired::run(&label, &mut arms, "branching");
                }
            }
        };
    }
    callbacks!("2d owned", owned2, windows_2d, 0x91);
    callbacks!("2d view", view2, windows_2d, 0x91);

    // ---- 3D: view, raycast ----
    let b3 = boxes_3d(0x3D, 60.0);
    let owned3 = build_3d(&b3);
    let bytes3 = owned3.to_bytes();
    let view3 = Index3DView::from_bytes(&bytes3).unwrap();
    for (i, (name, lo, hi, n)) in WINDOWS.iter().enumerate() {
        let qs = windows_3d(0x61 + i as u64, *lo, *hi, *n);
        let label = hits_label("3d view", name, &qs, |q| view3.count(q));
        pair!(
            label,
            |q| view3.any(q),
            &qs,
            |m: bool, q, out: &mut Vec<usize>| {
                if m {
                    view3.search_into_forced::<true>(q, out)
                } else {
                    view3.search_into_forced::<false>(q, out)
                }
            }
        );
    }
    callbacks!("3d owned", owned3, windows_3d, 0xA1);
    callbacks!("3d view", view3, windows_3d, 0xA1);
    // The window scene is too sparse for rays (about one hit per ray, where
    // the traversal shape decides nothing), so rays get denser scenes of their
    // own: tens and hundreds of candidates per ray.
    let rs = rays(0x7A);
    for (name, side) in [("sides <300", 300.0), ("sides <900", 900.0)] {
        let scene = build_3d(&boxes_3d(0x7B, side));
        let label = hits_label("3d raycast", name, &rs, |r| {
            let mut out = Vec::new();
            scene.raycast_into(r, &mut out);
            out.len()
        });
        let first_hit = |r| {
            scene
                .raycast_each(r, |_, _| ControlFlow::Break(()))
                .is_break()
        };
        pair!(label, first_hit, &rs, |m: bool, r, out: &mut Vec<usize>| {
            if m {
                scene.raycast_into_forced::<true>(r, out)
            } else {
                scene.raycast_into_forced::<false>(r, out)
            }
        });
    }

    // ---- f32 scalar ----
    let mut b = Index2DBuilder::new(b2.len());
    for &bx in &b2 {
        b.add(bx);
    }
    let f32_2 = b.finish_f32().unwrap();
    let mut b = Index3DBuilder::new(b3.len());
    for &bx in &b3 {
        b.add(bx);
    }
    let f32_3 = b.finish_f32().unwrap();
    for (i, (name, lo, hi, n)) in WINDOWS.iter().enumerate() {
        let qs = windows_2d(0x71 + i as u64, *lo, *hi, *n);
        let label = hits_label("2d f32", name, &qs, |q| f32_2.count(q));
        pair!(
            label,
            |q| f32_2.any(q),
            &qs,
            |m: bool, q, out: &mut Vec<usize>| {
                *out = if m {
                    f32_2.search_forced::<true>(q)
                } else {
                    f32_2.search_forced::<false>(q)
                };
            }
        );
        let qs = windows_3d(0x81 + i as u64, *lo, *hi, *n);
        let label = hits_label("3d f32", name, &qs, |q| f32_3.count(q));
        pair!(
            label,
            |q| f32_3.any(q),
            &qs,
            |m: bool, q, out: &mut Vec<usize>| {
                *out = if m {
                    f32_3.search_forced::<true>(q)
                } else {
                    f32_3.search_forced::<false>(q)
                };
            }
        );
    }
}
