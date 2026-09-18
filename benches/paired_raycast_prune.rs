//! Does exact closest-hit pruning pay on a mesh? Four shapes of the same
//! answer, in ONE binary, interleaved, with the two broad phases as controls.
//!
//! The question behind it: every ordering API in the crate keys on the *box*
//! (`raycast_closest` returns the nearest item whose box the ray enters), so a
//! caller who wants the nearest *triangle* does the refinement itself. The
//! shipped `examples/raycast_mesh.rs` does it the widest way there is — collect
//! every box the ray crosses over the whole segment, then run the exact test on
//! all of them. But `raycast_each` walks candidates in nondecreasing entry `t`,
//! and entry `t` is a lower bound on the exact hit `t`, so a caller can stop as
//! soon as the stream's entry `t` passes the best exact hit so far. That is an
//! exact answer with correct pruning and needs nothing new in the crate. This
//! bench asks what it is worth before any API is designed around it.
//!
//! **The two broad phases are not the same traversal**, which is the whole
//! tension: `raycast_into` is a depth-first stack sweep, `raycast_each` is
//! best-first over a `BinaryHeap`, so ordering costs more per candidate before
//! any triangle is tested. Pruning has to pay that back out of the narrow-phase
//! work it skips. Both broad phases are therefore arms here — the gap between
//! them is the ordering tax, read directly instead of inferred.
//!
//! Arms:
//!   - `raycast_into (control)` — unordered broad phase, no narrow phase.
//!   - `raycast_each (ordered broad)` — ordered broad phase, no narrow phase.
//!     Its ratio to the control IS the ordering tax.
//!   - `collect + gather + closest_triangle` — the shipped example's shape:
//!     copy the candidate triangles into a contiguous slice, one kernel call.
//!     The reference the ratios are taken against.
//!   - `collect, test per id` — same set, no gather copy, tested one at a time.
//!     Isolates the copy from the pruning; without it a win could be either.
//!   - `ordered + prune` — `raycast_each` in entry-`t` order, test per id,
//!     `Break` once entry `t` exceeds the best exact `t`.
//!
//! The last three must agree: the checksum column is the sum of `hit id + 1`
//! over the rays (0 for a miss) and pins that pruning did not change the answer.
//!
//! The sweep is over scene density, not item count: what decides the answer is
//! how many boxes a ray crosses and how early the real hit sits among them, and
//! both are printed, measured, in each case's label.
//!
//! `f64` on purpose. `closest_triangle` over `Triangle3DF32` tests 8 at a time
//! with the `simd` feature, so on that record the gather arm buys vectorization
//! that a per-candidate loop gives up, and the comparison would carry both
//! effects at once. Here every arm runs the same scalar kernel per triangle, so
//! what moves is the traversal. Whether a pruned form can keep the f32 batching
//! (3DGRT flushes a chunk of hits at a time) is a separate question about the
//! caller's kernel, not about the index.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --bench paired_raycast_prune

use std::hint::black_box;
use std::ops::ControlFlow;

use packed_spatial_index::{Index3D, Point3D, Ray3D, Triangle3D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const EXTENT: f64 = 60.0;
const RAYS: usize = 512;
const MAX_T: f64 = 1_000.0;

fn n_items() -> usize {
    std::env::var("PRUNE_N")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(200_000)
}

/// Triangle half-size. A ray along z crosses roughly `n * (2*size)^2 / EXTENT^2`
/// boxes, so this is the density knob: the sweep runs it from "the ray meets
/// almost nothing" to "the ray is buried in geometry".
fn sizes() -> Vec<f64> {
    match std::env::var("PRUNE_SIZES") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![0.35, 0.75, 1.5, 3.0, 6.0],
    }
}

fn mesh(seed: u64, n: usize, size: f64) -> Vec<Triangle3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let c = [
                rng.random_range(0.0..EXTENT),
                rng.random_range(0.0..EXTENT),
                rng.random_range(0.0..EXTENT),
            ];
            let mut v = || {
                [
                    c[0] + rng.random_range(-size..size),
                    c[1] + rng.random_range(-size..size),
                    c[2] + rng.random_range(-size..size),
                ]
            };
            Triangle3D::new(v(), v(), v())
        })
        .collect()
}

/// Rays entering the scene from below, so a ray crosses its whole depth and the
/// nearest hit sits early in the stream — the case pruning is for.
fn rays(seed: u64) -> Vec<Ray3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..RAYS)
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
                MAX_T,
            )
        })
        .collect()
}

fn main() {
    pin::pin_from_env();
    let n = n_items();

    for size in sizes() {
        let tris = mesh(0x3E11 ^ size.to_bits(), n, size);
        let index = Index3D::from_triangles(&tris).unwrap();
        let qs = rays(0xA11E5);

        // What the sweep is varying, measured rather than assumed: how many
        // boxes a ray crosses, and how far into that ordered stream the nearest
        // real hit sits. The second number is what pruning can save.
        let mut cand_total = 0usize;
        let mut prefix_total = 0usize;
        let mut walked_total = 0usize;
        let mut hits = 0usize;
        let mut buf = Vec::new();
        for ray in &qs {
            index.raycast_into(*ray, &mut buf);
            cand_total += buf.len();
            let mut best = f64::INFINITY;
            let mut prefix = 0usize;
            let mut seen = 0usize;
            let _: ControlFlow<()> = index.raycast_each(*ray, |id, enter_t| {
                if enter_t > best {
                    return ControlFlow::Break(());
                }
                seen += 1;
                if let Some(h) = ray.closest_triangle(&tris[id..id + 1]) {
                    if h.t < best {
                        best = h.t;
                        prefix = seen;
                    }
                }
                ControlFlow::Continue(())
            });
            walked_total += seen;
            if best.is_finite() {
                hits += 1;
                prefix_total += prefix;
            }
        }
        let label = format!(
            "size={size} ({:.1} candidates/ray, pruned walk tests {:.1}, hit at #{:.1}, {:.0}% of rays hit)",
            cand_total as f64 / qs.len() as f64,
            walked_total as f64 / qs.len() as f64,
            prefix_total as f64 / hits.max(1) as f64,
            100.0 * hits as f64 / qs.len() as f64,
        );

        let (mut ids_c, mut ids_g, mut ids_p) = (Vec::new(), Vec::new(), Vec::new());
        let mut cands: Vec<Triangle3D> = Vec::new();
        let mut arms = vec![
            // Control: the unordered broad phase on its own.
            paired::arm("raycast_into (control)", || {
                let mut t = 0;
                for ray in black_box(&qs) {
                    index.raycast_into(*ray, &mut ids_c);
                    t += ids_c.len();
                }
                t
            }),
            // Second control: the same set, walked in order, still no narrow
            // phase. Against the first arm this is the price of ordering.
            paired::arm("raycast_each (ordered broad)", || {
                let mut t = 0;
                for ray in black_box(&qs) {
                    let _: ControlFlow<()> = index.raycast_each(*ray, |_, _| {
                        t += 1;
                        ControlFlow::Continue(())
                    });
                }
                t
            }),
            paired::arm("collect + gather + closest_triangle", || {
                let mut sum = 0;
                for ray in black_box(&qs) {
                    index.raycast_into(*ray, &mut ids_g);
                    cands.clear();
                    cands.extend(ids_g.iter().map(|&id| tris[id]));
                    if let Some(h) = ray.closest_triangle(&cands) {
                        sum += ids_g[h.index] + 1;
                    }
                }
                sum
            }),
            paired::arm("collect, test per id", || {
                let mut sum = 0;
                for ray in black_box(&qs) {
                    index.raycast_into(*ray, &mut ids_p);
                    let mut best = f64::INFINITY;
                    let mut best_id = usize::MAX;
                    for &id in ids_p.iter() {
                        if let Some(h) = ray.closest_triangle(&tris[id..id + 1]) {
                            if h.t < best {
                                best = h.t;
                                best_id = id;
                            }
                        }
                    }
                    if best_id != usize::MAX {
                        sum += best_id + 1;
                    }
                }
                sum
            }),
            paired::arm("ordered + prune", || {
                let mut sum = 0;
                for ray in black_box(&qs) {
                    let mut best = f64::INFINITY;
                    let mut best_id = usize::MAX;
                    let _: ControlFlow<()> = index.raycast_each(*ray, |id, enter_t| {
                        if enter_t > best {
                            return ControlFlow::Break(());
                        }
                        if let Some(h) = ray.closest_triangle(&tris[id..id + 1]) {
                            if h.t < best {
                                best = h.t;
                                best_id = id;
                            }
                        }
                        ControlFlow::Continue(())
                    });
                    if best_id != usize::MAX {
                        sum += best_id + 1;
                    }
                }
                sum
            }),
        ];
        paired::run(&label, &mut arms, "collect + gather + closest_triangle");
    }
}
