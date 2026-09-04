// Gate measurement for a dual-tree kNN-join, per the handover: establish the
// number a kernel would have to beat BEFORE writing one.
//
// Arm A is the baseline that already ships: one `neighbors_of_box(a, k)` per
// item of A. Arm B is `join_epsilon` at a radius tuned to emit roughly the
// same n*k pairs — not the same answer, but a real dual-tree descent doing a
// comparable amount of output work, so its runtime is a proxy for the floor a
// dual-tree kNN-join could reach. The gap between them is the most a kernel
// could win; if it is small, the kernel is not worth building.
use packed_spatial_index::{Box2D, Index2D, Index2DBuilder};
use std::hint::black_box;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn uniform(seed: u64, n: usize, extent: f64, size: f64) -> Vec<Box2D> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            let x = r.next_f64() * extent;
            let y = r.next_f64() * extent;
            Box2D::new(x, y, x + r.next_f64() * size, y + r.next_f64() * size)
        })
        .collect()
}

fn clustered(seed: u64, n: usize, extent: f64, size: f64, clusters: usize) -> Vec<Box2D> {
    let mut r = Rng(seed);
    let centers: Vec<(f64, f64)> = (0..clusters)
        .map(|_| (r.next_f64() * extent, r.next_f64() * extent))
        .collect();
    let spread = extent / (clusters as f64).sqrt() * 0.15;
    (0..n)
        .map(|i| {
            let (cx, cy) = centers[i % clusters];
            let x = cx + (r.next_f64() - 0.5) * spread;
            let y = cy + (r.next_f64() - 0.5) * spread;
            Box2D::new(x, y, x + r.next_f64() * size, y + r.next_f64() * size)
        })
        .collect()
}

fn build(boxes: &[Box2D]) -> Index2D {
    let mut b = Index2DBuilder::new(boxes.len());
    for &x in boxes {
        b.add(x);
    }
    b.finish().unwrap()
}

fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Pairs at `eps`, stopping once `cap` is exceeded — an uncapped count at a
/// large epsilon on clustered data is quadratic and never returns.
fn run(label: &str, a_boxes: &[Box2D], b_boxes: &[Box2D], k: usize, rounds: usize, iters: usize) {
    let a = build(a_boxes);
    let b = build(b_boxes);
    let n = a_boxes.len();

    // Correctness guard before any timing: the two arms must agree on every
    // row's distances (ids can differ on a tie at the kth).
    let dual = a.knn_join(&b, k);
    for (i, row) in dual.iter().enumerate() {
        let want = b.neighbors_of_box(a_boxes[i], k);
        let d: Vec<f64> = row.iter().map(|&j| a_boxes[i].distance_to_box(b_boxes[j])).collect();
        let w: Vec<f64> = want.iter().map(|&j| a_boxes[i].distance_to_box(b_boxes[j])).collect();
        assert_eq!(d, w, "{label}: arms disagree on row {i}");
    }

    let mut abs: Vec<f64> = Vec::new();
    let mut ratio: Vec<f64> = Vec::new();
    let mut control: Vec<f64> = Vec::new();
    for round in 0..rounds {
        let mut t = [f64::NAN; 3];
        // Slot 2 is knn_join a second time: the control, which should not
        // move. What it shows is the floor, not an effect.
        let order: [usize; 3] = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
        for &slot in order.iter() {
            t[slot] = match slot {
                0 | 2 => time(iters, || {
                    black_box(a.knn_join(&b, k));
                }),
                _ => time(iters, || {
                    let mut sink = 0usize;
                    for &box_a in a_boxes {
                        sink += b.neighbors_of_box(box_a, k).len();
                    }
                    black_box(sink);
                }),
            };
        }
        if round == 0 {
            continue;
        }
        abs.push(t[0]);
        ratio.push(t[1] / t[0]);
        control.push(t[2] / t[0]);
    }
    let lo = control.iter().cloned().fold(f64::MAX, f64::min);
    let hi = control.iter().cloned().fold(f64::MIN, f64::max);
    println!(
        "{label:<12} n={n:<8} k={k:<3} knn_join {:>9.1}ms | naive/knn_join {:>6.2}x | CONTROL {:.3}x [{:.3}..{:.3}]",
        median(&mut abs.clone()) * 1e3,
        median(&mut ratio.clone()),
        median(&mut control.clone()),
        lo,
        hi,
    );
}

fn main() {
    let rounds = 7;
    let iters = 2;
    for n in [100_000usize] {
        for k in [1usize, 10, 50] {
            run(
                "uniform",
                &uniform(11, n, 10_000.0, 4.0),
                &uniform(12, n, 10_000.0, 4.0),
                k,
                rounds,
                iters,
            );
            run(
                "clustered",
                &clustered(13, n, 10_000.0, 4.0, 64),
                &clustered(14, n, 10_000.0, 4.0, 64),
                k,
                rounds,
                iters,
            );
        }
    }
}
