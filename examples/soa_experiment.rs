//! Experiment: AoS queries (current Index) vs SoA scalar vs SoA SIMD (f64x4).
//! Run: `cargo run --release --example soa_experiment`
//! (faster with `RUSTFLAGS="-C target-cpu=native"`).

use std::time::Instant;

use packed_spatial_index::experimental::ExperimentalSortKey;
use packed_spatial_index::{IndexBuilder, Rect};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N: usize = 100_000;
const NODE_SIZE: usize = 16;
const NQ: usize = 1_000;
const REPS: usize = 200;

fn main() {
    let mut rng = StdRng::seed_from_u64(0xB0B);
    let boxes: Vec<[f64; 4]> = (0..N)
        .map(|_| {
            let cx: f64 = rng.gen_range(0.0..10_000.0);
            let cy: f64 = rng.gen_range(0.0..10_000.0);
            let w: f64 = rng.gen_range(0.1..20.0);
            let h: f64 = rng.gen_range(0.1..20.0);
            [cx, cy, cx + w, cy + h]
        })
        .collect();

    let mut aos = IndexBuilder::new(N)
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut);
    let mut soa = IndexBuilder::new(N)
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut);
    for b in &boxes {
        aos.add_bounds(b[0], b[1], b[2], b[3]);
        soa.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let aos = aos.finish().unwrap();
    let soa = soa.finish_simd().unwrap();

    let mut qrng = StdRng::seed_from_u64(0xACE);
    let queries: Vec<[f64; 4]> = (0..NQ)
        .map(|_| {
            let qx: f64 = qrng.gen_range(0.0..10_000.0);
            let qy: f64 = qrng.gen_range(0.0..10_000.0);
            let qw: f64 = qrng.gen_range(10.0..200.0);
            let qh: f64 = qrng.gen_range(10.0..200.0);
            [qx, qy, qx + qw, qy + qh]
        })
        .collect();

    // correctness: SoA scalar and SoA SIMD == AoS
    {
        let (mut a, mut s, mut sm, mut av) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut st1, mut st2, mut st3, mut st4) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for q in &queries {
            let rect = Rect::new(q[0], q[1], q[2], q[3]);
            aos.search_into_stack(rect, &mut a, &mut st1);
            soa.search_scalar(rect, &mut s, &mut st2);
            soa.search_simd(rect, &mut sm, &mut st3);
            soa.search_avx512(rect, &mut av, &mut st4);
            a.sort_unstable();
            s.sort_unstable();
            sm.sort_unstable();
            av.sort_unstable();
            assert_eq!(a, s, "SoA-scalar != AoS");
            assert_eq!(a, sm, "SoA-SIMD != AoS");
            assert_eq!(a, av, "SoA-AVX512 != AoS");
        }
        println!(
            "avx512f available: {}",
            std::is_x86_feature_detected!("avx512f")
        );
        println!("correctness: scalar == SSE/AVX2 == AVX-512 == AoS OK\n");
    }

    fn bench<F: FnMut() -> usize>(label: &str, nq: usize, mut f: F) {
        let mut best = f64::INFINITY;
        let mut sink = 0;
        for _ in 0..REPS {
            let t = Instant::now();
            sink = f();
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        std::hint::black_box(sink);
        println!("{:<16} : {:>8.1} us / {} queries", label, best, nq);
    }

    let (mut buf, mut st) = (Vec::new(), Vec::new());
    bench("AoS", NQ, || {
        let mut t = 0;
        for x in &queries {
            aos.search_into_stack(Rect::new(x[0], x[1], x[2], x[3]), &mut buf, &mut st);
            t += buf.len();
        }
        t
    });
    bench("SoA-scalar", NQ, || {
        let mut t = 0;
        for x in &queries {
            soa.search_scalar(Rect::new(x[0], x[1], x[2], x[3]), &mut buf, &mut st);
            t += buf.len();
        }
        t
    });
    bench("SoA-SIMD(f64x4)", NQ, || {
        let mut t = 0;
        for x in &queries {
            soa.search_simd(Rect::new(x[0], x[1], x[2], x[3]), &mut buf, &mut st);
            t += buf.len();
        }
        t
    });
    bench("SoA-AVX512(x8)", NQ, || {
        let mut t = 0;
        for x in &queries {
            soa.search_avx512(Rect::new(x[0], x[1], x[2], x[3]), &mut buf, &mut st);
            t += buf.len();
        }
        t
    });
}
