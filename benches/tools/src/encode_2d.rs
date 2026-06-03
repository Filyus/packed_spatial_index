//! Local performance tool: batch encoding throughput comparison.
//! Run (with autovectorization for the native CPU):
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path benches/tools/Cargo.toml --bin encode_2d

use std::time::Instant;

use packed_spatial_index::benchmark_support as hilbert;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 1_000_000;
const REPS: usize = 50;

fn melem_s(elapsed_ms: f64) -> f64 {
    (N as f64) / (elapsed_ms / 1e3) / 1e6
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let xs: Vec<u16> = (0..N).map(|_| rng.random()).collect();
    let ys: Vec<u16> = (0..N).map(|_| rng.random()).collect();
    let mut out = vec![0u32; N];

    // scalar lut
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        for i in 0..N {
            out[i] = hilbert::lut(xs[i], ys[i]);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out[N - 1]);
    }
    println!("lut (scalar loop)        : {:>7.0} Melem/s", melem_s(best));

    // scalar magic_bits
    best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        for i in 0..N {
            out[i] = hilbert::magic_bits(xs[i], ys[i]);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out[N - 1]);
    }
    println!("magic_bits (scalar loop) : {:>7.0} Melem/s", melem_s(best));

    // batch magic_bits (autovec candidate)
    best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        hilbert::magic_bits_batch(&xs, &ys, &mut out);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out[N - 1]);
    }
    println!("magic_bits_batch         : {:>7.0} Melem/s", melem_s(best));
}
