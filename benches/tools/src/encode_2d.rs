//! Local performance tool: batch encoding throughput comparison.
//! Run (with autovectorization for the native CPU):
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path benches/tools/Cargo.toml --bin encode_2d

use std::time::Instant;

use packed_spatial_index::benchmark_support as hilbert;
use psi_perf::emit;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 1_000_000;
const REPS: usize = 50;

fn melem_s(elapsed_ms: f64) -> f64 {
    (N as f64) / (elapsed_ms / 1e3) / 1e6
}

fn main() {
    psi_perf::pin_from_env();
    emit(&serde_json::json!({ "tool": "encode_2d_meta", "n": N }));
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
    emit(&serde_json::json!({ "tool": "encode_2d", "encoder": "lut", "melem_s": melem_s(best) }));

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
    emit(
        &serde_json::json!({ "tool": "encode_2d", "encoder": "magic_bits", "melem_s": melem_s(best) }),
    );

    // batch magic_bits, written so LLVM can vectorize the loop
    best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        hilbert::magic_bits_batch(&xs, &ys, &mut out);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out[N - 1]);
    }
    emit(
        &serde_json::json!({ "tool": "encode_2d", "encoder": "magic_bits_batch", "melem_s": melem_s(best) }),
    );
}
