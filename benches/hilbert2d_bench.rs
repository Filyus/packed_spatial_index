//! Hilbert encoder benchmark: crate implementations against the reference
//! `static_aabb2d_index::hilbert_xy_to_index`.
//!
//! Measures **throughput**: encode a whole point array into an
//! output buffer. This reflects real index-build usage and lets the
//! compiler pipeline/vectorize independent iterations.
//!
//! IMPORTANT: do NOT wrap every element in `black_box`; that prevents vectorization and
//! degenerates the measurement into single-call latency (biasing the comparison toward
//! the table-driven version). `black_box` is applied only to input and output buffers.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use packed_spatial_index::benchmark_support as hilbert;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use static_aabb2d_index::hilbert_xy_to_index;

const N: usize = 100_000;

fn gen_points() -> Vec<(u16, u16)> {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    (0..N).map(|_| (rng.random(), rng.random())).collect()
}

fn bench_hilbert(c: &mut Criterion) {
    let points = gen_points();
    let mut out = vec![0u32; N];
    let mut group = c.benchmark_group("hilbert_encode");
    group.throughput(Throughput::Elements(N as u64));

    macro_rules! bench {
        ($name:expr, $f:path) => {
            group.bench_function($name, |b| {
                b.iter(|| {
                    let pts = black_box(&points[..]);
                    for (i, &(x, y)) in pts.iter().enumerate() {
                        out[i] = $f(x, y);
                    }
                    black_box(&out[..]);
                })
            });
        };
    }

    bench!("crate::hilbert_xy_to_index", hilbert_xy_to_index);
    bench!("magic_bits", hilbert::magic_bits);
    bench!("lut", hilbert::lut);
    bench!("loop_rotation", hilbert::loop_rotation);
    bench!("morton", hilbert::morton);

    group.finish();
}

criterion_group!(benches, bench_hilbert);
criterion_main!(benches);
