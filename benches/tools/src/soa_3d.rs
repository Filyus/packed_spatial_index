//! Local performance tool: scalar `Index3D` queries vs `SimdIndex3D` scalar/SIMD traversal.
//! Run: `cargo run --release --manifest-path benches/tools/Cargo.toml --bin soa_3d`
//! (faster with `RUSTFLAGS="-C target-cpu=native"`).

use std::time::Instant;

use packed_spatial_index::benchmark_support::SortKey3DStrategy;
use packed_spatial_index::{Box3D, Index3DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 100_000;
const NODE_SIZE: usize = 16;
const NQ: usize = 1_000;
const REPS: usize = 200;

fn main() {
    let mut rng = StdRng::seed_from_u64(0x3D0B);
    let boxes: Vec<Box3D> = (0..N)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect();

    let mut scalar = Index3DBuilder::new(N)
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey3DStrategy::Hilbert);
    let mut simd = Index3DBuilder::new(N)
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey3DStrategy::Hilbert);
    for &bounds in &boxes {
        scalar.add(bounds);
        simd.add(bounds);
    }
    let scalar = scalar.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    let mut qrng = StdRng::seed_from_u64(0x3ACE);
    let queries: Vec<Box3D> = (0..NQ)
        .map(|_| {
            let x: f64 = qrng.random_range(0.0..10_000.0);
            let y: f64 = qrng.random_range(0.0..10_000.0);
            let z: f64 = qrng.random_range(0.0..10_000.0);
            let w: f64 = qrng.random_range(10.0..200.0);
            Box3D::new(x, y, z, x + w, y + w, z + w)
        })
        .collect();

    let (mut scalar_hits, mut soa_scalar_hits, mut simd_hits, mut avx_hits) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut scalar_stack, mut soa_scalar_stack, mut simd_stack, mut avx_stack) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &query in &queries {
        scalar.search_into_stack(query, &mut scalar_hits, &mut scalar_stack);
        simd.search_scalar(query, &mut soa_scalar_hits, &mut soa_scalar_stack);
        simd.search_simd(query, &mut simd_hits, &mut simd_stack);
        simd.search_avx512(query, &mut avx_hits, &mut avx_stack);
        scalar_hits.sort_unstable();
        soa_scalar_hits.sort_unstable();
        simd_hits.sort_unstable();
        avx_hits.sort_unstable();
        assert_eq!(scalar_hits, soa_scalar_hits, "SoA scalar != Index3D");
        assert_eq!(scalar_hits, simd_hits, "SoA SIMD != Index3D");
        assert_eq!(scalar_hits, avx_hits, "SoA AVX-512 != Index3D");
    }
    println!(
        "avx512f available: {}",
        std::is_x86_feature_detected!("avx512f")
    );
    println!("correctness: Index3D == SoA scalar == SIMD == AVX-512 OK\n");

    fn bench<F: FnMut() -> usize>(label: &str, mut f: F) {
        let mut best = f64::INFINITY;
        let mut sink = 0usize;
        for _ in 0..REPS {
            let start = Instant::now();
            sink = f();
            best = best.min(start.elapsed().as_secs_f64() * 1e6);
        }
        std::hint::black_box(sink);
        println!("{label:<18}: {best:>8.1} us / {NQ} queries");
    }

    let (mut out, mut stack) = (Vec::new(), Vec::new());
    bench("Index3D", || {
        let mut total = 0usize;
        for &query in &queries {
            scalar.search_into_stack(query, &mut out, &mut stack);
            total += out.len();
        }
        total
    });
    bench("SoA scalar", || {
        let mut total = 0usize;
        for &query in &queries {
            simd.search_scalar(query, &mut out, &mut stack);
            total += out.len();
        }
        total
    });
    bench("SoA SIMD(f64x4)", || {
        let mut total = 0usize;
        for &query in &queries {
            simd.search_simd(query, &mut out, &mut stack);
            total += out.len();
        }
        total
    });
    bench("SoA AVX-512(x8)", || {
        let mut total = 0usize;
        for &query in &queries {
            simd.search_avx512(query, &mut out, &mut stack);
            total += out.len();
        }
        total
    });
}
