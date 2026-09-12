//! One-process-per-arm drivers for profiling the count and aggregate kernels
//! under `callgrind` (Windows has no valgrind; WSL2 runs it fine — simulation,
//! so the VM costs it nothing), and alternating-arm modes for `perf record`.
//!
//! The two open questions this instruments:
//! 1. `Index2D::count` vs `SimdIndex2D::count` on large windows (the scalar
//!    kernel stays ~1.2x ahead) — modes `owned` / `simd`, `time1`, `hot1`.
//! 2. `Index2D::aggregate` vs `count` on ~10k-hit windows — modes `count` /
//!    `aggregate`, `time2`, `hot2`.
//!
//! Every arm prints its total, and `owned`/`simd` must print the same number
//! (same boxes, same windows, same count) — the parity check before trusting
//! any profile of them.
//!
//! Run under callgrind (collect only the query loop; the index build would
//! otherwise swamp it):
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=2 cargo build --release --features simd \
//!     --example callgrind_count_agg
//! valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes \
//!     --collect-atstart=no --toggle-collect='callgrind_count_agg::run_*' \
//!     target/release/examples/callgrind_count_agg simd
//! callgrind_annotate callgrind.out.<pid>
//! ```
//! `hot1` / `hot2` alternate the arms many times for `perf record`, whose
//! per-symbol cycle split needs no isolation: the kernels are separate
//! symbols, the build is a handful of samples against tens of thousands.

use std::hint::black_box;

use packed_spatial_index::{Box2D, Index2D, Index2DBuilder};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N1: usize = 100_000;
const EXT1: f64 = 10_000.0;

const N2: usize = 1_000_000;
const EXT2: f64 = 1_000_000.0;

/// The `paired_simd_count` workload: small boxes, windows 2000..5000.
fn boxes1() -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(0xB0B);
    (0..N1)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXT1);
            let y: f64 = rng.random_range(0.0..EXT1);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn windows1() -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(0x1A96E);
    (0..200)
        .map(|_| {
            let s: f64 = rng.random_range(2000.0..5000.0);
            let x: f64 = rng.random_range(0.0..EXT1 - s);
            let y: f64 = rng.random_range(0.0..EXT1 - s);
            Box2D::new(x, y, x + s, y + s)
        })
        .collect()
}

/// The `paired_aggregate` workload: 1M boxes, ~10k-hit windows.
fn build_big() -> Index2D {
    let mut rng = StdRng::seed_from_u64(1);
    let mut builder = Index2DBuilder::new(N2).node_size(16);
    for _ in 0..N2 {
        let x: f64 = rng.random_range(0.0..EXT2);
        let y: f64 = rng.random_range(0.0..EXT2);
        let w: f64 = rng.random_range(0.0..50.0);
        let h: f64 = rng.random_range(0.0..50.0);
        builder.add(Box2D::new(x, y, x + w, y + h));
    }
    let scalars: Vec<f64> = (0..N2).map(|i| (i % 101) as f64).collect();
    let masks: Vec<u64> = (0..N2).map(|i| 1u64 << (i % 61)).collect();
    builder
        .aggregate_scalar(&scalars)
        .aggregate_mask(&masks)
        .finish()
        .unwrap()
}

fn windows2() -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(2);
    (0..200)
        .map(|_| {
            let side = 100_000.0f64;
            let x: f64 = rng.random_range(0.0..EXT2 - side);
            let y: f64 = rng.random_range(0.0..EXT2 - side);
            Box2D::new(x, y, x + side, y + side)
        })
        .collect()
}

/// Both count questions share this arm: the owned kernel is the control of
/// the aggregate question and the subject of the SIMD question.
#[inline(never)]
fn run_count(owned: &Index2D, qs: &[Box2D]) -> usize {
    let mut t = 0usize;
    for q in qs {
        t += owned.count(black_box(*q));
    }
    t
}

#[cfg(feature = "simd")]
#[inline(never)]
fn run_simd(simd: &packed_spatial_index::SimdIndex2D, qs: &[Box2D]) -> usize {
    let mut t = 0usize;
    for q in qs {
        t += simd.count(black_box(*q));
    }
    t
}

#[inline(never)]
fn run_aggregate(owned: &Index2D, qs: &[Box2D]) -> u64 {
    let mut t = 0u64;
    for q in qs {
        let a = owned.aggregate(black_box(*q)).unwrap();
        t += a.count + a.sum.unwrap_or(0.0) as u64 + a.mask.unwrap_or(0).count_ones() as u64;
    }
    t
}

fn main() {
    match std::env::args().nth(1).unwrap_or_default().as_str() {
        // One arm per process, for callgrind toggle-collect runs.
        "owned" => {
            let items = boxes1();
            let mut b = Index2DBuilder::new(N1);
            for &bx in &items {
                b.add(bx);
            }
            let owned = b.finish().unwrap();
            let qs = windows1();
            println!("{}", run_count(&owned, &qs));
        }
        "count" => {
            let owned = build_big();
            let qs = windows2();
            println!("{}", run_count(&owned, &qs));
        }
        "count2" => {
            let owned = build_big();
            let qs = windows2();
            println!("{}", run_count(&owned, &qs));
        }
        #[cfg(feature = "simd")]
        "simd" => {
            let items = boxes1();
            let mut s = Index2DBuilder::new(N1);
            for &bx in &items {
                s.add(bx);
            }
            let simd = s.finish_simd().unwrap();
            let qs = windows1();
            println!("{}", run_simd(&simd, &qs));
        }
        "aggregate" => {
            let owned = build_big();
            let qs = windows2();
            println!("{}", run_aggregate(&owned, &qs));
        }
        "time1" => time1(),
        "time2" => time2(),
        "hot1" => hot1(),
        "hot2" => hot2(),
        _ => eprintln!("modes: owned count2 simd aggregate time1 time2 hot1 hot2"),
    }
}

/// Interleaved wall-clock rounds, min per round: reproduces the paired-bench
/// ratios before any profiler run (the gap must reproduce on this platform to
/// be the thing you set out to explain).
#[inline(never)]
fn time1() {
    let items = boxes1();
    let mut b = Index2DBuilder::new(N1);
    let mut s = Index2DBuilder::new(N1);
    for &bx in &items {
        b.add(bx);
        s.add(bx);
    }
    let owned = b.finish().unwrap();
    let simd = s.finish_simd().unwrap();
    let qs = windows1();
    let mut o_min = f64::MAX;
    let mut s_min = f64::MAX;
    for _ in 0..30 {
        let t = std::time::Instant::now();
        black_box(run_count(&owned, &qs));
        let o = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        black_box(run_simd(&simd, &qs));
        let sp = t.elapsed().as_secs_f64();
        o_min = o_min.min(o);
        s_min = s_min.min(sp);
    }
    println!(
        "time1 owned {o_min:.6}s simd {s_min:.6}s simd/owned {:.3}",
        s_min / o_min
    );
}

#[inline(never)]
fn time2() {
    let owned = build_big();
    let qs = windows2();
    let mut c_min = f64::MAX;
    let mut a_min = f64::MAX;
    for _ in 0..30 {
        let t = std::time::Instant::now();
        black_box(run_count(&owned, &qs));
        let c = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        black_box(run_aggregate(&owned, &qs));
        let a = t.elapsed().as_secs_f64();
        c_min = c_min.min(c);
        a_min = a_min.min(a);
    }
    println!(
        "time2 count {c_min:.6}s aggregate {a_min:.6}s agg/count {:.3}",
        a_min / c_min
    );
}

/// Long alternation for `perf record`: the per-symbol cycle split
/// (`run_count`'s kernel vs `run_simd`) needs equal query counts and nothing
/// else, because each kernel is its own symbol.
#[inline(never)]
fn hot1() {
    let items = boxes1();
    let mut b = Index2DBuilder::new(N1);
    let mut s = Index2DBuilder::new(N1);
    for &bx in &items {
        b.add(bx);
        s.add(bx);
    }
    let owned = b.finish().unwrap();
    let simd = s.finish_simd().unwrap();
    let qs = windows1();
    let mut acc = 0usize;
    for _ in 0..2000 {
        acc = acc.wrapping_add(run_count(&owned, &qs));
        acc = acc.wrapping_add(run_simd(&simd, &qs));
    }
    println!("{acc}");
}

#[inline(never)]
fn hot2() {
    let owned = build_big();
    let qs = windows2();
    let mut acc = 0u64;
    for _ in 0..400 {
        acc = acc.wrapping_add(run_count(&owned, &qs) as u64);
        acc = acc.wrapping_add(run_aggregate(&owned, &qs));
    }
    println!("{acc}");
}
