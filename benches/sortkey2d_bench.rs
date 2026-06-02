//! Focused sort-key stage benchmarks.
//!
//! This separates three effects that are easy to blur together:
//! raw Hilbert/Morton encoding, encode+radix-sort order construction, and
//! full index building.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use packed_spatial_index::experimental::{self, ExperimentalSortKey2D, radix_sort_pairs};
use packed_spatial_index::{Bounds2D, Index2DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const ENCODE_N: usize = 100_000;
const NODE_SIZE: usize = 16;

const KEYS: &[(&str, ExperimentalSortKey2D)] = &[
    (
        "hilbert_magic_bits",
        ExperimentalSortKey2D::HilbertMagicBits,
    ),
    ("hilbert_lut", ExperimentalSortKey2D::HilbertLut),
    (
        "hilbert_loop_rotation",
        ExperimentalSortKey2D::HilbertLoopRotation,
    ),
    ("morton", ExperimentalSortKey2D::Morton),
];

fn gen_boxes(n: usize) -> Vec<Bounds2D> {
    let mut rng = StdRng::seed_from_u64(0xB0B);
    (0..n)
        .map(|_| {
            let cx: f64 = rng.random_range(0.0..10_000.0);
            let cy: f64 = rng.random_range(0.0..10_000.0);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            Bounds2D::new(cx, cy, cx + w, cy + h)
        })
        .collect()
}

fn normalized_points(boxes: &[Bounds2D]) -> Vec<(u16, u16)> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for b in boxes {
        min_x = min_x.min(b.min_x);
        min_y = min_y.min(b.min_y);
        max_x = max_x.max(b.max_x);
        max_y = max_y.max(b.max_y);
    }

    let scaled_width = u16::MAX as f64 / (max_x - min_x);
    let scaled_height = u16::MAX as f64 / (max_y - min_y);
    boxes
        .iter()
        .map(|b| {
            (
                hilbert_coord(scaled_width, b.min_x, b.max_x, min_x),
                hilbert_coord(scaled_height, b.min_y, b.max_y, min_y),
            )
        })
        .collect()
}

#[inline]
fn hilbert_coord(scaled: f64, lo: f64, hi: f64, extent_min: f64) -> u16 {
    let value = scaled * (0.5 * (lo + hi) - extent_min);
    if value.is_nan() {
        0
    } else if value > u16::MAX as f64 {
        u16::MAX
    } else if value < 0.0 {
        0
    } else {
        value as u16
    }
}

fn bench_encode(c: &mut Criterion) {
    let boxes = gen_boxes(ENCODE_N);
    let points = normalized_points(&boxes);
    let mut out = vec![0u32; ENCODE_N];

    let mut group = c.benchmark_group("sortkey_encode_normalized_points");
    group.throughput(Throughput::Elements(ENCODE_N as u64));

    macro_rules! bench_encode_case {
        ($name:expr, $f:path) => {
            group.bench_function($name, |b| {
                b.iter(|| {
                    let points = black_box(&points[..]);
                    for (i, &(x, y)) in points.iter().enumerate() {
                        out[i] = $f(x, y);
                    }
                    black_box(&out[..]);
                });
            });
        };
    }

    bench_encode_case!("hilbert_magic_bits", experimental::magic_bits);
    bench_encode_case!("hilbert_lut", experimental::lut);
    bench_encode_case!("hilbert_loop_rotation", experimental::loop_rotation);
    bench_encode_case!("morton", experimental::morton);
    group.finish();
}

fn bench_encode_sort(c: &mut Criterion) {
    let boxes = gen_boxes(ENCODE_N);
    let points = normalized_points(&boxes);

    let mut group = c.benchmark_group("sortkey_encode_radix_sort");
    group.throughput(Throughput::Elements(ENCODE_N as u64));

    macro_rules! bench_encode_sort_case {
        ($name:expr, $f:path) => {
            group.bench_function($name, |b| {
                b.iter(|| {
                    let points = black_box(&points[..]);
                    let mut order = Vec::with_capacity(points.len());
                    for (i, &(x, y)) in points.iter().enumerate() {
                        order.push(($f(x, y), i as u32));
                    }
                    radix_sort_pairs(&mut order, 8);
                    black_box(order);
                });
            });
        };
    }

    bench_encode_sort_case!("hilbert_magic_bits", experimental::magic_bits);
    bench_encode_sort_case!("hilbert_lut", experimental::lut);
    bench_encode_sort_case!("hilbert_loop_rotation", experimental::loop_rotation);
    bench_encode_sort_case!("morton", experimental::morton);
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("sortkey_full_build");
    for &n in &[17usize, 1_000, 100_000] {
        let boxes = gen_boxes(n);
        group.throughput(Throughput::Elements(n as u64));
        for &(name, key) in KEYS {
            group.bench_with_input(BenchmarkId::new(name, n), &boxes, |b, boxes| {
                b.iter(|| {
                    let mut builder = Index2DBuilder::new(boxes.len())
                        .node_size(NODE_SIZE)
                        .experimental_sort_key(key);
                    for &bounds in boxes {
                        builder.add(bounds);
                    }
                    black_box(builder.finish().unwrap());
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_encode_sort, bench_build);
criterion_main!(benches);
