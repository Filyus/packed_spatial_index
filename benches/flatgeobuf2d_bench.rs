//! Compare the core index against FlatGeobuf's packed Hilbert R-tree.
//!
//! Run:
//!   cargo bench --bench flatgeobuf2d_bench --no-default-features --features parallel,simd

use std::hint::black_box;
use std::io::Cursor;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use flatgeobuf::packed_r_tree::{NodeItem, PackedRTree, calc_extent, hilbert_sort};
use packed_spatial_index::experimental::ExperimentalSortKey2D;
use packed_spatial_index::{Bounds2D, Index2D, Index2DBuilder, Index2DView};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NODE_SIZE: usize = 16;
const QUERY_COUNT: usize = 1_000;

fn gen_boxes(n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let cx: f64 = rng.random_range(0.0..10_000.0);
            let cy: f64 = rng.random_range(0.0..10_000.0);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            [cx, cy, cx + w, cy + h]
        })
        .collect()
}

fn make_queries(n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let qx: f64 = rng.random_range(0.0..10_000.0);
            let qy: f64 = rng.random_range(0.0..10_000.0);
            let qw: f64 = rng.random_range(10.0..200.0);
            let qh: f64 = rng.random_range(10.0..200.0);
            [qx, qy, qx + qw, qy + qh]
        })
        .collect()
}

fn to_bounds(q: &[f64; 4]) -> Bounds2D {
    Bounds2D::new(q[0], q[1], q[2], q[3])
}

fn flatgeobuf_nodes(boxes: &[[f64; 4]]) -> Vec<NodeItem> {
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut node = NodeItem::bounds(b[0], b[1], b[2], b[3]);
            node.offset = i as u64;
            node
        })
        .collect()
}

fn build_flatgeobuf(boxes: &[[f64; 4]]) -> PackedRTree {
    let mut nodes = flatgeobuf_nodes(boxes);
    let extent = calc_extent(&nodes);
    hilbert_sort(&mut nodes, &extent);
    PackedRTree::build(&nodes, &extent, NODE_SIZE as u16).unwrap()
}

fn build_flatgeobuf_presorted(nodes: &[NodeItem], extent: &NodeItem) -> PackedRTree {
    PackedRTree::build(nodes, extent, NODE_SIZE as u16).unwrap()
}

fn build_index(boxes: &[[f64; 4]], parallel: bool) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .parallel(parallel)
        .experimental_sort_key(ExperimentalSortKey2D::HilbertLut);
    for b in boxes {
        builder.add(Bounds2D::new(b[0], b[1], b[2], b[3]));
    }
    builder.finish().unwrap()
}

fn build_simd_index(boxes: &[[f64; 4]]) -> packed_spatial_index::SimdIndex2D {
    let mut builder = Index2DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .parallel(true)
        .experimental_sort_key(ExperimentalSortKey2D::HilbertLut);
    for b in boxes {
        builder.add(Bounds2D::new(b[0], b[1], b[2], b[3]));
    }
    builder.finish_simd().unwrap()
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("flatgeobuf_build");
    for &n in &[NODE_SIZE, 1_000, 100_000] {
        let boxes = gen_boxes(n, 0xF6B);
        let mut sorted_nodes = flatgeobuf_nodes(&boxes);
        let extent = calc_extent(&sorted_nodes);
        hilbert_sort(&mut sorted_nodes, &extent);

        group.bench_with_input(
            BenchmarkId::new("flatgeobuf_full", n),
            &boxes,
            |b, boxes| b.iter(|| black_box(build_flatgeobuf(boxes))),
        );
        group.bench_with_input(
            BenchmarkId::new("flatgeobuf_presorted_build_only", n),
            &(sorted_nodes, extent),
            |b, (nodes, extent)| b.iter(|| black_box(build_flatgeobuf_presorted(nodes, extent))),
        );
        group.bench_with_input(BenchmarkId::new("index_serial", n), &boxes, |b, boxes| {
            b.iter(|| black_box(build_index(boxes, false)))
        });
        group.bench_with_input(BenchmarkId::new("index_parallel", n), &boxes, |b, boxes| {
            b.iter(|| black_box(build_index(boxes, true)))
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0xF6B);
    let flatgeobuf = build_flatgeobuf(&boxes);
    let index = build_index(&boxes, false);
    let simd = build_simd_index(&boxes);
    let queries = make_queries(QUERY_COUNT, 0xACE);

    let mut group = c.benchmark_group("flatgeobuf_search");
    group.bench_function("flatgeobuf_search", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += flatgeobuf.search(q[0], q[1], q[2], q[3]).unwrap().len();
            }
            black_box(total)
        })
    });
    group.bench_function("index_search_with", |b| {
        let mut workspace = packed_spatial_index::SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += index.search_with(to_bounds(q), &mut workspace).len();
            }
            black_box(total)
        })
    });
    group.bench_function("simd_search_with", |b| {
        let mut workspace = packed_spatial_index::SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += simd.search_with(to_bounds(q), &mut workspace).len();
            }
            black_box(total)
        })
    });
    group.finish();
}

fn bench_persistence(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0xF6B);
    let flatgeobuf = build_flatgeobuf(&boxes);
    let index = build_index(&boxes, false);
    let index_bytes = index.to_bytes();

    let mut flatgeobuf_bytes = Vec::new();
    flatgeobuf.stream_write(&mut flatgeobuf_bytes).unwrap();

    let mut group = c.benchmark_group("flatgeobuf_persistence");
    group.throughput(Throughput::Bytes(flatgeobuf_bytes.len() as u64));
    group.bench_function("flatgeobuf_stream_write", |b| {
        let mut out = Vec::with_capacity(flatgeobuf_bytes.len());
        b.iter(|| {
            out.clear();
            flatgeobuf.stream_write(&mut out).unwrap();
            black_box(out.len())
        })
    });
    group.bench_function("flatgeobuf_from_buf", |b| {
        b.iter(|| {
            black_box(
                PackedRTree::from_buf(Cursor::new(&flatgeobuf_bytes), n, NODE_SIZE as u16).unwrap(),
            )
        })
    });

    group.throughput(Throughput::Bytes(index_bytes.len() as u64));
    group.bench_function("index_to_bytes", |b| b.iter(|| black_box(index.to_bytes())));
    group.bench_function("index_from_bytes_owned", |b| {
        b.iter(|| black_box(Index2D::from_bytes(&index_bytes).unwrap()))
    });
    group.bench_function("index_from_bytes_view", |b| {
        b.iter(|| black_box(Index2DView::from_bytes(&index_bytes).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench_build, bench_search, bench_persistence);
criterion_main!(benches);
