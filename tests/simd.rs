#![cfg(feature = "simd")]

mod common;

use common::random_boxes;
use packed_spatial_index::experimental::ExperimentalSortKey;
use packed_spatial_index::{
    BuildError, IndexBuilder, NeighborWorkspace, Point, Rect, SearchWorkspace,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use static_aabb2d_index::StaticAABB2DIndexBuilder;
use std::ops::ControlFlow;

#[test]
fn simd_empty_and_small_indexes_behave_like_aos() {
    let empty = IndexBuilder::new(0).finish_simd().unwrap();
    assert_eq!(empty.num_items(), 0);
    assert_eq!(empty.bounds(), None);
    assert!(empty.search(Rect::new(-1.0, -1.0, 1.0, 1.0)).is_empty());

    let boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [2.0, 2.0, 3.0, 3.0],
        [-1.0, -1.0, 0.5, 0.5],
    ];
    let mut aos = IndexBuilder::new(boxes.len());
    let mut simd = IndexBuilder::new(boxes.len());
    for b in boxes {
        aos.add(Rect::new(b[0], b[1], b[2], b[3]));
        simd.add(Rect::new(b[0], b[1], b[2], b[3]));
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    assert_eq!(simd.bounds(), aos.bounds());

    let query = Rect::new(-0.25, -0.25, 2.25, 2.25);
    let mut expected = aos.search(query);
    let mut actual = simd.search(query);
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(expected, actual);
}

#[test]
fn simd_finish_reports_count_mismatch() {
    let mut builder = IndexBuilder::new(2);
    builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));

    assert!(matches!(
        builder.finish_simd(),
        Err(BuildError::ItemCount {
            added: 1,
            expected: 2
        })
    ));
}

#[test]
fn simd_search_apis_agree_with_aos() {
    let mut builder = IndexBuilder::new(3);
    builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));
    builder.add(Rect::new(0.5, 0.5, 2.0, 2.0));
    let simd = builder.finish_simd().unwrap();

    let query = Rect::new(0.0, 0.0, 2.0, 2.0);
    let mut expected = simd.search(query);
    expected.sort_unstable();

    let mut out = Vec::new();
    simd.search_into(query, &mut out);
    out.sort_unstable();
    assert_eq!(expected, out);

    let mut workspace = SearchWorkspace::new();
    let mut with = simd.search_with(query, &mut workspace).to_vec();
    with.sort_unstable();
    assert_eq!(expected, with);

    assert!(simd.any(query));
    assert!(!simd.any(Rect::new(10.0, 10.0, 11.0, 11.0)));
    assert!(matches!(simd.first(query), Some(0 | 2)));
    assert_eq!(simd.first(Rect::new(10.0, 10.0, 11.0, 11.0)), None);

    let mut visited = Vec::new();
    let completed: ControlFlow<()> = simd.visit(query, |idx| {
        visited.push(idx);
        ControlFlow::Continue(())
    });
    assert!(completed.is_continue());
    visited.sort_unstable();
    assert_eq!(expected, visited);
}

#[test]
fn simd_neighbors_match_aos() {
    let mut rng = StdRng::seed_from_u64(0x51D);
    let boxes = random_boxes(&mut rng, 1_000);

    let mut aos_builder = IndexBuilder::new(boxes.len()).node_size(16);
    let mut simd_builder = IndexBuilder::new(boxes.len()).node_size(16);
    for b in &boxes {
        aos_builder.add(Rect::new(b[0], b[1], b[2], b[3]));
        simd_builder.add(Rect::new(b[0], b[1], b[2], b[3]));
    }
    let aos = aos_builder.finish().unwrap();
    let simd = simd_builder.finish_simd().unwrap();

    for _ in 0..100 {
        let point = Point::new(rng.random_range(0.0..1000.0), rng.random_range(0.0..1000.0));
        assert_eq!(simd.neighbors(point, 16), aos.neighbors(point, 16));
        assert_eq!(
            simd.neighbors_within(point, 16, 100.0),
            aos.neighbors_within(point, 16, 100.0)
        );

        let mut out = Vec::new();
        simd.neighbors_into(point, 8, f64::INFINITY, &mut out);
        assert_eq!(out, aos.neighbors(point, 8));

        let mut workspace = NeighborWorkspace::new();
        assert_eq!(
            simd.neighbors_with(point, 8, f64::INFINITY, &mut workspace),
            aos.neighbors(point, 8).as_slice()
        );
    }
}

#[test]
fn simd_index_search_matches_reference() {
    let mut rng = StdRng::seed_from_u64(99);
    let n = 5_000usize;
    let node_size = 16usize;
    let boxes = random_boxes(&mut rng, n);

    let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(n, node_size);
    let mut builder = IndexBuilder::new(n)
        .node_size(node_size)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut);
    for b in &boxes {
        reference.add(b[0], b[1], b[2], b[3]);
        builder.add(Rect::new(b[0], b[1], b[2], b[3]));
    }
    let reference = reference.build().unwrap();
    let simd = builder.finish_simd().unwrap();

    let (mut scalar, mut simd_out, mut simd_prefetch, mut avx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut st1, mut st2, mut st3, mut st4) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..500 {
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let qw: f64 = rng.random_range(1.0..100.0);
        let qh: f64 = rng.random_range(1.0..100.0);
        let query = Rect::new(qx, qy, qx + qw, qy + qh);

        let mut expected = reference.query(qx, qy, qx + qw, qy + qh);
        simd.search_scalar(query, &mut scalar, &mut st1);
        simd.search_simd(query, &mut simd_out, &mut st2);
        simd.search_simd_prefetch(query, &mut simd_prefetch, &mut st3);
        simd.search_avx512(query, &mut avx, &mut st4);
        expected.sort_unstable();
        scalar.sort_unstable();
        simd_out.sort_unstable();
        simd_prefetch.sort_unstable();
        avx.sort_unstable();
        assert_eq!(expected, scalar, "SoA-scalar != reference");
        assert_eq!(expected, simd_out, "SoA-SIMD != reference");
        assert_eq!(expected, simd_prefetch, "SoA-SIMD-prefetch != reference");
        assert_eq!(expected, avx, "SoA-AVX512 != reference");
    }
}
