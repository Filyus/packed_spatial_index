#![cfg(feature = "simd")]

use packed_spatial_index::{
    Box3D, BuildError, Index3D, Index3DBuilder, NeighborWorkspace, Point3D, SearchWorkspace,
    SimdIndex3D,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::ops::ControlFlow;

fn random_boxes_3d(n: usize, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let z: f64 = rng.random_range(0.0..1_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn flat_z_boxes_3d(n: usize, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, 10.0, x + dx, y + dy, 10.0)
        })
        .collect()
}

fn degenerate_boxes_3d() -> Vec<Box3D> {
    (0..96)
        .map(|i| {
            let x = (i % 12) as f64;
            let y = ((i / 12) % 8) as f64;
            let z = (i % 4) as f64;
            Box3D::new(x, y, z, x, y, z)
        })
        .collect()
}

fn build_pair(boxes: &[Box3D], node_size: usize) -> (Index3D, SimdIndex3D) {
    let mut aos = Index3DBuilder::new(boxes.len()).node_size(node_size);
    let mut simd = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for &b in boxes {
        aos.add(b);
        simd.add(b);
    }
    (aos.finish().unwrap(), simd.finish_simd().unwrap())
}

fn assert_search_paths_match_aos(boxes: &[Box3D], node_size: usize) {
    let (aos, simd) = build_pair(boxes, node_size);
    let extent = aos.extent().unwrap();
    let mid_x = (extent.min_x + extent.max_x) * 0.5;
    let mid_y = (extent.min_y + extent.max_y) * 0.5;
    let mid_z = (extent.min_z + extent.max_z) * 0.5;
    let queries = [
        extent,
        Box3D::new(
            extent.min_x - 10.0,
            extent.min_y - 10.0,
            extent.min_z - 10.0,
            extent.min_x - 1.0,
            extent.min_y - 1.0,
            extent.min_z - 1.0,
        ),
        Box3D::new(
            mid_x - 25.0,
            mid_y - 25.0,
            mid_z - 25.0,
            mid_x + 25.0,
            mid_y + 25.0,
            mid_z + 25.0,
        ),
        Box3D::new(
            extent.min_x,
            extent.min_y,
            mid_z,
            extent.max_x,
            extent.max_y,
            mid_z,
        ),
        Box3D::new(
            extent.min_x,
            extent.min_y,
            extent.min_z,
            extent.min_x,
            extent.min_y,
            extent.min_z,
        ),
    ];

    let (mut scalar, mut wide, mut avx) = (Vec::new(), Vec::new(), Vec::new());
    let (mut s1, mut s2, mut s3) = (Vec::new(), Vec::new(), Vec::new());
    let mut workspace = SearchWorkspace::new();

    for query in queries {
        let mut expected = aos.search(query);
        expected.sort_unstable();

        simd.search_scalar(query, &mut scalar, &mut s1);
        simd.search_simd(query, &mut wide, &mut s2);
        simd.search_avx512(query, &mut avx, &mut s3);
        let mut public = simd.search_with(query, &mut workspace).to_vec();

        scalar.sort_unstable();
        wide.sort_unstable();
        avx.sort_unstable();
        public.sort_unstable();

        assert_eq!(expected, scalar, "SoA-scalar != AoS, node_size={node_size}");
        assert_eq!(expected, wide, "SoA-wide != AoS, node_size={node_size}");
        assert_eq!(expected, avx, "SoA-AVX512 != AoS, node_size={node_size}");
        assert_eq!(expected, public, "SoA-public != AoS, node_size={node_size}");

        assert_eq!(simd.any(query), !expected.is_empty());
        if let Some(index) = simd.first(query) {
            assert!(expected.binary_search(&index).is_ok());
        } else {
            assert!(expected.is_empty());
        }
    }
}

#[test]
fn simd3d_large_window_search_matches_scalar() {
    // `search_scalar` does not use the covered-range fast path, so it is an
    // independent oracle for the SIMD paths over large windows that fully contain
    // whole subtrees.
    let boxes = random_boxes_3d(8_000, 0xC0FFEE);
    let simd = {
        let mut builder = Index3DBuilder::new(boxes.len()).node_size(16);
        for &b in &boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    };
    let extent = simd.extent().unwrap();

    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let (mut scalar, mut wide, mut avx) = (Vec::new(), Vec::new(), Vec::new());
    let (mut s1, mut s2, mut s3) = (Vec::new(), Vec::new(), Vec::new());

    for size in [50.0, 250.0, 1_000.0, 5_000.0, 100_000.0] {
        for _ in 0..30 {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let z: f64 = rng.random_range(0.0..1_000.0);
            let query = Box3D::new(x, y, z, x + size, y + size, z + size);

            simd.search_scalar(query, &mut scalar, &mut s1);
            simd.search_simd(query, &mut wide, &mut s2);
            simd.search_avx512(query, &mut avx, &mut s3);
            scalar.sort_unstable();
            wide.sort_unstable();
            avx.sort_unstable();
            assert_eq!(scalar, wide, "SoA-wide large window != scalar");
            assert_eq!(scalar, avx, "SoA-AVX512 large window != scalar");
        }
    }

    let full = extent;
    simd.search_simd(full, &mut wide, &mut s2);
    simd.search_avx512(full, &mut avx, &mut s3);
    assert_eq!(wide.len(), boxes.len(), "full-extent SIMD must return all");
    assert_eq!(avx.len(), boxes.len(), "full-extent AVX512 must return all");
}

#[test]
fn simd3d_empty_and_small_indexes_behave_like_aos() {
    let empty = Index3DBuilder::new(0).finish_simd().unwrap();
    assert_eq!(empty.num_items(), 0);
    assert_eq!(empty.extent(), None);
    assert!(
        empty
            .search(Box3D::new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
            .is_empty()
    );

    let boxes = random_boxes_3d(5, 0xA11);
    let mut aos = Index3DBuilder::new(boxes.len());
    let mut simd = Index3DBuilder::new(boxes.len());
    for &b in &boxes {
        aos.add(b);
        simd.add(b);
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    assert_eq!(simd.extent(), aos.extent());

    let query = Box3D::new(0.0, 0.0, 0.0, 1_000.0, 1_000.0, 1_000.0);
    let mut expected = aos.search(query);
    let mut actual = simd.search(query);
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(expected, actual);
}

#[test]
fn simd3d_finish_reports_count_mismatch() {
    let mut builder = Index3DBuilder::new(2);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));

    assert!(matches!(
        builder.finish_simd(),
        Err(BuildError::ItemCount {
            added: 1,
            expected: 2
        })
    ));
}

#[test]
fn simd3d_search_apis_agree_with_aos() {
    let boxes = random_boxes_3d(2_000, 0x3D5);
    let (aos, simd) = build_pair(&boxes, 16);

    let mut rng = StdRng::seed_from_u64(0x9999);
    let (mut scalar, mut wide, mut avx, mut visit_wide, mut visit_avx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut s1, mut s2, mut s3, mut s4, mut s5) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut workspace = SearchWorkspace::new();

    for _ in 0..400 {
        let x: f64 = rng.random_range(0.0..1_000.0);
        let y: f64 = rng.random_range(0.0..1_000.0);
        let z: f64 = rng.random_range(0.0..1_000.0);
        let w: f64 = rng.random_range(1.0..150.0);
        let query = Box3D::new(x, y, z, x + w, y + w, z + w);

        let mut expected = aos.search(query);
        expected.sort_unstable();

        // Every SoA traversal path must agree with the AoS reference.
        simd.search_scalar(query, &mut scalar, &mut s1);
        simd.search_simd(query, &mut wide, &mut s2);
        simd.search_avx512(query, &mut avx, &mut s3);
        visit_wide.clear();
        visit_avx.clear();
        assert!(
            simd.visit_simd(query, &mut s4, |idx| {
                visit_wide.push(idx);
                ControlFlow::<()>::Continue(())
            })
            .is_continue()
        );
        assert!(
            simd.visit_avx512(query, &mut s5, |idx| {
                visit_avx.push(idx);
                ControlFlow::<()>::Continue(())
            })
            .is_continue()
        );
        let mut high = simd.search(query);
        let mut into = Vec::new();
        simd.search_into(query, &mut into);
        let mut with = simd.search_with(query, &mut workspace).to_vec();

        scalar.sort_unstable();
        wide.sort_unstable();
        avx.sort_unstable();
        visit_wide.sort_unstable();
        visit_avx.sort_unstable();
        high.sort_unstable();
        into.sort_unstable();
        with.sort_unstable();

        assert_eq!(expected, scalar, "SoA-scalar != AoS");
        assert_eq!(expected, wide, "SoA-wide != AoS");
        assert_eq!(expected, avx, "SoA-AVX512 != AoS");
        assert_eq!(expected, visit_wide, "SoA-visit-wide != AoS");
        assert_eq!(expected, visit_avx, "SoA-visit-AVX512 != AoS");
        assert_eq!(expected, high, "SoA-search != AoS");
        assert_eq!(expected, into, "SoA-search_into != AoS");
        assert_eq!(expected, with, "SoA-search_with != AoS");

        assert_eq!(simd.any(query), !expected.is_empty());
        if let Some(idx) = simd.first(query) {
            assert!(expected.binary_search(&idx).is_ok());
        } else {
            assert!(expected.is_empty());
        }

        let mut visited = Vec::new();
        let done: ControlFlow<()> = simd.visit(query, |idx| {
            visited.push(idx);
            ControlFlow::Continue(())
        });
        assert!(done.is_continue());
        visited.sort_unstable();
        assert_eq!(expected, visited, "SoA-visit != AoS");
    }

    let full = simd.extent().unwrap();
    let mut expected = (0..boxes.len()).collect::<Vec<_>>();
    scalar.clear();
    wide.clear();
    avx.clear();
    simd.search_scalar(full, &mut scalar, &mut s1);
    simd.search_simd(full, &mut wide, &mut s2);
    simd.search_avx512(full, &mut avx, &mut s3);
    visit_wide.clear();
    visit_avx.clear();
    let _: ControlFlow<()> = simd.visit_simd(full, &mut s4, |idx| {
        visit_wide.push(idx);
        ControlFlow::Continue(())
    });
    let _: ControlFlow<()> = simd.visit_avx512(full, &mut s5, |idx| {
        visit_avx.push(idx);
        ControlFlow::Continue(())
    });
    scalar.sort_unstable();
    wide.sort_unstable();
    avx.sort_unstable();
    visit_wide.sort_unstable();
    visit_avx.sort_unstable();
    expected.sort_unstable();
    assert_eq!(expected, scalar, "SoA-scalar full extent");
    assert_eq!(expected, wide, "SoA-wide full extent");
    assert_eq!(expected, avx, "SoA-AVX512 full extent");
    assert_eq!(expected, visit_wide, "SoA-visit-wide full extent");
    assert_eq!(expected, visit_avx, "SoA-visit-AVX512 full extent");
}

#[test]
fn simd3d_search_matches_aos_for_edge_shapes_and_node_sizes() {
    let random = random_boxes_3d(513, 0x513D);
    let flat_z = flat_z_boxes_3d(513, 0xF1A7);
    let degenerate = degenerate_boxes_3d();

    for boxes in [&random[..], &flat_z[..], &degenerate[..]] {
        for node_size in [8, 16, 32] {
            assert_search_paths_match_aos(boxes, node_size);
        }
    }
}

#[test]
fn simd3d_neighbors_match_aos() {
    let boxes = random_boxes_3d(2_000, 0x4B3D);
    let mut aos = Index3DBuilder::new(boxes.len()).node_size(16);
    let mut simd = Index3DBuilder::new(boxes.len()).node_size(16);
    for &b in &boxes {
        aos.add(b);
        simd.add(b);
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    let mut rng = StdRng::seed_from_u64(0x7777);
    let mut workspace = NeighborWorkspace::new();
    for _ in 0..100 {
        let point = Point3D::new(
            rng.random_range(0.0..1_000.0),
            rng.random_range(0.0..1_000.0),
            rng.random_range(0.0..1_000.0),
        );
        assert_eq!(simd.neighbors(point, 16), aos.neighbors(point, 16));
        assert_eq!(
            simd.neighbors_within(point, 16, 100.0),
            aos.neighbors_within(point, 16, 100.0)
        );

        let mut out = Vec::new();
        simd.neighbors_into(point, 8, f64::INFINITY, &mut out);
        assert_eq!(out, aos.neighbors(point, 8));

        assert_eq!(
            simd.neighbors_with(point, 8, f64::INFINITY, &mut workspace),
            aos.neighbors(point, 8).as_slice()
        );
    }
}

#[test]
fn simd3d_neighbors_match_aos_for_flat_z_node_sizes() {
    let boxes = flat_z_boxes_3d(513, 0xBEEF);
    let points = [
        Point3D::new(10.0, 10.0, 10.0),
        Point3D::new(250.0, 500.0, 10.0),
        Point3D::new(900.0, 100.0, 25.0),
    ];

    for node_size in [8, 16, 32] {
        let (aos, simd) = build_pair(&boxes, node_size);
        let mut workspace = NeighborWorkspace::new();
        for point in points {
            assert_eq!(simd.neighbors(point, 16), aos.neighbors(point, 16));
            assert_eq!(
                simd.neighbors_within(point, 16, 100.0),
                aos.neighbors_within(point, 16, 100.0)
            );
            assert_eq!(
                simd.neighbors_with(point, 8, f64::INFINITY, &mut workspace),
                aos.neighbors(point, 8).as_slice()
            );
        }
    }
}
