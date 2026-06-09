#![cfg(feature = "simd")]

//! Zero-copy SIMD views (`SimdIndex2DView`/`SimdIndex3DView`) must return the same
//! results as the AoS reference, over bytes produced by either the AoS or the SoA
//! serializer (the format is shared/canonical).

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index3DBuilder, NeighborWorkspace, Point2D, Point3D,
    SearchWorkspace, SimdIndex2DView, SimdIndex3DView,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::ops::ControlFlow;

fn boxes_2d(n: usize, seed: u64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn boxes_3d(n: usize, seed: u64) -> Vec<Box3D> {
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

fn assert_nondecreasing_distances(items: &[(usize, f64)]) {
    for pair in items.windows(2) {
        assert!(
            pair[0].1 <= pair[1].1,
            "neighbor distances must be nondecreasing: {pair:?}"
        );
    }
}

#[test]
fn simd2d_view_matches_aos_over_both_byte_sources() {
    let boxes = boxes_2d(4_000, 0x7E1);
    let mut aos = Index2DBuilder::new(boxes.len()).node_size(16);
    let mut simd = Index2DBuilder::new(boxes.len()).node_size(16);
    for &b in &boxes {
        aos.add(b);
        simd.add(b);
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    let aos_bytes = aos.to_bytes();
    let simd_bytes = simd.to_bytes();

    let mut ws = SearchWorkspace::new();
    let mut nws = NeighborWorkspace::new();
    let mut rng = StdRng::seed_from_u64(0x1234);

    for source in [&aos_bytes, &simd_bytes] {
        let view = SimdIndex2DView::from_bytes(source).unwrap();
        assert_eq!(view.num_items(), aos.num_items());
        assert_eq!(view.extent(), aos.extent());

        for _ in 0..200 {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            // Windows up to ~1.5x the extent exercise the covered-range fast path,
            // where the query fully contains whole subtrees.
            let w: f64 = rng.random_range(1.0..1_500.0);
            let q = Box2D::new(x, y, x + w, y + w);

            let mut expected = aos.search(q);
            expected.sort_unstable();

            let mut got = view.search(q);
            got.sort_unstable();
            assert_eq!(expected, got, "view.search");

            let mut with = view.search_with(q, &mut ws).to_vec();
            with.sort_unstable();
            assert_eq!(expected, with, "view.search_with");

            assert_eq!(view.any(q), !expected.is_empty());
            assert_eq!(view.first(q).is_some(), !expected.is_empty());

            let mut visited = Vec::new();
            let _: ControlFlow<()> = view.visit(q, |i| {
                visited.push(i);
                ControlFlow::Continue(())
            });
            visited.sort_unstable();
            assert_eq!(expected, visited, "view.visit");

            // KNN order at equal distances is implementation-defined and differs
            // between the AoS and SoA traversals, so the view (SoA family) is compared
            // exactly against the owned SoA index, and as a sorted set against AoS.
            let p = Point2D::new(x, y);
            assert_eq!(
                view.neighbors(p, 10),
                simd.neighbors(p, 10),
                "view vs SoA knn"
            );
            let (mut a, mut b) = (view.neighbors(p, 10), aos.neighbors(p, 10));
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "view knn set vs AoS");
            assert_eq!(
                view.neighbors_with(p, 5, f64::INFINITY, &mut nws),
                simd.neighbors(p, 5).as_slice()
            );

            let mut into = Vec::new();
            view.neighbors_into(p, 5, f64::INFINITY, &mut into);
            assert_eq!(into, simd.neighbors(p, 5), "view.neighbors_into");

            assert_eq!(
                view.neighbors_within(p, 10, 100.0),
                simd.neighbors_within(p, 10, 100.0),
                "view.neighbors_within"
            );

            let mut visited = Vec::new();
            let stopped = view.visit_neighbors(p, f64::INFINITY, |idx, dist| {
                visited.push((idx, dist));
                if visited.len() == 5 {
                    ControlFlow::Break(idx)
                } else {
                    ControlFlow::Continue(())
                }
            });
            assert!(stopped.is_break());
            assert_nondecreasing_distances(&visited);
            assert_eq!(
                visited.iter().map(|&(idx, _)| idx).collect::<Vec<_>>(),
                simd.neighbors(p, 5),
                "view.visit_neighbors"
            );
        }

        // Exact full-extent query must return every item via the contains shortcut.
        let full = view.extent().unwrap();
        assert_eq!(view.search(full).len(), boxes.len(), "view full-extent 2D");
    }
}

#[test]
fn simd3d_view_matches_aos_over_both_byte_sources() {
    let boxes = boxes_3d(4_000, 0x7E3);
    let mut aos = Index3DBuilder::new(boxes.len()).node_size(16);
    let mut simd = Index3DBuilder::new(boxes.len()).node_size(16);
    for &b in &boxes {
        aos.add(b);
        simd.add(b);
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    let aos_bytes = aos.to_bytes();
    let simd_bytes = simd.to_bytes();

    let mut ws = SearchWorkspace::new();
    let mut nws = NeighborWorkspace::new();
    let mut rng = StdRng::seed_from_u64(0x5678);

    for source in [&aos_bytes, &simd_bytes] {
        let view = SimdIndex3DView::from_bytes(source).unwrap();
        assert_eq!(view.num_items(), aos.num_items());
        assert_eq!(view.extent(), aos.extent());

        for _ in 0..200 {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let z: f64 = rng.random_range(0.0..1_000.0);
            // Windows up to ~1.5x the extent exercise the covered-range fast path,
            // where the query fully contains whole subtrees.
            let w: f64 = rng.random_range(1.0..1_500.0);
            let q = Box3D::new(x, y, z, x + w, y + w, z + w);

            let mut expected = aos.search(q);
            expected.sort_unstable();

            let mut got = view.search(q);
            got.sort_unstable();
            assert_eq!(expected, got, "view.search");

            let mut with = view.search_with(q, &mut ws).to_vec();
            with.sort_unstable();
            assert_eq!(expected, with, "view.search_with");

            assert_eq!(view.any(q), !expected.is_empty());
            assert_eq!(view.first(q).is_some(), !expected.is_empty());

            let mut visited = Vec::new();
            let _: ControlFlow<()> = view.visit(q, |i| {
                visited.push(i);
                ControlFlow::Continue(())
            });
            visited.sort_unstable();
            assert_eq!(expected, visited, "view.visit");

            let p = Point3D::new(x, y, z);
            assert_eq!(
                view.neighbors(p, 10),
                simd.neighbors(p, 10),
                "view vs SoA knn"
            );
            let (mut a, mut b) = (view.neighbors(p, 10), aos.neighbors(p, 10));
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "view knn set vs AoS");
            assert_eq!(
                view.neighbors_with(p, 5, f64::INFINITY, &mut nws),
                simd.neighbors(p, 5).as_slice()
            );

            let mut into = Vec::new();
            view.neighbors_into(p, 5, f64::INFINITY, &mut into);
            assert_eq!(into, simd.neighbors(p, 5), "view.neighbors_into");

            assert_eq!(
                view.neighbors_within(p, 10, 100.0),
                simd.neighbors_within(p, 10, 100.0),
                "view.neighbors_within"
            );

            let mut visited = Vec::new();
            let stopped = view.visit_neighbors(p, f64::INFINITY, |idx, dist| {
                visited.push((idx, dist));
                if visited.len() == 5 {
                    ControlFlow::Break(idx)
                } else {
                    ControlFlow::Continue(())
                }
            });
            assert!(stopped.is_break());
            assert_nondecreasing_distances(&visited);
            assert_eq!(
                visited.iter().map(|&(idx, _)| idx).collect::<Vec<_>>(),
                simd.neighbors(p, 5),
                "view.visit_neighbors"
            );
        }

        // Exact full-extent query must return every item via the contains shortcut.
        let full = view.extent().unwrap();
        assert_eq!(view.search(full).len(), boxes.len(), "view full-extent 3D");
    }
}

#[test]
fn simd_view_rejects_wrong_dimension() {
    let b2 = boxes_2d(64, 0x1);
    let mut a2 = Index2DBuilder::new(b2.len());
    for &b in &b2 {
        a2.add(b);
    }
    let bytes2d = a2.finish().unwrap().to_bytes();

    let b3 = boxes_3d(64, 0x2);
    let mut a3 = Index3DBuilder::new(b3.len());
    for &b in &b3 {
        a3.add(b);
    }
    let bytes3d = a3.finish().unwrap().to_bytes();

    assert!(SimdIndex2DView::from_bytes(&bytes3d).is_err());
    assert!(SimdIndex3DView::from_bytes(&bytes2d).is_err());
}
