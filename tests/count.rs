//! `count(query)` must agree with `search(query).len()` on every frontend that
//! offers it — it is the same traversal with the result buffer removed, so any
//! divergence is a bug in the counting path, not a cheaper approximation.

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView, Triangle2D,
};

fn boxes2d(n: usize) -> Vec<Box2D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            Box2D::new(x, y, x + 1.5, y + 2.0)
        })
        .collect()
}

fn boxes3d(n: usize) -> Vec<Box3D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            let z = ((i * 5077) % 983) as f64 / 983.0 * 200.0;
            Box3D::new(x, y, z, x + 1.5, y + 2.0, z + 1.0)
        })
        .collect()
}

fn queries2d() -> Vec<Box2D> {
    vec![
        Box2D::new(0.0, 0.0, 10.0, 10.0),           // a corner
        Box2D::new(90.0, 90.0, 120.0, 120.0),       // the middle
        Box2D::new(-50.0, -50.0, 500.0, 500.0),     // everything
        Box2D::new(1e6, 1e6, 1e6 + 1.0, 1e6 + 1.0), // nothing
    ]
}

fn queries3d() -> Vec<Box3D> {
    vec![
        Box3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
        Box3D::new(90.0, 90.0, 90.0, 120.0, 120.0, 120.0),
        Box3D::new(-50.0, -50.0, -50.0, 500.0, 500.0, 500.0),
        Box3D::new(1e6, 1e6, 1e6, 1e6 + 1.0, 1e6 + 1.0, 1e6 + 1.0),
    ]
}

fn build2d(boxes: &[Box2D]) -> packed_spatial_index::Index2D {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(8);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap()
}

fn build3d(boxes: &[Box3D]) -> packed_spatial_index::Index3D {
    let mut b = Index3DBuilder::new(boxes.len()).node_size(8);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap()
}

#[test]
fn owned_2d_count_matches_search_len() {
    let index = build2d(&boxes2d(5_000));
    for q in queries2d() {
        assert_eq!(index.count(q), index.search(q).len(), "query {q:?}");
    }
}

#[test]
fn owned_3d_count_matches_search_len() {
    let index = build3d(&boxes3d(5_000));
    for q in queries3d() {
        assert_eq!(index.count(q), index.search(q).len(), "query {q:?}");
    }
}

#[test]
fn count_accepts_region_queries() {
    let index = build2d(&boxes2d(5_000));
    let tri = Triangle2D::new([10.0, 10.0], [150.0, 20.0], [60.0, 140.0]);
    assert_eq!(index.count(&tri), index.search(&tri).len());
}

#[test]
fn empty_index_counts_zero() {
    let index = build2d(&[]);
    assert_eq!(index.count(Box2D::new(-1e9, -1e9, 1e9, 1e9)), 0);
}

#[test]
fn views_count_like_the_owned_indexes() {
    let index2 = build2d(&boxes2d(3_000));
    let bytes2 = index2.to_bytes();
    let view2 = Index2DView::from_bytes(&bytes2).unwrap();
    for q in queries2d() {
        assert_eq!(view2.count(q), index2.count(q), "2D query {q:?}");
    }

    let index3 = build3d(&boxes3d(3_000));
    let bytes3 = index3.to_bytes();
    let view3 = Index3DView::from_bytes(&bytes3).unwrap();
    for q in queries3d() {
        assert_eq!(view3.count(q), index3.count(q), "3D query {q:?}");
    }
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::{SimdIndex2DView, SimdIndex3DView};

    #[test]
    fn simd_and_its_view_count_like_the_scalar_index() {
        let boxes = boxes2d(3_000);
        let scalar = build2d(&boxes);
        let mut b = Index2DBuilder::new(boxes.len()).node_size(8);
        for bx in &boxes {
            b.add(*bx);
        }
        let simd = b.finish_simd().unwrap();
        let bytes = simd.to_bytes();
        let view = SimdIndex2DView::from_bytes(&bytes).unwrap();
        for q in queries2d() {
            assert_eq!(simd.count(q), scalar.count(q), "2D query {q:?}");
            assert_eq!(view.count(q), scalar.count(q), "2D view query {q:?}");
            assert_eq!(simd.count(q), simd.search(q).len());
        }

        let boxes = boxes3d(3_000);
        let scalar = build3d(&boxes);
        let mut b = Index3DBuilder::new(boxes.len()).node_size(8);
        for bx in &boxes {
            b.add(*bx);
        }
        let simd = b.finish_simd().unwrap();
        let bytes = simd.to_bytes();
        let view = SimdIndex3DView::from_bytes(&bytes).unwrap();
        for q in queries3d() {
            assert_eq!(simd.count(q), scalar.count(q), "3D query {q:?}");
            assert_eq!(view.count(q), scalar.count(q), "3D view query {q:?}");
            assert_eq!(simd.count(q), simd.search(q).len());
        }
    }
}

#[cfg(feature = "f32-storage")]
mod f32_storage {
    use super::*;

    #[test]
    fn scalar_f32_counts_its_own_superset() {
        let boxes = boxes2d(3_000);
        let mut b = Index2DBuilder::new(boxes.len()).node_size(8);
        for bx in &boxes {
            b.add(*bx);
        }
        let index = b.finish_f32().unwrap();
        for q in queries2d() {
            // The f32 answer is a superset of the exact one, so `count` is
            // pinned to that index's own `search`, not to the f64 index.
            assert_eq!(index.count(q), index.search(q).len(), "2D query {q:?}");
        }

        let boxes = boxes3d(3_000);
        let mut b = Index3DBuilder::new(boxes.len()).node_size(8);
        for bx in &boxes {
            b.add(*bx);
        }
        let index = b.finish_f32().unwrap();
        for q in queries3d() {
            assert_eq!(index.count(q), index.search(q).len(), "3D query {q:?}");
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_f32_and_its_view_count_alike() {
        use packed_spatial_index::{SimdIndex2DF32View, SimdIndex3DF32View};

        let boxes = boxes2d(3_000);
        let mut b = Index2DBuilder::new(boxes.len()).node_size(8);
        for bx in &boxes {
            b.add(*bx);
        }
        let index = b.finish_simd_f32().unwrap();
        let bytes = index.to_bytes();
        let view = SimdIndex2DF32View::from_bytes(&bytes).unwrap();
        for q in queries2d() {
            assert_eq!(index.count(q), index.search(q).len(), "2D query {q:?}");
            assert_eq!(view.count(q), index.count(q), "2D view query {q:?}");
        }

        let boxes = boxes3d(3_000);
        let mut b = Index3DBuilder::new(boxes.len()).node_size(8);
        for bx in &boxes {
            b.add(*bx);
        }
        let index = b.finish_simd_f32().unwrap();
        let bytes = index.to_bytes();
        let view = SimdIndex3DF32View::from_bytes(&bytes).unwrap();
        for q in queries3d() {
            assert_eq!(index.count(q), index.search(q).len(), "3D query {q:?}");
            assert_eq!(view.count(q), index.count(q), "3D view query {q:?}");
        }
    }
}

#[cfg(feature = "stream")]
mod stream {
    use super::*;
    use packed_spatial_index::{SliceReader, StreamIndex2D, StreamIndex3D};

    #[test]
    fn streamed_counts_match_the_owned_indexes() {
        let index2 = build2d(&boxes2d(3_000));
        let stream2 = StreamIndex2D::open(SliceReader::new(index2.to_bytes())).unwrap();
        for q in queries2d() {
            assert_eq!(stream2.count(q).unwrap(), index2.count(q), "2D query {q:?}");
        }

        let index3 = build3d(&boxes3d(3_000));
        let stream3 = StreamIndex3D::open(SliceReader::new(index3.to_bytes())).unwrap();
        for q in queries3d() {
            assert_eq!(stream3.count(q).unwrap(), index3.count(q), "3D query {q:?}");
        }
    }
}
