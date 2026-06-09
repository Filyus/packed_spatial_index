use std::collections::BTreeSet;

use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn random_boxes(rng: &mut StdRng, count: usize, extent: f64, max_size: f64) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..extent);
            let y: f64 = rng.random_range(0.0..extent);
            let z: f64 = rng.random_range(0.0..extent);
            let w: f64 = rng.random_range(0.0..max_size);
            let h: f64 = rng.random_range(0.0..max_size);
            let d: f64 = rng.random_range(0.0..max_size);
            Box3D::new(x, y, z, x + w, y + h, z + d)
        })
        .collect()
}

fn build(boxes: &[Box3D]) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len());
    for &b in boxes {
        builder.add(b);
    }
    builder.finish().unwrap()
}

fn naive_join(a: &[Box3D], b: &[Box3D]) -> BTreeSet<(usize, usize)> {
    let mut out = BTreeSet::new();
    for (i, box_a) in a.iter().enumerate() {
        for (j, box_b) in b.iter().enumerate() {
            if box_a.overlaps(*box_b) {
                out.insert((i, j));
            }
        }
    }
    out
}

fn naive_self_join(boxes: &[Box3D]) -> BTreeSet<(usize, usize)> {
    let mut out = BTreeSet::new();
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            if boxes[i].overlaps(boxes[j]) {
                out.insert((i, j));
            }
        }
    }
    out
}

fn normalized(pairs: Vec<(usize, usize)>) -> BTreeSet<(usize, usize)> {
    let normalized: BTreeSet<_> = pairs.iter().map(|&(i, j)| (i.min(j), i.max(j))).collect();
    assert_eq!(normalized.len(), pairs.len(), "duplicate pairs reported");
    normalized
}

#[test]
fn join_matches_naive_pairs_3d() {
    let mut rng = StdRng::seed_from_u64(111);
    for (n, m, max_size) in [(0, 5, 10.0), (1, 1, 10.0), (40, 9, 20.0), (400, 350, 8.0)] {
        let boxes_a = random_boxes(&mut rng, n, 100.0, max_size);
        let boxes_b = random_boxes(&mut rng, m, 100.0, max_size);
        let a = build(&boxes_a);
        let b = build(&boxes_b);

        let expected = naive_join(&boxes_a, &boxes_b);
        let actual: BTreeSet<_> = a.join(&b).into_iter().collect();
        assert_eq!(a.join(&b).len(), expected.len(), "duplicate pairs reported");
        assert_eq!(actual, expected, "n={n} m={m}");
    }
}

#[test]
fn self_join_matches_naive_pairs_3d() {
    let mut rng = StdRng::seed_from_u64(222);
    for (n, max_size) in [(0, 10.0), (1, 10.0), (2, 200.0), (50, 25.0), (600, 6.0)] {
        let boxes = random_boxes(&mut rng, n, 100.0, max_size);
        let index = build(&boxes);

        let expected = naive_self_join(&boxes);
        assert_eq!(normalized(index.self_join()), expected, "n={n}");
    }
}

#[test]
fn view_join_matches_owned_join_3d() {
    let mut rng = StdRng::seed_from_u64(333);
    let boxes_a = random_boxes(&mut rng, 220, 100.0, 12.0);
    let boxes_b = random_boxes(&mut rng, 260, 100.0, 12.0);
    let a = build(&boxes_a);
    let b = build(&boxes_b);
    let bytes_a = a.to_bytes();
    let bytes_b = b.to_bytes();
    let view_a = Index3DView::from_bytes(&bytes_a).unwrap();
    let view_b = Index3DView::from_bytes(&bytes_b).unwrap();

    let owned: BTreeSet<_> = a.join(&b).into_iter().collect();
    let viewed: BTreeSet<_> = view_a.join(&view_b).into_iter().collect();
    assert_eq!(owned, viewed);
    assert_eq!(normalized(view_a.self_join()), normalized(a.self_join()));
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::SimdIndex3DView;

    fn build_simd(boxes: &[Box3D]) -> packed_spatial_index::SimdIndex3D {
        let mut builder = Index3DBuilder::new(boxes.len());
        for &b in boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    }

    #[test]
    fn simd_join_matches_naive_pairs_3d() {
        let mut rng = StdRng::seed_from_u64(444);
        let boxes_a = random_boxes(&mut rng, 380, 100.0, 10.0);
        let boxes_b = random_boxes(&mut rng, 300, 100.0, 10.0);
        let a = build_simd(&boxes_a);
        let b = build_simd(&boxes_b);

        let expected = naive_join(&boxes_a, &boxes_b);
        let actual: BTreeSet<_> = a.join(&b).into_iter().collect();
        assert_eq!(a.join(&b).len(), expected.len());
        assert_eq!(actual, expected);

        let expected_self = naive_self_join(&boxes_a);
        assert_eq!(normalized(a.self_join()), expected_self);
    }

    #[test]
    fn simd_view_join_matches_owned_3d() {
        let mut rng = StdRng::seed_from_u64(555);
        let boxes_a = random_boxes(&mut rng, 210, 100.0, 14.0);
        let boxes_b = random_boxes(&mut rng, 190, 100.0, 14.0);
        let a = build_simd(&boxes_a);
        let b = build_simd(&boxes_b);
        let bytes_a = a.to_bytes();
        let bytes_b = b.to_bytes();
        let view_a = SimdIndex3DView::from_bytes(&bytes_a).unwrap();
        let view_b = SimdIndex3DView::from_bytes(&bytes_b).unwrap();

        let owned: BTreeSet<_> = a.join(&b).into_iter().collect();
        let viewed: BTreeSet<_> = view_a.join(&view_b).into_iter().collect();
        assert_eq!(owned, viewed);
        assert_eq!(normalized(view_a.self_join()), normalized(a.self_join()));
    }
}
