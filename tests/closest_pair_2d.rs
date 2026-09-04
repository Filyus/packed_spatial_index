use packed_spatial_index::{Box2D, Index2D, Index2DBuilder, Index2DView};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn random_boxes(rng: &mut StdRng, count: usize, extent: f64, max_size: f64) -> Vec<Box2D> {
    (0..count)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..extent);
            let y: f64 = rng.random_range(0.0..extent);
            let w: f64 = rng.random_range(0.0..max_size);
            let h: f64 = rng.random_range(0.0..max_size);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn build(boxes: &[Box2D]) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len());
    for &b in boxes {
        builder.add(b);
    }
    builder.finish().unwrap()
}

/// Brute-force closest cross pair: the distance only, because which pair wins a
/// tie is traversal order and not part of the API.
fn naive_closest(a: &[Box2D], b: &[Box2D]) -> Option<f64> {
    let mut best = f64::INFINITY;
    for box_a in a {
        for box_b in b {
            best = best.min(box_a.distance_to_box(*box_b));
        }
    }
    best.is_finite().then_some(best)
}

fn naive_self_closest(boxes: &[Box2D]) -> Option<f64> {
    let mut best = f64::INFINITY;
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            best = best.min(boxes[i].distance_to_box(boxes[j]));
        }
    }
    best.is_finite().then_some(best)
}

/// The reported pair must really be that far apart — the distance alone would
/// not catch a pair of ids that do not match it.
fn check_pair(a: &[Box2D], b: &[Box2D], found: Option<(usize, usize, f64)>, expected: Option<f64>) {
    match (found, expected) {
        (None, None) => {}
        (Some((i, j, d)), Some(want)) => {
            assert_eq!(d, want, "distance");
            assert_eq!(
                a[i].distance_to_box(b[j]),
                d,
                "ids do not match the distance"
            );
        }
        (found, expected) => panic!("mismatch: {found:?} vs {expected:?}"),
    }
}

#[test]
fn closest_pair_matches_brute_force() {
    let mut rng = StdRng::seed_from_u64(4201);
    for (n, m, max_size) in [
        (0, 5, 4.0),
        (5, 0, 4.0),
        (1, 1, 4.0),
        (23, 41, 8.0),
        (400, 300, 2.0),
        (400, 300, 30.0), // dense enough that many pairs overlap at distance 0
    ] {
        let boxes_a = random_boxes(&mut rng, n, 100.0, max_size);
        let boxes_b = random_boxes(&mut rng, m, 100.0, max_size);
        let a = build(&boxes_a);
        let b = build(&boxes_b);
        check_pair(
            &boxes_a,
            &boxes_b,
            a.closest_pair(&b),
            naive_closest(&boxes_a, &boxes_b),
        );
    }
}

#[test]
fn self_closest_pair_matches_brute_force() {
    let mut rng = StdRng::seed_from_u64(4202);
    for (n, max_size) in [
        (0, 4.0),
        (1, 4.0),
        (2, 4.0),
        (37, 8.0),
        (500, 2.0),
        (500, 30.0),
    ] {
        let boxes = random_boxes(&mut rng, n, 100.0, max_size);
        let index = build(&boxes);
        let found = index.self_closest_pair();
        let expected = naive_self_closest(&boxes);
        match (found, expected) {
            (None, None) => {}
            (Some((i, j, d)), Some(want)) => {
                assert_ne!(i, j, "an item was paired with itself");
                assert_eq!(d, want, "n={n}");
                assert_eq!(boxes[i].distance_to_box(boxes[j]), d);
            }
            (found, expected) => panic!("n={n}: {found:?} vs {expected:?}"),
        }
    }
}

#[test]
fn single_item_index_has_no_self_pair() {
    let index = build(&[Box2D::new(0.0, 0.0, 1.0, 1.0)]);
    assert_eq!(index.self_closest_pair(), None);
    assert_eq!(build(&[]).self_closest_pair(), None);
}

#[test]
fn overlapping_boxes_are_zero_apart() {
    let boxes = [
        Box2D::new(0.0, 0.0, 2.0, 2.0),
        Box2D::new(1.0, 1.0, 3.0, 3.0),
        Box2D::new(90.0, 90.0, 91.0, 91.0),
    ];
    let index = build(&boxes);
    let (i, j, d) = index.self_closest_pair().unwrap();
    assert_eq!((i.min(j), i.max(j)), (0, 1));
    assert_eq!(d, 0.0);
}

#[test]
fn touching_boxes_are_zero_apart() {
    // Edges are inclusive everywhere in this crate, so a shared edge is a
    // distance of exactly zero, not an epsilon above it.
    let a = build(&[Box2D::new(0.0, 0.0, 1.0, 1.0)]);
    let b = build(&[Box2D::new(1.0, 0.0, 2.0, 1.0)]);
    assert_eq!(a.closest_pair(&b), Some((0, 0, 0.0)));
}

#[test]
fn empty_index_has_no_closest_pair() {
    let a = build(&[Box2D::new(0.0, 0.0, 1.0, 1.0)]);
    let empty = build(&[]);
    assert_eq!(a.closest_pair(&empty), None);
    assert_eq!(empty.closest_pair(&a), None);
    assert_eq!(empty.closest_pair(&empty), None);
}

#[test]
fn view_matches_owned() {
    let mut rng = StdRng::seed_from_u64(4203);
    let boxes_a = random_boxes(&mut rng, 300, 100.0, 5.0);
    let boxes_b = random_boxes(&mut rng, 220, 100.0, 5.0);
    let a = build(&boxes_a);
    let b = build(&boxes_b);
    let bytes_a = a.to_bytes();
    let bytes_b = b.to_bytes();
    let view_a = Index2DView::from_bytes(&bytes_a).unwrap();
    let view_b = Index2DView::from_bytes(&bytes_b).unwrap();

    assert_eq!(
        view_a.closest_pair(&view_b).map(|(_, _, d)| d),
        a.closest_pair(&b).map(|(_, _, d)| d)
    );
    assert_eq!(
        view_a.self_closest_pair().map(|(_, _, d)| d),
        a.self_closest_pair().map(|(_, _, d)| d)
    );
}

/// The answer must not depend on which side is `self`.
#[test]
fn closest_pair_is_symmetric_in_distance() {
    let mut rng = StdRng::seed_from_u64(4204);
    for _ in 0..8 {
        let boxes_a = random_boxes(&mut rng, 120, 100.0, 4.0);
        let boxes_b = random_boxes(&mut rng, 90, 100.0, 4.0);
        let a = build(&boxes_a);
        let b = build(&boxes_b);
        assert_eq!(
            a.closest_pair(&b).map(|(_, _, d)| d),
            b.closest_pair(&a).map(|(_, _, d)| d)
        );
    }
}

/// The closest pair is the smallest epsilon at which the join is non-empty.
#[test]
fn closest_pair_agrees_with_join_epsilon() {
    let mut rng = StdRng::seed_from_u64(4205);
    let boxes_a = random_boxes(&mut rng, 250, 100.0, 3.0);
    let boxes_b = random_boxes(&mut rng, 200, 100.0, 3.0);
    let a = build(&boxes_a);
    let b = build(&boxes_b);
    let (_, _, d) = a.closest_pair(&b).unwrap();
    assert!(!a.join_within(&b, d).is_empty(), "empty at the answer");
    if d > 0.0 {
        let just_under = d - d * 1e-9;
        assert!(
            a.join_within(&b, just_under).is_empty(),
            "non-empty below the answer"
        );
    }

    let (_, _, d) = a.self_closest_pair().unwrap();
    assert!(!a.self_join_within(d).is_empty());
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::{SimdIndex2D, SimdIndex2DView};

    fn build_simd(boxes: &[Box2D]) -> SimdIndex2D {
        let mut builder = Index2DBuilder::new(boxes.len());
        for &b in boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    }

    #[test]
    fn simd_and_view_match_brute_force() {
        let mut rng = StdRng::seed_from_u64(4206);
        let boxes_a = random_boxes(&mut rng, 450, 100.0, 4.0);
        let boxes_b = random_boxes(&mut rng, 380, 100.0, 4.0);
        let a = build_simd(&boxes_a);
        let b = build_simd(&boxes_b);
        let bytes_a = a.to_bytes();
        let bytes_b = b.to_bytes();
        let view_a = SimdIndex2DView::from_bytes(&bytes_a).unwrap();
        let view_b = SimdIndex2DView::from_bytes(&bytes_b).unwrap();

        let expected = naive_closest(&boxes_a, &boxes_b);
        check_pair(&boxes_a, &boxes_b, a.closest_pair(&b), expected);
        check_pair(&boxes_a, &boxes_b, view_a.closest_pair(&view_b), expected);

        let expected_self = naive_self_closest(&boxes_a);
        assert_eq!(a.self_closest_pair().map(|(_, _, d)| d), expected_self);
        assert_eq!(view_a.self_closest_pair().map(|(_, _, d)| d), expected_self);
    }
}
