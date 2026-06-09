use std::collections::BTreeSet;
use std::ops::ControlFlow;

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

fn naive_join(a: &[Box2D], b: &[Box2D]) -> BTreeSet<(usize, usize)> {
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

fn naive_self_join(boxes: &[Box2D]) -> BTreeSet<(usize, usize)> {
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
fn join_matches_naive_pairs() {
    let mut rng = StdRng::seed_from_u64(101);
    for (n, m, max_size) in [(0, 7, 4.0), (1, 1, 4.0), (37, 5, 8.0), (500, 400, 3.0)] {
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
fn self_join_matches_naive_pairs() {
    let mut rng = StdRng::seed_from_u64(202);
    for (n, max_size) in [(0, 4.0), (1, 4.0), (2, 100.0), (33, 8.0), (700, 2.5)] {
        let boxes = random_boxes(&mut rng, n, 100.0, max_size);
        let index = build(&boxes);

        let expected = naive_self_join(&boxes);
        assert_eq!(normalized(index.self_join()), expected, "n={n}");
    }
}

#[test]
fn join_handles_touching_edges_inclusively() {
    let a = build(&[Box2D::new(0.0, 0.0, 1.0, 1.0)]);
    let b = build(&[Box2D::new(1.0, 1.0, 2.0, 2.0), Box2D::new(1.5, 0.0, 2.0, 0.5)]);
    assert_eq!(a.join(&b), vec![(0, 0)]);
}

#[test]
fn join_with_supports_early_exit() {
    let mut rng = StdRng::seed_from_u64(303);
    let boxes = random_boxes(&mut rng, 300, 50.0, 5.0);
    let index = build(&boxes);

    let total = index.self_join().len();
    assert!(total > 10, "test needs a pair-rich input, got {total}");

    let mut seen = 0usize;
    let flow = index.self_join_with(|_, _| {
        seen += 1;
        if seen == 10 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(flow, ControlFlow::Break(()));
    assert_eq!(seen, 10);
}

#[test]
fn view_join_matches_owned_join() {
    let mut rng = StdRng::seed_from_u64(404);
    let boxes_a = random_boxes(&mut rng, 250, 100.0, 6.0);
    let boxes_b = random_boxes(&mut rng, 180, 100.0, 6.0);
    let a = build(&boxes_a);
    let b = build(&boxes_b);
    let bytes_a = a.to_bytes();
    let bytes_b = b.to_bytes();
    let view_a = Index2DView::from_bytes(&bytes_a).unwrap();
    let view_b = Index2DView::from_bytes(&bytes_b).unwrap();

    let owned: BTreeSet<_> = a.join(&b).into_iter().collect();
    let viewed: BTreeSet<_> = view_a.join(&view_b).into_iter().collect();
    assert_eq!(owned, viewed);
    assert_eq!(normalized(view_a.self_join()), normalized(a.self_join()));
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::SimdIndex2DView;

    fn build_simd(boxes: &[Box2D]) -> packed_spatial_index::SimdIndex2D {
        let mut builder = Index2DBuilder::new(boxes.len());
        for &b in boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    }

    #[test]
    fn simd_join_matches_naive_pairs() {
        let mut rng = StdRng::seed_from_u64(505);
        let boxes_a = random_boxes(&mut rng, 450, 100.0, 4.0);
        let boxes_b = random_boxes(&mut rng, 350, 100.0, 4.0);
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
    fn simd_view_join_matches_owned() {
        let mut rng = StdRng::seed_from_u64(606);
        let boxes_a = random_boxes(&mut rng, 200, 100.0, 5.0);
        let boxes_b = random_boxes(&mut rng, 240, 100.0, 5.0);
        let a = build_simd(&boxes_a);
        let b = build_simd(&boxes_b);
        let bytes_a = a.to_bytes();
        let bytes_b = b.to_bytes();
        let view_a = SimdIndex2DView::from_bytes(&bytes_a).unwrap();
        let view_b = SimdIndex2DView::from_bytes(&bytes_b).unwrap();

        let owned: BTreeSet<_> = a.join(&b).into_iter().collect();
        let viewed: BTreeSet<_> = view_a.join(&view_b).into_iter().collect();
        assert_eq!(owned, viewed);
        assert_eq!(normalized(view_a.self_join()), normalized(a.self_join()));
    }
}
