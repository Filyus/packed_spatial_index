//! `estimate_count`: the bracket is exact, the estimate sits inside it, and
//! `stop_level = 0` reproduces `count`, on every 2D type that carries it.

use packed_spatial_index::{Box2D, Index2D, Index2DBuilder, Index2DView};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn random_boxes(rng: &mut StdRng, count: usize, extent: f64, max_size: f64) -> Vec<Box2D> {
    (0..count)
        .map(|_| {
            let x = rng.random_range(0.0..extent);
            let y = rng.random_range(0.0..extent);
            let w = rng.random_range(0.0..max_size);
            let h = rng.random_range(0.0..max_size);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn clustered_boxes(rng: &mut StdRng, count: usize) -> Vec<Box2D> {
    let centres = [(20.0, 20.0), (80.0, 30.0), (50.0, 90.0)];
    (0..count)
        .map(|i| {
            let (cx, cy) = centres[i % centres.len()];
            let x = cx + rng.random_range(-5.0..5.0);
            let y = cy + rng.random_range(-5.0..5.0);
            Box2D::new(x, y, x + 0.2, y + 0.2)
        })
        .collect()
}

fn build(boxes: &[Box2D], node_size: usize) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for b in boxes {
        builder.add(*b);
    }
    builder.finish().unwrap()
}

fn windows() -> Vec<Box2D> {
    vec![
        Box2D::new(10.0, 10.0, 30.0, 30.0),
        Box2D::new(0.0, 0.0, 100.0, 100.0),
        Box2D::new(-10.0, -10.0, 200.0, 200.0),
        Box2D::new(48.0, 48.0, 52.0, 52.0),
        Box2D::new(300.0, 300.0, 310.0, 310.0),
        Box2D::new(25.0, 0.0, 26.0, 100.0),
    ]
}

fn check_index(index: &Index2D, label: &str) {
    let view_bytes = index.to_bytes();
    let view = Index2DView::from_bytes(&view_bytes).unwrap();
    for window in windows() {
        let exact = index.count(window);
        let mut previous_spread = usize::MAX;
        for stop_level in (0..6).rev() {
            let est = index.estimate_count(window, stop_level);
            assert!(
                est.lower <= exact && exact <= est.upper,
                "{label} window {window:?} level {stop_level}: {exact} not in [{}, {}]",
                est.lower,
                est.upper
            );
            assert!(
                est.lower as f64 <= est.estimate && est.estimate <= est.upper as f64,
                "{label} estimate {} outside [{}, {}]",
                est.estimate,
                est.lower,
                est.upper
            );
            // Descending further can only tighten the bracket.
            assert!(
                est.spread() <= previous_spread,
                "{label} level {stop_level} widened"
            );
            previous_spread = est.spread();
            assert!(est.nodes_tested >= 1);

            assert_eq!(view.estimate_count(window, stop_level), est, "{label} view");
        }
        let exact_est = index.estimate_count(window, 0);
        assert_eq!(
            (exact_est.lower, exact_est.upper),
            (exact, exact),
            "{label} level 0"
        );
        assert_eq!(exact_est.estimate, exact as f64);
    }
}

#[test]
fn bracket_holds_on_uniform_data_at_every_node_size() {
    let mut rng = StdRng::seed_from_u64(3101);
    let boxes = random_boxes(&mut rng, 2000, 100.0, 3.0);
    for node_size in [2, 4, 16, 64] {
        check_index(&build(&boxes, node_size), &format!("uniform n{node_size}"));
    }
}

#[test]
fn bracket_holds_on_clustered_data() {
    let mut rng = StdRng::seed_from_u64(3102);
    let boxes = clustered_boxes(&mut rng, 3000);
    check_index(&build(&boxes, 16), "clustered");
}

#[test]
fn contained_root_and_missed_root_are_one_test_each() {
    let mut rng = StdRng::seed_from_u64(3103);
    let boxes = random_boxes(&mut rng, 500, 100.0, 2.0);
    let index = build(&boxes, 16);
    let all = index.estimate_count(Box2D::new(-1.0, -1.0, 200.0, 200.0), 3);
    assert_eq!((all.lower, all.upper, all.nodes_tested), (500, 500, 1));
    let none = index.estimate_count(Box2D::new(500.0, 500.0, 600.0, 600.0), 3);
    assert_eq!(
        (none.lower, none.upper, none.estimate, none.nodes_tested),
        (0, 0, 0.0, 1)
    );
}

#[test]
fn stopping_high_is_cheaper_than_stopping_low() {
    let mut rng = StdRng::seed_from_u64(3104);
    let boxes = random_boxes(&mut rng, 20_000, 1000.0, 2.0);
    let index = build(&boxes, 16);
    let window = Box2D::new(100.0, 100.0, 400.0, 400.0);
    let high = index.estimate_count(window, 2);
    let low = index.estimate_count(window, 1);
    let exact = index.estimate_count(window, 0);
    assert!(high.nodes_tested < low.nodes_tested && low.nodes_tested < exact.nodes_tested);
    assert!(high.spread() >= low.spread() && low.spread() >= exact.spread());
    assert_eq!(exact.spread(), 0);
    // On uniform data the level-1 estimate lands close to the truth.
    let truth = exact.lower as f64;
    assert!(
        (low.estimate - truth).abs() / truth < 0.05,
        "{} vs {truth}",
        low.estimate
    );
}

#[test]
fn empty_index_estimates_nothing() {
    let index = Index2DBuilder::new(0).finish().unwrap();
    let est = index.estimate_count(Box2D::new(0.0, 0.0, 1.0, 1.0), 1);
    assert_eq!((est.lower, est.upper, est.nodes_tested), (0, 0, 0));
}

#[test]
fn point_items_score_whole_when_overlapped() {
    // Degenerate boxes: a node of points cut by the window still brackets.
    let boxes: Vec<Box2D> = (0..64)
        .map(|i| {
            let v = i as f64;
            Box2D::new(v, 0.0, v, 0.0)
        })
        .collect();
    let index = build(&boxes, 4);
    let window = Box2D::new(10.0, -1.0, 20.0, 1.0);
    let exact = index.count(window);
    assert_eq!(exact, 11);
    let est = index.estimate_count(window, 1);
    assert!(est.lower <= exact && exact <= est.upper);
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::SimdIndex2DView;

    #[test]
    fn simd_index_and_view_agree_with_owned() {
        let mut rng = StdRng::seed_from_u64(3105);
        let boxes = random_boxes(&mut rng, 1500, 100.0, 3.0);
        let index = build(&boxes, 16);
        let mut builder = Index2DBuilder::new(boxes.len()).node_size(16);
        for b in &boxes {
            builder.add(*b);
        }
        let simd = builder.finish_simd().unwrap();
        let bytes = simd.to_bytes();
        let view = SimdIndex2DView::from_bytes(&bytes).unwrap();
        for window in windows() {
            for stop_level in 0..4 {
                let expected = index.estimate_count(window, stop_level);
                assert_eq!(simd.estimate_count(window, stop_level), expected);
                assert_eq!(view.estimate_count(window, stop_level), expected);
            }
        }
    }
}
