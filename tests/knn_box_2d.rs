use std::ops::ControlFlow;

use packed_spatial_index::{Box2D, Index2D, Index2DBuilder, Index2DView, Point2D};
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

fn gap(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> f64 {
    if a_max < b_min {
        b_min - a_max
    } else if b_max < a_min {
        a_min - b_max
    } else {
        0.0
    }
}

fn box_distance_squared(a: Box2D, b: Box2D) -> f64 {
    let dx = gap(a.min_x, a.max_x, b.min_x, b.max_x);
    let dy = gap(a.min_y, a.max_y, b.min_y, b.max_y);
    dx * dx + dy * dy
}

/// Naive nondecreasing distances of all items within `max_distance` of `query`.
fn naive_distances(boxes: &[Box2D], query: Box2D, max_distance: f64) -> Vec<f64> {
    let mut distances: Vec<f64> = boxes
        .iter()
        .map(|&b| box_distance_squared(b, query))
        .filter(|&d| d <= max_distance * max_distance)
        .collect();
    distances.sort_by(|a, b| a.total_cmp(b));
    distances
}

fn result_distances(boxes: &[Box2D], query: Box2D, ids: &[usize]) -> Vec<f64> {
    ids.iter()
        .map(|&i| box_distance_squared(boxes[i], query))
        .collect()
}

#[test]
fn box_neighbors_match_naive_distances() {
    let mut rng = StdRng::seed_from_u64(808);
    let boxes = random_boxes(&mut rng, 600, 100.0, 4.0);
    let index = build(&boxes);

    for k in [1usize, 2, 7, 50, 600, 1000] {
        let query = Box2D::new(40.0, 40.0, 45.0, 43.0);
        let ids = index.neighbors_of_box(query, k);
        assert_eq!(ids.len(), k.min(boxes.len()));

        let expected = naive_distances(&boxes, query, f64::INFINITY);
        let actual = result_distances(&boxes, query, &ids);
        assert_eq!(actual, expected[..ids.len()], "k={k}");

        // Returned ids must be unique.
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate ids for k={k}");
    }
}

#[test]
fn box_neighbors_within_respects_max_distance() {
    let mut rng = StdRng::seed_from_u64(909);
    let boxes = random_boxes(&mut rng, 400, 100.0, 3.0);
    let index = build(&boxes);
    let query = Box2D::new(10.0, 80.0, 12.0, 81.0);

    for max_distance in [0.0, 1.5, 6.0, 25.0] {
        let ids = index.neighbors_of_box_within(query, usize::MAX, max_distance);
        let expected = naive_distances(&boxes, query, max_distance);
        let actual = result_distances(&boxes, query, &ids);
        assert_eq!(actual, expected, "max_distance={max_distance}");
    }
}

#[test]
fn zero_size_query_box_matches_point_neighbors() {
    let mut rng = StdRng::seed_from_u64(1010);
    let boxes = random_boxes(&mut rng, 500, 100.0, 5.0);
    let index = build(&boxes);

    let point = Point2D::new(33.3, 66.6);
    let query = Box2D::from_point(point);
    let by_point = result_distances(&boxes, query, &index.neighbors(point, 25));
    let by_box = result_distances(&boxes, query, &index.neighbors_of_box(query, 25));
    assert_eq!(by_point, by_box);
}

#[test]
fn overlapping_items_come_first_with_zero_distance() {
    let index = build(&[
        Box2D::new(0.0, 0.0, 2.0, 2.0),
        Box2D::new(1.0, 1.0, 3.0, 3.0),
        Box2D::new(8.0, 8.0, 9.0, 9.0),
        Box2D::new(4.0, 0.0, 5.0, 1.0),
    ]);
    // Touching counts as distance 0.
    let query = Box2D::new(2.0, 0.0, 3.0, 1.0);
    let ids = index.neighbors_of_box(query, 3);
    let mut first_two: Vec<usize> = ids[..2].to_vec();
    first_two.sort_unstable();
    assert_eq!(first_two, vec![0, 1]);
    assert_eq!(ids[2], 3);
}

#[test]
fn visit_neighbors_of_box_is_nondecreasing_and_breaks() {
    let mut rng = StdRng::seed_from_u64(1111);
    let boxes = random_boxes(&mut rng, 300, 100.0, 4.0);
    let index = build(&boxes);
    let query = Box2D::new(50.0, 50.0, 52.0, 52.0);

    let mut last = 0.0f64;
    let mut seen = 0usize;
    let flow = index.visit_neighbors_of_box(query, f64::INFINITY, |_, dist| {
        assert!(dist >= last, "distances must be nondecreasing");
        last = dist;
        seen += 1;
        if seen == 40 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(flow, ControlFlow::Break(()));
    assert_eq!(seen, 40);
}

#[test]
fn view_box_neighbors_match_owned() {
    let mut rng = StdRng::seed_from_u64(1212);
    let boxes = random_boxes(&mut rng, 350, 100.0, 6.0);
    let index = build(&boxes);
    let bytes = index.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();

    let query = Box2D::new(20.0, 30.0, 24.0, 31.0);
    let owned = result_distances(&boxes, query, &index.neighbors_of_box(query, 30));
    let viewed = result_distances(&boxes, query, &view.neighbors_of_box(query, 30));
    assert_eq!(owned, viewed);
}

#[test]
fn empty_index_and_zero_results() {
    let index = build(&[]);
    assert!(
        index
            .neighbors_of_box(Box2D::new(0.0, 0.0, 1.0, 1.0), 5)
            .is_empty()
    );

    let index = build(&[Box2D::new(0.0, 0.0, 1.0, 1.0)]);
    assert!(
        index
            .neighbors_of_box(Box2D::new(2.0, 2.0, 3.0, 3.0), 0)
            .is_empty()
    );
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::{NeighborWorkspace, SimdIndex2DView};

    #[test]
    fn simd_box_neighbors_match_naive() {
        let mut rng = StdRng::seed_from_u64(1313);
        let boxes = random_boxes(&mut rng, 450, 100.0, 4.0);
        let mut builder = Index2DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| builder.add(b));
        let index = builder.finish_simd().unwrap();
        let bytes = index.to_bytes();
        let view = SimdIndex2DView::from_bytes(&bytes).unwrap();

        let query = Box2D::new(70.0, 10.0, 72.0, 14.0);
        let expected = naive_distances(&boxes, query, f64::INFINITY);
        for k in [1usize, 12, 200] {
            let owned = result_distances(&boxes, query, &index.neighbors_of_box(query, k));
            let viewed = result_distances(&boxes, query, &view.neighbors_of_box(query, k));
            assert_eq!(owned, expected[..k.min(boxes.len())], "k={k}");
            assert_eq!(viewed, expected[..k.min(boxes.len())], "k={k}");
        }

        let mut workspace = NeighborWorkspace::new();
        let with_ws = index
            .neighbors_of_box_with(query, 12, f64::INFINITY, &mut workspace)
            .to_vec();
        assert_eq!(
            result_distances(&boxes, query, &with_ws),
            expected[..12.min(boxes.len())]
        );
    }
}
