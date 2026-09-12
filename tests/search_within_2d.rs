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

fn naive_within(boxes: &[Box2D], query: Box2D, max_distance: f64) -> BTreeSet<usize> {
    (0..boxes.len())
        .filter(|&i| boxes[i].distance_to_box(query) <= max_distance)
        .collect()
}

fn as_set(ids: Vec<usize>) -> BTreeSet<usize> {
    let set: BTreeSet<_> = ids.iter().copied().collect();
    assert_eq!(set.len(), ids.len(), "duplicate ids reported");
    set
}

#[test]
fn search_within_matches_naive() {
    let mut rng = StdRng::seed_from_u64(2101);
    for (n, max_size, max_distance) in [
        (0, 4.0, 2.0),
        (1, 4.0, 2.0),
        (37, 8.0, 5.0),
        (700, 3.0, 1.0),
        (700, 3.0, 12.0),
    ] {
        let boxes = random_boxes(&mut rng, n, 100.0, max_size);
        let index = build(&boxes);
        for query in [
            Box2D::new(10.0, 10.0, 12.0, 12.0),
            Box2D::new(50.0, 50.0, 50.0, 50.0),
            Box2D::new(-20.0, -20.0, -19.0, -19.0),
            Box2D::new(0.0, 0.0, 100.0, 100.0),
        ] {
            let expected = naive_within(&boxes, query, max_distance);
            assert_eq!(
                as_set(index.search_within(query, max_distance)),
                expected,
                "n={n} eps={max_distance} query={query:?}"
            );
        }
    }
}

#[test]
fn max_distance_zero_reproduces_search() {
    let mut rng = StdRng::seed_from_u64(2102);
    let boxes = random_boxes(&mut rng, 500, 100.0, 5.0);
    let index = build(&boxes);
    for query in [
        Box2D::new(10.0, 10.0, 30.0, 30.0),
        Box2D::new(50.0, 50.0, 50.0, 50.0),
        Box2D::new(200.0, 200.0, 201.0, 201.0),
    ] {
        assert_eq!(
            as_set(index.search_within(query, 0.0)),
            as_set(index.search(query)),
            "query={query:?}"
        );
    }
}

#[test]
fn boundary_is_inclusive() {
    let index = build(&[
        Box2D::new(0.0, 0.0, 1.0, 1.0),
        Box2D::new(3.0, 0.0, 4.0, 1.0),
        Box2D::new(3.001, 0.0, 4.0, 1.0),
    ]);
    // Query touches item 1 at exactly 2.0 and item 2 at 2.001.
    let query = Box2D::new(0.5, 0.5, 1.0, 1.0);
    assert_eq!(as_set(index.search_within(query, 2.0)), as_set(vec![0, 1]));
    assert!(index.search_within_any(query, 2.0));
}

#[test]
fn degenerate_point_query() {
    let index = build(&[
        Box2D::new(0.0, 0.0, 1.0, 1.0),
        Box2D::new(5.0, 5.0, 6.0, 6.0),
    ]);
    let point = Box2D::new(2.0, 0.5, 2.0, 0.5);
    assert_eq!(as_set(index.search_within(point, 1.0)), as_set(vec![0]));
    assert_eq!(as_set(index.search_within(point, 0.5)), BTreeSet::new());
    // A point sitting inside an item box is at distance zero from it.
    let inside = Box2D::new(0.5, 0.5, 0.5, 0.5);
    assert_eq!(as_set(index.search_within(inside, 0.0)), as_set(vec![0]));
}

#[test]
fn negative_and_nan_max_distance_match_nothing() {
    let mut rng = StdRng::seed_from_u64(2103);
    let boxes = random_boxes(&mut rng, 200, 100.0, 5.0);
    let index = build(&boxes);
    let query = Box2D::new(10.0, 10.0, 20.0, 20.0);
    for max_distance in [-1.0, -0.0000001, f64::NAN] {
        assert!(
            index.search_within(query, max_distance).is_empty(),
            "{max_distance}"
        );
        assert!(
            !index.search_within_any(query, max_distance),
            "{max_distance}"
        );
    }
}

#[test]
fn into_visit_any_and_count_agree_with_search_within() {
    let mut rng = StdRng::seed_from_u64(2104);
    let boxes = random_boxes(&mut rng, 400, 100.0, 4.0);
    let index = build(&boxes);
    let mut buffer = vec![usize::MAX; 3];
    for max_distance in [0.0, 2.0, 30.0] {
        for query in [
            Box2D::new(10.0, 10.0, 12.0, 12.0),
            Box2D::new(300.0, 300.0, 301.0, 301.0),
        ] {
            let expected = index.search_within(query, max_distance);
            index.search_within_into(query, max_distance, &mut buffer);
            assert_eq!(buffer, expected, "eps={max_distance}");

            let mut visited = Vec::new();
            let _: ControlFlow<()> = index.search_within_each(query, max_distance, |i| {
                visited.push(i);
                ControlFlow::Continue(())
            });
            assert_eq!(visited, expected, "eps={max_distance}");

            assert_eq!(
                index.search_within_any(query, max_distance),
                !expected.is_empty()
            );
            assert_eq!(index.count_within(query, max_distance), expected.len());
        }
    }
}

#[test]
fn visit_within_stops_early() {
    let mut rng = StdRng::seed_from_u64(2105);
    let boxes = random_boxes(&mut rng, 300, 100.0, 4.0);
    let index = build(&boxes);
    let mut seen = 0usize;
    let flow = index.search_within_each(Box2D::new(0.0, 0.0, 100.0, 100.0), 5.0, |i| {
        seen += 1;
        ControlFlow::Break(i)
    });
    assert!(flow.is_break());
    assert_eq!(seen, 1);
}

#[test]
fn empty_index_matches_nothing() {
    let index = build(&[]);
    assert!(
        index
            .search_within(Box2D::new(0.0, 0.0, 1.0, 1.0), 10.0)
            .is_empty()
    );
    assert!(!index.search_within_any(Box2D::new(0.0, 0.0, 1.0, 1.0), 10.0));
    assert_eq!(index.count_within(Box2D::new(0.0, 0.0, 1.0, 1.0), 10.0), 0);
}

#[test]
fn view_matches_owned() {
    let mut rng = StdRng::seed_from_u64(2106);
    let boxes = random_boxes(&mut rng, 350, 100.0, 6.0);
    let index = build(&boxes);
    let bytes = index.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    for max_distance in [0.0, 2.5, 20.0] {
        let query = Box2D::new(20.0, 20.0, 25.0, 25.0);
        assert_eq!(
            as_set(view.search_within(query, max_distance)),
            as_set(index.search_within(query, max_distance)),
            "eps={max_distance}"
        );
        assert_eq!(
            view.search_within_any(query, max_distance),
            index.search_within_any(query, max_distance)
        );
        assert_eq!(
            view.count_within(query, max_distance),
            index.count_within(query, max_distance)
        );
    }
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
    fn simd_matches_naive_and_view_matches_owned() {
        let mut rng = StdRng::seed_from_u64(2107);
        let boxes = random_boxes(&mut rng, 600, 100.0, 4.0);
        let index = build_simd(&boxes);
        let bytes = index.to_bytes();
        let view = SimdIndex2DView::from_bytes(&bytes).unwrap();

        for max_distance in [0.0, 1.5, 15.0] {
            for query in [
                Box2D::new(30.0, 30.0, 33.0, 33.0),
                Box2D::new(70.0, 70.0, 70.0, 70.0),
            ] {
                let expected = naive_within(&boxes, query, max_distance);
                assert_eq!(
                    as_set(index.search_within(query, max_distance)),
                    expected,
                    "eps={max_distance}"
                );
                assert_eq!(
                    as_set(view.search_within(query, max_distance)),
                    expected,
                    "eps={max_distance}"
                );
                assert_eq!(
                    index.search_within_any(query, max_distance),
                    !expected.is_empty()
                );
                assert_eq!(
                    view.search_within_any(query, max_distance),
                    !expected.is_empty()
                );
                assert_eq!(index.count_within(query, max_distance), expected.len());
                assert_eq!(view.count_within(query, max_distance), expected.len());
            }
        }
    }
}

/// The collect forms switch traversal by estimated selectivity, and the two
/// traversals must answer identically. The sweep runs the radius from "hits
/// almost nothing" to "covers the whole extent" so it crosses the threshold in
/// both directions, and checks every collect form against both the callback
/// form (which never switches) and the brute-force oracle.
#[test]
fn collect_forms_agree_across_the_selectivity_switch() {
    let mut rng = StdRng::seed_from_u64(0x5717C4);
    let boxes = random_boxes(&mut rng, 2_000, 1_000.0, 5.0);
    let index = build(&boxes);
    let bytes = index.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    let mut buffer = Vec::new();

    for query in [
        Box2D::new(500.0, 500.0, 501.0, 501.0),
        Box2D::new(0.0, 0.0, 1.0, 1.0),
        Box2D::new(-50.0, -50.0, -49.0, -49.0),
    ] {
        for max_distance in [
            0.0, 0.1, 1.0, 5.0, 12.0, 25.0, 60.0, 150.0, 400.0, 900.0, 2_000.0,
        ] {
            let expected = naive_within(&boxes, query, max_distance);
            let label = format!("query={query:?} max_distance={max_distance}");

            let mut each = Vec::new();
            let _: ControlFlow<()> = index.search_within_each(query, max_distance, |i| {
                each.push(i);
                ControlFlow::Continue(())
            });
            assert_eq!(as_set(each), expected, "search_within_each: {label}");

            assert_eq!(
                as_set(index.search_within(query, max_distance)),
                expected,
                "search_within: {label}"
            );
            index.search_within_into(query, max_distance, &mut buffer);
            assert_eq!(
                as_set(buffer.clone()),
                expected,
                "search_within_into: {label}"
            );
            assert_eq!(
                index.count_within(query, max_distance),
                expected.len(),
                "count_within: {label}"
            );
            assert_eq!(
                as_set(view.search_within(query, max_distance)),
                expected,
                "view search_within: {label}"
            );
            assert_eq!(
                view.count_within(query, max_distance),
                expected.len(),
                "view count_within: {label}"
            );
        }
    }
}
