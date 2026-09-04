use std::collections::BTreeSet;
use std::ops::ControlFlow;

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

fn naive_within(boxes: &[Box3D], query: Box3D, max_distance: f64) -> BTreeSet<usize> {
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
    let mut rng = StdRng::seed_from_u64(3101);
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
            Box3D::new(10.0, 10.0, 10.0, 12.0, 12.0, 12.0),
            Box3D::new(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
            Box3D::new(-20.0, -20.0, -20.0, -19.0, -19.0, -19.0),
            Box3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0),
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
fn epsilon_zero_reproduces_search() {
    let mut rng = StdRng::seed_from_u64(3102);
    let boxes = random_boxes(&mut rng, 500, 100.0, 5.0);
    let index = build(&boxes);
    for query in [
        Box3D::new(10.0, 10.0, 10.0, 30.0, 30.0, 30.0),
        Box3D::new(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
        Box3D::new(200.0, 200.0, 200.0, 201.0, 201.0, 201.0),
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
        Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Box3D::new(3.0, 0.0, 0.0, 4.0, 1.0, 1.0),
        Box3D::new(3.001, 0.0, 0.0, 4.0, 1.0, 1.0),
    ]);
    // Query touches item 1 at exactly 2.0 and item 2 at 2.001.
    let query = Box3D::new(0.5, 0.5, 0.5, 1.0, 1.0, 1.0);
    assert_eq!(as_set(index.search_within(query, 2.0)), as_set(vec![0, 1]));
    assert!(index.any_within(query, 2.0));
}

#[test]
fn degenerate_point_query() {
    let index = build(&[
        Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0),
    ]);
    let point = Box3D::new(2.0, 0.5, 0.5, 2.0, 0.5, 0.5);
    assert_eq!(as_set(index.search_within(point, 1.0)), as_set(vec![0]));
    assert_eq!(as_set(index.search_within(point, 0.5)), BTreeSet::new());
    // A point sitting inside an item box is at distance zero from it.
    let inside = Box3D::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
    assert_eq!(as_set(index.search_within(inside, 0.0)), as_set(vec![0]));
}

#[test]
fn negative_and_nan_epsilon_match_nothing() {
    let mut rng = StdRng::seed_from_u64(3103);
    let boxes = random_boxes(&mut rng, 200, 100.0, 5.0);
    let index = build(&boxes);
    let query = Box3D::new(10.0, 10.0, 10.0, 20.0, 20.0, 20.0);
    for max_distance in [-1.0, -0.0000001, f64::NAN] {
        assert!(
            index.search_within(query, max_distance).is_empty(),
            "{max_distance}"
        );
        assert!(!index.any_within(query, max_distance), "{max_distance}");
    }
}

#[test]
fn into_visit_and_any_agree_with_search_within() {
    let mut rng = StdRng::seed_from_u64(3104);
    let boxes = random_boxes(&mut rng, 400, 100.0, 4.0);
    let index = build(&boxes);
    let mut buffer = vec![usize::MAX; 3];
    for max_distance in [0.0, 2.0, 30.0] {
        for query in [
            Box3D::new(10.0, 10.0, 10.0, 12.0, 12.0, 12.0),
            Box3D::new(300.0, 300.0, 300.0, 301.0, 301.0, 301.0),
        ] {
            let expected = index.search_within(query, max_distance);
            index.search_within_into(query, max_distance, &mut buffer);
            assert_eq!(buffer, expected, "eps={max_distance}");

            let mut visited = Vec::new();
            let _: ControlFlow<()> = index.visit_within(query, max_distance, |i| {
                visited.push(i);
                ControlFlow::Continue(())
            });
            assert_eq!(visited, expected, "eps={max_distance}");

            assert_eq!(index.any_within(query, max_distance), !expected.is_empty());
        }
    }
}

#[test]
fn visit_within_stops_early() {
    let mut rng = StdRng::seed_from_u64(3105);
    let boxes = random_boxes(&mut rng, 300, 100.0, 4.0);
    let index = build(&boxes);
    let mut seen = 0usize;
    let flow = index.visit_within(Box3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0), 5.0, |i| {
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
            .search_within(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 10.0)
            .is_empty()
    );
    assert!(!index.any_within(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 10.0));
}

#[test]
fn view_matches_owned() {
    let mut rng = StdRng::seed_from_u64(3106);
    let boxes = random_boxes(&mut rng, 350, 100.0, 6.0);
    let index = build(&boxes);
    let bytes = index.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    for max_distance in [0.0, 2.5, 20.0] {
        let query = Box3D::new(20.0, 20.0, 20.0, 25.0, 25.0, 25.0);
        assert_eq!(
            as_set(view.search_within(query, max_distance)),
            as_set(index.search_within(query, max_distance)),
            "eps={max_distance}"
        );
        assert_eq!(
            view.any_within(query, max_distance),
            index.any_within(query, max_distance)
        );
    }
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::{SimdIndex3D, SimdIndex3DView};

    fn build_simd(boxes: &[Box3D]) -> SimdIndex3D {
        let mut builder = Index3DBuilder::new(boxes.len());
        for &b in boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    }

    #[test]
    fn simd_matches_naive_and_view_matches_owned() {
        let mut rng = StdRng::seed_from_u64(3107);
        let boxes = random_boxes(&mut rng, 600, 100.0, 4.0);
        let index = build_simd(&boxes);
        let bytes = index.to_bytes();
        let view = SimdIndex3DView::from_bytes(&bytes).unwrap();

        for max_distance in [0.0, 1.5, 15.0] {
            for query in [
                Box3D::new(30.0, 30.0, 30.0, 33.0, 33.0, 33.0),
                Box3D::new(70.0, 70.0, 70.0, 70.0, 70.0, 70.0),
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
                assert_eq!(index.any_within(query, max_distance), !expected.is_empty());
                assert_eq!(view.any_within(query, max_distance), !expected.is_empty());
            }
        }
    }
}
