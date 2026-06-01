mod common;

use common::{brute_force_neighbors, build_index, random_boxes};
use packed_spatial_index::{NeighborWorkspace, Point};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::ops::ControlFlow;

#[test]
fn neighbors_match_brute_force() {
    let mut rng = StdRng::seed_from_u64(0xAABB);
    let boxes = random_boxes(&mut rng, 1_000);
    let index = build_index(&boxes, 16);

    for _ in 0..200 {
        let point = Point::new(rng.random_range(0.0..1000.0), rng.random_range(0.0..1000.0));
        for &(limit, max_distance) in &[
            (0, f64::INFINITY),
            (1, f64::INFINITY),
            (8, 80.0),
            (32, 250.0),
        ] {
            assert_eq!(
                index.neighbors_within(point, limit, max_distance),
                brute_force_neighbors(&boxes, point, limit, max_distance)
            );
        }
    }
}

#[test]
fn neighbor_apis_agree_and_support_early_exit() {
    let boxes = [
        [0.0, 0.0, 2.0, 2.0],
        [5.0, 0.0, 6.0, 1.0],
        [10.0, 0.0, 11.0, 1.0],
        [-5.0, 0.0, -4.0, 1.0],
    ];
    let index = build_index(&boxes, 2);
    let point = Point::new(1.0, 1.0);
    let expected = brute_force_neighbors(&boxes, point, 3, f64::INFINITY);
    assert_eq!(expected[0], 0);
    assert_eq!(index.neighbors(point, 3), expected);
    assert_eq!(index.neighbors_within(point, 4, 3.9), vec![0]);
    assert!(index.neighbors(point, 0).is_empty());
    assert!(index.neighbors_within(point, 3, -1.0).is_empty());

    let mut out = vec![usize::MAX];
    index.neighbors_into(point, 3, f64::INFINITY, &mut out);
    assert_eq!(out, expected);

    let mut workspace = NeighborWorkspace::with_capacity(8, 8);
    assert_eq!(
        index.neighbors_with(point, 3, f64::INFINITY, &mut workspace),
        expected.as_slice()
    );
    assert_eq!(workspace.results(), expected.as_slice());

    let mut visited = Vec::new();
    let completed: ControlFlow<()> = index.visit_neighbors(point, f64::INFINITY, |idx, dist| {
        visited.push((idx, dist));
        ControlFlow::Continue(())
    });
    assert!(completed.is_continue());
    assert_eq!(
        visited
            .iter()
            .map(|&(idx, _)| idx)
            .take(3)
            .collect::<Vec<_>>(),
        expected
    );
    assert!(visited.windows(2).all(|pair| pair[0].1 <= pair[1].1));

    let mut calls = 0usize;
    let stopped: ControlFlow<usize> = index.visit_neighbors(point, f64::INFINITY, |idx, _| {
        calls += 1;
        ControlFlow::Break(idx)
    });
    assert_eq!(calls, 1);
    assert_eq!(stopped, ControlFlow::Break(0));
}
