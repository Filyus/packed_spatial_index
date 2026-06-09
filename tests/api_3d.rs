use std::ops::ControlFlow;

#[cfg(feature = "bench-internals")]
use packed_spatial_index::benchmark_support::SortKey3DStrategy;
use packed_spatial_index::{
    BoundsError, Box3D, BuildError, DEFAULT_NODE_SIZE, Index3DBuilder, Index3DView,
    NeighborWorkspace, Point3D, SearchWorkspace, SortKey3D,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn bounds3d_helpers_use_inclusive_edges() {
    let outer = Box3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    let inner = Box3D::new(2.0, 2.0, 2.0, 4.0, 4.0, 4.0);
    let touching = Box3D::new(10.0, 10.0, 10.0, 12.0, 12.0, 12.0);
    let outside = Box3D::new(11.0, 11.0, 11.0, 12.0, 12.0, 12.0);

    assert!(outer.contains(inner));
    assert!(outer.overlaps(touching));
    assert!(!outer.overlaps(outside));
    assert!(outer.contains_point(Point3D::new(5.0, 5.0, 5.0)));
    assert!(outer.contains_point(Point3D::new(10.0, 10.0, 10.0)));
    assert!(!outer.contains_point(Point3D::new(10.1, 10.0, 10.0)));
}

#[test]
fn box3d_from_point_creates_zero_size_query_box() {
    let point = Point3D::new(2.0, 3.0, 4.0);
    assert_eq!(
        Box3D::from_point(point),
        Box3D::new(2.0, 3.0, 4.0, 2.0, 3.0, 4.0)
    );
}

#[test]
fn point3d_queries_find_containing_boxes() {
    let mut builder = Index3DBuilder::new(2);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0));
    builder.add(Box3D::new(2.0, 2.0, 2.0, 4.0, 4.0, 4.0));
    let index = builder.finish().unwrap();

    assert_eq!(
        index.search(Box3D::from_point(Point3D::new(1.0, 1.0, 1.0))),
        vec![0]
    );
    assert_eq!(
        index.search(Box3D::from_point(Point3D::new(2.0, 2.0, 2.0))),
        vec![0, 1]
    );
    assert!(
        index
            .search(Box3D::from_point(Point3D::new(5.0, 5.0, 5.0)))
            .is_empty()
    );
}

#[test]
fn bounds3d_try_new_validates_bounds() {
    assert_eq!(
        Box3D::try_new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Ok(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    );
    assert!(matches!(
        Box3D::try_new(2.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Err(BoundsError::InvalidBounds3D { .. })
    ));
    assert!(matches!(
        Box3D::try_new(0.0, 0.0, f64::NAN, 1.0, 1.0, 1.0),
        Err(BoundsError::InvalidBounds3D { .. })
    ));
}

#[test]
fn index3d_finish_reports_count_mismatch() {
    let mut builder = Index3DBuilder::new(2);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));

    assert!(matches!(
        builder.finish(),
        Err(BuildError::ItemCount {
            added: 1,
            expected: 2
        })
    ));
}

#[test]
fn index3d_huge_builder_count_does_not_panic_on_construction() {
    let builder = Index3DBuilder::new(usize::MAX);
    assert!(matches!(
        builder.finish(),
        Err(BuildError::ItemCount {
            added: 0,
            expected: usize::MAX
        })
    ));
}

#[test]
fn index3d_default_builder_uses_exported_node_size() {
    let mut builder = Index3DBuilder::new(DEFAULT_NODE_SIZE + 1);
    for i in 0..=DEFAULT_NODE_SIZE {
        let x = i as f64;
        builder.add(Box3D::new(x, x, x, x + 0.5, x + 0.5, x + 0.5));
    }
    let index = builder.finish().unwrap();
    assert_eq!(index.node_size(), DEFAULT_NODE_SIZE);
}

#[test]
fn index3d_empty_and_small_indexes_behave() {
    let empty = Index3DBuilder::new(0).finish().unwrap();
    assert_eq!(empty.num_items(), 0);
    assert_eq!(empty.node_size(), DEFAULT_NODE_SIZE);
    assert_eq!(empty.extent(), None);
    assert!(
        empty
            .search(Box3D::new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
            .is_empty()
    );
    assert!(!empty.any(Box3D::new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)));
    assert_eq!(
        empty.first(Box3D::new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)),
        None
    );
    assert!(empty.neighbors(Point3D::new(0.0, 0.0, 0.0), 1).is_empty());

    let boxes = [
        Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0),
        Box3D::new(-1.0, -1.0, -1.0, 0.5, 0.5, 0.5),
    ];
    let mut builder = Index3DBuilder::new(boxes.len());
    for bounds in boxes {
        builder.add(bounds);
    }
    let index = builder.finish().unwrap();
    assert_eq!(
        index.extent(),
        Some(Box3D::new(-1.0, -1.0, -1.0, 6.0, 6.0, 6.0))
    );

    let mut hits = index.search(Box3D::new(-0.25, -0.25, -0.25, 2.0, 2.0, 2.0));
    hits.sort_unstable();
    assert_eq!(hits, vec![0, 2]);
}

#[test]
fn index3d_search_apis_agree() {
    let boxes = random_boxes_3d(257, 0x3D);
    let index = build_index_3d(&boxes, 16);
    let query = Box3D::new(250.0, 250.0, 250.0, 650.0, 650.0, 650.0);

    let mut expected = brute_force_search(&boxes, query);
    expected.sort_unstable();

    let mut search = index.search(query);
    search.sort_unstable();
    assert_eq!(search, expected);

    let mut into = Vec::new();
    index.search_into(query, &mut into);
    into.sort_unstable();
    assert_eq!(into, expected);

    let mut workspace = SearchWorkspace::with_capacity(16, 16);
    let mut with = index.search_with(query, &mut workspace).to_vec();
    with.sort_unstable();
    assert_eq!(with, expected);
    assert_eq!(workspace.results().len(), expected.len());

    assert_eq!(index.any(query), !expected.is_empty());
    assert_eq!(index.first(query).is_some(), !expected.is_empty());

    let mut visited = Vec::new();
    let flow: ControlFlow<()> = index.visit(query, |item| {
        visited.push(item);
        ControlFlow::Continue(())
    });
    assert_eq!(flow, ControlFlow::Continue(()));
    visited.sort_unstable();
    assert_eq!(visited, expected);

    let found = index.visit(query, ControlFlow::Break);
    assert_eq!(found.is_break(), !expected.is_empty());
}

#[test]
fn index3d_full_extent_search_returns_all_items() {
    let n = 128usize;
    let mut builder = Index3DBuilder::new(n);
    for i in 0..n {
        let x = (i % 8) as f64;
        let y = ((i / 8) % 8) as f64;
        let z = (i / 64) as f64;
        builder.add(Box3D::new(x, y, z, x + 0.25, y + 0.25, z + 0.25));
    }
    let index = builder.finish().unwrap();

    let mut hits = index.search(index.extent().unwrap());
    hits.sort_unstable();
    assert_eq!(hits, (0..n).collect::<Vec<_>>());
    assert!(index.any(index.extent().unwrap()));
    assert!(index.first(index.extent().unwrap()).is_some());

    let mut visited = Vec::new();
    let flow: ControlFlow<()> = index.visit(index.extent().unwrap(), |item| {
        visited.push(item);
        ControlFlow::Continue(())
    });
    assert!(flow.is_continue());
    visited.sort_unstable();
    assert_eq!(visited, (0..n).collect::<Vec<_>>());

    let bytes = index.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let mut view_hits = view.search(view.extent().unwrap());
    view_hits.sort_unstable();
    assert_eq!(view_hits, (0..n).collect::<Vec<_>>());
}

#[test]
fn index3d_search_matches_brute_force_on_random_boxes() {
    let boxes = random_boxes_3d(2_000, 0xB0B);
    let queries = random_queries_3d(128, 0xACE);

    for node_size in [8, 16, 32] {
        let index = build_index_3d(&boxes, node_size);
        for &query in &queries {
            let mut actual = index.search(query);
            let mut expected = brute_force_search(&boxes, query);
            actual.sort_unstable();
            expected.sort_unstable();
            assert_eq!(actual, expected, "node_size={node_size}");
        }
    }
}

#[cfg(feature = "bench-internals")]
#[test]
fn index3d_sort_key_strategies_match_brute_force_on_random_boxes() {
    let boxes = random_boxes_3d(2_000, 0xB0B);
    let queries = random_queries_3d(128, 0xACE);

    for node_size in [8, 16, 32] {
        for sort_key in [SortKey3DStrategy::Hilbert, SortKey3DStrategy::Morton] {
            let index = build_index_3d_impl(&boxes, node_size, sort_key);
            for &query in &queries {
                let mut actual = index.search(query);
                let mut expected = brute_force_search(&boxes, query);
                actual.sort_unstable();
                expected.sort_unstable();
                assert_eq!(
                    actual, expected,
                    "node_size={node_size}, sort_key={sort_key:?}"
                );
            }
        }
    }
}

#[test]
fn index3d_neighbors_match_brute_force() {
    let boxes = random_boxes_3d(2_000, 0x5151);
    let index = build_index_3d(&boxes, 16);
    let points = random_points_3d(96, 0xC0FFEE);

    for point in points {
        assert_eq!(
            index.neighbors(point, 1),
            brute_force_neighbors(&boxes, point, 1, f64::INFINITY)
        );
        assert_eq!(
            index.neighbors(point, 10),
            brute_force_neighbors(&boxes, point, 10, f64::INFINITY)
        );
        assert_eq!(
            index.neighbors_within(point, 10, 80.0),
            brute_force_neighbors(&boxes, point, 10, 80.0)
        );
    }

    assert!(index.neighbors(Point3D::new(0.0, 0.0, 0.0), 0).is_empty());
    assert!(
        index
            .neighbors_within(Point3D::new(0.0, 0.0, 0.0), 10, -1.0)
            .is_empty()
    );
    assert!(
        index
            .neighbors_within(Point3D::new(0.0, 0.0, 0.0), 10, f64::NAN)
            .is_empty()
    );
}

#[test]
fn index3d_neighbor_apis_agree_and_support_early_exit() {
    let boxes = [
        Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Box3D::new(10.0, 10.0, 10.0, 11.0, 11.0, 11.0),
        Box3D::new(3.0, 3.0, 3.0, 4.0, 4.0, 4.0),
        Box3D::new(-5.0, -5.0, -5.0, -4.0, -4.0, -4.0),
    ];
    let index = build_index_3d(&boxes, 2);
    let point = Point3D::new(3.25, 3.25, 3.25);

    let expected = index.neighbors_within(point, 3, f64::INFINITY);
    let mut into = Vec::new();
    index.neighbors_into(point, 3, f64::INFINITY, &mut into);
    assert_eq!(into, expected);

    let mut workspace = NeighborWorkspace::with_capacity(4, 8);
    assert_eq!(
        index.neighbors_with(point, 3, f64::INFINITY, &mut workspace),
        expected.as_slice()
    );
    assert_eq!(workspace.results(), expected.as_slice());

    let mut visited = Vec::new();
    let flow: ControlFlow<()> = index.visit_neighbors(point, f64::INFINITY, |item, dist| {
        visited.push((item, dist));
        if visited.len() == 3 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert!(flow.is_break());
    assert_eq!(
        visited.iter().map(|&(item, _)| item).collect::<Vec<_>>(),
        expected
    );
    assert!(visited.windows(2).all(|pair| pair[0].1 <= pair[1].1));
}

#[cfg(feature = "parallel")]
#[test]
fn index3d_parallel_build_matches_serial() {
    let boxes = random_boxes_3d(10_000, 0xABA);
    let queries = random_queries_3d(64, 0xBAB);

    let mut serial = Index3DBuilder::new(boxes.len())
        .node_size(16)
        .parallel(false);
    let mut parallel = Index3DBuilder::new(boxes.len())
        .node_size(16)
        .parallel(true)
        .parallel_min_items(0);
    for &bounds in &boxes {
        serial.add(bounds);
        parallel.add(bounds);
    }
    let serial = serial.finish().unwrap();
    let parallel = parallel.finish().unwrap();

    for query in queries {
        let mut a = serial.search(query);
        let mut b = parallel.search(query);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

fn build_index_3d(boxes: &[Box3D], node_size: usize) -> packed_spatial_index::Index3D {
    let mut builder = Index3DBuilder::new(boxes.len())
        .node_size(node_size)
        .sort_key(SortKey3D::Hilbert);
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish().unwrap()
}

#[cfg(feature = "bench-internals")]
fn build_index_3d_impl(
    boxes: &[Box3D],
    node_size: usize,
    sort_key: SortKey3DStrategy,
) -> packed_spatial_index::Index3D {
    let mut builder = Index3DBuilder::new(boxes.len())
        .node_size(node_size)
        .sort_key_strategy(sort_key);
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish().unwrap()
}

fn random_boxes_3d(n: usize, seed: u64) -> Vec<Box3D> {
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

fn random_queries_3d(n: usize, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let z: f64 = rng.random_range(0.0..1_000.0);
            let dx: f64 = rng.random_range(10.0..120.0);
            let dy: f64 = rng.random_range(10.0..120.0);
            let dz: f64 = rng.random_range(10.0..120.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn random_points_3d(n: usize, seed: u64) -> Vec<Point3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Point3D::new(
                rng.random_range(0.0..1_000.0),
                rng.random_range(0.0..1_000.0),
                rng.random_range(0.0..1_000.0),
            )
        })
        .collect()
}

fn brute_force_search(items: &[Box3D], query: Box3D) -> Vec<usize> {
    items
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, bounds)| bounds.overlaps(query).then_some(index))
        .collect()
}

fn brute_force_neighbors(
    items: &[Box3D],
    point: Point3D,
    max_results: usize,
    max_distance: f64,
) -> Vec<usize> {
    if max_results == 0 || max_distance.is_nan() || max_distance.is_sign_negative() {
        return Vec::new();
    }
    let max_distance_squared = max_distance * max_distance;
    let mut pairs: Vec<(usize, f64)> = items
        .iter()
        .copied()
        .enumerate()
        .map(|(index, bounds)| (index, distance_squared_to(bounds, point)))
        .filter(|&(_, distance_squared)| distance_squared <= max_distance_squared)
        .collect();
    pairs.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    pairs
        .into_iter()
        .take(max_results)
        .map(|(index, _)| index)
        .collect()
}

fn distance_squared_to(bounds: Box3D, point: Point3D) -> f64 {
    let dx = axis_distance(point.x, bounds.min_x, bounds.max_x);
    let dy = axis_distance(point.y, bounds.min_y, bounds.max_y);
    let dz = axis_distance(point.z, bounds.min_z, bounds.max_z);
    dx * dx + dy * dy + dz * dz
}

fn axis_distance(point: f64, min: f64, max: f64) -> f64 {
    if point < min {
        min - point
    } else if point > max {
        point - max
    } else {
        0.0
    }
}
