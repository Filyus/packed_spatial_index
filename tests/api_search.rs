mod common;

use common::rect;
#[cfg(feature = "parallel")]
use packed_spatial_index::DEFAULT_PARALLEL_MIN_ITEMS;
#[cfg(feature = "parallel")]
use packed_spatial_index::experimental::ExperimentalSortKey;
use packed_spatial_index::{BuildError, DEFAULT_NODE_SIZE, IndexBuilder, Rect, SearchWorkspace};
#[cfg(feature = "parallel")]
use rand::rngs::StdRng;
#[cfg(feature = "parallel")]
use rand::{RngExt, SeedableRng};
use static_aabb2d_index::StaticAABB2DIndexBuilder;
use std::ops::ControlFlow;

#[test]
fn rect_helpers_use_inclusive_edges() {
    let outer = Rect::new(0.0, 0.0, 10.0, 10.0);
    let inner = Rect::new(2.0, 2.0, 4.0, 4.0);
    let touching = Rect::new(10.0, 10.0, 12.0, 12.0);
    let outside = Rect::new(11.0, 11.0, 12.0, 12.0);

    assert!(outer.overlaps(inner));
    assert!(outer.overlaps(touching));
    assert!(!outer.overlaps(outside));

    assert!(outer.contains(inner));
    assert!(outer.contains(outer));
    assert!(!outer.contains(touching));
    assert!(outer.contains_point(packed_spatial_index::Point::new(10.0, 10.0)));
    assert!(!outer.contains_point(packed_spatial_index::Point::new(10.1, 10.0)));
}

#[test]
fn default_builder_uses_exported_node_size() {
    let mut builder = IndexBuilder::new(17);
    for i in 0..17 {
        builder.add_bounds(i as f64, 0.0, i as f64 + 0.5, 1.0);
    }
    let index = builder.finish().unwrap();
    assert_eq!(index.node_size(), DEFAULT_NODE_SIZE);
}

#[cfg(feature = "parallel")]
#[test]
fn default_parallel_threshold_is_exported() {
    assert_eq!(DEFAULT_PARALLEL_MIN_ITEMS, 50_000);
}

#[test]
fn add_rect_and_add_bounds_produce_identical_results() {
    let boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [2.0, 2.0, 3.0, 3.0],
        [-1.0, -1.0, 0.5, 0.5],
    ];
    let mut by_rect = IndexBuilder::new(boxes.len());
    let mut by_bounds = IndexBuilder::new(boxes.len());
    for b in boxes {
        by_rect.add(rect(b));
        by_bounds.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let by_rect = by_rect.finish().unwrap();
    let by_bounds = by_bounds.finish().unwrap();

    let query = Rect::new(-0.25, -0.25, 2.25, 2.25);
    let mut a = by_rect.search(query);
    let mut b = by_bounds.search(query);
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}

#[test]
fn empty_and_small_indexes_behave_like_reference() {
    let reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(0, 16)
        .build()
        .unwrap();
    let empty = IndexBuilder::new(0).finish().unwrap();
    assert_eq!(empty.num_items(), 0);
    assert_eq!(empty.bounds(), None);
    assert_eq!(
        empty.search(Rect::new(-1.0, -1.0, 1.0, 1.0)),
        reference.query(-1.0, -1.0, 1.0, 1.0)
    );

    let boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [2.0, 2.0, 3.0, 3.0],
        [-1.0, -1.0, 0.5, 0.5],
    ];
    let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(boxes.len(), 16);
    let mut index = IndexBuilder::new(boxes.len());
    for b in boxes {
        reference.add(b[0], b[1], b[2], b[3]);
        index.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let reference = reference.build().unwrap();
    let index = index.finish().unwrap();

    assert_eq!(index.bounds(), Some(Rect::new(-1.0, -1.0, 3.0, 3.0)));

    let query = Rect::new(-0.25, -0.25, 2.25, 2.25);
    let mut expected = reference.query(query.min_x, query.min_y, query.max_x, query.max_y);
    let mut actual = index.search(query);
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(expected, actual);
}

#[test]
fn degenerate_extent_matches_reference() {
    let boxes = [
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
    ];
    let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(boxes.len(), 16);
    let mut index = IndexBuilder::new(boxes.len());
    for b in boxes {
        reference.add(b[0], b[1], b[2], b[3]);
        index.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let reference = reference.build().unwrap();
    let index = index.finish().unwrap();

    let query = Rect::new(9.0, 9.0, 11.0, 11.0);
    let mut expected = reference.query(query.min_x, query.min_y, query.max_x, query.max_y);
    let mut actual = index.search(query);
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(expected, actual);
}

#[test]
fn search_apis_agree() {
    let mut builder = IndexBuilder::new(3);
    builder.add_bounds(0.0, 0.0, 1.0, 1.0);
    builder.add_bounds(5.0, 5.0, 6.0, 6.0);
    builder.add(Rect::new(0.5, 0.5, 2.0, 2.0));
    let index = builder.finish().unwrap();

    let query = Rect::new(0.0, 0.0, 2.0, 2.0);
    let mut expected = index.search(query);
    expected.sort_unstable();

    let mut out = vec![usize::MAX];
    index.search_into(query, &mut out);
    out.sort_unstable();
    assert_eq!(expected, out);

    let mut workspace = SearchWorkspace::with_capacity(8, 8);
    let mut with = index.search_with(query, &mut workspace).to_vec();
    with.sort_unstable();
    assert_eq!(expected, with);
    assert_eq!(workspace.results().len(), 2);

    assert!(index.any(query));
    assert!(!index.any(Rect::new(10.0, 10.0, 11.0, 11.0)));
    assert!(matches!(index.first(query), Some(0 | 2)));
    assert_eq!(index.first(Rect::new(10.0, 10.0, 11.0, 11.0)), None);

    let mut visited = Vec::new();
    let completed: ControlFlow<()> = index.visit(query, |idx| {
        visited.push(idx);
        ControlFlow::Continue(())
    });
    assert!(completed.is_continue());
    visited.sort_unstable();
    assert_eq!(expected, visited);

    let mut calls = 0usize;
    let stopped: ControlFlow<usize> = index.visit(query, |idx| {
        calls += 1;
        ControlFlow::Break(idx)
    });
    assert_eq!(calls, 1);
    assert!(matches!(stopped, ControlFlow::Break(0 | 2)));
}

#[test]
fn hidden_stack_paths_reuse_and_clear_buffers() {
    let mut builder = IndexBuilder::new(2);
    builder.add_bounds(0.0, 0.0, 1.0, 1.0);
    builder.add_bounds(5.0, 5.0, 6.0, 6.0);
    let index = builder.finish().unwrap();

    let mut out = vec![usize::MAX];
    let mut stack = vec![usize::MAX, usize::MAX];
    index.search_into_stack(Rect::new(10.0, 10.0, 11.0, 11.0), &mut out, &mut stack);
    assert!(out.is_empty());
    assert!(stack.is_empty());

    index.search_into_stack_prefetch(Rect::new(0.0, 0.0, 2.0, 2.0), &mut out, &mut stack);
    assert_eq!(out, vec![0]);
}

#[test]
fn finish_reports_count_mismatch() {
    let mut builder = IndexBuilder::new(2);
    builder.add_bounds(0.0, 0.0, 1.0, 1.0);

    assert!(matches!(
        builder.finish(),
        Err(BuildError::ItemCount {
            added: 1,
            expected: 2
        })
    ));
}

#[test]
#[cfg(feature = "parallel")]
fn parallel_build_matches_serial() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = 20_000usize;
    let mut boxes = Vec::with_capacity(n);
    for _ in 0..n {
        let cx: f64 = rng.random_range(0.0..10_000.0);
        let cy: f64 = rng.random_range(0.0..10_000.0);
        boxes.push([cx, cy, cx + 10.0, cy + 10.0]);
    }

    let mut serial = IndexBuilder::new(n).experimental_sort_key(ExperimentalSortKey::HilbertLut);
    let mut parallel = IndexBuilder::new(n)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut)
        .parallel(true)
        .parallel_min_items(0);
    for b in &boxes {
        serial.add_bounds(b[0], b[1], b[2], b[3]);
        parallel.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let serial = serial.finish().unwrap();
    let parallel = parallel.finish().unwrap();

    for _ in 0..200 {
        let qx: f64 = rng.random_range(0.0..10_000.0);
        let qy: f64 = rng.random_range(0.0..10_000.0);
        let query = Rect::new(qx, qy, qx + 150.0, qy + 150.0);
        let mut a = serial.search(query);
        let mut b = parallel.search(query);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}
