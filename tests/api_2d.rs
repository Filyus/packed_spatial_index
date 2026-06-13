#[cfg(feature = "parallel")]
use packed_spatial_index::DEFAULT_PARALLEL_MIN_ITEMS;
use packed_spatial_index::{
    BoundsError, Box2D, BuildError, DEFAULT_NODE_SIZE, Index2DBuilder, Index2DView, SearchWorkspace,
};
#[cfg(feature = "parallel")]
use rand::rngs::StdRng;
#[cfg(feature = "parallel")]
use rand::{RngExt, SeedableRng};
use static_aabb2d_index::StaticAABB2DIndexBuilder;
use std::ops::ControlFlow;

#[test]
fn bounds_helpers_use_inclusive_edges() {
    let outer = Box2D::new(0.0, 0.0, 10.0, 10.0);
    let inner = Box2D::new(2.0, 2.0, 4.0, 4.0);
    let touching = Box2D::new(10.0, 10.0, 12.0, 12.0);
    let outside = Box2D::new(11.0, 11.0, 12.0, 12.0);

    assert!(outer.overlaps(inner));
    assert!(outer.overlaps(touching));
    assert!(!outer.overlaps(outside));

    assert!(outer.contains(inner));
    assert!(outer.contains(outer));
    assert!(!outer.contains(touching));
    assert!(outer.contains_point(packed_spatial_index::Point2D::new(10.0, 10.0)));
    assert!(!outer.contains_point(packed_spatial_index::Point2D::new(10.1, 10.0)));
}

#[test]
fn box_from_point_creates_zero_size_query_box() {
    let point = packed_spatial_index::Point2D::new(2.0, 3.0);
    assert_eq!(Box2D::from_point(point), Box2D::new(2.0, 3.0, 2.0, 3.0));
}

#[test]
fn point_queries_find_containing_boxes() {
    let mut builder = Index2DBuilder::new(3);
    builder.add(Box2D::new(0.0, 0.0, 2.0, 2.0));
    builder.add(Box2D::new(2.0, 2.0, 4.0, 4.0));
    builder.add(Box2D::new(10.0, 10.0, 11.0, 11.0));
    let index = builder.finish().unwrap();

    assert_eq!(
        index.search(Box2D::from_point(packed_spatial_index::Point2D::new(
            1.0, 1.0
        ))),
        vec![0]
    );
    assert_eq!(
        index.search(Box2D::from_point(packed_spatial_index::Point2D::new(
            2.0, 2.0
        ))),
        vec![0, 1]
    );
    assert!(
        index
            .search(Box2D::from_point(packed_spatial_index::Point2D::new(
                9.0, 9.0
            )))
            .is_empty()
    );
}

#[test]
fn bounds_try_new_validates_bounds() {
    assert_eq!(
        Box2D::try_new(0.0, 0.0, 1.0, 1.0),
        Ok(Box2D::new(0.0, 0.0, 1.0, 1.0))
    );

    assert!(matches!(
        Box2D::try_new(2.0, 0.0, 1.0, 1.0),
        Err(BoundsError::InvalidBounds { .. })
    ));
    assert!(matches!(
        Box2D::try_new(0.0, f64::NAN, 1.0, 1.0),
        Err(BoundsError::InvalidBounds { .. })
    ));
}

#[test]
fn default_builder_uses_exported_node_size() {
    let mut builder = Index2DBuilder::new(17);
    for i in 0..17 {
        builder.add(Box2D::new(i as f64, 0.0, i as f64 + 0.5, 1.0));
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
fn empty_and_small_indexes_behave_like_reference() {
    let reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(0, 16)
        .build()
        .unwrap();
    let empty = Index2DBuilder::new(0).finish().unwrap();
    assert_eq!(empty.num_items(), 0);
    assert_eq!(empty.extent(), None);
    assert_eq!(
        empty.search(Box2D::new(-1.0, -1.0, 1.0, 1.0)),
        reference.query(-1.0, -1.0, 1.0, 1.0)
    );

    let boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [2.0, 2.0, 3.0, 3.0],
        [-1.0, -1.0, 0.5, 0.5],
    ];
    let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(boxes.len(), 16);
    let mut index = Index2DBuilder::new(boxes.len());
    for b in boxes {
        reference.add(b[0], b[1], b[2], b[3]);
        index.add(Box2D::new(b[0], b[1], b[2], b[3]));
    }
    let reference = reference.build().unwrap();
    let index = index.finish().unwrap();

    assert_eq!(index.extent(), Some(Box2D::new(-1.0, -1.0, 3.0, 3.0)));

    let query = Box2D::new(-0.25, -0.25, 2.25, 2.25);
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
    let mut index = Index2DBuilder::new(boxes.len());
    for b in boxes {
        reference.add(b[0], b[1], b[2], b[3]);
        index.add(Box2D::new(b[0], b[1], b[2], b[3]));
    }
    let reference = reference.build().unwrap();
    let index = index.finish().unwrap();

    let query = Box2D::new(9.0, 9.0, 11.0, 11.0);
    let mut expected = reference.query(query.min_x, query.min_y, query.max_x, query.max_y);
    let mut actual = index.search(query);
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(expected, actual);
}

#[test]
fn search_apis_agree() {
    let mut builder = Index2DBuilder::new(3);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    builder.add(Box2D::new(0.5, 0.5, 2.0, 2.0));
    let index = builder.finish().unwrap();

    let query = Box2D::new(0.0, 0.0, 2.0, 2.0);
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
    assert!(!index.any(Box2D::new(10.0, 10.0, 11.0, 11.0)));
    assert!(matches!(index.first(query), Some(0 | 2)));
    assert_eq!(index.first(Box2D::new(10.0, 10.0, 11.0, 11.0)), None);

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
    let mut builder = Index2DBuilder::new(2);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    let index = builder.finish().unwrap();

    let mut out = vec![usize::MAX];
    let mut stack = vec![usize::MAX, usize::MAX];
    index.search_into_stack(Box2D::new(10.0, 10.0, 11.0, 11.0), &mut out, &mut stack);
    assert!(out.is_empty());
    assert!(stack.is_empty());

    index.search_into_stack_prefetch(Box2D::new(0.0, 0.0, 2.0, 2.0), &mut out, &mut stack);
    assert_eq!(out, vec![0]);
}

#[test]
fn full_extent_search_returns_all_items() {
    let n = 128usize;
    let mut builder = Index2DBuilder::new(n);
    for i in 0..n {
        let x = (i % 16) as f64;
        let y = (i / 16) as f64;
        builder.add(Box2D::new(x, y, x + 0.25, y + 0.25));
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
    let view = Index2DView::from_bytes(&bytes).unwrap();
    let mut view_hits = view.search(view.extent().unwrap());
    view_hits.sort_unstable();
    assert_eq!(view_hits, (0..n).collect::<Vec<_>>());
}

#[test]
fn finish_reports_count_mismatch() {
    let mut builder = Index2DBuilder::new(2);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));

    assert!(matches!(
        builder.finish(),
        Err(BuildError::ItemCount {
            added: 1,
            expected: 2
        })
    ));
}

#[test]
fn huge_builder_count_does_not_panic_on_construction() {
    let builder = Index2DBuilder::new(usize::MAX);
    assert!(matches!(builder.finish(), Err(BuildError::TreeTooLarge)));
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

    let mut serial = Index2DBuilder::new(n);
    let mut parallel = Index2DBuilder::new(n).parallel(true).parallel_min_items(0);
    for b in &boxes {
        serial.add(Box2D::new(b[0], b[1], b[2], b[3]));
        parallel.add(Box2D::new(b[0], b[1], b[2], b[3]));
    }
    let serial = serial.finish().unwrap();
    let parallel = parallel.finish().unwrap();

    for _ in 0..200 {
        let qx: f64 = rng.random_range(0.0..10_000.0);
        let qy: f64 = rng.random_range(0.0..10_000.0);
        let query = Box2D::new(qx, qy, qx + 150.0, qy + 150.0);
        let mut a = serial.search(query);
        let mut b = parallel.search(query);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

// A deterministic 20x20 grid of unit boxes, so search_iter tests run without
// the rng helpers (which are gated behind the `parallel` feature here).
fn grid_index() -> packed_spatial_index::Index2D {
    let mut builder = Index2DBuilder::new(400);
    for gy in 0..20 {
        for gx in 0..20 {
            let (x, y) = (gx as f64, gy as f64);
            builder.add(Box2D::new(x, y, x + 0.5, y + 0.5));
        }
    }
    builder.finish().unwrap()
}

#[test]
fn search_iter_matches_search() {
    let index = grid_index();
    for &q in &[
        Box2D::new(0.0, 0.0, 4.5, 4.5),
        Box2D::new(5.0, 5.0, 5.5, 5.5),
        Box2D::new(-10.0, -10.0, 100.0, 100.0),
        Box2D::new(100.0, 100.0, 200.0, 200.0),
        Box2D::new(3.2, 7.1, 9.9, 12.4),
    ] {
        let mut from_iter: Vec<usize> = index.search_iter(q).collect();
        let mut from_search = index.search(q);
        from_iter.sort_unstable();
        from_search.sort_unstable();
        assert_eq!(from_iter, from_search, "query {q:?}");
    }
}

#[test]
fn search_iter_partial_consumption_is_lazy_and_valid() {
    let index = grid_index();
    let q = Box2D::new(0.0, 0.0, 9.5, 9.5); // 100 boxes overlap
    let total = index.search(q).len();
    assert_eq!(total, 100);

    // Every yielded item is a genuine hit, and `take(k)` yields exactly k.
    let prefix: Vec<usize> = index.search_iter(q).take(7).collect();
    assert_eq!(prefix.len(), 7);
    let full: std::collections::HashSet<usize> = index.search(q).into_iter().collect();
    assert!(prefix.iter().all(|i| full.contains(i)));

    // `find` short-circuits and returns a real hit.
    let found = index.search_iter(q).find(|&i| i % 5 == 0);
    assert!(found.is_some_and(|i| full.contains(&i)));
}

#[test]
fn search_iter_empty_and_no_match() {
    let empty = Index2DBuilder::new(0).finish().unwrap();
    assert_eq!(empty.search_iter(Box2D::new(0.0, 0.0, 1.0, 1.0)).count(), 0);

    let index = grid_index();
    let miss = Box2D::new(1000.0, 1000.0, 1001.0, 1001.0);
    assert_eq!(index.search_iter(miss).next(), None);
}

#[test]
fn search_iter_size_hint_bounds_results() {
    let index = grid_index();
    let (lo, hi) = index
        .search_iter(Box2D::new(0.0, 0.0, 4.5, 4.5))
        .size_hint();
    assert_eq!(lo, 0);
    assert_eq!(hi, Some(index.num_items()));
}
