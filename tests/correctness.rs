//! Correctness tests:
//!  1) Hilbert encoders match the reference crate bit-for-bit;
//!  2) the encoder is bijective on a dense subrange;
//!  3) `Index` and `SimdIndex` searches match the reference as sets.

use std::ops::ControlFlow;

use packed_spatial_index::experimental::{ExperimentalSortKey, ENCODERS};
use packed_spatial_index::{BuildError, IndexBuilder, Rect, SearchWorkspace, SortKey};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use static_aabb2d_index::{hilbert_xy_to_index, StaticAABB2DIndexBuilder};

fn rect(bounds: [f64; 4]) -> Rect {
    Rect::new(bounds[0], bounds[1], bounds[2], bounds[3])
}

fn random_boxes(rng: &mut StdRng, n: usize) -> Vec<[f64; 4]> {
    let mut boxes = Vec::with_capacity(n);
    for _ in 0..n {
        let cx: f64 = rng.gen_range(0.0..1000.0);
        let cy: f64 = rng.gen_range(0.0..1000.0);
        let w: f64 = rng.gen_range(0.1..10.0);
        let h: f64 = rng.gen_range(0.1..10.0);
        boxes.push([cx, cy, cx + w, cy + h]);
    }
    boxes
}

#[test]
fn encoders_match_reference() {
    let step = 257u32;
    for xv in (0..=u16::MAX as u32).step_by(step as usize) {
        for yv in (0..=u16::MAX as u32).step_by(step as usize) {
            let (x, y) = (xv as u16, yv as u16);
            let expected = hilbert_xy_to_index(x, y);
            for (name, f) in ENCODERS {
                assert_eq!(f(x, y), expected, "encoder `{name}` mismatch at ({x}, {y})");
            }
        }
    }
}

#[test]
fn encoders_match_reference_random() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    for _ in 0..200_000 {
        let x: u16 = rng.gen();
        let y: u16 = rng.gen();
        let expected = hilbert_xy_to_index(x, y);
        for (name, f) in ENCODERS {
            assert_eq!(f(x, y), expected, "encoder `{name}` mismatch at ({x}, {y})");
        }
    }
}

#[test]
fn encoder_is_bijection_on_8bit() {
    for (name, f) in ENCODERS {
        let mut seen = std::collections::HashSet::with_capacity(256 * 256);
        for x in 0..256u16 {
            for y in 0..256u16 {
                let v = f(x, y);
                assert!(
                    seen.insert(v),
                    "encoder `{name}` not injective at ({x},{y})"
                );
            }
        }
        assert_eq!(seen.len(), 256 * 256, "encoder `{name}` lost values");
    }
}

fn check_experimental_sort_key_matches_reference(choice: ExperimentalSortKey) {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 5_000usize;
    let node_size = 16usize;
    let boxes = random_boxes(&mut rng, n);

    let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(n, node_size);
    let mut index = IndexBuilder::new(n)
        .node_size(node_size)
        .experimental_sort_key(choice);
    for b in &boxes {
        reference.add(b[0], b[1], b[2], b[3]);
        index.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let reference = reference.build().unwrap();
    let index = index.finish().unwrap();

    for _ in 0..500 {
        let qx: f64 = rng.gen_range(0.0..1000.0);
        let qy: f64 = rng.gen_range(0.0..1000.0);
        let qw: f64 = rng.gen_range(1.0..100.0);
        let qh: f64 = rng.gen_range(1.0..100.0);
        let query = Rect::new(qx, qy, qx + qw, qy + qh);

        let mut expected = reference.query(qx, qy, qx + qw, qy + qh);
        let mut actual = index.search(query);
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(
            expected, actual,
            "search results differ (choice={choice:?})"
        );
    }
}

#[test]
fn index_search_matches_reference_magic() {
    check_experimental_sort_key_matches_reference(ExperimentalSortKey::HilbertMagicBits);
}

#[test]
fn index_search_matches_reference_loop() {
    check_experimental_sort_key_matches_reference(ExperimentalSortKey::HilbertLoopRotation);
}

#[test]
fn index_search_matches_reference_lut() {
    check_experimental_sort_key_matches_reference(ExperimentalSortKey::HilbertLut);
}

#[test]
fn index_search_matches_reference_morton() {
    check_experimental_sort_key_matches_reference(ExperimentalSortKey::Morton);
}

#[test]
fn public_sort_keys_match_reference() {
    for key in [SortKey::Hilbert, SortKey::Morton] {
        let mut rng = StdRng::seed_from_u64(123);
        let n = 2_000usize;
        let boxes = random_boxes(&mut rng, n);

        let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(n, 16);
        let mut index = IndexBuilder::new(n).sort_key(key);
        for b in &boxes {
            reference.add(b[0], b[1], b[2], b[3]);
            index.add(rect(*b));
        }
        let reference = reference.build().unwrap();
        let index = index.finish().unwrap();

        let query = Rect::new(250.0, 250.0, 750.0, 750.0);
        let mut expected = reference.query(query.min_x, query.min_y, query.max_x, query.max_y);
        let mut actual = index.search(query);
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(expected, actual, "public sort key differs: {key:?}");
    }
}

#[test]
fn default_builder_uses_node_size_16() {
    let mut builder = IndexBuilder::new(17);
    for i in 0..17 {
        builder.add_bounds(i as f64, 0.0, i as f64 + 0.5, 1.0);
    }
    let index = builder.finish().unwrap();
    assert_eq!(index.node_size(), 16);
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
        let cx: f64 = rng.gen_range(0.0..10_000.0);
        let cy: f64 = rng.gen_range(0.0..10_000.0);
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
        let qx: f64 = rng.gen_range(0.0..10_000.0);
        let qy: f64 = rng.gen_range(0.0..10_000.0);
        let query = Rect::new(qx, qy, qx + 150.0, qy + 150.0);
        let mut a = serial.search(query);
        let mut b = parallel.search(query);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

#[cfg(feature = "simd")]
#[test]
fn simd_empty_and_small_indexes_behave_like_aos() {
    let empty = IndexBuilder::new(0).finish_simd().unwrap();
    assert_eq!(empty.num_items(), 0);
    assert!(empty.search(Rect::new(-1.0, -1.0, 1.0, 1.0)).is_empty());

    let boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [2.0, 2.0, 3.0, 3.0],
        [-1.0, -1.0, 0.5, 0.5],
    ];
    let mut aos = IndexBuilder::new(boxes.len());
    let mut simd = IndexBuilder::new(boxes.len());
    for b in boxes {
        aos.add_bounds(b[0], b[1], b[2], b[3]);
        simd.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    let query = Rect::new(-0.25, -0.25, 2.25, 2.25);
    let mut expected = aos.search(query);
    let mut actual = simd.search(query);
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(expected, actual);
}

#[cfg(feature = "simd")]
#[test]
fn simd_finish_reports_count_mismatch() {
    let mut builder = IndexBuilder::new(2);
    builder.add_bounds(0.0, 0.0, 1.0, 1.0);

    assert!(matches!(
        builder.finish_simd(),
        Err(BuildError::ItemCount {
            added: 1,
            expected: 2
        })
    ));
}

#[cfg(feature = "simd")]
#[test]
fn simd_search_apis_agree_with_aos() {
    let mut builder = IndexBuilder::new(3);
    builder.add_bounds(0.0, 0.0, 1.0, 1.0);
    builder.add_bounds(5.0, 5.0, 6.0, 6.0);
    builder.add_bounds(0.5, 0.5, 2.0, 2.0);
    let simd = builder.finish_simd().unwrap();

    let query = Rect::new(0.0, 0.0, 2.0, 2.0);
    let mut expected = simd.search(query);
    expected.sort_unstable();

    let mut out = Vec::new();
    simd.search_into(query, &mut out);
    out.sort_unstable();
    assert_eq!(expected, out);

    let mut workspace = SearchWorkspace::new();
    let mut with = simd.search_with(query, &mut workspace).to_vec();
    with.sort_unstable();
    assert_eq!(expected, with);

    assert!(simd.any(query));
    assert!(!simd.any(Rect::new(10.0, 10.0, 11.0, 11.0)));
    assert!(matches!(simd.first(query), Some(0 | 2)));
    assert_eq!(simd.first(Rect::new(10.0, 10.0, 11.0, 11.0)), None);

    let mut visited = Vec::new();
    let completed: ControlFlow<()> = simd.visit(query, |idx| {
        visited.push(idx);
        ControlFlow::Continue(())
    });
    assert!(completed.is_continue());
    visited.sort_unstable();
    assert_eq!(expected, visited);
}

#[cfg(feature = "simd")]
#[test]
fn simd_index_search_matches_reference() {
    let mut rng = StdRng::seed_from_u64(99);
    let n = 5_000usize;
    let node_size = 16usize;
    let boxes = random_boxes(&mut rng, n);

    let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(n, node_size);
    let mut builder = IndexBuilder::new(n)
        .node_size(node_size)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut);
    for b in &boxes {
        reference.add(b[0], b[1], b[2], b[3]);
        builder.add_bounds(b[0], b[1], b[2], b[3]);
    }
    let reference = reference.build().unwrap();
    let simd = builder.finish_simd().unwrap();

    let (mut scalar, mut simd_out, mut simd_prefetch, mut avx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut st1, mut st2, mut st3, mut st4) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..500 {
        let qx: f64 = rng.gen_range(0.0..1000.0);
        let qy: f64 = rng.gen_range(0.0..1000.0);
        let qw: f64 = rng.gen_range(1.0..100.0);
        let qh: f64 = rng.gen_range(1.0..100.0);
        let query = Rect::new(qx, qy, qx + qw, qy + qh);

        let mut expected = reference.query(qx, qy, qx + qw, qy + qh);
        simd.search_scalar(query, &mut scalar, &mut st1);
        simd.search_simd(query, &mut simd_out, &mut st2);
        simd.search_simd_prefetch(query, &mut simd_prefetch, &mut st3);
        simd.search_avx512(query, &mut avx, &mut st4);
        expected.sort_unstable();
        scalar.sort_unstable();
        simd_out.sort_unstable();
        simd_prefetch.sort_unstable();
        avx.sort_unstable();
        assert_eq!(expected, scalar, "SoA-scalar != reference");
        assert_eq!(expected, simd_out, "SoA-SIMD != reference");
        assert_eq!(expected, simd_prefetch, "SoA-SIMD-prefetch != reference");
        assert_eq!(expected, avx, "SoA-AVX512 != reference");
    }
}
