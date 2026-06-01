mod common;

use common::{random_boxes, rect};
use packed_spatial_index::experimental::{ENCODERS, ExperimentalSortKey};
use packed_spatial_index::{IndexBuilder, Rect, SortKey};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use static_aabb2d_index::{StaticAABB2DIndexBuilder, hilbert_xy_to_index};

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
        let x: u16 = rng.random();
        let y: u16 = rng.random();
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
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let qw: f64 = rng.random_range(1.0..100.0);
        let qh: f64 = rng.random_range(1.0..100.0);
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
    let mut rng = StdRng::seed_from_u64(123);
    let n = 2_000usize;
    let boxes = random_boxes(&mut rng, n);

    let mut reference = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(n, 16);
    let mut index = IndexBuilder::new(n).sort_key(SortKey::Hilbert);
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
    assert_eq!(expected, actual);
}
