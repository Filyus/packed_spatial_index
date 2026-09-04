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

/// The baseline the dual-tree join replaces: one kNN query per item.
fn naive_knn_join(a: &[Box2D], b: &Index2D, k: usize) -> Vec<Vec<usize>> {
    a.iter().map(|&q| b.neighbors_of_box(q, k)).collect()
}

/// Distances of a row, which is what the API actually pins down — ties at the
/// kth distance are traversal order, so two correct implementations may pick
/// different ids for the same distance.
fn row_distances(query: Box2D, boxes: &[Box2D], row: &[usize]) -> Vec<f64> {
    row.iter()
        .map(|&i| query.distance_to_box(boxes[i]))
        .collect()
}

fn assert_matches_naive(a_boxes: &[Box2D], b_boxes: &[Box2D], k: usize) {
    let b = build(b_boxes);
    let a = build(a_boxes);
    let got = a.knn_join(&b, k);
    let want = naive_knn_join(a_boxes, &b, k);
    assert_eq!(got.len(), a_boxes.len());
    for (i, (got_row, want_row)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(got_row.len(), want_row.len(), "row {i} length");
        let got_d = row_distances(a_boxes[i], b_boxes, got_row);
        let want_d = row_distances(a_boxes[i], b_boxes, want_row);
        assert_eq!(got_d, want_d, "row {i} distances (k={k})");
        // Nearest first, like `neighbors_of_box`.
        assert!(
            got_d.windows(2).all(|w| w[0] <= w[1]),
            "row {i} not in nondecreasing distance order: {got_d:?}"
        );
    }
}

#[test]
fn knn_join_matches_the_per_item_loop() {
    let mut rng = StdRng::seed_from_u64(6101);
    for (n, m, max_size) in [
        (1, 1, 4.0),
        (17, 40, 8.0),
        (200, 150, 3.0),
        (150, 200, 25.0), // dense: many overlapping, so many zero distances
    ] {
        let a_boxes = random_boxes(&mut rng, n, 100.0, max_size);
        let b_boxes = random_boxes(&mut rng, m, 100.0, max_size);
        for k in [1, 2, 5, 10] {
            assert_matches_naive(&a_boxes, &b_boxes, k);
        }
    }
}

#[test]
fn k_larger_than_other_returns_every_item() {
    let mut rng = StdRng::seed_from_u64(6102);
    let a_boxes = random_boxes(&mut rng, 20, 100.0, 4.0);
    let b_boxes = random_boxes(&mut rng, 7, 100.0, 4.0);
    let a = build(&a_boxes);
    let b = build(&b_boxes);
    for row in a.knn_join(&b, 50) {
        assert_eq!(row.len(), 7, "every item of b, and no more");
        let mut sorted = row.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 7, "an item was reported twice");
    }
}

#[test]
fn degenerate_inputs_give_empty_rows() {
    let a_boxes = [
        Box2D::new(0.0, 0.0, 1.0, 1.0),
        Box2D::new(5.0, 5.0, 6.0, 6.0),
    ];
    let a = build(&a_boxes);
    let b = build(&[Box2D::new(2.0, 2.0, 3.0, 3.0)]);
    let empty = build(&[]);

    // k = 0 asks for nothing.
    assert_eq!(a.knn_join(&b, 0), vec![Vec::<usize>::new(); 2]);
    // Nothing to be near.
    assert_eq!(a.knn_join(&empty, 3), vec![Vec::<usize>::new(); 2]);
    // No items to answer for.
    assert!(empty.knn_join(&b, 3).is_empty());
}

#[test]
fn rows_are_indexed_by_item_id_not_traversal_order() {
    // Item ids are the order they were added; the tree stores them in spatial
    // sort order, so a row landing in the wrong slot would still look
    // plausible without this.
    let a_boxes = [
        Box2D::new(90.0, 90.0, 91.0, 91.0), // id 0, far from b0
        Box2D::new(0.0, 0.0, 1.0, 1.0),     // id 1, on top of b0
    ];
    let a = build(&a_boxes);
    let b = build(&[
        Box2D::new(0.0, 0.0, 1.0, 1.0),
        Box2D::new(89.0, 89.0, 90.0, 90.0),
    ]);
    assert_eq!(a.knn_join(&b, 1), vec![vec![1], vec![0]]);
}

#[test]
fn self_join_reports_the_item_itself_first() {
    // Joined against itself, every item's nearest neighbour is itself at
    // distance zero. Worth pinning: it is the reason there is no `self_`
    // variant here that silently drops the diagonal.
    let boxes = [
        Box2D::new(0.0, 0.0, 1.0, 1.0),
        Box2D::new(10.0, 0.0, 11.0, 1.0),
        Box2D::new(50.0, 0.0, 51.0, 1.0),
    ];
    let index = build(&boxes);
    let rows = index.knn_join(&index, 1);
    assert_eq!(rows, vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn view_matches_owned() {
    let mut rng = StdRng::seed_from_u64(6103);
    let a_boxes = random_boxes(&mut rng, 120, 100.0, 5.0);
    let b_boxes = random_boxes(&mut rng, 90, 100.0, 5.0);
    let a = build(&a_boxes);
    let b = build(&b_boxes);
    let bytes_a = a.to_bytes();
    let bytes_b = b.to_bytes();
    let view_a = Index2DView::from_bytes(&bytes_a).unwrap();
    let view_b = Index2DView::from_bytes(&bytes_b).unwrap();
    for k in [1, 4] {
        assert_eq!(view_a.knn_join(&view_b, k), a.knn_join(&b, k));
    }
}

/// Independently of the per-item loop: no item of `other` may be strictly
/// closer than the row's own kth, and the row must be full when it can be.
/// Strict inequality, so ties at the kth distance — which are traversal order
/// — cannot make this flap, and unlike a radius query it never has to round a
/// squared distance back through a square root.
#[test]
fn no_closer_item_is_left_out_of_a_row() {
    let mut rng = StdRng::seed_from_u64(6104);
    let a_boxes = random_boxes(&mut rng, 150, 100.0, 3.0);
    let b_boxes = random_boxes(&mut rng, 200, 100.0, 3.0);
    let a = build(&a_boxes);
    let b = build(&b_boxes);
    let k = 5;
    for (i, row) in a.knn_join(&b, k).iter().enumerate() {
        assert_eq!(row.len(), k, "row {i} is short of k");
        let kth = a_boxes[i].distance_to_box(b_boxes[*row.last().unwrap()]);
        let named: std::collections::BTreeSet<usize> = row.iter().copied().collect();
        for (j, &box_b) in b_boxes.iter().enumerate() {
            if a_boxes[i].distance_to_box(box_b) < kth {
                assert!(
                    named.contains(&j),
                    "row {i} omits {j}, which is closer than its own kth"
                );
            }
        }
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
    fn simd_and_view_match_the_scalar_oracle() {
        let mut rng = StdRng::seed_from_u64(6105);
        let a_boxes = random_boxes(&mut rng, 300, 100.0, 4.0);
        let b_boxes = random_boxes(&mut rng, 260, 100.0, 4.0);
        let scalar_a = build(&a_boxes);
        let scalar_b = build(&b_boxes);
        let a = build_simd(&a_boxes);
        let b = build_simd(&b_boxes);
        let bytes_a = a.to_bytes();
        let bytes_b = b.to_bytes();
        let view_a = SimdIndex2DView::from_bytes(&bytes_a).unwrap();
        let view_b = SimdIndex2DView::from_bytes(&bytes_b).unwrap();

        for k in [1, 3, 8] {
            let want = scalar_a.knn_join(&scalar_b, k);
            for (label, got) in [
                ("simd", a.knn_join(&b, k)),
                ("view", view_a.knn_join(&view_b, k)),
            ] {
                for (i, (got_row, want_row)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        row_distances(a_boxes[i], &b_boxes, got_row),
                        row_distances(a_boxes[i], &b_boxes, want_row),
                        "{label} row {i} (k={k})"
                    );
                }
            }
        }
    }
}
