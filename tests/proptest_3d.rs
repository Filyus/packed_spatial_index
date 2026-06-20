//! Property-based correctness and robustness tests for the 3D index.
//!
//! Mirrors `proptest_2d.rs`: `search` (scalar, view, SIMD) agrees with a
//! brute-force scan, `neighbors` (kNN) and `self_join` match brute force, and
//! `from_bytes` never panics on arbitrary or mutated byte buffers despite using
//! `*_unchecked` accessors after header validation.

#[cfg(feature = "simd")]
use packed_spatial_index::SimdIndex3D;
use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView, Point3D};
use proptest::prelude::*;

/// Boxes on a small integer grid so edges collide on exact boundaries.
fn boxes_strategy() -> impl Strategy<Value = Vec<[f64; 6]>> {
    let single =
        (0i64..12, 0i64..12, 0i64..12, 0i64..5, 0i64..5, 0i64..5).prop_map(|(x, y, z, w, h, d)| {
            [
                x as f64,
                y as f64,
                z as f64,
                (x + w) as f64,
                (y + h) as f64,
                (z + d) as f64,
            ]
        });
    prop::collection::vec(single, 0..64)
}

fn query_strategy() -> impl Strategy<Value = [f64; 6]> {
    (0i64..12, 0i64..12, 0i64..12, 0i64..7, 0i64..7, 0i64..7).prop_map(|(x, y, z, w, h, d)| {
        [
            x as f64,
            y as f64,
            z as f64,
            (x + w) as f64,
            (y + h) as f64,
            (z + d) as f64,
        ]
    })
}

fn box3d(b: &[f64; 6]) -> Box3D {
    Box3D::new(b[0], b[1], b[2], b[3], b[4], b[5])
}

fn build(boxes: &[[f64; 6]]) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len());
    for b in boxes {
        builder.add(box3d(b));
    }
    builder.finish().unwrap()
}

fn brute_force(boxes: &[[f64; 6]], query: Box3D) -> Vec<usize> {
    boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| query.overlaps(box3d(b)))
        .map(|(i, _)| i)
        .collect()
}

fn point_strategy() -> impl Strategy<Value = [f64; 3]> {
    (0i64..12, 0i64..12, 0i64..12).prop_map(|(x, y, z)| [x as f64, y as f64, z as f64])
}

/// Squared point-to-box distance (0 when inside), the metric `neighbors` uses.
fn point_box_dist2(b: &[f64; 6], p: [f64; 3]) -> f64 {
    let dx = (b[0] - p[0]).max(0.0).max(p[0] - b[3]);
    let dy = (b[1] - p[1]).max(0.0).max(p[1] - b[4]);
    let dz = (b[2] - p[2]).max(0.0).max(p[2] - b[5]);
    dx * dx + dy * dy + dz * dz
}

fn brute_force_knn_dists(boxes: &[[f64; 6]], p: [f64; 3], k: usize) -> Vec<f64> {
    let mut d: Vec<f64> = boxes.iter().map(|b| point_box_dist2(b, p)).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d.truncate(k);
    d
}

fn brute_force_self_join(boxes: &[[f64; 6]]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, bi) in boxes.iter().enumerate() {
        let bi = box3d(bi);
        for (j, bj) in boxes.iter().enumerate().skip(i + 1) {
            if bi.overlaps(box3d(bj)) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Sorted distances of the returned kNN ids — tie-safe (compares distances, not
/// ids, since the grid produces equal distances where item choice is unspecified).
fn knn_dists(ids: &[usize], boxes: &[[f64; 6]], p: [f64; 3]) -> Vec<f64> {
    let mut d: Vec<f64> = ids.iter().map(|&i| point_box_dist2(&boxes[i], p)).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d
}

fn norm(pair: (usize, usize)) -> (usize, usize) {
    if pair.0 <= pair.1 {
        pair
    } else {
        (pair.1, pair.0)
    }
}

proptest! {
    #[test]
    fn search_matches_brute_force(boxes in boxes_strategy(), q in query_strategy()) {
        let query = box3d(&q);

        let mut expected = brute_force(&boxes, query);
        expected.sort_unstable();

        let index = build(&boxes);
        let mut scalar = index.search(query);
        scalar.sort_unstable();
        prop_assert_eq!(&scalar, &expected);

        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        let mut borrowed = view.search(query);
        borrowed.sort_unstable();
        prop_assert_eq!(&borrowed, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index3DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box3d(b));
            }
            let simd = builder.finish_simd().unwrap();
            let mut simd_hits = simd.search(query);
            simd_hits.sort_unstable();
            prop_assert_eq!(&simd_hits, &expected);
        }
    }

    /// kNN: returned items' distances (sorted) equal the k smallest brute-force
    /// distances — scalar index, view, and SIMD index.
    #[test]
    fn neighbors_match_brute_force(boxes in boxes_strategy(), p in point_strategy(), k in 1usize..8) {
        let point = Point3D::new(p[0], p[1], p[2]);
        let expected = brute_force_knn_dists(&boxes, p, k);

        let index = build(&boxes);
        prop_assert_eq!(knn_dists(&index.neighbors(point, k), &boxes, p), expected.clone());

        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        prop_assert_eq!(knn_dists(&view.neighbors(point, k), &boxes, p), expected.clone());

        #[cfg(feature = "simd")]
        {
            let mut builder = Index3DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box3d(b));
            }
            let simd = builder.finish_simd().unwrap();
            prop_assert_eq!(knn_dists(&simd.neighbors(point, k), &boxes, p), expected.clone());
        }
    }

    /// `self_join`: exactly the brute-force set of intersecting pairs (ids within
    /// a pair are order-independent) — scalar index, view, and SIMD index.
    #[test]
    fn self_join_matches_brute_force(boxes in boxes_strategy()) {
        let mut expected = brute_force_self_join(&boxes);
        expected.sort_unstable();

        let index = build(&boxes);
        let mut got: Vec<(usize, usize)> = index.self_join().into_iter().map(norm).collect();
        got.sort_unstable();
        prop_assert_eq!(&got, &expected);

        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        let mut got_v: Vec<(usize, usize)> = view.self_join().into_iter().map(norm).collect();
        got_v.sort_unstable();
        prop_assert_eq!(&got_v, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index3DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box3d(b));
            }
            let simd = builder.finish_simd().unwrap();
            let mut got_s: Vec<(usize, usize)> = simd.self_join().into_iter().map(norm).collect();
            got_s.sort_unstable();
            prop_assert_eq!(&got_s, &expected);
        }
    }

    #[test]
    fn from_bytes_never_panics_on_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = Index3D::from_bytes(&bytes);
        let _ = Index3DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex3D::from_bytes(&bytes);
        }
    }

    #[test]
    fn from_bytes_tolerates_mutated_valid_bytes(
        boxes in boxes_strategy(),
        pos in any::<prop::sample::Index>(),
        xor in 1u8..=255,
        truncate in any::<bool>(),
    ) {
        let mut bytes = build(&boxes).to_bytes();
        if !bytes.is_empty() {
            let idx = pos.index(bytes.len());
            bytes[idx] ^= xor;
        }
        if truncate && !bytes.is_empty() {
            bytes.truncate(bytes.len() / 2);
        }

        let _ = Index3D::from_bytes(&bytes);
        let _ = Index3DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex3D::from_bytes(&bytes);
        }
    }
}
