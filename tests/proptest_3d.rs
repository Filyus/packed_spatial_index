//! Property-based correctness and robustness tests for the 3D index.
//!
//! Mirrors `proptest_2d.rs`: `search` (scalar, view, SIMD) agrees with a
//! brute-force scan; `neighbors` (kNN), `self_join`, `raycast` /
//! `raycast_closest`, and frustum culling match their brute-force oracle; and
//! `from_bytes` never panics on arbitrary or mutated byte buffers despite using
//! `*_unchecked` accessors after header validation.

#[cfg(feature = "simd")]
use packed_spatial_index::SimdIndex3D;
use packed_spatial_index::{
    Box3D, Frustum3D, Index3D, Index3DBuilder, Index3DView, Point3D, Ray3D,
};
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

/// Grid origin, a non-zero integer direction, generous max_t covering the field.
fn ray_strategy() -> impl Strategy<Value = (f64, f64, f64, f64, f64, f64)> {
    (
        0i64..12,
        0i64..12,
        0i64..12,
        -2i64..=2,
        -2i64..=2,
        -2i64..=2,
    )
        .prop_filter("non-zero direction", |(_, _, _, dx, dy, dz)| {
            *dx != 0 || *dy != 0 || *dz != 0
        })
        .prop_map(|(ox, oy, oz, dx, dy, dz)| {
            (
                ox as f64, oy as f64, oz as f64, dx as f64, dy as f64, dz as f64,
            )
        })
}

/// An axis-aligned box frustum `[lo, hi]^3` (lo < hi always).
fn frustum_strategy() -> impl Strategy<Value = Frustum3D> {
    (0i64..6, 6i64..12).prop_map(|(lo, hi)| {
        let (lo, hi) = (lo as f64, hi as f64);
        Frustum3D::from_planes([
            [1.0, 0.0, 0.0, -lo],
            [-1.0, 0.0, 0.0, hi],
            [0.0, 1.0, 0.0, -lo],
            [0.0, -1.0, 0.0, hi],
            [0.0, 0.0, 1.0, -lo],
            [0.0, 0.0, -1.0, hi],
        ])
    })
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

    /// All-hits raycast returns exactly the boxes the ray segment enters (oracle is
    /// the public `Ray3D::intersects_box`) — scalar index, view, and SIMD index.
    #[test]
    fn raycast_matches_predicate(
        (ox, oy, oz, dx, dy, dz) in ray_strategy(),
        boxes in boxes_strategy(),
    ) {
        let ray = Ray3D::new(Point3D::new(ox, oy, oz), dx, dy, dz, 48.0);
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| ray.intersects_box(box3d(b)))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let index = build(&boxes);
        let mut got = index.raycast(ray);
        got.sort_unstable();
        prop_assert_eq!(&got, &expected);

        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        let mut got_v = view.raycast(ray);
        got_v.sort_unstable();
        prop_assert_eq!(&got_v, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index3DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box3d(b));
            }
            let simd = builder.finish_simd().unwrap();
            let mut got_s = simd.raycast(ray);
            got_s.sort_unstable();
            prop_assert_eq!(&got_s, &expected);
        }
    }

    /// Closest-hit raycast returns the minimum entry `t` (compare the `t`, not the
    /// id, so ties are safe) — scalar index, view, and SIMD index.
    #[test]
    fn raycast_closest_matches_brute_force(
        (ox, oy, oz, dx, dy, dz) in ray_strategy(),
        boxes in boxes_strategy(),
    ) {
        let ray = Ray3D::new(Point3D::new(ox, oy, oz), dx, dy, dz, 48.0);
        let expected = boxes
            .iter()
            .filter_map(|b| ray.enter_t(box3d(b)))
            .fold(None, |acc: Option<f64>, t| Some(acc.map_or(t, |a| a.min(t))));

        let index = build(&boxes);
        prop_assert_eq!(index.raycast_closest(ray).map(|(_, t)| t), expected);

        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        prop_assert_eq!(view.raycast_closest(ray).map(|(_, t)| t), expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index3DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box3d(b));
            }
            let simd = builder.finish_simd().unwrap();
            prop_assert_eq!(simd.raycast_closest(ray).map(|(_, t)| t), expected);
        }
    }

    /// Frustum culling returns exactly the boxes overlapping the frustum (oracle is
    /// `Frustum3D::overlaps_box`) — scalar index and view (f64-only).
    #[test]
    fn search_frustum_matches_predicate(frustum in frustum_strategy(), boxes in boxes_strategy()) {
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| frustum.overlaps_box(box3d(b)))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let index = build(&boxes);
        let mut got = index.search_frustum(frustum);
        got.sort_unstable();
        prop_assert_eq!(&got, &expected);

        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        let mut got_v = view.search_frustum(frustum);
        got_v.sort_unstable();
        prop_assert_eq!(&got_v, &expected);
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
