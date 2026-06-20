//! Property-based correctness and robustness tests for the 2D index.
//!
//! These complement the hand-written cases in `api_2d.rs` and `persistence_2d.rs`
//! by sweeping a wide space of inputs:
//!   * `search` (scalar, view, and SIMD) agrees with a brute-force scan;
//!   * `neighbors` (kNN) returns the k nearest by distance, tie-safe;
//!   * `self_join` returns exactly the brute-force set of intersecting pairs;
//!   * `raycast` / `raycast_closest` and the triangle / convex-polygon region
//!     queries agree with their public predicate run over every box;
//!   * `from_bytes` never panics on arbitrary or mutated byte buffers, even
//!     though it relies on `*_unchecked` accessors after header validation.

#[cfg(feature = "simd")]
use packed_spatial_index::SimdIndex2D;
use packed_spatial_index::{
    Box2D, ConvexPolygon2D, Index2D, Index2DBuilder, Index2DView, Point2D, Ray2D, Triangle2D,
};
use proptest::prelude::*;

/// Generate boxes on a small integer grid so edges collide often. Inclusive-edge
/// overlap bugs hide on exact boundaries, which random floats almost never hit.
fn boxes_strategy() -> impl Strategy<Value = Vec<[f64; 4]>> {
    let single = (0i64..16, 0i64..16, 0i64..6, 0i64..6)
        .prop_map(|(x, y, w, h)| [x as f64, y as f64, (x + w) as f64, (y + h) as f64]);
    prop::collection::vec(single, 0..64)
}

fn query_strategy() -> impl Strategy<Value = [f64; 4]> {
    (0i64..16, 0i64..16, 0i64..8, 0i64..8)
        .prop_map(|(x, y, w, h)| [x as f64, y as f64, (x + w) as f64, (y + h) as f64])
}

fn build(boxes: &[[f64; 4]]) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len());
    for b in boxes {
        builder.add(Box2D::new(b[0], b[1], b[2], b[3]));
    }
    builder.finish().unwrap()
}

fn brute_force(boxes: &[[f64; 4]], query: Box2D) -> Vec<usize> {
    boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| query.overlaps(Box2D::new(b[0], b[1], b[2], b[3])))
        .map(|(i, _)| i)
        .collect()
}

fn point_strategy() -> impl Strategy<Value = [f64; 2]> {
    (0i64..16, 0i64..16).prop_map(|(x, y)| [x as f64, y as f64])
}

/// Squared point-to-box distance (0 when the point is inside), the same metric
/// `neighbors` uses. Integer-grid boxes make this bit-exact, so the k smallest
/// distances can be compared with `==`.
fn point_box_dist2(b: &[f64; 4], px: f64, py: f64) -> f64 {
    let dx = (b[0] - px).max(0.0).max(px - b[2]);
    let dy = (b[1] - py).max(0.0).max(py - b[3]);
    dx * dx + dy * dy
}

fn brute_force_knn_dists(boxes: &[[f64; 4]], px: f64, py: f64, k: usize) -> Vec<f64> {
    let mut d: Vec<f64> = boxes.iter().map(|b| point_box_dist2(b, px, py)).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d.truncate(k);
    d
}

fn brute_force_self_join(boxes: &[[f64; 4]]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, bi) in boxes.iter().enumerate() {
        let bi = Box2D::new(bi[0], bi[1], bi[2], bi[3]);
        for (j, bj) in boxes.iter().enumerate().skip(i + 1) {
            if bi.overlaps(Box2D::new(bj[0], bj[1], bj[2], bj[3])) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Sorted distances of the returned kNN ids, for comparing against a brute-force
/// list. Comparing distances (not ids) is tie-safe: the grid produces equal
/// distances often, where which specific item is returned is unspecified.
fn knn_dists(ids: &[usize], boxes: &[[f64; 4]], px: f64, py: f64) -> Vec<f64> {
    let mut d: Vec<f64> = ids
        .iter()
        .map(|&i| point_box_dist2(&boxes[i], px, py))
        .collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d
}

/// Pair with the two ids ordered, since `self_join` does not promise an order
/// within a pair.
fn norm(pair: (usize, usize)) -> (usize, usize) {
    if pair.0 <= pair.1 {
        pair
    } else {
        (pair.1, pair.0)
    }
}

fn box2d(b: &[f64; 4]) -> Box2D {
    Box2D::new(b[0], b[1], b[2], b[3])
}

/// Grid origin, a non-zero integer direction, generous max_t covering the field.
fn ray_strategy() -> impl Strategy<Value = (f64, f64, f64, f64)> {
    (0i64..16, 0i64..16, -2i64..=2, -2i64..=2)
        .prop_filter("non-zero direction", |(_, _, dx, dy)| *dx != 0 || *dy != 0)
        .prop_map(|(ox, oy, dx, dy)| (ox as f64, oy as f64, dx as f64, dy as f64))
}

fn triangle_strategy() -> impl Strategy<Value = Triangle2D> {
    (0i64..16, 0i64..16, 0i64..16, 0i64..16, 0i64..16, 0i64..16).prop_map(
        |(ax, ay, bx, by, cx, cy)| {
            Triangle2D::new(
                [ax as f64, ay as f64],
                [bx as f64, by as f64],
                [cx as f64, cy as f64],
            )
        },
    )
}

/// A regular n-gon at a random center/radius — always convex, as the SAT
/// predicate requires.
fn polygon_strategy() -> impl Strategy<Value = ConvexPolygon2D> {
    (1i64..15, 1i64..15, 2i64..7, 3usize..=6).prop_map(|(cx, cy, r, n)| {
        let verts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * (i as f64) / (n as f64);
                [
                    cx as f64 + (r as f64) * a.cos(),
                    cy as f64 + (r as f64) * a.sin(),
                ]
            })
            .collect();
        ConvexPolygon2D::new(verts)
    })
}

proptest! {
    /// Scalar, view, and SIMD searches must all return exactly the brute-force set.
    #[test]
    fn search_matches_brute_force(boxes in boxes_strategy(), q in query_strategy()) {
        let query = Box2D::new(q[0], q[1], q[2], q[3]);

        let mut expected = brute_force(&boxes, query);
        expected.sort_unstable();

        let index = build(&boxes);
        let mut scalar = index.search(query);
        scalar.sort_unstable();
        prop_assert_eq!(&scalar, &expected);

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        let mut borrowed = view.search(query);
        borrowed.sort_unstable();
        prop_assert_eq!(&borrowed, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index2DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(Box2D::new(b[0], b[1], b[2], b[3]));
            }
            let simd = builder.finish_simd().unwrap();
            let mut simd_hits = simd.search(query);
            simd_hits.sort_unstable();
            prop_assert_eq!(&simd_hits, &expected);
        }
    }

    /// kNN agrees with brute force: the returned items' distances (sorted) equal
    /// the k smallest brute-force distances — scalar index, view, and SIMD index.
    #[test]
    fn neighbors_match_brute_force(boxes in boxes_strategy(), p in point_strategy(), k in 1usize..8) {
        let point = Point2D::new(p[0], p[1]);
        let expected = brute_force_knn_dists(&boxes, p[0], p[1], k);

        let index = build(&boxes);
        prop_assert_eq!(knn_dists(&index.neighbors(point, k), &boxes, p[0], p[1]), expected.clone());

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        prop_assert_eq!(knn_dists(&view.neighbors(point, k), &boxes, p[0], p[1]), expected.clone());

        #[cfg(feature = "simd")]
        {
            let mut builder = Index2DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(Box2D::new(b[0], b[1], b[2], b[3]));
            }
            let simd = builder.finish_simd().unwrap();
            prop_assert_eq!(knn_dists(&simd.neighbors(point, k), &boxes, p[0], p[1]), expected.clone());
        }
    }

    /// `self_join` returns exactly the brute-force set of intersecting pairs (ids
    /// within a pair are order-independent) — scalar index, view, and SIMD index.
    #[test]
    fn self_join_matches_brute_force(boxes in boxes_strategy()) {
        let mut expected = brute_force_self_join(&boxes);
        expected.sort_unstable();

        let index = build(&boxes);
        let mut got: Vec<(usize, usize)> = index.self_join().into_iter().map(norm).collect();
        got.sort_unstable();
        prop_assert_eq!(&got, &expected);

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        let mut got_v: Vec<(usize, usize)> = view.self_join().into_iter().map(norm).collect();
        got_v.sort_unstable();
        prop_assert_eq!(&got_v, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index2DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(Box2D::new(b[0], b[1], b[2], b[3]));
            }
            let simd = builder.finish_simd().unwrap();
            let mut got_s: Vec<(usize, usize)> = simd.self_join().into_iter().map(norm).collect();
            got_s.sort_unstable();
            prop_assert_eq!(&got_s, &expected);
        }
    }

    /// All-hits raycast returns exactly the boxes the ray segment enters (oracle is
    /// the public `Ray2D::intersects_box`) — scalar index, view, and SIMD index.
    #[test]
    fn raycast_matches_predicate((ox, oy, dx, dy) in ray_strategy(), boxes in boxes_strategy()) {
        let ray = Ray2D::new(Point2D::new(ox, oy), dx, dy, 64.0);
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| ray.intersects_box(box2d(b)))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let index = build(&boxes);
        let mut got = index.raycast(ray);
        got.sort_unstable();
        prop_assert_eq!(&got, &expected);

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        let mut got_v = view.raycast(ray);
        got_v.sort_unstable();
        prop_assert_eq!(&got_v, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index2DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box2d(b));
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
    fn raycast_closest_matches_brute_force((ox, oy, dx, dy) in ray_strategy(), boxes in boxes_strategy()) {
        let ray = Ray2D::new(Point2D::new(ox, oy), dx, dy, 64.0);
        let expected = boxes
            .iter()
            .filter_map(|b| ray.enter_t(box2d(b)))
            .fold(None, |acc: Option<f64>, t| Some(acc.map_or(t, |a| a.min(t))));

        let index = build(&boxes);
        prop_assert_eq!(index.raycast_closest(ray).map(|(_, t)| t), expected);

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        prop_assert_eq!(view.raycast_closest(ray).map(|(_, t)| t), expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index2DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box2d(b));
            }
            let simd = builder.finish_simd().unwrap();
            prop_assert_eq!(simd.raycast_closest(ray).map(|(_, t)| t), expected);
        }
    }

    /// Triangle region search returns exactly the boxes overlapping the triangle
    /// (oracle is `Triangle2D::overlaps_box`) — scalar index and view (f64-only).
    #[test]
    fn search_triangle_matches_predicate(tri in triangle_strategy(), boxes in boxes_strategy()) {
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| tri.overlaps_box(box2d(b)))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let index = build(&boxes);
        let mut got = index.search_triangle(tri);
        got.sort_unstable();
        prop_assert_eq!(&got, &expected);

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        let mut got_v = view.search_triangle(tri);
        got_v.sort_unstable();
        prop_assert_eq!(&got_v, &expected);
    }

    /// Convex-polygon region search returns exactly the boxes overlapping the
    /// polygon (oracle is `ConvexPolygon2D::overlaps_box`) — scalar index and view.
    #[test]
    fn search_polygon_matches_predicate(poly in polygon_strategy(), boxes in boxes_strategy()) {
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| poly.overlaps_box(box2d(b)))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let index = build(&boxes);
        let mut got = index.search_polygon(&poly);
        got.sort_unstable();
        prop_assert_eq!(&got, &expected);

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        let mut got_v = view.search_polygon(&poly);
        got_v.sort_unstable();
        prop_assert_eq!(&got_v, &expected);
    }

    /// Arbitrary bytes must yield `Err`, never a panic or out-of-bounds read.
    #[test]
    fn from_bytes_never_panics_on_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = Index2D::from_bytes(&bytes);
        let _ = Index2DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex2D::from_bytes(&bytes);
        }
    }

    /// Mutating a valid buffer must still be handled gracefully (Ok or Err, no panic).
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

        let _ = Index2D::from_bytes(&bytes);
        let _ = Index2DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex2D::from_bytes(&bytes);
        }
    }
}
