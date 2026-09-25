//! `closest_pairs` / `closest_pairs_to` (and their `_within` forms) against a
//! brute force over every pair, on every frontend that has them.

use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

type Pair = (usize, usize, f64);

fn random_boxes(rng: &mut StdRng, count: usize, extent: f64, max_size: f64) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..extent);
            let y: f64 = rng.random_range(0.0..extent);
            let z: f64 = rng.random_range(0.0..extent);
            let w: f64 = rng.random_range(0.0..max_size);
            let h: f64 = rng.random_range(0.0..max_size);
            let d: f64 = rng.random_range(0.0..max_size);
            Box3D::new(x, y, z, x + w, y + h, z + d)
        })
        .collect()
}

/// Unit boxes on an integer grid: distances are small integers and square
/// roots of them, so ties are everywhere, and duplicates overlap exactly.
fn grid_boxes(rng: &mut StdRng, count: usize, extent: u32) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let x = f64::from(rng.random_range(0..extent)) * 2.0;
            let y = f64::from(rng.random_range(0..extent)) * 2.0;
            let z = f64::from(rng.random_range(0..extent)) * 2.0;
            Box3D::new(x, y, z, x + 1.0, y + 1.0, z + 1.0)
        })
        .collect()
}

fn build(boxes: &[Box3D], node_size: usize) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for &b in boxes {
        builder.add(b);
    }
    builder.finish().unwrap()
}

fn distance(a: Box3D, b: Box3D) -> f64 {
    a.distance_squared_to_box(b).sqrt()
}

/// Every cross pair within `max_distance`, sorted by `(distance, i, j)`, cut
/// to `k`: the whole contract in five lines.
fn naive_pairs_to(a: &[Box3D], b: &[Box3D], k: usize, max_distance: f64) -> Vec<Pair> {
    let mut all = Vec::new();
    for (i, &box_a) in a.iter().enumerate() {
        for (j, &box_b) in b.iter().enumerate() {
            let d = distance(box_a, box_b);
            if d <= max_distance {
                all.push((i, j, d));
            }
        }
    }
    all.sort_by(|x, y| x.2.total_cmp(&y.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    all.truncate(k);
    all
}

/// Every unordered pair of distinct items as `(i, j)` with `i < j`.
fn naive_pairs(boxes: &[Box3D], k: usize, max_distance: f64) -> Vec<Pair> {
    let mut all = Vec::new();
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            let d = distance(boxes[i], boxes[j]);
            if d <= max_distance {
                all.push((i, j, d));
            }
        }
    }
    all.sort_by(|x, y| x.2.total_cmp(&y.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    all.truncate(k);
    all
}

/// Scenes: `(boxes_a, boxes_b)` covering sparse, overlapping and tied data.
fn scenes(seed: u64) -> Vec<(Vec<Box3D>, Vec<Box3D>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![
        (Vec::new(), Vec::new()),
        (Vec::new(), random_boxes(&mut rng, 5, 100.0, 4.0)),
        (random_boxes(&mut rng, 5, 100.0, 4.0), Vec::new()),
        (
            random_boxes(&mut rng, 1, 100.0, 4.0),
            random_boxes(&mut rng, 1, 100.0, 4.0),
        ),
        (
            random_boxes(&mut rng, 2, 100.0, 4.0),
            random_boxes(&mut rng, 3, 100.0, 4.0),
        ),
    ];
    for _ in 0..3 {
        out.push((
            random_boxes(&mut rng, 70, 100.0, 3.0),
            random_boxes(&mut rng, 55, 100.0, 3.0),
        ));
        // Dense enough that a good share of the pairs overlap at distance 0.
        out.push((
            random_boxes(&mut rng, 60, 100.0, 40.0),
            random_boxes(&mut rng, 45, 100.0, 40.0),
        ));
        out.push((grid_boxes(&mut rng, 80, 4), grid_boxes(&mut rng, 50, 4)));
    }
    out
}

const KS: [usize; 9] = [0, 1, 2, 3, 7, 20, 100, 1000, 100_000];

#[test]
fn closest_pairs_match_brute_force() {
    for (boxes, _) in scenes(5201) {
        for node_size in [2, 4, 16] {
            let index = build(&boxes, node_size);
            for k in KS {
                let want = naive_pairs(&boxes, k, f64::INFINITY);
                assert_eq!(index.closest_pairs(k), want, "n={} k={k}", boxes.len());
            }
        }
    }
}

#[test]
fn closest_pairs_to_match_brute_force() {
    for (boxes_a, boxes_b) in scenes(5202) {
        for node_size in [2, 4, 16] {
            let a = build(&boxes_a, node_size);
            let b = build(&boxes_b, node_size.max(3));
            for k in KS {
                let want = naive_pairs_to(&boxes_a, &boxes_b, k, f64::INFINITY);
                assert_eq!(a.closest_pairs_to(&b, k), want, "k={k}");
            }
        }
    }
}

#[test]
fn within_forms_match_brute_force() {
    for (boxes_a, boxes_b) in scenes(5203) {
        let a = build(&boxes_a, 4);
        let b = build(&boxes_b, 4);
        for max_distance in [0.0, 1.0, 2.0, 5.5, 1e9, f64::INFINITY] {
            for k in [1, 5, 50, 100_000] {
                assert_eq!(
                    a.closest_pairs_within(k, max_distance),
                    naive_pairs(&boxes_a, k, max_distance),
                    "k={k} max_distance={max_distance}"
                );
                assert_eq!(
                    a.closest_pairs_to_within(&b, k, max_distance),
                    naive_pairs_to(&boxes_a, &boxes_b, k, max_distance),
                    "k={k} max_distance={max_distance}"
                );
            }
        }
        for bad in [-1.0, f64::NAN] {
            assert!(a.closest_pairs_within(10, bad).is_empty());
            assert!(a.closest_pairs_to_within(&b, 10, bad).is_empty());
        }
    }
}

/// Growing `k` only appends, and `k = 1` is at `closest_pair`'s distance.
#[test]
fn k_is_a_prefix_of_k_plus_one() {
    for (boxes_a, boxes_b) in scenes(5204) {
        let a = build(&boxes_a, 4);
        let b = build(&boxes_b, 4);
        let all_self = a.closest_pairs(usize::MAX);
        let all_cross = a.closest_pairs_to(&b, usize::MAX);
        for k in 0..all_self.len().min(60) {
            assert_eq!(a.closest_pairs(k), all_self[..k]);
        }
        for k in 0..all_cross.len().min(60) {
            assert_eq!(a.closest_pairs_to(&b, k), all_cross[..k]);
        }
        assert_eq!(
            a.closest_pairs(1).first().map(|p| p.2),
            a.closest_pair().map(|p| p.2)
        );
        assert_eq!(
            a.closest_pairs_to(&b, 1).first().map(|p| p.2),
            a.closest_pair_to(&b).map(|p| p.2)
        );
    }
}

/// Where the nearest distance is unique, `k = 1` is `closest_pair` exactly.
#[test]
fn k_one_is_closest_pair_without_ties() {
    let mut rng = StdRng::seed_from_u64(5205);
    for _ in 0..20 {
        let boxes_a = random_boxes(&mut rng, 200, 100.0, 1.0);
        let boxes_b = random_boxes(&mut rng, 150, 100.0, 1.0);
        let a = build(&boxes_a, 16);
        let b = build(&boxes_b, 16);
        let pairs = a.closest_pairs(2);
        if pairs[0].2 < pairs[1].2 {
            let (i, j, d) = a.closest_pair().unwrap();
            assert_eq!(pairs[0], (i.min(j), i.max(j), d));
        }
        let pairs = a.closest_pairs_to(&b, 2);
        if pairs[0].2 < pairs[1].2 {
            assert_eq!(Some(pairs[0]), a.closest_pair_to(&b));
        }
    }
}

#[test]
fn view_matches_owned() {
    for (boxes_a, boxes_b) in scenes(5206) {
        let a = build(&boxes_a, 4);
        let b = build(&boxes_b, 4);
        let bytes_a = a.to_bytes();
        let bytes_b = b.to_bytes();
        let view_a = Index3DView::from_bytes(&bytes_a).unwrap();
        let view_b = Index3DView::from_bytes(&bytes_b).unwrap();
        for k in [1, 9, 1000] {
            assert_eq!(view_a.closest_pairs(k), a.closest_pairs(k));
            assert_eq!(
                view_a.closest_pairs_within(k, 3.0),
                a.closest_pairs_within(k, 3.0)
            );
            assert_eq!(
                view_a.closest_pairs_to(&view_b, k),
                a.closest_pairs_to(&b, k)
            );
            assert_eq!(
                view_a.closest_pairs_to_within(&view_b, k, 3.0),
                a.closest_pairs_to_within(&b, k, 3.0)
            );
        }
    }
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::{SimdIndex3D, SimdIndex3DView};

    fn build_simd(boxes: &[Box3D], node_size: usize) -> SimdIndex3D {
        let mut builder = Index3DBuilder::new(boxes.len()).node_size(node_size);
        for &b in boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    }

    #[test]
    fn simd_and_view_match_brute_force() {
        for (boxes_a, boxes_b) in scenes(5207) {
            let a = build_simd(&boxes_a, 4);
            let b = build_simd(&boxes_b, 4);
            let bytes_a = a.to_bytes();
            let bytes_b = b.to_bytes();
            let view_a = SimdIndex3DView::from_bytes(&bytes_a).unwrap();
            let view_b = SimdIndex3DView::from_bytes(&bytes_b).unwrap();
            for k in [1, 9, 1000] {
                let want = naive_pairs(&boxes_a, k, f64::INFINITY);
                assert_eq!(a.closest_pairs(k), want);
                assert_eq!(view_a.closest_pairs(k), want);
                let want = naive_pairs(&boxes_a, k, 3.0);
                assert_eq!(a.closest_pairs_within(k, 3.0), want);
                assert_eq!(view_a.closest_pairs_within(k, 3.0), want);
                let want = naive_pairs_to(&boxes_a, &boxes_b, k, f64::INFINITY);
                assert_eq!(a.closest_pairs_to(&b, k), want);
                assert_eq!(view_a.closest_pairs_to(&view_b, k), want);
                let want = naive_pairs_to(&boxes_a, &boxes_b, k, 3.0);
                assert_eq!(a.closest_pairs_to_within(&b, k, 3.0), want);
                assert_eq!(view_a.closest_pairs_to_within(&view_b, k, 3.0), want);
            }
        }
    }
}
