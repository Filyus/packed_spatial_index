//! `search_heaviest`: top-k by the aggregate scalar against a brute-force sort,
//! on every frontend that carries the `AGGR` chunk.

use std::ops::ControlFlow;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use packed_spatial_index::{
    Box2D, Box3D, HalfSpace2D, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView,
    Overlaps2D, Overlaps3D,
};

fn random_boxes_2d(rng: &mut StdRng, count: usize) -> Vec<Box2D> {
    (0..count)
        .map(|_| {
            let (x, y): (f64, f64) = (rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0));
            let (w, h): (f64, f64) = (rng.random_range(0.0..5.0), rng.random_range(0.0..5.0));
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn random_boxes_3d(rng: &mut StdRng, count: usize) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let (x, y, z): (f64, f64, f64) = (
                rng.random_range(-50.0..50.0),
                rng.random_range(-50.0..50.0),
                rng.random_range(-50.0..50.0),
            );
            let side: f64 = rng.random_range(0.0..5.0);
            Box3D::new(x, y, z, x + side, y + side, z + side)
        })
        .collect()
}

fn random_window_2d(rng: &mut StdRng) -> Box2D {
    let (x, y): (f64, f64) = (rng.random_range(-60.0..60.0), rng.random_range(-60.0..60.0));
    let side: f64 = rng.random_range(0.0..120.0);
    Box2D::new(x, y, x + side, y + side)
}

fn random_window_3d(rng: &mut StdRng) -> Box3D {
    let (x, y, z): (f64, f64, f64) = (
        rng.random_range(-60.0..60.0),
        rng.random_range(-60.0..60.0),
        rng.random_range(-60.0..60.0),
    );
    let side: f64 = rng.random_range(0.0..120.0);
    Box3D::new(x, y, z, x + side, y + side, z + side)
}

/// Weights drawn from a handful of values, so ties are everywhere; a few NaN
/// and a signed zero ride along.
fn tied_weights(rng: &mut StdRng, count: usize) -> Vec<f64> {
    (0..count)
        .map(|_| match rng.random_range(0..40) {
            0 => f64::NAN,
            1 => -0.0,
            2 => 0.0,
            n => f64::from(n % 6) - 2.0,
        })
        .collect()
}

/// The oracle: every hit, sorted heaviest first, ties by index, NaN dropped.
fn oracle(hits: impl Iterator<Item = usize>, weights: &[f64], k: usize) -> Vec<usize> {
    let mut hits: Vec<usize> = hits.filter(|&i| !weights[i].is_nan()).collect();
    hits.sort_by(|&a, &b| weights[b].partial_cmp(&weights[a]).unwrap().then(a.cmp(&b)));
    hits.truncate(k);
    hits
}

fn oracle_2d(items: &[Box2D], weights: &[f64], region: &impl Overlaps2D, k: usize) -> Vec<usize> {
    oracle(
        (0..items.len()).filter(|&i| region.overlaps_box(items[i])),
        weights,
        k,
    )
}

fn oracle_3d(items: &[Box3D], weights: &[f64], region: &impl Overlaps3D, k: usize) -> Vec<usize> {
    oracle(
        (0..items.len()).filter(|&i| region.overlaps_box(items[i])),
        weights,
        k,
    )
}

#[test]
fn matches_the_sorted_oracle_2d_on_every_frontend() {
    let mut rng = StdRng::seed_from_u64(211);
    for (count, node_size) in [(1, 4), (7, 2), (300, 4), (2_000, 16)] {
        let items = random_boxes_2d(&mut rng, count);
        let weights = tied_weights(&mut rng, count);
        let mut builder = Index2DBuilder::new(count).node_size(node_size);
        for &b in &items {
            builder.add(b);
        }
        let index = builder.aggregate_scalar(&weights).finish().unwrap();
        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd = packed_spatial_index::SimdIndex2D::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd_view = packed_spatial_index::SimdIndex2DView::from_bytes(&bytes).unwrap();

        for _ in 0..40 {
            let window = random_window_2d(&mut rng);
            for k in [1, 3, 10, count + 1] {
                let want = oracle_2d(&items, &weights, &window, k);
                assert_eq!(index.search_heaviest(window, k), Some(want.clone()));
                assert_eq!(view.search_heaviest(window, k), Some(want.clone()));
                #[cfg(feature = "simd")]
                {
                    assert_eq!(simd.search_heaviest(window, k), Some(want.clone()));
                    assert_eq!(simd_view.search_heaviest(window, k), Some(want.clone()));
                }
            }
        }
        // A region that is not a box.
        let half = HalfSpace2D::new(1.0, 0.5, 3.0);
        assert_eq!(
            index.search_heaviest(half, 25),
            Some(oracle_2d(&items, &weights, &half, 25))
        );
    }
}

#[test]
fn matches_the_sorted_oracle_3d_on_every_frontend() {
    let mut rng = StdRng::seed_from_u64(311);
    for (count, node_size) in [(5, 2), (1_500, 8)] {
        let items = random_boxes_3d(&mut rng, count);
        let weights = tied_weights(&mut rng, count);
        let mut builder = Index3DBuilder::new(count).node_size(node_size);
        for &b in &items {
            builder.add(b);
        }
        let index = builder.aggregate_scalar(&weights).finish().unwrap();
        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd = packed_spatial_index::SimdIndex3D::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd_view = packed_spatial_index::SimdIndex3DView::from_bytes(&bytes).unwrap();

        for _ in 0..30 {
            let window = random_window_3d(&mut rng);
            for k in [1, 4, 20] {
                let want = oracle_3d(&items, &weights, &window, k);
                assert_eq!(index.search_heaviest(window, k), Some(want.clone()));
                assert_eq!(view.search_heaviest(window, k), Some(want.clone()));
                #[cfg(feature = "simd")]
                {
                    assert_eq!(simd.search_heaviest(window, k), Some(want.clone()));
                    assert_eq!(simd_view.search_heaviest(window, k), Some(want.clone()));
                }
            }
        }
    }
}

#[test]
fn k_is_a_prefix_of_k_plus_one_and_each_reports_weights() {
    let mut rng = StdRng::seed_from_u64(7);
    let items = random_boxes_2d(&mut rng, 500);
    let weights = tied_weights(&mut rng, 500);
    let mut builder = Index2DBuilder::new(items.len()).node_size(5);
    for &b in &items {
        builder.add(b);
    }
    let index = builder.aggregate_scalar(&weights).finish().unwrap();
    let window = Box2D::new(-30.0, -30.0, 30.0, 30.0);

    let all = index.search_heaviest(window, usize::MAX).unwrap();
    for k in 0..all.len() {
        assert_eq!(index.search_heaviest(window, k).unwrap(), all[..k]);
    }

    // The visitor sees the weights in nonincreasing order; a threshold stops it.
    let mut seen = Vec::new();
    let flow = index
        .search_heaviest_each(window, |id, w| {
            assert_eq!(w.to_bits(), weights[id].to_bits());
            if w < 1.0 {
                return ControlFlow::Break(id);
            }
            seen.push(id);
            ControlFlow::Continue(())
        })
        .unwrap();
    let above: Vec<usize> = all.iter().copied().filter(|&i| weights[i] >= 1.0).collect();
    assert_eq!(seen, above);
    assert_eq!(flow, ControlFlow::Break(all[above.len()]));
}

#[test]
fn none_without_a_scalar_column_and_edge_cases() {
    let mut builder = Index2DBuilder::new(2);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(2.0, 2.0, 3.0, 3.0));
    let window = Box2D::new(-9.0, -9.0, 9.0, 9.0);
    let bare = builder.finish().unwrap();
    assert_eq!(bare.search_heaviest(window, 1), None);
    assert_eq!(bare.search_heaviest(window, 0), None);

    let mut builder = Index2DBuilder::new(2);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(2.0, 2.0, 3.0, 3.0));
    let masked = builder.aggregate_mask(&[1, 2]).finish().unwrap();
    assert_eq!(masked.search_heaviest(window, 1), None);
    let flow: Option<ControlFlow<()>> = masked.search_heaviest_each(window, |_, _| unreachable!());
    assert_eq!(flow, None);

    let mut builder = Index2DBuilder::new(2);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(2.0, 2.0, 3.0, 3.0));
    let index = builder.aggregate_scalar(&[1.0, 2.0]).finish().unwrap();
    assert_eq!(index.search_heaviest(window, 0), Some(vec![]));
    assert_eq!(
        index.search_heaviest(Box2D::new(50.0, 50.0, 60.0, 60.0), 3),
        Some(vec![])
    );

    let empty = Index2DBuilder::new(0)
        .aggregate_scalar(&[])
        .finish()
        .unwrap();
    assert_eq!(empty.search_heaviest(window, 3), Some(vec![]));

    // Infinite weights order like any other.
    let mut builder = Index3DBuilder::new(3);
    for i in 0..3 {
        let x = f64::from(i);
        builder.add(Box3D::new(x, 0.0, 0.0, x + 0.5, 1.0, 1.0));
    }
    let index = builder
        .aggregate_scalar(&[f64::NEG_INFINITY, f64::INFINITY, 0.0])
        .finish()
        .unwrap();
    let all = Box3D::new(-1.0, -1.0, -1.0, 9.0, 9.0, 9.0);
    assert_eq!(index.search_heaviest(all, 3), Some(vec![1, 2, 0]));
}

/// Both forms behind the switch — the region collected and ranked, and the
/// best-first descent — against the oracle, on every frontend, for boxes and
/// regions that are not boxes, every `k` from none to more than there are.
#[test]
fn both_forms_match_the_oracle_on_every_frontend() {
    use packed_spatial_index::{Capsule2D, Capsule3D, ConvexPolygon2D, HalfSpace3D, Triangle2D};

    macro_rules! check {
        ($idx:expr, $region:expr, $want:expr, $k:expr) => {{
            let (region, want, k): (_, Vec<usize>, usize) = ($region, $want.clone(), $k);
            assert_eq!($idx.search_heaviest(region, k), Some(want.clone()));
            assert_eq!(
                $idx.search_heaviest_forced::<true, _>(region, k),
                Some(want.clone())
            );
            assert_eq!(
                $idx.search_heaviest_forced::<false, _>(region, k),
                Some(want.clone())
            );
            for collect in [false, true] {
                let mut seen = Vec::new();
                let mut visit = |id, _| {
                    seen.push(id);
                    if seen.len() == k {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                };
                let _ = if collect {
                    $idx.search_heaviest_each_forced::<true, _, _, _>(region, &mut visit)
                } else {
                    $idx.search_heaviest_each_forced::<false, _, _, _>(region, &mut visit)
                };
                if k > 0 {
                    assert_eq!(seen, want);
                }
            }
        }};
    }

    let mut rng = StdRng::seed_from_u64(213);
    for (count, node_size) in [(1, 4), (9, 2), (400, 4), (3_000, 16)] {
        let items = random_boxes_2d(&mut rng, count);
        let weights = tied_weights(&mut rng, count);
        let mut builder = Index2DBuilder::new(count).node_size(node_size);
        for &b in &items {
            builder.add(b);
        }
        let index = builder.aggregate_scalar(&weights).finish().unwrap();
        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd = packed_spatial_index::SimdIndex2D::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd_view = packed_spatial_index::SimdIndex2DView::from_bytes(&bytes).unwrap();
        for _ in 0..40 {
            let window = random_window_2d(&mut rng);
            let c = [window.min_x, window.min_y];
            let tri = Triangle2D::new(c, [window.max_x, window.min_y], [c[0], window.max_y]);
            let poly = ConvexPolygon2D::new(vec![
                c,
                [window.max_x, window.min_y],
                [window.max_x, window.max_y],
            ]);
            let capsule = Capsule2D::new(c, [window.max_x, window.max_y], 3.0);
            let half = HalfSpace2D::new(1.0, -0.5, rng.random_range(-60.0..60.0));
            for k in [0, 1, 3, 10, count + 1] {
                let want = oracle_2d(&items, &weights, &window, k);
                check!(index, window, want, k);
                check!(view, window, want, k);
                #[cfg(feature = "simd")]
                {
                    check!(simd, window, want, k);
                    check!(simd_view, window, want, k);
                }
                check!(index, tri, oracle_2d(&items, &weights, &tri, k), k);
                check!(view, &poly, oracle_2d(&items, &weights, &poly, k), k);
                check!(index, capsule, oracle_2d(&items, &weights, &capsule, k), k);
                check!(view, half, oracle_2d(&items, &weights, &half, k), k);
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(313);
    for (count, node_size) in [(5, 2), (2_000, 8)] {
        let items = random_boxes_3d(&mut rng, count);
        let weights = tied_weights(&mut rng, count);
        let mut builder = Index3DBuilder::new(count).node_size(node_size);
        for &b in &items {
            builder.add(b);
        }
        let index = builder.aggregate_scalar(&weights).finish().unwrap();
        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd = packed_spatial_index::SimdIndex3D::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd_view = packed_spatial_index::SimdIndex3DView::from_bytes(&bytes).unwrap();
        for _ in 0..30 {
            let window = random_window_3d(&mut rng);
            let capsule = Capsule3D::new(
                [window.min_x, window.min_y, window.min_z],
                [window.max_x, window.max_y, window.max_z],
                2.0,
            );
            let half = HalfSpace3D::new(0.5, 1.0, -1.0, rng.random_range(-60.0..60.0));
            for k in [0, 1, 4, 20, count + 1] {
                let want = oracle_3d(&items, &weights, &window, k);
                check!(index, window, want, k);
                check!(view, window, want, k);
                #[cfg(feature = "simd")]
                {
                    check!(simd, window, want, k);
                    check!(simd_view, window, want, k);
                }
                check!(index, capsule, oracle_3d(&items, &weights, &capsule, k), k);
                check!(view, half, oracle_3d(&items, &weights, &half, k), k);
            }
        }
    }
}
