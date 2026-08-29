//! Region-shape queries on the f32 frontends.
//!
//! They test the stored `f32` box widened back to `f64`. Boxes were rounded
//! outward at build time, so widening can only grow them: the answer must be a
//! superset of the f64 index's answer (no misses), and every extra must be a box
//! that genuinely grazes the region — checked by re-testing the predicate against
//! the original box inflated by a margin far larger than `f32` spacing at these
//! coordinates.

#![cfg(feature = "f32-storage")]

use std::ops::ControlFlow;

use packed_spatial_index::{
    Box2D, Box3D, ConvexPolygon2D, Frustum3D, Index2DBuilder, Index2DF32, Index3DBuilder,
    Index3DF32,
};

const COUNTS: [usize; 6] = [0, 1, 5, 17, 64, 1000];
const NODE_SIZES: [usize; 2] = [4, 16];
/// Coordinates run to ~200, where consecutive `f32` are ~1.5e-5 apart. A 1e-3
/// margin swallows the rounding many times over while still rejecting a box that
/// has no business being in the answer.
const MARGIN: f64 = 1e-3;

fn boxes2d(n: usize) -> Vec<Box2D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            let w = 0.2 + ((i * 13) % 5) as f64;
            let h = 0.2 + ((i * 17) % 5) as f64;
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn boxes3d(n: usize) -> Vec<Box3D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            let z = ((i * 5077) % 983) as f64 / 983.0 * 200.0;
            Box3D::new(x, y, z, x + 1.0, y + 1.5, z + 0.8)
        })
        .collect()
}

fn inflate2(b: Box2D) -> Box2D {
    Box2D::new(
        b.min_x - MARGIN,
        b.min_y - MARGIN,
        b.max_x + MARGIN,
        b.max_y + MARGIN,
    )
}

fn inflate3(b: Box3D) -> Box3D {
    Box3D::new(
        b.min_x - MARGIN,
        b.min_y - MARGIN,
        b.min_z - MARGIN,
        b.max_x + MARGIN,
        b.max_y + MARGIN,
        b.max_z + MARGIN,
    )
}

fn box_frustum(lo: f64, hi: f64) -> Frustum3D {
    Frustum3D::from_planes([
        [1.0, 0.0, 0.0, -lo],
        [-1.0, 0.0, 0.0, hi],
        [0.0, 1.0, 0.0, -lo],
        [0.0, -1.0, 0.0, hi],
        [0.0, 0.0, 1.0, -lo],
        [0.0, 0.0, -1.0, hi],
    ])
}

fn trapezoid() -> ConvexPolygon2D {
    ConvexPolygon2D::new(vec![
        [10.0, 10.0],
        [190.0, 30.0],
        [170.0, 180.0],
        [40.0, 150.0],
    ])
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

fn f32_2d(boxes: &[Box2D], node_size: usize) -> Index2DF32 {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish_f32().unwrap()
}

fn f32_3d(boxes: &[Box3D], node_size: usize) -> Index3DF32 {
    let mut b = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish_f32().unwrap()
}

fn owned2d(boxes: &[Box2D], node_size: usize) -> packed_spatial_index::Index2D {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap()
}

fn owned3d(boxes: &[Box3D], node_size: usize) -> packed_spatial_index::Index3D {
    let mut b = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap()
}

/// The two halves of the conservative-superset contract.
fn assert_conservative<T: Copy>(
    hits: &[usize],
    exact: &[usize],
    boxes: &[T],
    accepts: impl Fn(T) -> bool,
    label: &str,
) {
    for id in exact {
        assert!(hits.contains(id), "{label}: missed item {id}");
    }
    for &id in hits {
        assert!(
            accepts(boxes[id]),
            "{label}: item {id} is not even a near-boundary hit"
        );
    }
}

#[test]
fn f32_2d_region_is_a_conservative_superset() {
    for n in COUNTS {
        let boxes = boxes2d(n);
        for node_size in NODE_SIZES {
            let compact = f32_2d(&boxes, node_size);
            let owned = owned2d(&boxes, node_size);
            let poly = trapezoid();

            let hits = sorted(compact.search_region(&poly));
            let exact = sorted(owned.search(&poly));
            assert_conservative(
                &hits,
                &exact,
                &boxes,
                |b| poly.overlaps_box(inflate2(b)),
                &format!("polygon n={n} node_size={node_size}"),
            );
            assert_eq!(compact.count_region(&poly), hits.len());
            assert_eq!(compact.any_region(&poly), !hits.is_empty());
            assert_eq!(compact.first_region(&poly).is_some(), !hits.is_empty());

            // A Box2D region must agree with the frontend's own rounded `search`.
            let window = Box2D::new(20.0, 20.0, 150.0, 160.0);
            assert_eq!(
                sorted(compact.search_region(window)),
                sorted(compact.search(window))
            );
        }
    }
}

#[test]
fn f32_3d_region_is_a_conservative_superset() {
    for n in COUNTS {
        let boxes = boxes3d(n);
        for node_size in NODE_SIZES {
            let compact = f32_3d(&boxes, node_size);
            let owned = owned3d(&boxes, node_size);
            let frustum = box_frustum(30.0, 170.0);

            let hits = sorted(compact.search_region(frustum));
            let exact = sorted(owned.search(&frustum));
            assert_conservative(
                &hits,
                &exact,
                &boxes,
                |b| frustum.overlaps_box(inflate3(b)),
                &format!("frustum n={n} node_size={node_size}"),
            );
            assert_eq!(compact.count_region(frustum), hits.len());

            let window = Box3D::new(20.0, 20.0, 20.0, 150.0, 160.0, 140.0);
            assert_eq!(
                sorted(compact.search_region(window)),
                sorted(compact.search(window))
            );
        }
    }
}

#[test]
fn region_visitor_can_break_and_buffers_are_cleared() {
    let boxes = boxes3d(1000);
    let compact = f32_3d(&boxes, 16);
    let frustum = box_frustum(30.0, 170.0);

    let mut seen = 0usize;
    let flow = compact.visit_region(frustum, |id| {
        seen += 1;
        if seen == 3 {
            ControlFlow::Break(id)
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(seen, 3);
    assert!(flow.is_break());

    let mut buf = vec![usize::MAX; 7];
    compact.search_region_into(frustum, &mut buf);
    assert!(!buf.contains(&usize::MAX));
    assert_eq!(buf.len(), compact.count_region(frustum));
}

#[test]
fn region_on_empty_index() {
    let compact = f32_3d(&[], 16);
    let frustum = box_frustum(30.0, 170.0);
    assert!(compact.search_region(frustum).is_empty());
    assert_eq!(compact.count_region(frustum), 0);
    assert!(!compact.any_region(frustum));
    assert_eq!(compact.first_region(frustum), None);
}

#[cfg(feature = "simd")]
mod simd_f32 {
    use super::*;
    use packed_spatial_index::{
        SimdIndex2DF32, SimdIndex2DF32View, SimdIndex3DF32, SimdIndex3DF32View,
    };

    fn simd_f32_2d(boxes: &[Box2D], node_size: usize) -> SimdIndex2DF32 {
        let mut b = Index2DBuilder::new(boxes.len()).node_size(node_size);
        for bx in boxes {
            b.add(*bx);
        }
        b.finish_simd_f32().unwrap()
    }

    fn simd_f32_3d(boxes: &[Box3D], node_size: usize) -> SimdIndex3DF32 {
        let mut b = Index3DBuilder::new(boxes.len()).node_size(node_size);
        for bx in boxes {
            b.add(*bx);
        }
        b.finish_simd_f32().unwrap()
    }

    /// The scalar and SIMD f32 frontends hold identical columns, so their region
    /// answers must be identical too — not merely both conservative.
    #[test]
    fn simd_f32_matches_scalar_f32() {
        for n in COUNTS {
            let boxes2 = boxes2d(n);
            let boxes3 = boxes3d(n);
            for node_size in NODE_SIZES {
                let poly = trapezoid();
                assert_eq!(
                    sorted(simd_f32_2d(&boxes2, node_size).search_region(&poly)),
                    sorted(f32_2d(&boxes2, node_size).search_region(&poly)),
                    "2D n={n} node_size={node_size}"
                );
                let frustum = box_frustum(30.0, 170.0);
                assert_eq!(
                    sorted(simd_f32_3d(&boxes3, node_size).search_region(frustum)),
                    sorted(f32_3d(&boxes3, node_size).search_region(frustum)),
                    "3D n={n} node_size={node_size}"
                );
            }
        }
    }

    #[test]
    fn simd_f32_views_match_their_indexes() {
        let boxes2 = boxes2d(1000);
        let index2 = simd_f32_2d(&boxes2, 16);
        let bytes2 = index2.to_bytes();
        let view2 = SimdIndex2DF32View::from_bytes(&bytes2).unwrap();
        let poly = trapezoid();
        assert_eq!(
            sorted(view2.search_region(&poly)),
            sorted(index2.search_region(&poly))
        );

        let boxes3 = boxes3d(1000);
        let index3 = simd_f32_3d(&boxes3, 16);
        let bytes3 = index3.to_bytes();
        let view3 = SimdIndex3DF32View::from_bytes(&bytes3).unwrap();
        let frustum = box_frustum(30.0, 170.0);
        assert_eq!(
            sorted(view3.search_region(frustum)),
            sorted(index3.search_region(frustum))
        );
        assert_eq!(view3.count_region(frustum), index3.count_region(frustum));
    }
}

mod ordered {
    use super::*;
    use packed_spatial_index::view_depth_3d;

    const EYE3: [f64; 3] = [-40.0, -30.0, -20.0];
    const DIR3: [f64; 3] = [0.6, 0.7, 0.4];

    /// The ordered query must answer exactly the set its own `search_region`
    /// answers — the same conservative superset, only sequenced.
    #[test]
    fn ordered_matches_the_region_it_orders() {
        for n in COUNTS {
            let boxes = boxes3d(n);
            for node_size in NODE_SIZES {
                let compact = f32_3d(&boxes, node_size);
                let frustum = box_frustum(30.0, 170.0);
                let key = |b| view_depth_3d(EYE3, DIR3, b);

                assert_eq!(
                    sorted(compact.search_ordered(frustum, key, usize::MAX, f64::INFINITY)),
                    sorted(compact.search_region(frustum)),
                    "n={n} node_size={node_size}"
                );

                let mut keys = Vec::new();
                let _: ControlFlow<()> =
                    compact.visit_ordered(frustum, key, f64::INFINITY, |_, k| {
                        keys.push(k);
                        ControlFlow::Continue(())
                    });
                for pair in keys.windows(2) {
                    assert!(pair[0] <= pair[1], "keys out of order");
                }
            }
        }
    }

    /// Pins the ORDER, which set equality cannot: with the boxes jittered so no
    /// two share a depth, a budget must return the nearest prefix. The jitter
    /// (1e-3) is ~65x the f32 spacing here, so ordering by the stored widened box
    /// and by the original box agree.
    #[test]
    fn budget_returns_the_nearest_prefix() {
        let boxes: Vec<Box3D> = boxes3d(1000)
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                Box3D::new(
                    b.min_x + i as f64 * 0.001,
                    b.min_y,
                    b.min_z,
                    b.max_x + i as f64 * 0.001,
                    b.max_y,
                    b.max_z,
                )
            })
            .collect();
        let compact = f32_3d(&boxes, 16);
        let frustum = box_frustum(30.0, 170.0);
        let key = |b| view_depth_3d(EYE3, DIR3, b);

        // The oracle is this frontend's own (conservative) answer, ordered.
        let mut expected: Vec<(usize, f64)> = compact
            .search_region(frustum)
            .into_iter()
            .map(|id| (id, key(boxes[id])))
            .collect();
        expected.sort_by(|a, c| a.1.total_cmp(&c.1).then(a.0.cmp(&c.0)));

        for budget in [1usize, 7, 50] {
            let want: Vec<usize> = expected.iter().take(budget).map(|(i, _)| *i).collect();
            assert_eq!(
                compact.search_ordered(frustum, key, budget, f64::INFINITY),
                want,
                "budget={budget}"
            );
        }
    }

    /// No exact hit may be lost by ordering: the widened box has a depth no
    /// larger than the true one, so the bound stays admissible.
    #[test]
    fn ordered_loses_no_exact_hit() {
        let boxes = boxes3d(1000);
        let compact = f32_3d(&boxes, 16);
        let frustum = box_frustum(30.0, 170.0);
        let exact = sorted(owned3d(&boxes, 16).search(&frustum));
        let hits = sorted(compact.search_ordered(
            frustum,
            |b| view_depth_3d(EYE3, DIR3, b),
            usize::MAX,
            f64::INFINITY,
        ));
        for id in &exact {
            assert!(hits.contains(id), "missed item {id}");
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_f32_ordered_matches_scalar_f32() {
        use packed_spatial_index::SimdIndex3DF32;
        let boxes = boxes3d(1000);
        let mut b = Index3DBuilder::new(boxes.len()).node_size(16);
        for bx in &boxes {
            b.add(*bx);
        }
        let simd: SimdIndex3DF32 = b.finish_simd_f32().unwrap();
        let scalar = f32_3d(&boxes, 16);
        let frustum = box_frustum(30.0, 170.0);
        let key = |b| view_depth_3d(EYE3, DIR3, b);
        for budget in [5usize, usize::MAX] {
            assert_eq!(
                simd.search_ordered(frustum, key, budget, f64::INFINITY),
                scalar.search_ordered(frustum, key, budget, f64::INFINITY),
                "budget={budget}"
            );
        }
    }
}
