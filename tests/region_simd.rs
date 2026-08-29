//! Region-shape queries on the SIMD f64 frontends must answer exactly what the
//! owned f64 index answers — same boxes, same predicate, only a different layout.

#![cfg(feature = "simd")]

use std::ops::ControlFlow;

use packed_spatial_index::{
    Box2D, Box3D, ConvexPolygon2D, Frustum3D, Index2DBuilder, Index3DBuilder, SimdIndex2D,
    SimdIndex2DView, SimdIndex3D, SimdIndex3DView, Triangle2D,
};

const COUNTS: [usize; 6] = [0, 1, 5, 17, 64, 1000];
const NODE_SIZES: [usize; 2] = [4, 16];

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

fn simd2d(boxes: &[Box2D], node_size: usize) -> SimdIndex2D {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish_simd().unwrap()
}

fn simd3d(boxes: &[Box3D], node_size: usize) -> SimdIndex3D {
    let mut b = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish_simd().unwrap()
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

#[test]
fn simd_2d_region_matches_owned() {
    for n in COUNTS {
        let boxes = boxes2d(n);
        for node_size in NODE_SIZES {
            let simd = simd2d(&boxes, node_size);
            let owned = owned2d(&boxes, node_size);
            let poly = trapezoid();
            let tri = Triangle2D::new([20.0, 20.0], [180.0, 40.0], [60.0, 170.0]);
            let window = Box2D::new(20.0, 20.0, 150.0, 160.0);

            let expected_poly = sorted(owned.search(&poly));
            assert_eq!(
                sorted(simd.search_region(&poly)),
                expected_poly,
                "polygon: n={n} node_size={node_size}"
            );
            assert_eq!(simd.count_region(&poly), expected_poly.len());
            assert_eq!(simd.any_region(&poly), !expected_poly.is_empty());
            assert_eq!(
                simd.first_region(&poly).is_some(),
                !expected_poly.is_empty()
            );

            assert_eq!(sorted(simd.search_region(&tri)), sorted(owned.search(&tri)));
            // A Box2D is an Overlaps2D too, so the shape path must agree with the
            // SIMD kernel it sits next to.
            assert_eq!(
                sorted(simd.search_region(window)),
                sorted(simd.search(window))
            );
        }
    }
}

#[test]
fn simd_3d_region_matches_owned() {
    for n in COUNTS {
        let boxes = boxes3d(n);
        for node_size in NODE_SIZES {
            let simd = simd3d(&boxes, node_size);
            let owned = owned3d(&boxes, node_size);
            let frustum = box_frustum(30.0, 170.0);
            let expected = sorted(owned.search(&frustum));

            assert_eq!(
                sorted(simd.search_region(frustum)),
                expected,
                "n={n} node_size={node_size}"
            );
            assert_eq!(simd.count_region(frustum), expected.len());
            assert_eq!(simd.any_region(frustum), !expected.is_empty());

            let window = Box3D::new(20.0, 20.0, 20.0, 150.0, 160.0, 140.0);
            assert_eq!(
                sorted(simd.search_region(window)),
                sorted(simd.search(window))
            );
        }
    }
}

#[test]
fn simd_views_match_owned_indexes() {
    let boxes2 = boxes2d(1000);
    let simd2 = simd2d(&boxes2, 16);
    let bytes2 = simd2.to_bytes();
    let view2 = SimdIndex2DView::from_bytes(&bytes2).unwrap();
    let poly = trapezoid();
    assert_eq!(
        sorted(view2.search_region(&poly)),
        sorted(simd2.search_region(&poly))
    );
    assert_eq!(view2.count_region(&poly), simd2.count_region(&poly));

    let boxes3 = boxes3d(1000);
    let simd3 = simd3d(&boxes3, 16);
    let bytes3 = simd3.to_bytes();
    let view3 = SimdIndex3DView::from_bytes(&bytes3).unwrap();
    let frustum = box_frustum(30.0, 170.0);
    assert_eq!(
        sorted(view3.search_region(frustum)),
        sorted(simd3.search_region(frustum))
    );
    assert_eq!(view3.count_region(frustum), simd3.count_region(frustum));
}

#[test]
fn region_visitor_can_break_and_buffers_are_cleared() {
    let boxes = boxes3d(1000);
    let simd = simd3d(&boxes, 16);
    let frustum = box_frustum(30.0, 170.0);

    let mut seen = 0usize;
    let flow = simd.visit_region(frustum, |id| {
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
    simd.search_region_into(frustum, &mut buf);
    assert!(!buf.contains(&usize::MAX));
    assert_eq!(buf.len(), simd.count_region(frustum));
}

#[test]
fn region_on_empty_index() {
    let simd = simd3d(&[], 16);
    let frustum = box_frustum(30.0, 170.0);
    assert!(simd.search_region(frustum).is_empty());
    assert_eq!(simd.count_region(frustum), 0);
    assert!(!simd.any_region(frustum));
    assert_eq!(simd.first_region(frustum), None);
}

mod ordered {
    use super::*;
    use packed_spatial_index::{view_depth_2d, view_depth_3d};

    const EYE3: [f64; 3] = [-40.0, -30.0, -20.0];
    const DIR3: [f64; 3] = [0.6, 0.7, 0.4];

    /// Jittered so no two boxes share a view depth: the budgeted prefix then has
    /// exactly one right answer.
    fn jittered3d(n: usize) -> Vec<Box3D> {
        boxes3d(n)
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
            .collect()
    }

    #[test]
    fn ordered_matches_the_region_it_orders() {
        for n in COUNTS {
            let boxes = jittered3d(n);
            for node_size in NODE_SIZES {
                let simd = simd3d(&boxes, node_size);
                let frustum = box_frustum(30.0, 170.0);
                let key = |b| view_depth_3d(EYE3, DIR3, b);

                let hits = simd.search_ordered(frustum, key, usize::MAX, f64::INFINITY);
                assert_eq!(
                    sorted(hits.clone()),
                    sorted(simd.search_region(frustum)),
                    "n={n} node_size={node_size}"
                );

                let mut keys = Vec::new();
                let _: ControlFlow<()> = simd.visit_ordered(frustum, key, f64::INFINITY, |_, k| {
                    keys.push(k);
                    ControlFlow::Continue(())
                });
                for pair in keys.windows(2) {
                    assert!(pair[0] <= pair[1], "keys out of order");
                }

                // The budget must be the nearest prefix of the brute-force order.
                let mut expected: Vec<(usize, f64)> = boxes
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| frustum.overlaps_box(**b))
                    .map(|(i, b)| (i, key(*b)))
                    .collect();
                expected.sort_by(|a, c| a.1.total_cmp(&c.1).then(a.0.cmp(&c.0)));
                for budget in [1usize, 7, 50] {
                    let want: Vec<usize> = expected.iter().take(budget).map(|(i, _)| *i).collect();
                    assert_eq!(
                        simd.search_ordered(frustum, key, budget, f64::INFINITY),
                        want,
                        "budget={budget} n={n} node_size={node_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn ordered_on_views_and_2d() {
        let boxes3 = jittered3d(1000);
        let simd3 = simd3d(&boxes3, 16);
        let bytes3 = simd3.to_bytes();
        let view3 = SimdIndex3DView::from_bytes(&bytes3).unwrap();
        let frustum = box_frustum(30.0, 170.0);
        let key3 = |b| view_depth_3d(EYE3, DIR3, b);
        for budget in [5usize, usize::MAX] {
            assert_eq!(
                view3.search_ordered(frustum, key3, budget, f64::INFINITY),
                simd3.search_ordered(frustum, key3, budget, f64::INFINITY),
                "budget={budget}"
            );
        }

        let boxes2 = boxes2d(1000);
        let simd2 = simd2d(&boxes2, 16);
        let bytes2 = simd2.to_bytes();
        let view2 = SimdIndex2DView::from_bytes(&bytes2).unwrap();
        let poly = trapezoid();
        let key2 = |b| view_depth_2d([-40.0, -30.0], [0.6, 0.8], b);
        assert_eq!(
            sorted(view2.search_ordered(&poly, key2, usize::MAX, f64::INFINITY)),
            sorted(simd2.search_region(&poly))
        );
    }
}
