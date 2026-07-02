//! Correctness for 2D ray traversal, cross-checked against brute force —
//! including axis-parallel rays that exercise the masked SIMD slab path.

use packed_spatial_index::{Box2D, Index2DBuilder, NeighborWorkspace, Point2D, Ray2D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const EPS: f64 = 1e-9;

fn boxes_2d(n: usize, seed: u64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let dx: f64 = rng.random_range(0.5..30.0);
            let dy: f64 = rng.random_range(0.5..30.0);
            Box2D::new(x, y, x + dx, y + dy)
        })
        .collect()
}

fn brute_closest(boxes: &[Box2D], ray: Ray2D) -> Option<f64> {
    let mut best: Option<f64> = None;
    for &b in boxes {
        if let Some(t) = ray.enter_t(b)
            && best.is_none_or(|bt| t < bt)
        {
            best = Some(t);
        }
    }
    best
}

fn brute_all_hits(boxes: &[Box2D], ray: Ray2D) -> Vec<usize> {
    let mut out: Vec<usize> = (0..boxes.len())
        .filter(|&i| ray.intersects_box(boxes[i]))
        .collect();
    out.sort_unstable();
    out
}

fn close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => (x - y).abs() <= EPS * y.abs().max(1.0),
        _ => false,
    }
}

fn random_ray(rng: &mut StdRng, i: usize) -> Ray2D {
    let ox: f64 = rng.random_range(-200.0..1_200.0);
    let oy: f64 = rng.random_range(-200.0..1_200.0);
    let mut dx: f64 = rng.random_range(-1.0..1.0);
    let mut dy: f64 = rng.random_range(-1.0..1.0);
    match i % 3 {
        1 => dy = 0.0,
        2 => dx = 0.0,
        _ => {}
    }
    if dx == 0.0 && dy == 0.0 {
        dx = 1.0;
    }
    Ray2D::new(Point2D::new(ox, oy), dx, dy, 2_500.0)
}

#[test]
fn raycast_closest_2d_matches_brute_including_axis_parallel() {
    let boxes = boxes_2d(3_000, 0x4A22);
    let mut builder = Index2DBuilder::new(boxes.len()).node_size(8);
    boxes.iter().for_each(|&b| builder.add(b));
    let aos = builder.finish().unwrap();

    let mut ws = NeighborWorkspace::new();
    let mut rng = StdRng::seed_from_u64(0x0DD2);
    for i in 0..4_000 {
        let ray = random_ray(&mut rng, i);
        let expected = brute_closest(&boxes, ray);
        let aos_hit = aos.raycast_closest_with(ray, &mut ws).map(|(_, t)| t);
        assert!(
            close(aos_hit, expected),
            "AoS 2D {aos_hit:?} vs {expected:?}"
        );
    }
}

#[test]
fn raycast_all_hits_2d_matches_brute() {
    let boxes = boxes_2d(2_000, 0x4A23);
    let mut builder = Index2DBuilder::new(boxes.len());
    boxes.iter().for_each(|&b| builder.add(b));
    let index = builder.finish().unwrap();

    let mut rng = StdRng::seed_from_u64(0x0DD6);
    let mut results = Vec::new();
    for i in 0..1_000 {
        let ray = random_ray(&mut rng, i);
        index.raycast_into(ray, &mut results);
        results.sort_unstable();
        assert_eq!(results, brute_all_hits(&boxes, ray));
    }
}

#[test]
fn raycast_closest_2d_includes_exact_max_distance_hit_and_rejects_invalid_rays() {
    let mut builder = Index2DBuilder::new(1);
    builder.add(Box2D::new(10.0, 0.0, 11.0, 1.0));
    let index = builder.finish().unwrap();

    let boundary = Ray2D::new(Point2D::new(0.0, 0.5), 1.0, 0.0, 10.0);
    assert_eq!(index.raycast_closest(boundary), Some((0, 10.0)));

    let nan_origin = Ray2D::new(Point2D::new(f64::NAN, 0.5), 1.0, 0.0, 20.0);
    assert!(index.raycast(nan_origin).is_empty());
    assert_eq!(index.raycast_closest(nan_origin), None);

    let inf_dir = Ray2D::new(Point2D::new(0.0, 0.5), f64::INFINITY, 0.0, 20.0);
    assert!(index.raycast(inf_dir).is_empty());
    assert_eq!(index.raycast_closest(inf_dir), None);
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;

    #[test]
    fn simd_raycast_closest_2d_matches_brute_including_axis_parallel() {
        let boxes = boxes_2d(3_000, 0x4A22);
        let mut builder = Index2DBuilder::new(boxes.len()).node_size(8);
        boxes.iter().for_each(|&b| builder.add(b));
        let simd = builder.finish_simd().unwrap();

        let mut ws = NeighborWorkspace::new();
        let mut rng = StdRng::seed_from_u64(0x0DD2);
        for i in 0..4_000 {
            let ray = random_ray(&mut rng, i);
            let expected = brute_closest(&boxes, ray);
            let simd_hit = simd.raycast_closest_with(ray, &mut ws).map(|(_, t)| t);
            assert!(
                close(simd_hit, expected),
                "SoA 2D {simd_hit:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn simd_raycast_all_hits_2d_matches_brute() {
        let boxes = boxes_2d(1_500, 0x4A24);
        let mut builder = Index2DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| builder.add(b));
        let simd = builder.finish_simd().unwrap();

        let mut rng = StdRng::seed_from_u64(0x0DD7);
        let mut results = Vec::new();
        for i in 0..800 {
            let ray = random_ray(&mut rng, i);
            simd.raycast_into(ray, &mut results);
            results.sort_unstable();
            assert_eq!(results, brute_all_hits(&boxes, ray));
        }
    }

    #[test]
    fn simd_raycast_closest_2d_ray_through_exact_face() {
        // A ray parallel to Y whose origin lies *exactly* on a box face: the
        // degenerate case that produces `0 * inf = NaN` in an unmasked slab.
        let boxes = [
            Box2D::new(0.0, 0.0, 10.0, 10.0),
            Box2D::new(50.0, 0.0, 60.0, 10.0),
        ];
        let mut builder = Index2DBuilder::new(boxes.len()).node_size(8);
        boxes.iter().for_each(|&b| builder.add(b));
        let simd = builder.finish_simd().unwrap();

        // origin.y == 0.0 == box.min_y, dir = +X, dir_y == 0.
        let ray = Ray2D::new(Point2D::new(-5.0, 0.0), 1.0, 0.0, 100.0);
        let mut ws = NeighborWorkspace::new();
        let hit = simd.raycast_closest_with(ray, &mut ws);
        assert_eq!(hit.map(|(_, t)| t), brute_closest(&boxes, ray));
        assert!((hit.unwrap().1 - 5.0).abs() <= EPS, "entry t should be 5.0");
    }

    #[test]
    fn simd_raycast_2d_subnormal_direction_matches_scalar() {
        let boxes = [Box2D::new(0.0, 0.0, 1.0, 1.0)];
        let mut scalar_builder = Index2DBuilder::new(boxes.len());
        let mut simd_builder = Index2DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| {
            scalar_builder.add(b);
            simd_builder.add(b);
        });
        let scalar = scalar_builder.finish().unwrap();
        let simd = simd_builder.finish_simd().unwrap();

        let ray = Ray2D::new(Point2D::new(0.5, -1.0), f64::from_bits(1), 1.0, 10.0);
        assert_eq!(simd.raycast(ray), scalar.raycast(ray));
        assert_eq!(simd.raycast_closest(ray), scalar.raycast_closest(ray));
    }
}

mod ordered_and_views {
    use super::*;
    use packed_spatial_index::Index2DView;
    use std::ops::ControlFlow;

    #[test]
    fn visit_raycast_2d_is_ordered_and_complete() {
        let boxes = boxes_2d(1_500, 0x5B01);
        let mut builder = Index2DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| builder.add(b));
        let index = builder.finish().unwrap();

        let mut rng = StdRng::seed_from_u64(0x5B02);
        for i in 0..500 {
            let ray = random_ray(&mut rng, i);
            let mut expected: Vec<(f64, usize)> = (0..boxes.len())
                .filter_map(|j| ray.enter_t(boxes[j]).map(|t| (t, j)))
                .collect();
            expected.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut visited: Vec<(f64, usize)> = Vec::new();
            let flow: ControlFlow<()> = index.visit_raycast(ray, |id, t| {
                visited.push((t, id));
                ControlFlow::Continue(())
            });
            assert_eq!(flow, ControlFlow::Continue(()));

            let visited_ts: Vec<f64> = visited.iter().map(|&(t, _)| t).collect();
            let expected_ts: Vec<f64> = expected.iter().map(|&(t, _)| t).collect();
            assert_eq!(visited_ts, expected_ts);
        }
    }

    #[test]
    fn view_raycast_2d_matches_owned() {
        let boxes = boxes_2d(1_200, 0x5B03);
        let mut builder = Index2DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| builder.add(b));
        let index = builder.finish().unwrap();
        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();

        let mut rng = StdRng::seed_from_u64(0x5B04);
        for i in 0..400 {
            let ray = random_ray(&mut rng, i);

            let mut owned_hits = index.raycast(ray);
            let mut view_hits = view.raycast(ray);
            owned_hits.sort_unstable();
            view_hits.sort_unstable();
            assert_eq!(owned_hits, view_hits);

            assert_eq!(index.raycast_closest(ray), view.raycast_closest(ray));
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_view_raycast_2d_matches_owned() {
        use packed_spatial_index::SimdIndex2DView;

        let boxes = boxes_2d(1_200, 0x5B05);
        let mut builder = Index2DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| builder.add(b));
        let simd = builder.finish_simd().unwrap();
        let bytes = simd.to_bytes();
        let view = SimdIndex2DView::from_bytes(&bytes).unwrap();

        let mut rng = StdRng::seed_from_u64(0x5B06);
        for i in 0..400 {
            let ray = random_ray(&mut rng, i);

            let mut owned_hits = simd.raycast(ray);
            let mut view_hits = view.raycast(ray);
            owned_hits.sort_unstable();
            view_hits.sort_unstable();
            assert_eq!(owned_hits, view_hits);

            let owned_closest = simd.raycast_closest(ray).map(|(_, t)| t);
            let view_closest = view.raycast_closest(ray).map(|(_, t)| t);
            assert!(close(owned_closest, view_closest));

            let mut simd_ts = Vec::new();
            let _: ControlFlow<()> = simd.visit_raycast(ray, |_, t| {
                simd_ts.push(t);
                ControlFlow::Continue(())
            });
            let mut view_ts = Vec::new();
            let _: ControlFlow<()> = view.visit_raycast(ray, |_, t| {
                view_ts.push(t);
                ControlFlow::Continue(())
            });
            assert_eq!(simd_ts, view_ts);
        }
    }
}
