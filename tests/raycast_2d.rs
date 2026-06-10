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
}
