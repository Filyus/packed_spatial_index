//! Correctness for ray traversal: closest hit (nearest box the ray *segment*
//! enters, by entry `t`) and all-hits raycast, cross-checked against brute
//! force. Both the AoS scalar path (`Index3D`) and the SoA/SIMD path
//! (`SimdIndex3D`, which dispatches to AVX-512 or masked `f64x4`) must agree —
//! including axis-parallel rays (a zero direction component), where the
//! multiply-only vector slab is not NaN-safe.

use packed_spatial_index::{Box3D, Index3DBuilder, NeighborWorkspace, Point3D, Ray3D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const EPS: f64 = 1e-9;

fn boxes_3d(n: usize, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let z: f64 = rng.random_range(0.0..1_000.0);
            let dx: f64 = rng.random_range(0.5..30.0);
            let dy: f64 = rng.random_range(0.5..30.0);
            let dz: f64 = rng.random_range(0.5..30.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn brute_closest(boxes: &[Box3D], ray: Ray3D) -> Option<f64> {
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

fn brute_all_hits(boxes: &[Box3D], ray: Ray3D) -> Vec<usize> {
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

fn random_ray(rng: &mut StdRng, i: usize) -> Ray3D {
    let ox: f64 = rng.random_range(-200.0..1_200.0);
    let oy: f64 = rng.random_range(-200.0..1_200.0);
    let oz: f64 = rng.random_range(-200.0..1_200.0);
    let mut dx: f64 = rng.random_range(-1.0..1.0);
    let mut dy: f64 = rng.random_range(-1.0..1.0);
    let mut dz: f64 = rng.random_range(-1.0..1.0);
    // Force a zero component on most rays to exercise the masked SIMD path.
    match i % 4 {
        1 => dx = 0.0,
        2 => dy = 0.0,
        3 => dz = 0.0,
        _ => {}
    }
    if dx == 0.0 && dy == 0.0 && dz == 0.0 {
        dx = 1.0;
    }
    Ray3D::new(Point3D::new(ox, oy, oz), dx, dy, dz, 2_500.0)
}

#[test]
fn raycast_closest_3d_matches_brute_including_axis_parallel() {
    let boxes = boxes_3d(3_000, 0x4A33);
    let mut builder = Index3DBuilder::new(boxes.len()).node_size(8);
    boxes.iter().for_each(|&b| builder.add(b));
    let aos = builder.finish().unwrap();

    let mut ws = NeighborWorkspace::new();
    let mut rng = StdRng::seed_from_u64(0x0DD3);

    for i in 0..4_000 {
        let ray = random_ray(&mut rng, i);
        let expected = brute_closest(&boxes, ray);
        let aos_hit = aos.raycast_closest_with(ray, &mut ws).map(|(_, t)| t);
        assert!(
            close(aos_hit, expected),
            "AoS 3D {aos_hit:?} vs {expected:?}"
        );
    }
}

#[test]
fn raycast_all_hits_3d_matches_brute() {
    let boxes = boxes_3d(2_000, 0x4A34);
    let mut builder = Index3DBuilder::new(boxes.len());
    boxes.iter().for_each(|&b| builder.add(b));
    let index = builder.finish().unwrap();

    let mut rng = StdRng::seed_from_u64(0x0DD4);
    let mut results = Vec::new();
    for i in 0..1_000 {
        let ray = random_ray(&mut rng, i);
        index.raycast_into(ray, &mut results);
        results.sort_unstable();
        assert_eq!(results, brute_all_hits(&boxes, ray));
    }
}

#[test]
fn raycast_3d_empty_and_degenerate_rays() {
    let index = Index3DBuilder::new(0).finish().unwrap();
    let ray = Ray3D::new(Point3D::new(0.0, 0.0, 0.0), 1.0, 0.0, 0.0, 10.0);
    assert!(index.raycast(ray).is_empty());
    assert_eq!(index.raycast_closest(ray), None);

    let mut builder = Index3DBuilder::new(1);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    let index = builder.finish().unwrap();
    // Negative and NaN max_distance never hit.
    let bad = Ray3D::new(Point3D::new(0.5, 0.5, 0.5), 1.0, 0.0, 0.0, -1.0);
    assert!(index.raycast(bad).is_empty());
    assert_eq!(index.raycast_closest(bad), None);
    let nan = Ray3D::new(Point3D::new(0.5, 0.5, 0.5), 1.0, 0.0, 0.0, f64::NAN);
    assert!(index.raycast(nan).is_empty());
    // Origin inside the box: entry t is 0.
    let inside = Ray3D::new(Point3D::new(0.5, 0.5, 0.5), 1.0, 0.0, 0.0, 10.0);
    assert_eq!(index.raycast_closest(inside), Some((0, 0.0)));
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;

    #[test]
    fn simd_raycast_closest_3d_matches_brute_including_axis_parallel() {
        let boxes = boxes_3d(3_000, 0x4A33);
        let mut builder = Index3DBuilder::new(boxes.len()).node_size(8);
        boxes.iter().for_each(|&b| builder.add(b));
        let simd = builder.finish_simd().unwrap();

        let mut ws = NeighborWorkspace::new();
        let mut rng = StdRng::seed_from_u64(0x0DD3);
        for i in 0..4_000 {
            let ray = random_ray(&mut rng, i);
            let expected = brute_closest(&boxes, ray);
            let simd_hit = simd.raycast_closest_with(ray, &mut ws).map(|(_, t)| t);
            assert!(
                close(simd_hit, expected),
                "SoA 3D {simd_hit:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn simd_raycast_all_hits_3d_matches_brute() {
        let boxes = boxes_3d(1_500, 0x4A35);
        let mut builder = Index3DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| builder.add(b));
        let simd = builder.finish_simd().unwrap();

        let mut rng = StdRng::seed_from_u64(0x0DD5);
        let mut results = Vec::new();
        for i in 0..800 {
            let ray = random_ray(&mut rng, i);
            simd.raycast_into(ray, &mut results);
            results.sort_unstable();
            assert_eq!(results, brute_all_hits(&boxes, ray));
        }
    }

    #[test]
    fn simd_raycast_closest_3d_ray_through_exact_face() {
        // A ray that is parallel to Y and Z and whose origin lies *exactly* on a box
        // face: the degenerate case that produces `0 * inf = NaN` in an unmasked slab.
        let boxes = [
            Box3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
            Box3D::new(50.0, 0.0, 0.0, 60.0, 10.0, 10.0),
        ];
        let mut builder = Index3DBuilder::new(boxes.len()).node_size(8);
        boxes.iter().for_each(|&b| builder.add(b));
        let simd = builder.finish_simd().unwrap();

        // origin.z == 0.0 == box.min_z, dir = +X, dir_y == dir_z == 0.
        let ray = Ray3D::new(Point3D::new(-5.0, 5.0, 0.0), 1.0, 0.0, 0.0, 100.0);
        let mut ws = NeighborWorkspace::new();
        let hit = simd.raycast_closest_with(ray, &mut ws);
        assert_eq!(hit.map(|(_, t)| t), brute_closest(&boxes, ray));
        assert!((hit.unwrap().1 - 5.0).abs() <= EPS, "entry t should be 5.0");
    }
}
