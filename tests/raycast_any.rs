//! `raycast_any` answers `!raycast(ray).is_empty()` on every frontend, in both
//! forms of the scalar child test.

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView, Point2D, Point3D,
    Ray2D, Ray3D,
};

const N: usize = 2_000;

fn rays_2d() -> Vec<Ray2D> {
    let mut rays = Vec::new();
    for i in 0..60 {
        let f = i as f64;
        let origin = Point2D::new(f * 17.0 % 1000.0, 1000.0 - f * 13.0 % 1000.0);
        // Short and long, oblique and axis-parallel, and one of zero length.
        for (dx, dy, len) in [
            (0.3, -0.7, 5.0),
            (0.3, -0.7, 800.0),
            (1.0, 0.0, 40.0),
            (0.0, 1.0, 0.0),
        ] {
            rays.push(Ray2D::new(origin, dx, dy, len));
        }
    }
    rays.push(Ray2D::new(Point2D::new(-50.0, -50.0), -1.0, -1.0, 100.0));
    rays
}

fn rays_3d() -> Vec<Ray3D> {
    let mut rays = Vec::new();
    for i in 0..60 {
        let f = i as f64;
        let origin = Point3D::new(
            f * 17.0 % 1000.0,
            1000.0 - f * 13.0 % 1000.0,
            f * 7.0 % 1000.0,
        );
        for (dx, dy, dz, len) in [
            (0.3, -0.7, 0.2, 5.0),
            (0.3, -0.7, 0.2, 800.0),
            (1.0, 0.0, 0.0, 40.0),
            (0.0, 0.0, 1.0, 0.0),
        ] {
            rays.push(Ray3D::new(origin, dx, dy, dz, len));
        }
    }
    rays.push(Ray3D::new(
        Point3D::new(-50.0, -50.0, -50.0),
        -1.0,
        -1.0,
        -1.0,
        100.0,
    ));
    rays
}

#[test]
fn raycast_any_matches_raycast_in_2d() {
    let boxes: Vec<Box2D> = (0..N)
        .map(|i| {
            let x = ((i * 7919) % 1000) as f64;
            let y = ((i * 104_729) % 1000) as f64;
            let s = 1.0 + (i % 11) as f64;
            Box2D::new(x, y, x + s, y + s)
        })
        .collect();
    for node_size in [4, 16, 128] {
        let build = || {
            let mut b = Index2DBuilder::new(N).node_size(node_size);
            for &bx in &boxes {
                b.add(bx);
            }
            b
        };
        let owned = build().finish().unwrap();
        let bytes = owned.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd = build().finish_simd().unwrap();
        let (mut hits, mut some) = (0, 0);
        for ray in rays_2d() {
            let expected = !owned.raycast(ray).is_empty();
            some += usize::from(expected);
            hits += 1;
            assert_eq!(
                owned.raycast_any(ray),
                expected,
                "owned {node_size} {ray:?}"
            );
            assert_eq!(
                owned.raycast_any_forced::<false>(ray),
                expected,
                "owned branch {ray:?}"
            );
            assert_eq!(view.raycast_any(ray), expected, "view {node_size} {ray:?}");
            assert_eq!(
                view.raycast_any_forced::<false>(ray),
                expected,
                "view branch {ray:?}"
            );
            #[cfg(feature = "simd")]
            assert_eq!(simd.raycast_any(ray), expected, "simd {node_size} {ray:?}");
        }
        // Both answers occur, or the test pins nothing.
        assert!(some > 0 && some < hits, "{some} of {hits}");
    }
}

#[test]
fn raycast_any_matches_raycast_in_3d() {
    let boxes: Vec<Box3D> = (0..N)
        .map(|i| {
            let x = ((i * 7919) % 1000) as f64;
            let y = ((i * 104_729) % 1000) as f64;
            let z = ((i * 1_299_709) % 1000) as f64;
            let s = 5.0 + (i % 23) as f64;
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect();
    for node_size in [4, 16, 128] {
        let build = || {
            let mut b = Index3DBuilder::new(N).node_size(node_size);
            for &bx in &boxes {
                b.add(bx);
            }
            b
        };
        let owned = build().finish().unwrap();
        let bytes = owned.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        #[cfg(feature = "simd")]
        let simd = build().finish_simd().unwrap();
        let (mut hits, mut some) = (0, 0);
        for ray in rays_3d() {
            let expected = !owned.raycast(ray).is_empty();
            some += usize::from(expected);
            hits += 1;
            assert_eq!(
                owned.raycast_any(ray),
                expected,
                "owned {node_size} {ray:?}"
            );
            assert_eq!(
                owned.raycast_any_forced::<false>(ray),
                expected,
                "owned branch {ray:?}"
            );
            assert_eq!(view.raycast_any(ray), expected, "view {node_size} {ray:?}");
            assert_eq!(
                view.raycast_any_forced::<false>(ray),
                expected,
                "view branch {ray:?}"
            );
            #[cfg(feature = "simd")]
            assert_eq!(simd.raycast_any(ray), expected, "simd {node_size} {ray:?}");
        }
        assert!(some > 0 && some < hits, "{some} of {hits}");
    }
}
