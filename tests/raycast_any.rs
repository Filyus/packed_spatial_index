//! `raycast_any` answers `!raycast(ray).is_empty()` on every frontend, in both
//! forms of the scalar child test, and on the SIMD frontends through every slab
//! kernel of their depth-first descent.

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
        #[cfg(feature = "simd")]
        let simd_bytes = simd.to_bytes();
        #[cfg(feature = "simd")]
        let simd_view = packed_spatial_index::SimdIndex2DView::from_bytes(&simd_bytes).unwrap();
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
            {
                assert_eq!(simd.raycast_any(ray), expected, "simd {node_size} {ray:?}");
                assert_eq!(simd.raycast_any_kernel::<0>(ray), expected, "wide {ray:?}");
                assert_eq!(simd.raycast_any_kernel::<1>(ray), expected, "avx2 {ray:?}");
                assert_eq!(simd.raycast_any_queue(ray), expected, "queue {ray:?}");
                assert_eq!(simd_view.raycast_any(ray), expected, "simd view {ray:?}");
            }
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
        #[cfg(feature = "simd")]
        let simd_bytes = simd.to_bytes();
        #[cfg(feature = "simd")]
        let simd_view = packed_spatial_index::SimdIndex3DView::from_bytes(&simd_bytes).unwrap();
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
            {
                assert_eq!(simd.raycast_any(ray), expected, "simd {node_size} {ray:?}");
                assert_eq!(simd.raycast_any_kernel::<0>(ray), expected, "wide {ray:?}");
                assert_eq!(simd.raycast_any_kernel::<1>(ray), expected, "avx2 {ray:?}");
                assert_eq!(simd.raycast_any_queue(ray), expected, "queue {ray:?}");
                assert_eq!(simd_view.raycast_any(ray), expected, "simd view {ray:?}");
            }
        }
        assert!(some > 0 && some < hits, "{some} of {hits}");
    }
}

/// Boxes on an integer grid with gaps, so rays from grid coordinates run along
/// faces and start on edges, inside boxes and in the gaps between them.
#[cfg(feature = "simd")]
fn grid_3d() -> Vec<Box3D> {
    let mut boxes = Vec::new();
    for x in 0..9 {
        for y in 0..9 {
            for z in 0..9 {
                if (x + 2 * y + 3 * z) % 4 == 0 {
                    continue;
                }
                let (x, y, z) = (f64::from(x) * 3.0, f64::from(y) * 3.0, f64::from(z) * 3.0);
                boxes.push(Box3D::new(x, y, z, x + 2.0, y + 2.0, z + 2.0));
            }
        }
    }
    boxes
}

/// Rays that reach the edges of the new descent: axis-parallel (the `wide`
/// `select` path) along faces and through gaps, zero length, the point probe,
/// origins inside and on boxes, oblique rays both ways (the AVX-512 and AVX2
/// paths), and rays that can hit nothing.
#[cfg(feature = "simd")]
fn edge_rays_3d() -> Vec<Ray3D> {
    let mut rays = Vec::new();
    let dirs = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.3, -0.7, 0.2),
        (-0.5, 0.25, 0.8),
    ];
    for o in [-1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 13.0, 26.0, 27.0] {
        for &(dx, dy, dz) in &dirs {
            for len in [0.0, 0.4, 1.0, 5.0, 40.0] {
                rays.push(Ray3D::new(Point3D::new(o, 2.5, o), dx, dy, dz, len));
                rays.push(Ray3D::new(Point3D::new(2.0, o, 1.0), dx, dy, dz, len));
            }
        }
    }
    let o = Point3D::new(1.0, 1.0, 1.0);
    rays.push(Ray3D::new(o, 1.0, 0.0, 0.0, -1.0));
    rays.push(Ray3D::new(o, 1.0, 0.0, 0.0, f64::NAN));
    rays.push(Ray3D::new(o, f64::INFINITY, 0.0, 0.0, 10.0));
    rays.push(Ray3D::new(
        Point3D::new(f64::NAN, 1.0, 1.0),
        1.0,
        0.0,
        0.0,
        10.0,
    ));
    rays
}

#[cfg(feature = "simd")]
#[test]
fn simd_raycast_any_edges_in_3d() {
    use packed_spatial_index::SimdIndex3DView;
    let boxes = grid_3d();
    // Node sizes that leave remainders past the four- and eight-wide groups.
    for node_size in [2, 5, 8, 13, 16] {
        let build = || {
            let mut b = Index3DBuilder::new(boxes.len()).node_size(node_size);
            for &bx in &boxes {
                b.add(bx);
            }
            b
        };
        let owned = build().finish().unwrap();
        let simd = build().finish_simd().unwrap();
        let bytes = simd.to_bytes();
        let view = SimdIndex3DView::from_bytes(&bytes).unwrap();
        let (mut total, mut some) = (0, 0);
        for ray in edge_rays_3d() {
            let expected = !owned.raycast(ray).is_empty();
            total += 1;
            some += usize::from(expected);
            let at = format!("node {node_size} {ray:?}");
            assert_eq!(simd.raycast_any(ray), expected, "simd (avx512) {at}");
            assert_eq!(simd.raycast_any_kernel::<0>(ray), expected, "wide {at}");
            assert_eq!(simd.raycast_any_kernel::<1>(ray), expected, "avx2 {at}");
            assert_eq!(view.raycast_any(ray), expected, "view {at}");
        }
        assert!(some > 0 && some < total, "{some} of {total}");
    }

    let empty = Index3DBuilder::new(0).finish_simd().unwrap();
    let ray = Ray3D::new(Point3D::new(0.0, 0.0, 0.0), 1.0, 0.0, 0.0, 10.0);
    assert!(!empty.raycast_any(ray));
    assert!(!empty.raycast_any_kernel::<0>(ray));
}

#[cfg(feature = "simd")]
#[test]
fn simd_raycast_any_edges_in_2d() {
    use packed_spatial_index::SimdIndex2DView;
    let mut boxes = Vec::new();
    for x in 0..30 {
        for y in 0..30 {
            if (x + 2 * y) % 5 != 0 {
                let (x, y) = (f64::from(x) * 3.0, f64::from(y) * 3.0);
                boxes.push(Box2D::new(x, y, x + 2.0, y + 2.0));
            }
        }
    }
    let dirs = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.0, 0.0),
        (0.3, -0.7),
        (-0.6, 0.8),
    ];
    let mut rays = Vec::new();
    for o in [-1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 45.0, 87.0, 88.0] {
        for &(dx, dy) in &dirs {
            for len in [0.0, 0.4, 1.0, 5.0, 40.0] {
                rays.push(Ray2D::new(Point2D::new(o, 2.5), dx, dy, len));
                rays.push(Ray2D::new(Point2D::new(2.0, o), dx, dy, len));
            }
        }
    }
    let o = Point2D::new(1.0, 1.0);
    rays.push(Ray2D::new(o, 1.0, 0.0, -1.0));
    rays.push(Ray2D::new(o, 1.0, 0.0, f64::NAN));
    rays.push(Ray2D::new(o, f64::INFINITY, 0.0, 10.0));

    for node_size in [2, 5, 8, 13, 16] {
        let build = || {
            let mut b = Index2DBuilder::new(boxes.len()).node_size(node_size);
            for &bx in &boxes {
                b.add(bx);
            }
            b
        };
        let owned = build().finish().unwrap();
        let simd = build().finish_simd().unwrap();
        let bytes = simd.to_bytes();
        let view = SimdIndex2DView::from_bytes(&bytes).unwrap();
        let (mut total, mut some) = (0, 0);
        for &ray in &rays {
            let expected = !owned.raycast(ray).is_empty();
            total += 1;
            some += usize::from(expected);
            let at = format!("node {node_size} {ray:?}");
            assert_eq!(simd.raycast_any(ray), expected, "simd (avx512) {at}");
            assert_eq!(simd.raycast_any_kernel::<0>(ray), expected, "wide {at}");
            assert_eq!(simd.raycast_any_kernel::<1>(ray), expected, "avx2 {at}");
            assert_eq!(view.raycast_any(ray), expected, "view {at}");
        }
        assert!(some > 0 && some < total, "{some} of {total}");
    }

    let empty = Index2DBuilder::new(0).finish_simd().unwrap();
    let ray = Ray2D::new(Point2D::new(0.0, 0.0), 1.0, 0.0, 10.0);
    assert!(!empty.raycast_any(ray));
    assert!(!empty.raycast_any_kernel::<0>(ray));
}
