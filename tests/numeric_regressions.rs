use packed_spatial_index::{
    Box2D, Box3D, Capsule2D, Capsule3D, Cone3D, ConvexPolygon2D, Point3D, Ray3D, Triangle3D,
};

#[test]
fn translated_convex_polygon_keeps_its_winding() {
    let origin = (1_u64 << 32) as f64;
    let a = [origin, origin];
    let b = [origin + 4.0, origin];
    let c = [origin, origin + 4.0];
    let inside = Box2D::new(origin + 1.0, origin + 1.0, origin + 1.0, origin + 1.0);
    let outside = Box2D::new(origin + 3.0, origin + 3.0, origin + 3.0, origin + 3.0);
    for vertices in [vec![a, b, c], vec![a, c, b]] {
        let polygon = ConvexPolygon2D::new(vertices);
        assert!(polygon.overlaps_box(inside));
        assert!(polygon.contains_box(inside));
        assert!(!polygon.overlaps_box(outside));
    }
}

#[test]
fn triangle_at_both_ray_limits_is_included() {
    let tri = Triangle3D::new([-1.0, -1.0, 0.1], [1.0, -1.0, 0.1], [0.0, 12.0, 0.1]);
    let ray = Ray3D::new(Point3D::new(0.0, 0.0, 0.0), 0.0, 0.0, 1.0, 0.1);
    for count in [1, 4] {
        let triangles = [tri; 4];
        let triangles = &triangles[..count];
        assert_eq!(ray.closest_triangle(triangles).unwrap().t, 0.1);
        let hit = ray
            .closest_triangle_watertight(triangles, 0.1, false)
            .unwrap();
        assert_eq!(hit.t, 0.1);
        assert_eq!(hit.index, 0);
        let shorter = Ray3D::new(ray.origin, 0.0, 0.0, 1.0, 0.1_f64.next_down());
        assert!(shorter.closest_triangle(triangles).is_none());
        let longer = Ray3D::new(ray.origin, 0.0, 0.0, 1.0, 0.2);
        assert!(
            longer
                .closest_triangle_watertight(triangles, 0.1_f64.next_up(), false)
                .is_none()
        );
    }

    let oblique = Triangle3D::new([-1.0, -1.0, 0.0], [1.0, -1.0, 1.0], [0.0, 2.0, 5.0]);
    let ray = Ray3D::new(Point3D::new(0.0, 0.0, 0.0), 0.0, 0.0, 1.0, 2.0);
    let hit = ray
        .closest_triangle_watertight(&[oblique; 4], 2.0, false)
        .unwrap();
    assert_eq!((hit.index, hit.t), (0, 2.0));
}

#[test]
fn huge_triangles_still_have_finite_hits_in_scalar_and_simd() {
    let ray = Ray3D::new(Point3D::new(0.0, 0.0, 0.0), 0.0, 0.0, 1.0, 10.0);
    let near = Triangle3D::new([-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]);
    let huge = Triangle3D::new(
        [-1e155, -1e155, 2.0],
        [1e155, -1e155, 2.0],
        [0.0, 1e155, 2.0],
    );
    let triangles = [near, huge, near, huge];
    for count in [2, 4] {
        let hit = ray.closest_triangle(&triangles[..count]).unwrap();
        assert_eq!((hit.index, hit.t), (0, 1.0));
    }
    for count in [1, 4] {
        let huge_only = [huge; 4];
        let hit = ray
            .closest_triangle_watertight(&huge_only[..count], 0.0, false)
            .unwrap();
        assert_eq!((hit.index, hit.t, hit.uv), (0, 2.0, [0.25, 0.5]));
    }
}

#[test]
fn large_capsule_distance_brackets_an_interior_minimum() {
    let s = 1e154_f64;
    let exact = (s * 0.5) * s;
    let cap2 = Capsule2D::new([-s, s], [s, -s], 0.0);
    let cap3 = Capsule3D::new([-s, s, 0.0], [s, -s, 0.0], 0.0);
    let bounds2 = cap2.segment_box_gap_squared_bounds(Box2D::new(0.0, s, 0.0, s));
    let bounds3 = cap3.segment_box_gap_squared_bounds(Box3D::new(0.0, s, 0.0, 0.0, s, 0.0));
    for (lo, hi) in [bounds2, bounds3] {
        assert!(lo <= exact && exact <= hi, "{lo:e} <= {exact:e} <= {hi:e}");
        // Both nonzero gaps are s/2, so the per-axis pad of 8 eps · s
        // allows 32 eps relative error in each squared bound before summing.
        assert!(hi - lo <= 72.0 * f64::EPSILON * exact);
    }
}

#[test]
fn cone_keeps_flat_slice_when_a_breakpoint_overflows() {
    for x in [-f64::from_bits(1), 0.0, f64::from_bits(1)] {
        let cone = Cone3D::try_new([0.0; 3], [x, 0.0, 1.0], 0.5, 10.0).unwrap();
        assert!(cone.overlaps_box(Box3D::new(1.0, -10.0, 5.0, 1.0, 10.0, 5.0)));
        assert!(!cone.overlaps_box(Box3D::new(3.0, -10.0, 5.0, 3.0, 10.0, 5.0)));
    }
}
