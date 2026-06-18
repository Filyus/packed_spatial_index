//! Region queries on the zero-copy byte views must match the owned indexes
//! (which are themselves checked against brute force), including the
//! contained-subtree fast path.

use packed_spatial_index::{
    Box2D, Box3D, ConvexPolygon2D, Frustum3D, Index2DBuilder, Index2DView, Index3DBuilder,
    Index3DView, Triangle2D,
};

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

fn build2d(boxes: &[Box2D]) -> Vec<u8> {
    let mut b = Index2DBuilder::new(boxes.len());
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap().to_bytes()
}

fn build3d(boxes: &[Box3D]) -> Vec<u8> {
    let mut b = Index3DBuilder::new(boxes.len());
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap().to_bytes()
}

#[test]
fn view_triangle_matches_owned() {
    let boxes = boxes2d(4000);
    let bytes = build2d(&boxes);
    let owned = {
        let mut b = Index2DBuilder::new(boxes.len());
        for bx in &boxes {
            b.add(*bx);
        }
        b.finish().unwrap()
    };
    let view = Index2DView::from_bytes(&bytes).unwrap();

    for tri in [
        Triangle2D::new([10.0, 10.0], [180.0, 30.0], [40.0, 170.0]),
        Triangle2D::new([20.0, 20.0], [190.0, 25.0], [100.0, 28.0]), // sliver
        Triangle2D::new([-500.0, -500.0], [900.0, -500.0], [-500.0, 900.0]), // contains all
    ] {
        let mut o = owned.search_triangle(tri);
        let mut v = view.search_triangle(tri);
        o.sort_unstable();
        v.sort_unstable();
        assert_eq!(v, o, "triangle {tri:?}");
        assert_eq!(view.any_triangle(tri), !v.is_empty());

        let mut buf = vec![usize::MAX; 2];
        view.search_triangle_into(tri, &mut buf);
        buf.sort_unstable();
        assert_eq!(buf, v);
    }
}

#[test]
fn view_polygon_matches_owned() {
    let boxes = boxes2d(4000);
    let bytes = build2d(&boxes);
    let owned = {
        let mut b = Index2DBuilder::new(boxes.len());
        for bx in &boxes {
            b.add(*bx);
        }
        b.finish().unwrap()
    };
    let view = Index2DView::from_bytes(&bytes).unwrap();

    let polys = [
        ConvexPolygon2D::new(vec![
            [10.0, 10.0],
            [190.0, 30.0],
            [170.0, 180.0],
            [40.0, 150.0],
        ]),
        ConvexPolygon2D::new(vec![
            [-500.0, -500.0],
            [900.0, -500.0],
            [900.0, 900.0],
            [-500.0, 900.0],
        ]), // contains all
    ];
    for poly in &polys {
        let mut o = owned.search_polygon(poly);
        let mut v = view.search_polygon(poly);
        o.sort_unstable();
        v.sort_unstable();
        assert_eq!(v, o);
        assert_eq!(view.any_polygon(poly), !v.is_empty());
    }
}

#[test]
fn view_frustum_matches_owned() {
    let boxes = boxes3d(4000);
    let bytes = build3d(&boxes);
    let owned = {
        let mut b = Index3DBuilder::new(boxes.len());
        for bx in &boxes {
            b.add(*bx);
        }
        b.finish().unwrap()
    };
    let view = Index3DView::from_bytes(&bytes).unwrap();

    // Axis-aligned box frustum [lo,hi]^3 and a contains-all one.
    let box_frustum = |lo: f64, hi: f64| {
        Frustum3D::from_planes([
            [1.0, 0.0, 0.0, -lo],
            [-1.0, 0.0, 0.0, hi],
            [0.0, 1.0, 0.0, -lo],
            [0.0, -1.0, 0.0, hi],
            [0.0, 0.0, 1.0, -lo],
            [0.0, 0.0, -1.0, hi],
        ])
    };
    for frustum in [box_frustum(20.0, 160.0), box_frustum(-1000.0, 1000.0)] {
        let mut o = owned.search_frustum(frustum);
        let mut v = view.search_frustum(frustum);
        o.sort_unstable();
        v.sort_unstable();
        assert_eq!(v, o);
        assert_eq!(view.any_frustum(frustum), !v.is_empty());

        let mut buf = vec![usize::MAX; 2];
        view.search_frustum_into(frustum, &mut buf);
        buf.sort_unstable();
        assert_eq!(buf, v);
    }
}

#[test]
fn view_region_empty_index() {
    let bytes2 = build2d(&[]);
    let v2 = Index2DView::from_bytes(&bytes2).unwrap();
    assert!(
        v2.search_triangle(Triangle2D::new([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]))
            .is_empty()
    );
    assert!(
        v2.search_polygon(&ConvexPolygon2D::new(vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]))
        .is_empty()
    );

    let bytes3 = build3d(&[]);
    let v3 = Index3DView::from_bytes(&bytes3).unwrap();
    let f = Frustum3D::from_planes([[1.0, 0.0, 0.0, 0.0]; 6]);
    assert!(v3.search_frustum(f).is_empty());
    assert!(!v3.any_frustum(f));
}
