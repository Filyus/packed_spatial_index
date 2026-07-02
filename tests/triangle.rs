//! Fixed-width triangle payload: round-trip through a view, by-id and zero-copy,
//! in both `f64` (`Triangle3D`) and compact `f32` (`Triangle3DF32`).

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index2DView, Index3D, Index3DBuilder, Index3DView,
    Point3D, Ray3D, Triangle2, Triangle2D, Triangle3, Triangle3D, Triangle3DF32,
};

fn tri3(i: usize) -> Triangle3D {
    let v = i as f64;
    Triangle3D::new([v, v, v], [v + 1.0, v, v], [v, v + 1.0, v + 2.0])
}

#[test]
fn triangle3d_payload_round_trips() {
    let n = 500;
    let tris: Vec<Triangle3D> = (0..n).map(tri3).collect();

    // `from_triangles` builds the index over each triangle's bounding box, in
    // slice order, and must match a hand-rolled builder loop.
    let index = Index3D::from_triangles(&tris).unwrap();
    let mut manual = Index3DBuilder::new(n);
    for t in &tris {
        manual.add(t.aabb());
    }
    assert_eq!(index.to_bytes(), manual.finish().unwrap().to_bytes());
    let bytes = index.serialize().triangles(&tris).to_bytes().unwrap();

    let view = Index3DView::from_bytes(&bytes).unwrap();
    assert!(view.has_payload());

    // By-id access (always available) returns each item's triangle.
    for (id, t) in tris.iter().enumerate() {
        assert_eq!(view.triangle(id), Some(*t), "triangle {id}");
    }
    assert_eq!(view.triangle::<Triangle3D>(n), None); // out of range

    // Zero-copy typed slice (leaf order) when the buffer is aligned.
    if let Some(slice) = view.triangles::<Triangle3D>() {
        assert_eq!(slice.len(), n);
        let mut seen: Vec<Triangle3D> = slice.to_vec();
        seen.sort_by(|a, b| a.a[0].partial_cmp(&b.a[0]).unwrap());
        let mut want = tris.clone();
        want.sort_by(|a, b| a.a[0].partial_cmp(&b.a[0]).unwrap());
        assert_eq!(seen, want);
    }

    // A spatial query maps results back to triangles via their bounding boxes.
    let q = Box3D::new(9.0, 9.0, 9.0, 12.0, 12.0, 20.0);
    for id in index.search(q) {
        let t = view.triangle::<Triangle3D>(id).unwrap();
        assert!(t.aabb().overlaps(q));
    }
}

#[test]
fn triangle3d_f32_is_compact_and_round_trips() {
    let n = 400;
    let tris: Vec<Triangle3DF32> = (0..n)
        .map(|i| {
            let v = i as f32;
            Triangle3DF32::new([v, v, v], [v + 1.0, v, v], [v, v + 1.0, v + 2.0])
        })
        .collect();
    let index = Index3D::from_triangles(&tris).unwrap();
    let f32_bytes = index.serialize().triangles(&tris).to_bytes().unwrap();

    // The f32 records are half the size of the equivalent f64 ones.
    let tris64: Vec<Triangle3D> = (0..n)
        .map(|i| {
            let v = i as f64;
            Triangle3D::new([v, v, v], [v + 1.0, v, v], [v, v + 1.0, v + 2.0])
        })
        .collect();
    let f64_bytes = Index3D::from_triangles(&tris64)
        .unwrap()
        .serialize()
        .triangles(&tris64)
        .to_bytes()
        .unwrap();
    assert!(f32_bytes.len() < f64_bytes.len());

    let view = Index3DView::from_bytes(&f32_bytes).unwrap();
    for (id, t) in tris.iter().enumerate() {
        assert_eq!(view.triangle(id), Some(*t));
    }
    // Asking for the wrong record type (f64) on an f32 payload yields None.
    assert_eq!(view.triangle::<Triangle3D>(0), None);
}

#[test]
fn triangle2d_payload_round_trips() {
    let n = 300;
    let tris: Vec<Triangle2D> = (0..n)
        .map(|i| {
            let v = i as f64;
            Triangle2D::new([v, v], [v + 1.0, v], [v, v + 2.0])
        })
        .collect();

    let index = Index2D::from_triangles(&tris).unwrap();
    let bytes = index.serialize().triangles(&tris).to_bytes().unwrap();

    let view = Index2DView::from_bytes(&bytes).unwrap();
    for (id, t) in tris.iter().enumerate() {
        assert_eq!(view.triangle(id), Some(*t));
    }
    if let Some(slice) = view.triangles::<Triangle2D>() {
        assert_eq!(slice.len(), n);
    }
    // A plain byte payload is not exposed as triangles.
    let plain = index.serialize().to_bytes().unwrap();
    assert_eq!(
        Index2DView::from_bytes(&plain)
            .unwrap()
            .triangle::<Triangle2D>(0),
        None
    );
}

#[test]
fn triangle_aabb_bounds_vertices() {
    let t = Triangle3D::new([1.0, 5.0, -2.0], [3.0, 2.0, 4.0], [-1.0, 0.0, 1.0]);
    let bb = t.aabb();
    assert_eq!(bb, Box3D::new(-1.0, 0.0, -2.0, 3.0, 5.0, 4.0));
}

#[test]
fn closest_triangle_accepts_tiny_valid_triangles() {
    let tiny = 1e-15;
    let tri = Triangle3D::new([0.0, 0.0, 1.0], [tiny, 0.0, 1.0], [0.0, tiny, 1.0]);
    let ray = Ray3D::new(
        Point3D::new(tiny * 0.25, tiny * 0.25, 0.0),
        0.0,
        0.0,
        1.0,
        2.0,
    );
    let hit = ray.closest_triangle(&[tri]).unwrap();
    assert_eq!(hit.index, 0);
    assert!((hit.t - 1.0).abs() < 1e-12, "t={}", hit.t);

    let tiny = 1e-10f32;
    let tri = Triangle3DF32::new([0.0, 0.0, 1.0], [tiny, 0.0, 1.0], [0.0, tiny, 1.0]);
    let ray = Ray3D::new(
        Point3D::new(tiny as f64 * 0.25, tiny as f64 * 0.25, 0.0),
        0.0,
        0.0,
        1.0,
        2.0,
    );
    let hit = ray.closest_triangle(&[tri]).unwrap();
    assert_eq!(hit.index, 0);
    assert!((hit.t - 1.0).abs() < 1e-5, "t={}", hit.t);
}

/// A scattered field of small boxes and an assortment of query triangles (fat,
/// thin, axis-aligned, degenerate). `triangle_search` must equal the brute-force
/// triangle-AABB overlap, and stay a subset of the bounding-box query.
#[test]
fn triangle_search_matches_brute_force() {
    let extent = 200.0;
    let n = 4000;
    let boxes: Vec<Box2D> = (0..n)
        .map(|i| {
            // Cheap deterministic spread without an rng dependency.
            let x = ((i * 7919) % 977) as f64 / 977.0 * extent;
            let y = ((i * 6121) % 991) as f64 / 991.0 * extent;
            let w = 0.2 + ((i * 13) % 5) as f64;
            let h = 0.2 + ((i * 17) % 5) as f64;
            Box2D::new(x, y, x + w, y + h)
        })
        .collect();

    let mut builder = Index2DBuilder::new(n);
    for b in &boxes {
        builder.add(*b);
    }
    let index = builder.finish().unwrap();

    let queries = [
        // Fat triangle covering a big region.
        Triangle2D::new([10.0, 10.0], [180.0, 30.0], [40.0, 170.0]),
        // Thin sliver.
        Triangle2D::new([20.0, 20.0], [190.0, 25.0], [100.0, 28.0]),
        // Right triangle hugging an axis.
        Triangle2D::new([0.0, 0.0], [150.0, 0.0], [0.0, 150.0]),
        // Tiny triangle that may catch nothing.
        Triangle2D::new([50.0, 50.0], [51.0, 50.0], [50.0, 51.0]),
        // Degenerate (collinear) triangle: zero area, must match brute force
        // (which is "the filled area" — empty).
        Triangle2D::new([0.0, 0.0], [10.0, 10.0], [20.0, 20.0]),
        // Reversed winding (clockwise) should behave the same.
        Triangle2D::new([40.0, 170.0], [180.0, 30.0], [10.0, 10.0]),
    ];

    for tri in queries {
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| tri.overlaps_box(**b))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let mut got = index.search(&tri);
        got.sort_unstable();
        assert_eq!(got, expected, "triangle {tri:?}");

        // Tight: a subset of the bounding-box query.
        let mut bbox_hits = index.search(tri.aabb());
        bbox_hits.sort_unstable();
        assert!(
            got.iter().all(|i| bbox_hits.binary_search(i).is_ok()),
            "triangle result must be a subset of the bbox result"
        );

        // `any` agrees with `triangle_search` being non-empty.
        assert_eq!(index.any(&tri), !got.is_empty(), "any {tri:?}");

        // `search_into` matches `triangle_search`.
        let mut buf = vec![usize::MAX; 3];
        index.search_into(&tri, &mut buf);
        buf.sort_unstable();
        assert_eq!(buf, got);
    }
}

/// The contained-subtree fast path must not change results: a triangle large
/// enough to swallow whole subtrees returns exactly the brute-force set.
#[test]
fn triangle_search_contained_fast_path_is_correct() {
    let n = 2000;
    let boxes: Vec<Box2D> = (0..n)
        .map(|i| {
            let x = (i % 50) as f64 * 2.0;
            let y = (i / 50) as f64 * 2.0;
            Box2D::new(x, y, x + 0.5, y + 0.5)
        })
        .collect();
    let mut builder = Index2DBuilder::new(n);
    for b in &boxes {
        builder.add(*b);
    }
    let index = builder.finish().unwrap();

    // Huge triangle containing the entire point field — exercises the root and
    // subtree fast-accept paths.
    let tri = Triangle2D::new([-1000.0, -1000.0], [3000.0, -1000.0], [-1000.0, 3000.0]);
    let mut got = index.search(&tri);
    got.sort_unstable();
    let all: Vec<usize> = (0..n).filter(|&i| tri.overlaps_box(boxes[i])).collect();
    assert_eq!(got, all);
}

#[test]
fn triangle_search_empty_index() {
    let index = Index2DBuilder::new(0).finish().unwrap();
    let tri = Triangle2D::new([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
    assert!(index.search(&tri).is_empty());
    assert!(!index.any(&tri));
}
