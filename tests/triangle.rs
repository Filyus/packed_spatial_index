//! Fixed-width triangle payload: round-trip through a view, by-id and zero-copy,
//! in both `f64` (`Triangle3D`) and compact `f32` (`Triangle3DF32`).

use packed_spatial_index::{
    Box3D, Index2D, Index2DView, Index3D, Index3DBuilder, Index3DView, Triangle2D, Triangle3,
    Triangle3D, Triangle3DF32,
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
