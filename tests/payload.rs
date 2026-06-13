//! Optional per-item payload: write, then read back via the zero-copy views and
//! confirm the index-only loaders tolerate payload files.

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView,
};

fn build_2d(n: usize) -> Index2D {
    let mut builder = Index2DBuilder::new(n).node_size(16);
    for i in 0..n {
        let v = i as f64;
        builder.add(Box2D::new(v, v, v + 1.0, v + 1.0));
    }
    builder.finish().unwrap()
}

#[test]
fn view_2d_payload_round_trip_and_search() {
    let n = 500;
    let index = build_2d(n);
    let payloads: Vec<Vec<u8>> = (0..n).map(|i| format!("blob-{i}").into_bytes()).collect();
    let bytes = index.to_bytes_with_payloads(&payloads).unwrap();

    let view = Index2DView::from_bytes(&bytes).unwrap();
    assert!(view.has_payload());

    // Search results address payloads directly.
    let hits = view.search(Box2D::new(0.0, 0.0, 10.5, 10.5));
    assert_eq!(hits, index.search(Box2D::new(0.0, 0.0, 10.5, 10.5)));
    for id in hits {
        assert_eq!(view.payload(id), Some(payloads[id].as_slice()));
    }

    assert_eq!(view.payload(0), Some(b"blob-0".as_slice()));
    assert_eq!(view.payload(n - 1), Some(payloads[n - 1].as_slice()));
    assert_eq!(view.payload(n), None); // out of range
}

#[test]
fn view_3d_payload_round_trip() {
    let n = 300;
    let mut builder = Index3DBuilder::new(n).node_size(16);
    for i in 0..n {
        let v = i as f64;
        builder.add(Box3D::new(v, v, v, v + 1.0, v + 1.0, v + 1.0));
    }
    let index = builder.finish().unwrap();
    let payloads: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; (i % 7) + 1]).collect();
    let bytes = index.to_bytes_with_payloads(&payloads).unwrap();

    let view = Index3DView::from_bytes(&bytes).unwrap();
    assert!(view.has_payload());
    for (id, want) in payloads.iter().enumerate() {
        assert_eq!(view.payload(id), Some(want.as_slice()));
    }
}

#[test]
fn index_only_file_has_no_payload() {
    let bytes = build_2d(50).to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    assert!(!view.has_payload());
    assert_eq!(view.payload(0), None);
}

#[test]
fn owned_loader_ignores_payload() {
    // A payload file loads as an owned index (payload dropped), giving the same
    // query results as the index-only file.
    let index = build_2d(200);
    let payloads: Vec<Vec<u8>> = (0..200).map(|i| format!("x{i}").into_bytes()).collect();
    let with = index.to_bytes_with_payloads(&payloads).unwrap();

    let owned = Index2D::from_bytes(&with).unwrap();
    let query = Box2D::new(0.0, 0.0, 50.5, 50.5);
    assert_eq!(owned.search(query), index.search(query));
}
