use packed_spatial_index::{
    Box2D, Box3D, FileMetadata, Index2DBuilder, Index2DView, Index3DBuilder, read_metadata,
};

fn build_2d(n: usize) -> packed_spatial_index::Index2D {
    let mut b = Index2DBuilder::new(n);
    for i in 0..n {
        let v = i as f64;
        b.add(Box2D::new(v, v, v + 1.0, v + 1.0));
    }
    b.finish().unwrap()
}

#[test]
fn metadata_round_trips_with_payloads() {
    let index = build_2d(3);
    let payloads = [b"a".as_slice(), b"bb", b"ccc"];
    let bytes = index
        .serialize()
        .crs("EPSG:4326")
        .content_type("application/geo+json")
        .attribution("© Example")
        .payloads(&payloads)
        .to_bytes()
        .unwrap();

    let md = read_metadata(&bytes).unwrap();
    assert_eq!(md.crs.as_deref(), Some("EPSG:4326"));
    assert_eq!(md.content_type.as_deref(), Some("application/geo+json"));
    assert_eq!(md.attribution.as_deref(), Some("© Example"));

    // The index and payload still load — the loaders skip the optional META chunk.
    let view = Index2DView::from_bytes(&bytes).unwrap();
    assert_eq!(view.num_items(), 3);
    assert!(view.has_payload());
    let hits = view.search_payloads(Box2D::new(-1.0, -1.0, 10.0, 10.0));
    assert_eq!(hits.len(), 3);
}

#[test]
fn partial_and_absent_metadata() {
    let index = build_2d(2);

    // Only one field set.
    let bytes = index.serialize().crs("EPSG:3857").to_bytes().unwrap();
    let md = read_metadata(&bytes).unwrap();
    assert_eq!(md.crs.as_deref(), Some("EPSG:3857"));
    assert_eq!(md.content_type, None);
    assert_eq!(md.attribution, None);

    // No metadata at all → empty (no META chunk).
    let plain = index.to_bytes();
    assert_eq!(read_metadata(&plain).unwrap(), FileMetadata::default());
}

#[test]
fn metadata_3d() {
    let mut b = Index3DBuilder::new(2);
    b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    b.add(Box3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0));
    let index = b.finish().unwrap();
    let bytes = index.serialize().crs("EPSG:4979").to_bytes().unwrap();
    assert_eq!(
        read_metadata(&bytes).unwrap().crs.as_deref(),
        Some("EPSG:4979")
    );
}

#[cfg(feature = "stream")]
#[test]
fn metadata_with_interleaved_streams() {
    use packed_spatial_index::{SliceReader, StreamIndex2D};

    let index = build_2d(2_000);
    let payloads: Vec<Vec<u8>> = (0..2_000).map(|i| format!("f{i}").into_bytes()).collect();
    let bytes = index
        .serialize()
        .interleaved()
        .crs("EPSG:4326")
        .content_type("application/x-protobuf")
        .payloads(&payloads)
        .to_bytes()
        .unwrap();

    // Metadata reads back without loading the index.
    let md = read_metadata(&bytes).unwrap();
    assert_eq!(md.crs.as_deref(), Some("EPSG:4326"));
    assert_eq!(md.content_type.as_deref(), Some("application/x-protobuf"));

    // The streaming reader opens the (interleaved) file and serves payloads,
    // skipping the optional META chunk.
    let stream = StreamIndex2D::open(SliceReader::new(bytes)).unwrap();
    let pairs = stream
        .search_payloads(Box2D::new(-1.0, -1.0, 5000.0, 5000.0))
        .unwrap();
    assert_eq!(pairs.len(), 2_000);
    for (id, blob) in &pairs {
        assert_eq!(blob, &payloads[*id]);
    }
}
