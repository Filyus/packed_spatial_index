//! Compact f32 mesh end to end WITHOUT simd: build a scalar `Index3DF32` over
//! triangle boxes, serialize the triangles as a fixed-width payload, then stream
//! `(id, triangle)` back through `StreamIndex3DF32`.
#![cfg(all(feature = "f32-storage", feature = "stream"))]

use packed_spatial_index::{Box3D, Index3DF32, SliceReader, StreamIndex3DF32, Triangle3DF32};

fn mesh(n: usize) -> Vec<Triangle3DF32> {
    (0..n)
        .map(|i| {
            // Vertex a.x encodes the item id, so streamed blobs are self-checking.
            let v = i as f32;
            Triangle3DF32::new([v, 0.0, 0.0], [v + 1.0, 0.0, 0.0], [v, 1.0, 1.0])
        })
        .collect()
}

#[test]
fn f32_mesh_payload_round_trips_over_stream() {
    let n = 5_000;
    let tris = mesh(n);
    let index = Index3DF32::from_triangles(&tris).unwrap();
    let bytes = index.serialize().triangles(&tris).to_bytes().unwrap();

    let stream = StreamIndex3DF32::open(SliceReader::new(bytes)).unwrap();
    assert_eq!(stream.num_items(), n);
    assert!(stream.has_payload());

    // Full extent: every triangle streams back, keyed by insertion id.
    let all = stream
        .search_payloads(Box3D::new(-1.0, -1.0, -1.0, 1e9, 1e9, 1e9))
        .unwrap();
    assert_eq!(all.len(), n);
    for (id, blob) in &all {
        assert_eq!(blob.len(), 36); // 9 f32
        let ax = f32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
        assert_eq!(ax, *id as f32, "triangle {id} payload mismatch");
    }

    // A windowed query returns a subset, still self-consistent.
    let win = stream
        .search_payloads(Box3D::new(100.0, -1.0, -1.0, 140.0, 2.0, 2.0))
        .unwrap();
    assert!(!win.is_empty() && win.len() < n);
    for (id, blob) in &win {
        let ax = f32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
        assert_eq!(ax, *id as f32);
    }
}

#[test]
fn index_only_f32_has_no_payload() {
    let index = Index3DF32::from_triangles(&mesh(100)).unwrap();
    let bytes = index.to_bytes(); // no payload attached
    let stream = StreamIndex3DF32::open(SliceReader::new(bytes)).unwrap();
    assert!(!stream.has_payload());
    assert_eq!(stream.num_items(), 100);
}

#[test]
fn corrupt_f32_mesh_bytes_never_panic() {
    // Flip a byte across the whole f32 mesh file (boxes + payload) and confirm
    // open + search/search_payloads never panic (Ok or Err, never UB/panic).
    let tris = mesh(400);
    let base = Index3DF32::from_triangles(&tris)
        .unwrap()
        .serialize()
        .triangles(&tris)
        .to_bytes()
        .unwrap();
    let q = Box3D::new(-1.0, -1.0, -1.0, 1e9, 1e9, 1e9);
    for i in (0..base.len()).step_by(31) {
        let mut bytes = base.clone();
        bytes[i] ^= 0xFF;
        if let Ok(s) = StreamIndex3DF32::open(SliceReader::new(bytes)) {
            let _ = s.search(q);
            let _ = s.search_payloads(q);
        }
    }
}

#[test]
fn fixed_width_records_payload_streams() {
    const STRIDE: usize = 12;
    let n = 1_000;
    let index = Index3DF32::from_triangles(&mesh(n)).unwrap();
    let mut flat = vec![0u8; n * STRIDE];
    for id in 0..n {
        flat[id * STRIDE..id * STRIDE + 8].copy_from_slice(&(id as u64).to_le_bytes());
    }
    let bytes = index.serialize().records(STRIDE, &flat).to_bytes().unwrap();
    let stream = StreamIndex3DF32::open(SliceReader::new(bytes)).unwrap();
    let all = stream
        .search_payloads(Box3D::new(-1.0, -1.0, -1.0, 1e9, 1e9, 1e9))
        .unwrap();
    assert_eq!(all.len(), n);
    for (id, blob) in &all {
        assert_eq!(blob.len(), STRIDE);
        assert_eq!(&blob[..8], &(*id as u64).to_le_bytes());
    }
}

#[test]
fn interleaved_f32_streams_and_matches_soa() {
    let tris = mesh(20_000);
    let index = Index3DF32::from_triangles(&tris).unwrap();
    let soa = index.serialize().to_bytes().unwrap();
    let inter = index.serialize().interleaved().to_bytes().unwrap();
    // Same file size (box+index reordered, not resized).
    assert_eq!(soa.len(), inter.len());

    let q = Box3D::new(100.0, -1.0, -1.0, 200.0, 2.0, 2.0);
    let mut a = StreamIndex3DF32::open(SliceReader::new(soa))
        .unwrap()
        .search(q)
        .unwrap();
    let mut b = StreamIndex3DF32::open(SliceReader::new(inter))
        .unwrap()
        .search(q)
        .unwrap();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);

    // Interleaved + payload also streams back correctly.
    let with_payload = index
        .serialize()
        .interleaved()
        .triangles(&tris)
        .to_bytes()
        .unwrap();
    let stream = StreamIndex3DF32::open(SliceReader::new(with_payload)).unwrap();
    assert!(stream.has_payload());
    let all = stream
        .search_payloads(Box3D::new(-1.0, -1.0, -1.0, 1e9, 1e9, 1e9))
        .unwrap();
    assert_eq!(all.len(), tris.len());
}
