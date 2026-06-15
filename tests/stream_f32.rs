//! Streaming a compact `f32` index: `StreamIndex*F32` matches the owned scalar
//! `Index*F32` (same outward-rounded boxes, same widen-on-read overlap).
#![cfg(all(feature = "f32-storage", feature = "simd", feature = "stream"))]

use std::collections::HashSet;

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index3DBuilder, SliceReader, StreamIndex2DF32, StreamIndex3DF32,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn stream_f32_2d_matches_owned_scalar() {
    let n = 20_000;
    let mut rng = StdRng::seed_from_u64(0x2F32);
    let mut b_scalar = Index2DBuilder::new(n).node_size(16);
    let mut b_simd = Index2DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let (x, y) = (
            rng.random_range(0.0..1000.0f64),
            rng.random_range(0.0..1000.0),
        );
        let bx = Box2D::new(x, y, x + 5.0, y + 5.0);
        b_scalar.add(bx);
        b_simd.add(bx);
    }
    // Oracle: the owned scalar f32 index (widen-on-read overlap, same as stream).
    let owned = b_scalar.finish_f32().unwrap();
    // Streamed file: the SIMD f32 index serializes the identical f32 boxes.
    let bytes = b_simd.finish_simd_f32().unwrap().to_bytes();

    let stream = StreamIndex2DF32::open(SliceReader::new(bytes)).unwrap();
    assert_eq!(stream.num_items(), n);
    assert!(!stream.has_payload());

    let mut rng = StdRng::seed_from_u64(0xBEEF);
    for _ in 0..200 {
        let x = rng.random_range(0.0..1000.0);
        let y = rng.random_range(0.0..1000.0);
        let q = Box2D::new(x, y, x + 50.0, y + 50.0);
        let streamed: HashSet<usize> = stream.search(q).unwrap().into_iter().collect();
        let owned_hits: HashSet<usize> = owned.search(q).into_iter().collect();
        assert_eq!(streamed, owned_hits);
    }
}

#[test]
fn stream_f32_3d_matches_owned_scalar() {
    let n = 20_000;
    let mut rng = StdRng::seed_from_u64(0x3F32);
    let mut b_scalar = Index3DBuilder::new(n).node_size(16);
    let mut b_simd = Index3DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let c = [
            rng.random_range(0.0..1000.0f64),
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
        ];
        let bx = Box3D::new(c[0], c[1], c[2], c[0] + 5.0, c[1] + 5.0, c[2] + 5.0);
        b_scalar.add(bx);
        b_simd.add(bx);
    }
    let owned = b_scalar.finish_f32().unwrap();
    let bytes = b_simd.finish_simd_f32().unwrap().to_bytes();

    let stream = StreamIndex3DF32::open(SliceReader::new(bytes)).unwrap();
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    for _ in 0..200 {
        let c = [
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
        ];
        let q = Box3D::new(c[0], c[1], c[2], c[0] + 60.0, c[1] + 60.0, c[2] + 60.0);
        let streamed: HashSet<usize> = stream.search(q).unwrap().into_iter().collect();
        let owned_hits: HashSet<usize> = owned.search(q).into_iter().collect();
        assert_eq!(streamed, owned_hits);
    }
}

#[test]
fn stream_f32_is_half_the_bytes_of_f64() {
    let n = 5_000;
    let mut b64 = Index2DBuilder::new(n).node_size(16);
    let mut b32 = Index2DBuilder::new(n).node_size(16);
    for i in 0..n {
        let v = i as f64;
        let bx = Box2D::new(v, v, v + 1.0, v + 1.0);
        b64.add(bx);
        b32.add(bx);
    }
    let f64_bytes = b64.finish().unwrap().to_bytes();
    let f32_bytes = b32.finish_simd_f32().unwrap().to_bytes();
    // f32 boxes are half the size, so the streamable file is much smaller.
    assert!(f32_bytes.len() < f64_bytes.len());
}

#[cfg(feature = "async")]
#[test]
fn stream_f32_async_matches_sync() {
    use packed_spatial_index::RangeReader;
    use std::io;

    struct AsyncSlice(Vec<u8>);
    impl packed_spatial_index::AsyncRangeReader for AsyncSlice {
        async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            SliceReader::new(self.0.as_slice()).read_exact_at(offset, buf)
        }
        fn len(&self) -> Option<u64> {
            Some(self.0.len() as u64)
        }
    }

    let n = 20_000;
    let mut rng = StdRng::seed_from_u64(0xA5F3);
    let mut b = Index2DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let (x, y) = (
            rng.random_range(0.0..1000.0f64),
            rng.random_range(0.0..1000.0),
        );
        b.add(Box2D::new(x, y, x + 5.0, y + 5.0));
    }
    let bytes = b.finish_simd_f32().unwrap().to_bytes();
    let sync = StreamIndex2DF32::open(SliceReader::new(bytes.clone())).unwrap();
    let astream = pollster::block_on(StreamIndex2DF32::open_async(AsyncSlice(bytes))).unwrap();

    let q = Box2D::new(300.0, 300.0, 380.0, 380.0);
    let mut s = sync.search(q).unwrap();
    let mut a = pollster::block_on(astream.search_async(q)).unwrap();
    s.sort_unstable();
    a.sort_unstable();
    assert_eq!(s, a);
}
