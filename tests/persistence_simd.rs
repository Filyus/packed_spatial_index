#![cfg(feature = "simd")]

//! Persistence interop between the AoS (`Index2D`/`Index3D`) and SoA/SIMD
//! (`SimdIndex2D`/`SimdIndex3D`) indexes. They share the canonical `PSINDEX`
//! format, so bytes written by one must load into the other.

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index3D, Index3DBuilder, SimdIndex2D, SimdIndex3D,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn boxes_2d(n: usize, seed: u64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn boxes_3d(n: usize, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1_000.0);
            let y: f64 = rng.random_range(0.0..1_000.0);
            let z: f64 = rng.random_range(0.0..1_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

#[test]
fn simd2d_bytes_match_aos_and_round_trip() {
    let boxes = boxes_2d(5_000, 0x501);
    let mut aos = Index2DBuilder::new(boxes.len()).node_size(16);
    let mut simd = Index2DBuilder::new(boxes.len()).node_size(16);
    for &b in &boxes {
        aos.add(b);
        simd.add(b);
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    // Byte-for-byte identical serialization.
    assert_eq!(simd.to_bytes(), aos.to_bytes());

    // to_bytes_into matches to_bytes.
    let mut buf = Vec::new();
    simd.to_bytes_into(&mut buf);
    assert_eq!(buf, simd.to_bytes());

    // Cross-load both directions, then compare query results.
    let from_aos_bytes = SimdIndex2D::from_bytes(&aos.to_bytes()).unwrap();
    let from_simd_bytes = Index2D::from_bytes(&simd.to_bytes()).unwrap();
    let round = SimdIndex2D::from_bytes(&simd.to_bytes()).unwrap();

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    for _ in 0..300 {
        let x: f64 = rng.random_range(0.0..1_000.0);
        let y: f64 = rng.random_range(0.0..1_000.0);
        let w: f64 = rng.random_range(1.0..100.0);
        let q = Box2D::new(x, y, x + w, y + w);

        let mut expected = aos.search(q);
        expected.sort_unstable();
        for got in [
            from_aos_bytes.search(q),
            from_simd_bytes.search(q),
            round.search(q),
        ] {
            let mut got = got;
            got.sort_unstable();
            assert_eq!(expected, got);
        }
    }
}

#[test]
fn simd3d_bytes_match_aos_and_round_trip() {
    let boxes = boxes_3d(5_000, 0x503);
    let mut aos = Index3DBuilder::new(boxes.len()).node_size(16);
    let mut simd = Index3DBuilder::new(boxes.len()).node_size(16);
    for &b in &boxes {
        aos.add(b);
        simd.add(b);
    }
    let aos = aos.finish().unwrap();
    let simd = simd.finish_simd().unwrap();

    assert_eq!(simd.to_bytes(), aos.to_bytes());

    let mut buf = Vec::new();
    simd.to_bytes_into(&mut buf);
    assert_eq!(buf, simd.to_bytes());

    let from_aos_bytes = SimdIndex3D::from_bytes(&aos.to_bytes()).unwrap();
    let from_simd_bytes = Index3D::from_bytes(&simd.to_bytes()).unwrap();
    let round = SimdIndex3D::from_bytes(&simd.to_bytes()).unwrap();

    let mut rng = StdRng::seed_from_u64(0xBEEF);
    for _ in 0..300 {
        let x: f64 = rng.random_range(0.0..1_000.0);
        let y: f64 = rng.random_range(0.0..1_000.0);
        let z: f64 = rng.random_range(0.0..1_000.0);
        let w: f64 = rng.random_range(1.0..120.0);
        let q = Box3D::new(x, y, z, x + w, y + w, z + w);

        let mut expected = aos.search(q);
        expected.sort_unstable();
        for got in [
            from_aos_bytes.search(q),
            from_simd_bytes.search(q),
            round.search(q),
        ] {
            let mut got = got;
            got.sort_unstable();
            assert_eq!(expected, got);
        }
    }
}

#[test]
fn simd_from_bytes_rejects_wrong_dimension() {
    let b2 = boxes_2d(64, 0x1);
    let mut a2 = Index2DBuilder::new(b2.len());
    for &b in &b2 {
        a2.add(b);
    }
    let bytes2d = a2.finish().unwrap().to_bytes();

    let b3 = boxes_3d(64, 0x2);
    let mut a3 = Index3DBuilder::new(b3.len());
    for &b in &b3 {
        a3.add(b);
    }
    let bytes3d = a3.finish().unwrap().to_bytes();

    // A 3D blob must not load as a 2D SIMD index, and vice versa.
    assert!(SimdIndex2D::from_bytes(&bytes3d).is_err());
    assert!(SimdIndex3D::from_bytes(&bytes2d).is_err());
}
