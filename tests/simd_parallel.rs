#![cfg(all(feature = "simd", feature = "parallel"))]

//! The parallel SoA build (parallel sort + parallel reorder) must return the same
//! query results as the serial build and the AoS reference. Tree layout may differ
//! on tie-breaking, so results are compared as sorted sets, not byte-for-byte.

use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index3DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn simd2d_parallel_build_matches_serial() {
    let n = 20_000usize;
    let mut rng = StdRng::seed_from_u64(0x2A11);
    let boxes: Vec<Box2D> = (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..2_000.0);
            let y: f64 = rng.random_range(0.0..2_000.0);
            let w: f64 = rng.random_range(0.1..25.0);
            let h: f64 = rng.random_range(0.1..25.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect();

    let build = |parallel: bool| {
        let mut b = Index2DBuilder::new(n)
            .node_size(16)
            .parallel(parallel)
            .parallel_min_items(1);
        for &bx in &boxes {
            b.add(bx);
        }
        b.finish_simd().unwrap()
    };
    let serial = build(false);
    let parallel = build(true);

    let mut aos = Index2DBuilder::new(n).node_size(16);
    for &bx in &boxes {
        aos.add(bx);
    }
    let aos = aos.finish().unwrap();

    for _ in 0..300 {
        let x: f64 = rng.random_range(0.0..2_000.0);
        let y: f64 = rng.random_range(0.0..2_000.0);
        let w: f64 = rng.random_range(1.0..120.0);
        let q = Box2D::new(x, y, x + w, y + w);
        let mut expected = aos.search(q);
        expected.sort_unstable();
        let mut s = serial.search(q);
        s.sort_unstable();
        let mut p = parallel.search(q);
        p.sort_unstable();
        assert_eq!(expected, s, "serial SoA != AoS");
        assert_eq!(expected, p, "parallel SoA != AoS");
    }
}

#[test]
fn simd3d_parallel_build_matches_serial() {
    let n = 20_000usize;
    let mut rng = StdRng::seed_from_u64(0x2A33);
    let boxes: Vec<Box3D> = (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..2_000.0);
            let y: f64 = rng.random_range(0.0..2_000.0);
            let z: f64 = rng.random_range(0.0..2_000.0);
            let dx: f64 = rng.random_range(0.1..25.0);
            let dy: f64 = rng.random_range(0.1..25.0);
            let dz: f64 = rng.random_range(0.1..25.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect();

    let build = |parallel: bool| {
        let mut b = Index3DBuilder::new(n)
            .node_size(16)
            .parallel(parallel)
            .parallel_min_items(1);
        for &bx in &boxes {
            b.add(bx);
        }
        b.finish_simd().unwrap()
    };
    let serial = build(false);
    let parallel = build(true);

    let mut aos = Index3DBuilder::new(n).node_size(16);
    for &bx in &boxes {
        aos.add(bx);
    }
    let aos = aos.finish().unwrap();

    for _ in 0..300 {
        let x: f64 = rng.random_range(0.0..2_000.0);
        let y: f64 = rng.random_range(0.0..2_000.0);
        let z: f64 = rng.random_range(0.0..2_000.0);
        let w: f64 = rng.random_range(1.0..150.0);
        let q = Box3D::new(x, y, z, x + w, y + w, z + w);
        let mut expected = aos.search(q);
        expected.sort_unstable();
        let mut s = serial.search(q);
        s.sort_unstable();
        let mut p = parallel.search(q);
        p.sort_unstable();
        assert_eq!(expected, s, "serial SoA != AoS");
        assert_eq!(expected, p, "parallel SoA != AoS");
    }
}
