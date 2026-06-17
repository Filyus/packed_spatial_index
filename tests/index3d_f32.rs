//! Scalar `Index3DF32`: native f32 build, conservative superset of `Index3D`.
#![cfg(feature = "f32-storage")]

use std::collections::HashSet;

use packed_spatial_index::{Box3D, Index3DBuilder, Index3DF32, Point3D, Ray3D, Triangle3D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn boxes(seed: u64, n: usize) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let c = [
                rng.random_range(0.0..1000.0f64),
                rng.random_range(0.0..1000.0),
                rng.random_range(0.0..1000.0),
            ];
            Box3D::new(
                c[0],
                c[1],
                c[2],
                c[0] + rng.random_range(0.1..8.0),
                c[1] + rng.random_range(0.1..8.0),
                c[2] + rng.random_range(0.1..8.0),
            )
        })
        .collect()
}

fn build_both(
    seed: u64,
    n: usize,
) -> (
    packed_spatial_index::Index3D,
    packed_spatial_index::Index3DF32,
) {
    let bs = boxes(seed, n);
    let mut b1 = Index3DBuilder::new(n).node_size(16);
    let mut b2 = Index3DBuilder::new(n).node_size(16);
    for &b in &bs {
        b1.add(b);
        b2.add(b);
    }
    (b1.finish().unwrap(), b2.finish_f32().unwrap())
}

#[test]
fn scalar_f32_neighbors_exact_matches_f64_and_simd() {
    let bs = boxes(0x77AB, 5_000);
    let mut b1 = Index3DBuilder::new(bs.len()).node_size(16);
    let mut b2 = Index3DBuilder::new(bs.len()).node_size(16);
    for &b in &bs {
        b1.add(b);
        b2.add(b);
    }
    let f64_index = b1.finish().unwrap();
    let scalar: Index3DF32 = b2.finish_f32().unwrap();
    #[cfg(feature = "simd")]
    let simd = {
        let mut bsim = Index3DBuilder::new(bs.len()).node_size(16);
        for &b in &bs {
            bsim.add(b);
        }
        bsim.finish_simd_f32().unwrap()
    };

    let mut q = StdRng::seed_from_u64(0x9191);
    for _ in 0..150 {
        let p = Point3D::new(
            q.random_range(0.0..1000.0),
            q.random_range(0.0..1000.0),
            q.random_range(0.0..1000.0),
        );
        let exact: HashSet<usize> = scalar
            .neighbors_exact(p, 5, |i| bs[i])
            .into_iter()
            .collect();
        let f64n: HashSet<usize> = f64_index.neighbors(p, 5).into_iter().collect();
        assert_eq!(
            exact, f64n,
            "scalar f32 exact kNN must match f64 kNN at {p:?}"
        );
        assert_eq!(scalar.neighbors(p, 5).len(), 5);
        #[cfg(feature = "simd")]
        assert_eq!(scalar.neighbors(p, 5), simd.neighbors(p, 5));
    }
}

#[test]
fn search_is_a_conservative_superset() {
    let (f64_index, f32_index) = build_both(0xF32A, 20_000);
    assert_eq!(f32_index.num_items(), 20_000);
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let (mut tot64, mut tot32) = (0usize, 0usize);
    for _ in 0..200 {
        let q = {
            let x = rng.random_range(0.0..1000.0);
            let y = rng.random_range(0.0..1000.0);
            let z = rng.random_range(0.0..1000.0);
            Box3D::new(x, y, z, x + 40.0, y + 40.0, z + 40.0)
        };
        let exact: HashSet<usize> = f64_index.search(q).into_iter().collect();
        let got: HashSet<usize> = f32_index.search(q).into_iter().collect();
        assert!(exact.is_subset(&got), "compact index missed a hit");
        tot64 += exact.len();
        tot32 += got.len();
    }
    // Outward rounding is sub-ulp at this scale: few false positives.
    assert!(
        tot32 <= tot64 + tot64 / 100 + 200,
        "fp blowup: {tot32} vs {tot64}"
    );
}

#[test]
fn raycast_is_a_conservative_superset() {
    let (f64_index, f32_index) = build_both(0xF32B, 10_000);
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    for _ in 0..200 {
        let o = Point3D::new(
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
            -10.0,
        );
        let ray = Ray3D::new(
            o,
            rng.random_range(-0.3..0.3),
            rng.random_range(-0.3..0.3),
            1.0,
            2000.0,
        );
        let exact: HashSet<usize> = f64_index.raycast(ray).into_iter().collect();
        let got: HashSet<usize> = f32_index.raycast(ray).into_iter().collect();
        assert!(exact.is_subset(&got));
    }
}

#[test]
fn from_triangles_and_edge_cases() {
    // Built from triangles, queried, mapped back via bounding boxes.
    let tris: Vec<Triangle3D> = (0..300)
        .map(|i| {
            let v = i as f64;
            Triangle3D::new([v, v, v], [v + 1.0, v, v], [v, v + 1.0, v + 2.0])
        })
        .collect();
    let index = packed_spatial_index::Index3DF32::from_triangles(&tris).unwrap();
    assert_eq!(index.num_items(), 300);
    let hits = index.search(Box3D::new(9.0, 9.0, 9.0, 12.0, 12.0, 20.0));
    assert!(!hits.is_empty());

    // Empty and single-node.
    let empty = packed_spatial_index::Index3DF32::from_triangles::<Triangle3D>(&[]).unwrap();
    assert!(
        empty
            .search(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
            .is_empty()
    );
    let (f64_index, f32_index) = build_both(0x1, 5);
    let q = Box3D::new(-1.0, -1.0, -1.0, 2000.0, 2000.0, 2000.0);
    let mut a = f64_index.search(q);
    let mut b = f32_index.search(q);
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b); // full extent: all items, no false positives possible
}

#[test]
fn to_bytes_from_bytes_round_trips() {
    let (_, compact) = build_both(0x5E11, 5_000);
    let bytes = compact.to_bytes();
    let loaded = Index3DF32::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.num_items(), compact.num_items());
    assert_eq!(loaded.is_empty(), compact.is_empty());
    assert_eq!(loaded.node_size(), compact.node_size());
    assert_eq!(loaded.extent(), compact.extent());
    let q = Box3D::new(300.0, 300.0, 300.0, 360.0, 360.0, 360.0);
    let mut a = compact.search(q);
    let mut b = loaded.search(q);
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}

#[test]
fn any_first_visit_match_search() {
    use std::ops::ControlFlow;
    let (_, compact) = build_both(0xA1F1, 5_000);
    let q = Box3D::new(300.0, 300.0, 300.0, 360.0, 360.0, 360.0);
    let hits = compact.search(q);
    assert_eq!(compact.any(q), !hits.is_empty());
    assert_eq!(compact.first(q).is_some(), !hits.is_empty());
    if let Some(f) = compact.first(q) {
        assert!(hits.contains(&f));
    }
    let mut visited = Vec::new();
    let _ = compact.visit(q, |i| {
        visited.push(i);
        ControlFlow::<()>::Continue(())
    });
    visited.sort_unstable();
    let mut s = hits;
    s.sort_unstable();
    assert_eq!(visited, s);
    // empty query -> any false, first None
    let empty_q = Box3D::new(-100.0, -100.0, -100.0, -99.0, -99.0, -99.0);
    assert!(!compact.any(empty_q));
    assert_eq!(compact.first(empty_q), None);
}

#[test]
fn metadata_round_trips_on_f32_file() {
    use packed_spatial_index::read_metadata;
    let (_, compact) = build_both(0xC0DE, 200);
    let bytes = compact
        .serialize()
        .crs("EPSG:4979")
        .content_type("application/x-mesh")
        .to_bytes()
        .unwrap();
    let md = read_metadata(&bytes).unwrap();
    assert_eq!(md.crs.as_deref(), Some("EPSG:4979"));
    assert_eq!(md.content_type.as_deref(), Some("application/x-mesh"));
}

#[cfg(feature = "simd")]
#[test]
fn simd_f32_search_matches_scalar_f32_exactly() {
    // Both round the query inward, so the SIMD and scalar f32 indexes return the
    // identical tight superset (not just each a superset of f64).
    let bs = boxes(0x51D3, 20_000);
    let mut b_scalar = Index3DBuilder::new(bs.len()).node_size(16);
    let mut b_simd = Index3DBuilder::new(bs.len()).node_size(16);
    for &b in &bs {
        b_scalar.add(b);
        b_simd.add(b);
    }
    let scalar = b_scalar.finish_f32().unwrap();
    let simd = b_simd.finish_simd_f32().unwrap();
    let mut rng = StdRng::seed_from_u64(0x6060);
    for _ in 0..200 {
        let x = rng.random_range(0.0..1000.0);
        let y = rng.random_range(0.0..1000.0);
        let z = rng.random_range(0.0..1000.0);
        let q = Box3D::new(x, y, z, x + 40.0, y + 40.0, z + 40.0);
        let mut a = scalar.search(q);
        let mut b = simd.search(q);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

#[test]
fn search_exact_matches_f64_exactly() {
    // With box_at returning the true f64 boxes, the f32 index's exact search has
    // no false positives -> identical to the f64 index's search.
    let bs = boxes(0xE7AC, 20_000);
    let mut b1 = Index3DBuilder::new(bs.len()).node_size(16);
    let mut b2 = Index3DBuilder::new(bs.len()).node_size(16);
    for &b in &bs {
        b1.add(b);
        b2.add(b);
    }
    let f64_index = b1.finish().unwrap();
    let compact = b2.finish_f32().unwrap();

    let mut rng = StdRng::seed_from_u64(0x9A9A);
    for _ in 0..200 {
        let x = rng.random_range(0.0..1000.0);
        let y = rng.random_range(0.0..1000.0);
        let z = rng.random_range(0.0..1000.0);
        let q = Box3D::new(x, y, z, x + 40.0, y + 40.0, z + 40.0);
        let mut exact = compact.search_exact(q, |id| bs[id]);
        let mut f64_hits = f64_index.search(q);
        exact.sort_unstable();
        f64_hits.sort_unstable();
        assert_eq!(exact, f64_hits);
        // exact is a subset of the conservative search
        let conservative: HashSet<usize> = compact.search(q).into_iter().collect();
        assert!(exact.iter().all(|id| conservative.contains(id)));
        // any_exact / first_exact agree
        assert_eq!(compact.any_exact(q, |id| bs[id]), !exact.is_empty());
        if let Some(f) = compact.first_exact(q, |id| bs[id]) {
            assert!(exact.contains(&f));
        }
    }
}
