//! Scalar `Index2DF32`: native f32 build, conservative superset of `Index2D`.
#![cfg(feature = "f32-storage")]

use std::collections::HashSet;

use packed_spatial_index::{Box2D, Index2DBuilder, Triangle2D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn build_both(
    seed: u64,
    n: usize,
) -> (
    packed_spatial_index::Index2D,
    packed_spatial_index::Index2DF32,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b1 = Index2DBuilder::new(n).node_size(16);
    let mut b2 = Index2DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let (x, y) = (
            rng.random_range(0.0..1000.0f64),
            rng.random_range(0.0..1000.0),
        );
        let b = Box2D::new(
            x,
            y,
            x + rng.random_range(0.1..8.0),
            y + rng.random_range(0.1..8.0),
        );
        b1.add(b);
        b2.add(b);
    }
    (b1.finish().unwrap(), b2.finish_f32().unwrap())
}

#[test]
fn search_is_a_conservative_superset() {
    let (f64_index, f32_index) = build_both(0x2F32, 20_000);
    assert_eq!(f32_index.num_items(), 20_000);
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let (mut tot64, mut tot32) = (0usize, 0usize);
    for _ in 0..200 {
        let x = rng.random_range(0.0..1000.0);
        let y = rng.random_range(0.0..1000.0);
        let q = Box2D::new(x, y, x + 40.0, y + 40.0);
        let exact: HashSet<usize> = f64_index.search(q).into_iter().collect();
        let got: HashSet<usize> = f32_index.search(q).into_iter().collect();
        assert!(exact.is_subset(&got), "compact index missed a hit");
        tot64 += exact.len();
        tot32 += got.len();
    }
    assert!(
        tot32 <= tot64 + tot64 / 100 + 200,
        "fp blowup: {tot32} vs {tot64}"
    );
}

#[test]
fn search_exact_matches_f64_exactly() {
    // The inward-rounded f32 query compares f32-vs-f32 yet, with box_at returning
    // the true f64 boxes, exact search has no false positives -> identical to the
    // f64 index's search.
    let mut rng = StdRng::seed_from_u64(0x2E7A);
    let bs: Vec<Box2D> = (0..20_000)
        .map(|_| {
            let (x, y) = (
                rng.random_range(0.0..1000.0f64),
                rng.random_range(0.0..1000.0),
            );
            Box2D::new(
                x,
                y,
                x + rng.random_range(0.1..8.0),
                y + rng.random_range(0.1..8.0),
            )
        })
        .collect();
    let mut b1 = Index2DBuilder::new(bs.len()).node_size(16);
    let mut b2 = Index2DBuilder::new(bs.len()).node_size(16);
    for &b in &bs {
        b1.add(b);
        b2.add(b);
    }
    let f64_index = b1.finish().unwrap();
    let compact = b2.finish_f32().unwrap();

    let mut rng = StdRng::seed_from_u64(0x9B9B);
    for _ in 0..200 {
        let x = rng.random_range(0.0..1000.0);
        let y = rng.random_range(0.0..1000.0);
        let q = Box2D::new(x, y, x + 40.0, y + 40.0);
        let mut exact = compact.search_exact(q, |id| bs[id]);
        let mut f64_hits = f64_index.search(q);
        exact.sort_unstable();
        f64_hits.sort_unstable();
        assert_eq!(exact, f64_hits);
        let conservative: HashSet<usize> = compact.search(q).into_iter().collect();
        assert!(exact.iter().all(|id| conservative.contains(id)));
        assert_eq!(compact.any_exact(q, |id| bs[id]), !exact.is_empty());
        if let Some(f) = compact.first_exact(q, |id| bs[id]) {
            assert!(exact.contains(&f));
        }
    }
}

#[test]
fn from_triangles_builds_a_queryable_index() {
    let tris: Vec<Triangle2D> = (0..300)
        .map(|i| {
            let v = i as f64;
            Triangle2D::new([v, v], [v + 1.0, v], [v, v + 2.0])
        })
        .collect();
    let index = packed_spatial_index::Index2DF32::from_triangles(&tris).unwrap();
    assert_eq!(index.num_items(), 300);
    assert!(!index.search(Box2D::new(9.0, 9.0, 14.0, 14.0)).is_empty());
}
