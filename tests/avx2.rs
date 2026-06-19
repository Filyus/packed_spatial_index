//! The AVX2 runtime tier (left-pack search / visit / all-hits raycast) must
//! return exactly what the scalar indexes do. These call the doc-hidden `*_avx2`
//! entries directly so the AVX2 path is exercised even on an AVX-512 machine
//! (where the public dispatch would otherwise pick AVX-512). Skipped on hosts
//! without AVX2.

#![cfg(all(feature = "simd", target_arch = "x86_64"))]

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index3D, Index3DBuilder, Point2D, Point3D, Ray2D, Ray3D,
};
use std::ops::ControlFlow;

fn avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

fn boxes2(n: usize, seed: u64) -> Vec<Box2D> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed as usize) as u32;
            let x = (h % 977) as f64 / 977.0 * 1000.0;
            let y = ((h >> 7) % 991) as f64 / 991.0 * 1000.0;
            let w = 0.2 + ((h >> 3) % 6) as f64;
            Box2D::new(x, y, x + w, y + 0.3 + ((h >> 5) % 5) as f64)
        })
        .collect()
}
fn boxes3(n: usize, seed: u64) -> Vec<Box3D> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed as usize) as u32;
            let x = (h % 977) as f64 / 977.0 * 1000.0;
            let y = ((h >> 7) % 991) as f64 / 991.0 * 1000.0;
            let z = ((h >> 13) % 983) as f64 / 983.0 * 1000.0;
            Box3D::new(x, y, z, x + 1.5, y + 1.5, z + 1.5)
        })
        .collect()
}

const SIZES: &[usize] = &[0, 1, 3, 5, 7, 16, 17, 31, 63, 1000, 20_000];
const NODE_SIZES: &[usize] = &[4, 8, 16, 31];

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

#[test]
fn avx2_search_matches_scalar_f64() {
    if !avx2() {
        return;
    }
    for &n in SIZES {
        for &ns in NODE_SIZES {
            let b2 = boxes2(n, 1);
            let b3 = boxes3(n, 2);
            let s2 = {
                let mut b = Index2DBuilder::new(n).node_size(ns);
                b2.iter().for_each(|&x| b.add(x));
                b.finish().unwrap()
            };
            let m2 = {
                let mut b = Index2DBuilder::new(n).node_size(ns);
                b2.iter().for_each(|&x| b.add(x));
                b.finish_simd().unwrap()
            };
            let s3 = {
                let mut b = Index3DBuilder::new(n).node_size(ns);
                b3.iter().for_each(|&x| b.add(x));
                b.finish().unwrap()
            };
            let m3 = {
                let mut b = Index3DBuilder::new(n).node_size(ns);
                b3.iter().for_each(|&x| b.add(x));
                b.finish_simd().unwrap()
            };
            let q2 = [
                Box2D::new(-1e9, -1e9, 1e9, 1e9),
                Box2D::new(100.0, 100.0, 400.0, 400.0),
                Box2D::new(500.0, 500.0, 500.0, 500.0),
                Box2D::new(5000.0, 5000.0, 6000.0, 6000.0),
            ];
            let (mut o, mut st) = (Vec::new(), Vec::new());
            for q in q2 {
                m2.search_avx2(q, &mut o, &mut st);
                assert_eq!(sorted(o.clone()), sorted(s2.search(q)), "2d n={n} ns={ns}");
                // visit yields the same set
                let mut vis = Vec::new();
                let _ = m2.visit_avx2::<(), _>(q, &mut st, |i| {
                    vis.push(i);
                    ControlFlow::Continue(())
                });
                assert_eq!(sorted(vis), sorted(s2.search(q)), "2d visit n={n} ns={ns}");

                let q3 = Box3D::new(q.min_x, q.min_y, q.min_x, q.max_x, q.max_y, q.max_x);
                m3.search_avx2(q3, &mut o, &mut st);
                assert_eq!(sorted(o.clone()), sorted(s3.search(q3)), "3d n={n} ns={ns}");
                let mut vis3 = Vec::new();
                let _ = m3.visit_avx2::<(), _>(q3, &mut st, |i| {
                    vis3.push(i);
                    ControlFlow::Continue(())
                });
                assert_eq!(
                    sorted(vis3),
                    sorted(s3.search(q3)),
                    "3d visit n={n} ns={ns}"
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "f32-storage")]
fn avx2_search_matches_scalar_f32() {
    if !avx2() {
        return;
    }
    use packed_spatial_index::{Index2DF32, Index3DF32};
    for &n in SIZES {
        for &ns in NODE_SIZES {
            let b2 = boxes2(n, 3);
            let b3 = boxes3(n, 4);
            let s2: Index2DF32 = {
                let mut b = Index2DBuilder::new(n).node_size(ns);
                b2.iter().for_each(|&x| b.add(x));
                b.finish_f32().unwrap()
            };
            let m2 = {
                let mut b = Index2DBuilder::new(n).node_size(ns);
                b2.iter().for_each(|&x| b.add(x));
                b.finish_simd_f32().unwrap()
            };
            let s3: Index3DF32 = {
                let mut b = Index3DBuilder::new(n).node_size(ns);
                b3.iter().for_each(|&x| b.add(x));
                b.finish_f32().unwrap()
            };
            let m3 = {
                let mut b = Index3DBuilder::new(n).node_size(ns);
                b3.iter().for_each(|&x| b.add(x));
                b.finish_simd_f32().unwrap()
            };
            let q2 = [
                Box2D::new(-1e9, -1e9, 1e9, 1e9),
                Box2D::new(100.0, 100.0, 400.0, 400.0),
                Box2D::new(5000.0, 5000.0, 6000.0, 6000.0),
            ];
            let mut o = Vec::new();
            for q in q2 {
                m2.search_avx2_into(q, &mut o);
                assert_eq!(
                    sorted(o.clone()),
                    sorted(s2.search(q)),
                    "f32 2d n={n} ns={ns}"
                );
                let q3 = Box3D::new(q.min_x, q.min_y, q.min_x, q.max_x, q.max_y, q.max_x);
                m3.search_avx2_into(q3, &mut o);
                assert_eq!(
                    sorted(o.clone()),
                    sorted(s3.search(q3)),
                    "f32 3d n={n} ns={ns}"
                );
            }
        }
    }
}

#[test]
fn avx2_raycast_all_hits_matches_scalar() {
    if !avx2() {
        return;
    }
    for &n in &[0usize, 1, 5, 17, 1000, 20_000] {
        for &ns in &[4usize, 16] {
            let b2 = boxes2(n, 5);
            let b3 = boxes3(n, 6);
            let s2: Index2D = {
                let mut b = Index2DBuilder::new(n).node_size(ns);
                b2.iter().for_each(|&x| b.add(x));
                b.finish().unwrap()
            };
            let m2 = {
                let mut b = Index2DBuilder::new(n).node_size(ns);
                b2.iter().for_each(|&x| b.add(x));
                b.finish_simd().unwrap()
            };
            let s3: Index3D = {
                let mut b = Index3DBuilder::new(n).node_size(ns);
                b3.iter().for_each(|&x| b.add(x));
                b.finish().unwrap()
            };
            let m3 = {
                let mut b = Index3DBuilder::new(n).node_size(ns);
                b3.iter().for_each(|&x| b.add(x));
                b.finish_simd().unwrap()
            };
            let mut o = Vec::new();
            for k in 0..40 {
                let a = k as f64 / 40.0 * std::f64::consts::TAU;
                let r2 = Ray2D::new(Point2D::new(500.0, 500.0), a.cos(), a.sin(), 1000.0);
                m2.raycast_avx2_into(r2, &mut o);
                assert_eq!(
                    sorted(o.clone()),
                    sorted(s2.raycast(r2)),
                    "ray2 n={n} ns={ns}"
                );

                let r3 = Ray3D::new(
                    Point3D::new(500.0, 500.0, 500.0),
                    a.cos(),
                    a.sin(),
                    (a * 0.5).cos(),
                    1000.0,
                );
                m3.raycast_avx2_into(r3, &mut o);
                assert_eq!(
                    sorted(o.clone()),
                    sorted(s3.raycast(r3)),
                    "ray3 n={n} ns={ns}"
                );
            }
        }
    }
}
