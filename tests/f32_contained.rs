//! The scalar f32 `search` and `count`, and the SIMD f32 `count` (owned and
//! view), take a subtree the query covers as its leaf range, without testing
//! each item. That is only sound if a covered node
//! implies every stored box under it passes the same overlap test the leaves
//! would have run, including after rounding to f32. `visit` still tests every
//! leaf and never takes the shortcut, so it is the reference: the same set of
//! items (its descent orders children differently, and the order is not part
//! of the API), on coordinates f32 cannot represent and windows whose edges
//! fall between neighbouring f32 values.
#![cfg(feature = "f32-storage")]

use std::ops::ControlFlow;

use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index3DBuilder};

/// Coordinates offset far enough that f32 spacing is about 0.06, with
/// fractional parts f32 cannot hold.
const BASE: f64 = 1.0e6;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn visit_2d(index: &packed_spatial_index::Index2DF32, q: Box2D) -> Vec<usize> {
    let mut out = Vec::new();
    let _: ControlFlow<()> = index.visit(q, |i| {
        out.push(i);
        ControlFlow::Continue(())
    });
    out
}

fn visit_3d(index: &packed_spatial_index::Index3DF32, q: Box3D) -> Vec<usize> {
    let mut out = Vec::new();
    let _: ControlFlow<()> = index.visit(q, |i| {
        out.push(i);
        ControlFlow::Continue(())
    });
    out
}

#[test]
fn f32_search_with_covered_subtrees_matches_the_per_leaf_test_2d() {
    let mut s = 0x2D;
    let n = 20_000;
    let mut b = Index2DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let x = BASE + lcg(&mut s) * 1000.0;
        let y = BASE + lcg(&mut s) * 1000.0;
        let (w, h) = (lcg(&mut s) * 3.0, lcg(&mut s) * 3.0);
        b.add(Box2D::new(x, y, x + w, y + h));
    }
    let index = b.finish_f32().unwrap();

    let mut windows = vec![
        // Everything, and more than everything.
        Box2D::new(BASE - 10.0, BASE - 10.0, BASE + 1010.0, BASE + 1010.0),
        Box2D::new(BASE, BASE, BASE + 1000.0, BASE + 1000.0),
    ];
    for _ in 0..300 {
        let side = 1.0 + lcg(&mut s) * 600.0;
        let x = BASE + lcg(&mut s) * (1000.0 - side);
        let y = BASE + lcg(&mut s) * (1000.0 - side);
        // Nudge edges off the f32 grid so inward rounding actually moves them.
        let e = 0.013 + lcg(&mut s) * 0.03;
        windows.push(Box2D::new(x + e, y - e, x + side - e, y + side + e));
    }

    let mut covered_any = false;
    for q in windows {
        let mut reference = visit_2d(&index, q);
        reference.sort_unstable();
        let mut got = index.search(q);
        got.sort_unstable();
        assert_eq!(got, reference, "{q:?}");
        assert_eq!(index.count(q), reference.len(), "{q:?}");
        covered_any |= reference.len() > 64;
    }
    assert!(covered_any, "no window was wide enough to cover a subtree");
}

#[test]
fn f32_search_with_covered_subtrees_matches_the_per_leaf_test_3d() {
    let mut s = 0x3D;
    let n = 20_000;
    let mut b = Index3DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let x = BASE + lcg(&mut s) * 1000.0;
        let y = BASE + lcg(&mut s) * 1000.0;
        let z = BASE + lcg(&mut s) * 1000.0;
        let k = lcg(&mut s) * 8.0;
        b.add(Box3D::new(x, y, z, x + k, y + k, z + k));
    }
    let index = b.finish_f32().unwrap();

    let mut windows = vec![Box3D::new(
        BASE - 10.0,
        BASE - 10.0,
        BASE - 10.0,
        BASE + 1010.0,
        BASE + 1010.0,
        BASE + 1010.0,
    )];
    for _ in 0..200 {
        let side = 10.0 + lcg(&mut s) * 700.0;
        let (x, y, z) = (
            BASE + lcg(&mut s) * (1000.0 - side),
            BASE + lcg(&mut s) * (1000.0 - side),
            BASE + lcg(&mut s) * (1000.0 - side),
        );
        let e = 0.013 + lcg(&mut s) * 0.03;
        windows.push(Box3D::new(
            x + e,
            y - e,
            z + e,
            x + side - e,
            y + side + e,
            z + side - e,
        ));
    }

    for q in windows {
        let mut reference = visit_3d(&index, q);
        reference.sort_unstable();
        let mut got = index.search(q);
        got.sort_unstable();
        assert_eq!(got, reference, "{q:?}");
        assert_eq!(index.count(q), reference.len(), "{q:?}");
    }
}

/// The same boxes and windows as the scalar tests above, for the SIMD f32
/// `count` (owned and view), which adds a covered subtree's leaf range and a
/// leaf's hit popcount instead of visiting each item. Node size 12 leaves a
/// tail after the eight-lane blocks; 16 has none.
#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::{SimdIndex2DF32View, SimdIndex3DF32View};

    fn visit_len<F>(visit: F) -> usize
    where
        F: FnOnce(&mut dyn FnMut(usize) -> ControlFlow<()>) -> ControlFlow<()>,
    {
        let mut n = 0usize;
        let _ = visit(&mut |_| {
            n += 1;
            ControlFlow::Continue(())
        });
        n
    }

    #[test]
    fn simd_f32_count_with_covered_subtrees_matches_the_per_leaf_test_2d() {
        for node_size in [12, 16] {
            let mut s = 0x2D;
            let n = 20_000;
            let mut b = Index2DBuilder::new(n).node_size(node_size);
            for _ in 0..n {
                let x = BASE + lcg(&mut s) * 1000.0;
                let y = BASE + lcg(&mut s) * 1000.0;
                let (w, h) = (lcg(&mut s) * 3.0, lcg(&mut s) * 3.0);
                b.add(Box2D::new(x, y, x + w, y + h));
            }
            let index = b.finish_simd_f32().unwrap();
            let bytes = index.to_bytes();
            let view = SimdIndex2DF32View::from_bytes(&bytes).unwrap();

            let mut windows = vec![
                Box2D::new(BASE - 10.0, BASE - 10.0, BASE + 1010.0, BASE + 1010.0),
                Box2D::new(BASE, BASE, BASE + 1000.0, BASE + 1000.0),
            ];
            for _ in 0..300 {
                let side = 1.0 + lcg(&mut s) * 600.0;
                let x = BASE + lcg(&mut s) * (1000.0 - side);
                let y = BASE + lcg(&mut s) * (1000.0 - side);
                let e = 0.013 + lcg(&mut s) * 0.03;
                windows.push(Box2D::new(x + e, y - e, x + side - e, y + side + e));
            }

            let mut covered_any = false;
            for q in windows {
                let reference = visit_len(|f| index.visit(q, f));
                assert_eq!(index.count(q), reference, "owned {node_size} {q:?}");
                assert_eq!(view.count(q), reference, "view {node_size} {q:?}");
                assert_eq!(index.search(q).len(), reference, "search {node_size} {q:?}");
                covered_any |= reference > 64;
            }
            assert!(covered_any, "no window was wide enough to cover a subtree");
        }
    }

    #[test]
    fn simd_f32_count_with_covered_subtrees_matches_the_per_leaf_test_3d() {
        for node_size in [12, 16] {
            let mut s = 0x3D;
            let n = 20_000;
            let mut b = Index3DBuilder::new(n).node_size(node_size);
            for _ in 0..n {
                let x = BASE + lcg(&mut s) * 1000.0;
                let y = BASE + lcg(&mut s) * 1000.0;
                let z = BASE + lcg(&mut s) * 1000.0;
                let k = lcg(&mut s) * 8.0;
                b.add(Box3D::new(x, y, z, x + k, y + k, z + k));
            }
            let index = b.finish_simd_f32().unwrap();
            let bytes = index.to_bytes();
            let view = SimdIndex3DF32View::from_bytes(&bytes).unwrap();

            let mut windows = vec![Box3D::new(
                BASE - 10.0,
                BASE - 10.0,
                BASE - 10.0,
                BASE + 1010.0,
                BASE + 1010.0,
                BASE + 1010.0,
            )];
            for _ in 0..200 {
                let side = 10.0 + lcg(&mut s) * 700.0;
                let (x, y, z) = (
                    BASE + lcg(&mut s) * (1000.0 - side),
                    BASE + lcg(&mut s) * (1000.0 - side),
                    BASE + lcg(&mut s) * (1000.0 - side),
                );
                let e = 0.013 + lcg(&mut s) * 0.03;
                windows.push(Box3D::new(
                    x + e,
                    y - e,
                    z + e,
                    x + side - e,
                    y + side + e,
                    z + side - e,
                ));
            }

            let mut covered_any = false;
            for q in windows {
                let reference = visit_len(|f| index.visit(q, f));
                assert_eq!(index.count(q), reference, "owned {node_size} {q:?}");
                assert_eq!(view.count(q), reference, "view {node_size} {q:?}");
                assert_eq!(index.search(q).len(), reference, "search {node_size} {q:?}");
                covered_any |= reference > 64;
            }
            assert!(covered_any, "no window was wide enough to cover a subtree");
        }
    }
}
