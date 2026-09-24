//! The scalar f32 `search` and `count` emit a subtree the query covers as its
//! leaf range, without testing each item. That is only sound if a covered node
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
