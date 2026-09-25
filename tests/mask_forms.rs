//! Every collect path keeps both child tests -- the branch-free mask the
//! shipping code runs and the per-child branch it replaced -- behind a
//! `const MASKED: bool`, so the two can be timed in one binary on a target the
//! mask was never measured on (`benches/paired_mask_forms.rs`). That is only
//! sound while they agree, so pin it: same items, same order, and the same as
//! the public search.

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView, Point3D, Ray3D,
};

const N: usize = 3_000;

fn boxes_2d() -> Vec<Box2D> {
    // A deterministic scatter with overlap, no RNG dependency needed.
    (0..N)
        .map(|i| {
            let x = ((i * 7919) % 1000) as f64;
            let y = ((i * 104_729) % 1000) as f64;
            let s = 1.0 + (i % 17) as f64;
            Box2D::new(x, y, x + s, y + s)
        })
        .collect()
}

fn boxes_3d() -> Vec<Box3D> {
    (0..N)
        .map(|i| {
            let x = ((i * 7919) % 1000) as f64;
            let y = ((i * 104_729) % 1000) as f64;
            let z = ((i * 1_299_709) % 1000) as f64;
            let s = 1.0 + (i % 23) as f64;
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect()
}

fn windows_2d() -> Vec<Box2D> {
    vec![
        Box2D::new(0.0, 0.0, 1000.0, 1000.0),
        Box2D::new(100.0, 100.0, 180.0, 160.0),
        Box2D::new(500.0, 20.0, 900.0, 700.0),
        Box2D::new(2000.0, 2000.0, 2100.0, 2100.0),
        Box2D::new(333.0, 333.0, 333.0, 333.0),
    ]
}

fn windows_3d() -> Vec<Box3D> {
    vec![
        Box3D::new(0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0),
        Box3D::new(100.0, 100.0, 100.0, 300.0, 250.0, 400.0),
        Box3D::new(600.0, 10.0, 200.0, 950.0, 500.0, 800.0),
        Box3D::new(2000.0, 2000.0, 2000.0, 2100.0, 2100.0, 2100.0),
    ]
}

#[test]
fn owned_2d_and_views_agree_in_both_forms() {
    let mut b = Index2DBuilder::new(N).node_size(16);
    for bx in boxes_2d() {
        b.add(bx);
    }
    let owned = b.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    let (mut a, mut m, mut shipped) = (Vec::new(), Vec::new(), Vec::new());
    for q in windows_2d() {
        owned.search_into_forced::<false>(q, &mut a);
        owned.search_into_forced::<true>(q, &mut m);
        owned.search_into(q, &mut shipped);
        assert_eq!(a, m, "owned 2d {q:?}");
        assert_eq!(m, shipped, "owned 2d shipped {q:?}");

        view.search_into_forced::<false>(q, &mut a);
        view.search_into_forced::<true>(q, &mut m);
        view.search_into(q, &mut shipped);
        assert_eq!(a, m, "view 2d {q:?}");
        assert_eq!(m, shipped, "view 2d shipped {q:?}");
    }
}

#[test]
fn owned_2d_callback_paths_agree_in_both_forms() {
    use std::ops::ControlFlow;
    // Node size 128 takes a node in two mask chunks.
    for node_size in [16, 128] {
        let mut b = Index2DBuilder::new(N).node_size(node_size);
        for bx in boxes_2d() {
            b.add(bx);
        }
        let owned = b.finish().unwrap();
        let mut stack = Vec::new();
        for q in windows_2d() {
            let (mut a, mut m) = (Vec::new(), Vec::new());
            let _ = owned.visit_with_stack_forced::<false, (), _>(q, &mut stack, |i| {
                a.push(i);
                ControlFlow::Continue(())
            });
            let _ = owned.visit_with_stack_forced::<true, (), _>(q, &mut stack, |i| {
                m.push(i);
                ControlFlow::Continue(())
            });
            assert_eq!(a, m, "visit {node_size} {q:?}");
            let first_b =
                owned.visit_with_stack_forced::<false, _, _>(q, &mut stack, ControlFlow::Break);
            let first_m =
                owned.visit_with_stack_forced::<true, _, _>(q, &mut stack, ControlFlow::Break);
            assert_eq!(first_b, first_m, "first {node_size} {q:?}");
            let find_b =
                owned.find_with_stack_forced::<false, _, _>(q, &mut stack, ControlFlow::Break);
            let find_m =
                owned.find_with_stack_forced::<true, _, _>(q, &mut stack, ControlFlow::Break);
            assert_eq!(find_b, first_b, "find branching {node_size} {q:?}");
            assert_eq!(find_m, first_b, "find masked {node_size} {q:?}");
            assert_eq!(
                owned.first(q),
                a.first().copied(),
                "shipped first {node_size} {q:?}"
            );
            assert_eq!(owned.any(q), !a.is_empty(), "shipped any {node_size} {q:?}");
        }
    }
}

#[test]
fn view_3d_and_raycast_agree_in_both_forms() {
    let mut b = Index3DBuilder::new(N).node_size(16);
    for bx in boxes_3d() {
        b.add(bx);
    }
    let owned = b.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let (mut a, mut m, mut shipped) = (Vec::new(), Vec::new(), Vec::new());
    for q in windows_3d() {
        view.search_into_forced::<false>(q, &mut a);
        view.search_into_forced::<true>(q, &mut m);
        view.search_into(q, &mut shipped);
        assert_eq!(a, m, "view 3d {q:?}");
        assert_eq!(m, shipped, "view 3d shipped {q:?}");
    }
    for i in 0..40 {
        let f = i as f64;
        let ray = Ray3D::new(
            Point3D::new(f * 25.0, 1000.0 - f * 20.0, -5.0),
            0.1 + f * 0.01,
            -0.05,
            1.0,
            3000.0,
        );
        owned.raycast_into_forced::<false>(ray, &mut a);
        owned.raycast_into_forced::<true>(ray, &mut m);
        owned.raycast_into(ray, &mut shipped);
        assert_eq!(a, m, "raycast {i}");
        assert_eq!(m, shipped, "raycast shipped {i}");
    }
}

#[cfg(feature = "f32-storage")]
#[test]
fn f32_indexes_agree_in_both_forms() {
    let mut b = Index2DBuilder::new(N).node_size(16);
    for bx in boxes_2d() {
        b.add(bx);
    }
    let f2 = b.finish_f32().unwrap();
    for q in windows_2d() {
        let a = f2.search_forced::<false>(q);
        assert_eq!(a, f2.search_forced::<true>(q), "f32 2d {q:?}");
        assert_eq!(a, f2.search(q), "f32 2d shipped {q:?}");
    }
    let mut b = Index3DBuilder::new(N).node_size(16);
    for bx in boxes_3d() {
        b.add(bx);
    }
    let f3 = b.finish_f32().unwrap();
    for q in windows_3d() {
        let a = f3.search_forced::<false>(q);
        assert_eq!(a, f3.search_forced::<true>(q), "f32 3d {q:?}");
        assert_eq!(a, f3.search(q), "f32 3d shipped {q:?}");
    }
}
