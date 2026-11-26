use std::ops::ControlFlow;

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView, NeighborWorkspace,
    Point2D, Point3D, Ray2D, Ray3D, view_depth_2d, view_depth_3d,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn each_with_matches_each_2d_across_reused_workspace() {
    let mut rng = StdRng::seed_from_u64(0xEACB_0002);
    let mut b = Index2DBuilder::new(200);
    for _ in 0..200 {
        let x: f64 = rng.random_range(-50.0..50.0);
        let y: f64 = rng.random_range(-50.0..50.0);
        b.add(Box2D::new(x, y, x + 2.0, y + 2.0));
    }
    let idx = b.finish().unwrap();
    let bytes = idx.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    let mut ws = NeighborWorkspace::new();

    for i in 0..10 {
        let pt = Point2D::new(i as f64 * 5.0 - 20.0, i as f64 * 3.0 - 15.0);
        let mut a = Vec::new();
        let _ = idx.neighbors_each(pt, f64::INFINITY, |id, d2| {
            a.push((id, d2));
            if a.len() == 12 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut b_owned = Vec::new();
        let _ = idx.neighbors_each_with(pt, f64::INFINITY, &mut ws, |id, d2| {
            b_owned.push((id, d2));
            if b_owned.len() == 12 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut b_view = Vec::new();
        let _ = view.neighbors_each_with(pt, f64::INFINITY, &mut ws, |id, d2| {
            b_view.push((id, d2));
            if b_view.len() == 12 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(a, b_owned);
        assert_eq!(a, b_view);

        let qbox = Box2D::new(pt.x - 1.0, pt.y - 1.0, pt.x + 1.0, pt.y + 1.0);
        let mut ba = Vec::new();
        let _ = idx.neighbors_of_box_each(qbox, f64::INFINITY, |id, d2| {
            ba.push((id, d2));
            if ba.len() == 12 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut bb_owned = Vec::new();
        let _ = idx.neighbors_of_box_each_with(qbox, f64::INFINITY, &mut ws, |id, d2| {
            bb_owned.push((id, d2));
            if bb_owned.len() == 12 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut bb_view = Vec::new();
        let _ = view.neighbors_of_box_each_with(qbox, f64::INFINITY, &mut ws, |id, d2| {
            bb_view.push((id, d2));
            if bb_view.len() == 12 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(ba, bb_owned);
        assert_eq!(ba, bb_view);

        let ray = Ray2D::new(pt, 1.0, 0.3, 100.0);
        let mut ra = Vec::new();
        let _ = idx.raycast_each(ray, |id, t| {
            ra.push((id, t));
            ControlFlow::<()>::Continue(())
        });
        let mut rb = Vec::new();
        let _ = idx.raycast_each_with(ray, &mut ws, |id, t| {
            rb.push((id, t));
            ControlFlow::<()>::Continue(())
        });
        let mut rv = Vec::new();
        let _ = view.raycast_each_with(ray, &mut ws, |id, t| {
            rv.push((id, t));
            ControlFlow::<()>::Continue(())
        });
        assert_eq!(ra, rb);
        assert_eq!(ra, rv);

        let region = Box2D::new(-40.0, -40.0, 40.0, 40.0);
        let mut oa = Vec::new();
        let _ = idx.search_ordered_each(
            region,
            |bx| view_depth_2d([pt.x, pt.y], [1.0, 0.0], bx),
            f64::INFINITY,
            |id, k| {
                oa.push((id, k));
                if oa.len() == 10 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        let mut ob = Vec::new();
        let _ = idx.search_ordered_each_with(
            region,
            |bx| view_depth_2d([pt.x, pt.y], [1.0, 0.0], bx),
            f64::INFINITY,
            &mut ws,
            |id, k| {
                ob.push((id, k));
                if ob.len() == 10 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        assert_eq!(oa, ob);
    }
}

#[test]
fn each_with_matches_each_3d_across_reused_workspace() {
    let mut rng = StdRng::seed_from_u64(0xEACB_0003);
    let mut b = Index3DBuilder::new(200);
    for _ in 0..200 {
        let x: f64 = rng.random_range(-30.0..30.0);
        let y: f64 = rng.random_range(-30.0..30.0);
        let z: f64 = rng.random_range(-30.0..30.0);
        b.add(Box3D::new(x, y, z, x + 2.0, y + 2.0, z + 2.0));
    }
    let idx = b.finish().unwrap();
    let bytes = idx.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let mut ws = NeighborWorkspace::new();

    for i in 0..10 {
        let pt = Point3D::new(i as f64 * 4.0 - 15.0, i as f64 * 2.0 - 10.0, 0.0);
        let mut a = Vec::new();
        let _ = idx.neighbors_each(pt, f64::INFINITY, |id, d2| {
            a.push((id, d2));
            if a.len() == 15 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut b_owned = Vec::new();
        let _ = idx.neighbors_each_with(pt, f64::INFINITY, &mut ws, |id, d2| {
            b_owned.push((id, d2));
            if b_owned.len() == 15 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut b_view = Vec::new();
        let _ = view.neighbors_each_with(pt, f64::INFINITY, &mut ws, |id, d2| {
            b_view.push((id, d2));
            if b_view.len() == 15 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(a, b_owned);
        assert_eq!(a, b_view);

        let qbox = Box3D::new(
            pt.x - 1.0,
            pt.y - 1.0,
            pt.z - 1.0,
            pt.x + 1.0,
            pt.y + 1.0,
            pt.z + 1.0,
        );
        let mut ba = Vec::new();
        let _ = idx.neighbors_of_box_each(qbox, f64::INFINITY, |id, d2| {
            ba.push((id, d2));
            if ba.len() == 15 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut bb_owned = Vec::new();
        let _ = idx.neighbors_of_box_each_with(qbox, f64::INFINITY, &mut ws, |id, d2| {
            bb_owned.push((id, d2));
            if bb_owned.len() == 15 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let mut bb_view = Vec::new();
        let _ = view.neighbors_of_box_each_with(qbox, f64::INFINITY, &mut ws, |id, d2| {
            bb_view.push((id, d2));
            if bb_view.len() == 15 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(ba, bb_owned);
        assert_eq!(ba, bb_view);

        let ray = Ray3D::new(pt, 1.0, 0.2, -0.3, 100.0);
        let mut ra = Vec::new();
        let _ = idx.raycast_each(ray, |id, t| {
            ra.push((id, t));
            ControlFlow::<()>::Continue(())
        });
        let mut rb = Vec::new();
        let _ = idx.raycast_each_with(ray, &mut ws, |id, t| {
            rb.push((id, t));
            ControlFlow::<()>::Continue(())
        });
        let mut rv = Vec::new();
        let _ = view.raycast_each_with(ray, &mut ws, |id, t| {
            rv.push((id, t));
            ControlFlow::<()>::Continue(())
        });
        assert_eq!(ra, rb);
        assert_eq!(ra, rv);

        let region = Box3D::new(-30.0, -30.0, -30.0, 30.0, 30.0, 30.0);
        let mut oa = Vec::new();
        let _ = idx.search_ordered_each(
            region,
            |bx| view_depth_3d([pt.x, pt.y, pt.z], [1.0, 0.0, 0.0], bx),
            f64::INFINITY,
            |id, k| {
                oa.push((id, k));
                if oa.len() == 10 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        let mut ob = Vec::new();
        let _ = idx.search_ordered_each_with(
            region,
            |bx| view_depth_3d([pt.x, pt.y, pt.z], [1.0, 0.0, 0.0], bx),
            f64::INFINITY,
            &mut ws,
            |id, k| {
                ob.push((id, k));
                if ob.len() == 10 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        assert_eq!(oa, ob);
    }
}
