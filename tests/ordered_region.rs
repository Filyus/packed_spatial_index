//! Ordered region queries: `search_ordered` / `visit_ordered` must return exactly
//! the set `search` returns, emit it in nondecreasing key order, honor the
//! `max_results` budget and the `max_key` cutoff, and agree between the owned
//! indexes and their byte views.

use std::ops::ControlFlow;

use packed_spatial_index::{
    Box2D, Box3D, ConvexPolygon2D, Frustum3D, Index2D, Index2DBuilder, Index2DView, Index3D,
    Index3DBuilder, Index3DView, view_depth_2d, view_depth_3d,
};

const NODE_SIZES: [usize; 2] = [4, 16];
const COUNTS: [usize; 6] = [0, 1, 5, 17, 64, 1000];

const EYE2: [f64; 2] = [-40.0, -30.0];
const DIR2: [f64; 2] = [0.6, 0.8];
const EYE3: [f64; 3] = [-40.0, -30.0, -20.0];
const DIR3: [f64; 3] = [0.6, 0.7, 0.4];

/// Deterministic scatter; the `i * 0.001` jitter keeps the view depths distinct,
/// so the budgeted `max_results` case has an unambiguous expected answer.
fn boxes2d(n: usize) -> Vec<Box2D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0 + i as f64 * 0.001;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            let w = 0.2 + ((i * 13) % 5) as f64;
            let h = 0.2 + ((i * 17) % 5) as f64;
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn boxes3d(n: usize) -> Vec<Box3D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0 + i as f64 * 0.001;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            let z = ((i * 5077) % 983) as f64 / 983.0 * 200.0;
            Box3D::new(x, y, z, x + 1.0, y + 1.5, z + 0.8)
        })
        .collect()
}

fn build2d(boxes: &[Box2D], node_size: usize) -> Index2D {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap()
}

fn build3d(boxes: &[Box3D], node_size: usize) -> Index3D {
    let mut b = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for bx in boxes {
        b.add(*bx);
    }
    b.finish().unwrap()
}

/// Six inward planes bounding the axis-aligned box `[lo, hi]^3`.
fn box_frustum(lo: f64, hi: f64) -> Frustum3D {
    Frustum3D::from_planes([
        [1.0, 0.0, 0.0, -lo],
        [-1.0, 0.0, 0.0, hi],
        [0.0, 1.0, 0.0, -lo],
        [0.0, -1.0, 0.0, hi],
        [0.0, 0.0, 1.0, -lo],
        [0.0, 0.0, -1.0, hi],
    ])
}

fn trapezoid() -> ConvexPolygon2D {
    ConvexPolygon2D::new(vec![
        [10.0, 10.0],
        [190.0, 30.0],
        [170.0, 180.0],
        [40.0, 150.0],
    ])
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

/// Brute-force answer: every box the region accepts, ordered by key then index.
fn brute<T: Copy>(
    boxes: &[T],
    overlaps: impl Fn(T) -> bool,
    key: impl Fn(T) -> f64,
) -> Vec<(usize, f64)> {
    let mut hits: Vec<(usize, f64)> = boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| overlaps(**b))
        .map(|(i, b)| (i, key(*b)))
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    hits
}

fn assert_nondecreasing(keys: &[f64]) {
    for pair in keys.windows(2) {
        assert!(
            pair[0] <= pair[1],
            "keys out of order: {} then {}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn ordered_matches_search_set_2d() {
    for n in COUNTS {
        let boxes = boxes2d(n);
        for node_size in NODE_SIZES {
            let index = build2d(&boxes, node_size);
            let window = Box2D::new(20.0, 20.0, 150.0, 160.0);
            let poly = trapezoid();

            for (label, hits, expected) in [
                (
                    "box",
                    index.search_ordered(
                        window,
                        |b| view_depth_2d(EYE2, DIR2, b),
                        usize::MAX,
                        f64::INFINITY,
                    ),
                    index.search(window),
                ),
                (
                    "polygon",
                    index.search_ordered(
                        &poly,
                        |b| view_depth_2d(EYE2, DIR2, b),
                        usize::MAX,
                        f64::INFINITY,
                    ),
                    index.search(&poly),
                ),
            ] {
                assert_eq!(
                    sorted(hits),
                    sorted(expected),
                    "{label}: n={n} node_size={node_size}"
                );
            }
        }
    }
}

#[test]
fn ordered_matches_search_set_3d() {
    for n in COUNTS {
        let boxes = boxes3d(n);
        for node_size in NODE_SIZES {
            let index = build3d(&boxes, node_size);
            let frustum = box_frustum(30.0, 170.0);

            let hits = index.search_ordered(
                frustum,
                |b| view_depth_3d(EYE3, DIR3, b),
                usize::MAX,
                f64::INFINITY,
            );
            assert_eq!(
                sorted(hits),
                sorted(index.search(&frustum)),
                "n={n} node_size={node_size}"
            );
        }
    }
}

#[test]
fn ordered_emits_nondecreasing_keys() {
    for n in COUNTS {
        let boxes3 = boxes3d(n);
        for node_size in NODE_SIZES {
            let index = build3d(&boxes3, node_size);
            let frustum = box_frustum(30.0, 170.0);
            let mut keys = Vec::new();
            let flow: ControlFlow<()> = index.visit_ordered(
                frustum,
                |b| view_depth_3d(EYE3, DIR3, b),
                f64::INFINITY,
                |_, key| {
                    keys.push(key);
                    ControlFlow::Continue(())
                },
            );
            assert!(flow.is_continue());
            assert_nondecreasing(&keys);
            assert_eq!(keys.len(), index.search(&frustum).len());
            // Nondecreasing alone is vacuous for a constant key, so pin the keys
            // themselves against the brute-force order.
            let want: Vec<f64> = brute(
                &boxes3,
                |b| frustum.overlaps_box(b),
                |b| view_depth_3d(EYE3, DIR3, b),
            )
            .iter()
            .map(|(_, key)| *key)
            .collect();
            assert_eq!(keys, want, "n={n} node_size={node_size}");
        }
    }

    let boxes2 = boxes2d(500);
    let index = build2d(&boxes2, 8);
    let window = Box2D::new(20.0, 20.0, 150.0, 160.0);
    let mut keys = Vec::new();
    let _: ControlFlow<()> = index.visit_ordered(
        window,
        |b| view_depth_2d(EYE2, DIR2, b),
        f64::INFINITY,
        |_, key| {
            keys.push(key);
            ControlFlow::Continue(())
        },
    );
    assert_nondecreasing(&keys);
}

#[test]
fn budget_returns_the_nearest_prefix() {
    let boxes = boxes3d(1000);
    let frustum = box_frustum(30.0, 170.0);
    let expected = brute(
        &boxes,
        |b| frustum.overlaps_box(b),
        |b| view_depth_3d(EYE3, DIR3, b),
    );

    for node_size in NODE_SIZES {
        let index = build3d(&boxes, node_size);
        for k in [1usize, 7, 50, expected.len()] {
            let hits =
                index.search_ordered(frustum, |b| view_depth_3d(EYE3, DIR3, b), k, f64::INFINITY);
            let want: Vec<usize> = expected.iter().take(k).map(|(i, _)| *i).collect();
            assert_eq!(hits, want, "k={k} node_size={node_size}");
        }
        // Asking for more than exists yields everything, still in order.
        let all = index.search_ordered(
            frustum,
            |b| view_depth_3d(EYE3, DIR3, b),
            usize::MAX,
            f64::INFINITY,
        );
        assert_eq!(all.len(), expected.len());
    }
}

#[test]
fn max_key_cuts_off_exactly() {
    let boxes = boxes3d(1000);
    let frustum = box_frustum(30.0, 170.0);
    let expected = brute(
        &boxes,
        |b| frustum.overlaps_box(b),
        |b| view_depth_3d(EYE3, DIR3, b),
    );
    let cutoff = expected[expected.len() / 3].1;
    let want: Vec<usize> = expected
        .iter()
        .filter(|(_, key)| *key <= cutoff)
        .map(|(i, _)| *i)
        .collect();

    let index = build3d(&boxes, 16);
    let hits = index.search_ordered(
        frustum,
        |b| view_depth_3d(EYE3, DIR3, b),
        usize::MAX,
        cutoff,
    );
    assert!(!want.is_empty() && want.len() < expected.len());
    assert_eq!(sorted(hits), sorted(want));
}

#[test]
fn visitor_can_break_early() {
    let boxes = boxes3d(1000);
    let index = build3d(&boxes, 16);
    let frustum = box_frustum(30.0, 170.0);

    let mut seen = 0usize;
    let flow = index.visit_ordered(
        frustum,
        |b| view_depth_3d(EYE3, DIR3, b),
        f64::INFINITY,
        |id, _| {
            seen += 1;
            if seen == 3 {
                ControlFlow::Break(id)
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    assert_eq!(seen, 3);
    let ordered = index.search_ordered(frustum, |b| view_depth_3d(EYE3, DIR3, b), 3, f64::INFINITY);
    assert_eq!(flow, ControlFlow::Break(ordered[2]));
}

#[test]
fn views_match_owned() {
    let boxes3 = boxes3d(1000);
    let owned3 = build3d(&boxes3, 16);
    let bytes3 = owned3.to_bytes();
    let view3 = Index3DView::from_bytes(&bytes3).unwrap();
    let frustum = box_frustum(30.0, 170.0);
    for k in [5usize, usize::MAX] {
        assert_eq!(
            view3.search_ordered(frustum, |b| view_depth_3d(EYE3, DIR3, b), k, f64::INFINITY),
            owned3.search_ordered(frustum, |b| view_depth_3d(EYE3, DIR3, b), k, f64::INFINITY),
            "3D k={k}"
        );
    }

    let boxes2 = boxes2d(1000);
    let owned2 = build2d(&boxes2, 16);
    let bytes2 = owned2.to_bytes();
    let view2 = Index2DView::from_bytes(&bytes2).unwrap();
    let poly = trapezoid();
    for k in [5usize, usize::MAX] {
        assert_eq!(
            view2.search_ordered(&poly, |b| view_depth_2d(EYE2, DIR2, b), k, f64::INFINITY),
            owned2.search_ordered(&poly, |b| view_depth_2d(EYE2, DIR2, b), k, f64::INFINITY),
            "2D k={k}"
        );
    }

    // The view's visitor path too.
    let mut keys = Vec::new();
    let _: ControlFlow<()> = view3.visit_ordered(
        frustum,
        |b| view_depth_3d(EYE3, DIR3, b),
        f64::INFINITY,
        |_, key| {
            keys.push(key);
            ControlFlow::Continue(())
        },
    );
    assert_nondecreasing(&keys);
    assert_eq!(keys.len(), owned3.search(&frustum).len());
}

#[test]
fn degenerate_inputs() {
    let empty = build3d(&[], 16);
    let frustum = box_frustum(30.0, 170.0);
    assert!(
        empty
            .search_ordered(
                frustum,
                |b| view_depth_3d(EYE3, DIR3, b),
                usize::MAX,
                f64::INFINITY
            )
            .is_empty()
    );

    let boxes = boxes3d(200);
    let index = build3d(&boxes, 16);
    let key = |b| view_depth_3d(EYE3, DIR3, b);

    assert!(
        index
            .search_ordered(frustum, key, 0, f64::INFINITY)
            .is_empty()
    );
    assert!(
        index
            .search_ordered(frustum, key, usize::MAX, f64::NAN)
            .is_empty()
    );
    // Every box is in front of the eye here, so a negative cutoff excludes all.
    assert!(
        index
            .search_ordered(frustum, key, usize::MAX, -1.0)
            .is_empty()
    );

    // A zero direction makes every key 0.0 — degenerate but still a valid
    // (constant) admissible bound, so the set is unchanged.
    let flat = index.search_ordered(
        frustum,
        |b| view_depth_3d(EYE3, [0.0; 3], b),
        usize::MAX,
        f64::INFINITY,
    );
    assert_eq!(sorted(flat), sorted(index.search(&frustum)));

    // Reused buffers are cleared first.
    let mut buf = vec![999usize; 5];
    index.search_ordered_into(frustum, key, 3, f64::INFINITY, &mut buf);
    assert_eq!(buf.len(), 3);
    assert!(!buf.contains(&999));
}

#[test]
fn view_depth_is_a_lower_bound_on_the_box() {
    let b = Box3D::new(1.0, 2.0, 3.0, 5.0, 7.0, 11.0);
    let eye = [0.0, 0.0, 0.0];

    // Along +x from the origin the depth is simply min_x.
    assert_eq!(view_depth_3d(eye, [1.0, 0.0, 0.0], b), 1.0);
    // Along -x it is -max_x.
    assert_eq!(view_depth_3d(eye, [-1.0, 0.0, 0.0], b), -5.0);
    assert_eq!(
        view_depth_2d(eye_2d(), [0.0, 1.0], Box2D::new(1.0, 2.0, 5.0, 7.0)),
        2.0
    );

    // Never exceeds the depth of any corner.
    let dir = [0.3, -0.7, 0.5];
    let depth = view_depth_3d(EYE3, dir, b);
    for &x in &[b.min_x, b.max_x] {
        for &y in &[b.min_y, b.max_y] {
            for &z in &[b.min_z, b.max_z] {
                let corner =
                    (x - EYE3[0]) * dir[0] + (y - EYE3[1]) * dir[1] + (z - EYE3[2]) * dir[2];
                assert!(depth <= corner, "{depth} > {corner}");
            }
        }
    }

    // A longer direction rescales the key without changing the order.
    let far = Box3D::new(50.0, 2.0, 3.0, 55.0, 7.0, 11.0);
    let unit = [1.0, 0.0, 0.0];
    let long = [3.0, 0.0, 0.0];
    assert!(view_depth_3d(EYE3, unit, b) < view_depth_3d(EYE3, unit, far));
    assert!(view_depth_3d(EYE3, long, b) < view_depth_3d(EYE3, long, far));
    assert_eq!(
        view_depth_3d(EYE3, long, b),
        3.0 * view_depth_3d(EYE3, unit, b)
    );
}

fn eye_2d() -> [f64; 2] {
    [0.0, 0.0]
}
