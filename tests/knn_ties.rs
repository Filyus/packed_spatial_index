//! Asking for one nearest neighbour must answer the same item as asking for
//! several and taking the first.
//!
//! This is not a free-standing wish: `max_results == 1` is routed to a separate
//! traversal (`best_first::nearest_one`) that keeps no item queue, while every
//! larger `k` goes through the two-queue kernel. Two kernels answering one
//! question have to agree, and the only place they can disagree is a tie.
//!
//! The rule both must obey is the one the two-queue kernel's heap order already
//! encodes: among items at equal distance, the smaller item index wins. That
//! makes `neighbors(q, k)` a prefix of `neighbors(q, k + 1)` for every `k`,
//! which is the property a caller actually leans on.
//!
//! Every fixture here puts all four items at distance exactly zero, so the tie
//! set is the whole index and the expected first answer is item `0` whatever
//! order the boxes were inserted in.

use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index3DBuilder, Point2D, Point3D};

const NODE_SIZES: [usize; 3] = [2, 4, 16];

/// All 24 orderings of four items, so insertion order cannot be what decides.
fn permutations4() -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let p = [a, b, c, d];
                    let mut seen = [false; 4];
                    if p.iter().all(|&i| {
                        let fresh = !seen[i];
                        seen[i] = true;
                        fresh
                    }) {
                        out.push(p);
                    }
                }
            }
        }
    }
    assert_eq!(out.len(), 24);
    out
}

/// Four boxes with distinct centres that all contain the origin, so every
/// distance is zero while the Hilbert order among them is not degenerate.
fn tie_boxes_2d() -> [Box2D; 4] {
    [
        Box2D::new(-1.0, -1.0, 0.5, 0.5),
        Box2D::new(-0.5, -0.5, 1.0, 1.0),
        Box2D::new(-1.0, -0.5, 1.0, 0.5),
        Box2D::new(-0.5, -1.0, 0.5, 1.0),
    ]
}

fn tie_boxes_3d() -> [Box3D; 4] {
    [
        Box3D::new(-1.0, -1.0, -1.0, 0.5, 0.5, 0.5),
        Box3D::new(-0.5, -0.5, -0.5, 1.0, 1.0, 1.0),
        Box3D::new(-1.0, -0.5, -1.0, 1.0, 0.5, 1.0),
        Box3D::new(-0.5, -1.0, -0.5, 0.5, 1.0, 0.5),
    ]
}

/// `one` is what the k = 1 traversal answered, `many` what the k > 1 kernel did.
fn check(frontend: &str, node_size: usize, perm: &[usize; 4], one: &[usize], many: &[usize]) {
    assert_eq!(
        many.len(),
        4,
        "{frontend}: every item is at distance 0, so all four should come back \
         (node_size {node_size}, insertion order {perm:?})"
    );
    assert_eq!(
        many[0], 0,
        "{frontend}: among equal distances the smallest item index must come \
         first (node_size {node_size}, insertion order {perm:?}), got {many:?}"
    );
    assert_eq!(
        one,
        &many[..1],
        "{frontend}: neighbors(q, 1) must equal the first of neighbors(q, 4) \
         (node_size {node_size}, insertion order {perm:?})"
    );
}

#[test]
fn k1_agrees_with_k_many_on_ties_2d() {
    let boxes = tie_boxes_2d();
    let point = Point2D { x: 0.0, y: 0.0 };
    let query = Box2D::new(0.0, 0.0, 0.0, 0.0);

    for node_size in NODE_SIZES {
        for perm in permutations4() {
            let mut builder = Index2DBuilder::new(4).node_size(node_size);
            for &i in perm.iter() {
                builder.add(boxes[i]);
            }
            let owned = builder.finish().unwrap();

            check(
                "Index2D point",
                node_size,
                &perm,
                &owned.neighbors(point, 1),
                &owned.neighbors(point, 4),
            );
            check(
                "Index2D box",
                node_size,
                &perm,
                &owned.neighbors_of_box(query, 1),
                &owned.neighbors_of_box(query, 4),
            );

            let bytes = owned.to_bytes();
            let view = packed_spatial_index::Index2DView::from_bytes(&bytes).unwrap();
            check(
                "Index2DView point",
                node_size,
                &perm,
                &view.neighbors(point, 1),
                &view.neighbors(point, 4),
            );
            check(
                "Index2DView box",
                node_size,
                &perm,
                &view.neighbors_of_box(query, 1),
                &view.neighbors_of_box(query, 4),
            );
        }
    }
}

#[test]
fn k1_agrees_with_k_many_on_ties_3d() {
    let boxes = tie_boxes_3d();
    let point = Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let query = Box3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for node_size in NODE_SIZES {
        for perm in permutations4() {
            let mut builder = Index3DBuilder::new(4).node_size(node_size);
            for &i in perm.iter() {
                builder.add(boxes[i]);
            }
            let owned = builder.finish().unwrap();

            check(
                "Index3D point",
                node_size,
                &perm,
                &owned.neighbors(point, 1),
                &owned.neighbors(point, 4),
            );
            check(
                "Index3D box",
                node_size,
                &perm,
                &owned.neighbors_of_box(query, 1),
                &owned.neighbors_of_box(query, 4),
            );

            let bytes = owned.to_bytes();
            let view = packed_spatial_index::Index3DView::from_bytes(&bytes).unwrap();
            check(
                "Index3DView point",
                node_size,
                &perm,
                &view.neighbors(point, 1),
                &view.neighbors(point, 4),
            );
        }
    }
}

/// The prefix property is what the tie rule buys, so state it directly:
/// growing `k` may only append.
#[test]
fn growing_k_only_appends() {
    let boxes = tie_boxes_2d();
    let point = Point2D { x: 0.0, y: 0.0 };

    for node_size in NODE_SIZES {
        for perm in permutations4() {
            let mut builder = Index2DBuilder::new(4).node_size(node_size);
            for &i in perm.iter() {
                builder.add(boxes[i]);
            }
            let index = builder.finish().unwrap();

            for k in 1..4 {
                let short = index.neighbors(point, k);
                let long = index.neighbors(point, k + 1);
                assert!(
                    long.starts_with(&short),
                    "neighbors(q, {k}) must be a prefix of neighbors(q, {}) \
                     (node_size {node_size}, insertion order {perm:?}): {short:?} vs {long:?}",
                    k + 1
                );
            }
        }
    }
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::{SimdIndex2DView, SimdIndex3DView};

    #[test]
    fn k1_agrees_with_k_many_on_ties_simd_2d() {
        let boxes = tie_boxes_2d();
        let point = Point2D { x: 0.0, y: 0.0 };
        let query = Box2D::new(0.0, 0.0, 0.0, 0.0);

        for node_size in NODE_SIZES {
            for perm in permutations4() {
                let mut builder = Index2DBuilder::new(4).node_size(node_size);
                for &i in perm.iter() {
                    builder.add(boxes[i]);
                }
                let simd = builder.finish_simd().unwrap();

                check(
                    "SimdIndex2D point",
                    node_size,
                    &perm,
                    &simd.neighbors(point, 1),
                    &simd.neighbors(point, 4),
                );
                check(
                    "SimdIndex2D box",
                    node_size,
                    &perm,
                    &simd.neighbors_of_box(query, 1),
                    &simd.neighbors_of_box(query, 4),
                );

                let bytes = simd.to_bytes();
                let view = SimdIndex2DView::from_bytes(&bytes).unwrap();
                check(
                    "SimdIndex2DView point",
                    node_size,
                    &perm,
                    &view.neighbors(point, 1),
                    &view.neighbors(point, 4),
                );
                check(
                    "SimdIndex2DView box",
                    node_size,
                    &perm,
                    &view.neighbors_of_box(query, 1),
                    &view.neighbors_of_box(query, 4),
                );
            }
        }
    }

    #[test]
    fn k1_agrees_with_k_many_on_ties_simd_3d() {
        let boxes = tie_boxes_3d();
        let point = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };

        for node_size in NODE_SIZES {
            for perm in permutations4() {
                let mut builder = Index3DBuilder::new(4).node_size(node_size);
                for &i in perm.iter() {
                    builder.add(boxes[i]);
                }
                let simd = builder.finish_simd().unwrap();

                check(
                    "SimdIndex3D point",
                    node_size,
                    &perm,
                    &simd.neighbors(point, 1),
                    &simd.neighbors(point, 4),
                );

                let bytes = simd.to_bytes();
                let view = SimdIndex3DView::from_bytes(&bytes).unwrap();
                check(
                    "SimdIndex3DView point",
                    node_size,
                    &perm,
                    &view.neighbors(point, 1),
                    &view.neighbors(point, 4),
                );
            }
        }
    }
}

#[cfg(feature = "f32-storage")]
mod f32_storage {
    use super::*;

    #[test]
    fn k1_agrees_with_k_many_on_ties_f32() {
        let boxes2 = tie_boxes_2d();
        let boxes3 = tie_boxes_3d();
        let point2 = Point2D { x: 0.0, y: 0.0 };
        let point3 = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };

        for node_size in NODE_SIZES {
            for perm in permutations4() {
                let mut builder = Index2DBuilder::new(4).node_size(node_size);
                for &i in perm.iter() {
                    builder.add(boxes2[i]);
                }
                let index2 = builder.finish_f32().unwrap();
                check(
                    "Index2DF32 point",
                    node_size,
                    &perm,
                    &index2.neighbors(point2, 1),
                    &index2.neighbors(point2, 4),
                );

                let mut builder = Index3DBuilder::new(4).node_size(node_size);
                for &i in perm.iter() {
                    builder.add(boxes3[i]);
                }
                let index3 = builder.finish_f32().unwrap();
                check(
                    "Index3DF32 point",
                    node_size,
                    &perm,
                    &index3.neighbors(point3, 1),
                    &index3.neighbors(point3, 4),
                );
            }
        }
    }
}
