//! 3D frustum culling queries on `Index3D`: the traversal must return exactly the
//! boxes the conservative `Frustum3D::overlaps_box` predicate accepts, the
//! contained-subtree fast path must not change that, and `from_view_projection`
//! must agree with the plane convention.

use packed_spatial_index::{Box3D, ClipSpaceZ, Frustum3D, Index3DBuilder};

/// Six inward planes bounding the axis-aligned box `[lo, hi]^3`.
fn box_frustum(lo: f64, hi: f64) -> Frustum3D {
    Frustum3D::from_planes([
        [1.0, 0.0, 0.0, -lo], // x >= lo
        [-1.0, 0.0, 0.0, hi], // x <= hi
        [0.0, 1.0, 0.0, -lo], // y >= lo
        [0.0, -1.0, 0.0, hi], // y <= hi
        [0.0, 0.0, 1.0, -lo], // z >= lo
        [0.0, 0.0, -1.0, hi], // z <= hi
    ])
}

fn scattered_boxes(n: usize) -> Vec<Box3D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            let z = ((i * 5077) % 983) as f64 / 983.0 * 200.0;
            let w = 0.2 + ((i * 13) % 5) as f64;
            let h = 0.2 + ((i * 17) % 5) as f64;
            let d = 0.2 + ((i * 19) % 5) as f64;
            Box3D::new(x, y, z, x + w, y + h, z + d)
        })
        .collect()
}

fn build(boxes: &[Box3D]) -> packed_spatial_index::Index3D {
    let mut builder = Index3DBuilder::new(boxes.len());
    for b in boxes {
        builder.add(*b);
    }
    builder.finish().unwrap()
}

#[test]
fn search_frustum_matches_predicate() {
    let boxes = scattered_boxes(4000);
    let index = build(&boxes);

    // A tilted frustum (not axis aligned) so the planes actually slant.
    let tilted = Frustum3D::from_planes([
        [1.0, 0.2, 0.0, -10.0],
        [-1.0, 0.1, 0.0, 150.0],
        [0.1, 1.0, 0.0, -10.0],
        [0.0, -1.0, 0.2, 150.0],
        [0.0, 0.1, 1.0, -10.0],
        [0.0, 0.0, -1.0, 150.0],
    ]);

    for frustum in [box_frustum(20.0, 160.0), box_frustum(0.0, 100.0), tilted] {
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| frustum.overlaps_box(**b))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let mut got = index.search_frustum(frustum);
        got.sort_unstable();
        assert_eq!(got, expected);

        // any agrees with non-empty; into matches owned.
        assert_eq!(index.any_frustum(frustum), !got.is_empty());
        let mut buf = vec![usize::MAX; 3];
        index.search_frustum_into(frustum, &mut buf);
        buf.sort_unstable();
        assert_eq!(buf, got);

        // Every reported box really passes the predicate (no leakage), and every
        // contained box is also an overlap (fast path stays sound).
        for &i in &got {
            assert!(frustum.overlaps_box(boxes[i]));
        }
    }
}

#[test]
fn search_frustum_contained_fast_path_is_correct() {
    let boxes = scattered_boxes(3000);
    let index = build(&boxes);

    // A frustum that swallows the whole field — exercises root + subtree accepts.
    let frustum = box_frustum(-1000.0, 1000.0);
    let mut got = index.search_frustum(frustum);
    got.sort_unstable();
    let all: Vec<usize> = (0..boxes.len())
        .filter(|&i| frustum.overlaps_box(boxes[i]))
        .collect();
    assert_eq!(got, all);
    assert_eq!(got.len(), boxes.len(), "all boxes lie inside");

    let (results, _visited, _planes, contained) = index.search_frustum_visited(frustum);
    assert_eq!(results, boxes.len());
    assert!(contained > 0, "expected contained subtrees to be accepted");
}

#[test]
fn from_view_projection_identity_is_ndc_cube() {
    // vp = identity => Gribb-Hartmann yields the clip cube [-1, 1]^3.
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let frustum = Frustum3D::from_view_projection(identity, ClipSpaceZ::NegOneToOne);

    let inside = Box3D::new(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5);
    let outside = Box3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0);
    let straddle = Box3D::new(0.9, 0.9, 0.9, 1.5, 1.5, 1.5);

    assert!(frustum.contains_box(inside));
    assert!(frustum.overlaps_box(inside));
    assert!(!frustum.overlaps_box(outside));
    assert!(frustum.overlaps_box(straddle));
    assert!(!frustum.contains_box(straddle));

    // And it drives a query.
    let boxes = vec![inside, outside, straddle];
    let index = build(&boxes);
    let mut got = index.search_frustum(frustum);
    got.sort_unstable();
    assert_eq!(got, vec![0, 2]);
}

#[test]
fn search_frustum_empty_index() {
    let index = Index3DBuilder::new(0).finish().unwrap();
    let frustum = box_frustum(0.0, 1.0);
    assert!(index.search_frustum(frustum).is_empty());
    assert!(!index.any_frustum(frustum));
}
