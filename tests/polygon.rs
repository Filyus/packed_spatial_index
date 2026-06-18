//! 2D convex-polygon queries on `Index2D`: the traversal must return exactly the
//! boxes the exact `ConvexPolygon2D::overlaps_box` predicate accepts, the
//! contained fast path must not change that, and a 3-vertex polygon must agree
//! with the specialized `search_triangle`.

use packed_spatial_index::{Box2D, ConvexPolygon2D, Index2D, Index2DBuilder, Triangle2D};

fn scattered_boxes(n: usize) -> Vec<Box2D> {
    (0..n)
        .map(|i| {
            let x = ((i * 7919) % 977) as f64 / 977.0 * 200.0;
            let y = ((i * 6121) % 991) as f64 / 991.0 * 200.0;
            let w = 0.2 + ((i * 13) % 5) as f64;
            let h = 0.2 + ((i * 17) % 5) as f64;
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn build(boxes: &[Box2D]) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len());
    for b in boxes {
        builder.add(*b);
    }
    builder.finish().unwrap()
}

#[test]
fn search_polygon_matches_predicate() {
    let boxes = scattered_boxes(4000);
    let index = build(&boxes);

    let trapezoid = ConvexPolygon2D::new(vec![
        [10.0, 10.0],
        [190.0, 30.0],
        [170.0, 180.0],
        [40.0, 150.0],
    ]);
    let hexagon = ConvexPolygon2D::new(vec![
        [100.0, 20.0],
        [170.0, 60.0],
        [170.0, 140.0],
        [100.0, 180.0],
        [30.0, 140.0],
        [30.0, 60.0],
    ]);
    let triangle = ConvexPolygon2D::new(vec![[0.0, 0.0], [150.0, 0.0], [0.0, 150.0]]);

    for poly in [&trapezoid, &hexagon, &triangle] {
        let mut expected: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| poly.overlaps_box(**b))
            .map(|(i, _)| i)
            .collect();
        expected.sort_unstable();

        let mut got = index.search_polygon(poly);
        got.sort_unstable();
        assert_eq!(got, expected);

        assert_eq!(index.any_polygon(poly), !got.is_empty());
        let mut buf = vec![usize::MAX; 3];
        index.search_polygon_into(poly, &mut buf);
        buf.sort_unstable();
        assert_eq!(buf, got);
    }
}

#[test]
fn polygon_with_three_vertices_equals_search_triangle() {
    let boxes = scattered_boxes(4000);
    let index = build(&boxes);

    let verts = [[5.0, 5.0], [180.0, 25.0], [60.0, 175.0]];
    let poly = ConvexPolygon2D::new(verts.to_vec());
    let tri = Triangle2D::new(verts[0], verts[1], verts[2]);

    let mut a = index.search_polygon(&poly);
    let mut b = index.search_triangle(tri);
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}

#[test]
fn search_polygon_contained_fast_path_is_correct() {
    let boxes = scattered_boxes(3000);
    let index = build(&boxes);

    // A polygon swallowing the whole field — exercises root + subtree accepts.
    let poly = ConvexPolygon2D::new(vec![
        [-1000.0, -1000.0],
        [1000.0, -1000.0],
        [1000.0, 1000.0],
        [-1000.0, 1000.0],
    ]);
    let mut got = index.search_polygon(&poly);
    got.sort_unstable();
    let all: Vec<usize> = (0..boxes.len())
        .filter(|&i| poly.overlaps_box(boxes[i]))
        .collect();
    assert_eq!(got, all);
    assert_eq!(got.len(), boxes.len());

    let (results, _v, _s, contained) = index.search_polygon_visited(&poly);
    assert_eq!(results, boxes.len());
    assert!(contained > 0, "expected contained subtrees to be accepted");
}

#[test]
fn degenerate_polygon_behaves_like_its_predicate() {
    let boxes = scattered_boxes(500);
    let index = build(&boxes);

    // A collinear (zero-area) polygon is a segment: `overlaps_box` is the exact
    // SAT (no filled area), so the query equals the brute-force predicate, and
    // `contains_box` never fires (no fast-accept) — same as a degenerate triangle.
    let collinear = ConvexPolygon2D::new(vec![[0.0, 0.0], [80.0, 80.0], [160.0, 160.0]]);
    let mut got = index.search_polygon(&collinear);
    got.sort_unstable();
    let mut expected: Vec<usize> = (0..boxes.len())
        .filter(|&i| collinear.overlaps_box(boxes[i]))
        .collect();
    expected.sort_unstable();
    assert_eq!(got, expected);
    let (_r, _v, _s, contained) = index.search_polygon_visited(&collinear);
    assert_eq!(
        contained, 0,
        "a zero-area polygon must not fast-accept subtrees"
    );

    // Fewer than three vertices is not a region: matches nothing.
    let two = ConvexPolygon2D::new(vec![[0.0, 0.0], [100.0, 100.0]]);
    assert!(index.search_polygon(&two).is_empty());
    assert!(!index.any_polygon(&two));
}

#[test]
fn search_polygon_empty_index() {
    let index = Index2DBuilder::new(0).finish().unwrap();
    let poly = ConvexPolygon2D::new(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    assert!(index.search_polygon(&poly).is_empty());
    assert!(!index.any_polygon(&poly));
}
