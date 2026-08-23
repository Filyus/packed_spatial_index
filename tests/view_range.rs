//! The zero-copy views' range search runs the contained-subtree traversal,
//! which emits a fully covered subtree without testing its items. That shortcut
//! is only correct if it fires on exactly the subtrees the query contains, and
//! a wrong containment predicate does not fail loudly — it silently returns
//! extra items. So compare against the owned index over windows of every size,
//! and assert the shortcut is actually exercised rather than trusting that a
//! window "looks big".

use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView,
};

const EXTENT: f64 = 1000.0;

/// Deterministic xorshift: a failure has to be reproducible.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Clustered, so subtrees are spatially tight and a mid-sized window really can
/// contain whole ones. Uniform data would make the fast path rare and the test
/// weak without saying so.
fn clustered(n: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = Rng(seed | 1);
    let centers: Vec<(f64, f64)> = (0..64)
        .map(|_| (rng.unit() * EXTENT, rng.unit() * EXTENT))
        .collect();
    (0..n)
        .map(|i| {
            let (cx, cy) = centers[i % centers.len()];
            (
                (cx + (rng.unit() - 0.5) * 30.0).clamp(0.0, EXTENT),
                (cy + (rng.unit() - 0.5) * 30.0).clamp(0.0, EXTENT),
            )
        })
        .collect()
}

fn windows(side: f64, count: usize, seed: u64) -> Vec<Box2D> {
    let mut rng = Rng(seed | 1);
    let span = EXTENT * side;
    (0..count)
        .map(|_| {
            let x = rng.unit() * (EXTENT - span).max(0.0);
            let y = rng.unit() * (EXTENT - span).max(0.0);
            Box2D::new(x, y, x + span, y + span)
        })
        .collect()
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

#[test]
fn view_range_search_matches_the_owned_index_2d() {
    let points = clustered(20_000, 0x5EED_0002);
    let mut builder = Index2DBuilder::new(points.len()).node_size(8);
    for &(x, y) in &points {
        builder.add(Box2D::new(x, y, x + 1.0, y + 1.5));
    }
    let owned = builder.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();

    let mut biggest = 0usize;
    for side in [0.0005, 0.01, 0.05, 0.15, 0.4, 1.0] {
        for q in windows(side, 24, (side * 1e6) as u64 + 7) {
            let expected = sorted(owned.search(q));
            assert_eq!(
                sorted(view.search(q)),
                expected,
                "search, side {side}, {q:?}"
            );
            let mut into = Vec::new();
            view.search_into(q, &mut into);
            assert_eq!(sorted(into), expected, "search_into, side {side}, {q:?}");
            assert_eq!(view.count(q), expected.len(), "count, side {side}, {q:?}");
            assert_eq!(view.any(q), !expected.is_empty(), "any, side {side}");
            assert_eq!(
                view.first(q).is_some(),
                !expected.is_empty(),
                "first, side {side}"
            );
            if let Some(first) = view.first(q) {
                assert!(expected.contains(&first), "first returned a non-hit");
            }
            biggest = biggest.max(expected.len());
        }
    }

    // Without this the test could pass on windows that never covered a whole
    // subtree — that is, without ever running the code it exists to check.
    assert!(
        biggest > points.len() / 2,
        "no window covered a large part of the tree ({biggest} hits), so the \
         contained-subtree path was never exercised"
    );
}

#[test]
fn view_range_search_matches_the_owned_index_3d() {
    let points = clustered(15_000, 0x5EED_0004);
    let mut builder = Index3DBuilder::new(points.len()).node_size(8);
    for (i, &(x, y)) in points.iter().enumerate() {
        let z = (i % 997) as f64 / 997.0 * EXTENT;
        builder.add(Box3D::new(x, y, z, x + 1.0, y + 1.5, z + 0.8));
    }
    let owned = builder.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();

    let mut biggest = 0usize;
    for side in [0.01, 0.15, 0.4, 1.0] {
        for q2 in windows(side, 16, (side * 1e6) as u64 + 11) {
            let q = Box3D::new(
                q2.min_x,
                q2.min_y,
                -1.0,
                q2.max_x,
                q2.max_y,
                EXTENT * side + 1.0,
            );
            let expected = sorted(owned.search(q));
            assert_eq!(sorted(view.search(q)), expected, "side {side}, {q:?}");
            assert_eq!(view.count(q), expected.len(), "count, side {side}");
            assert_eq!(view.any(q), !expected.is_empty(), "any, side {side}");
            biggest = biggest.max(expected.len());
        }
    }
    assert!(
        biggest > points.len() / 2,
        "no window covered a large part of the tree ({biggest} hits)"
    );
}

/// A query that contains the entire tree takes the root shortcut in both the
/// owned index and the view; it must still return every item exactly once.
#[test]
fn a_query_covering_the_root_returns_every_item_once() {
    let points = clustered(5_000, 0x5EED_0003);
    let mut builder = Index2DBuilder::new(points.len()).node_size(8);
    for &(x, y) in &points {
        builder.add(Box2D::new(x, y, x + 1.0, y + 1.0));
    }
    let owned = builder.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();

    let extent = owned.extent().unwrap();
    let covering = Box2D::new(
        extent.min_x - 1.0,
        extent.min_y - 1.0,
        extent.max_x + 1.0,
        extent.max_y + 1.0,
    );
    let hits = sorted(view.search(covering));
    assert_eq!(hits, (0..points.len()).collect::<Vec<_>>());
    assert_eq!(view.count(covering), points.len());
}
