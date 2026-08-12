//! The canonical flatbush fixture: the 100 boxes every port of the packed Hilbert R-tree
//! (flatbush, `static_aabb2d_index`, FlatGeobuf) tests against, with the expected results
//! those ports agree on. The expectations here are written as literals rather than taken
//! from a reference implementation, so they pin behaviour that a shared bug could not.

use packed_spatial_index::{Box2D, Index2D, Index2DBuilder, Point2D};

/// `[min_x, min_y, max_x, max_y]` quadruples, flattened, from the flatbush test suite.
const DATA: [f64; 400] = [
    8.0, 62.0, 11.0, 66.0, 57.0, 17.0, 57.0, 19.0, 76.0, 26.0, 79.0, 29.0, 36.0, 56.0, 38.0, 56.0,
    92.0, 77.0, 96.0, 80.0, 87.0, 70.0, 90.0, 74.0, 43.0, 41.0, 47.0, 43.0, 0.0, 58.0, 2.0, 62.0,
    76.0, 86.0, 80.0, 89.0, 27.0, 13.0, 27.0, 15.0, 71.0, 63.0, 75.0, 67.0, 25.0, 2.0, 27.0, 2.0,
    87.0, 6.0, 88.0, 6.0, 22.0, 90.0, 23.0, 93.0, 22.0, 89.0, 22.0, 93.0, 57.0, 11.0, 61.0, 13.0,
    61.0, 55.0, 63.0, 56.0, 17.0, 85.0, 21.0, 87.0, 33.0, 43.0, 37.0, 43.0, 6.0, 1.0, 7.0, 3.0,
    80.0, 87.0, 80.0, 87.0, 23.0, 50.0, 26.0, 52.0, 58.0, 89.0, 58.0, 89.0, 12.0, 30.0, 15.0, 34.0,
    32.0, 58.0, 36.0, 61.0, 41.0, 84.0, 44.0, 87.0, 44.0, 18.0, 44.0, 19.0, 13.0, 63.0, 15.0, 67.0,
    52.0, 70.0, 54.0, 74.0, 57.0, 59.0, 58.0, 59.0, 17.0, 90.0, 20.0, 92.0, 48.0, 53.0, 52.0, 56.0,
    92.0, 68.0, 92.0, 72.0, 26.0, 52.0, 30.0, 52.0, 56.0, 23.0, 57.0, 26.0, 88.0, 48.0, 88.0, 48.0,
    66.0, 13.0, 67.0, 15.0, 7.0, 82.0, 8.0, 86.0, 46.0, 68.0, 50.0, 68.0, 37.0, 33.0, 38.0, 36.0,
    6.0, 15.0, 8.0, 18.0, 85.0, 36.0, 89.0, 38.0, 82.0, 45.0, 84.0, 48.0, 12.0, 2.0, 16.0, 3.0,
    26.0, 15.0, 26.0, 16.0, 55.0, 23.0, 59.0, 26.0, 76.0, 37.0, 79.0, 39.0, 86.0, 74.0, 90.0, 77.0,
    16.0, 75.0, 18.0, 78.0, 44.0, 18.0, 45.0, 21.0, 52.0, 67.0, 54.0, 71.0, 59.0, 78.0, 62.0, 78.0,
    24.0, 5.0, 24.0, 8.0, 64.0, 80.0, 64.0, 83.0, 66.0, 55.0, 70.0, 55.0, 0.0, 17.0, 2.0, 19.0,
    15.0, 71.0, 18.0, 74.0, 87.0, 57.0, 87.0, 59.0, 6.0, 34.0, 7.0, 37.0, 34.0, 30.0, 37.0, 32.0,
    51.0, 19.0, 53.0, 19.0, 72.0, 51.0, 73.0, 55.0, 29.0, 45.0, 30.0, 45.0, 94.0, 94.0, 96.0, 95.0,
    7.0, 22.0, 11.0, 24.0, 86.0, 45.0, 87.0, 48.0, 33.0, 62.0, 34.0, 65.0, 18.0, 10.0, 21.0, 14.0,
    64.0, 66.0, 67.0, 67.0, 64.0, 25.0, 65.0, 28.0, 27.0, 4.0, 31.0, 6.0, 84.0, 4.0, 85.0, 5.0,
    48.0, 80.0, 50.0, 81.0, 1.0, 61.0, 3.0, 61.0, 71.0, 89.0, 74.0, 92.0, 40.0, 42.0, 43.0, 43.0,
    27.0, 64.0, 28.0, 66.0, 46.0, 26.0, 50.0, 26.0, 53.0, 83.0, 57.0, 87.0, 14.0, 75.0, 15.0, 79.0,
    31.0, 45.0, 34.0, 45.0, 89.0, 84.0, 92.0, 88.0, 84.0, 51.0, 85.0, 53.0, 67.0, 87.0, 67.0, 89.0,
    39.0, 26.0, 43.0, 27.0, 47.0, 61.0, 47.0, 63.0, 23.0, 49.0, 25.0, 53.0, 12.0, 3.0, 14.0, 5.0,
    16.0, 50.0, 19.0, 53.0, 63.0, 80.0, 64.0, 84.0, 22.0, 63.0, 22.0, 64.0, 26.0, 66.0, 29.0, 66.0,
    2.0, 15.0, 3.0, 15.0, 74.0, 77.0, 77.0, 79.0, 64.0, 11.0, 68.0, 11.0, 38.0, 4.0, 39.0, 8.0,
    83.0, 73.0, 87.0, 77.0, 85.0, 52.0, 89.0, 56.0, 74.0, 60.0, 76.0, 63.0, 62.0, 66.0, 65.0, 67.0,
];

fn boxes() -> Vec<Box2D> {
    DATA.chunks_exact(4)
        .map(|b| Box2D::new(b[0], b[1], b[2], b[3]))
        .collect()
}

fn build(node_size: usize) -> Index2D {
    let items = boxes();
    let mut builder = Index2DBuilder::new(items.len()).node_size(node_size);
    for b in &items {
        builder.add(*b);
    }
    builder.finish().unwrap()
}

fn sorted_search(index: &Index2D, query: Box2D) -> Vec<usize> {
    let mut hits = index.search(query);
    hits.sort_unstable();
    hits
}

#[test]
fn extent_matches_the_fixture() {
    assert_eq!(build(16).extent(), Some(Box2D::new(0.0, 1.0, 96.0, 95.0)));
}

#[test]
fn search_matches_the_fixture() {
    let index = build(16);
    assert_eq!(
        sorted_search(&index, Box2D::new(40.0, 40.0, 60.0, 60.0)),
        vec![6, 29, 31, 75]
    );
    assert_eq!(
        sorted_search(&index, Box2D::new(0.0, 0.0, 100.0, 100.0)).len(),
        100
    );
    assert!(sorted_search(&index, Box2D::new(200.0, 200.0, 300.0, 300.0)).is_empty());
}

#[test]
fn search_is_independent_of_node_size() {
    let expected = sorted_search(&build(16), Box2D::new(40.0, 40.0, 60.0, 60.0));
    // 2 is the smallest branching factor, 100 puts every item under the root: both must
    // return the same items as the default shape, since node size is a layout choice only.
    for node_size in [2usize, 4, 7, 16, 64, 100] {
        assert_eq!(
            sorted_search(&build(node_size), Box2D::new(40.0, 40.0, 60.0, 60.0)),
            expected,
            "node_size={node_size}"
        );
    }
}

#[test]
fn neighbors_match_the_fixture() {
    let index = build(16);
    let mut nearest = index.neighbors(Point2D::new(50.0, 50.0), 3);
    nearest.sort_unstable();
    assert_eq!(nearest, vec![6, 31, 75]);

    // Within 12 units of (50, 50), unlimited count.
    let mut within = index.neighbors_within(Point2D::new(50.0, 50.0), 100, 12.0);
    within.sort_unstable();
    assert_eq!(within, vec![6, 29, 31, 75, 85]);

    let mut all = index.neighbors(Point2D::new(50.0, 50.0), 100);
    all.sort_unstable();
    assert_eq!(all, (0..100).collect::<Vec<_>>());
}

#[test]
fn every_item_is_found_by_its_own_box() {
    let index = build(16);
    for (i, b) in boxes().iter().enumerate() {
        assert!(
            index.search(*b).contains(&i),
            "item {i} did not find itself"
        );
    }
}
