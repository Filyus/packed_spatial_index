use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView, Point3D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn random_boxes(rng: &mut StdRng, count: usize, extent: f64, max_size: f64) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..extent);
            let y: f64 = rng.random_range(0.0..extent);
            let z: f64 = rng.random_range(0.0..extent);
            let w: f64 = rng.random_range(0.0..max_size);
            let h: f64 = rng.random_range(0.0..max_size);
            let d: f64 = rng.random_range(0.0..max_size);
            Box3D::new(x, y, z, x + w, y + h, z + d)
        })
        .collect()
}

fn build(boxes: &[Box3D]) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len());
    for &b in boxes {
        builder.add(b);
    }
    builder.finish().unwrap()
}

fn gap(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> f64 {
    if a_max < b_min {
        b_min - a_max
    } else if b_max < a_min {
        a_min - b_max
    } else {
        0.0
    }
}

fn box_distance_squared(a: Box3D, b: Box3D) -> f64 {
    let dx = gap(a.min_x, a.max_x, b.min_x, b.max_x);
    let dy = gap(a.min_y, a.max_y, b.min_y, b.max_y);
    let dz = gap(a.min_z, a.max_z, b.min_z, b.max_z);
    dx * dx + dy * dy + dz * dz
}

fn naive_distances(boxes: &[Box3D], query: Box3D, max_distance: f64) -> Vec<f64> {
    let mut distances: Vec<f64> = boxes
        .iter()
        .map(|&b| box_distance_squared(b, query))
        .filter(|&d| d <= max_distance * max_distance)
        .collect();
    distances.sort_by(|a, b| a.total_cmp(b));
    distances
}

fn result_distances(boxes: &[Box3D], query: Box3D, ids: &[usize]) -> Vec<f64> {
    ids.iter()
        .map(|&i| box_distance_squared(boxes[i], query))
        .collect()
}

#[test]
fn box_neighbors_match_naive_distances_3d() {
    let mut rng = StdRng::seed_from_u64(2020);
    let boxes = random_boxes(&mut rng, 500, 100.0, 8.0);
    let index = build(&boxes);
    let query = Box3D::new(40.0, 40.0, 40.0, 44.0, 42.0, 41.0);

    let expected = naive_distances(&boxes, query, f64::INFINITY);
    for k in [1usize, 5, 60, 500] {
        let ids = index.neighbors_of_box(query, k);
        assert_eq!(
            result_distances(&boxes, query, &ids),
            expected[..k.min(boxes.len())],
            "k={k}"
        );
    }
}

#[test]
fn box_neighbors_within_respects_max_distance_3d() {
    let mut rng = StdRng::seed_from_u64(2121);
    let boxes = random_boxes(&mut rng, 400, 100.0, 6.0);
    let index = build(&boxes);
    let query = Box3D::new(10.0, 80.0, 50.0, 12.0, 81.0, 52.0);

    for max_distance in [0.0, 4.0, 20.0] {
        let ids = index.neighbors_of_box_within(query, usize::MAX, max_distance);
        let expected = naive_distances(&boxes, query, max_distance);
        assert_eq!(
            result_distances(&boxes, query, &ids),
            expected,
            "max_distance={max_distance}"
        );
    }
}

#[test]
fn zero_size_query_box_matches_point_neighbors_3d() {
    let mut rng = StdRng::seed_from_u64(2222);
    let boxes = random_boxes(&mut rng, 400, 100.0, 10.0);
    let index = build(&boxes);

    let point = Point3D::new(33.3, 66.6, 50.0);
    let query = Box3D::from_point(point);
    let by_point = result_distances(&boxes, query, &index.neighbors(point, 20));
    let by_box = result_distances(&boxes, query, &index.neighbors_of_box(query, 20));
    assert_eq!(by_point, by_box);
}

#[test]
fn view_box_neighbors_match_owned_3d() {
    let mut rng = StdRng::seed_from_u64(2323);
    let boxes = random_boxes(&mut rng, 300, 100.0, 12.0);
    let index = build(&boxes);
    let bytes = index.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();

    let query = Box3D::new(20.0, 30.0, 60.0, 24.0, 31.0, 62.0);
    let owned = result_distances(&boxes, query, &index.neighbors_of_box(query, 30));
    let viewed = result_distances(&boxes, query, &view.neighbors_of_box(query, 30));
    assert_eq!(owned, viewed);
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::SimdIndex3DView;

    #[test]
    fn simd_box_neighbors_match_naive_3d() {
        let mut rng = StdRng::seed_from_u64(2424);
        let boxes = random_boxes(&mut rng, 400, 100.0, 8.0);
        let mut builder = Index3DBuilder::new(boxes.len());
        boxes.iter().for_each(|&b| builder.add(b));
        let index = builder.finish_simd().unwrap();
        let bytes = index.to_bytes();
        let view = SimdIndex3DView::from_bytes(&bytes).unwrap();

        let query = Box3D::new(70.0, 10.0, 30.0, 72.0, 14.0, 33.0);
        let expected = naive_distances(&boxes, query, f64::INFINITY);
        for k in [1usize, 10, 150] {
            let owned = result_distances(&boxes, query, &index.neighbors_of_box(query, k));
            let viewed = result_distances(&boxes, query, &view.neighbors_of_box(query, k));
            assert_eq!(owned, expected[..k.min(boxes.len())], "k={k}");
            assert_eq!(viewed, expected[..k.min(boxes.len())], "k={k}");
        }
    }
}
