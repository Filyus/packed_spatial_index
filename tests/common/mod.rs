#![allow(dead_code)]

use packed_spatial_index::{Bounds2D, Index2D, Index2DBuilder, Point2D};
use rand::RngExt;
use rand::rngs::StdRng;

pub fn bounds(coords: [f64; 4]) -> Bounds2D {
    Bounds2D::new(coords[0], coords[1], coords[2], coords[3])
}

pub fn random_boxes(rng: &mut StdRng, n: usize) -> Vec<[f64; 4]> {
    let mut boxes = Vec::with_capacity(n);
    for _ in 0..n {
        let cx: f64 = rng.random_range(0.0..1000.0);
        let cy: f64 = rng.random_range(0.0..1000.0);
        let w: f64 = rng.random_range(0.1..10.0);
        let h: f64 = rng.random_range(0.1..10.0);
        boxes.push([cx, cy, cx + w, cy + h]);
    }
    boxes
}

pub fn build_index(boxes: &[[f64; 4]], node_size: usize) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for b in boxes {
        builder.add(Bounds2D::new(b[0], b[1], b[2], b[3]));
    }
    builder.finish().unwrap()
}

fn distance_squared(point: Point2D, bounds: [f64; 4]) -> f64 {
    fn axis(point: f64, min: f64, max: f64) -> f64 {
        if point < min {
            min - point
        } else if point > max {
            point - max
        } else {
            0.0
        }
    }

    let dx = axis(point.x, bounds[0], bounds[2]);
    let dy = axis(point.y, bounds[1], bounds[3]);
    dx * dx + dy * dy
}

pub fn brute_force_neighbors(
    boxes: &[[f64; 4]],
    point: Point2D,
    max_results: usize,
    max_distance: f64,
) -> Vec<usize> {
    if max_results == 0 || max_distance.is_nan() || max_distance.is_sign_negative() {
        return Vec::new();
    }
    let max_dist_sq = max_distance * max_distance;
    let mut pairs: Vec<(usize, f64)> = boxes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, b)| (index, distance_squared(point, b)))
        .filter(|&(_, dist)| dist <= max_dist_sq)
        .collect();
    pairs.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    pairs
        .into_iter()
        .take(max_results)
        .map(|(index, _)| index)
        .collect()
}
