//! `estimate_count` in 3D: the bracket holds against `count` on owned, view
//! and SIMD indexes.

use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn random_boxes(rng: &mut StdRng, count: usize, extent: f64, max_size: f64) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let x = rng.random_range(0.0..extent);
            let y = rng.random_range(0.0..extent);
            let z = rng.random_range(0.0..extent);
            let w = rng.random_range(0.0..max_size);
            let h = rng.random_range(0.0..max_size);
            let d = rng.random_range(0.0..max_size);
            Box3D::new(x, y, z, x + w, y + h, z + d)
        })
        .collect()
}

fn build(boxes: &[Box3D], node_size: usize) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for b in boxes {
        builder.add(*b);
    }
    builder.finish().unwrap()
}

fn windows() -> Vec<Box3D> {
    vec![
        Box3D::new(10.0, 10.0, 10.0, 30.0, 30.0, 30.0),
        Box3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0),
        Box3D::new(-10.0, -10.0, -10.0, 200.0, 200.0, 200.0),
        Box3D::new(48.0, 48.0, 48.0, 52.0, 52.0, 52.0),
        Box3D::new(300.0, 300.0, 300.0, 310.0, 310.0, 310.0),
        Box3D::new(25.0, 0.0, 0.0, 26.0, 100.0, 100.0),
    ]
}

#[test]
fn bracket_holds_and_level_zero_is_exact() {
    let mut rng = StdRng::seed_from_u64(3201);
    let boxes = random_boxes(&mut rng, 3000, 100.0, 3.0);
    for node_size in [2, 8, 16] {
        let index = build(&boxes, node_size);
        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        for window in windows() {
            let exact = index.count(window);
            let mut previous_spread = usize::MAX;
            for stop_level in (0..6).rev() {
                let est = index.estimate_count(window, stop_level);
                assert!(
                    est.lower <= exact && exact <= est.upper,
                    "n{node_size} {window:?} level {stop_level}: {exact} not in [{}, {}]",
                    est.lower,
                    est.upper
                );
                assert!(est.lower as f64 <= est.estimate && est.estimate <= est.upper as f64);
                assert!(est.spread() <= previous_spread);
                previous_spread = est.spread();
                assert_eq!(view.estimate_count(window, stop_level), est);
            }
            let exact_est = index.estimate_count(window, 0);
            assert_eq!((exact_est.lower, exact_est.upper), (exact, exact));
        }
    }
}

#[test]
fn flat_items_still_bracket() {
    // Every item is flat in z: the z axis contributes 1 to every fraction.
    let boxes: Vec<Box3D> = (0..128)
        .map(|i| {
            let v = (i % 16) as f64;
            let w = (i / 16) as f64;
            Box3D::new(v, w, 5.0, v + 0.5, w + 0.5, 5.0)
        })
        .collect();
    let index = build(&boxes, 8);
    for window in [
        Box3D::new(2.0, 2.0, 4.0, 6.0, 6.0, 6.0),
        Box3D::new(2.0, 2.0, 6.0, 6.0, 6.0, 7.0),
        Box3D::new(0.0, 0.0, 5.0, 16.0, 8.0, 5.0),
    ] {
        let exact = index.count(window);
        for stop_level in 0..4 {
            let est = index.estimate_count(window, stop_level);
            assert!(
                est.lower <= exact && exact <= est.upper,
                "{window:?} {stop_level}"
            );
        }
    }
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::SimdIndex3DView;

    #[test]
    fn simd_index_and_view_agree_with_owned() {
        let mut rng = StdRng::seed_from_u64(3202);
        let boxes = random_boxes(&mut rng, 1500, 100.0, 3.0);
        let index = build(&boxes, 16);
        let mut builder = Index3DBuilder::new(boxes.len()).node_size(16);
        for b in &boxes {
            builder.add(*b);
        }
        let simd = builder.finish_simd().unwrap();
        let bytes = simd.to_bytes();
        let view = SimdIndex3DView::from_bytes(&bytes).unwrap();
        for window in windows() {
            for stop_level in 0..4 {
                let expected = index.estimate_count(window, stop_level);
                assert_eq!(simd.estimate_count(window, stop_level), expected);
                assert_eq!(view.estimate_count(window, stop_level), expected);
            }
        }
    }
}
