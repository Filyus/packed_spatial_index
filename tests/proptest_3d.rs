//! Property-based correctness and robustness tests for the 3D index.
//!
//! Mirrors `proptest_2d.rs`: `search` (scalar, view, SIMD) must agree with a
//! brute-force scan, and `from_bytes` must never panic on arbitrary or mutated
//! byte buffers despite using `*_unchecked` accessors after header validation.

use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView};
#[cfg(feature = "simd")]
use packed_spatial_index::SimdIndex3D;
use proptest::prelude::*;

/// Boxes on a small integer grid so edges collide on exact boundaries.
fn boxes_strategy() -> impl Strategy<Value = Vec<[f64; 6]>> {
    let single = (0i64..12, 0i64..12, 0i64..12, 0i64..5, 0i64..5, 0i64..5).prop_map(
        |(x, y, z, w, h, d)| {
            [
                x as f64,
                y as f64,
                z as f64,
                (x + w) as f64,
                (y + h) as f64,
                (z + d) as f64,
            ]
        },
    );
    prop::collection::vec(single, 0..64)
}

fn query_strategy() -> impl Strategy<Value = [f64; 6]> {
    (0i64..12, 0i64..12, 0i64..12, 0i64..7, 0i64..7, 0i64..7).prop_map(|(x, y, z, w, h, d)| {
        [
            x as f64,
            y as f64,
            z as f64,
            (x + w) as f64,
            (y + h) as f64,
            (z + d) as f64,
        ]
    })
}

fn box3d(b: &[f64; 6]) -> Box3D {
    Box3D::new(b[0], b[1], b[2], b[3], b[4], b[5])
}

fn build(boxes: &[[f64; 6]]) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len());
    for b in boxes {
        builder.add(box3d(b));
    }
    builder.finish().unwrap()
}

fn brute_force(boxes: &[[f64; 6]], query: Box3D) -> Vec<usize> {
    boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| query.overlaps(box3d(b)))
        .map(|(i, _)| i)
        .collect()
}

proptest! {
    #[test]
    fn search_matches_brute_force(boxes in boxes_strategy(), q in query_strategy()) {
        let query = box3d(&q);

        let mut expected = brute_force(&boxes, query);
        expected.sort_unstable();

        let index = build(&boxes);
        let mut scalar = index.search(query);
        scalar.sort_unstable();
        prop_assert_eq!(&scalar, &expected);

        let bytes = index.to_bytes();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        let mut borrowed = view.search(query);
        borrowed.sort_unstable();
        prop_assert_eq!(&borrowed, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index3DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(box3d(b));
            }
            let simd = builder.finish_simd().unwrap();
            let mut simd_hits = simd.search(query);
            simd_hits.sort_unstable();
            prop_assert_eq!(&simd_hits, &expected);
        }
    }

    #[test]
    fn from_bytes_never_panics_on_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = Index3D::from_bytes(&bytes);
        let _ = Index3DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex3D::from_bytes(&bytes);
        }
    }

    #[test]
    fn from_bytes_tolerates_mutated_valid_bytes(
        boxes in boxes_strategy(),
        pos in any::<prop::sample::Index>(),
        xor in 1u8..=255,
        truncate in any::<bool>(),
    ) {
        let mut bytes = build(&boxes).to_bytes();
        if !bytes.is_empty() {
            let idx = pos.index(bytes.len());
            bytes[idx] ^= xor;
        }
        if truncate && !bytes.is_empty() {
            bytes.truncate(bytes.len() / 2);
        }

        let _ = Index3D::from_bytes(&bytes);
        let _ = Index3DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex3D::from_bytes(&bytes);
        }
    }
}
