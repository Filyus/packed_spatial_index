//! Property-based correctness and robustness tests for the 2D index.
//!
//! These complement the hand-written cases in `api_2d.rs` and `persistence_2d.rs`
//! by sweeping a wide space of inputs:
//!   * `search` (scalar, view, and SIMD) agrees with a brute-force scan;
//!   * `from_bytes` never panics on arbitrary or mutated byte buffers, even
//!     though it relies on `*_unchecked` accessors after header validation.

#[cfg(feature = "simd")]
use packed_spatial_index::SimdIndex2D;
use packed_spatial_index::{Box2D, Index2D, Index2DBuilder, Index2DView};
use proptest::prelude::*;

/// Generate boxes on a small integer grid so edges collide often. Inclusive-edge
/// overlap bugs hide on exact boundaries, which random floats almost never hit.
fn boxes_strategy() -> impl Strategy<Value = Vec<[f64; 4]>> {
    let single = (0i64..16, 0i64..16, 0i64..6, 0i64..6)
        .prop_map(|(x, y, w, h)| [x as f64, y as f64, (x + w) as f64, (y + h) as f64]);
    prop::collection::vec(single, 0..64)
}

fn query_strategy() -> impl Strategy<Value = [f64; 4]> {
    (0i64..16, 0i64..16, 0i64..8, 0i64..8)
        .prop_map(|(x, y, w, h)| [x as f64, y as f64, (x + w) as f64, (y + h) as f64])
}

fn build(boxes: &[[f64; 4]]) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len());
    for b in boxes {
        builder.add(Box2D::new(b[0], b[1], b[2], b[3]));
    }
    builder.finish().unwrap()
}

fn brute_force(boxes: &[[f64; 4]], query: Box2D) -> Vec<usize> {
    boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| query.overlaps(Box2D::new(b[0], b[1], b[2], b[3])))
        .map(|(i, _)| i)
        .collect()
}

proptest! {
    /// Scalar, view, and SIMD searches must all return exactly the brute-force set.
    #[test]
    fn search_matches_brute_force(boxes in boxes_strategy(), q in query_strategy()) {
        let query = Box2D::new(q[0], q[1], q[2], q[3]);

        let mut expected = brute_force(&boxes, query);
        expected.sort_unstable();

        let index = build(&boxes);
        let mut scalar = index.search(query);
        scalar.sort_unstable();
        prop_assert_eq!(&scalar, &expected);

        let bytes = index.to_bytes();
        let view = Index2DView::from_bytes(&bytes).unwrap();
        let mut borrowed = view.search(query);
        borrowed.sort_unstable();
        prop_assert_eq!(&borrowed, &expected);

        #[cfg(feature = "simd")]
        {
            let mut builder = Index2DBuilder::new(boxes.len());
            for b in &boxes {
                builder.add(Box2D::new(b[0], b[1], b[2], b[3]));
            }
            let simd = builder.finish_simd().unwrap();
            let mut simd_hits = simd.search(query);
            simd_hits.sort_unstable();
            prop_assert_eq!(&simd_hits, &expected);
        }
    }

    /// Arbitrary bytes must yield `Err`, never a panic or out-of-bounds read.
    #[test]
    fn from_bytes_never_panics_on_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = Index2D::from_bytes(&bytes);
        let _ = Index2DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex2D::from_bytes(&bytes);
        }
    }

    /// Mutating a valid buffer must still be handled gracefully (Ok or Err, no panic).
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

        let _ = Index2D::from_bytes(&bytes);
        let _ = Index2DView::from_bytes(&bytes);
        #[cfg(feature = "simd")]
        {
            let _ = SimdIndex2D::from_bytes(&bytes);
        }
    }
}
