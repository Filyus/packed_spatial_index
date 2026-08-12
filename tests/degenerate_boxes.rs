//! Boxes with `min > max`, which the unchecked `Box2D::new` / `Box3D::new` allow.
//!
//! Such a box covers no region, yet it is *contained* by queries it does not
//! overlap — so a tree holding one answered the same query differently depending on
//! whether the search descended to it or took the whole-subtree shortcut. Found by
//! `fuzz/fuzz_targets/load.rs`, comparing an owned index against its own view.
//!
//! Two things close it, and both are tested here. The builder now refuses these
//! boxes, so no index built through the API can contain one. A file can still
//! carry one, and for that case every search path must at least agree with every
//! other — which is what the shortcut's added `overlaps` test buys.

use std::fs;
use std::path::Path;

use packed_spatial_index::{
    Box2D, Box3D, BuildError, Index2D, Index2DBuilder, Index2DView, Index3DBuilder,
};

const INVERTED_2D: Box2D = Box2D {
    min_x: 4.7e226,
    min_y: 1.0e-321,
    max_x: 2.0,
    max_y: 2.0,
};
const WIDE_2D: Box2D = Box2D {
    min_x: -1e30,
    min_y: -1e30,
    max_x: 1e30,
    max_y: 1e30,
};

fn builder_2d(bad: Box2D, n: usize, at: usize) -> Index2DBuilder {
    let mut builder = Index2DBuilder::new(n);
    for i in 0..n {
        if i == at {
            builder.add(bad);
        } else {
            let x = i as f64;
            builder.add(Box2D::new(x, x, x + 1.0, x + 1.0));
        }
    }
    builder
}

#[test]
fn the_2d_builder_rejects_crossed_and_nan_bounds() {
    let cases = [
        ("crossed x", INVERTED_2D),
        ("crossed y", Box2D::new(0.0, 9.0, 1.0, 1.0)),
        ("nan min", Box2D::new(f64::NAN, 0.0, 1.0, 1.0)),
        ("nan max", Box2D::new(0.0, 0.0, 1.0, f64::NAN)),
    ];
    for (what, bad) in cases {
        // Position varies so the reported index is the added position, not a
        // sorted one — after sorting, "item 3" would mean nothing to the caller.
        for (n, at) in [(1usize, 0usize), (5, 3), (200, 137)] {
            let err = builder_2d(bad, n, at)
                .finish()
                .err()
                .expect("crossed bounds must be rejected");
            assert!(
                matches!(err, BuildError::InvalidItemBounds { at: reported } if reported == at),
                "{what}, n={n}: expected InvalidItemBounds at {at}, got {err:?}"
            );
        }
    }
}

#[cfg(feature = "simd")]
#[test]
fn every_2d_finish_variant_rejects_them() {
    assert!(matches!(
        builder_2d(INVERTED_2D, 5, 2)
            .finish_simd()
            .err()
            .expect("crossed bounds must be rejected"),
        BuildError::InvalidItemBounds { at: 2 }
    ));
    #[cfg(feature = "f32-storage")]
    {
        assert!(matches!(
            builder_2d(INVERTED_2D, 5, 2)
                .finish_f32()
                .err()
                .expect("crossed bounds must be rejected"),
            BuildError::InvalidItemBounds { at: 2 }
        ));
        assert!(matches!(
            builder_2d(INVERTED_2D, 5, 2)
                .finish_simd_f32()
                .err()
                .expect("crossed bounds must be rejected"),
            BuildError::InvalidItemBounds { at: 2 }
        ));
    }
}

#[test]
fn the_3d_builder_rejects_them_including_the_z_axis() {
    let mut builder = Index3DBuilder::new(2);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    builder.add(Box3D::new(0.0, 0.0, 5.0, 1.0, 1.0, 1.0));
    assert!(matches!(
        builder
            .finish()
            .err()
            .expect("crossed bounds must be rejected"),
        BuildError::InvalidItemBounds { at: 1 }
    ));
}

#[test]
fn well_formed_boxes_still_build() {
    // The guard must not reject the degenerate-but-valid shapes real data carries:
    // zero-area boxes, points, and infinities.
    let mut builder = Index2DBuilder::new(3);
    builder.add(Box2D::new(1.0, 1.0, 1.0, 1.0));
    builder.add(Box2D::new(-0.0, 0.0, 0.0, -0.0));
    builder.add(Box2D::new(f64::NEG_INFINITY, 0.0, f64::INFINITY, 0.0));
    let index = builder.finish().expect("valid bounds");
    assert_eq!(index.num_items(), 3);
}

/// A 160-byte index carrying one crossed box, as `fuzz/fuzz_targets/load.rs` found
/// it. The builder can no longer produce this, so the bytes are carried instead:
/// a loaded file is outside the builder's guarantee, and the paths must still agree.
#[test]
fn a_loaded_crossed_box_reads_the_same_through_every_path() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inverted_box_2d.psindex");
    let bytes = fs::read(&path).expect("fixture is carried by the repository");

    let index = Index2D::from_bytes(&bytes).expect("the fixture is a valid container");
    let view = Index2DView::from_bytes(&bytes).expect("the fixture is a valid container");
    let expected = index.search(WIDE_2D);

    assert_eq!(view.search(WIDE_2D), expected, "zero-copy view");
    assert_eq!(
        index.search_iter(WIDE_2D).collect::<Vec<_>>(),
        expected,
        "search_iter"
    );
    assert_eq!(index.any(WIDE_2D), !expected.is_empty(), "any");

    let (mut results, mut stack) = (Vec::new(), Vec::new());
    index.search_into_stack(WIDE_2D, &mut results, &mut stack);
    assert_eq!(results, expected, "search_into_stack");
    index.search_into_stack_prefetch(WIDE_2D, &mut results, &mut stack);
    assert_eq!(results, expected, "search_into_stack_prefetch");

    #[cfg(feature = "simd")]
    {
        use packed_spatial_index::{SimdIndex2D, SimdIndex2DView};
        if let Ok(simd) = SimdIndex2D::from_bytes(&bytes) {
            assert_eq!(simd.search(WIDE_2D), expected, "SimdIndex2D");
        }
        if let Ok(simd) = SimdIndex2DView::from_bytes(&bytes) {
            assert_eq!(simd.search(WIDE_2D), expected, "SimdIndex2DView");
        }
    }
}
