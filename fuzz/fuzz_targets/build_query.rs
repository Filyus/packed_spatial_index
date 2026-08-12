//! An exploring fuzzer for the build and query paths, judged by brute force.
//!
//! `load.rs` asks whether a hostile buffer can get past validation. This one is
//! not about hostility — the input is the caller's own boxes — it is about the
//! three things a built index *promises*, each checkable on every case:
//!
//! 1. **A search returns exactly the overlapping items**, judged by a linear scan
//!    over the same boxes. Never by another index of ours: two paths that share a
//!    misreading of `overlaps` agree with each other perfectly. This is where a
//!    tree-shape bug shows up as a wrong answer rather than as a crash.
//! 2. **The SIMD index answers the same as the scalar one.** They differ in
//!    layout, in kernel and in whether the leaf scan writes through `unsafe`, so
//!    the fuzzer is choosing lane counts and node fills that the fixed tests
//!    cannot enumerate — the tail of a node, an 8-wide block that is 3 long.
//! 3. **Bytes round-trip.** `to_bytes` then `from_bytes` must answer identically;
//!    a field that serializes but does not validate would show here first.
//!
//! `cargo-fuzz` builds with debug assertions on, so every internal
//! `debug_assert!` in the builder is live in each of these runs too.
//!
//! Not wired into CI: it needs nightly, and a fuzz run has no natural end.
//! See `load.rs` in this directory for the two Windows failures it takes to get
//! a run started.
//!
//! ```text
//! cargo +nightly fuzz run build_query
//! ```
//!
//! Last run: 400 000 executions, no crashes.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use packed_spatial_index::{Box2D, Index2D, Index2DBuilder, Index2DView};

#[derive(Arbitrary, Debug)]
struct Case {
    /// `f32` bit patterns rather than `f64`: four bytes per coordinate lets the
    /// fuzzer reach the interesting values — subnormals, huge exponents, signed
    /// zero — while spending its budget on tree shape rather than on mantissas.
    boxes: Vec<[f32; 4]>,
    query: [f32; 4],
    node_size: u16,
}

/// Non-finite coordinates are excluded so that the brute-force oracle stays a
/// judge: with a NaN in play, "overlaps" is a question about comparison rules,
/// not about the tree, and `tests/api_2d.rs` pins those rules directly.
fn to_box(raw: [f32; 4]) -> Option<Box2D> {
    if raw.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let (min_x, max_x) = (f64::from(raw[0]), f64::from(raw[2]));
    let (min_y, max_y) = (f64::from(raw[1]), f64::from(raw[3]));
    Some(Box2D::new(
        min_x.min(max_x),
        min_y.min(max_y),
        min_x.max(max_x),
        min_y.max(max_y),
    ))
}

fn brute_force(boxes: &[Box2D], query: Box2D) -> Vec<usize> {
    boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| b.overlaps(query))
        .map(|(i, _)| i)
        .collect()
}

fuzz_target!(|case: Case| {
    let boxes: Vec<Box2D> = case.boxes.iter().copied().filter_map(to_box).collect();
    let Some(query) = to_box(case.query) else {
        return;
    };
    // 4096 keeps a case inside a few milliseconds; the shapes worth finding are
    // node fills and level boundaries, and those appear at every size.
    if boxes.len() > 4096 {
        return;
    }

    let node_size = usize::from(case.node_size.max(2));
    let mut builder = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for b in &boxes {
        builder.add(*b);
    }
    let Ok(index) = builder.finish() else {
        return;
    };

    let expected = brute_force(&boxes, query);
    let mut actual = index.search(query);
    actual.sort_unstable();
    assert_eq!(actual, expected, "search disagreed with a linear scan");

    let mut simd_builder = Index2DBuilder::new(boxes.len()).node_size(node_size);
    for b in &boxes {
        simd_builder.add(*b);
    }
    if let Ok(simd) = simd_builder.finish_simd() {
        let mut from_simd = simd.search(query);
        from_simd.sort_unstable();
        assert_eq!(from_simd, expected, "the SIMD index disagreed with a linear scan");
        assert_eq!(
            simd.any(query),
            !expected.is_empty(),
            "`any` disagreed with whether a linear scan found anything"
        );
    }

    let bytes = index.to_bytes();
    let reloaded = Index2D::from_bytes(&bytes).expect("bytes this crate just wrote must load");
    let mut after = reloaded.search(query);
    after.sort_unstable();
    assert_eq!(after, expected, "the index answered differently after a round-trip");

    // Every entry point over a tree this crate built. They differ in whether they
    // take the whole-subtree shortcut, prefetch, or scan lazily, and the shortcut
    // is the one that has been wrong before, so a disagreement here is always a bug
    // rather than a corrupt file.
    let mut from_view = Index2DView::from_bytes(&bytes)
        .expect("bytes this crate just wrote must load as a view")
        .search(query);
    from_view.sort_unstable();
    assert_eq!(from_view, expected, "the zero-copy view answered differently");

    let mut from_iter = index.search_iter(query).collect::<Vec<_>>();
    from_iter.sort_unstable();
    assert_eq!(from_iter, expected, "search_iter answered differently");

    assert_eq!(
        index.any(query),
        !expected.is_empty(),
        "`any` disagreed with `search`"
    );

    let (mut results, mut stack) = (Vec::new(), Vec::new());
    index.search_into_stack_prefetch(query, &mut results, &mut stack);
    results.sort_unstable();
    assert_eq!(results, expected, "the prefetching traversal answered differently");
});
