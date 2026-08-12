//! An exploring fuzzer for the loaders, to sit alongside the deterministic tests
//! rather than replace it.
//!
//! `tests/proptest_2d.rs` already throws arbitrary and mutated-valid bytes at
//! `from_bytes` on every `cargo test`, and that is the part CI can hold. What it
//! cannot do is find the shape nobody thought of, which is what coverage-guided
//! fuzzing is for — and it matters here more than the panic-freedom alone
//! suggests. SAFETY.md's argument is that validation runs first and traversal
//! then *trusts* the structure, so a hole in validation is not a wrong answer,
//! it is an out-of-bounds read in a path with `unsafe` in it.
//!
//! Properties, checked on every case the fuzzer invents:
//!
//! 1. **No panic, on any input** — libFuzzer checks it by running at all, since
//!    a panic aborts.
//! 2. **A loaded index only ever hands back item indices below `num_items`.**
//!    Callers use those to index their own parallel arrays, so a validated index
//!    that returns 4 000 000 for a 3-item tree is a caller-side out-of-bounds
//!    that no amount of caller care would prevent.
//! 3. **The owned index and the zero-copy view agree about accepting the buffer.**
//!    They validate it through different code, so a disagreement means one of them
//!    is reading a field the other is not.
//!
//!    Their *answers* are deliberately not compared. A search may take the
//!    whole-subtree shortcut, which relies on the tree invariant that a parent's
//!    box covers its children — true of anything this crate builds, and breakable
//!    by a corrupt file. Checking it at load costs a pass over the whole tree,
//!    which is what the zero-copy view exists to avoid, so on a corrupt buffer the
//!    paths may legitimately return different sets. SAFETY.md states this, and
//!    `tests/degenerate_boxes.rs` pins the part that is guaranteed. An earlier
//!    version of this target did compare the answers, and that is how the
//!    inconsistency it *was* right about — the shortcut testing containment without
//!    overlap — was found. That comparison now lives in `build_query.rs`, over trees
//!    this crate built, where the invariant holds and a difference is always a bug.
//! 4. **Accepting a buffer means the query paths survive it**, including the SIMD
//!    ones, which write left-packed hits into reserved slack through `unsafe`.
//!
//! Not wired into CI: it needs nightly, and a fuzz run has no natural end.
//!
//! ```text
//! cargo install cargo-fuzz
//! cargo +nightly fuzz run load
//! ```
//!
//! On Windows that fails twice before it works, and both failures look like
//! something else:
//!
//! * `STATUS_DLL_NOT_FOUND` (`0xc0000135`) means the AddressSanitizer runtime is
//!   not on `PATH`. It ships with MSVC, at
//!   `…/VC/Tools/MSVC/<version>/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll`;
//!   put that directory on `PATH` for the run.
//! * `-s none` looks like the way around that and is not: without the sanitizer
//!   the coverage instrumentation loses its symbols and the link fails on
//!   `__start___sancov_cntrs`.
//!
//! Seed the corpus before a long run, or the fuzzer spends its first minutes
//! rediscovering the magic and the chunk table. `examples/` writes real
//! artifacts; any `.psindex` file dropped in `fuzz/corpus/load` works.
//!
//! Last run: 400 000 executions from a seeded corpus, no crashes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DView, Index3D, Index3DView, SimdIndex2D, SimdIndex2DView,
};

/// A query that reaches every corner of a tree built anywhere in `f64` space,
/// so acceptance is tested by walking the whole structure, not one branch.
const WIDE_2D: Box2D = Box2D {
    min_x: -1e30,
    min_y: -1e30,
    max_x: 1e30,
    max_y: 1e30,
};
const WIDE_3D: Box3D = Box3D {
    min_x: -1e30,
    min_y: -1e30,
    min_z: -1e30,
    max_x: 1e30,
    max_y: 1e30,
    max_z: 1e30,
};

/// Write the offending case out ourselves before failing.
///
/// On Windows a Rust panic under the sanitizer leaves through `abort`, which
/// libFuzzer's crash handler does not see: the run stops with
/// `STATUS_STACK_BUFFER_OVERRUN` and `artifacts/` stays empty, so the input that
/// found the bug is lost. Writing it first costs nothing on a path that is about
/// to end the process anyway.
fn keep(data: &[u8]) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in data {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/load");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(format!("case-{hash:016x}")), data);
}

fn check_hits(hits: &[usize], num_items: usize, what: &str) {
    for &hit in hits {
        assert!(
            hit < num_items,
            "{what} returned item index {hit} for a tree of {num_items} items"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    let owned = Index2D::from_bytes(data);
    let view = Index2DView::from_bytes(data);

    // Two validators over one buffer: if they disagree about acceptance, one of
    // them is reading a field the other is not.
    if owned.is_ok() != view.is_ok() {
        keep(data);
        panic!("Index2D and Index2DView disagreed about accepting the same bytes");
    }

    if let Ok(index) = &owned {
        let hits = index.search(WIDE_2D);
        check_hits(&hits, index.num_items(), "Index2D::search");
        check_hits(
            &index.neighbors(packed_spatial_index::Point2D::new(0.0, 0.0), 8),
            index.num_items(),
            "Index2D::neighbors",
        );

    }

    // The SIMD paths write left-packed hits into reserved slack through `unsafe`,
    // so they are the ones a validation hole would turn into a memory error.
    if let Ok(simd) = SimdIndex2D::from_bytes(data) {
        let mut hits = Vec::new();
        let mut stack = Vec::new();
        simd.search_simd(WIDE_2D, &mut hits, &mut stack);
        check_hits(&hits, simd.num_items(), "SimdIndex2D::search_simd");
        assert!(simd.any(WIDE_2D) == !hits.is_empty());
    }
    if let Ok(view) = SimdIndex2DView::from_bytes(data) {
        check_hits(&view.search(WIDE_2D), view.num_items(), "SimdIndex2DView");
    }

    // The same buffer is a candidate 3D index; the layouts share a container, so
    // a 2D tree read as 3D is exactly the confusion validation must reject.
    if let Ok(index) = Index3D::from_bytes(data) {
        check_hits(&index.search(WIDE_3D), index.num_items(), "Index3D::search");
    }
    if let Ok(view) = Index3DView::from_bytes(data) {
        check_hits(&view.search(WIDE_3D), view.num_items(), "Index3DView");
    }
});
