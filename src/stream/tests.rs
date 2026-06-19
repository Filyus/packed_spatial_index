use super::*;
use crate::LoadError;
use crate::{Box2D, Index2DBuilder};
use std::cell::RefCell;
use std::io;

/// Build a deterministic index of `n` unit boxes on a diagonal.
fn build_bytes(n: usize, node_size: usize) -> Vec<u8> {
    let mut builder = Index2DBuilder::new(n).node_size(node_size);
    for i in 0..n {
        let v = i as f64;
        builder.add(Box2D::new(v, v, v + 0.5, v + 0.5));
    }
    builder.finish().unwrap().to_bytes()
}

/// A `RangeReader` that counts reads and bytes, to prove `open` is bounded.
struct CountingReader<R> {
    inner: R,
    reads: RefCell<usize>,
    bytes: RefCell<u64>,
}

impl<R: RangeReader> CountingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            reads: RefCell::new(0),
            bytes: RefCell::new(0),
        }
    }
}

impl<R: RangeReader> RangeReader for CountingReader<R> {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        *self.reads.borrow_mut() += 1;
        *self.bytes.borrow_mut() += buf.len() as u64;
        self.inner.read_exact_at(offset, buf)
    }

    fn len(&self) -> Option<u64> {
        self.inner.len()
    }
}

fn open_slice(bytes: Vec<u8>) -> StreamIndex2D<SliceReader<Vec<u8>>> {
    StreamIndex2D::open(SliceReader::new(bytes)).expect("open should succeed")
}

#[test]
fn metadata_matches_owned_across_sizes() {
    for &n in &[0usize, 1, 16, 17, 1000] {
        let mut builder = Index2DBuilder::new(n).node_size(16);
        for i in 0..n {
            let v = i as f64;
            builder.add(Box2D::new(v, v, v + 0.5, v + 0.5));
        }
        let owned = builder.finish().unwrap();
        let bytes = owned.to_bytes();

        let stream = open_slice(bytes);
        assert_eq!(stream.num_items(), owned.num_items(), "n={n}");
        assert_eq!(stream.node_size(), owned.node_size(), "n={n}");
        assert_eq!(stream.is_empty(), n == 0, "n={n}");
        assert_eq!(stream.extent(), owned.extent(), "n={n}");
    }
}

#[test]
fn from_directory_reuses_directory_without_io() {
    // Large enough that the leaf level exceeds the directory budget, so a
    // query genuinely streams leaves (and the reattach saving is visible).
    let bytes = build_bytes(50_000, 16);
    let q = Box2D::new(100.0, 100.0, 140.0, 140.0);

    // Open once over a counting reader: this pays the directory reads.
    let idx = StreamIndex2D::open(CountingReader::new(SliceReader::new(bytes.clone()))).unwrap();
    let expected = idx.search(q).unwrap();
    let (dir, reader) = idx.into_directory();
    let open_plus_query_reads = *reader.reads.borrow();
    assert!(open_plus_query_reads > 0);

    // Reattach a FRESH reader from the cached directory: zero I/O until a query.
    let idx2 =
        StreamIndex2D::from_directory(&dir, CountingReader::new(SliceReader::new(bytes.clone())))
            .unwrap();
    assert_eq!(
        *idx2.core.reader.reads.borrow(),
        0,
        "from_directory must not read"
    );

    // The query returns the identical result and reads only its own descent,
    // never re-reading the directory.
    let got = idx2.search(q).unwrap();
    assert_eq!(got, expected);
    let query_only_reads = *idx2.core.reader.reads.borrow();
    assert!(query_only_reads > 0);
    assert!(
        query_only_reads < open_plus_query_reads,
        "reattached query ({query_only_reads}) should read less than open+query ({open_plus_query_reads})"
    );
    assert_eq!(dir.num_items(), 50_000);
}

#[test]
fn larger_directory_budget_cuts_per_query_reads() {
    let bytes = build_bytes(50_000, 16);
    let q = Box2D::new(100.0, 100.0, 140.0, 140.0);

    // Per-query reads (excluding the one-time open) and the hit set, for a
    // given directory budget.
    let query_cost = |budget: Option<u64>| -> (usize, Vec<usize>) {
        let limits = StreamLimits {
            directory_budget_bytes: budget,
            ..Default::default()
        };
        let idx = StreamIndex2D::open_with_limits(
            CountingReader::new(SliceReader::new(bytes.clone())),
            limits,
        )
        .unwrap();
        let after_open = *idx.core.reader.reads.borrow();
        let hits = idx.search(q).unwrap();
        let after_query = *idx.core.reader.reads.borrow();
        (after_query - after_open, hits)
    };

    let (default_reads, default_hits) = query_cost(None);
    // A budget large enough to cache the whole tree: the query reads nothing.
    let (big_reads, big_hits) = query_cost(Some(64 * 1024 * 1024));

    assert_eq!(default_hits, big_hits, "budget must not change results");
    assert!(
        default_reads > 0,
        "default budget should still stream leaves"
    );
    assert!(
        big_reads < default_reads,
        "bigger directory ({big_reads}) should read less per query than default ({default_reads})"
    );
}

/// Boxes scattered across the extent (multiplicative hash), so a window
/// query's hits land on non-contiguous leaf runs — what makes a wider
/// coalesce gap merge reads. The diagonal `build_bytes` keeps hits
/// contiguous, which already coalesce into one run.
fn build_scattered_bytes(n: usize) -> Vec<u8> {
    let mut b = Index2DBuilder::new(n).node_size(16);
    for i in 0..n {
        let x = (i.wrapping_mul(2_654_435_761) % 1000) as f64;
        let y = (i.wrapping_mul(40_503) % 1000) as f64;
        b.add(Box2D::new(x, y, x + 2.0, y + 2.0));
    }
    b.finish().unwrap().to_bytes()
}

#[test]
fn larger_coalesce_gap_cuts_per_query_reads() {
    let bytes = build_scattered_bytes(50_000);
    let q = Box2D::new(0.0, 0.0, 200.0, 200.0);

    let query_cost = |gap: Option<u64>| -> (usize, Vec<usize>) {
        let limits = StreamLimits {
            coalesce_gap_bytes: gap,
            ..Default::default()
        };
        let idx = StreamIndex2D::open_with_limits(
            CountingReader::new(SliceReader::new(bytes.clone())),
            limits,
        )
        .unwrap();
        let after_open = *idx.core.reader.reads.borrow();
        let mut hits = idx.search(q).unwrap();
        hits.sort_unstable();
        let after_query = *idx.core.reader.reads.borrow();
        (after_query - after_open, hits)
    };

    let (default_reads, default_hits) = query_cost(None);
    let (wide_reads, wide_hits) = query_cost(Some(256 * 1024));

    assert_eq!(default_hits, wide_hits, "gap must not change results");
    assert!(default_reads > 0, "leaves should stream at this size");
    assert!(
        wide_reads < default_reads,
        "wider gap ({wide_reads}) should read less than default ({default_reads})"
    );
}

#[test]
fn from_directory_rejects_dimension_mismatch() {
    // A 3D directory reattached as a 2D index must be refused, not misread.
    let mut b = crate::Index3DBuilder::new(64);
    for i in 0..64 {
        let v = i as f64;
        b.add(crate::Box3D::new(v, v, v, v + 1.0, v + 1.0, v + 1.0));
    }
    let bytes = b.finish().unwrap().to_bytes();
    let idx3d = StreamIndex3D::open(SliceReader::new(bytes)).unwrap();
    let (dir3d, _reader) = idx3d.into_directory();
    match StreamIndex2D::from_directory(&dir3d, SliceReader::new(Vec::new())) {
        Err(StreamError::Format(LoadError::UnsupportedVersion)) => {}
        Err(other) => panic!("expected dimension-mismatch rejection, got {other:?}"),
        Ok(_) => panic!("a 3D directory must not reattach as a 2D index"),
    }
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = build_bytes(10, 16);
    bytes[0] ^= 0xFF;
    match StreamIndex2D::open(SliceReader::new(bytes)) {
        Err(StreamError::Format(LoadError::BadMagic)) => {}
        Ok(_) => panic!("expected BadMagic, got a valid index"),
        Err(other) => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn rejects_wrong_variant() {
    // 3D bytes opened as a 2D stream must be rejected on the flags check.
    let mut builder = crate::Index3DBuilder::new(8);
    for i in 0..8 {
        let v = i as f64;
        builder.add(crate::Box3D::new(v, v, v, v + 1.0, v + 1.0, v + 1.0));
    }
    let bytes = builder.finish().unwrap().to_bytes();
    match StreamIndex2D::open(SliceReader::new(bytes)) {
        Err(StreamError::Format(LoadError::UnsupportedVersion)) => {}
        Ok(_) => panic!("expected a flag-mismatch rejection, got a valid index"),
        Err(other) => panic!("expected UnsupportedVersion (flag mismatch), got {other:?}"),
    }
}

#[test]
fn rejects_length_mismatch() {
    let mut bytes = build_bytes(10, 16);
    bytes.push(0); // one trailing byte the header does not account for
    match StreamIndex2D::open(SliceReader::new(bytes)) {
        Err(StreamError::Format(LoadError::LengthMismatch { .. })) => {}
        Ok(_) => panic!("expected LengthMismatch, got a valid index"),
        Err(other) => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_truncated_header() {
    let bytes = build_bytes(10, 16);
    let short = bytes[..40].to_vec(); // shorter than the 32-byte superblock
    match StreamIndex2D::open(SliceReader::new(short)) {
        Err(StreamError::Io(err)) if err.kind() == io::ErrorKind::UnexpectedEof => {}
        Ok(_) => panic!("expected UnexpectedEof, got a valid index"),
        Err(other) => panic!("expected UnexpectedEof, got {other:?}"),
    }
}

#[test]
fn open_is_bounded_and_does_not_read_everything() {
    // A large index: open must touch only header + level_bounds + the two
    // directory ranges, reading far less than the whole file.
    let bytes = build_bytes(100_000, 16);
    let file_len = bytes.len() as u64;
    let reader = CountingReader::new(SliceReader::new(bytes));
    let stream = StreamIndex2D::open(reader).unwrap();

    let reads = *stream.core.reader.reads.borrow();
    let read_bytes = *stream.core.reader.bytes.borrow();
    // open: leading read + directory + TREE descriptor + two directory ranges.
    assert!(reads <= 6, "open should issue at most 6 reads, did {reads}");
    assert!(
        read_bytes * 4 < file_len,
        "open read {read_bytes} of {file_len} bytes; should be a small fraction"
    );
}

#[test]
fn directory_covers_all_levels_above_the_leaves() {
    // With the default budget the directory should reach down to (but not
    // include) the leaf level for a mid-sized index, so traversal only ever
    // streams the leaves.
    let bytes = build_bytes(50_000, 16);
    let stream = open_slice(bytes);
    // Leaf level ends at level_bounds[0] = num_items; the directory starting
    // exactly there means every internal level is cached.
    assert_eq!(stream.core.dir_node_start, stream.core.level_bounds[0]);
}

/// Build random boxes; return both the owned index and its serialized bytes.
fn random_owned(n: usize, seed: u64) -> (crate::Index2D, Vec<u8>) {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let mut builder = Index2DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let cx: f64 = rng.random_range(0.0..1000.0);
        let cy: f64 = rng.random_range(0.0..1000.0);
        let w: f64 = rng.random_range(0.1..10.0);
        let h: f64 = rng.random_range(0.1..10.0);
        builder.add(Box2D::new(cx, cy, cx + w, cy + h));
    }
    let owned = builder.finish().unwrap();
    let bytes = owned.to_bytes();
    (owned, bytes)
}

#[test]
fn streamed_search_matches_owned() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    // 20k items so the leaf level (> the 8192-node directory budget) is
    // genuinely streamed and coalesced, not served entirely from cache.
    let (owned, bytes) = random_owned(20_000, 0xC0FFEE);
    let stream = open_slice(bytes);
    assert!(stream.core.dir_node_start > 0, "leaves should be streamed");

    let mut rng = StdRng::seed_from_u64(0xBEEF);
    for _ in 0..200 {
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let qw: f64 = rng.random_range(0.0..200.0);
        let qh: f64 = rng.random_range(0.0..200.0);
        let query = Box2D::new(qx, qy, qx + qw, qy + qh);

        let mut streamed = stream.search(query).unwrap();
        let mut owned_hits = owned.search(query);
        streamed.sort_unstable();
        owned_hits.sort_unstable();
        assert_eq!(streamed, owned_hits, "query {query:?}");
    }
}

#[test]
fn edge_queries_match_owned() {
    let (owned, bytes) = random_owned(20_000, 0x1234);
    let stream = open_slice(bytes);

    // Full extent: every item.
    let full = Box2D::new(-1.0, -1.0, 2000.0, 2000.0);
    let mut a = stream.search(full).unwrap();
    let mut b = owned.search(full);
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
    assert_eq!(a.len(), 20_000);

    // No match: far away.
    assert!(
        stream
            .search(Box2D::new(1e9, 1e9, 1e9 + 1.0, 1e9 + 1.0))
            .unwrap()
            .is_empty()
    );

    // Empty index.
    let empty = open_slice(build_bytes(0, 16));
    assert!(
        empty
            .search(Box2D::new(0.0, 0.0, 1.0, 1.0))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn query_streams_only_a_small_part_of_the_leaves() {
    // A tight query over a large index should fetch only a few leaf groups,
    // not the whole leaf section.
    let (_, bytes) = random_owned(50_000, 0x77);
    let file_len = bytes.len() as u64;
    let stream = StreamIndex2D::open(CountingReader::new(SliceReader::new(bytes))).unwrap();

    let reads_after_open = *stream.core.reader.reads.borrow();
    let bytes_after_open = *stream.core.reader.bytes.borrow();

    let _ = stream
        .search(Box2D::new(500.0, 500.0, 505.0, 505.0))
        .unwrap();

    let query_reads = *stream.core.reader.reads.borrow() - reads_after_open;
    let query_bytes = *stream.core.reader.bytes.borrow() - bytes_after_open;
    assert!(query_reads <= 8, "tight query issued {query_reads} reads");
    assert!(
        query_bytes * 8 < file_len,
        "tight query read {query_bytes} of {file_len} bytes"
    );
}

#[test]
#[cfg(any(unix, windows))]
fn file_reader_search_matches_owned() {
    let (owned, bytes) = random_owned(20_000, 0xF11E);
    let path = std::env::temp_dir().join(format!(
        "psi_stream_{}_{}.psindex",
        std::process::id(),
        "search"
    ));
    std::fs::write(&path, &bytes).unwrap();

    let stream = StreamIndex2D::open(FileReader::open(&path).unwrap()).unwrap();
    let query = Box2D::new(400.0, 400.0, 460.0, 460.0);
    let mut streamed = stream.search(query).unwrap();
    let mut owned_hits = owned.search(query);
    streamed.sort_unstable();
    owned_hits.sort_unstable();
    assert_eq!(streamed, owned_hits);

    std::fs::remove_file(&path).ok();
}

#[test]
fn streamed_search_matches_owned_3d() {
    use crate::{Box3D, Index3DBuilder};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(0x3D3D);
    let n = 20_000;
    let mut builder = Index3DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let cx: f64 = rng.random_range(0.0..1000.0);
        let cy: f64 = rng.random_range(0.0..1000.0);
        let cz: f64 = rng.random_range(0.0..1000.0);
        let w: f64 = rng.random_range(0.1..10.0);
        let h: f64 = rng.random_range(0.1..10.0);
        let d: f64 = rng.random_range(0.1..10.0);
        builder.add(Box3D::new(cx, cy, cz, cx + w, cy + h, cz + d));
    }
    let owned = builder.finish().unwrap();
    let stream = StreamIndex3D::open(SliceReader::new(owned.to_bytes())).unwrap();
    assert!(stream.core.dir_node_start > 0, "leaves should be streamed");

    for _ in 0..200 {
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let qz: f64 = rng.random_range(0.0..1000.0);
        let q = Box3D::new(qx, qy, qz, qx + 200.0, qy + 200.0, qz + 200.0);
        let mut streamed = stream.search(q).unwrap();
        let mut owned_hits = owned.search(q);
        streamed.sort_unstable();
        owned_hits.sort_unstable();
        assert_eq!(streamed, owned_hits, "query {q:?}");
    }
}

#[test]
fn three_d_bytes_rejected_as_2d_and_vice_versa() {
    // A 2D index opened as a 3D stream (and the reverse) must be rejected on
    // the flags check, never misread.
    let two_d = build_bytes(64, 16);
    match StreamIndex3D::open(SliceReader::new(two_d)) {
        Err(StreamError::Format(LoadError::UnsupportedVersion)) => {}
        Ok(_) => panic!("2D-as-3D should be rejected, got a valid index"),
        Err(other) => panic!("2D-as-3D should be rejected, got {other:?}"),
    }
}

// ---- Hardening: untrusted / adversarial input ----

/// A reader that hides its length, like a plain HTTP source without a HEAD.
/// `open` then skips the exact-length cross-check.
struct NoLenReader<R>(R);

impl<R: RangeReader> RangeReader for NoLenReader<R> {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.0.read_exact_at(offset, buf)
    }
    fn len(&self) -> Option<u64> {
        None
    }
}

/// Byte offset of the index section (start of the `TREE` chunk's indices, for
/// the SoA layout) — the streaming core already resolved it as `idx0`.
fn indices_offset(stream: &StreamIndex2D<SliceReader<Vec<u8>>>) -> usize {
    stream.core.idx0 as usize
}

#[test]
fn fully_cached_small_index_search_matches_owned() {
    // Small enough that the whole tree (incl. leaves) fits the directory
    // budget, so search is served entirely from cache — exercises the
    // cached-copy path of `gather` end to end.
    let (owned, bytes) = random_owned(500, 0x5A5A);
    let stream = open_slice(bytes);
    assert_eq!(stream.core.dir_node_start, 0, "whole tree should be cached");

    for q in [
        Box2D::new(0.0, 0.0, 500.0, 500.0),
        Box2D::new(100.0, 100.0, 120.0, 120.0),
        Box2D::new(-9.0, -9.0, -8.0, -8.0),
    ] {
        let mut a = stream.search(q).unwrap();
        let mut b = owned.search(q);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "query {q:?}");
    }
}

#[test]
fn unknown_length_reader_works() {
    let (owned, bytes) = random_owned(20_000, 0xA11);
    let stream = StreamIndex2D::open(NoLenReader(SliceReader::new(bytes))).unwrap();
    let q = Box2D::new(300.0, 300.0, 360.0, 360.0);
    let mut a = stream.search(q).unwrap();
    let mut b = owned.search(q);
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}

#[test]
fn too_short_body_rejected() {
    let mut bytes = build_bytes(1000, 16);
    bytes.truncate(bytes.len() - 8); // drop one index entry
    // The TREE chunk now claims more bytes than the file holds.
    match StreamIndex2D::open(SliceReader::new(bytes)) {
        Err(StreamError::Format(LoadError::InvalidTree | LoadError::LengthMismatch { .. })) => {}
        Ok(_) => panic!("expected rejection, got a valid index"),
        Err(other) => panic!("expected InvalidTree/LengthMismatch, got {other:?}"),
    }
}

#[test]
fn corrupt_leaf_index_is_rejected_not_misread() {
    let (_, mut bytes) = random_owned(1000, 0x9);
    let idx0 = indices_offset(&open_slice(bytes.clone()));
    // Leaf position 0 -> an item id far beyond num_items.
    bytes[idx0..idx0 + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let stream = open_slice(bytes); // open does not validate indices
    match stream.search(Box2D::new(-1.0, -1.0, 2000.0, 2000.0)) {
        Err(StreamError::Format(LoadError::InvalidTree | LoadError::IntegerOverflow)) => {}
        other => panic!("expected a rejection, got {other:?}"),
    }
}

#[test]
fn corrupt_internal_pointer_is_rejected_not_misread() {
    let (_, mut bytes) = random_owned(1000, 0xA);
    let opened = open_slice(bytes.clone());
    let idx0 = indices_offset(&opened);
    let num_items = opened.core.num_items;
    // First internal node (position num_items) -> a child pointer out of range.
    let off = idx0 + num_items * 8;
    bytes[off..off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let stream = open_slice(bytes);
    match stream.search(Box2D::new(-1.0, -1.0, 2000.0, 2000.0)) {
        Err(StreamError::Format(LoadError::InvalidTree | LoadError::IntegerOverflow)) => {}
        other => panic!("expected a rejection, got {other:?}"),
    }
}

#[test]
fn deep_tree_small_node_size_matches_owned() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    // node_size 4 + 30k items: a deep tree where both the leaves and the
    // level above them are streamed (directory caches only higher levels),
    // exercising coalesced streaming of internal nodes, not just leaves.
    let mut rng = StdRng::seed_from_u64(0xDEE9);
    let n = 30_000;
    let mut builder = Index2DBuilder::new(n).node_size(4);
    for _ in 0..n {
        let cx: f64 = rng.random_range(0.0..1000.0);
        let cy: f64 = rng.random_range(0.0..1000.0);
        let w: f64 = rng.random_range(0.1..10.0);
        let h: f64 = rng.random_range(0.1..10.0);
        builder.add(Box2D::new(cx, cy, cx + w, cy + h));
    }
    let owned = builder.finish().unwrap();
    let stream = open_slice(owned.to_bytes());
    assert!(stream.core.level_count >= 7, "tree should be deep");
    assert!(
        stream.core.dir_node_start > stream.core.level_bounds[0],
        "at least leaves and the level above should be streamed"
    );

    for _ in 0..100 {
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let q = Box2D::new(qx, qy, qx + 150.0, qy + 150.0);
        let mut a = stream.search(q).unwrap();
        let mut b = owned.search(q);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "query {q:?}");
    }
}

#[test]
fn concurrent_queries_on_shared_reader() {
    // The `&self` positioned-read contract should let one reader serve many
    // queries at once.
    let (owned, bytes) = random_owned(20_000, 0xCAFE);
    let stream = open_slice(bytes);
    std::thread::scope(|scope| {
        for t in 0..4 {
            let stream = &stream;
            let owned = &owned;
            scope.spawn(move || {
                let base = t as f64 * 200.0;
                let q = Box2D::new(base, base, base + 120.0, base + 120.0);
                let mut a = stream.search(q).unwrap();
                let mut b = owned.search(q);
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b);
            });
        }
    });
}

#[test]
fn corrupt_bytes_never_panic() {
    // Flip a byte at many positions across a valid index and confirm neither
    // `open` nor a full-extent query ever panics — they return Ok or Err.
    // Covers in-range-but-reordered/aliased pointers (the frontier sort/dedup
    // guard) and arbitrary box/level corruption.
    let (_, base) = random_owned(800, 0xF0F0);
    let query = Box2D::new(-1.0, -1.0, 2000.0, 2000.0);
    for i in (0..base.len()).step_by(37) {
        let mut bytes = base.clone();
        bytes[i] ^= 0xFF;
        if let Ok(stream) = StreamIndex2D::open(SliceReader::new(bytes)) {
            // Must terminate without panicking; result correctness is not
            // asserted for a corrupt index, only that it does not crash.
            let _ = stream.search(query);
        }
    }
}

#[test]
fn corrupt_payload_bytes_never_panic() {
    // Flip a byte across the WHOLE payload file (header, index, offset table,
    // blobs) and confirm `open` + `search_payloads` never panic. Covers the
    // untrusted-offset path (e.g. a run with out-of-order offsets, which must
    // be rejected, not underflow the blob slice).
    let (_, _, base) = random_with_payloads(500, 0xF0F1);
    let query = Box2D::new(-1.0, -1.0, 2000.0, 2000.0);
    for i in (0..base.len()).step_by(31) {
        let mut bytes = base.clone();
        bytes[i] ^= 0xFF;
        if let Ok(stream) = StreamIndex2D::open(SliceReader::new(bytes)) {
            let _ = stream.search_payloads(query);
        }
    }
}

#[test]
fn corrupt_fixed_width_payload_bytes_never_panic() {
    // Same fuzz for the table-less layout: flip a byte across the whole file
    // (including the record_stride field and the blob region) and confirm
    // open + search_payloads never panic.
    const STRIDE: usize = 12;
    let (owned, _) = random_owned(500, 0xF1F2);
    let n = owned.num_items();
    let flat = vec![0x5Au8; n * STRIDE];
    let base = owned.serialize().records(STRIDE, &flat).to_bytes().unwrap();
    let query = Box2D::new(-1.0, -1.0, 2000.0, 2000.0);
    for i in (0..base.len()).step_by(29) {
        let mut bytes = base.clone();
        bytes[i] ^= 0xFF;
        if let Ok(stream) = StreamIndex2D::open(SliceReader::new(bytes)) {
            let _ = stream.search_payloads(query);
        }
    }
}

#[test]
fn out_of_order_payload_offset_is_rejected() {
    // Directly craft an out-of-order offset entry and confirm search_payloads
    // rejects it (InvalidTree) rather than panicking.
    let (_, _, mut bytes) = random_with_payloads(1_000, 0x0FF5);
    let stream = open_slice(bytes.clone());
    let offsets_start = stream.core.payload.as_ref().unwrap().offsets_start as usize;
    // Set offset entry 1 to a huge value (> later entries -> out of order).
    bytes[offsets_start + 8..offsets_start + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    let stream = open_slice(bytes);
    match stream.search_payloads(Box2D::new(-1.0, -1.0, 2000.0, 2000.0)) {
        Err(StreamError::Format(LoadError::InvalidTree | LoadError::IntegerOverflow)) => {}
        other => panic!("expected rejection, got {:?}", other.map(|v| v.len())),
    }
}

// ---- Payload ----

/// Build a random index plus a variable-length payload per item; return the
/// owned index, the payloads, and the payload-carrying bytes.
fn random_with_payloads(n: usize, seed: u64) -> (crate::Index2D, Vec<Vec<u8>>, Vec<u8>) {
    let (owned, _) = random_owned(n, seed);
    let payloads: Vec<Vec<u8>> = (0..n)
        .map(|i| format!("payload-for-item-{i}").into_bytes())
        .collect();
    let bytes = owned.to_bytes_with_payloads(&payloads).unwrap();
    (owned, payloads, bytes)
}

#[test]
fn streamed_payloads_round_trip_with_search() {
    // 20k items so leaves stream; payloads come back paired with ids.
    let (owned, payloads, bytes) = random_with_payloads(20_000, 0x9EED);
    let stream = open_slice(bytes);
    assert!(stream.has_payload());

    let query = Box2D::new(400.0, 400.0, 460.0, 460.0);
    let pairs = stream.search_payloads(query).unwrap();

    // The id set equals a plain search, and each blob matches the original.
    let mut got_ids: Vec<usize> = pairs.iter().map(|(id, _)| *id).collect();
    let mut want_ids = owned.search(query);
    got_ids.sort_unstable();
    want_ids.sort_unstable();
    assert_eq!(got_ids, want_ids);
    for (id, blob) in &pairs {
        assert_eq!(blob, &payloads[*id]);
    }

    // Full-extent: every payload streams back.
    let all = stream
        .search_payloads(Box2D::new(-1.0, -1.0, 2000.0, 2000.0))
        .unwrap();
    assert_eq!(all.len(), 20_000);
    for (id, blob) in &all {
        assert_eq!(blob, &payloads[*id]);
    }
}

#[test]
fn fixed_width_payload_streams_table_less() {
    const STRIDE: usize = 12;
    let (owned, _) = random_owned(20_000, 0x713A);
    let n = owned.num_items();
    // Record `id` encodes its own id, so streamed blobs are self-checking.
    let mut flat = vec![0u8; n * STRIDE];
    for id in 0..n {
        flat[id * STRIDE..id * STRIDE + 8].copy_from_slice(&(id as u64).to_le_bytes());
        flat[id * STRIDE + 8..id * STRIDE + STRIDE].copy_from_slice(&[0xAB, 0xCD, id as u8, 0]);
    }
    let fixed_bytes = owned.serialize().records(STRIDE, &flat).to_bytes().unwrap();
    let variable: Vec<Vec<u8>> = (0..n)
        .map(|id| flat[id * STRIDE..(id + 1) * STRIDE].to_vec())
        .collect();
    let var_bytes = owned.serialize().payloads(&variable).to_bytes().unwrap();

    let stream = open_slice(fixed_bytes.clone());
    assert!(stream.has_payload());
    assert!(stream.core.dir_node_start > 0, "leaves should be streamed");

    let query = Box2D::new(400.0, 400.0, 460.0, 460.0);
    let pairs = stream.search_payloads(query).unwrap();
    let mut got: Vec<usize> = pairs.iter().map(|(id, _)| *id).collect();
    let mut want = owned.search(query);
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
    for (id, blob) in &pairs {
        assert_eq!(blob.as_slice(), &flat[*id * STRIDE..(*id + 1) * STRIDE]);
        assert_eq!(&blob[..8], &(*id as u64).to_le_bytes());
    }

    // Full-extent: every record streams back.
    let all = stream
        .search_payloads(Box2D::new(-1.0, -1.0, 2000.0, 2000.0))
        .unwrap();
    assert_eq!(all.len(), n);

    // Table-less wins a round trip: the same windowed query reads strictly
    // fewer times than the variable layout (which also reads the offset table).
    let fixed_r = StreamIndex2D::open(CountingReader::new(SliceReader::new(fixed_bytes))).unwrap();
    let before = *fixed_r.core.reader.reads.borrow();
    let _ = fixed_r.search_payloads(query).unwrap();
    let fixed_reads = *fixed_r.core.reader.reads.borrow() - before;

    let var_r = StreamIndex2D::open(CountingReader::new(SliceReader::new(var_bytes))).unwrap();
    let before = *var_r.core.reader.reads.borrow();
    let _ = var_r.search_payloads(query).unwrap();
    let var_reads = *var_r.core.reader.reads.borrow() - before;

    assert!(
        fixed_reads < var_reads,
        "fixed {fixed_reads} should read fewer than variable {var_reads}"
    );
}

#[test]
fn interleaved_search_matches_soa_and_owned() {
    // The interleaved layout must return identical results to the default SoA
    // layout (and the owned index) for plain search and search_payloads.
    for &n in &[0usize, 1, 16, 17, 1000, 20_000] {
        let (owned, _) = random_owned(n, 0xC0FFEE ^ n as u64);
        let payloads: Vec<Vec<u8>> = (0..n)
            .map(|i| format!("blob-{i}-xx").into_bytes())
            .collect();

        let soa = open_slice(owned.to_bytes());
        let inter = open_slice(owned.to_bytes_interleaved());
        let inter_pay = open_slice(owned.to_bytes_interleaved_with_payloads(&payloads).unwrap());
        assert!(inter_pay.has_payload(), "n={n}");

        for q in [
            Box2D::new(400.0, 400.0, 460.0, 460.0),
            Box2D::new(-1.0, -1.0, 2000.0, 2000.0),
            Box2D::new(0.0, 0.0, 100.0, 100.0),
        ] {
            let mut want = owned.search(q);
            want.sort_unstable();
            let mut from_soa = soa.search(q).unwrap();
            from_soa.sort_unstable();
            let mut from_inter = inter.search(q).unwrap();
            from_inter.sort_unstable();
            assert_eq!(from_soa, want, "soa n={n}");
            assert_eq!(from_inter, want, "interleaved n={n}");

            let pairs = inter_pay.search_payloads(q).unwrap();
            let mut ids: Vec<usize> = pairs.iter().map(|(id, _)| *id).collect();
            ids.sort_unstable();
            assert_eq!(ids, want, "interleaved payloads n={n}");
            for (id, blob) in &pairs {
                assert_eq!(blob, &payloads[*id], "blob n={n}");
            }
        }
    }
}

#[test]
fn interleaved_uses_fewer_reads_than_soa() {
    // The interleaved layout fetches a node's box and pointer together, so a
    // query issues fewer reads than the SoA layout's separate box/index passes.
    let (owned, _) = random_owned(50_000, 0x5EED);
    let query = Box2D::new(300.0, 300.0, 360.0, 360.0);

    let soa = StreamIndex2D::open(CountingReader::new(SliceReader::new(owned.to_bytes()))).unwrap();
    soa.search(query).unwrap();
    let soa_reads = *soa.core.reader.reads.borrow();

    let inter = StreamIndex2D::open(CountingReader::new(SliceReader::new(
        owned.to_bytes_interleaved(),
    )))
    .unwrap();
    inter.search(query).unwrap();
    let inter_reads = *inter.core.reader.reads.borrow();

    assert!(
        inter_reads < soa_reads,
        "interleaved {inter_reads} should be fewer reads than SoA {soa_reads}"
    );
}

#[test]
fn interleaved_rejected_by_soa_loaders() {
    // An interleaved file is streaming-targeted; the in-memory loaders and
    // views read the SoA layout only and must reject it cleanly.
    let (owned, _) = random_owned(100, 0x1);
    let bytes = owned.to_bytes_interleaved();
    assert!(matches!(
        crate::Index2D::from_bytes(&bytes),
        Err(LoadError::UnsupportedVersion)
    ));
    assert!(matches!(
        crate::Index2DView::from_bytes(&bytes),
        Err(LoadError::UnsupportedVersion)
    ));
}

#[test]
fn interleaved_corrupt_bytes_never_panic() {
    // Fuzz: flipping bytes of an interleaved payload file must never panic;
    // open / search / search_payloads return Ok or Err.
    let (owned, _) = random_owned(500, 0x77);
    let payloads: Vec<Vec<u8>> = (0..500).map(|i| vec![i as u8; (i % 7) + 1]).collect();
    let clean = owned.to_bytes_interleaved_with_payloads(&payloads).unwrap();
    let query = Box2D::new(-1.0, -1.0, 2000.0, 2000.0);
    for i in (0..clean.len()).step_by(31) {
        let mut bytes = clean.clone();
        bytes[i] ^= 0xA5;
        if let Ok(stream) = StreamIndex2D::open(SliceReader::new(bytes)) {
            let _ = stream.search(query);
            let _ = stream.search_payloads(query);
        }
    }
}

/// A `RangeReader` that serves a clean header + `level_bounds` (so `open`
/// succeeds) but returns adversarial garbage for every byte of the node and
/// payload sections, varying per read. Models a hostile or inconsistent
/// backing store: the streaming reader reads each range once and validates
/// every file-derived value at use, so the descent must yield `Ok` or `Err`,
/// never panic, no matter what bytes come back.
struct HostileReader {
    clean: Vec<u8>,
    clean_below: u64,
    counter: RefCell<u8>,
}

impl HostileReader {
    fn new(clean: Vec<u8>) -> Self {
        // level_bounds ends at 64 + 8 * level_count (header field at offset 56).
        let level_count = u64::from_le_bytes(clean[56..64].try_into().unwrap());
        let clean_below = 64 + 8 * level_count;
        Self {
            clean,
            clean_below,
            counter: RefCell::new(1),
        }
    }
}

impl RangeReader for HostileReader {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset).map_err(|_| unexpected_eof())?;
        let end = start.checked_add(buf.len()).ok_or_else(unexpected_eof)?;
        let src = self.clean.get(start..end).ok_or_else(unexpected_eof)?;
        let mut c = self.counter.borrow_mut();
        for (i, (dst, &b)) in buf.iter_mut().zip(src).enumerate() {
            let pos = offset + i as u64;
            // Pristine header + level_bounds; everything else is corrupted with
            // a per-read-varying mask so no two reads agree.
            *dst = if pos < self.clean_below {
                b
            } else {
                b ^ c.wrapping_add(pos as u8) ^ 0x5A
            };
        }
        *c = c.wrapping_add(31);
        Ok(())
    }

    fn len(&self) -> Option<u64> {
        Some(self.clean.len() as u64)
    }
}

#[test]
fn hostile_reader_never_panics() {
    // The node/payload bytes are adversarial (and inconsistent across reads),
    // but the header/level_bounds are valid. The descent reads each range once
    // and validates every file-derived value at use, so it must never panic.
    let (owned, _) = random_owned(2_000, 0xDEAD);
    let payloads: Vec<Vec<u8>> = (0..2_000).map(|i| vec![i as u8; (i % 5) + 1]).collect();
    let queries = [
        Box2D::new(-1.0, -1.0, 2000.0, 2000.0),
        Box2D::new(400.0, 400.0, 460.0, 460.0),
        Box2D::new(0.0, 0.0, 10.0, 10.0),
    ];

    // Index-only files have no blob total for `open` to read from the hostile
    // region, so `open` succeeds and the search descent runs entirely against
    // hostile node bytes.
    for clean in [owned.to_bytes(), owned.to_bytes_interleaved()] {
        let stream = StreamIndex2D::open(HostileReader::new(clean)).unwrap();
        for q in queries {
            let _ = stream.search(q);
        }
    }

    // Payload files: `open` reads the (hostile) blob total and may reject the
    // file on the length cross-check — a valid outcome. When it does open, the
    // payload descent must still never panic.
    for clean in [
        owned.to_bytes_with_payloads(&payloads).unwrap(),
        owned.to_bytes_interleaved_with_payloads(&payloads).unwrap(),
    ] {
        if let Ok(stream) = StreamIndex2D::open(HostileReader::new(clean)) {
            for q in queries {
                let _ = stream.search(q);
                let _ = stream.search_payloads(q);
            }
        }
    }
}

#[test]
fn search_payloads_absent_is_nopayload() {
    let (_, bytes) = random_owned(100, 0x1);
    let stream = open_slice(bytes);
    assert!(!stream.has_payload());
    assert!(matches!(
        stream.search_payloads(Box2D::new(0.0, 0.0, 1000.0, 1000.0)),
        Err(StreamError::NoPayload)
    ));
}

#[test]
fn search_payloads_via_file_and_unknown_length_readers() {
    let (_, payloads, bytes) = random_with_payloads(5_000, 0x3);
    let query = Box2D::new(0.0, 0.0, 1000.0, 1000.0);
    let check = |stream: &dyn Fn() -> Vec<(usize, Vec<u8>)>| {
        for (id, blob) in stream() {
            assert_eq!(blob, payloads[id]);
        }
    };

    let path = std::env::temp_dir().join(format!("psi_payload_{}.psindex", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let fstream = StreamIndex2D::open(FileReader::open(&path).unwrap()).unwrap();
    check(&|| fstream.search_payloads(query).unwrap());
    std::fs::remove_file(&path).ok();

    let nstream = StreamIndex2D::open(NoLenReader(SliceReader::new(bytes))).unwrap();
    check(&|| nstream.search_payloads(query).unwrap());
}

#[test]
fn empty_payload_blobs_round_trip() {
    let (owned, _) = random_owned(50, 0x4);
    let payloads: Vec<Vec<u8>> = vec![Vec::new(); 50];
    let bytes = owned.to_bytes_with_payloads(&payloads).unwrap();
    let stream = open_slice(bytes);
    let all = stream
        .search_payloads(Box2D::new(-1.0, -1.0, 2000.0, 2000.0))
        .unwrap();
    assert!(!all.is_empty());
    assert!(all.iter().all(|(_, blob)| blob.is_empty()));
}

// ---- Per-query limits ----

#[test]
fn limits_bound_broad_queries() {
    let (owned, bytes) = random_owned(50_000, 0x71117);
    let full = Box2D::new(-1.0, -1.0, 2000.0, 2000.0);
    let narrow = Box2D::new(500.0, 500.0, 510.0, 510.0);

    // max_items: a broad query aborts; a narrow one (few hits) succeeds.
    let item_capped = StreamIndex2D::open_with_limits(
        SliceReader::new(bytes.clone()),
        StreamLimits {
            max_items: Some(100),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        item_capped.search(full),
        Err(StreamError::LimitExceeded)
    ));
    let mut hits = item_capped.search(narrow).unwrap();
    let mut want = owned.search(narrow);
    hits.sort_unstable();
    want.sort_unstable();
    assert!(hits.len() < 100 && hits == want);

    // max_reads: coalescing keeps the count low even for a huge result (the
    // leaf section is a couple of big reads), so `max_reads` mainly guards
    // scattered queries; a budget below the minimum still aborts.
    let read_capped = StreamIndex2D::open_with_limits(
        SliceReader::new(bytes.clone()),
        StreamLimits {
            max_reads: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        read_capped.search(full),
        Err(StreamError::LimitExceeded)
    ));

    // max_read_bytes: tiny budget aborts the broad query.
    let byte_capped = StreamIndex2D::open_with_limits(
        SliceReader::new(bytes.clone()),
        StreamLimits {
            max_read_bytes: Some(4096),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        byte_capped.search(full),
        Err(StreamError::LimitExceeded)
    ));

    // Default (no limits): the full query returns everything.
    let unlimited = open_slice(bytes);
    assert_eq!(unlimited.search(full).unwrap().len(), 50_000);
}

#[test]
fn limits_bound_payload_queries() {
    let (_, _, bytes) = random_with_payloads(20_000, 0x71118);
    let capped = StreamIndex2D::open_with_limits(
        SliceReader::new(bytes),
        StreamLimits {
            max_items: Some(50),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        capped.search_payloads(Box2D::new(-1.0, -1.0, 2000.0, 2000.0)),
        Err(StreamError::LimitExceeded)
    ));
}

#[cfg(feature = "async")]
#[test]
fn async_limits_match_sync() {
    let (_, bytes) = random_owned(50_000, 0x71119);
    let limits = StreamLimits {
        max_items: Some(100),
        ..Default::default()
    };
    let astream = pollster::block_on(StreamIndex2D::open_with_limits_async(
        AsyncSlice(bytes),
        limits,
    ))
    .unwrap();
    let full = Box2D::new(-1.0, -1.0, 2000.0, 2000.0);
    assert!(matches!(
        pollster::block_on(astream.search_async(full)),
        Err(StreamError::LimitExceeded)
    ));
}

#[test]
fn search_payloads_streams_few_reads() {
    // A tight query over a payload index should fetch payloads in a handful
    // of coalesced reads, not one per hit.
    let (_, _, bytes) = random_with_payloads(50_000, 0x55);
    let stream = StreamIndex2D::open(CountingReader::new(SliceReader::new(bytes))).unwrap();
    let reads_before = *stream.core.reader.reads.borrow();
    let pairs = stream
        .search_payloads(Box2D::new(500.0, 500.0, 540.0, 540.0))
        .unwrap();
    let query_reads = *stream.core.reader.reads.borrow() - reads_before;
    assert!(!pairs.is_empty());
    assert!(
        query_reads <= 16,
        "search_payloads issued {query_reads} reads for {} hits",
        pairs.len()
    );
}

#[test]
fn streamed_3d_payload_round_trips_with_search() {
    use crate::{Box3D, Index3DBuilder};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(0x3D_0AD);
    let n = 20_000;
    let mut builder = Index3DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let c: [f64; 3] = [
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
        ];
        builder.add(Box3D::new(
            c[0],
            c[1],
            c[2],
            c[0] + 2.0,
            c[1] + 2.0,
            c[2] + 2.0,
        ));
    }
    let owned = builder.finish().unwrap();
    let payloads: Vec<Vec<u8>> = (0..n)
        .map(|i| format!("3d-blob-{i}").into_bytes())
        .collect();
    let bytes = owned.to_bytes_with_payloads(&payloads).unwrap();

    let stream = StreamIndex3D::open(SliceReader::new(bytes)).unwrap();
    assert!(stream.has_payload());

    let query = Box3D::new(400.0, 400.0, 400.0, 460.0, 460.0, 460.0);
    let pairs = stream.search_payloads(query).unwrap();
    let mut got: Vec<usize> = pairs.iter().map(|(id, _)| *id).collect();
    let mut want = owned.search(query);
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
    for (id, blob) in &pairs {
        assert_eq!(blob, &payloads[*id]);
    }
}

// ---- Async (equivalence with the sync path) ----

#[cfg(feature = "async")]
struct AsyncSlice(Vec<u8>);

#[cfg(feature = "async")]
impl AsyncRangeReader for AsyncSlice {
    async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset).map_err(|_| unexpected_eof())?;
        let end = start.checked_add(buf.len()).ok_or_else(unexpected_eof)?;
        let src = self.0.get(start..end).ok_or_else(unexpected_eof)?;
        buf.copy_from_slice(src);
        Ok(())
    }
    fn len(&self) -> Option<u64> {
        Some(self.0.len() as u64)
    }
}

/// A future that returns `Pending` exactly once (waking itself) so that a
/// read appears in flight for one poll round before completing.
#[cfg(feature = "async")]
struct YieldOnce(bool);

#[cfg(feature = "async")]
impl std::future::Future for YieldOnce {
    type Output = ();
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

/// An async reader that yields once per read and tracks the peak number of
/// reads in flight at the same time — `> 1` proves a level's reads were
/// issued concurrently rather than awaited one by one.
#[cfg(feature = "async")]
struct YieldReader {
    inner: Vec<u8>,
    in_flight: std::cell::Cell<usize>,
    peak: std::cell::Cell<usize>,
}

#[cfg(feature = "async")]
impl AsyncRangeReader for YieldReader {
    async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.in_flight.set(self.in_flight.get() + 1);
        self.peak.set(self.peak.get().max(self.in_flight.get()));
        YieldOnce(false).await;
        self.in_flight.set(self.in_flight.get() - 1);
        let start = usize::try_from(offset).map_err(|_| unexpected_eof())?;
        let end = start.checked_add(buf.len()).ok_or_else(unexpected_eof)?;
        let src = self.inner.get(start..end).ok_or_else(unexpected_eof)?;
        buf.copy_from_slice(src);
        Ok(())
    }
    fn len(&self) -> Option<u64> {
        Some(self.inner.len() as u64)
    }
}

#[cfg(feature = "async")]
#[test]
fn async_reads_a_level_concurrently() {
    // A 2D window query crosses the Hilbert curve several times, so the leaf
    // gather has multiple coalesced runs; the async path must issue them
    // concurrently (peak in-flight > 1).
    let (owned, bytes) = random_owned(50_000, 0xC04C);
    let reader = YieldReader {
        inner: bytes,
        in_flight: std::cell::Cell::new(0),
        peak: std::cell::Cell::new(0),
    };
    let stream = pollster::block_on(StreamIndex2D::open_async(reader)).unwrap();
    let query = Box2D::new(200.0, 200.0, 600.0, 600.0);

    let mut got = pollster::block_on(stream.search_async(query)).unwrap();
    let mut want = owned.search(query);
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
    let peak = stream.core.reader.peak.get();
    assert!(
        peak > 1,
        "expected concurrent reads, peak in-flight was {peak}"
    );
}

#[cfg(feature = "async")]
#[test]
fn async_search_matches_sync() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let (_, bytes) = random_owned(20_000, 0xA5);
    let sync = open_slice(bytes.clone());
    let astream = pollster::block_on(StreamIndex2D::open_async(AsyncSlice(bytes))).unwrap();

    let mut rng = StdRng::seed_from_u64(0xA51);
    for _ in 0..100 {
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let q = Box2D::new(qx, qy, qx + 150.0, qy + 150.0);
        let mut s = sync.search(q).unwrap();
        let mut a = pollster::block_on(astream.search_async(q)).unwrap();
        s.sort_unstable();
        a.sort_unstable();
        assert_eq!(s, a, "query {q:?}");
    }
}

#[cfg(feature = "async")]
#[test]
fn async_search_payloads_matches_sync() {
    let (_, payloads, bytes) = random_with_payloads(20_000, 0xA6);
    let sync = open_slice(bytes.clone());
    let astream = pollster::block_on(StreamIndex2D::open_async(AsyncSlice(bytes))).unwrap();

    let q = Box2D::new(300.0, 300.0, 380.0, 380.0);
    let mut sync_pairs = sync.search_payloads(q).unwrap();
    let mut async_pairs = pollster::block_on(astream.search_payloads_async(q)).unwrap();
    sync_pairs.sort();
    async_pairs.sort();
    assert_eq!(sync_pairs, async_pairs);
    for (id, blob) in &async_pairs {
        assert_eq!(blob, &payloads[*id]);
    }
    assert!(astream.has_payload_async());
}

#[cfg(feature = "async")]
#[test]
fn async_fixed_width_payload_matches_sync() {
    const STRIDE: usize = 12;
    let (owned, _) = random_owned(20_000, 0xA6F);
    let n = owned.num_items();
    let mut flat = vec![0u8; n * STRIDE];
    for id in 0..n {
        flat[id * STRIDE..id * STRIDE + 8].copy_from_slice(&(id as u64).to_le_bytes());
    }
    let bytes = owned.serialize().records(STRIDE, &flat).to_bytes().unwrap();
    let sync = open_slice(bytes.clone());
    let astream = pollster::block_on(StreamIndex2D::open_async(AsyncSlice(bytes))).unwrap();

    let q = Box2D::new(300.0, 300.0, 380.0, 380.0);
    let mut sync_pairs = sync.search_payloads(q).unwrap();
    let mut async_pairs = pollster::block_on(astream.search_payloads_async(q)).unwrap();
    sync_pairs.sort();
    async_pairs.sort();
    assert_eq!(sync_pairs, async_pairs);
    for (id, blob) in &async_pairs {
        assert_eq!(&blob[..8], &(*id as u64).to_le_bytes());
    }
}

#[cfg(feature = "async")]
#[test]
fn async_3d_search_payloads_matches_sync() {
    use crate::{Box3D, Index3DBuilder};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(0xA7);
    let n = 20_000;
    let mut builder = Index3DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let c: [f64; 3] = [
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
        ];
        builder.add(Box3D::new(
            c[0],
            c[1],
            c[2],
            c[0] + 2.0,
            c[1] + 2.0,
            c[2] + 2.0,
        ));
    }
    let owned = builder.finish().unwrap();
    let payloads: Vec<Vec<u8>> = (0..n).map(|i| format!("a3d-{i}").into_bytes()).collect();
    let bytes = owned.to_bytes_with_payloads(&payloads).unwrap();

    let astream = pollster::block_on(StreamIndex3D::open_async(AsyncSlice(bytes))).unwrap();
    let q = Box3D::new(300.0, 300.0, 300.0, 380.0, 380.0, 380.0);
    let pairs = pollster::block_on(astream.search_payloads_async(q)).unwrap();
    let mut got: Vec<usize> = pairs.iter().map(|(id, _)| *id).collect();
    let mut want = owned.search(q);
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
    for (id, blob) in &pairs {
        assert_eq!(blob, &payloads[*id]);
    }
}
