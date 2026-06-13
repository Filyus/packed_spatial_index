//! Streaming reader for the packed spatial index binary format.
//!
//! Where [`Index2D::from_bytes`](crate::Index2D::from_bytes) needs the whole
//! serialized index in memory, the streaming reader answers queries by fetching
//! only the byte ranges a traversal actually touches, over a [`RangeReader`].
//! That backing store can be a local file ([`FileReader`]), an in-memory buffer
//! ([`SliceReader`]), or — by implementing the one-method [`RangeReader`] trait
//! — a remote object served through HTTP range requests.
//!
//! This module is the foundation layer: it validates the header and level
//! bounds at [`open`](StreamIndex2D::open) time and prefetches the small upper
//! levels of the tree (the "directory"), so later queries stream only the lower
//! levels they need. The query traversal itself builds on top of this.
//!
//! Available behind the `stream` feature.

use std::io;

use crate::geometry::Box2D;
use crate::persistence::{
    FORMAT_FLAGS_2D, FORMAT_HEADER_LEN, LoadError, parse_and_validate_header,
    read_f64_le_unchecked, read_u64_at, section_layout, validate_level_bounds,
};

/// Upper bound on how many nodes the open-time "directory" prefetch caches.
///
/// The tree is stored leaves-first with the root last, so the upper levels form
/// a contiguous suffix of the box section. We cache levels from the top down
/// while their combined node count stays within this budget; queries then reach
/// those levels with zero I/O and stream only the levels below. 8192 nodes is a
/// few hundred KiB of boxes — small to hold, yet enough to cover every level
/// above the leaves for indexes into the millions of items.
const DIRECTORY_NODE_BUDGET: usize = 8192;

/// A source of bytes addressable by absolute offset.
///
/// This is the only capability [`StreamIndex2D`] needs from its backing store,
/// so a local file, an in-memory slice, or a remote object behind HTTP range
/// requests can all drive the same streaming queries.
///
/// Implementations must read from an absolute offset **without** disturbing any
/// shared cursor (hence `&self`, not `&mut self`), so one reader can serve
/// concurrent queries safely.
// `len` reports the source's total byte length if known; "emptiness" is not a
// meaningful concept for a random-access byte source, so no `is_empty`.
#[allow(clippy::len_without_is_empty)]
pub trait RangeReader {
    /// Read exactly `buf.len()` bytes starting at byte `offset`, filling `buf`.
    ///
    /// Returns an [`io::ErrorKind::UnexpectedEof`] error if fewer bytes are
    /// available. A zero-length `buf` always succeeds.
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Total length in bytes, if known.
    ///
    /// Local files report their size; a remote reader may return [`None`], in
    /// which case [`open`](StreamIndex2D::open) skips the exact-length
    /// cross-check and instead relies on reads past the end failing.
    fn len(&self) -> Option<u64> {
        None
    }
}

fn unexpected_eof() -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "read past the end of the range source",
    )
}

/// A [`RangeReader`] over an in-memory byte buffer (`&[u8]`, `Vec<u8>`, a memory
/// map, ...). Reads are bounds-checked copies out of the buffer.
pub struct SliceReader<T> {
    data: T,
}

impl<T: AsRef<[u8]>> SliceReader<T> {
    /// Wrap an in-memory buffer.
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Return the wrapped buffer.
    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<T: AsRef<[u8]>> RangeReader for SliceReader<T> {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let data = self.data.as_ref();
        let start = usize::try_from(offset).map_err(|_| unexpected_eof())?;
        let end = start.checked_add(buf.len()).ok_or_else(unexpected_eof)?;
        let src = data.get(start..end).ok_or_else(unexpected_eof)?;
        buf.copy_from_slice(src);
        Ok(())
    }

    fn len(&self) -> Option<u64> {
        Some(self.data.as_ref().len() as u64)
    }
}

/// A [`RangeReader`] over a local file using positioned reads.
///
/// Positioned reads (`pread` on Unix, `seek_read` on Windows) don't move a
/// shared cursor, so the reader takes `&self` and one open file can serve many
/// concurrent queries. Available on Unix and Windows; other targets can
/// implement [`RangeReader`] directly.
#[cfg(any(unix, windows))]
pub struct FileReader {
    file: std::fs::File,
    len: u64,
}

#[cfg(any(unix, windows))]
impl FileReader {
    /// Open a file at `path` for streaming reads.
    pub fn open(path: impl AsRef<std::path::Path>) -> io::Result<Self> {
        Self::from_file(std::fs::File::open(path)?)
    }

    /// Wrap an already-open file. Its length is queried once via metadata.
    pub fn from_file(file: std::fs::File) -> io::Result<Self> {
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }
}

#[cfg(any(unix, windows))]
impl RangeReader for FileReader {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        #[cfg(unix)]
        {
            std::os::unix::fs::FileExt::read_exact_at(&self.file, buf, offset)
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut filled = 0usize;
            while filled < buf.len() {
                let n = self
                    .file
                    .seek_read(&mut buf[filled..], offset + filled as u64)?;
                if n == 0 {
                    return Err(unexpected_eof());
                }
                filled += n;
            }
            Ok(())
        }
    }

    fn len(&self) -> Option<u64> {
        Some(self.len)
    }
}

/// Error returned by the streaming reader.
#[derive(Debug)]
pub enum StreamError {
    /// An I/O error from the backing [`RangeReader`].
    Io(io::Error),
    /// The bytes are not a valid index of the expected variant. Carries the same
    /// [`LoadError`] categories as the in-memory loader.
    Format(LoadError),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Io(err) => write!(f, "streaming read failed: {err}"),
            StreamError::Format(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamError::Io(err) => Some(err),
            StreamError::Format(err) => Some(err),
        }
    }
}

impl From<io::Error> for StreamError {
    fn from(err: io::Error) -> Self {
        StreamError::Io(err)
    }
}

impl From<LoadError> for StreamError {
    fn from(err: LoadError) -> Self {
        StreamError::Format(err)
    }
}

/// Dimension-independent streaming state: validated header counts, section
/// offsets, the parsed level bounds, and the cached upper-level directory.
///
/// Both the 2D and (future) 3D streaming indexes wrap one of these; only box
/// parsing and query traversal differ between dimensions.
pub(crate) struct StreamCore<R> {
    // These five describe how to navigate and fetch nodes during a streamed
    // traversal. They are populated and validated at open time; the query
    // traversal that reads them lands in the next layer, so they are
    // `allow(dead_code)` until then.
    #[allow(dead_code)]
    reader: R,
    #[allow(dead_code)]
    level_count: usize,
    /// Exclusive end offset of each level, in node positions (`level_bounds[i]`).
    #[allow(dead_code)]
    level_bounds: Vec<usize>,
    /// Byte offset of the box section.
    #[allow(dead_code)]
    box0: u64,
    /// Byte offset of the index section.
    #[allow(dead_code)]
    idx0: u64,
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    /// Box record size in bytes.
    record: usize,
    /// First node position covered by the cached directory.
    dir_node_start: usize,
    /// Cached box bytes for node positions `[dir_node_start, num_nodes)`.
    dir_boxes: Vec<u8>,
    /// Cached index bytes for the same node positions, fetched together with the
    /// boxes; read by the traversal layer (next step).
    #[allow(dead_code)]
    dir_indices: Vec<u8>,
}

impl<R: RangeReader> StreamCore<R> {
    /// Open and validate an index of the given variant from `reader`.
    fn open(
        reader: R,
        expected_flags: u64,
        dimensions: usize,
        coord_bytes: usize,
    ) -> Result<Self, StreamError> {
        // 1. Header (fixed 64 bytes): magic, version, flags, counts, tree shape.
        let mut header = [0u8; FORMAT_HEADER_LEN];
        reader.read_exact_at(0, &mut header)?;
        let fields = parse_and_validate_header(&header, expected_flags)?;

        // 2. Section offsets, derived purely from the validated header counts.
        let layout = section_layout(
            fields.level_count,
            fields.num_nodes,
            dimensions,
            coord_bytes,
        )?;

        // 3. Cross-check the declared length against the source, when known.
        if let Some(actual) = reader.len()
            && actual != layout.total_len as u64
        {
            return Err(StreamError::Format(LoadError::LengthMismatch {
                expected: layout.total_len,
                actual: usize::try_from(actual).unwrap_or(usize::MAX),
            }));
        }

        // 4. Level bounds (small): read fully, validate, parse to positions.
        let level_bounds_len = fields.level_count * 8;
        let mut level_bounds_bytes = vec![0u8; level_bounds_len];
        reader.read_exact_at(layout.level_bounds_start as u64, &mut level_bounds_bytes)?;
        validate_level_bounds(
            &level_bounds_bytes,
            fields.num_items,
            fields.num_nodes,
            fields.node_size,
            fields.level_count,
        )?;
        let mut level_bounds = Vec::with_capacity(fields.level_count);
        for level in 0..fields.level_count {
            let value = read_u64_at(&level_bounds_bytes, level * 8)
                .and_then(|v| usize::try_from(v).map_err(|_| LoadError::IntegerOverflow))?;
            level_bounds.push(value);
        }

        // 5. Directory: cache the upper levels (a contiguous suffix of the box
        //    and index sections) up to the node budget.
        let dir_node_start =
            directory_start(&level_bounds, fields.level_count, DIRECTORY_NODE_BUDGET);
        let cached_nodes = fields.num_nodes - dir_node_start;

        let mut dir_boxes = vec![0u8; cached_nodes * layout.record];
        if !dir_boxes.is_empty() {
            let offset = layout.box0 + (dir_node_start * layout.record);
            reader.read_exact_at(offset as u64, &mut dir_boxes)?;
        }

        let mut dir_indices = vec![0u8; cached_nodes * 8];
        if !dir_indices.is_empty() {
            let offset = layout.idx0 + (dir_node_start * 8);
            reader.read_exact_at(offset as u64, &mut dir_indices)?;
        }

        Ok(StreamCore {
            reader,
            node_size: fields.node_size,
            num_items: fields.num_items,
            num_nodes: fields.num_nodes,
            level_count: fields.level_count,
            level_bounds,
            record: layout.record,
            box0: layout.box0 as u64,
            idx0: layout.idx0 as u64,
            dir_node_start,
            dir_boxes,
            dir_indices,
        })
    }

    /// Cached box record bytes for node `position`, if the directory covers it.
    fn cached_box_bytes(&self, position: usize) -> Option<&[u8]> {
        if position < self.dir_node_start || position >= self.num_nodes {
            return None;
        }
        let start = (position - self.dir_node_start) * self.record;
        self.dir_boxes.get(start..start + self.record)
    }
}

/// Choose the first node position to cache in the directory: walk levels from
/// the top down while their combined node count stays within `budget`. Always
/// includes the top level; never the leaves unless the whole tree fits.
fn directory_start(level_bounds: &[usize], level_count: usize, budget: usize) -> usize {
    // Node count of level `l` = level_bounds[l] - level_bounds[l-1] (or - 0).
    let width = |level: usize| -> usize {
        let end = level_bounds[level];
        let start = if level == 0 {
            0
        } else {
            level_bounds[level - 1]
        };
        end - start
    };

    let mut first_level = level_count - 1;
    let mut cached_nodes = width(first_level);
    while first_level > 0 {
        let next = first_level - 1;
        let next_width = width(next);
        if cached_nodes + next_width > budget {
            break;
        }
        cached_nodes += next_width;
        first_level = next;
    }

    if first_level == 0 {
        0
    } else {
        level_bounds[first_level - 1]
    }
}

/// Streaming reader for a 2D `f64` packed spatial index.
///
/// Open one over any [`RangeReader`] — a local [`FileReader`], an in-memory
/// [`SliceReader`], or a custom remote source — and query it by fetching only
/// the byte ranges a traversal needs, instead of loading the whole serialized
/// index. [`open`](Self::open) validates the header and level bounds and
/// prefetches the upper levels of the tree.
pub struct StreamIndex2D<R> {
    core: StreamCore<R>,
}

impl<R: RangeReader> StreamIndex2D<R> {
    /// Open and validate a 2D `f64` index from `reader`.
    ///
    /// Reads and validates the header and level bounds and prefetches the upper
    /// levels of the tree. Returns [`StreamError::Format`] for a corrupt or
    /// wrong-variant index and [`StreamError::Io`] for a read failure.
    pub fn open(reader: R) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open(reader, FORMAT_FLAGS_2D, 2, 8)?,
        })
    }

    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.core.num_items
    }

    /// Whether the index has no items.
    pub fn is_empty(&self) -> bool {
        self.core.num_items == 0
    }

    /// Packed node size of the index.
    pub fn node_size(&self) -> usize {
        self.core.node_size
    }

    /// Total extent of all indexed items, or [`None`] for an empty index.
    ///
    /// Read from the cached root box, so this costs no I/O.
    pub fn extent(&self) -> Option<Box2D> {
        if self.core.num_items == 0 {
            return None;
        }
        // The root is the final node and always sits in the cached directory.
        let root = self.core.num_nodes - 1;
        let bytes = self.core.cached_box_bytes(root)?;
        Some(parse_box2d(bytes))
    }
}

/// Parse one 2D box record (`[min_x, min_y, max_x, max_y]` little-endian f64).
fn parse_box2d(bytes: &[u8]) -> Box2D {
    Box2D::new(
        read_f64_le_unchecked(bytes, 0),
        read_f64_le_unchecked(bytes, 8),
        read_f64_le_unchecked(bytes, 16),
        read_f64_le_unchecked(bytes, 24),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Box2D, Index2DBuilder};
    use std::cell::RefCell;

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
        let short = bytes[..40].to_vec(); // shorter than the 64-byte header
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
        assert!(reads <= 4, "open should issue at most 4 reads, did {reads}");
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
}
