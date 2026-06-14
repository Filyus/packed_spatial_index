//! Streaming reader for the packed spatial index binary format.
//!
//! Where [`Index2D::from_bytes`](crate::Index2D::from_bytes) needs the whole
//! serialized index in memory, the streaming reader answers queries by fetching
//! only the byte ranges a traversal actually touches, over a [`RangeReader`].
//! That backing store can be a local file ([`FileReader`]), an in-memory buffer
//! ([`SliceReader`]), or — by implementing the one-method [`RangeReader`] trait
//! — a remote object served through HTTP range requests.
//!
//! [`open`](StreamIndex2D::open) validates the header and level bounds and
//! prefetches the small upper levels of the tree (the "directory"). A range
//! query then descends the tree level by level, fetching each level's boxes
//! from the directory or in coalesced reads, so it touches only the lower levels
//! and the few leaf runs the query actually overlaps. [`StreamIndex2D`] and
//! [`StreamIndex3D`] expose `search` / `search_into` / `visit`, and — when the
//! index carries a payload section — `search_payloads` / `visit_payloads`, which
//! also stream each matching item's stored blob (the payload is laid out in leaf
//! order, so a query fetches its blobs in coalesced reads).
//!
//! Pointers are validated as they are followed, so the reader is safe to point
//! at untrusted data. Available behind the `stream` feature. See [`RangeReader`]
//! for implementing a remote (e.g. HTTP range) source.

use std::io;

use crate::geometry::{Box2D, Box3D};
use crate::persistence::{
    FORMAT_FLAGS_2D, FORMAT_FLAGS_3D, FORMAT_HEADER_LEN, LoadError, parse_and_validate_header,
    payload_layout, read_f64_le_unchecked, read_u64_at, read_u64_le_unchecked, section_layout,
    validate_level_bounds,
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

/// When streaming a level, node records whose byte gap is no larger than this
/// are fetched in a single read. Coalescing trades a little re-read for far
/// fewer round trips, which dominates on high-latency (e.g. HTTP) sources.
const COALESCE_GAP_BYTES: u64 = 4096;

/// A source of bytes addressable by absolute offset.
///
/// This is the only capability [`StreamIndex2D`] needs from its backing store,
/// so a local file, an in-memory slice, or a remote object behind HTTP range
/// requests can all drive the same streaming queries.
///
/// Implementations must read from an absolute offset **without** disturbing any
/// shared cursor (hence `&self`, not `&mut self`), so one reader can serve
/// concurrent queries safely.
///
/// # A remote (HTTP range) reader
///
/// Implement the single required method to query an index that lives in object
/// storage — no crate dependency on any HTTP client:
///
/// ```ignore
/// use std::io;
/// use packed_spatial_index::RangeReader;
///
/// struct HttpRange {
///     url: String,
///     client: reqwest::blocking::Client,
/// }
///
/// impl RangeReader for HttpRange {
///     fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
///         let end = offset + buf.len() as u64 - 1;
///         let bytes = self
///             .client
///             .get(&self.url)
///             .header("Range", format!("bytes={offset}-{end}"))
///             .send()
///             .and_then(|r| r.error_for_status())
///             .and_then(|r| r.bytes())
///             .map_err(io::Error::other)?;
///         if bytes.len() != buf.len() {
///             return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short range"));
///         }
///         buf.copy_from_slice(&bytes);
///         Ok(())
///     }
///     // `len` defaults to `None`; `open` then skips the length cross-check and
///     // relies on reads past the end failing. Override it (e.g. from a HEAD
///     // request) for a stricter check.
/// }
/// ```
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
    /// Payloads were requested but the index has no payload section.
    NoPayload,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Io(err) => write!(f, "streaming read failed: {err}"),
            StreamError::Format(err) => write!(f, "{err}"),
            StreamError::NoPayload => write!(f, "index has no payload section"),
        }
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamError::Io(err) => Some(err),
            StreamError::Format(err) => Some(err),
            StreamError::NoPayload => None,
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
    reader: R,
    node_size: usize,
    num_items: usize,
    num_nodes: usize,
    level_count: usize,
    /// Exclusive end offset of each level, in node positions (`level_bounds[i]`).
    level_bounds: Vec<usize>,
    /// Box record size in bytes.
    record: usize,
    /// Byte offset of the box section.
    box0: u64,
    /// Byte offset of the index section.
    idx0: u64,
    /// First node position covered by the cached directory.
    dir_node_start: usize,
    /// Cached box bytes for node positions `[dir_node_start, num_nodes)`.
    dir_boxes: Vec<u8>,
    /// Cached index bytes for the same node positions.
    dir_indices: Vec<u8>,
    /// Optional payload section. `None` when the index carries no payload.
    payload: Option<PayloadSection>,
}

/// Byte locations of a streamed index's payload section.
struct PayloadSection {
    /// Byte offset of the `(num_items + 1)` u64 prefix-offset table.
    offsets_start: u64,
    /// Byte offset of the blob region.
    blobs_start: u64,
    /// Total blob bytes (validated against the file length at open).
    blob_total: u64,
}

impl<R> StreamCore<R> {
    /// Whether the index carries a payload section. No I/O, so available for
    /// both sync and async readers.
    fn has_payload(&self) -> bool {
        self.payload.is_some()
    }
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
        let fields = parse_and_validate_header(&header, expected_flags, true)?;

        // 2. Section offsets, derived purely from the validated header counts.
        let layout = section_layout(
            fields.level_count,
            fields.num_nodes,
            dimensions,
            coord_bytes,
        )?;

        // 3. Locate the optional payload section. Its blob total lives in the
        //    last offset-table entry, so read just that one u64 (the table itself
        //    is read on demand, per item, never wholesale).
        let (payload, full_total) = if fields.has_payload {
            let offsets_start = layout.total_len;
            let last_offset_at = offsets_start
                .checked_add(
                    fields
                        .num_items
                        .checked_mul(8)
                        .ok_or(LoadError::IntegerOverflow)?,
                )
                .ok_or(LoadError::IntegerOverflow)?;
            let mut last = [0u8; 8];
            reader.read_exact_at(last_offset_at as u64, &mut last)?;
            let blob_total = u64::from_le_bytes(last);
            let blob_total_usize =
                usize::try_from(blob_total).map_err(|_| LoadError::IntegerOverflow)?;
            let plan = payload_layout(fields.num_items, offsets_start, blob_total_usize)?;
            (
                Some(PayloadSection {
                    offsets_start: plan.offsets_start as u64,
                    blobs_start: plan.blobs_start as u64,
                    blob_total,
                }),
                plan.full_total,
            )
        } else {
            (None, layout.total_len)
        };

        // 4. Cross-check the declared length against the source, when known.
        if let Some(actual) = reader.len()
            && actual != full_total as u64
        {
            return Err(StreamError::Format(LoadError::LengthMismatch {
                expected: full_total,
                actual: usize::try_from(actual).unwrap_or(usize::MAX),
            }));
        }

        // 5. Level bounds (small): read fully, validate, parse to positions.
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

        // 6. Directory: cache the upper levels (a contiguous suffix of the box
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
            payload,
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

    /// Gather `stride`-byte records for `positions` (sorted, ascending) from the
    /// section beginning at `section0`, into `out` (cleared, then filled so that
    /// record `i` lands at `out[i*stride..]`). Records covered by the directory
    /// `cache` are copied; the rest are streamed with adjacent ranges coalesced.
    /// `cache` holds records for node positions `[dir_node_start, num_nodes)`.
    fn gather(
        &self,
        positions: &[usize],
        section0: u64,
        stride: usize,
        cache: &[u8],
        out: &mut Vec<u8>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), StreamError> {
        out.clear();
        out.resize(positions.len() * stride, 0);

        // Copy cached records; collect the streamed ones as (out index, position).
        let mut streamed: Vec<(usize, usize)> = Vec::new();
        for (i, &pos) in positions.iter().enumerate() {
            if pos >= self.dir_node_start {
                let src = (pos - self.dir_node_start) * stride;
                out[i * stride..i * stride + stride].copy_from_slice(&cache[src..src + stride]);
            } else {
                streamed.push((i, pos));
            }
        }

        // Stream the rest, coalescing runs whose byte gap is within the budget.
        let mut j = 0;
        while j < streamed.len() {
            let lo = section0 + (streamed[j].1 * stride) as u64;
            let mut k = j;
            // One past the last position bundled into this read, in node units.
            let mut end_pos = streamed[j].1 + 1;
            while k + 1 < streamed.len() {
                let next_pos = streamed[k + 1].1;
                let gap = (next_pos - end_pos) as u64 * stride as u64;
                if gap > COALESCE_GAP_BYTES {
                    break;
                }
                k += 1;
                end_pos = next_pos + 1;
            }
            let hi = section0 + (end_pos * stride) as u64;
            scratch.clear();
            scratch.resize((hi - lo) as usize, 0);
            self.reader.read_exact_at(lo, scratch)?;
            for &(out_i, pos) in &streamed[j..=k] {
                let within = (section0 + (pos * stride) as u64 - lo) as usize;
                out[out_i * stride..out_i * stride + stride]
                    .copy_from_slice(&scratch[within..within + stride]);
            }
            j = k + 1;
        }
        Ok(())
    }

    /// Descend the tree level by level, calling `leaf` once at the leaf level
    /// with the surviving leaf positions (sorted) and their gathered index bytes
    /// (the insertion ids, in the same order). `overlaps` decides box
    /// intersection; this keeps the traversal dimension- and payload-independent.
    ///
    /// At each level the frontier's boxes are fetched (cached or
    /// coalesced-streamed) and tested; survivors expand to their child groups,
    /// and a parent that fails the test prunes its whole subtree.
    fn traverse<O, L>(&self, overlaps: O, mut leaf: L) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        L: FnMut(&[usize], &[u8]) -> Result<(), StreamError>,
    {
        if self.num_items == 0 {
            return Ok(());
        }

        let mut frontier = vec![self.num_nodes - 1];
        let mut level = self.level_count - 1;
        let mut boxes = Vec::new();
        let mut indices = Vec::new();
        let mut scratch = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();

        loop {
            self.gather(
                &frontier,
                self.box0,
                self.record,
                &self.dir_boxes,
                &mut boxes,
                &mut scratch,
            )?;
            survivors.clear();
            for (i, &pos) in frontier.iter().enumerate() {
                if overlaps(&boxes[i * self.record..(i + 1) * self.record]) {
                    survivors.push(pos);
                }
            }
            if survivors.is_empty() {
                return Ok(());
            }

            self.gather(
                &survivors,
                self.idx0,
                8,
                &self.dir_indices,
                &mut indices,
                &mut scratch,
            )?;

            if level == 0 {
                // `survivors` are sorted leaf positions; `indices` their ids.
                return leaf(&survivors, &indices);
            }

            // Expand survivors to their child groups at the level below.
            let child_level_end = self.level_bounds[level - 1];
            let child_level_start = if level >= 2 {
                self.level_bounds[level - 2]
            } else {
                0
            };
            let mut next = Vec::new();
            for i in 0..survivors.len() {
                let child0 = read_index(&indices, i)?;
                // Validate the pointer against the child level (untrusted source).
                if child0 < child_level_start
                    || child0 >= child_level_end
                    || (child0 - child_level_start) % self.node_size != 0
                {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                let end = (child0 + self.node_size).min(child_level_end);
                next.extend(child0..end);
            }
            // A well-formed tree already yields a sorted, disjoint frontier, but
            // an untrusted index may have in-range yet reordered or aliased child
            // pointers. Sorting and deduping keeps `gather` fed ascending
            // positions (it computes byte gaps by subtraction) and caps the
            // frontier at the level width, so a crafted file cannot trigger an
            // underflow or blow the frontier up level over level. For a valid
            // tree this is a no-op on the result set.
            next.sort_unstable();
            next.dedup();
            frontier = next;
            level -= 1;
        }
    }

    /// Visit the insertion id of every leaf whose box satisfies `overlaps`.
    fn visit_ids<O, F>(&self, overlaps: O, mut visit: F) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize),
    {
        self.traverse(overlaps, |survivors, indices| {
            for i in 0..survivors.len() {
                let id = read_index(indices, i)?;
                if id >= self.num_items {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                visit(id);
            }
            Ok(())
        })
    }

    /// Visit `(insertion id, payload blob)` for every leaf whose box satisfies
    /// `overlaps`, streaming the payload section in leaf order during the leaf
    /// pass so the offset table and blobs are read in coalesced runs.
    fn visit_payloads<O, F>(&self, overlaps: O, mut emit: F) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize, &[u8]),
    {
        let section = self.payload.as_ref().ok_or(StreamError::NoPayload)?;
        let mut off_buf = Vec::new();
        let mut blob_buf = Vec::new();
        self.traverse(overlaps, |survivors, indices| {
            self.gather_payloads(
                section,
                survivors,
                indices,
                &mut off_buf,
                &mut blob_buf,
                &mut emit,
            )
        })
    }

    /// Stream the blobs for `leaf_positions` (sorted leaf ranks) and their
    /// `indices` (insertion ids, same order), coalescing the leaf-ordered offset
    /// table and blob region into runs. Emits `(id, blob)` per leaf.
    fn gather_payloads<F>(
        &self,
        section: &PayloadSection,
        leaf_positions: &[usize],
        indices: &[u8],
        off_buf: &mut Vec<u8>,
        blob_buf: &mut Vec<u8>,
        emit: &mut F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        let mut j = 0;
        while j < leaf_positions.len() {
            // Coalesce leaf positions whose offset-table gap is within budget.
            let mut k = j;
            while k + 1 < leaf_positions.len() {
                let gap = (leaf_positions[k + 1] - leaf_positions[k]) as u64 * 8;
                if gap > COALESCE_GAP_BYTES {
                    break;
                }
                k += 1;
            }
            let lo = leaf_positions[j];
            let hi = leaf_positions[k];

            // Read offset entries [lo ..= hi+1] (one extra for the last blob end).
            let entries = hi + 2 - lo;
            off_buf.clear();
            off_buf.resize(entries * 8, 0);
            self.reader
                .read_exact_at(section.offsets_start + (lo * 8) as u64, off_buf)?;

            let blob_lo = read_u64_le_unchecked(off_buf, 0);
            let blob_hi = read_u64_le_unchecked(off_buf, (hi + 1 - lo) * 8);
            if blob_hi < blob_lo || blob_hi > section.blob_total {
                return Err(StreamError::Format(LoadError::InvalidTree));
            }
            blob_buf.clear();
            blob_buf.resize((blob_hi - blob_lo) as usize, 0);
            if !blob_buf.is_empty() {
                self.reader
                    .read_exact_at(section.blobs_start + blob_lo, blob_buf)?;
            }

            for (offset, &p) in leaf_positions[j..=k].iter().enumerate() {
                let i = j + offset;
                let o0 = read_u64_le_unchecked(off_buf, (p - lo) * 8);
                let o1 = read_u64_le_unchecked(off_buf, (p + 1 - lo) * 8);
                if o1 < o0 || o1 > blob_hi {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                let id = read_index(indices, i)?;
                if id >= self.num_items {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                emit(
                    id,
                    &blob_buf[(o0 - blob_lo) as usize..(o1 - blob_lo) as usize],
                );
            }
            j = k + 1;
        }
        Ok(())
    }
}

/// Read index entry `i` (a little-endian `u64`) from gathered index bytes.
fn read_index(bytes: &[u8], i: usize) -> Result<usize, StreamError> {
    let value = read_u64_le_unchecked(bytes, i * 8);
    usize::try_from(value).map_err(|_| StreamError::Format(LoadError::IntegerOverflow))
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
///
/// Queries are fallible (a backing read can fail; a corrupt index is reported
/// as [`StreamError::Format`]) and otherwise mirror [`Index2D`](crate::Index2D)
/// range search. Results are item insertion indices, in traversal order.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box2D, Index2DBuilder, SliceReader, StreamIndex2D};
///
/// // Serialize an index once...
/// let mut builder = Index2DBuilder::new(2);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
/// let bytes = builder.finish().unwrap().to_bytes();
///
/// // ...then query it through a RangeReader without rebuilding it in memory.
/// let index = StreamIndex2D::open(SliceReader::new(bytes))?;
/// assert_eq!(index.search(Box2D::new(0.0, 0.0, 2.0, 2.0))?, vec![0]);
/// # Ok::<(), packed_spatial_index::StreamError>(())
/// ```
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

    /// Stream the indices of every item whose box intersects `query`, passing
    /// each to `visitor`.
    ///
    /// Fallible: a read from the backing [`RangeReader`] can fail mid-query, and
    /// a corrupt index is reported as [`StreamError::Format`]. Items are yielded
    /// in tree-traversal order, which is not part of the API.
    pub fn visit<F: FnMut(usize)>(&self, query: Box2D, visitor: F) -> Result<(), StreamError> {
        self.core
            .visit_ids(|record| parse_box2d(record).overlaps(query), visitor)
    }

    /// Stream the indices of every item whose box intersects `query`.
    pub fn search(&self, query: Box2D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.search_into(query, &mut out)?;
        Ok(out)
    }

    /// Like [`search`](Self::search), but appends into a reused buffer (cleared
    /// first) to avoid reallocating across queries.
    pub fn search_into(&self, query: Box2D, out: &mut Vec<usize>) -> Result<(), StreamError> {
        out.clear();
        self.visit(query, |index| out.push(index))
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload(&self) -> bool {
        self.core.has_payload()
    }

    /// Visit `(item index, payload blob)` for every item intersecting `query`.
    ///
    /// The payload section is stored in leaf order, so a spatial query fetches
    /// its blobs (and their offset table) in coalesced reads — a handful of
    /// round trips even over a remote source, instead of one per item. The blob
    /// slice is valid only for the duration of each call. Returns
    /// [`StreamError::NoPayload`] if the index has no payload section.
    pub fn visit_payloads<F: FnMut(usize, &[u8])>(
        &self,
        query: Box2D,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads(|record| parse_box2d(record).overlaps(query), visitor)
    }

    /// Collect `(item index, payload blob)` for every item intersecting `query`.
    /// The owning counterpart of [`visit_payloads`](Self::visit_payloads).
    pub fn search_payloads(&self, query: Box2D) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads(query, |id, blob| out.push((id, blob.to_vec())))?;
        Ok(out)
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

/// Streaming reader for a 3D `f64` packed spatial index.
///
/// The 3D counterpart of [`StreamIndex2D`]: it shares the same open, validation,
/// directory prefetch, and coalesced traversal, differing only in the 48-byte
/// box record. See [`StreamIndex2D`] for the streaming model.
pub struct StreamIndex3D<R> {
    core: StreamCore<R>,
}

impl<R: RangeReader> StreamIndex3D<R> {
    /// Open and validate a 3D `f64` index from `reader`.
    pub fn open(reader: R) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open(reader, FORMAT_FLAGS_3D, 3, 8)?,
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
    /// Read from the cached root box, so this costs no I/O.
    pub fn extent(&self) -> Option<Box3D> {
        if self.core.num_items == 0 {
            return None;
        }
        let root = self.core.num_nodes - 1;
        let bytes = self.core.cached_box_bytes(root)?;
        Some(parse_box3d(bytes))
    }

    /// Stream the indices of every item whose box intersects `query`, passing
    /// each to `visitor`. Fallible; see [`StreamIndex2D::visit`].
    pub fn visit<F: FnMut(usize)>(&self, query: Box3D, visitor: F) -> Result<(), StreamError> {
        self.core
            .visit_ids(|record| parse_box3d(record).overlaps(query), visitor)
    }

    /// Stream the indices of every item whose box intersects `query`.
    pub fn search(&self, query: Box3D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.search_into(query, &mut out)?;
        Ok(out)
    }

    /// Like [`search`](Self::search), but appends into a reused buffer.
    pub fn search_into(&self, query: Box3D, out: &mut Vec<usize>) -> Result<(), StreamError> {
        out.clear();
        self.visit(query, |index| out.push(index))
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload(&self) -> bool {
        self.core.has_payload()
    }

    /// Visit `(item index, payload blob)` for every item intersecting `query`.
    /// See [`StreamIndex2D::visit_payloads`].
    pub fn visit_payloads<F: FnMut(usize, &[u8])>(
        &self,
        query: Box3D,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads(|record| parse_box3d(record).overlaps(query), visitor)
    }

    /// Collect `(item index, payload blob)` for every item intersecting `query`.
    pub fn search_payloads(&self, query: Box3D) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads(query, |id, blob| out.push((id, blob.to_vec())))?;
        Ok(out)
    }
}

/// Parse one 3D box record (`[min_x, min_y, min_z, max_x, max_y, max_z]` LE f64).
fn parse_box3d(bytes: &[u8]) -> Box3D {
    Box3D::new(
        read_f64_le_unchecked(bytes, 0),
        read_f64_le_unchecked(bytes, 8),
        read_f64_le_unchecked(bytes, 16),
        read_f64_le_unchecked(bytes, 24),
        read_f64_le_unchecked(bytes, 32),
        read_f64_le_unchecked(bytes, 40),
    )
}

// ---- Async streaming (behind the `async` feature) ----
//
// Mirror of the synchronous traversal for sources whose reads are async (browser
// / edge worker over HTTP range or object storage). The descent logic is the
// same — only the reads are awaited; the overlap test and the result sink stay
// synchronous closures so no async closures are needed. (The sync and async
// paths are kept in lockstep by an equivalence test; a future sans-io refactor
// could share one core.)

/// Async counterpart of [`RangeReader`]: read a byte range, returning a future.
///
/// Implement this to query an index that lives behind async I/O — an HTTP range
/// request from WebAssembly, an object-storage `get(range)` in an edge worker.
/// The returned futures need not be `Send` (edge/browser executors are
/// single-threaded). See [`RangeReader`] for the sync analogue and an HTTP
/// implementation sketch.
#[cfg(feature = "async")]
#[allow(async_fn_in_trait, clippy::len_without_is_empty)]
pub trait AsyncRangeReader {
    /// Read exactly `buf.len()` bytes starting at `offset`.
    async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Total length in bytes, if known.
    fn len(&self) -> Option<u64> {
        None
    }
}

/// What a traversal collects at the leaves.
#[cfg(feature = "async")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    Ids,
    Payloads,
}

#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamCore<R> {
    async fn open_async(
        reader: R,
        expected_flags: u64,
        dimensions: usize,
        coord_bytes: usize,
    ) -> Result<Self, StreamError> {
        let mut header = [0u8; FORMAT_HEADER_LEN];
        reader.read_exact_at(0, &mut header).await?;
        let fields = parse_and_validate_header(&header, expected_flags, true)?;
        let layout = section_layout(
            fields.level_count,
            fields.num_nodes,
            dimensions,
            coord_bytes,
        )?;

        let (payload, full_total) = if fields.has_payload {
            let offsets_start = layout.total_len;
            let last_offset_at = offsets_start
                .checked_add(
                    fields
                        .num_items
                        .checked_mul(8)
                        .ok_or(LoadError::IntegerOverflow)?,
                )
                .ok_or(LoadError::IntegerOverflow)?;
            let mut last = [0u8; 8];
            reader
                .read_exact_at(last_offset_at as u64, &mut last)
                .await?;
            let blob_total = u64::from_le_bytes(last);
            let blob_total_usize =
                usize::try_from(blob_total).map_err(|_| LoadError::IntegerOverflow)?;
            let plan = payload_layout(fields.num_items, offsets_start, blob_total_usize)?;
            (
                Some(PayloadSection {
                    offsets_start: plan.offsets_start as u64,
                    blobs_start: plan.blobs_start as u64,
                    blob_total,
                }),
                plan.full_total,
            )
        } else {
            (None, layout.total_len)
        };

        if let Some(actual) = reader.len()
            && actual != full_total as u64
        {
            return Err(StreamError::Format(LoadError::LengthMismatch {
                expected: full_total,
                actual: usize::try_from(actual).unwrap_or(usize::MAX),
            }));
        }

        let level_bounds_len = fields.level_count * 8;
        let mut level_bounds_bytes = vec![0u8; level_bounds_len];
        reader
            .read_exact_at(layout.level_bounds_start as u64, &mut level_bounds_bytes)
            .await?;
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

        let dir_node_start =
            directory_start(&level_bounds, fields.level_count, DIRECTORY_NODE_BUDGET);
        let cached_nodes = fields.num_nodes - dir_node_start;

        let mut dir_boxes = vec![0u8; cached_nodes * layout.record];
        if !dir_boxes.is_empty() {
            let offset = layout.box0 + (dir_node_start * layout.record);
            reader.read_exact_at(offset as u64, &mut dir_boxes).await?;
        }
        let mut dir_indices = vec![0u8; cached_nodes * 8];
        if !dir_indices.is_empty() {
            let offset = layout.idx0 + (dir_node_start * 8);
            reader
                .read_exact_at(offset as u64, &mut dir_indices)
                .await?;
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
            payload,
        })
    }

    /// Async mirror of [`gather`](StreamCore::gather).
    async fn gather_async(
        &self,
        positions: &[usize],
        section0: u64,
        stride: usize,
        cache: &[u8],
        out: &mut Vec<u8>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), StreamError> {
        out.clear();
        out.resize(positions.len() * stride, 0);
        let mut streamed: Vec<(usize, usize)> = Vec::new();
        for (i, &pos) in positions.iter().enumerate() {
            if pos >= self.dir_node_start {
                let src = (pos - self.dir_node_start) * stride;
                out[i * stride..i * stride + stride].copy_from_slice(&cache[src..src + stride]);
            } else {
                streamed.push((i, pos));
            }
        }
        let mut j = 0;
        while j < streamed.len() {
            let lo = section0 + (streamed[j].1 * stride) as u64;
            let mut k = j;
            let mut end_pos = streamed[j].1 + 1;
            while k + 1 < streamed.len() {
                let next_pos = streamed[k + 1].1;
                let gap = (next_pos - end_pos) as u64 * stride as u64;
                if gap > COALESCE_GAP_BYTES {
                    break;
                }
                k += 1;
                end_pos = next_pos + 1;
            }
            let hi = section0 + (end_pos * stride) as u64;
            scratch.clear();
            scratch.resize((hi - lo) as usize, 0);
            self.reader.read_exact_at(lo, scratch).await?;
            for &(out_i, pos) in &streamed[j..=k] {
                let within = (section0 + (pos * stride) as u64 - lo) as usize;
                out[out_i * stride..out_i * stride + stride]
                    .copy_from_slice(&scratch[within..within + stride]);
            }
            j = k + 1;
        }
        Ok(())
    }

    /// Async mirror of [`gather_payloads`](StreamCore::gather_payloads).
    async fn gather_payloads_async<F>(
        &self,
        section: &PayloadSection,
        leaf_positions: &[usize],
        indices: &[u8],
        off_buf: &mut Vec<u8>,
        blob_buf: &mut Vec<u8>,
        sink: &mut F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(usize, &[u8]),
    {
        let mut j = 0;
        while j < leaf_positions.len() {
            let mut k = j;
            while k + 1 < leaf_positions.len() {
                let gap = (leaf_positions[k + 1] - leaf_positions[k]) as u64 * 8;
                if gap > COALESCE_GAP_BYTES {
                    break;
                }
                k += 1;
            }
            let lo = leaf_positions[j];
            let hi = leaf_positions[k];
            let entries = hi + 2 - lo;
            off_buf.clear();
            off_buf.resize(entries * 8, 0);
            self.reader
                .read_exact_at(section.offsets_start + (lo * 8) as u64, off_buf)
                .await?;
            let blob_lo = read_u64_le_unchecked(off_buf, 0);
            let blob_hi = read_u64_le_unchecked(off_buf, (hi + 1 - lo) * 8);
            if blob_hi < blob_lo || blob_hi > section.blob_total {
                return Err(StreamError::Format(LoadError::InvalidTree));
            }
            blob_buf.clear();
            blob_buf.resize((blob_hi - blob_lo) as usize, 0);
            if !blob_buf.is_empty() {
                self.reader
                    .read_exact_at(section.blobs_start + blob_lo, blob_buf)
                    .await?;
            }
            for (offset, &p) in leaf_positions[j..=k].iter().enumerate() {
                let i = j + offset;
                let o0 = read_u64_le_unchecked(off_buf, (p - lo) * 8);
                let o1 = read_u64_le_unchecked(off_buf, (p + 1 - lo) * 8);
                if o1 < o0 || o1 > blob_hi {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                let id = read_index(indices, i)?;
                if id >= self.num_items {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                sink(
                    id,
                    &blob_buf[(o0 - blob_lo) as usize..(o1 - blob_lo) as usize],
                );
            }
            j = k + 1;
        }
        Ok(())
    }

    /// Async mirror of the synchronous traversal, parameterized by `want` (ids or
    /// id+payload). `overlaps` and `sink` are synchronous; only reads are awaited.
    async fn traverse_async<O, F>(
        &self,
        overlaps: O,
        want: Want,
        mut sink: F,
    ) -> Result<(), StreamError>
    where
        O: Fn(&[u8]) -> bool,
        F: FnMut(usize, &[u8]),
    {
        let section = if want == Want::Payloads {
            Some(self.payload.as_ref().ok_or(StreamError::NoPayload)?)
        } else {
            None
        };
        if self.num_items == 0 {
            return Ok(());
        }

        let mut frontier = vec![self.num_nodes - 1];
        let mut level = self.level_count - 1;
        let mut boxes = Vec::new();
        let mut indices = Vec::new();
        let mut scratch = Vec::new();
        let mut off_buf = Vec::new();
        let mut blob_buf = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();

        loop {
            self.gather_async(
                &frontier,
                self.box0,
                self.record,
                &self.dir_boxes,
                &mut boxes,
                &mut scratch,
            )
            .await?;
            survivors.clear();
            for (i, &pos) in frontier.iter().enumerate() {
                if overlaps(&boxes[i * self.record..(i + 1) * self.record]) {
                    survivors.push(pos);
                }
            }
            if survivors.is_empty() {
                return Ok(());
            }

            self.gather_async(
                &survivors,
                self.idx0,
                8,
                &self.dir_indices,
                &mut indices,
                &mut scratch,
            )
            .await?;

            if level == 0 {
                match section {
                    Some(section) => {
                        self.gather_payloads_async(
                            section,
                            &survivors,
                            &indices,
                            &mut off_buf,
                            &mut blob_buf,
                            &mut sink,
                        )
                        .await?;
                    }
                    None => {
                        for i in 0..survivors.len() {
                            let id = read_index(&indices, i)?;
                            if id >= self.num_items {
                                return Err(StreamError::Format(LoadError::InvalidTree));
                            }
                            sink(id, &[]);
                        }
                    }
                }
                return Ok(());
            }

            let child_level_end = self.level_bounds[level - 1];
            let child_level_start = if level >= 2 {
                self.level_bounds[level - 2]
            } else {
                0
            };
            let mut next = Vec::new();
            for i in 0..survivors.len() {
                let child0 = read_index(&indices, i)?;
                if child0 < child_level_start
                    || child0 >= child_level_end
                    || (child0 - child_level_start) % self.node_size != 0
                {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                let end = (child0 + self.node_size).min(child_level_end);
                next.extend(child0..end);
            }
            next.sort_unstable();
            next.dedup();
            frontier = next;
            level -= 1;
        }
    }
}

/// Streaming reader for a 2D `f64` index over async I/O. Mirrors
/// [`StreamIndex2D`]; use it when reads return futures (e.g. browser / edge
/// worker). Behind the `async` feature.
#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamIndex2D<R> {
    /// Open and validate a 2D `f64` index from an async `reader`.
    pub async fn open_async(reader: R) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open_async(reader, FORMAT_FLAGS_2D, 2, 8).await?,
        })
    }

    /// Stream the indices of every item whose box intersects `query`.
    pub async fn search_async(&self, query: Box2D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box2d(r).overlaps(query),
                Want::Ids,
                |id, _| out.push(id),
            )
            .await?;
        Ok(out)
    }

    /// Stream `(item index, payload blob)` for every item intersecting `query`.
    pub async fn search_payloads_async(
        &self,
        query: Box2D,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box2d(r).overlaps(query),
                Want::Payloads,
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload_async(&self) -> bool {
        self.core.has_payload()
    }
}

/// Streaming reader for a 3D `f64` index over async I/O. See [`StreamIndex2D`]'s
/// async methods. Behind the `async` feature.
#[cfg(feature = "async")]
impl<R: AsyncRangeReader> StreamIndex3D<R> {
    /// Open and validate a 3D `f64` index from an async `reader`.
    pub async fn open_async(reader: R) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open_async(reader, FORMAT_FLAGS_3D, 3, 8).await?,
        })
    }

    /// Stream the indices of every item whose box intersects `query`.
    pub async fn search_async(&self, query: Box3D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box3d(r).overlaps(query),
                Want::Ids,
                |id, _| out.push(id),
            )
            .await?;
        Ok(out)
    }

    /// Stream `(item index, payload blob)` for every item intersecting `query`.
    pub async fn search_payloads_async(
        &self,
        query: Box3D,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.core
            .traverse_async(
                |r| parse_box3d(r).overlaps(query),
                Want::Payloads,
                |id, blob| out.push((id, blob.to_vec())),
            )
            .await?;
        Ok(out)
    }
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

    /// Byte offset of the index section, recomputed from the validated header.
    fn indices_offset(stream: &StreamIndex2D<SliceReader<Vec<u8>>>) -> usize {
        FORMAT_HEADER_LEN + 8 * stream.core.level_count + stream.core.record * stream.core.num_nodes
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
        match StreamIndex2D::open(SliceReader::new(bytes)) {
            Err(StreamError::Format(LoadError::LengthMismatch { .. })) => {}
            Ok(_) => panic!("expected LengthMismatch, got a valid index"),
            Err(other) => panic!("expected LengthMismatch, got {other:?}"),
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
}
