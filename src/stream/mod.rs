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

use crate::geometry::{Box2D, Box3D, Overlaps2D, Overlaps3D};
use crate::persistence::{read_f32_le_unchecked, read_f64_le_unchecked};

#[cfg(feature = "async")]
mod async_io;
mod core;
mod directory;
mod error;
mod limits;
mod payload;
mod planner;
mod readers;

#[cfg(feature = "async")]
pub use self::async_io::AsyncRangeReader;
pub(crate) use self::core::{StreamCore, read_index};
pub use self::directory::StreamDirectory;
pub use self::error::StreamError;
pub use self::limits::StreamLimits;
#[cfg(any(unix, windows))]
pub use self::readers::FileReader;
#[cfg(test)]
use self::readers::unexpected_eof;
pub use self::readers::{RangeReader, SliceReader};

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

impl<R> StreamIndex2D<R> {
    /// Split off the reader, keeping the reusable [`StreamDirectory`]. No I/O.
    pub fn into_directory(self) -> (StreamDirectory, R) {
        let (parts, reader) = self.core.into_parts();
        (StreamDirectory { parts }, reader)
    }

    /// Rebuild a 2D `f64` index from a cached directory and a fresh reader. No
    /// I/O: the directory reads were paid when it was first opened.
    pub fn from_directory(dir: &StreamDirectory, reader: R) -> Result<Self, StreamError> {
        Self::from_directory_with_limits(dir, reader, StreamLimits::default())
    }

    /// [`from_directory`](Self::from_directory) with per-query [`StreamLimits`].
    pub fn from_directory_with_limits(
        dir: &StreamDirectory,
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: dir.reattach(reader, limits, 2 * 2 * 8)?,
        })
    }
}

impl<R: RangeReader> StreamIndex2D<R> {
    /// Open and validate a 2D `f64` index from `reader`.
    ///
    /// Reads and validates the header and level bounds and prefetches the upper
    /// levels of the tree. Returns [`StreamError::Format`] for a corrupt or
    /// wrong-variant index and [`StreamError::Io`] for a read failure.
    pub fn open(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits(reader, StreamLimits::default())
    }

    /// Open with per-query cost [`StreamLimits`]. Every query then aborts with
    /// [`StreamError::LimitExceeded`] if it would exceed a limit — use this to
    /// bound a broad query's reads / bytes / results to your environment (e.g. a
    /// worker's subrequest and memory budgets).
    pub fn open_with_limits(reader: R, limits: StreamLimits) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open(reader, 2, 8, limits)?,
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

    /// Stream the indices of every item whose box overlaps the region `query` —
    /// any [`Overlaps2D`] shape (a polygon, triangle, …), not just a box.
    ///
    /// Subtrees whose node box falls outside `query` are pruned during the
    /// streamed descent, so a region query touches (and fetches) only the leaves
    /// it overlaps — fewer reads than its bounding box would take.
    pub fn visit_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize),
    {
        self.core
            .visit_ids(|record| query.overlaps_box(parse_box2d(record)), visitor)
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub fn search_region<Q: Overlaps2D>(&self, query: &Q) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region(query, |index| out.push(index))?;
        Ok(out)
    }

    /// Visit `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`. Like [`visit_payloads`](Self::visit_payloads) but for a
    /// custom [`Overlaps2D`] shape; node-box pruning fetches only the leaves the
    /// region touches.
    pub fn visit_payloads_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize, &[u8]),
    {
        self.core
            .visit_payloads(|record| query.overlaps_box(parse_box2d(record)), visitor)
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`. The owning counterpart of
    /// [`visit_payloads_region`](Self::visit_payloads_region).
    pub fn search_payloads_region<Q: Overlaps2D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region(query, |id, blob| out.push((id, blob.to_vec())))?;
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

impl<R> StreamIndex3D<R> {
    /// Split off the reader, keeping the reusable [`StreamDirectory`]. No I/O.
    pub fn into_directory(self) -> (StreamDirectory, R) {
        let (parts, reader) = self.core.into_parts();
        (StreamDirectory { parts }, reader)
    }

    /// Rebuild a 3D `f64` index from a cached directory and a fresh reader. No I/O.
    pub fn from_directory(dir: &StreamDirectory, reader: R) -> Result<Self, StreamError> {
        Self::from_directory_with_limits(dir, reader, StreamLimits::default())
    }

    /// [`from_directory`](Self::from_directory) with per-query [`StreamLimits`].
    pub fn from_directory_with_limits(
        dir: &StreamDirectory,
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: dir.reattach(reader, limits, 3 * 2 * 8)?,
        })
    }
}

impl<R: RangeReader> StreamIndex3D<R> {
    /// Open and validate a 3D `f64` index from `reader`.
    pub fn open(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits(reader, StreamLimits::default())
    }

    /// Open with per-query cost [`StreamLimits`]. See
    /// [`StreamIndex2D::open_with_limits`].
    pub fn open_with_limits(reader: R, limits: StreamLimits) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open(reader, 3, 8, limits)?,
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

    /// Stream the indices of every item whose box overlaps the region `query` —
    /// any [`Overlaps3D`] shape (e.g. a [`Frustum3D`](crate::Frustum3D)), not just
    /// a box. Subtrees outside `query` are pruned during the streamed descent.
    pub fn visit_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize),
    {
        self.core
            .visit_ids(|record| query.overlaps_box(parse_box3d(record)), visitor)
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub fn search_region<Q: Overlaps3D>(&self, query: &Q) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region(query, |index| out.push(index))?;
        Ok(out)
    }

    /// Visit `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`; node-box pruning fetches only the leaves it touches.
    pub fn visit_payloads_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize, &[u8]),
    {
        self.core
            .visit_payloads(|record| query.overlaps_box(parse_box3d(record)), visitor)
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`. The owning counterpart of
    /// [`visit_payloads_region`](Self::visit_payloads_region).
    pub fn search_payloads_region<Q: Overlaps3D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region(query, |id, blob| out.push((id, blob.to_vec())))?;
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

/// Parse one 2D `f32` box record (16 bytes), widening to `f64`. The stored f32
/// bounds were rounded outward, so the widened box is a superset and search stays
/// conservative (no misses, a few near-boundary false positives).
fn parse_box2d_f32(bytes: &[u8]) -> Box2D {
    Box2D::new(
        read_f32_le_unchecked(bytes, 0) as f64,
        read_f32_le_unchecked(bytes, 4) as f64,
        read_f32_le_unchecked(bytes, 8) as f64,
        read_f32_le_unchecked(bytes, 12) as f64,
    )
}

/// Parse one 3D `f32` box record (24 bytes), widening to `f64` (see
/// [`parse_box2d_f32`]).
fn parse_box3d_f32(bytes: &[u8]) -> Box3D {
    Box3D::new(
        read_f32_le_unchecked(bytes, 0) as f64,
        read_f32_le_unchecked(bytes, 4) as f64,
        read_f32_le_unchecked(bytes, 8) as f64,
        read_f32_le_unchecked(bytes, 12) as f64,
        read_f32_le_unchecked(bytes, 16) as f64,
        read_f32_le_unchecked(bytes, 20) as f64,
    )
}

/// Streaming reader for a compact `f32` 2D index — the bytes of
/// [`SimdIndex2DF32`](crate::SimdIndex2DF32) / [`Index2DF32`](crate::Index2DF32).
/// Half the box bytes on the wire of [`StreamIndex2D`]; results are a
/// conservative superset (the stored f32 boxes are rounded outward). Behind the
/// `stream` feature.
pub struct StreamIndex2DF32<R> {
    core: StreamCore<R>,
}

impl<R> StreamIndex2DF32<R> {
    /// Split off the reader, keeping the reusable [`StreamDirectory`]. No I/O.
    pub fn into_directory(self) -> (StreamDirectory, R) {
        let (parts, reader) = self.core.into_parts();
        (StreamDirectory { parts }, reader)
    }

    /// Rebuild a 2D `f32` index from a cached directory and a fresh reader. No I/O.
    pub fn from_directory(dir: &StreamDirectory, reader: R) -> Result<Self, StreamError> {
        Self::from_directory_with_limits(dir, reader, StreamLimits::default())
    }

    /// [`from_directory`](Self::from_directory) with per-query [`StreamLimits`].
    pub fn from_directory_with_limits(
        dir: &StreamDirectory,
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: dir.reattach(reader, limits, 2 * 2 * 4)?,
        })
    }
}

impl<R: RangeReader> StreamIndex2DF32<R> {
    /// Open and validate a 2D `f32` index from `reader`.
    pub fn open(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits(reader, StreamLimits::default())
    }

    /// Open with per-query cost [`StreamLimits`]. See
    /// [`StreamIndex2D::open_with_limits`].
    pub fn open_with_limits(reader: R, limits: StreamLimits) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open(reader, 2, 4, limits)?,
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

    /// Total extent of all indexed items (widened f32 root box), or `None` when
    /// empty. Costs no I/O.
    pub fn extent(&self) -> Option<Box2D> {
        if self.core.num_items == 0 {
            return None;
        }
        let root = self.core.num_nodes - 1;
        Some(parse_box2d_f32(self.core.cached_box_bytes(root)?))
    }

    /// Stream the indices of every item whose (rounded) box intersects `query`.
    pub fn visit<F: FnMut(usize)>(&self, query: Box2D, visitor: F) -> Result<(), StreamError> {
        self.core
            .visit_ids(|r| parse_box2d_f32(r).overlaps(query), visitor)
    }

    /// Stream the indices of every item whose (rounded) box intersects `query`.
    pub fn search(&self, query: Box2D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.search_into(query, &mut out)?;
        Ok(out)
    }

    /// Like [`search`](Self::search), into a reused buffer (cleared first).
    pub fn search_into(&self, query: Box2D, out: &mut Vec<usize>) -> Result<(), StreamError> {
        out.clear();
        self.visit(query, |index| out.push(index))
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload(&self) -> bool {
        self.core.has_payload()
    }

    /// Visit `(item index, payload blob)` for every item intersecting `query`.
    pub fn visit_payloads<F: FnMut(usize, &[u8])>(
        &self,
        query: Box2D,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads(|r| parse_box2d_f32(r).overlaps(query), visitor)
    }

    /// Collect `(item index, payload blob)` for every item intersecting `query`.
    pub fn search_payloads(&self, query: Box2D) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads(query, |id, blob| out.push((id, blob.to_vec())))?;
        Ok(out)
    }

    /// Stream the indices of every item whose (rounded) box overlaps the region
    /// `query` — any [`Overlaps2D`] shape. Subtrees outside `query` are pruned.
    pub fn visit_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize),
    {
        self.core.visit_ids(
            |record| query.overlaps_box(parse_box2d_f32(record)),
            visitor,
        )
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub fn search_region<Q: Overlaps2D>(&self, query: &Q) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region(query, |index| out.push(index))?;
        Ok(out)
    }

    /// Visit `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`; node-box pruning fetches only the leaves it touches.
    pub fn visit_payloads_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps2D,
        F: FnMut(usize, &[u8]),
    {
        self.core.visit_payloads(
            |record| query.overlaps_box(parse_box2d_f32(record)),
            visitor,
        )
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub fn search_payloads_region<Q: Overlaps2D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region(query, |id, blob| out.push((id, blob.to_vec())))?;
        Ok(out)
    }
}

/// Streaming reader for a compact `f32` 3D index. The 3D counterpart of
/// [`StreamIndex2DF32`] (24-byte box record). Behind the `stream` feature.
pub struct StreamIndex3DF32<R> {
    core: StreamCore<R>,
}

impl<R> StreamIndex3DF32<R> {
    /// Split off the reader, keeping the reusable [`StreamDirectory`]. No I/O.
    pub fn into_directory(self) -> (StreamDirectory, R) {
        let (parts, reader) = self.core.into_parts();
        (StreamDirectory { parts }, reader)
    }

    /// Rebuild a 3D `f32` index from a cached directory and a fresh reader. No I/O.
    pub fn from_directory(dir: &StreamDirectory, reader: R) -> Result<Self, StreamError> {
        Self::from_directory_with_limits(dir, reader, StreamLimits::default())
    }

    /// [`from_directory`](Self::from_directory) with per-query [`StreamLimits`].
    pub fn from_directory_with_limits(
        dir: &StreamDirectory,
        reader: R,
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        Ok(Self {
            core: dir.reattach(reader, limits, 3 * 2 * 4)?,
        })
    }
}

impl<R: RangeReader> StreamIndex3DF32<R> {
    /// Open and validate a 3D `f32` index from `reader`.
    pub fn open(reader: R) -> Result<Self, StreamError> {
        Self::open_with_limits(reader, StreamLimits::default())
    }

    /// Open with per-query cost [`StreamLimits`].
    pub fn open_with_limits(reader: R, limits: StreamLimits) -> Result<Self, StreamError> {
        Ok(Self {
            core: StreamCore::open(reader, 3, 4, limits)?,
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

    /// Total extent (widened f32 root box), or `None` when empty. Costs no I/O.
    pub fn extent(&self) -> Option<Box3D> {
        if self.core.num_items == 0 {
            return None;
        }
        let root = self.core.num_nodes - 1;
        Some(parse_box3d_f32(self.core.cached_box_bytes(root)?))
    }

    /// Stream the indices of every item whose (rounded) box intersects `query`.
    pub fn visit<F: FnMut(usize)>(&self, query: Box3D, visitor: F) -> Result<(), StreamError> {
        self.core
            .visit_ids(|r| parse_box3d_f32(r).overlaps(query), visitor)
    }

    /// Stream the indices of every item whose (rounded) box intersects `query`.
    pub fn search(&self, query: Box3D) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.search_into(query, &mut out)?;
        Ok(out)
    }

    /// Like [`search`](Self::search), into a reused buffer (cleared first).
    pub fn search_into(&self, query: Box3D, out: &mut Vec<usize>) -> Result<(), StreamError> {
        out.clear();
        self.visit(query, |index| out.push(index))
    }

    /// Whether this index was written with a payload section.
    pub fn has_payload(&self) -> bool {
        self.core.has_payload()
    }

    /// Visit `(item index, payload blob)` for every item intersecting `query`.
    pub fn visit_payloads<F: FnMut(usize, &[u8])>(
        &self,
        query: Box3D,
        visitor: F,
    ) -> Result<(), StreamError> {
        self.core
            .visit_payloads(|r| parse_box3d_f32(r).overlaps(query), visitor)
    }

    /// Collect `(item index, payload blob)` for every item intersecting `query`.
    pub fn search_payloads(&self, query: Box3D) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads(query, |id, blob| out.push((id, blob.to_vec())))?;
        Ok(out)
    }

    /// Stream the indices of every item whose (rounded) box overlaps the region
    /// `query` — any [`Overlaps3D`] shape. Subtrees outside `query` are pruned.
    pub fn visit_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize),
    {
        self.core.visit_ids(
            |record| query.overlaps_box(parse_box3d_f32(record)),
            visitor,
        )
    }

    /// Collect the indices of every item whose box overlaps the region `query`.
    pub fn search_region<Q: Overlaps3D>(&self, query: &Q) -> Result<Vec<usize>, StreamError> {
        let mut out = Vec::new();
        self.visit_region(query, |index| out.push(index))?;
        Ok(out)
    }

    /// Visit `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`; node-box pruning fetches only the leaves it touches.
    pub fn visit_payloads_region<Q, F>(&self, query: &Q, visitor: F) -> Result<(), StreamError>
    where
        Q: Overlaps3D,
        F: FnMut(usize, &[u8]),
    {
        self.core.visit_payloads(
            |record| query.overlaps_box(parse_box3d_f32(record)),
            visitor,
        )
    }

    /// Collect `(item index, payload blob)` for every item whose box overlaps the
    /// region `query`.
    pub fn search_payloads_region<Q: Overlaps3D>(
        &self,
        query: &Q,
    ) -> Result<Vec<(usize, Vec<u8>)>, StreamError> {
        let mut out = Vec::new();
        self.visit_payloads_region(query, |id, blob| out.push((id, blob.to_vec())))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
