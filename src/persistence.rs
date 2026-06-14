use std::{error::Error, fmt};

use crate::geometry::{Box2D, Box3D};

/// Error returned when loading an index from bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// The buffer does not start with the expected `PSINDEX\0` magic marker.
    BadMagic,
    /// The buffer uses a newer or otherwise unsupported format version.
    UnsupportedVersion,
    /// The buffer ended before a complete header or section could be read.
    Truncated,
    /// The buffer length does not match the length declared by the header.
    LengthMismatch {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// The stored node size is outside the supported range.
    InvalidNodeSize {
        /// Stored node size.
        node_size: usize,
    },
    /// A stored integer does not fit this platform or a byte-size calculation overflowed.
    IntegerOverflow,
    /// The level bounds or child pointers do not describe a valid packed tree.
    InvalidTree,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::BadMagic => write!(f, "buffer is not a packed_spatial_index index"),
            LoadError::UnsupportedVersion => write!(f, "unsupported packed_spatial_index format"),
            LoadError::Truncated => write!(f, "buffer is truncated"),
            LoadError::LengthMismatch { expected, actual } => write!(
                f,
                "buffer length mismatch (expected {expected} bytes, got {actual})"
            ),
            LoadError::InvalidNodeSize { node_size } => {
                write!(f, "invalid node size in buffer ({node_size})")
            }
            LoadError::IntegerOverflow => write!(f, "buffer integer value is too large"),
            LoadError::InvalidTree => write!(f, "buffer does not contain a valid packed tree"),
        }
    }
}

impl Error for LoadError {}

/// Error returned when serializing an index together with item payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadError {
    /// The number of payloads does not equal the index's item count.
    CountMismatch {
        /// Expected payload count (the index's `num_items`).
        expected: usize,
        /// Number of payloads supplied.
        got: usize,
    },
    /// The combined payload size overflows the serialized-length calculation.
    TooLarge,
    /// A fixed-width record's length does not equal the declared stride.
    RecordSizeMismatch {
        /// The declared fixed record stride.
        stride: usize,
        /// The length of the offending record.
        got: usize,
    },
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayloadError::CountMismatch { expected, got } => write!(
                f,
                "payload count {got} does not match item count {expected}"
            ),
            PayloadError::TooLarge => write!(f, "combined payload size is too large to serialize"),
            PayloadError::RecordSizeMismatch { stride, got } => write!(
                f,
                "fixed-width record length {got} does not match stride {stride}"
            ),
        }
    }
}

impl Error for PayloadError {}

// Packed Spatial Index file signature, at the start of the superblock; the
// format version follows it as a little-endian u64.
const FORMAT_MAGIC: &[u8; 8] = b"PSINDEX\0";
/// Validated slices of the optional trailing payload section. Borrowed by the
/// zero-copy views to serve `payload(id)`.
pub(crate) struct ParsedPayload<'a> {
    /// `(num_items + 1)` little-endian `u64` prefix offsets into `blobs`. Empty
    /// for a fixed-width payload (`stride != 0`), which needs no table.
    pub(crate) offsets: &'a [u8],
    /// Concatenated per-item payload bytes.
    pub(crate) blobs: &'a [u8],
    /// Fixed record stride in bytes, or `0` for a variable-width payload (use
    /// `offsets`). When non-zero every blob is exactly `stride` bytes, so the
    /// blob at leaf rank `r` is `blobs[r * stride ..][.. stride]`.
    pub(crate) stride: usize,
}

/// Slice the payload at leaf rank `r`: by arithmetic for a fixed-width payload,
/// or out of the leaf-ordered offset table for a variable-width one.
#[inline]
pub(crate) fn payload_slice<'a>(payload: &ParsedPayload<'a>, r: usize) -> &'a [u8] {
    if payload.stride != 0 {
        let start = r * payload.stride;
        &payload.blobs[start..start + payload.stride]
    } else {
        let start = read_u64_le_unchecked(payload.offsets, r * 8) as usize;
        let end = read_u64_le_unchecked(payload.offsets, (r + 1) * 8) as usize;
        &payload.blobs[start..end]
    }
}

/// Build the `insertion id -> leaf rank` map by inverting the leaf entries of
/// `indices` (which map leaf rank -> insertion id). The payload section is
/// leaf-ordered, so a view needs this to serve random `payload(id)` lookups.
pub(crate) fn build_id_to_leaf(indices: &[u8], num_items: usize) -> Vec<u32> {
    let mut id_to_leaf = vec![0u32; num_items];
    for r in 0..num_items {
        let id = read_u64_le_unchecked(indices, r * 8) as usize;
        id_to_leaf[id] = r as u32;
    }
    id_to_leaf
}

pub(crate) fn expected_tree_shape(
    num_items: usize,
    node_size: usize,
) -> Result<(usize, usize), LoadError> {
    let mut num_nodes = num_items;
    let mut levels = 1usize;
    let mut n = num_items;
    if num_items > 0 {
        loop {
            n = n.div_ceil(node_size);
            num_nodes = num_nodes.checked_add(n).ok_or(LoadError::IntegerOverflow)?;
            levels = levels.checked_add(1).ok_or(LoadError::IntegerOverflow)?;
            if n == 1 {
                break;
            }
        }
    }
    Ok((num_nodes, levels))
}

pub(crate) struct ByteWriter<'a> {
    bytes: &'a mut Vec<u8>,
    len: usize,
}

impl<'a> ByteWriter<'a> {
    /// Write into a caller-owned buffer. The buffer is cleared and reserved to at least
    /// `len` bytes; reusing the same buffer across calls amortizes the allocation and
    /// page-fault cost, which dominates serialization of large indexes. Every byte is
    /// written exactly once via `extend_from_slice`, so the buffer is never zero-filled.
    pub(crate) fn new(bytes: &'a mut Vec<u8>, len: usize) -> Self {
        bytes.clear();
        bytes.reserve_exact(len);
        Self { bytes, len }
    }

    pub(crate) fn write_magic(&mut self) {
        self.write_bytes(FORMAT_MAGIC);
    }

    #[inline]
    #[cfg(feature = "f32-storage")]
    pub(crate) fn write_f32(&mut self, value: f32) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Write 2D box records from f32 structure-of-arrays columns (one `[min_x,
    /// min_y, max_x, max_y]` record per node).
    #[cfg(feature = "f32-storage")]
    pub(crate) fn write_soa_boxes_f32_2d(
        &mut self,
        min_xs: &[f32],
        min_ys: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        for i in 0..min_xs.len() {
            self.write_f32(min_xs[i]);
            self.write_f32(min_ys[i]);
            self.write_f32(max_xs[i]);
            self.write_f32(max_ys[i]);
        }
    }

    /// Write 3D box records from f32 structure-of-arrays columns (one `[min_x,
    /// min_y, min_z, max_x, max_y, max_z]` record per node).
    #[cfg(feature = "f32-storage")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_soa_boxes_f32_3d(
        &mut self,
        min_xs: &[f32],
        min_ys: &[f32],
        min_zs: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        max_zs: &[f32],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), min_zs.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        debug_assert_eq!(min_xs.len(), max_zs.len());
        for i in 0..min_xs.len() {
            self.write_f32(min_xs[i]);
            self.write_f32(min_ys[i]);
            self.write_f32(min_zs[i]);
            self.write_f32(max_xs[i]);
            self.write_f32(max_ys[i]);
            self.write_f32(max_zs[i]);
        }
    }

    #[inline]
    pub(crate) fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    // Only the big-endian box-writing fallback uses this; on little-endian targets
    // the bulk memcpy path makes it dead, so suppress the lint there.
    #[inline]
    #[cfg_attr(target_endian = "little", allow(dead_code))]
    pub(crate) fn write_f64(&mut self, value: f64) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Write 2D boxes as little-endian box records.
    ///
    /// On little-endian targets the slice is copied in one bulk memcpy. Other targets
    /// fall back to per-field writes.
    #[inline]
    pub(crate) fn write_box2d_slice(&mut self, values: &[Box2D]) {
        #[cfg(target_endian = "little")]
        {
            debug_assert_eq!(
                core::mem::size_of::<Box2D>(),
                4 * core::mem::size_of::<f64>()
            );
            debug_assert_eq!(core::mem::align_of::<Box2D>(), core::mem::align_of::<f64>());
            // SAFETY: `Box2D` is `repr(C)` with exactly four contiguous `f64`
            // fields in the persisted order and no padding; on little-endian targets
            // those native bytes are the little-endian file encoding.
            unsafe { self.write_raw_slice_bytes(values) };
        }
        #[cfg(not(target_endian = "little"))]
        for item in values {
            self.write_f64(item.min_x);
            self.write_f64(item.min_y);
            self.write_f64(item.max_x);
            self.write_f64(item.max_y);
        }
    }

    /// Write 3D boxes as little-endian box records.
    ///
    /// On little-endian targets the slice is copied in one bulk memcpy. Other targets
    /// fall back to per-field writes.
    #[inline]
    pub(crate) fn write_box3d_slice(&mut self, values: &[Box3D]) {
        #[cfg(target_endian = "little")]
        {
            debug_assert_eq!(
                core::mem::size_of::<Box3D>(),
                6 * core::mem::size_of::<f64>()
            );
            debug_assert_eq!(core::mem::align_of::<Box3D>(), core::mem::align_of::<f64>());
            // SAFETY: `Box3D` is `repr(C)` with exactly six contiguous `f64`
            // fields in the persisted order and no padding; on little-endian targets
            // those native bytes are the little-endian file encoding.
            unsafe { self.write_raw_slice_bytes(values) };
        }
        #[cfg(not(target_endian = "little"))]
        for item in values {
            self.write_f64(item.min_x);
            self.write_f64(item.min_y);
            self.write_f64(item.min_z);
            self.write_f64(item.max_x);
            self.write_f64(item.max_y);
            self.write_f64(item.max_z);
        }
    }

    /// Write 2D nodes **interleaved**: each `[min_x, min_y, max_x, max_y]` box
    /// record immediately followed by its `u64` index entry. `entries` and
    /// `indices` have one element per node (same length). Produces the node data
    /// of an interleaved-layout `TREE` chunk, trading the bulk box memcpy for a
    /// layout a streaming reader fetches in one read per level.
    #[cfg(feature = "stream")]
    pub(crate) fn write_interleaved_2d(&mut self, entries: &[Box2D], indices: &[usize]) {
        debug_assert_eq!(entries.len(), indices.len());
        for (b, &idx) in entries.iter().zip(indices) {
            self.write_f64(b.min_x);
            self.write_f64(b.min_y);
            self.write_f64(b.max_x);
            self.write_f64(b.max_y);
            self.write_u64(idx as u64);
        }
    }

    /// Write 3D nodes **interleaved**: each `[min_x, min_y, min_z, max_x, max_y,
    /// max_z]` box record immediately followed by its `u64` index entry. See
    /// [`write_interleaved_2d`](Self::write_interleaved_2d).
    #[cfg(feature = "stream")]
    pub(crate) fn write_interleaved_3d(&mut self, entries: &[Box3D], indices: &[usize]) {
        debug_assert_eq!(entries.len(), indices.len());
        for (b, &idx) in entries.iter().zip(indices) {
            self.write_f64(b.min_x);
            self.write_f64(b.min_y);
            self.write_f64(b.min_z);
            self.write_f64(b.max_x);
            self.write_f64(b.max_y);
            self.write_f64(b.max_z);
            self.write_u64(idx as u64);
        }
    }

    /// Write 2D box records from structure-of-arrays columns (one `[min_x, min_y,
    /// max_x, max_y]` record per node). Produces the same bytes as
    /// [`write_box2d_slice`](Self::write_box2d_slice) on an equivalent AoS slice.
    #[cfg(feature = "simd")]
    pub(crate) fn write_soa_boxes_2d(
        &mut self,
        min_xs: &[f64],
        min_ys: &[f64],
        max_xs: &[f64],
        max_ys: &[f64],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        for i in 0..min_xs.len() {
            self.write_f64(min_xs[i]);
            self.write_f64(min_ys[i]);
            self.write_f64(max_xs[i]);
            self.write_f64(max_ys[i]);
        }
    }

    /// Write 3D box records from structure-of-arrays columns (one `[min_x, min_y,
    /// min_z, max_x, max_y, max_z]` record per node). Produces the same bytes as
    /// [`write_box3d_slice`](Self::write_box3d_slice) on an equivalent AoS slice.
    #[cfg(feature = "simd")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_soa_boxes_3d(
        &mut self,
        min_xs: &[f64],
        min_ys: &[f64],
        min_zs: &[f64],
        max_xs: &[f64],
        max_ys: &[f64],
        max_zs: &[f64],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), min_zs.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        debug_assert_eq!(min_xs.len(), max_zs.len());
        for i in 0..min_xs.len() {
            self.write_f64(min_xs[i]);
            self.write_f64(min_ys[i]);
            self.write_f64(min_zs[i]);
            self.write_f64(max_xs[i]);
            self.write_f64(max_ys[i]);
            self.write_f64(max_zs[i]);
        }
    }

    /// Write a slice of `usize` as little-endian `u64`. Bulk-copied on 64-bit
    /// little-endian targets (where `usize` and `u64` share a layout); other targets
    /// fall back to per-element widening writes.
    #[inline]
    pub(crate) fn write_usize_slice_as_u64(&mut self, values: &[usize]) {
        #[cfg(all(target_endian = "little", target_pointer_width = "64"))]
        {
            // SAFETY: on a 64-bit little-endian target `usize` has the same size and
            // byte order as `u64`, so the slice's bytes equal the LE encoding that the
            // per-element `write_u64` path would produce.
            unsafe { self.write_raw_slice_bytes(values) };
        }
        #[cfg(not(all(target_endian = "little", target_pointer_width = "64")))]
        for &value in values {
            self.write_u64(value as u64);
        }
    }

    /// Write the optional payload section in **leaf order**: a `(num_items + 1)`
    /// prefix-offset table plus the concatenated blobs, both ordered by leaf rank
    /// so a spatial query (which visits leaves in contiguous runs) fetches them
    /// in coalesced reads. `leaf_order[r]` is the insertion id of the item at
    /// leaf rank `r` (i.e. the leaf entry of `indices`); `payloads` is indexed by
    /// insertion id.
    pub(crate) fn write_payload_offsets_and_blobs<P: AsRef<[u8]>>(
        &mut self,
        payloads: &[P],
        leaf_order: &[usize],
    ) {
        let mut acc: u64 = 0;
        self.write_u64(0);
        for &id in leaf_order {
            acc += payloads[id].as_ref().len() as u64;
            self.write_u64(acc);
        }
        for &id in leaf_order {
            self.write_bytes(payloads[id].as_ref());
        }
    }

    /// Write a fixed-width payload section in **leaf order**: just the blobs,
    /// each exactly `stride` bytes, with no offset table (the reader addresses
    /// blob `r` at `r * stride`). `leaf_order[r]` is the insertion id at leaf
    /// rank `r`; callers must have validated every blob is `stride` bytes.
    pub(crate) fn write_payload_blobs_fixed(&mut self, payloads: &[&[u8]], leaf_order: &[usize]) {
        for &id in leaf_order {
            self.write_bytes(payloads[id]);
        }
    }

    pub(crate) fn finish(self) {
        debug_assert_eq!(self.bytes.len(), self.len);
    }

    #[inline]
    fn write_bytes(&mut self, source: &[u8]) {
        debug_assert!(self.bytes.len() + source.len() <= self.len);
        // Capacity is pre-reserved in `new`, so this appends without reallocating
        // and without re-zeroing the destination.
        self.bytes.extend_from_slice(source);
    }

    #[inline]
    unsafe fn write_raw_slice_bytes<T>(&mut self, values: &[T]) {
        // SAFETY: the caller guarantees that `values` can be persisted by copying its
        // native bytes directly. The slice itself is valid and initialized.
        let source = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        self.write_bytes(source);
    }
}

pub(crate) fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, LoadError> {
    let end = offset.checked_add(8).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

#[inline]
pub(crate) fn read_u64_le_unchecked(bytes: &[u8], offset: usize) -> u64 {
    debug_assert!(offset <= bytes.len());
    debug_assert!(bytes.len() - offset >= 8);

    let mut value = 0u64;
    // SAFETY: callers only use this for slices and offsets validated by
    // `parse_index`; unaligned byte buffers are copied into an aligned u64.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().add(offset),
            (&mut value as *mut u64).cast::<u8>(),
            8,
        );
    }
    u64::from_le(value)
}

#[inline]
pub(crate) fn read_f64_le_unchecked(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64_le_unchecked(bytes, offset))
}

#[inline]
#[cfg(feature = "f32-storage")]
pub(crate) fn read_f32_le_unchecked(bytes: &[u8], offset: usize) -> f32 {
    debug_assert!(offset + 4 <= bytes.len());
    let mut value = 0u32;
    // SAFETY: callers only use this for slices and offsets validated by
    // the f32 byte parsers; unaligned byte buffers are copied into an aligned u32.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().add(offset),
            (&mut value as *mut u32).cast::<u8>(),
            4,
        );
    }
    f32::from_bits(u32::from_le(value))
}

fn usize_from_u64(value: u64) -> Result<usize, LoadError> {
    usize::try_from(value).map_err(|_| LoadError::IntegerOverflow)
}

// ===========================================================================
// PSINDEX — chunk container
//
// A file is a small superblock, a flat chunk directory (typed, length + offset,
// with a critical/optional bit), then the chunks. A reader skips an unknown
// *optional* chunk and rejects an unknown *critical* one, so every future
// addition is non-breaking. Only non-derivable data is stored: the tree's
// `num_nodes`, `level_count`, and `level_bounds` are all recomputed from
// `num_items` + `node_size` at load (no second source of truth to drift).
// ===========================================================================

/// Stored `version` value. Bumped only on a breaking layout change; a reader
/// rejects any other value.
pub(crate) const FORMAT_VERSION: u64 = 2;
/// Fixed superblock length: magic(8) + version(8) + chunk_count(4) + reserved(12).
pub(crate) const SUPERBLOCK_LEN: usize = 32;
/// One chunk-directory entry: tag(4) + flags(4) + offset(8) + length(8).
pub(crate) const CHUNK_ENTRY_LEN: usize = 24;
/// `flags` bit marking a chunk a reader must understand (else reject the file).
pub(crate) const CHUNK_FLAG_CRITICAL: u32 = 1;
// Tag namespace (see FORMAT.md): an uppercase-first ASCII tag is reserved for
// this format; lowercase-first tags are free for application-private chunks.
/// The packed tree (descriptor + raw node data). Critical.
pub(crate) const TAG_TREE: [u8; 4] = *b"TREE";
/// The optional payload section (descriptor + offset table + blobs). Optional —
/// an index-only reader skips it.
pub(crate) const TAG_PYLD: [u8; 4] = *b"PYLD";

/// A located chunk from the directory. Criticality is enforced during parsing
/// (an unknown critical chunk is rejected), so it is not retained here.
#[derive(Debug)]
pub(crate) struct ChunkRef {
    pub(crate) tag: [u8; 4],
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

/// Parse and validate the superblock + chunk directory. Every entry's byte range
/// is checked against the buffer; an unknown **critical** chunk (tag not in
/// `known_critical`) is rejected. Does not read chunk contents.
pub(crate) fn parse_container(
    bytes: &[u8],
    known_critical: &[[u8; 4]],
) -> Result<Vec<ChunkRef>, LoadError> {
    if bytes.len() < SUPERBLOCK_LEN {
        return Err(LoadError::Truncated);
    }
    if &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(LoadError::BadMagic);
    }
    if read_u64_at(bytes, 8)? != FORMAT_VERSION {
        return Err(LoadError::UnsupportedVersion);
    }
    let chunk_count = read_u32_at(bytes, 16)? as usize;

    let dir_len = chunk_count
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or(LoadError::IntegerOverflow)?;
    let dir_end = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or(LoadError::IntegerOverflow)?;
    if bytes.len() < dir_end {
        return Err(LoadError::Truncated);
    }

    let mut chunks = Vec::with_capacity(chunk_count);
    let mut max_end = dir_end;
    for i in 0..chunk_count {
        let base = SUPERBLOCK_LEN + i * CHUNK_ENTRY_LEN;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&bytes[base..base + 4]);
        let flags = read_u32_at(bytes, base + 4)?;
        let offset = read_u64_at(bytes, base + 8).and_then(usize_from_u64)?;
        let len = read_u64_at(bytes, base + 16).and_then(usize_from_u64)?;
        let end = offset.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
        if offset < dir_end || end > bytes.len() {
            return Err(LoadError::InvalidTree);
        }
        max_end = max_end.max(end);
        let critical = flags & CHUNK_FLAG_CRITICAL != 0;
        if critical && !known_critical.contains(&tag) {
            return Err(LoadError::UnsupportedVersion);
        }
        chunks.push(ChunkRef { tag, offset, len });
    }
    // Reject trailing bytes the directory does not account for (beyond the
    // last chunk's 8-byte alignment pad).
    if bytes.len() > ((max_end + 7) & !7) {
        return Err(LoadError::LengthMismatch {
            expected: max_end,
            actual: bytes.len(),
        });
    }
    Ok(chunks)
}

/// Find the first chunk with `tag`.
pub(crate) fn find_chunk(chunks: &[ChunkRef], tag: [u8; 4]) -> Option<&ChunkRef> {
    chunks.iter().find(|c| c.tag == tag)
}

/// Decoded `TREE` descriptor. `num_nodes` / `level_count` are *not* stored; the
/// caller derives them from `num_items` + `node_size`.
pub(crate) struct TreeDesc {
    pub(crate) dimensions: usize,
    pub(crate) coord_bytes: usize,
    pub(crate) interleaved: bool,
    pub(crate) num_items: usize,
    pub(crate) node_size: usize,
    /// Byte length of the descriptor (so node data starts at `desc_len`). Only
    /// the streaming reader consults it; the in-memory parser uses the node-data
    /// slice this comes paired with.
    #[cfg_attr(not(feature = "stream"), allow(dead_code))]
    pub(crate) desc_len: usize,
}

/// Minimum `TREE` descriptor length this version writes.
pub(crate) const TREE_DESC_LEN: usize = 24;
/// Minimum `PYLD` descriptor length an older reader must tolerate (`desc_len`
/// floor). Readers accept any `desc_len >= PYLD_DESC_LEN` and skip to the body.
pub(crate) const PYLD_DESC_LEN: usize = 8;
/// `PYLD` descriptor length this version writes: the 8-byte base plus the
/// `record_stride` u32. A reader before this field existed reads only the base
/// and treats the payload as variable-width (`record_stride = 0`).
pub(crate) const PYLD_DESC_LEN_FIXED: usize = 12;

/// Parse a `TREE` chunk's descriptor; returns it plus the node-data slice that
/// follows. Validates the fixed fields but not the node data (the dimension
/// parsers do that against the derived tree shape).
pub(crate) fn parse_tree_chunk(chunk: &[u8]) -> Result<(TreeDesc, &[u8]), LoadError> {
    if chunk.len() < TREE_DESC_LEN {
        return Err(LoadError::Truncated);
    }
    let desc_len = read_u32_at(chunk, 0)? as usize;
    if desc_len < TREE_DESC_LEN || desc_len > chunk.len() {
        return Err(LoadError::InvalidTree);
    }
    let dimensions = chunk[4] as usize;
    let coord_bytes = chunk[5] as usize;
    let layout = chunk[6];
    if (dimensions != 2 && dimensions != 3) || (coord_bytes != 4 && coord_bytes != 8) || layout > 1
    {
        return Err(LoadError::UnsupportedVersion);
    }
    let num_items = read_u64_at(chunk, 8).and_then(usize_from_u64)?;
    let node_size = read_u16_at(chunk, 16)? as usize;
    if !(2..=65535).contains(&node_size) {
        return Err(LoadError::InvalidNodeSize { node_size });
    }
    Ok((
        TreeDesc {
            dimensions,
            coord_bytes,
            interleaved: layout == 1,
            num_items,
            node_size,
            desc_len,
        },
        &chunk[desc_len..],
    ))
}

/// Decoded `PYLD` descriptor; returns it plus the slice that follows (offset
/// table + blobs).
pub(crate) struct PyldDesc {
    /// Only the streaming reader consults this (to locate the body); the
    /// in-memory parser uses the body slice it comes paired with.
    #[cfg_attr(not(feature = "stream"), allow(dead_code))]
    pub(crate) desc_len: usize,
    /// Fixed record stride in bytes, or `0` for a variable-width payload (offset
    /// table present). Read from the descriptor when `desc_len` covers it.
    pub(crate) record_stride: usize,
}

pub(crate) fn parse_pyld_chunk(chunk: &[u8]) -> Result<(PyldDesc, &[u8]), LoadError> {
    if chunk.len() < PYLD_DESC_LEN {
        return Err(LoadError::Truncated);
    }
    let desc_len = read_u32_at(chunk, 0)? as usize;
    if desc_len < PYLD_DESC_LEN || desc_len > chunk.len() {
        return Err(LoadError::InvalidTree);
    }
    let ordering = chunk[4];
    let compression = chunk[5];
    // Only leaf-rank ordering, uncompressed blobs exist in this version.
    if ordering != 0 || compression != 0 {
        return Err(LoadError::UnsupportedVersion);
    }
    // `record_stride` was appended after the 8-byte base; an older file without
    // it has `desc_len == 8` and is read as variable-width (stride 0).
    let record_stride = if desc_len >= PYLD_DESC_LEN_FIXED {
        read_u32_at(chunk, 8)? as usize
    } else {
        0
    };
    Ok((
        PyldDesc {
            desc_len,
            record_stride,
        },
        &chunk[desc_len..],
    ))
}

/// Optional descriptive metadata chunk (CRS / content type / attribution).
/// Optional — readers that do not care skip it.
pub(crate) const TAG_META: [u8; 4] = *b"META";

// `META` field ids. The chunk is a flat list of `(id: u16, len: u32, bytes)`
// fields read until the chunk ends; an unknown id is skipped, so new fields are
// non-breaking. Values are opaque UTF-8 strings the writer supplied.
const META_CRS: u16 = 0;
const META_CONTENT_TYPE: u16 = 1;
const META_ATTRIBUTION: u16 = 2;

/// Descriptive fields to write into a `META` chunk (borrowed, write side).
#[derive(Default)]
pub(crate) struct MetaFields<'a> {
    pub(crate) crs: Option<&'a str>,
    pub(crate) content_type: Option<&'a str>,
    pub(crate) attribution: Option<&'a str>,
}

impl MetaFields<'_> {
    pub(crate) fn is_empty(&self) -> bool {
        self.crs.is_none() && self.content_type.is_none() && self.attribution.is_none()
    }

    /// Byte length of the `META` chunk content for these fields.
    pub(crate) fn content_len(&self) -> usize {
        [self.crs, self.content_type, self.attribution]
            .into_iter()
            .flatten()
            .map(|s| 6 + s.len()) // id(2) + len(4) + bytes
            .sum()
    }
}

/// Descriptive metadata read from a file's `META` chunk. Every field is an opaque
/// string the writer supplied; this crate does not interpret them (e.g. the CRS
/// is whatever identifier the producer chose, such as `"EPSG:4326"`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileMetadata {
    /// Coordinate reference system identifier, if present.
    pub crs: Option<String>,
    /// Payload content type (media type), if present.
    pub content_type: Option<String>,
    /// Attribution / license string, if present.
    pub attribution: Option<String>,
}

impl ByteWriter<'_> {
    pub(crate) fn write_meta(&mut self, fields: &MetaFields<'_>) {
        for (id, value) in [
            (META_CRS, fields.crs),
            (META_CONTENT_TYPE, fields.content_type),
            (META_ATTRIBUTION, fields.attribution),
        ] {
            if let Some(s) = value {
                self.write_u16(id);
                self.write_u32(s.len() as u32);
                self.write_bytes(s.as_bytes());
            }
        }
    }
}

/// Parse a `META` chunk's flat field list into owned strings.
fn parse_meta(content: &[u8]) -> Result<FileMetadata, LoadError> {
    let mut md = FileMetadata::default();
    let mut off = 0;
    while off < content.len() {
        let id = read_u16_at(content, off)?;
        let len = read_u32_at(content, off + 2)? as usize;
        let start = off + 6;
        let end = start.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
        let bytes = content.get(start..end).ok_or(LoadError::Truncated)?;
        let s = std::str::from_utf8(bytes).map_err(|_| LoadError::InvalidTree)?;
        match id {
            META_CRS => md.crs = Some(s.to_owned()),
            META_CONTENT_TYPE => md.content_type = Some(s.to_owned()),
            META_ATTRIBUTION => md.attribution = Some(s.to_owned()),
            _ => {} // unknown field: skip
        }
        off = end;
    }
    Ok(md)
}

/// Read the optional descriptive metadata from a serialized index, without
/// loading the index. Returns an empty [`FileMetadata`] when there is no `META`
/// chunk.
pub fn read_metadata(bytes: &[u8]) -> Result<FileMetadata, LoadError> {
    let chunks = parse_container(bytes, &[TAG_TREE])?;
    match find_chunk(&chunks, TAG_META) {
        Some(m) => parse_meta(&bytes[m.offset..m.offset + m.len]),
        None => Ok(FileMetadata::default()),
    }
}

/// Frame a `TREE` chunk (+ optional `PYLD` / `META`) into a container in `out`.
/// The dimension-specific node bytes are written by `write_nodes`, called once
/// after the `TREE` descriptor; everything else (sizing, directory, alignment,
/// payload, metadata) is shared by the 2D and 3D serializers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_index_container(
    out: &mut Vec<u8>,
    dimensions: u8,
    coord_bytes: u8,
    interleaved: bool,
    num_items: usize,
    num_nodes: usize,
    node_size: usize,
    write_nodes: impl FnOnce(&mut ByteWriter),
    payloads: Option<&[&[u8]]>,
    record_stride: Option<u32>,
    leaf_order: &[usize],
    meta: &MetaFields<'_>,
) -> Result<(), PayloadError> {
    // Node bytes are `record + 8` per node regardless of layout (SoA splits them,
    // interleaved keeps them adjacent), so the TREE length is layout-independent.
    let record = dimensions as usize * 2 * coord_bytes as usize;
    let tree_len = TREE_DESC_LEN + num_nodes * (record + 8);

    let pyld_len = match payloads {
        Some(p) => {
            if p.len() != num_items {
                return Err(PayloadError::CountMismatch {
                    expected: num_items,
                    got: p.len(),
                });
            }
            match record_stride {
                // Fixed-width: every blob is `stride` bytes, no offset table.
                Some(stride) => {
                    let stride = stride as usize;
                    for b in p {
                        if b.len() != stride {
                            return Err(PayloadError::RecordSizeMismatch {
                                stride,
                                got: b.len(),
                            });
                        }
                    }
                    let blob_total = num_items
                        .checked_mul(stride)
                        .ok_or(PayloadError::TooLarge)?;
                    Some(PYLD_DESC_LEN_FIXED + blob_total)
                }
                // Variable-width: prefix-offset table plus the blobs.
                None => {
                    let mut blob_total: u64 = 0;
                    for b in p {
                        blob_total = blob_total
                            .checked_add(b.len() as u64)
                            .ok_or(PayloadError::TooLarge)?;
                    }
                    let blob_total =
                        usize::try_from(blob_total).map_err(|_| PayloadError::TooLarge)?;
                    Some(PYLD_DESC_LEN + (num_items + 1) * 8 + blob_total)
                }
            }
        }
        None => None,
    };
    let meta_len = (!meta.is_empty()).then(|| meta.content_len());

    // Chunks in write order: TREE, then optional PYLD, then optional META.
    let mut lens = vec![tree_len];
    if let Some(pl) = pyld_len {
        lens.push(pl);
    }
    if let Some(ml) = meta_len {
        lens.push(ml);
    }
    let (total, off) = plan_container(&lens).map_err(|_| PayloadError::TooLarge)?;
    let pyld_idx = pyld_len.map(|_| 1);
    let meta_idx = meta_len.map(|_| if pyld_len.is_some() { 2 } else { 1 });

    let mut bytes = ByteWriter::new(out, total);
    bytes.write_superblock(lens.len() as u32);
    bytes.write_chunk_entry(&TAG_TREE, true, off[0], tree_len);
    if let Some(i) = pyld_idx {
        bytes.write_chunk_entry(&TAG_PYLD, false, off[i], lens[i]);
    }
    if let Some(i) = meta_idx {
        bytes.write_chunk_entry(&TAG_META, false, off[i], lens[i]);
    }

    let mut pos = SUPERBLOCK_LEN + lens.len() * CHUNK_ENTRY_LEN;
    bytes.write_zeros(off[0] - pos);
    bytes.write_tree_desc(dimensions, coord_bytes, interleaved, num_items, node_size);
    write_nodes(&mut bytes);
    pos = off[0] + tree_len;
    if let (Some(i), Some(p)) = (pyld_idx, payloads) {
        bytes.write_zeros(off[i] - pos);
        bytes.write_pyld_desc(record_stride);
        match record_stride {
            Some(_) => bytes.write_payload_blobs_fixed(p, leaf_order),
            None => bytes.write_payload_offsets_and_blobs(p, leaf_order),
        }
        pos = off[i] + lens[i];
    }
    if let Some(i) = meta_idx {
        bytes.write_zeros(off[i] - pos);
        bytes.write_meta(meta);
        pos = off[i] + lens[i];
    }
    bytes.write_zeros(total - pos);
    bytes.finish();
    Ok(())
}

/// Round up to the next 8-byte boundary (chunks are 8-aligned).
fn align8(x: usize) -> Result<usize, LoadError> {
    x.checked_add(7)
        .map(|v| v & !7)
        .ok_or(LoadError::IntegerOverflow)
}

/// Plan a container from each chunk's content length: returns the total file
/// length and the absolute byte offset of each chunk's content (8-aligned).
pub(crate) fn plan_container(content_lens: &[usize]) -> Result<(usize, Vec<usize>), LoadError> {
    let dir_len = content_lens
        .len()
        .checked_mul(CHUNK_ENTRY_LEN)
        .ok_or(LoadError::IntegerOverflow)?;
    let mut cur = SUPERBLOCK_LEN
        .checked_add(dir_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let mut offsets = Vec::with_capacity(content_lens.len());
    for &len in content_lens {
        cur = align8(cur)?;
        offsets.push(cur);
        cur = cur.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
    }
    Ok((align8(cur)?, offsets))
}

/// Container writers. Kept in a separate `impl` block alongside the container
/// codec; they reuse the pre-sized append path of the main `ByteWriter`.
impl ByteWriter<'_> {
    pub(crate) fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_u8(&mut self, value: u8) {
        self.write_bytes(&[value]);
    }

    /// Append a raw byte slice. Used by the container round-trip tests to frame
    /// arbitrary chunk content; the real writers append typed fields directly.
    #[cfg(test)]
    pub(crate) fn write_raw(&mut self, bytes: &[u8]) {
        self.write_bytes(bytes);
    }

    /// Append `n` zero bytes (chunk descriptor reserved fields / alignment pads).
    pub(crate) fn write_zeros(&mut self, n: usize) {
        for _ in 0..n {
            self.write_bytes(&[0]);
        }
    }

    pub(crate) fn write_superblock(&mut self, chunk_count: u32) {
        self.write_magic();
        self.write_u64(FORMAT_VERSION);
        self.write_u32(chunk_count);
        self.write_zeros(12);
    }

    pub(crate) fn write_chunk_entry(
        &mut self,
        tag: &[u8; 4],
        critical: bool,
        offset: usize,
        len: usize,
    ) {
        self.write_bytes(tag);
        self.write_u32(if critical { CHUNK_FLAG_CRITICAL } else { 0 });
        self.write_u64(offset as u64);
        self.write_u64(len as u64);
    }

    pub(crate) fn write_tree_desc(
        &mut self,
        dimensions: u8,
        coord_bytes: u8,
        interleaved: bool,
        num_items: usize,
        node_size: usize,
    ) {
        self.write_u32(TREE_DESC_LEN as u32);
        self.write_u8(dimensions);
        self.write_u8(coord_bytes);
        self.write_u8(if interleaved { 1 } else { 0 });
        self.write_u8(0);
        self.write_u64(num_items as u64);
        self.write_u16(node_size as u16);
        self.write_zeros(6);
    }

    /// Write the `PYLD` descriptor. A variable-width payload keeps the original
    /// 8-byte descriptor (byte-identical to older files); a fixed-width one
    /// appends the `record_stride` field, growing `desc_len` to 12.
    pub(crate) fn write_pyld_desc(&mut self, record_stride: Option<u32>) {
        let desc_len = match record_stride {
            Some(_) => PYLD_DESC_LEN_FIXED,
            None => PYLD_DESC_LEN,
        };
        self.write_u32(desc_len as u32);
        self.write_u8(0); // ordering = leaf rank
        self.write_u8(0); // compression = none
        self.write_u16(0); // reserved
        if let Some(stride) = record_stride {
            self.write_u32(stride);
        }
    }
}

pub(crate) fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, LoadError> {
    let end = offset.checked_add(4).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

pub(crate) fn read_u16_at(bytes: &[u8], offset: usize) -> Result<u16, LoadError> {
    let end = offset.checked_add(2).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

/// Derive the level-bound positions (cumulative node counts per level) from the
/// item count and node size. The format does not store them — they are a function of the
/// tree shape, so deriving avoids a second source of truth that could drift.
pub(crate) fn derive_level_bounds(
    num_items: usize,
    node_size: usize,
    level_count: usize,
) -> Vec<usize> {
    let mut bounds = Vec::with_capacity(level_count);
    let mut n = num_items;
    let mut total = n;
    bounds.push(total);
    while bounds.len() < level_count {
        n = n.div_ceil(node_size);
        total += n;
        bounds.push(total);
    }
    bounds
}

/// A parsed SoA tree: the node bytes (borrowed) plus the derived shape. The
/// `level_bounds` are owned because they are derived at load, not stored.
pub(crate) struct ParsedTree<'a> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) entries: &'a [u8],
    pub(crate) indices: &'a [u8],
}

/// Parse a container's `TREE` chunk for an in-memory (SoA) reader, plus the
/// optional payload. Rejects an interleaved tree (that layout is streaming-only).
pub(crate) fn parse_index(
    bytes: &[u8],
    dimensions: usize,
    coord_bytes: usize,
) -> Result<(ParsedTree<'_>, Option<ParsedPayload<'_>>), LoadError> {
    let chunks = parse_container(bytes, &[TAG_TREE])?;
    let tree_ref = find_chunk(&chunks, TAG_TREE).ok_or(LoadError::InvalidTree)?;
    let (desc, node_data) =
        parse_tree_chunk(&bytes[tree_ref.offset..tree_ref.offset + tree_ref.len])?;
    if desc.dimensions != dimensions || desc.coord_bytes != coord_bytes {
        return Err(LoadError::UnsupportedVersion);
    }
    if desc.interleaved {
        // The interleaved layout is read only by the streaming reader.
        return Err(LoadError::UnsupportedVersion);
    }

    let (num_nodes, level_count) = expected_tree_shape(desc.num_items, desc.node_size)?;
    let record = dimensions
        .checked_mul(2 * coord_bytes)
        .ok_or(LoadError::IntegerOverflow)?;
    let entries_len = num_nodes
        .checked_mul(record)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_len = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;
    let node_len = entries_len
        .checked_add(indices_len)
        .ok_or(LoadError::IntegerOverflow)?;
    if node_data.len() != node_len {
        return Err(LoadError::InvalidTree);
    }

    let parsed = ParsedTree {
        node_size: desc.node_size,
        num_items: desc.num_items,
        num_nodes,
        level_count,
        level_bounds: derive_level_bounds(desc.num_items, desc.node_size, level_count),
        entries: &node_data[..entries_len],
        indices: &node_data[entries_len..],
    };
    validate_tree_indices(&parsed)?;

    let payload = match find_chunk(&chunks, TAG_PYLD) {
        Some(p) => {
            let (pd, body) = parse_pyld_chunk(&bytes[p.offset..p.offset + p.len])?;
            Some(parse_payload_body(body, desc.num_items, pd.record_stride)?)
        }
        None => None,
    };
    Ok((parsed, payload))
}

/// Validate leaf and internal child pointers against the derived level bounds.
fn validate_tree_indices(p: &ParsedTree<'_>) -> Result<(), LoadError> {
    for pos in 0..p.num_items {
        let index = read_u64_at(p.indices, pos * 8).and_then(usize_from_u64)?;
        if index >= p.num_items {
            return Err(LoadError::InvalidTree);
        }
    }
    for level in 1..p.level_count {
        let level_start = p.level_bounds[level - 1];
        let level_end = p.level_bounds[level];
        let child_level_start = if level == 1 {
            0
        } else {
            p.level_bounds[level - 2]
        };
        let child_level_end = level_start;
        for pos in level_start..level_end {
            let index = read_u64_at(p.indices, pos * 8).and_then(usize_from_u64)?;
            if index < child_level_start || index >= child_level_end {
                return Err(LoadError::InvalidTree);
            }
            if (index - child_level_start) % p.node_size != 0 {
                return Err(LoadError::InvalidTree);
            }
        }
    }
    Ok(())
}

/// Validate and slice a `PYLD` chunk's post-descriptor bytes. A fixed-width
/// payload (`stride != 0`) is just `num_items * stride` blob bytes with no table;
/// a variable-width one is a `(num_items + 1)` prefix-offset table followed by the
/// blob region.
fn parse_payload_body(
    body: &[u8],
    num_items: usize,
    stride: usize,
) -> Result<ParsedPayload<'_>, LoadError> {
    if stride != 0 {
        let total = num_items
            .checked_mul(stride)
            .ok_or(LoadError::IntegerOverflow)?;
        if body.len() != total {
            return Err(LoadError::LengthMismatch {
                expected: total,
                actual: body.len(),
            });
        }
        return Ok(ParsedPayload {
            offsets: &[],
            blobs: body,
            stride,
        });
    }
    let offsets_len = num_items
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or(LoadError::IntegerOverflow)?;
    if body.len() < offsets_len {
        return Err(LoadError::Truncated);
    }
    let offsets = &body[..offsets_len];
    let mut prev = 0u64;
    for i in 0..=num_items {
        let off = read_u64_at(offsets, i * 8)?;
        if (i == 0 && off != 0) || off < prev {
            return Err(LoadError::InvalidTree);
        }
        prev = off;
    }
    let blob_total = usize_from_u64(prev)?;
    let blobs = &body[offsets_len..];
    if blobs.len() != blob_total {
        return Err(LoadError::LengthMismatch {
            expected: offsets_len + blob_total,
            actual: body.len(),
        });
    }
    Ok(ParsedPayload {
        offsets,
        blobs,
        stride: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Box2D, Index2D, Index2DBuilder};

    fn build(n: usize) -> Index2D {
        let mut builder = Index2DBuilder::new(n).node_size(16);
        for i in 0..n {
            let v = i as f64;
            builder.add(Box2D::new(v, v, v + 1.0, v + 1.0));
        }
        builder.finish().unwrap()
    }

    // ---- chunk container ----

    /// Frame chunks `(tag, critical, content)` into a container buffer.
    fn encode_container(chunks: &[([u8; 4], bool, Vec<u8>)]) -> Vec<u8> {
        let lens: Vec<usize> = chunks.iter().map(|(_, _, c)| c.len()).collect();
        let (total, offsets) = plan_container(&lens).unwrap();
        let mut buf = Vec::new();
        let mut w = ByteWriter::new(&mut buf, total);
        w.write_superblock(chunks.len() as u32);
        for (i, (tag, critical, c)) in chunks.iter().enumerate() {
            w.write_chunk_entry(tag, *critical, offsets[i], c.len());
        }
        let mut pos = SUPERBLOCK_LEN + chunks.len() * CHUNK_ENTRY_LEN;
        for (i, (_, _, c)) in chunks.iter().enumerate() {
            w.write_zeros(offsets[i] - pos);
            w.write_raw(c);
            pos = offsets[i] + c.len();
        }
        w.write_zeros(total - pos);
        w.finish();
        buf
    }

    fn tree_content(
        interleaved: bool,
        num_items: usize,
        node_size: usize,
        node_data: &[u8],
    ) -> Vec<u8> {
        let mut v = Vec::new();
        let mut w = ByteWriter::new(&mut v, TREE_DESC_LEN + node_data.len());
        w.write_tree_desc(2, 8, interleaved, num_items, node_size);
        w.write_raw(node_data);
        w.finish();
        v
    }

    #[test]
    fn v2_container_round_trips_tree_and_payload() {
        let node_data = vec![0xABu8; 40 * 5]; // 5 interleaved 2D nodes, arbitrary bytes
        let tree = tree_content(true, 4, 16, &node_data);

        let mut pyld = Vec::new();
        {
            let mut w = ByteWriter::new(&mut pyld, PYLD_DESC_LEN + 8 + b"blob".len());
            w.write_pyld_desc(None); // variable-width
            w.write_u64(0); // one-entry offset table fragment, just bytes here
            w.write_raw(b"blob");
            w.finish();
        }

        let buf = encode_container(&[(TAG_TREE, true, tree), (TAG_PYLD, false, pyld)]);

        let chunks = parse_container(&buf, &[TAG_TREE]).unwrap();
        assert_eq!(chunks.len(), 2);

        let tree_ref = find_chunk(&chunks, TAG_TREE).unwrap();
        let (desc, nd) =
            parse_tree_chunk(&buf[tree_ref.offset..tree_ref.offset + tree_ref.len]).unwrap();
        assert_eq!(desc.dimensions, 2);
        assert_eq!(desc.coord_bytes, 8);
        assert!(desc.interleaved);
        assert_eq!(desc.num_items, 4);
        assert_eq!(desc.node_size, 16);
        assert_eq!(nd, &node_data[..]);

        let pyld_ref = find_chunk(&chunks, TAG_PYLD).unwrap();
        let (_pd, body) =
            parse_pyld_chunk(&buf[pyld_ref.offset..pyld_ref.offset + pyld_ref.len]).unwrap();
        assert_eq!(&body[8..], b"blob"); // after the 8-byte table fragment
    }

    #[test]
    fn v2_unknown_critical_chunk_rejected() {
        let buf = encode_container(&[(*b"WHAT", true, vec![1, 2, 3, 4])]);
        // A reader that only knows TREE must reject an unknown critical chunk.
        assert_eq!(
            parse_container(&buf, &[TAG_TREE]).unwrap_err(),
            LoadError::UnsupportedVersion
        );
    }

    #[test]
    fn v2_unknown_optional_chunk_skipped() {
        let tree = tree_content(false, 4, 16, &[0u8; 32]);
        // An unknown OPTIONAL chunk is fine — the directory parses, reader ignores it.
        let buf = encode_container(&[(TAG_TREE, true, tree), (*b"XTRA", false, vec![9; 7])]);
        let chunks = parse_container(&buf, &[TAG_TREE]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(find_chunk(&chunks, TAG_TREE).is_some());
        assert!(find_chunk(&chunks, *b"XTRA").is_some());
    }

    #[test]
    fn v2_truncated_chunk_range_rejected() {
        let tree = tree_content(false, 4, 16, &[0u8; 16]);
        let mut buf = encode_container(&[(TAG_TREE, true, tree)]);
        buf.truncate(buf.len() - 4); // chunk now claims more bytes than exist
        assert!(matches!(
            parse_container(&buf, &[TAG_TREE]),
            Err(LoadError::InvalidTree | LoadError::Truncated)
        ));
    }

    #[test]
    fn meta_parses_known_fields_and_skips_unknown() {
        // A META chunk content with crs(0), an unknown future field(99), and
        // attribution(2). The unknown field must be skipped, not break parsing.
        let mut content = Vec::new();
        let put = |c: &mut Vec<u8>, id: u16, value: &[u8]| {
            c.extend_from_slice(&id.to_le_bytes());
            c.extend_from_slice(&(value.len() as u32).to_le_bytes());
            c.extend_from_slice(value);
        };
        put(&mut content, 0, b"EPSG:4326"); // crs
        put(&mut content, 99, b"from-the-future"); // unknown -> skipped
        put(&mut content, 2, b"attribution-text"); // attribution

        let md = parse_meta(&content).unwrap();
        assert_eq!(md.crs.as_deref(), Some("EPSG:4326"));
        assert_eq!(md.attribution.as_deref(), Some("attribution-text"));
        assert_eq!(md.content_type, None);

        // Empty content -> all fields absent.
        assert_eq!(parse_meta(&[]).unwrap(), FileMetadata::default());
    }

    #[test]
    fn payload_round_trip() {
        for &n in &[0usize, 1, 17, 100] {
            let index = build(n);
            let payloads: Vec<Vec<u8>> = (0..n).map(|i| format!("item-{i}").into_bytes()).collect();
            let bytes = index.to_bytes_with_payloads(&payloads).unwrap();

            // The TREE chunk is byte-identical to the index-only file's; only an
            // extra PYLD chunk (and the directory entry for it) is added.
            let index_only = index.to_bytes();
            let with = parse_container(&bytes, &[TAG_TREE]).unwrap();
            let plain = parse_container(&index_only, &[TAG_TREE]).unwrap();
            let wt = find_chunk(&with, TAG_TREE).unwrap();
            let pt = find_chunk(&plain, TAG_TREE).unwrap();
            assert_eq!(
                &bytes[wt.offset..wt.offset + wt.len],
                &index_only[pt.offset..pt.offset + pt.len]
            );
            assert!(find_chunk(&with, TAG_PYLD).is_some());

            let (_parsed, payload) = parse_index(&bytes, 2, 8).unwrap();
            let parsed = payload.expect("payload present");
            // The payload is leaf-ordered: slot `r` holds the blob of the item at
            // leaf rank `r`, whose insertion id is `index.indices[r]`.
            for r in 0..n {
                let insertion_id = index.indices[r];
                assert_eq!(payload_slice(&parsed, r), payloads[insertion_id].as_slice());
            }
        }
    }

    #[test]
    fn payload_count_mismatch_rejected() {
        let index = build(5);
        let payloads = vec![vec![1u8]; 3];
        assert_eq!(
            index.to_bytes_with_payloads(&payloads),
            Err(PayloadError::CountMismatch {
                expected: 5,
                got: 3
            })
        );
    }

    #[test]
    fn payload_file_loads_index_only_via_scalar_loader() {
        let index = build(10);
        let payloads: Vec<Vec<u8>> = (0..10).map(|_| vec![0u8; 4]).collect();
        let with = index.to_bytes_with_payloads(&payloads).unwrap();
        // The scalar loader reads the index from a payload file, ignoring the
        // payload (it validates the trailing section but does not retain it).
        let owned = Index2D::from_bytes(&with).unwrap();
        let query = Box2D::new(0.0, 0.0, 100.0, 100.0);
        assert_eq!(owned.search(query), index.search(query));
    }

    #[test]
    fn variable_length_payloads_round_trip() {
        let index = build(20);
        let payloads: Vec<Vec<u8>> = (0..20).map(|i| vec![i as u8; i]).collect(); // len 0..19
        let bytes = index.to_bytes_with_payloads(&payloads).unwrap();
        let (_parsed, payload) = parse_index(&bytes, 2, 8).unwrap();
        let parsed = payload.expect("payload present");
        for r in 0..20 {
            assert_eq!(
                payload_slice(&parsed, r),
                payloads[index.indices[r]].as_slice()
            );
        }
    }

    #[test]
    fn fixed_width_payload_round_trips_without_a_table() {
        const STRIDE: usize = 12;
        let n = 20;
        let index = build(n);
        // One 12-byte record per item, in item order.
        let mut flat = Vec::with_capacity(n * STRIDE);
        for i in 0..n {
            flat.extend_from_slice(&(i as u32).to_le_bytes());
            flat.extend_from_slice(&[i as u8; STRIDE - 4]);
        }
        let fixed = index.serialize().records(STRIDE, &flat).to_bytes().unwrap();
        let variable: Vec<Vec<u8>> = (0..n)
            .map(|i| flat[i * STRIDE..(i + 1) * STRIDE].to_vec())
            .collect();
        let var_bytes = index.to_bytes_with_payloads(&variable).unwrap();

        // The table-less layout is smaller by roughly the dropped offset table
        // (minus the 4-byte stride field the fixed descriptor adds, plus padding).
        let saving = var_bytes.len() - fixed.len();
        assert!((n * 8..=(n + 1) * 8).contains(&saving), "saving {saving}");

        let (_parsed, payload) = parse_index(&fixed, 2, 8).unwrap();
        let parsed = payload.expect("payload present");
        assert_eq!(parsed.stride, STRIDE);
        assert!(parsed.offsets.is_empty());
        for r in 0..n {
            let id = index.indices[r];
            assert_eq!(
                payload_slice(&parsed, r),
                &flat[id * STRIDE..(id + 1) * STRIDE]
            );
        }
    }

    #[test]
    fn fixed_width_wrong_record_size_rejected() {
        let index = build(4);
        // Three 8-byte records for four items, then declare stride 8: the count
        // (3 != 4) is caught first.
        let flat = vec![0u8; 3 * 8];
        assert!(matches!(
            index.serialize().records(8, &flat).to_bytes(),
            Err(PayloadError::CountMismatch { .. })
        ));
    }
}
