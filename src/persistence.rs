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
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayloadError::CountMismatch { expected, got } => write!(
                f,
                "payload count {got} does not match item count {expected}"
            ),
            PayloadError::TooLarge => write!(f, "combined payload size is too large to serialize"),
        }
    }
}

impl Error for PayloadError {}

// Packed Spatial Index file signature. The format version is stored separately
// as a little-endian u64 in the header.
const FORMAT_MAGIC: &[u8; 8] = b"PSINDEX\0";
const FORMAT_VERSION: u64 = 1;
pub(crate) const FORMAT_FLAGS_2D: u64 = 0;
pub(crate) const FORMAT_FLAGS_3D: u64 = 1;
#[cfg(feature = "f32-storage")]
const FORMAT_FLAGS_2D_F32: u64 = 2;
#[cfg(feature = "f32-storage")]
const FORMAT_FLAGS_3D_F32: u64 = 3;
pub(crate) const FORMAT_HEADER_LEN: usize = 64;
/// `flags` bit set when an optional payload section follows the index. Orthogonal
/// to the dimension/coord bits, so a payload index keeps its variant flag and the
/// index bytes stay byte-identical; only the trailing payload sections are added.
pub(crate) const FORMAT_FLAG_PAYLOAD: u64 = 1 << 8;

/// Validated header fields shared by the in-memory parser and the streaming
/// reader. Every value here has already passed magic / version / flags /
/// node-size-range / tree-shape validation.
pub(crate) struct HeaderFields {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    /// Whether a payload section follows the index (the `FORMAT_FLAG_PAYLOAD` bit).
    pub(crate) has_payload: bool,
}

/// Byte offsets of the three sections that follow the header, plus the box
/// record size. Computed purely from validated header counts, so it is the one
/// source of truth for where `level_bounds`, `boxes`, and `indices` live —
/// shared by the in-memory parser and the streaming reader.
pub(crate) struct SectionLayout {
    /// Box record size in bytes (`dimensions * 2 * coord_bytes`). Read by the
    /// streaming reader; the in-memory parser derives section sizes without it.
    #[allow(dead_code)]
    pub(crate) record: usize,
    /// Start of the `level_bounds` section (always `FORMAT_HEADER_LEN`).
    pub(crate) level_bounds_start: usize,
    /// Start of the `boxes` section.
    pub(crate) box0: usize,
    /// Start of the `indices` section.
    pub(crate) idx0: usize,
    /// Total serialized length (end of the `indices` section).
    pub(crate) total_len: usize,
}

/// Parse and validate the fixed 64-byte header from `bytes` (which must be at
/// least `FORMAT_HEADER_LEN` long; only the first 64 bytes are read). Performs
/// every header-level check the in-memory parser does, except the
/// whole-buffer length and section validation, which need the full sections.
pub(crate) fn parse_and_validate_header(
    bytes: &[u8],
    expected_flags: u64,
    allow_payload: bool,
) -> Result<HeaderFields, LoadError> {
    if bytes.len() < FORMAT_MAGIC.len() {
        return Err(LoadError::Truncated);
    }
    if &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(LoadError::BadMagic);
    }
    if bytes.len() < FORMAT_HEADER_LEN {
        return Err(LoadError::Truncated);
    }

    let version = read_u64_at(bytes, 8)?;
    if version != FORMAT_VERSION {
        return Err(LoadError::UnsupportedVersion);
    }

    let header_len = read_u64_at(bytes, 16).and_then(usize_from_u64)?;
    let flags = read_u64_at(bytes, 24)?;
    let has_payload = flags & FORMAT_FLAG_PAYLOAD != 0;
    let dimension_flags = flags & !FORMAT_FLAG_PAYLOAD;
    // The dimension flag must match; the payload bit is only accepted when the
    // caller knows how to read the trailing payload sections.
    if header_len != FORMAT_HEADER_LEN
        || dimension_flags != expected_flags
        || (has_payload && !allow_payload)
    {
        return Err(LoadError::UnsupportedVersion);
    }

    let node_size = read_u64_at(bytes, 32).and_then(usize_from_u64)?;
    let num_items = read_u64_at(bytes, 40).and_then(usize_from_u64)?;
    let num_nodes = read_u64_at(bytes, 48).and_then(usize_from_u64)?;
    let level_count = read_u64_at(bytes, 56).and_then(usize_from_u64)?;

    if !(2..=65535).contains(&node_size) {
        return Err(LoadError::InvalidNodeSize { node_size });
    }

    let (expected_nodes, expected_levels) = expected_tree_shape(num_items, node_size)?;
    if num_nodes != expected_nodes || level_count != expected_levels {
        return Err(LoadError::InvalidTree);
    }

    Ok(HeaderFields {
        node_size,
        num_items,
        num_nodes,
        level_count,
        has_payload,
    })
}

/// Compute the section byte offsets from validated header counts.
pub(crate) fn section_layout(
    level_count: usize,
    num_nodes: usize,
    dimensions: usize,
    coord_bytes: usize,
) -> Result<SectionLayout, LoadError> {
    let level_bounds_len = level_count
        .checked_mul(8)
        .ok_or(LoadError::IntegerOverflow)?;
    let record = dimensions
        .checked_mul(2 * coord_bytes)
        .ok_or(LoadError::IntegerOverflow)?;
    let entries_len = num_nodes
        .checked_mul(record)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_len = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;

    let level_bounds_start = FORMAT_HEADER_LEN;
    let box0 = level_bounds_start
        .checked_add(level_bounds_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let idx0 = box0
        .checked_add(entries_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let total_len = idx0
        .checked_add(indices_len)
        .ok_or(LoadError::IntegerOverflow)?;

    Ok(SectionLayout {
        record,
        level_bounds_start,
        box0,
        idx0,
        total_len,
    })
}

/// Byte layout of the optional payload section that follows the index.
pub(crate) struct PayloadLayout {
    /// Byte offset of the `(num_items + 1)` `u64` offset table. Used by the
    /// streaming payload reader (next step).
    #[allow(dead_code)]
    pub(crate) offsets_start: usize,
    /// Byte offset of the blob region. Used by the streaming payload reader.
    #[allow(dead_code)]
    pub(crate) blobs_start: usize,
    /// Full serialized length (end of the blob region).
    pub(crate) full_total: usize,
}

/// Compute payload section offsets from the index length and total blob bytes.
pub(crate) fn payload_layout(
    num_items: usize,
    index_total_len: usize,
    blob_total: usize,
) -> Result<PayloadLayout, LoadError> {
    let offsets_len = num_items
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or(LoadError::IntegerOverflow)?;
    let offsets_start = index_total_len;
    let blobs_start = offsets_start
        .checked_add(offsets_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let full_total = blobs_start
        .checked_add(blob_total)
        .ok_or(LoadError::IntegerOverflow)?;
    Ok(PayloadLayout {
        offsets_start,
        blobs_start,
        full_total,
    })
}

/// Validated slices of the optional trailing payload section. Borrowed by the
/// zero-copy views to serve `payload(id)`.
pub(crate) struct ParsedPayload<'a> {
    /// `(num_items + 1)` little-endian `u64` prefix offsets into `blobs`.
    pub(crate) offsets: &'a [u8],
    /// Concatenated per-item payload bytes.
    pub(crate) blobs: &'a [u8],
}

/// Validate and slice the payload section from a full byte buffer.
/// `index_total_len` is where the index ends and the payload section begins.
pub(crate) fn parse_payload_section(
    bytes: &[u8],
    num_items: usize,
    index_total_len: usize,
) -> Result<ParsedPayload<'_>, LoadError> {
    let offsets_len = num_items
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or(LoadError::IntegerOverflow)?;
    let offsets_end = index_total_len
        .checked_add(offsets_len)
        .ok_or(LoadError::IntegerOverflow)?;
    if bytes.len() < offsets_end {
        return Err(LoadError::Truncated);
    }
    let offsets = &bytes[index_total_len..offsets_end];

    // offsets[0] must be 0 and the table must be non-decreasing.
    let mut prev = 0u64;
    for i in 0..=num_items {
        let off = read_u64_at(offsets, i * 8)?;
        if (i == 0 && off != 0) || off < prev {
            return Err(LoadError::InvalidTree);
        }
        prev = off;
    }
    let blob_total = usize_from_u64(prev)?;
    let full_total = offsets_end
        .checked_add(blob_total)
        .ok_or(LoadError::IntegerOverflow)?;
    if bytes.len() < full_total {
        return Err(LoadError::Truncated);
    }
    if bytes.len() != full_total {
        return Err(LoadError::LengthMismatch {
            expected: full_total,
            actual: bytes.len(),
        });
    }

    Ok(ParsedPayload {
        offsets,
        blobs: &bytes[offsets_end..full_total],
    })
}

/// Slice item `id`'s payload out of a validated offset table and blob region.
#[inline]
pub(crate) fn payload_slice<'a>(offsets: &[u8], blobs: &'a [u8], id: usize) -> &'a [u8] {
    let start = read_u64_le_unchecked(offsets, id * 8) as usize;
    let end = read_u64_le_unchecked(offsets, (id + 1) * 8) as usize;
    &blobs[start..end]
}

pub(crate) struct ParsedIndexBytes<'a> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    pub(crate) level_bounds: &'a [u8],
    pub(crate) entries: &'a [u8],
    pub(crate) indices: &'a [u8],
}

pub(crate) fn parse_index_bytes(bytes: &[u8]) -> Result<ParsedIndexBytes<'_>, LoadError> {
    parse_index_bytes_with_flags(bytes, FORMAT_FLAGS_2D, 2, 8)
}

pub(crate) fn parse_index3d_bytes(bytes: &[u8]) -> Result<ParsedIndexBytes<'_>, LoadError> {
    parse_index_bytes_with_flags(bytes, FORMAT_FLAGS_3D, 3, 8)
}

#[cfg(feature = "f32-storage")]
pub(crate) fn parse_index2d_f32_bytes(bytes: &[u8]) -> Result<ParsedIndexBytes<'_>, LoadError> {
    parse_index_bytes_with_flags(bytes, FORMAT_FLAGS_2D_F32, 2, 4)
}

#[cfg(feature = "f32-storage")]
pub(crate) fn parse_index3d_f32_bytes(bytes: &[u8]) -> Result<ParsedIndexBytes<'_>, LoadError> {
    parse_index_bytes_with_flags(bytes, FORMAT_FLAGS_3D_F32, 3, 4)
}

fn parse_index_bytes_with_flags(
    bytes: &[u8],
    expected_flags: u64,
    dimensions: usize,
    coord_bytes: usize,
) -> Result<ParsedIndexBytes<'_>, LoadError> {
    // Index-only readers do not understand a trailing payload section, so a
    // payload index is rejected (clean) rather than misread.
    let header = parse_and_validate_header(bytes, expected_flags, false)?;
    let layout = section_layout(
        header.level_count,
        header.num_nodes,
        dimensions,
        coord_bytes,
    )?;

    if bytes.len() < layout.total_len {
        return Err(LoadError::Truncated);
    }
    if bytes.len() != layout.total_len {
        return Err(LoadError::LengthMismatch {
            expected: layout.total_len,
            actual: bytes.len(),
        });
    }
    slice_and_validate_index(bytes, &header, &layout)
}

/// Parse the index allowing an optional trailing payload section. Used by the
/// zero-copy views, which can borrow the payload; index-only loaders use
/// [`parse_index_bytes`] and its siblings instead.
pub(crate) fn parse_index_and_payload<'a>(
    bytes: &'a [u8],
    expected_flags: u64,
    dimensions: usize,
    coord_bytes: usize,
) -> Result<(ParsedIndexBytes<'a>, Option<ParsedPayload<'a>>), LoadError> {
    let header = parse_and_validate_header(bytes, expected_flags, true)?;
    let layout = section_layout(
        header.level_count,
        header.num_nodes,
        dimensions,
        coord_bytes,
    )?;

    if bytes.len() < layout.total_len {
        return Err(LoadError::Truncated);
    }
    let parsed = slice_and_validate_index(bytes, &header, &layout)?;

    if header.has_payload {
        let payload = parse_payload_section(bytes, header.num_items, layout.total_len)?;
        Ok((parsed, Some(payload)))
    } else {
        if bytes.len() != layout.total_len {
            return Err(LoadError::LengthMismatch {
                expected: layout.total_len,
                actual: bytes.len(),
            });
        }
        Ok((parsed, None))
    }
}

/// Slice and validate the index sections from `bytes` (which must be at least
/// `layout.total_len` long). Does not check for trailing bytes, so callers add
/// the length check appropriate to whether a payload section may follow.
fn slice_and_validate_index<'a>(
    bytes: &'a [u8],
    header: &HeaderFields,
    layout: &SectionLayout,
) -> Result<ParsedIndexBytes<'a>, LoadError> {
    let parsed = ParsedIndexBytes {
        node_size: header.node_size,
        num_items: header.num_items,
        num_nodes: header.num_nodes,
        level_count: header.level_count,
        level_bounds: &bytes[layout.level_bounds_start..layout.box0],
        entries: &bytes[layout.box0..layout.idx0],
        indices: &bytes[layout.idx0..layout.total_len],
    };
    validate_level_bounds(
        parsed.level_bounds,
        parsed.num_items,
        parsed.num_nodes,
        parsed.node_size,
        parsed.level_count,
    )?;
    validate_indices(&parsed)?;
    Ok(parsed)
}

/// Validate the `level_bounds` section against the declared tree shape. Reads
/// only the (small) `level_bounds` bytes, so the streaming reader can call it at
/// open time without the full buffer.
pub(crate) fn validate_level_bounds(
    level_bounds: &[u8],
    num_items: usize,
    num_nodes: usize,
    node_size: usize,
    level_count: usize,
) -> Result<(), LoadError> {
    let mut n = num_items;
    let mut running_total = n;
    for level in 0..level_count {
        let actual = read_u64_at(level_bounds, level * 8).and_then(usize_from_u64)?;
        if actual != running_total {
            return Err(LoadError::InvalidTree);
        }
        if level + 1 == level_count {
            break;
        }
        if n == 0 {
            return Err(LoadError::InvalidTree);
        }
        n = n.div_ceil(node_size);
        running_total = running_total
            .checked_add(n)
            .ok_or(LoadError::IntegerOverflow)?;
    }
    if read_u64_at(level_bounds, (level_count - 1) * 8).and_then(usize_from_u64)? != num_nodes {
        return Err(LoadError::InvalidTree);
    }
    Ok(())
}

fn validate_indices(parsed: &ParsedIndexBytes<'_>) -> Result<(), LoadError> {
    for pos in 0..parsed.num_items {
        let index = read_u64_at(parsed.indices, pos * 8).and_then(usize_from_u64)?;
        if index >= parsed.num_items {
            return Err(LoadError::InvalidTree);
        }
    }

    for level in 1..parsed.level_count {
        let level_start =
            read_u64_at(parsed.level_bounds, (level - 1) * 8).and_then(usize_from_u64)?;
        let level_end = read_u64_at(parsed.level_bounds, level * 8).and_then(usize_from_u64)?;
        let child_level_start = if level == 1 {
            0
        } else {
            read_u64_at(parsed.level_bounds, (level - 2) * 8).and_then(usize_from_u64)?
        };
        let child_level_end = level_start;

        for pos in level_start..level_end {
            let index = read_u64_at(parsed.indices, pos * 8).and_then(usize_from_u64)?;
            if index < child_level_start || index >= child_level_end {
                return Err(LoadError::InvalidTree);
            }
            if (index - child_level_start) % parsed.node_size != 0 {
                return Err(LoadError::InvalidTree);
            }
        }
    }

    Ok(())
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

pub(crate) fn serialized_len(level_count: usize, num_nodes: usize) -> Result<usize, LoadError> {
    serialized_len_for_dimensions(level_count, num_nodes, 2, 8)
}

pub(crate) fn serialized_len_3d(level_count: usize, num_nodes: usize) -> Result<usize, LoadError> {
    serialized_len_for_dimensions(level_count, num_nodes, 3, 8)
}

#[cfg(feature = "f32-storage")]
pub(crate) fn serialized_len_2d_f32(
    level_count: usize,
    num_nodes: usize,
) -> Result<usize, LoadError> {
    serialized_len_for_dimensions(level_count, num_nodes, 2, 4)
}

#[cfg(feature = "f32-storage")]
pub(crate) fn serialized_len_3d_f32(
    level_count: usize,
    num_nodes: usize,
) -> Result<usize, LoadError> {
    serialized_len_for_dimensions(level_count, num_nodes, 3, 4)
}

fn serialized_len_for_dimensions(
    level_count: usize,
    num_nodes: usize,
    dimensions: usize,
    coord_bytes: usize,
) -> Result<usize, LoadError> {
    let levels = level_count
        .checked_mul(8)
        .ok_or(LoadError::IntegerOverflow)?;
    let entries = num_nodes
        .checked_mul(dimensions)
        .and_then(|len| len.checked_mul(2 * coord_bytes))
        .ok_or(LoadError::IntegerOverflow)?;
    let indices = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;
    FORMAT_HEADER_LEN
        .checked_add(levels)
        .and_then(|len| len.checked_add(entries))
        .and_then(|len| len.checked_add(indices))
        .ok_or(LoadError::IntegerOverflow)
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

    pub(crate) fn write_format_version(&mut self) {
        self.write_u64(FORMAT_VERSION);
    }

    pub(crate) fn write_header_len(&mut self) {
        self.write_u64(FORMAT_HEADER_LEN as u64);
    }

    pub(crate) fn write_flags(&mut self) {
        self.write_u64(FORMAT_FLAGS_2D);
    }

    pub(crate) fn write_3d_flags(&mut self) {
        self.write_u64(FORMAT_FLAGS_3D);
    }

    #[cfg(feature = "f32-storage")]
    pub(crate) fn write_2d_f32_flags(&mut self) {
        self.write_u64(FORMAT_FLAGS_2D_F32);
    }

    #[cfg(feature = "f32-storage")]
    pub(crate) fn write_3d_f32_flags(&mut self) {
        self.write_u64(FORMAT_FLAGS_3D_F32);
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

    /// Write the optional payload section: a `(num_items + 1)` prefix-offset
    /// table followed by the concatenated blobs. `payloads` is in item order.
    pub(crate) fn write_payload_offsets_and_blobs<P: AsRef<[u8]>>(&mut self, payloads: &[P]) {
        let mut acc: u64 = 0;
        self.write_u64(0);
        for payload in payloads {
            acc += payload.as_ref().len() as u64;
            self.write_u64(acc);
        }
        for payload in payloads {
            self.write_bytes(payload.as_ref());
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
    // `parse_index_bytes`; unaligned byte buffers are copied into an aligned u64.
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

    #[test]
    fn payload_round_trip() {
        for &n in &[0usize, 1, 17, 100] {
            let index = build(n);
            let payloads: Vec<Vec<u8>> = (0..n).map(|i| format!("item-{i}").into_bytes()).collect();
            let bytes = index.to_bytes_with_payloads(&payloads).unwrap();

            // The index sections (after the header) are byte-identical to the
            // index-only file; only the header's payload flag bit differs.
            let index_only = index.to_bytes();
            assert_eq!(&bytes[64..index_only.len()], &index_only[64..]);

            let header = parse_and_validate_header(&bytes, FORMAT_FLAGS_2D, true).unwrap();
            assert!(header.has_payload);
            let layout = section_layout(header.level_count, header.num_nodes, 2, 8).unwrap();
            let parsed = parse_payload_section(&bytes, header.num_items, layout.total_len).unwrap();
            for (i, want) in payloads.iter().enumerate() {
                assert_eq!(
                    payload_slice(parsed.offsets, parsed.blobs, i),
                    want.as_slice()
                );
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
        let header = parse_and_validate_header(&bytes, FORMAT_FLAGS_2D, true).unwrap();
        let layout = section_layout(header.level_count, header.num_nodes, 2, 8).unwrap();
        let parsed = parse_payload_section(&bytes, header.num_items, layout.total_len).unwrap();
        for (i, want) in payloads.iter().enumerate() {
            assert_eq!(
                payload_slice(parsed.offsets, parsed.blobs, i),
                want.as_slice()
            );
        }
    }
}
