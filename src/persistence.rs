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

// Packed Spatial Index file signature. The format version is stored separately
// as a little-endian u64 in the header.
const FORMAT_MAGIC: &[u8; 8] = b"PSINDEX\0";
const FORMAT_VERSION: u64 = 1;
const FORMAT_FLAGS_2D: u64 = 0;
const FORMAT_FLAGS_3D: u64 = 1;
const FORMAT_HEADER_LEN: usize = 64;

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
    parse_index_bytes_with_flags(bytes, FORMAT_FLAGS_2D, 2)
}

pub(crate) fn parse_index3d_bytes(bytes: &[u8]) -> Result<ParsedIndexBytes<'_>, LoadError> {
    parse_index_bytes_with_flags(bytes, FORMAT_FLAGS_3D, 3)
}

fn parse_index_bytes_with_flags(
    bytes: &[u8],
    expected_flags: u64,
    dimensions: usize,
) -> Result<ParsedIndexBytes<'_>, LoadError> {
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
    if header_len != FORMAT_HEADER_LEN || flags != expected_flags {
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

    let expected_len = serialized_len_for_dimensions(level_count, num_nodes, dimensions)?;
    if bytes.len() < expected_len {
        return Err(LoadError::Truncated);
    }
    if bytes.len() != expected_len {
        return Err(LoadError::LengthMismatch {
            expected: expected_len,
            actual: bytes.len(),
        });
    }

    let level_bounds_len = level_count
        .checked_mul(8)
        .ok_or(LoadError::IntegerOverflow)?;
    let entries_len = num_nodes
        .checked_mul(dimensions)
        .and_then(|len| len.checked_mul(16))
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_len = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;

    let level_start = FORMAT_HEADER_LEN;
    let entries_start = level_start
        .checked_add(level_bounds_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_start = entries_start
        .checked_add(entries_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let end = indices_start
        .checked_add(indices_len)
        .ok_or(LoadError::IntegerOverflow)?;

    let parsed = ParsedIndexBytes {
        node_size,
        num_items,
        num_nodes,
        level_count,
        level_bounds: &bytes[level_start..entries_start],
        entries: &bytes[entries_start..indices_start],
        indices: &bytes[indices_start..end],
    };
    validate_level_bounds(&parsed)?;
    validate_indices(&parsed)?;
    Ok(parsed)
}

fn validate_level_bounds(parsed: &ParsedIndexBytes<'_>) -> Result<(), LoadError> {
    let mut n = parsed.num_items;
    let mut running_total = n;
    for level in 0..parsed.level_count {
        let actual = read_u64_at(parsed.level_bounds, level * 8).and_then(usize_from_u64)?;
        if actual != running_total {
            return Err(LoadError::InvalidTree);
        }
        if level + 1 == parsed.level_count {
            break;
        }
        if n == 0 {
            return Err(LoadError::InvalidTree);
        }
        n = n.div_ceil(parsed.node_size);
        running_total = running_total
            .checked_add(n)
            .ok_or(LoadError::IntegerOverflow)?;
    }
    if read_u64_at(parsed.level_bounds, (parsed.level_count - 1) * 8).and_then(usize_from_u64)?
        != parsed.num_nodes
    {
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

fn expected_tree_shape(num_items: usize, node_size: usize) -> Result<(usize, usize), LoadError> {
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
    serialized_len_for_dimensions(level_count, num_nodes, 2)
}

pub(crate) fn serialized_len_3d(level_count: usize, num_nodes: usize) -> Result<usize, LoadError> {
    serialized_len_for_dimensions(level_count, num_nodes, 3)
}

fn serialized_len_for_dimensions(
    level_count: usize,
    num_nodes: usize,
    dimensions: usize,
) -> Result<usize, LoadError> {
    let levels = level_count
        .checked_mul(8)
        .ok_or(LoadError::IntegerOverflow)?;
    let entries = num_nodes
        .checked_mul(dimensions)
        .and_then(|len| len.checked_mul(16))
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

fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, LoadError> {
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

fn usize_from_u64(value: u64) -> Result<usize, LoadError> {
    usize::try_from(value).map_err(|_| LoadError::IntegerOverflow)
}
