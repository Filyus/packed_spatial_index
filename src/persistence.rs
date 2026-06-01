use std::{error::Error, fmt};

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
const FORMAT_FLAGS: u64 = 0;
const FORMAT_HEADER_LEN: usize = 64;

pub(crate) struct ParsedIndexBytes<'a> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    pub(crate) level_bounds: &'a [u8],
    pub(crate) boxes: &'a [u8],
    pub(crate) indices: &'a [u8],
}

pub(crate) fn parse_index_bytes(bytes: &[u8]) -> Result<ParsedIndexBytes<'_>, LoadError> {
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
    if header_len != FORMAT_HEADER_LEN || flags != FORMAT_FLAGS {
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

    let expected_len = serialized_len(level_count, num_nodes)?;
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
    let boxes_len = num_nodes
        .checked_mul(32)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_len = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;

    let level_start = FORMAT_HEADER_LEN;
    let boxes_start = level_start
        .checked_add(level_bounds_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_start = boxes_start
        .checked_add(boxes_len)
        .ok_or(LoadError::IntegerOverflow)?;
    let end = indices_start
        .checked_add(indices_len)
        .ok_or(LoadError::IntegerOverflow)?;

    let parsed = ParsedIndexBytes {
        node_size,
        num_items,
        num_nodes,
        level_count,
        level_bounds: &bytes[level_start..boxes_start],
        boxes: &bytes[boxes_start..indices_start],
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
    let levels = level_count
        .checked_mul(8)
        .ok_or(LoadError::IntegerOverflow)?;
    let boxes = num_nodes
        .checked_mul(32)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;
    FORMAT_HEADER_LEN
        .checked_add(levels)
        .and_then(|len| len.checked_add(boxes))
        .and_then(|len| len.checked_add(indices))
        .ok_or(LoadError::IntegerOverflow)
}

pub(crate) fn push_magic(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(FORMAT_MAGIC);
}

pub(crate) fn push_format_version(bytes: &mut Vec<u8>) {
    push_u64(bytes, FORMAT_VERSION);
}

pub(crate) fn push_header_len(bytes: &mut Vec<u8>) {
    push_u64(bytes, FORMAT_HEADER_LEN as u64);
}

pub(crate) fn push_flags(bytes: &mut Vec<u8>) {
    push_u64(bytes, FORMAT_FLAGS);
}

pub(crate) fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_le_bytes());
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
