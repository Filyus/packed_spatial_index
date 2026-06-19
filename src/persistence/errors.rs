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
