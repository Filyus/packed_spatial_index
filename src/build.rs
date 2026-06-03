use std::{error::Error, fmt};

/// Build error for finishing an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The builder received the wrong number of items.
    ItemCount {
        /// Number actually added through `add`.
        added: usize,
        /// Expected by `Index*Builder::new(count)`.
        expected: usize,
    },
    /// The requested item count would overflow the packed tree layout.
    TreeTooLarge,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::ItemCount { added, expected } => write!(
                f,
                "added item count must match declared count (added {added}, expected {expected})"
            ),
            BuildError::TreeTooLarge => write!(f, "packed tree is too large"),
        }
    }
}

impl Error for BuildError {}
