use std::{error::Error, fmt};

/// Build error for finishing an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
    /// An added box has `min > max` on some axis, or a `NaN` bound.
    ///
    /// The unchecked `Box2D::new` / `Box3D::new` accept these; `try_new` does not.
    /// A box with crossed bounds covers no region, yet it is *contained* by queries
    /// it does not overlap, so a tree holding one answers the same query differently
    /// depending on whether the search descends to it or takes the whole-subtree
    /// shortcut. Rejecting it here is what keeps every search path in agreement.
    InvalidItemBounds {
        /// Position of the offending box in the order it was added.
        at: usize,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::ItemCount { added, expected } => write!(
                f,
                "added item count must match declared count (added {added}, expected {expected})"
            ),
            BuildError::TreeTooLarge => write!(f, "packed tree is too large"),
            BuildError::InvalidItemBounds { at } => write!(
                f,
                "item {at} has crossed or NaN bounds (min > max); build from `Box::try_new` bounds"
            ),
        }
    }
}

impl Error for BuildError {}
