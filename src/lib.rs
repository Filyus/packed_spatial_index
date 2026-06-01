//! Packed static spatial index for 2D axis-aligned bounding boxes.
//!
//! The canonical flow is [`IndexBuilder`] -> [`Index`] -> [`Index::search`].
//! With the `simd` feature, `IndexBuilder::finish_simd` builds `SimdIndex`,
//! which has the same search API backed by a SoA layout and SIMD traversal.
//! `Index` can also be serialized with [`Index::to_bytes`] and viewed without
//! copying through [`IndexView`].
//!
//! # Quick Start
//! ```
//! use packed_spatial_index::{IndexBuilder, Rect};
//!
//! let mut builder = IndexBuilder::new(2);
//! builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
//! builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));
//! let index = builder.finish().unwrap();
//!
//! assert_eq!(index.search(Rect::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
//! ```
//!
//! # Cargo Features
//! * `parallel` (enabled by default): adaptive parallel builds through rayon.
//! * `simd` (enabled by default): `SimdIndex` with SIMD searches through `wide`
//!   and x86-64 AVX-512 intrinsics where available.

mod builder;
mod config;
mod geometry;
mod hilbert;
mod index;
#[cfg(feature = "simd")]
mod index_soa;
mod neighbors;
mod persistence;
mod sort;

pub use builder::{BuildError, IndexBuilder};
pub use config::DEFAULT_NODE_SIZE;
#[cfg(feature = "parallel")]
pub use config::DEFAULT_PARALLEL_MIN_ITEMS;
pub use geometry::{Point, Rect, RectError};
pub use index::{Index, IndexView, SearchWorkspace};
#[cfg(feature = "simd")]
pub use index_soa::SimdIndex;
pub use neighbors::NeighborWorkspace;
pub use persistence::LoadError;
pub use sort::SortKey;

/// Experimental internals kept public for benchmarks and research notebooks.
#[doc(hidden)]
pub mod experimental {
    pub use crate::hilbert::{
        ENCODERS, HilbertFn, loop_rotation, lut, magic_bits, magic_bits_batch, morton,
    };
    pub use crate::sort::{ExperimentalSortKey, radix_sort_pairs};
}
