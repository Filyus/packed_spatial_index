//! Packed static spatial index for 2D axis-aligned bounding boxes.
//!
//! The canonical flow is [`Index2DBuilder`] -> [`Index2D`] -> [`Index2D::search`].
//! With the `simd` feature, `Index2DBuilder::finish_simd` builds `SimdIndex2D`,
//! which has the same search API backed by a SoA layout and SIMD traversal.
//! `Index2D` can also be serialized with [`Index2D::to_bytes`] and viewed without
//! copying through [`Index2DView`].
//!
//! # Quick Start
//! ```
//! use packed_spatial_index::{Index2DBuilder, Bounds2D};
//!
//! let mut builder = Index2DBuilder::new(2);
//! builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
//! builder.add(Bounds2D::new(5.0, 5.0, 6.0, 6.0));
//! let index = builder.finish().unwrap();
//!
//! assert_eq!(index.search(Bounds2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
//! ```
//!
//! # Cargo Features
//! * `parallel` (enabled by default): adaptive parallel builds through rayon.
//! * `simd` (enabled by default): `SimdIndex2D` with SIMD searches through `wide`
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

pub use builder::{BuildError, Index2DBuilder};
pub use config::DEFAULT_NODE_SIZE;
#[cfg(feature = "parallel")]
pub use config::DEFAULT_PARALLEL_MIN_ITEMS;
pub use geometry::{Bounds2D, BoundsError, Point2D};
pub use index::{Index2D, Index2DView, SearchWorkspace};
#[cfg(feature = "simd")]
pub use index_soa::SimdIndex2D;
pub use neighbors::NeighborWorkspace;
pub use persistence::LoadError;
pub use sort::SortKey2D;

/// Experimental internals kept public for benchmarks and research notebooks.
#[doc(hidden)]
pub mod experimental {
    pub use crate::hilbert::{
        ENCODERS, HilbertFn, loop_rotation, lut, magic_bits, magic_bits_batch, morton,
    };
    pub use crate::sort::{ExperimentalSortKey2D, radix_sort_pairs};
}
