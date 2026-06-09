//! Packed static spatial index for 2D and 3D axis-aligned bounding boxes.
//!
//! The canonical flow is [`Index2DBuilder`] -> [`Index2D`] -> [`Index2D::search`].
//! With the `simd` feature, `Index2DBuilder::finish_simd` and
//! `Index3DBuilder::finish_simd` build `SimdIndex2D` and `SimdIndex3D`, which
//! have the same search APIs backed by SoA layouts and SIMD traversal.
//! `Index2D`, `Index3D`, and their SIMD counterparts can also be serialized
//! with `to_bytes`; scalar indexes can be viewed without copying through
//! [`Index2DView`] and [`Index3DView`].
//!
//! # Quick Start
//! ```
//! use packed_spatial_index::{Index2DBuilder, Box2D};
//!
//! let mut builder = Index2DBuilder::new(2);
//! builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
//! builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
//! let index = builder.finish().unwrap();
//!
//! assert_eq!(index.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
//! ```
//!
//! # Cargo Features
//! * `parallel` (enabled by default): adaptive parallel builds through rayon.
//! * `simd` (enabled by default): SIMD searches through `wide` and x86-64
//!   AVX-512 intrinsics where available.
//! * `f32-storage`: compact f32-storage SIMD indexes.

// On docs.rs (built with `--cfg docsrs` on nightly), auto-render "Available on
// crate feature X" badges for feature-gated items from their `#[cfg]`s.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

mod build;
mod builder2d;
mod builder3d;
mod config;
mod geometry;
mod hilbert2d;
mod index2d;
#[cfg(feature = "f32-storage")]
mod index2d_f32;
#[cfg(feature = "simd")]
mod index2d_soa;
mod index3d;
#[cfg(feature = "f32-storage")]
mod index3d_f32;
#[cfg(feature = "simd")]
mod index3d_soa;
mod join;
mod neighbors;
mod persistence;
mod sort2d;
mod sort3d;
mod traversal;
mod tree;

pub use build::BuildError;
pub use builder2d::Index2DBuilder;
pub use builder3d::Index3DBuilder;
pub use config::DEFAULT_NODE_SIZE;
#[cfg(feature = "parallel")]
pub use config::DEFAULT_PARALLEL_MIN_ITEMS;
pub use geometry::{BoundsError, Box2D, Box3D, Point2D, Point3D};
pub use index2d::{Index2D, Index2DView};
#[cfg(feature = "f32-storage")]
pub use index2d_f32::{SimdIndex2DF32, SimdIndex2DF32View};
#[cfg(feature = "simd")]
pub use index2d_soa::{SimdIndex2D, SimdIndex2DView};
pub use index3d::{Index3D, Index3DView};
#[cfg(feature = "f32-storage")]
pub use index3d_f32::{SimdIndex3DF32, SimdIndex3DF32View};
#[cfg(feature = "simd")]
pub use index3d_soa::{SimdIndex3D, SimdIndex3DView};
pub use neighbors::NeighborWorkspace;
pub use persistence::LoadError;
pub use sort2d::SortKey2D;
pub use sort3d::SortKey3D;
pub use traversal::SearchWorkspace;

/// Internal helpers exposed only for crate benchmarks and local performance tools.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod benchmark_support {
    pub use crate::hilbert2d::{
        ENCODERS, HilbertFn, loop_rotation, lut, magic_bits, magic_bits_batch, morton,
    };
    pub use crate::sort2d::{SortKey2DStrategy, radix_sort_pairs};
    pub use crate::sort3d::{
        SortKey3DStrategy, encode_hilbert3_nibble_lut, encode_hilbert3_pair_lut, encode_morton3,
        radix_sort_pairs_u64,
    };
}
