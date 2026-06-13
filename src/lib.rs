// The crate-level docs (the docs.rs landing page) are the README, so its Rust
// examples are compiled as doctests and cannot rot. README links to in-repo
// files use absolute URLs so they resolve on docs.rs too.
#![doc = include_str!("../README.md")]
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
mod ray;
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
pub use ray::{Ray2D, Ray3D};
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
