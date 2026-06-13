//! Packed static spatial index for 2D and 3D axis-aligned bounding boxes.
//!
//! The canonical flow is [`Index2DBuilder`] -> [`Index2D`] -> [`Index2D::search`].
//! With the `simd` feature, `Index2DBuilder::finish_simd` and
//! `Index3DBuilder::finish_simd` build `SimdIndex2D` and `SimdIndex3D`, which
//! share the same query API backed by SoA layouts and SIMD traversal.
//! Indexes also serialize with `to_bytes` and load back as owned indexes or as
//! zero-copy views ([`Index2DView`], [`Index3DView`]).
//!
//! # Queries
//!
//! Every query is a method on the index types ([`Index2D`] / [`Index3D`], the
//! SIMD indexes, and the zero-copy views) — the crate has no free functions, so
//! docs.rs lists these on each type's page (e.g. [`Index2D`]) rather than in a
//! crate-level "Functions" section. Range and ray results are item indices in
//! insertion order.
//!
//! * **Range / intersection** — [`search`](Index2D::search) (plus `search_into`
//!   / `search_with` / lazy [`search_iter`](Index2D::search_iter)),
//!   [`any`](Index2D::any), [`first`](Index2D::first),
//!   [`visit`](Index2D::visit).
//! * **Nearest neighbors** — from a point [`neighbors`](Index2D::neighbors)
//!   (plus `_within` / `_into` / `_with` /
//!   [`visit_neighbors`](Index2D::visit_neighbors)) or from a box
//!   [`neighbors_of_box`](Index2D::neighbors_of_box) and its variants.
//! * **Ray segment** — [`raycast`](Index2D::raycast) (all hits),
//!   [`raycast_closest`](Index2D::raycast_closest) (nearest box entered), and
//!   [`visit_raycast`](Index2D::visit_raycast).
//! * **Spatial join** — [`join`](Index2D::join) /
//!   [`join_with`](Index2D::join_with) between two indexes,
//!   [`self_join`](Index2D::self_join) /
//!   [`self_join_with`](Index2D::self_join_with) within one.
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
//! * `parallel` (default): adaptive parallel builds through rayon.
//! * `simd` (default): SIMD search/raycast through `wide` and x86-64 AVX-512.
//! * `f32-storage`: compact f32-storage SIMD indexes.
//! * `stream`: query a serialized index over a `RangeReader` (local file or
//!   remote object) without loading the whole file. No extra dependencies.

// On docs.rs (built with `--cfg docsrs` on nightly), auto-render "Available on
// crate feature X" badges for feature-gated items from their `#[cfg]`s.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

// Compile the README's Rust examples as doctests (so they cannot rot) without
// making the README the docs.rs landing page. This keeps the README's in-repo
// links relative — they resolve on GitHub and crates.io, which render the
// README; docs.rs renders the crate-level docs above instead.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

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
#[cfg(feature = "stream")]
mod stream;
mod traversal;
mod tree;

pub use build::BuildError;
pub use builder2d::Index2DBuilder;
pub use builder3d::Index3DBuilder;
pub use config::DEFAULT_NODE_SIZE;
#[cfg(feature = "parallel")]
pub use config::DEFAULT_PARALLEL_MIN_ITEMS;
pub use geometry::{BoundsError, Box2D, Box3D, Point2D, Point3D};
pub use index2d::{Index2D, Index2DView, Search2DIter};
#[cfg(feature = "f32-storage")]
pub use index2d_f32::{SimdIndex2DF32, SimdIndex2DF32View};
#[cfg(feature = "simd")]
pub use index2d_soa::{SimdIndex2D, SimdIndex2DView};
pub use index3d::{Index3D, Index3DView, Search3DIter};
#[cfg(feature = "f32-storage")]
pub use index3d_f32::{SimdIndex3DF32, SimdIndex3DF32View};
#[cfg(feature = "simd")]
pub use index3d_soa::{SimdIndex3D, SimdIndex3DView};
pub use neighbors::NeighborWorkspace;
pub use persistence::{LoadError, PayloadError};
pub use ray::{Ray2D, Ray3D};
pub use sort2d::SortKey2D;
pub use sort3d::SortKey3D;
#[cfg(all(feature = "stream", any(unix, windows)))]
pub use stream::FileReader;
#[cfg(feature = "stream")]
pub use stream::{RangeReader, SliceReader, StreamError, StreamIndex2D, StreamIndex3D};
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
