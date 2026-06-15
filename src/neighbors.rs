use std::collections::BinaryHeap;

use crate::geometry::{Box2D, Box3D, Point2D, Point3D};

/// What a 2D nearest-neighbor traversal measures distance from: a point or a
/// query box (box queries use box-to-box gap distance, `0.0` on overlap).
#[derive(Clone, Copy)]
pub(crate) enum NeighborQuery2D {
    Point(Point2D),
    Box(Box2D),
}

impl NeighborQuery2D {
    #[inline]
    pub(crate) fn distance_squared_to(self, bounds: Box2D) -> f64 {
        match self {
            NeighborQuery2D::Point(point) => bounds.distance_squared_to(point),
            NeighborQuery2D::Box(query) => bounds.distance_squared_to_box(query),
        }
    }
}

/// 3D counterpart of [`NeighborQuery2D`].
#[derive(Clone, Copy)]
pub(crate) enum NeighborQuery3D {
    Point(Point3D),
    Box(Box3D),
}

impl NeighborQuery3D {
    #[inline]
    pub(crate) fn distance_squared_to(self, bounds: Box3D) -> f64 {
        match self {
            NeighborQuery3D::Point(point) => bounds.distance_squared_to(point),
            NeighborQuery3D::Box(query) => bounds.distance_squared_to_box(query),
        }
    }
}

/// Reusable buffers for allocation-free repeated nearest-neighbor searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, NeighborWorkspace, Point2D, Box2D};
///
/// let mut builder = Index2DBuilder::new(2);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// builder.add(Box2D::new(10.0, 10.0, 11.0, 11.0));
/// let index = builder.finish().unwrap();
///
/// let mut workspace = NeighborWorkspace::new();
/// let hits = index.neighbors_with(Point2D::new(0.5, 0.5), 1, f64::INFINITY, &mut workspace);
/// assert_eq!(hits, &[0]);
/// assert_eq!(workspace.results(), &[0]);
/// ```
#[derive(Debug, Default)]
pub struct NeighborWorkspace {
    pub(crate) results: Vec<usize>,
    pub(crate) queue: BinaryHeap<NeighborState>,
    pub(crate) node_queue: BinaryHeap<NeighborNodeState>,
    #[cfg(all(feature = "f32-storage", feature = "simd"))]
    pub(crate) exact_queue: BinaryHeap<ExactNeighborState>,
}

impl NeighborWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a workspace with preallocated result and priority-queue capacity.
    pub fn with_capacity(results: usize, queue: usize) -> Self {
        Self {
            results: Vec::with_capacity(results),
            queue: BinaryHeap::with_capacity(queue),
            node_queue: BinaryHeap::with_capacity(queue),
            #[cfg(all(feature = "f32-storage", feature = "simd"))]
            exact_queue: BinaryHeap::with_capacity(results),
        }
    }

    /// Results from the latest nearest-neighbor search that used this workspace.
    pub fn results(&self) -> &[usize] {
        &self.results
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NeighborState {
    pub(crate) index: usize,
    pub(crate) is_leaf: bool,
    pub(crate) dist: f64,
}

impl NeighborState {
    #[inline]
    pub(crate) fn new(index: usize, is_leaf: bool, dist: f64) -> Self {
        Self {
            index,
            is_leaf,
            dist,
        }
    }
}

impl Eq for NeighborState {}

impl Ord for NeighborState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| self.is_leaf.cmp(&other.is_leaf))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for NeighborState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NeighborNodeState {
    pub(crate) index: usize,
    pub(crate) dist: f64,
}

impl NeighborNodeState {
    #[inline]
    pub(crate) fn new(index: usize, dist: f64) -> Self {
        Self { index, dist }
    }
}

impl Eq for NeighborNodeState {}

impl Ord for NeighborNodeState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for NeighborNodeState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(all(feature = "f32-storage", feature = "simd"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExactNeighborState {
    pub(crate) index: usize,
    pub(crate) dist: f64,
}

#[cfg(all(feature = "f32-storage", feature = "simd"))]
impl ExactNeighborState {
    #[inline]
    pub(crate) fn new(index: usize, dist: f64) -> Self {
        Self { index, dist }
    }
}

#[cfg(all(feature = "f32-storage", feature = "simd"))]
impl Eq for ExactNeighborState {}

#[cfg(all(feature = "f32-storage", feature = "simd"))]
impl Ord for ExactNeighborState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.index.cmp(&other.index))
    }
}

#[cfg(all(feature = "f32-storage", feature = "simd"))]
impl PartialOrd for ExactNeighborState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn max_distance_squared(max_distance: f64) -> Option<f64> {
    if max_distance.is_nan() || max_distance.is_sign_negative() {
        None
    } else {
        Some(max_distance * max_distance)
    }
}
