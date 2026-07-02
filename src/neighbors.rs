use std::collections::BinaryHeap;

use crate::geometry::{Box2D, Box3D, Point2D, Point3D};

/// Mean Earth radius in meters (IUGG), a reasonable default for
/// [`haversine_distance_2d`].
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Great-circle (haversine) distance in meters from a `(lon, lat)` query point
/// (degrees) to the closest point of `bounds`, a lon/lat AABB in degrees.
///
/// Drop this into a [`neighbors_metric`](crate::Index2D::neighbors_metric)
/// closure for geographic nearest-neighbor queries (`x` = longitude,
/// `y` = latitude). `earth_radius_m` is usually [`EARTH_RADIUS_M`].
///
/// The closest point is the per-axis clamp of the query onto the box — exact for
/// the small boxes typical of feature data. For a very large or near-polar box it
/// is a slight *over*-estimate of the true geodesic minimum, so a marginally
/// closer item in such a box could be missed; inflate the boxes there if you need
/// a strict bound.
pub fn haversine_distance_2d(query_lon_lat: (f64, f64), bounds: Box2D, earth_radius_m: f64) -> f64 {
    let (lon, lat) = query_lon_lat;
    let clamped_lon = lon.clamp(bounds.min_x, bounds.max_x);
    let clamped_lat = lat.clamp(bounds.min_y, bounds.max_y);
    let lat1 = lat.to_radians();
    let lat2 = clamped_lat.to_radians();
    let half_dlat = (clamped_lat - lat).to_radians() * 0.5;
    let half_dlon = (clamped_lon - lon).to_radians() * 0.5;
    let a = half_dlat.sin().powi(2) + lat1.cos() * lat2.cos() * half_dlon.sin().powi(2);
    2.0 * earth_radius_m * a.sqrt().asin()
}

/// What a 2D nearest-neighbor traversal measures distance from: a point or a
/// query box (box queries use box-to-box gap distance, `0.0` on overlap).
#[derive(Clone, Copy)]
pub(crate) enum NeighborQuery2D {
    Point(Point2D),
    Box(Box2D),
}

impl NeighborQuery2D {
    #[inline]
    pub(crate) fn is_valid(self) -> bool {
        match self {
            NeighborQuery2D::Point(point) => point.x.is_finite() && point.y.is_finite(),
            NeighborQuery2D::Box(_) => true,
        }
    }

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
    pub(crate) fn is_valid(self) -> bool {
        match self {
            NeighborQuery3D::Point(point) => {
                point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
            }
            NeighborQuery3D::Box(_) => true,
        }
    }

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
    #[cfg(feature = "f32-storage")]
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
            #[cfg(feature = "f32-storage")]
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

#[cfg(feature = "f32-storage")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExactNeighborState {
    pub(crate) index: usize,
    pub(crate) dist: f64,
}

#[cfg(feature = "f32-storage")]
impl ExactNeighborState {
    #[inline]
    pub(crate) fn new(index: usize, dist: f64) -> Self {
        Self { index, dist }
    }
}

#[cfg(feature = "f32-storage")]
impl Eq for ExactNeighborState {}

#[cfg(feature = "f32-storage")]
impl Ord for ExactNeighborState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.index.cmp(&other.index))
    }
}

#[cfg(feature = "f32-storage")]
impl PartialOrd for ExactNeighborState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn max_distance_squared(max_distance: f64) -> Option<f64> {
    if max_distance.is_nan() || max_distance < 0.0 {
        None
    } else {
        Some(max_distance * max_distance)
    }
}

/// A finite or `+inf` distance cutoff in the metric's own units (no squaring).
/// `None` (nan or negative) means "match nothing".
pub(crate) fn valid_max_distance(max_distance: f64) -> Option<f64> {
    if max_distance.is_nan() || max_distance < 0.0 {
        None
    } else {
        Some(max_distance)
    }
}

pub(crate) mod best_first;
pub(crate) mod metric_knn;

#[cfg(feature = "f32-storage")]
pub(crate) use best_first::level_end_of;

#[cfg(feature = "f32-storage")]
mod f32_knn;

#[cfg(feature = "f32-storage")]
pub(crate) use f32_knn::*;
