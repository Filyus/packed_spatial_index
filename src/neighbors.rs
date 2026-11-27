use std::collections::BinaryHeap;

use crate::geometry::{Box2D, Box3D, Point2D, Point3D};

/// Mean Earth radius in meters (IUGG), a reasonable default for
/// [`haversine_distance_2d`].
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Exact great-circle (haversine) distance in meters from a `(lon, lat)` query
/// point (degrees) to the closest point of `bounds`, a lon/lat AABB in degrees.
///
/// Drop this into a [`neighbors_metric`](crate::Index2D::neighbors_metric)
/// closure for geographic nearest-neighbor queries (`x` = longitude,
/// `y` = latitude). `earth_radius_m` is usually [`EARTH_RADIUS_M`].
///
/// Uses the exact closed-form minimum over the lon/lat box on the sphere
/// (accounting for 360° longitude periodicity and the true meridian optimum
/// `atan2(sin(lat1), cos(lat1) * cos(dlon))`). Near the antipode the
/// supplementary haversine is evaluated independently to preserve
/// distance resolution.
pub fn haversine_distance_2d(query_lon_lat: (f64, f64), bounds: Box2D, earth_radius_m: f64) -> f64 {
    let (lon, lat) = query_lon_lat;
    let span = bounds.max_x - bounds.min_x;
    let dlon_deg = if span >= 360.0 {
        0.0
    } else {
        // A signed remainder preserves tiny negative offsets at the lower edge.
        let offset = (lon - bounds.min_x) % 360.0;
        if (0.0..=span).contains(&offset) || offset <= span - 360.0 {
            0.0
        } else {
            let start = offset.abs();
            let end = ((lon - bounds.max_x) % 360.0).abs();
            start.min(360.0 - start).min(end.min(360.0 - end))
        }
    };

    let lat1 = lat.to_radians();
    let dlon = dlon_deg.to_radians();

    if dlon == 0.0 {
        let clamped_lat = lat.clamp(bounds.min_y, bounds.max_y);
        return earth_radius_m * (clamped_lat - lat).abs().to_radians();
    }

    let (sin_lat1, cos_lat1) = lat1.sin_cos();
    let sin_half_dlon_sq = (dlon * 0.5).sin().powi(2);
    let central_angle = |lat2: f64| -> f64 {
        let half_dlat = (lat2 - lat).to_radians() * 0.5;
        let cos_lats = if lat.abs() == 90.0 || lat2.abs() == 90.0 {
            0.0
        } else {
            cos_lat1 * lat2.to_radians().cos()
        };
        let hav = half_dlat.sin().powi(2) + cos_lats * sin_half_dlon_sq;
        let complement = if hav <= 0.5 {
            // Here the subtraction retains at least half the operand's scale.
            1.0 - hav
        } else {
            let half_sum = (lat + lat2).to_radians() * 0.5;
            let half_supplement = (180.0 - dlon_deg).to_radians() * 0.5;
            half_sum.sin().powi(2) + cos_lats * half_supplement.sin().powi(2)
        };
        2.0 * hav.sqrt().atan2(complement.sqrt())
    };

    let mut best = central_angle(bounds.min_y);
    if bounds.min_y == bounds.max_y {
        return earth_radius_m * best;
    }
    best = best.min(central_angle(bounds.max_y));
    let cos_dlon = dlon.cos();
    if cos_dlon > 0.0 {
        let phi_opt = sin_lat1.atan2(cos_lat1 * cos_dlon).to_degrees();
        if phi_opt > bounds.min_y && phi_opt < bounds.max_y {
            best = best.min(central_angle(phi_opt));
        }
    }
    earth_radius_m * best
}

/// Depth of the closest point of `bounds` along a view axis: the smallest
/// `dot(p - eye, direction)` over every point `p` of the box.
///
/// The ready-made ordering key for a front-to-back
/// [`search_ordered`](crate::Index3D::search_ordered): pair it with a
/// [`Frustum3D`](crate::Frustum3D) and items come out near-to-far, so a caller
/// on a budget stops early instead of collecting and sorting everything.
///
/// Admissible by construction — a node box encloses its children, so its depth
/// never exceeds theirs. `direction` need not be normalized: a longer vector
/// only rescales the key (and any `max_key` cutoff), it never changes the
/// order. Boxes behind the eye simply get negative depths.
pub fn view_depth_3d(eye: [f64; 3], direction: [f64; 3], bounds: Box3D) -> f64 {
    let mins = [bounds.min_x, bounds.min_y, bounds.min_z];
    let maxs = [bounds.max_x, bounds.max_y, bounds.max_z];
    let mut depth = 0.0;
    for axis in 0..3 {
        let d = direction[axis];
        let near = if d >= 0.0 { mins[axis] } else { maxs[axis] };
        depth += (near - eye[axis]) * d;
    }
    depth
}

/// 2D counterpart of [`view_depth_3d`].
pub fn view_depth_2d(eye: [f64; 2], direction: [f64; 2], bounds: Box2D) -> f64 {
    let mins = [bounds.min_x, bounds.min_y];
    let maxs = [bounds.max_x, bounds.max_y];
    let mut depth = 0.0;
    for axis in 0..2 {
        let d = direction[axis];
        let near = if d >= 0.0 { mins[axis] } else { maxs[axis] };
        depth += (near - eye[axis]) * d;
    }
    depth
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
            NeighborQuery2D::Box(bx) => {
                !bx.min_x.is_nan() && !bx.min_y.is_nan() && !bx.max_x.is_nan() && !bx.max_y.is_nan()
            }
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
            NeighborQuery3D::Box(bx) => {
                !bx.min_x.is_nan()
                    && !bx.min_y.is_nan()
                    && !bx.min_z.is_nan()
                    && !bx.max_x.is_nan()
                    && !bx.max_y.is_nan()
                    && !bx.max_z.is_nan()
            }
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExactNeighborState {
    pub(crate) index: usize,
    pub(crate) dist: f64,
}

impl ExactNeighborState {
    #[inline]
    pub(crate) fn new(index: usize, dist: f64) -> Self {
        Self { index, dist }
    }
}

impl Eq for ExactNeighborState {}

impl Ord for ExactNeighborState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for ExactNeighborState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Insert `(index, dist)` into a bounded max-heap of size `max_results`,
/// tightening `cutoff` whenever the k-th worst distance decreases.
/// Returns `true` if `cutoff` was updated.
#[inline]
pub(crate) fn push_top_k(
    top_k: &mut BinaryHeap<ExactNeighborState>,
    max_results: usize,
    index: usize,
    dist: f64,
    cutoff: &mut f64,
) -> bool {
    if top_k.len() < max_results {
        top_k.push(ExactNeighborState::new(index, dist));
        if top_k.len() == max_results {
            *cutoff = top_k.peek().unwrap().dist;
            return true;
        }
        false
    } else {
        let worst = top_k.peek().unwrap();
        if dist < worst.dist || (dist == worst.dist && index < worst.index) {
            *top_k.peek_mut().unwrap() = ExactNeighborState::new(index, dist);
            *cutoff = top_k.peek().unwrap().dist;
            true
        } else {
            false
        }
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
