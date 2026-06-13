//! Finite ray segments for ray/AABB traversal over the packed indexes.
//!
//! A ray is defined by an origin, an unnormalized direction, and a maximum ray
//! parameter `max_distance` (the segment covers `origin + t * dir` for
//! `t in [0, max_distance]`). Box intersections use the standard slab test with
//! precomputed reciprocal directions; axis-parallel rays (a direction component
//! that is exactly zero) are handled explicitly so a ray lying exactly on a box
//! face still hits.

use crate::geometry::{Box2D, Box3D, Point2D, Point3D};

/// Finite 2D ray segment used by `raycast` searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box2D, Point2D, Ray2D};
///
/// let ray = Ray2D::new(Point2D::new(-1.0, 0.5), 1.0, 0.0, 10.0);
/// assert!(ray.intersects_box(Box2D::new(0.0, 0.0, 1.0, 1.0)));
/// assert_eq!(ray.enter_t(Box2D::new(0.0, 0.0, 1.0, 1.0)), Some(1.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray2D {
    /// Ray origin.
    pub origin: Point2D,
    /// X component of the ray direction.
    pub dir_x: f64,
    /// Y component of the ray direction.
    pub dir_y: f64,
    // Precomputed to remove one division from every slab test.
    pub(crate) inv_dir_x: f64,
    pub(crate) inv_dir_y: f64,
    /// Maximum ray parameter to consider.
    pub max_distance: f64,
}

impl Ray2D {
    /// Create a finite ray segment covering `origin + t * dir` for
    /// `t in [0, max_distance]`.
    ///
    /// The direction does not need to be normalized. `max_distance` and every
    /// returned entry `t` are in units of the direction length, so the
    /// Euclidean distance to a hit is `t * hypot(dir_x, dir_y)`; normalize the
    /// direction (length 1) if you want `t` and `max_distance` in world units.
    ///
    /// A fully zero direction (`dir_x == dir_y == 0.0`) is a point probe: it
    /// hits only boxes that contain `origin`, all at `t == 0.0`. Direction
    /// components should be finite; a `NaN` direction produces unspecified
    /// results.
    #[inline]
    pub const fn new(origin: Point2D, dir_x: f64, dir_y: f64, max_distance: f64) -> Self {
        Self {
            origin,
            dir_x,
            dir_y,
            inv_dir_x: 1.0 / dir_x,
            inv_dir_y: 1.0 / dir_y,
            max_distance,
        }
    }

    /// `true` if any direction component is exactly zero (an axis-parallel ray). The
    /// vectorized slab test uses `1/dir = inf` and is not NaN-safe at a box face, so
    /// such rays take a masked path.
    #[inline]
    pub(crate) fn has_zero_direction(self) -> bool {
        self.dir_x == 0.0 || self.dir_y == 0.0
    }

    /// `true` when the ray segment touches `bounds` (edges inclusive).
    #[inline]
    pub fn intersects_box(self, bounds: Box2D) -> bool {
        if self.max_distance < 0.0 || self.max_distance.is_nan() {
            return false;
        }
        let mut t_min: f64 = 0.0;
        let mut t_max = self.max_distance;
        slab(
            self.origin.x,
            self.dir_x,
            self.inv_dir_x,
            bounds.min_x,
            bounds.max_x,
            &mut t_min,
            &mut t_max,
        ) && slab(
            self.origin.y,
            self.dir_y,
            self.inv_dir_y,
            bounds.min_y,
            bounds.max_y,
            &mut t_min,
            &mut t_max,
        )
    }

    /// Entry parameter `t` where the ray segment first touches `bounds` (`0.0` if the
    /// origin is inside), or `None` if the segment misses. Used by ordered closest-hit
    /// traversal.
    #[inline]
    pub fn enter_t(self, bounds: Box2D) -> Option<f64> {
        if self.max_distance < 0.0 || self.max_distance.is_nan() {
            return None;
        }
        let mut t_min: f64 = 0.0;
        let mut t_max = self.max_distance;
        let hit = slab(
            self.origin.x,
            self.dir_x,
            self.inv_dir_x,
            bounds.min_x,
            bounds.max_x,
            &mut t_min,
            &mut t_max,
        ) && slab(
            self.origin.y,
            self.dir_y,
            self.inv_dir_y,
            bounds.min_y,
            bounds.max_y,
            &mut t_min,
            &mut t_max,
        );
        hit.then_some(t_min)
    }
}

/// Finite 3D ray segment used by `raycast` searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, Point3D, Ray3D};
///
/// let ray = Ray3D::new(Point3D::new(-1.0, 0.5, 0.5), 1.0, 0.0, 0.0, 10.0);
/// assert!(ray.intersects_box(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray3D {
    /// Ray origin.
    pub origin: Point3D,
    /// X component of the ray direction.
    pub dir_x: f64,
    /// Y component of the ray direction.
    pub dir_y: f64,
    /// Z component of the ray direction.
    pub dir_z: f64,
    // Precomputed to remove one division from every slab test.
    pub(crate) inv_dir_x: f64,
    pub(crate) inv_dir_y: f64,
    pub(crate) inv_dir_z: f64,
    /// Maximum ray parameter to consider.
    pub max_distance: f64,
}

impl Ray3D {
    /// Create a finite ray segment covering `origin + t * dir` for
    /// `t in [0, max_distance]`.
    ///
    /// The direction does not need to be normalized. `max_distance` and every
    /// returned entry `t` are in units of the direction length, so the
    /// Euclidean distance to a hit is `t * (dir_x.hypot(dir_y).hypot(dir_z))`;
    /// normalize the direction (length 1) if you want `t` and `max_distance` in
    /// world units.
    ///
    /// A fully zero direction (`dir_x == dir_y == dir_z == 0.0`) is a point
    /// probe: it hits only boxes that contain `origin`, all at `t == 0.0`.
    /// Direction components should be finite; a `NaN` direction produces
    /// unspecified results.
    #[inline]
    pub const fn new(
        origin: Point3D,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        max_distance: f64,
    ) -> Self {
        Self {
            origin,
            dir_x,
            dir_y,
            dir_z,
            inv_dir_x: 1.0 / dir_x,
            inv_dir_y: 1.0 / dir_y,
            inv_dir_z: 1.0 / dir_z,
            max_distance,
        }
    }

    /// `true` if any direction component is exactly zero (an axis-parallel ray). The
    /// vectorized slab test uses `1/dir = inf` and is not NaN-safe at a box face, so
    /// such rays take a masked path.
    #[inline]
    pub(crate) fn has_zero_direction(self) -> bool {
        self.dir_x == 0.0 || self.dir_y == 0.0 || self.dir_z == 0.0
    }

    /// `true` when the ray segment touches `bounds` (faces inclusive).
    #[inline]
    pub fn intersects_box(self, bounds: Box3D) -> bool {
        if self.max_distance < 0.0 || self.max_distance.is_nan() {
            return false;
        }
        let mut t_min: f64 = 0.0;
        let mut t_max = self.max_distance;
        slab(
            self.origin.x,
            self.dir_x,
            self.inv_dir_x,
            bounds.min_x,
            bounds.max_x,
            &mut t_min,
            &mut t_max,
        ) && slab(
            self.origin.y,
            self.dir_y,
            self.inv_dir_y,
            bounds.min_y,
            bounds.max_y,
            &mut t_min,
            &mut t_max,
        ) && slab(
            self.origin.z,
            self.dir_z,
            self.inv_dir_z,
            bounds.min_z,
            bounds.max_z,
            &mut t_min,
            &mut t_max,
        )
    }

    /// Entry parameter `t` where the ray segment first touches `bounds` (`0.0` if the
    /// origin is inside), or `None` if the segment misses. Used by ordered closest-hit
    /// traversal.
    #[inline]
    pub fn enter_t(self, bounds: Box3D) -> Option<f64> {
        if self.max_distance < 0.0 || self.max_distance.is_nan() {
            return None;
        }
        let mut t_min: f64 = 0.0;
        let mut t_max = self.max_distance;
        let hit = slab(
            self.origin.x,
            self.dir_x,
            self.inv_dir_x,
            bounds.min_x,
            bounds.max_x,
            &mut t_min,
            &mut t_max,
        ) && slab(
            self.origin.y,
            self.dir_y,
            self.inv_dir_y,
            bounds.min_y,
            bounds.max_y,
            &mut t_min,
            &mut t_max,
        ) && slab(
            self.origin.z,
            self.dir_z,
            self.inv_dir_z,
            bounds.min_z,
            bounds.max_z,
            &mut t_min,
            &mut t_max,
        );
        hit.then_some(t_min)
    }
}

#[inline]
fn slab(
    origin: f64,
    direction: f64,
    inverse: f64,
    min: f64,
    max: f64,
    t_min: &mut f64,
    t_max: &mut f64,
) -> bool {
    if direction == 0.0 {
        return origin >= min && origin <= max;
    }

    let mut near = (min - origin) * inverse;
    let mut far = (max - origin) * inverse;
    if near > far {
        core::mem::swap(&mut near, &mut far);
    }

    *t_min = (*t_min).max(near);
    *t_max = (*t_max).min(far);
    *t_min <= *t_max
}
