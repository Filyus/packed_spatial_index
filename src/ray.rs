//! Finite ray segments for ray/AABB traversal over the packed indexes.
//!
//! A ray is defined by an origin, an unnormalized direction, and a maximum ray
//! parameter `max_distance` (the segment covers `origin + t * dir` for
//! `t in [0, max_distance]`). Box intersections use the standard slab test with
//! precomputed reciprocal directions; axis-parallel or subnormal direction
//! components are handled explicitly so a ray lying exactly on a box face still
//! hits.

use crate::geometry::{Box2D, Box3D, Point2D, Point3D};
use crate::triangle::{Triangle3, TriangleHit};

/// Conservative multiplier on `t_max` in the ray-AABB slab test (Ize 2013,
/// *Robust BVH Ray Traversal*, JCGT 2(2)).
///
/// Each slab bound `(bound - origin) * (1.0 / dir)` incurs three `f64`
/// roundings (`1/d`, `bound - o`, product), so its relative error is bounded by
/// `gamma_3 = 3u / (1 - 3u)` where `u = f64::EPSILON / 2 = 2^-53`. For any true
/// intersection (`t_near_exact <= t_far_exact`, both `>= 0`):
///
/// `t_near_comp <= t_near_exact * (1 + gamma_3) <= t_far_comp * (1 + gamma_3) / (1 - gamma_3)`.
///
/// Because `(1 + gamma_3) / (1 - gamma_3) = 1 + 6u / (1 - 3u) < 1 + 8u = 1 + 4 * f64::EPSILON`,
/// comparing `t_min <= t_max * SLAB_FAR_SCALE` never drops a true hit (including
/// grazing faces/edges and hits at `t == max_distance`). When `t_max < 0`,
/// `t_max * SLAB_FAR_SCALE < 0 <= t_min`, so behind-origin misses stay misses.
pub(crate) const SLAB_FAR_SCALE: f64 = 1.0 + 4.0 * f64::EPSILON;

/// Conservative multiplier on `t_min` returned by `enter_t`.
///
/// Since `t_near_comp <= t_near_exact * (1 + gamma_3)` and
/// `1 / (1 + gamma_3) > 1 - 4u = 1 - 2 * f64::EPSILON`, scaling the positive
/// entry parameter by `SLAB_NEAR_SCALE` guarantees a lower bound on the true
/// entry parameter `t_near_exact` (and keeps it `<= max_distance` even when
/// `t_min` overshoots `max_distance` by up to `gamma_3`).
pub(crate) const SLAB_NEAR_SCALE: f64 = 1.0 - 2.0 * f64::EPSILON;

#[inline]
pub(crate) fn inclusive_ray_cutoff(max_distance: f64) -> f64 {
    if !max_distance.is_finite() {
        return f64::INFINITY;
    }
    let max_distance = if max_distance == 0.0 {
        0.0
    } else {
        max_distance
    };
    f64::from_bits(max_distance.to_bits() + 1)
}

/// Finite 2D ray segment used by `raycast` searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box2D, Point2D, Ray2D};
///
/// let ray = Ray2D::new(Point2D::new(-1.0, 0.5), 1.0, 0.0, 10.0);
/// assert!(ray.intersects_box(Box2D::new(0.0, 0.0, 1.0, 1.0)));
/// let t = ray.enter_t(Box2D::new(0.0, 0.0, 1.0, 1.0)).unwrap();
/// assert!(t <= 1.0 && (1.0 - t) <= 4.0 * f64::EPSILON);
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
    /// components must be finite; a non-finite origin or direction never hits.
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

    /// `true` if any direction component needs the masked SIMD slab path. Exact
    /// zero and subnormal directions can produce `inf` reciprocals, and the
    /// multiply-only vector slab is not NaN-safe at a box face.
    #[inline]
    #[allow(dead_code)] // used by the SIMD raycast paths only
    pub(crate) fn has_zero_direction(self) -> bool {
        self.dir_x == 0.0
            || self.dir_y == 0.0
            || self.dir_x.is_subnormal()
            || self.dir_y.is_subnormal()
    }

    #[inline]
    pub(crate) fn has_non_finite_component(self) -> bool {
        !self.origin.x.is_finite()
            || !self.origin.y.is_finite()
            || !self.dir_x.is_finite()
            || !self.dir_y.is_finite()
    }

    /// `true` when the ray segment touches `bounds` (edges inclusive, conservative
    /// against `f64` slab rounding).
    #[inline]
    pub fn intersects_box(self, bounds: Box2D) -> bool {
        if self.max_distance < 0.0 || self.max_distance.is_nan() || self.has_non_finite_component()
        {
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
        ) && t_min <= t_max * SLAB_FAR_SCALE
    }

    /// Conservative lower bound on the entry parameter `t` where the ray segment
    /// first touches `bounds` (`0.0` if the origin is inside), or `None` if the
    /// segment misses. Used by ordered closest-hit traversal.
    #[inline]
    pub fn enter_t(self, bounds: Box2D) -> Option<f64> {
        if self.max_distance < 0.0 || self.max_distance.is_nan() || self.has_non_finite_component()
        {
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
        ) && t_min <= t_max * SLAB_FAR_SCALE;
        hit.then_some(t_min * SLAB_NEAR_SCALE)
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
    /// Direction components must be finite; a non-finite origin or direction
    /// never hits.
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

    /// `true` if any direction component needs the masked SIMD slab path. Exact
    /// zero and subnormal directions can produce `inf` reciprocals, and the
    /// multiply-only vector slab is not NaN-safe at a box face.
    #[inline]
    #[allow(dead_code)] // used by the SIMD raycast paths only
    pub(crate) fn has_zero_direction(self) -> bool {
        self.dir_x == 0.0
            || self.dir_y == 0.0
            || self.dir_z == 0.0
            || self.dir_x.is_subnormal()
            || self.dir_y.is_subnormal()
            || self.dir_z.is_subnormal()
    }

    #[inline]
    pub(crate) fn has_non_finite_component(self) -> bool {
        !self.origin.x.is_finite()
            || !self.origin.y.is_finite()
            || !self.origin.z.is_finite()
            || !self.dir_x.is_finite()
            || !self.dir_y.is_finite()
            || !self.dir_z.is_finite()
    }

    /// `true` when the ray segment touches `bounds` (faces inclusive, conservative
    /// against `f64` slab rounding).
    #[inline]
    pub fn intersects_box(self, bounds: Box3D) -> bool {
        if self.max_distance < 0.0 || self.max_distance.is_nan() || self.has_non_finite_component()
        {
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
        ) && t_min <= t_max * SLAB_FAR_SCALE
    }

    /// Conservative lower bound on the entry parameter `t` where the ray segment
    /// first touches `bounds` (`0.0` if the origin is inside), or `None` if the
    /// segment misses. Used by ordered closest-hit traversal.
    #[inline]
    pub fn enter_t(self, bounds: Box3D) -> Option<f64> {
        if self.max_distance < 0.0 || self.max_distance.is_nan() || self.has_non_finite_component()
        {
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
        ) && t_min <= t_max * SLAB_FAR_SCALE;
        hit.then_some(t_min * SLAB_NEAR_SCALE)
    }

    /// Exact squared Euclidean distance from this ray segment to `bounds`,
    /// or `0.0` when the segment intersects the box.
    ///
    /// The minimum of the (convex, piecewise-quadratic) squared point-to-box
    /// distance along `t in [0, max_distance]`, computed in closed form over the
    /// slab breakpoints — no iteration, no allocation. This is a lower bound on
    /// the distance from the ray to anything inside the box, which is what makes
    /// it a sound best-first ordering key: a box's value can only under-estimate
    /// the true distance of the geometry the user clicked. A non-finite origin or
    /// direction, or a negative/NaN `max_distance`, yields [`f64::INFINITY`]
    /// (the segment touches nothing).
    ///
    /// Used by [`Index3D::search_pick`](crate::Index3D::search_pick) as the
    /// perpendicular component of its ordering key.
    pub fn distance_squared_to_box(self, bounds: Box3D) -> f64 {
        // `enter_t` carries the same guards, and answers `None` under all of
        // them; the miss path re-checks them for callers that skip it.
        if self.enter_t(bounds).is_some() {
            return 0.0;
        }
        self.distance_squared_to_missed_box(bounds)
    }

    /// [`Self::distance_squared_to_box`] for a box the caller has already found
    /// the segment to miss — the same closed form without the entry test, which
    /// the pick key would otherwise pay twice per node.
    pub(crate) fn distance_squared_to_missed_box(self, bounds: Box3D) -> f64 {
        if self.max_distance < 0.0 || self.max_distance.is_nan() || self.has_non_finite_component()
        {
            return f64::INFINITY;
        }
        let org = [self.origin.x, self.origin.y, self.origin.z];
        let dir = [self.dir_x, self.dir_y, self.dir_z];
        let lo = [bounds.min_x, bounds.min_y, bounds.min_z];
        let hi = [bounds.max_x, bounds.max_y, bounds.max_z];
        // Breakpoints where an axis coordinate crosses a slab boundary,
        // ascending, always starting at 0 and ending at max_distance.
        let mut ts = [0.0f64; 8];
        let mut nt = 2usize;
        ts[1] = self.max_distance;
        // Every non-zero direction axis can cross a slab boundary; finite
        // crossings strictly inside `(0, max_distance)` split the quadratic
        // pieces. Overflowing divisions (tiny `dir[a]`) yield non-finite `t`
        // and are skipped without an arbitrary threshold.
        for a in 0..3 {
            if dir[a] != 0.0 {
                let t0 = (lo[a] - org[a]) / dir[a];
                let t1 = (hi[a] - org[a]) / dir[a];
                for t in [t0, t1] {
                    if t.is_finite() && t > 0.0 && t < self.max_distance {
                        let mut i = nt;
                        ts[nt] = t;
                        nt += 1;
                        while i > 1 && ts[i - 1] > ts[i] {
                            ts.swap(i - 1, i);
                            i -= 1;
                        }
                    }
                }
            }
        }
        let dist_at = |t: f64| -> f64 {
            let mut s = 0.0;
            for a in 0..3 {
                let p = org[a] + t * dir[a];
                let h = if p < lo[a] {
                    lo[a] - p
                } else if p > hi[a] {
                    p - hi[a]
                } else {
                    0.0
                };
                s += h * h;
            }
            s
        };
        let mut best = f64::INFINITY;
        for w in 0..nt - 1 {
            let (t0, t1) = (ts[w], ts[w + 1]);
            // On this piece each axis is either inside its slab (contributes 0)
            // or offset linearly: f(t) = sum (c_a + m_a t)^2, minimized in
            // closed form on [t0, t1].
            let mut c = [0.0f64; 3];
            let mut m = [0.0f64; 3];
            // The active set is constant across the piece, so any interior
            // point identifies it. With an infinite max_distance the last
            // piece ends at infinity, where `origin + inf * 0.0` is NaN for a
            // zero-direction axis and would silently read as "inside".
            let span = t1 - t0;
            let mid = if span.is_finite() {
                t0 + 0.5 * span
            } else {
                t0 + 1.0
            };
            let mut active = false;
            for a in 0..3 {
                let p = org[a] + mid * dir[a];
                if p < lo[a] {
                    c[a] = lo[a] - org[a];
                    m[a] = -dir[a];
                    active = true;
                } else if p > hi[a] {
                    c[a] = org[a] - hi[a];
                    m[a] = dir[a];
                    active = true;
                }
            }
            if !active {
                return 0.0; // the segment passes through the box on this piece
            }
            let mut a2 = 0.0;
            let mut a1 = 0.0;
            for a in 0..3 {
                a2 += m[a] * m[a];
                a1 += 2.0 * c[a] * m[a];
            }
            let t_star = if a2 > 0.0 {
                (-a1 / (2.0 * a2)).clamp(t0, t1)
            } else {
                t0
            };
            // Evaluate the sum of squared axis gaps directly at `t_star` rather
            // than the expanded `a2*t² + a1*t + a0`, which cancels when `|c|`
            // is large relative to the residual gap.
            best = best.min(dist_at(t_star));
        }
        for t in &ts[..nt] {
            best = best.min(dist_at(*t));
        }
        best
    }

    /// The closest triangle in `triangles` hit by this ray segment, as a
    /// [`TriangleHit`] (`index` into the slice and `t` in direction-length
    /// units), or `None` if the ray misses them all, by the Moller-Trumbore test.
    ///
    /// Works over either record type: [`Triangle3D`](crate::Triangle3D) tests in
    /// `f64`, [`Triangle3DF32`](crate::Triangle3DF32) in `f32` (8 at a time with
    /// the `simd` feature). Pair this with a query that yields candidate triangles
    /// — an [`Index3DView`](crate::Index3DView)'s
    /// [`triangles`](crate::Index3DView::triangles) payload, or the items a
    /// `raycast` / `search` returns — for exact ray-mesh intersection.
    #[inline]
    pub fn closest_triangle<T: Triangle3>(&self, triangles: &[T]) -> Option<TriangleHit> {
        let o = [self.origin.x, self.origin.y, self.origin.z];
        let d = [self.dir_x, self.dir_y, self.dir_z];
        T::closest_hit(o, d, self.max_distance, triangles)
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
    true
}

#[cfg(test)]
mod distance_tests {
    use super::*;
    use crate::Point3D;

    #[test]
    fn infinite_max_distance_still_measures_perpendicular_offset() {
        // Regression: the last quadratic piece runs to infinity, where
        // `origin + inf * 0.0` is NaN for a zero-direction axis; the NaN used
        // to read as "inside the slab" and dropped the perpendicular offset.
        let ray = Ray3D::new(
            Point3D {
                x: -100.0,
                y: 0.0,
                z: 0.0,
            },
            1.0,
            0.0,
            0.0,
            f64::INFINITY,
        );
        let point = Box3D::new(20.0, 0.0, 10.0, 20.0, 0.0, 10.0);
        assert!(ray.enter_t(point).is_none());
        assert!((ray.distance_squared_to_box(point) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn measures_the_segment_end_not_the_infinite_line() {
        let ray = Ray3D::new(
            Point3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            1.0,
            0.0,
            0.0,
            5.0,
        );
        let b = Box3D::new(10.0, -1.0, -1.0, 12.0, 1.0, 1.0);
        let d2 = ray.distance_squared_to_box(b);
        assert!((d2 - 25.0).abs() < 1e-9, "{d2}"); // endpoint (5,0,0), dx gap 5
    }

    #[test]
    fn zero_distance_when_the_segment_enters() {
        let ray = Ray3D::new(
            Point3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            1.0,
            0.0,
            0.0,
            100.0,
        );
        let b = Box3D::new(10.0, -1.0, -1.0, 12.0, 1.0, 1.0);
        assert_eq!(ray.distance_squared_to_box(b), 0.0);
        // and behind the origin: clamped to t=0
        let behind = Box3D::new(-12.0, -1.0, -1.0, -10.0, 1.0, 1.0);
        let d2 = ray.distance_squared_to_box(behind);
        assert!((d2 - 100.0).abs() < 1e-9, "{d2}"); // origin (0,0,0), dx gap 10
    }

    #[test]
    fn conservative_slab_hits_grazing_corner_and_lower_bounds_enter_t() {
        // Non-power-of-two direction where `1.0 / dir` rounds; aim directly at
        // a box corner `origin + t_true * dir` so a non-conservative slab can
        // miss on ULP mismatch between axes.
        let origin = Point3D::new(0.1, 0.2, 0.3);
        let dir = [3.0, 7.0, 11.0];
        let t_true = 1.0 / 3.0;
        let corner = [
            origin.x + t_true * dir[0],
            origin.y + t_true * dir[1],
            origin.z + t_true * dir[2],
        ];
        let bx = Box3D::new(
            corner[0],
            corner[1],
            corner[2],
            corner[0] + 1.0,
            corner[1] + 1.0,
            corner[2] + 1.0,
        );
        let ray = Ray3D::new(origin, dir[0], dir[1], dir[2], t_true);
        assert!(ray.intersects_box(bx));
        let t_enter = ray
            .enter_t(bx)
            .expect("grazing corner at max_distance must hit");
        assert!(
            t_enter <= t_true,
            "enter_t {t_enter} exceeded true t {t_true}"
        );
    }

    #[test]
    fn distance_to_missed_box_tracks_tiny_nonzero_axis_crossing() {
        // Direction has a tiny nonzero y component (`1e-14`), smaller than the
        // old `1e-12` relative cutoff, yet over `max_distance = 1e16` it
        // crosses the box's y slab completely.
        let ray = Ray3D::new(Point3D::new(0.0, 0.0, 0.0), 1.0, 1e-14, 0.0, 1e16);
        let bx = Box3D::new(1e15, 10.0, 1.0, 1e15 + 1.0, 11.0, 2.0);
        // At t = 1e15: p = (1e15, 10.0, 0.0), which is inside x and y slabs
        // and only 1.0 away in z -> d² = 1.0.
        let d2 = ray.distance_squared_to_box(bx);
        assert!((d2 - 1.0).abs() < 1e-9, "expected ~1.0, got {d2}");
    }
}

#[cfg(test)]
mod triangle_tests {
    use super::*;
    use crate::{Point3D, Triangle3D};

    fn ray(o: [f64; 3], d: [f64; 3]) -> Ray3D {
        Ray3D::new(Point3D::new(o[0], o[1], o[2]), d[0], d[1], d[2], 100.0)
    }

    #[test]
    fn closest_triangle_hits_nearest_and_misses() {
        // Two triangles facing the ray at z = 2 and z = 5; a +z ray should pick
        // the nearer one regardless of slice order.
        let near = Triangle3D::new([0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0]);
        let far = Triangle3D::new([0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [0.0, 2.0, 5.0]);
        let tris = [far, near];
        let hit = ray([0.25, 0.25, 0.0], [0.0, 0.0, 1.0])
            .closest_triangle(&tris)
            .unwrap();
        assert_eq!(hit.index, 1);
        assert!((hit.t - 2.0).abs() < 1e-4, "t={}", hit.t);

        // Pointing away, and an empty slice, both miss.
        assert!(
            ray([0.25, 0.25, 0.0], [0.0, 0.0, -1.0])
                .closest_triangle(&tris)
                .is_none()
        );
        assert!(
            ray([0.25, 0.25, 0.0], [0.0, 0.0, 1.0])
                .closest_triangle::<Triangle3D>(&[])
                .is_none()
        );
    }

    #[test]
    fn closest_triangle_f32_records() {
        use crate::Triangle3DF32;
        let near = Triangle3DF32::new([0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0]);
        let far = Triangle3DF32::new([0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [0.0, 2.0, 5.0]);
        let hit = ray([0.25, 0.25, 0.0], [0.0, 0.0, 1.0])
            .closest_triangle(&[far, near])
            .unwrap();
        assert_eq!(hit.index, 1);
        assert!((hit.t - 2.0).abs() < 1e-4, "t={}", hit.t);
    }
}
