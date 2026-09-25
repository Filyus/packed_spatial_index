//! A solid cone in 3D: apex, axis direction, half-angle and height — the
//! sensor-FOV / spotlight query shape.
//!
//! The box-vs-cone overlap test reduces to a one-dimensional convex question
//! along the axis. Slice the box by the plane at height `t` above the apex:
//! the slice is a convex polygon, and the cone's cross-section there is a disk
//! of radius `t * tan(half_angle)` centred on the axis. The box meets the cone
//! iff for some `t` in `[0, height]`
//!
//! ```text
//! dist(axis(t), box ∩ plane(t)) - t * tan(half_angle) <= 0
//! ```
//!
//! The first term is the partial minimisation, over the box, of a function
//! that is jointly convex in the point and `t`, so it is convex in `t`; the
//! whole left-hand side is convex, and its minimum over the box's height range
//! settles the question. Each evaluation is exact: the distance from the axis
//! point to the slice is the projection of that point onto the box
//! intersected with a plane, a three-variable quadratic knapsack with a
//! closed form (clamp along the plane normal). The outer minimum is found by
//! golden-section search on the convex function, cut short as soon as a probe
//! lands inside the cone or convexity bounds the minimum above zero; the only
//! inexactness is
//! floating-point, so boxes within floating-point reach of the cone's surface
//! may be reported either way. (The simpler "distance from the axis point to
//! the box against the disk radius" test is *not* equivalent: that compares
//! against a ball, whose union along the axis is a fatter cone.)
//!
//! Containment is exact by convexity: the solid cone contains a box iff it
//! contains the box's eight corners.

use crate::capsule::min_segment_box_gap_squared;
use crate::geometry::{Box3D, Overlaps3D};

/// Why [`Cone3D::try_new`] refused its inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cone3DError {
    /// The axis direction is zero or non-finite, so no cone axis exists.
    ZeroDirection,
    /// The half-angle is not in `(0°, 90°)` (or not finite).
    HalfAngle(f64),
    /// The height is not finite or is not positive.
    Height(f64),
}

impl std::fmt::Display for Cone3DError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDirection => write!(f, "the cone axis is zero or non-finite"),
            Self::HalfAngle(a) => {
                write!(f, "half-angle must be in (0, pi/2), got {a}")
            }
            Self::Height(h) => write!(f, "height must be finite and positive, got {h}"),
        }
    }
}

impl std::error::Error for Cone3DError {}

/// A solid circular cone: every point within `half_angle` of the axis, from
/// the `apex` out to `height` along it.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, Cone3D, Index3DBuilder};
///
/// let mut b = Index3DBuilder::new(2);
/// b.add(Box3D::new(-1.0, -1.0, 5.0, 1.0, 1.0, 6.0)); // on the axis
/// b.add(Box3D::new(20.0, 0.0, 5.0, 21.0, 1.0, 6.0)); // far off to the side
/// let index = b.finish().unwrap();
///
/// // A spotlight at the origin pointing up +z, 30 degrees wide, 10 units long.
/// let spot = Cone3D::try_new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 30f64.to_radians(), 10.0)?;
/// assert_eq!(index.search(&spot), vec![0]);
/// # Ok::<(), packed_spatial_index::Cone3DError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cone3D {
    /// The cone's tip.
    pub apex: [f64; 3],
    /// Unit axis direction, from the apex toward the base.
    pub axis: [f64; 3],
    /// Half-angle in radians, in `(0, pi/2)`.
    pub half_angle: f64,
    /// Distance from the apex to the base, positive.
    pub height: f64,
    /// `tan(half_angle)`, cached: the radius per unit height.
    tan_half: f64,
    /// `cos(half_angle)`, cached: the point-in-wedge threshold.
    cos_half: f64,
}

/// Golden-section steps for the convex minimisation in `overlaps_box`. Each
/// step shrinks the bracket by the golden ratio; 70 of them take a bracket
/// the size of the cone's height down to ~2e-15 of it, below the precision of
/// any coordinate the bracket could have been computed from.
const OVERLAP_STEPS: usize = 70;
/// `1 / φ`: the golden-section split.
const INV_PHI: f64 = 0.618_033_988_749_894_9;

impl Cone3D {
    /// Build a cone, validating the pieces.
    ///
    /// The `axis` need not be unit length (it is normalized here), but must be
    /// finite and non-zero; the half-angle is radians in `(0, pi/2)`; the
    /// height is measured along the axis from the apex.
    pub fn try_new(
        apex: [f64; 3],
        axis: [f64; 3],
        half_angle: f64,
        height: f64,
    ) -> Result<Self, Cone3DError> {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if !len.is_finite() || len == 0.0 {
            return Err(Cone3DError::ZeroDirection);
        }
        if !(half_angle > 0.0 && half_angle < std::f64::consts::FRAC_PI_2 && half_angle.is_finite())
        {
            return Err(Cone3DError::HalfAngle(half_angle));
        }
        if !(height.is_finite() && height > 0.0) {
            return Err(Cone3DError::Height(height));
        }
        Ok(Self {
            apex,
            axis: [axis[0] / len, axis[1] / len, axis[2] / len],
            half_angle,
            height,
            tan_half: half_angle.tan(),
            cos_half: half_angle.cos(),
        })
    }

    /// The box's extent along the axis, as heights above the apex:
    /// `[min, max]` of `axis · (q - apex)` over the box.
    #[inline]
    fn height_range(&self, lo: &[f64; 3], hi: &[f64; 3]) -> (f64, f64) {
        let base =
            self.axis[0] * self.apex[0] + self.axis[1] * self.apex[1] + self.axis[2] * self.apex[2];
        let (mut t_min, mut t_max) = (-base, -base);
        for i in 0..3 {
            let a = self.axis[i];
            if a >= 0.0 {
                t_min += a * lo[i];
                t_max += a * hi[i];
            } else {
                t_min += a * hi[i];
                t_max += a * lo[i];
            }
        }
        (t_min, t_max)
    }

    /// Distance from the axis point at height `t` to the box's slice by the
    /// plane at that height — the perpendicular clearance between the axis and
    /// the box, measured in that plane. Requires `t` inside
    /// [`height_range`](Self::height_range) so the slice is non-empty.
    ///
    /// This is the projection of `P = apex + t·axis` onto
    /// `box ∩ { q : axis · (q - apex) = t }`. Its KKT conditions give
    /// `q_i = clamp(P_i + λ·axis_i, lo_i, hi_i)` for the unique `λ` that puts
    /// `q` back on the plane; `φ(λ) = axis · q(λ)` is nondecreasing and
    /// piecewise linear with at most six kinks (each coordinate hits each of
    /// its two bounds once), so `λ` is read off the piece where `φ` crosses
    /// the target — no iteration.
    fn slice_gap(&self, t: f64, lo: &[f64; 3], hi: &[f64; 3]) -> f64 {
        let a = self.axis;
        let p = [
            self.apex[0] + t * a[0],
            self.apex[1] + t * a[1],
            self.apex[2] + t * a[2],
        ];
        // The plane through `P`: axis · q = axis · P.
        let target = a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
        let q_at = |lambda: f64| {
            [
                (p[0] + lambda * a[0]).clamp(lo[0], hi[0]),
                (p[1] + lambda * a[1]).clamp(lo[1], hi[1]),
                (p[2] + lambda * a[2]).clamp(lo[2], hi[2]),
            ]
        };
        let phi = |q: &[f64; 3]| a[0] * q[0] + a[1] * q[1] + a[2] * q[2];

        // Kinks of φ, sorted.
        let mut kinks = [0.0f64; 6];
        let mut n = 0;
        for i in 0..3 {
            if a[i] != 0.0 {
                kinks[n] = (lo[i] - p[i]) / a[i];
                kinks[n + 1] = (hi[i] - p[i]) / a[i];
                n += 2;
            }
        }
        kinks[..n].sort_by(|x, y| x.total_cmp(y));

        // Left of the smallest kink every coordinate is clamped at the end the
        // axis points away from, right of the largest at the other end, so φ
        // is flat on both outer pieces and its whole range is
        // [φ(k_0), φ(k_{n-1})]; the caller keeps `t` inside the box's height
        // range, which is exactly that interval. Between consecutive kinks φ
        // is linear, so the crossing is read off by interpolation.
        let mut values = [0.0f64; 6];
        for k in 0..n {
            values[k] = phi(&q_at(kinks[k]));
        }
        let lambda = if target <= values[0] {
            kinks[0]
        } else if target >= values[n - 1] {
            kinks[n - 1]
        } else {
            let j = (1..n)
                .find(|&j| target <= values[j])
                .expect("target is below the last kink's value");
            let (v0, v1) = (values[j - 1], values[j]);
            // `v0 < target <= v1`, so the piece has positive slope.
            kinks[j - 1] + (target - v0) / (v1 - v0) * (kinks[j] - kinks[j - 1])
        };
        let q = q_at(lambda);
        let (dx, dy, dz) = (q[0] - p[0], q[1] - p[1], q[2] - p[2]);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// `dist(axis(t), box ∩ plane(t)) - t * tan(half_angle)`: negative where the
    /// box's slice reaches inside the cone's disk at height `t`. Convex in `t`
    /// over the box's height range.
    #[inline]
    fn clearance(&self, t: f64, tan: f64, lo: &[f64; 3], hi: &[f64; 3]) -> f64 {
        self.slice_gap(t, lo, hi) - t * tan
    }

    /// Whether any part of `bx` intersects the solid cone.
    pub fn overlaps_box(&self, bx: Box3D) -> bool {
        let lo = [bx.min_x, bx.min_y, bx.min_z];
        let hi = [bx.max_x, bx.max_y, bx.max_z];
        let (t_min, t_max) = self.height_range(&lo, &hi);
        let (mut lo_t, mut hi_t) = (t_min.max(0.0), t_max.min(self.height));
        if lo_t > hi_t {
            // The box lies wholly behind the apex or beyond the base.
            return false;
        }
        let tan = self.tan_half;
        // Cheap decisions first; the search below is the expensive general
        // case and most boxes a traversal tests are nowhere near the cone.
        // A corner inside the cone settles it.
        if self.contains_point(lo)
            || self.contains_point(hi)
            || self.contains_point([lo[0], lo[1], hi[2]])
            || self.contains_point([lo[0], hi[1], lo[2]])
            || self.contains_point([hi[0], lo[1], lo[2]])
            || self.contains_point([lo[0], hi[1], hi[2]])
            || self.contains_point([hi[0], lo[1], hi[2]])
            || self.contains_point([hi[0], hi[1], lo[2]])
        {
            return true;
        }
        // The axis segment's clearance to the box: zero means the axis passes
        // through the box (a hit — every axis point is in the cone). A point
        // of the box inside the cone sits within the cone's radius at its own
        // height, and no box point is higher than `hi_t`, so a clearance above
        // the radius at `hi_t` rules the whole box out.
        let base = [
            self.apex[0] + self.height * self.axis[0],
            self.apex[1] + self.height * self.axis[1],
            self.apex[2] + self.height * self.axis[2],
        ];
        let gap2 = min_segment_box_gap_squared(&self.apex, &base, &lo, &hi, 3, 0.0, 1.0);
        if gap2 == 0.0 {
            return true;
        }
        let top_radius = hi_t * tan;
        if gap2 > top_radius * top_radius {
            return false;
        }
        // Golden-section search on a convex function over [lo_t, hi_t]: one
        // new evaluation per step. Any probe that reaches inside the cone is
        // already a proof of overlap, and convexity gives a lower bound on the
        // minimum from the four values in hand (secant slopes only grow left
        // to right), so a box that stays clear is dismissed after a few steps
        // instead of at full precision.
        let mut flo = self.clearance(lo_t, tan, &lo, &hi);
        let mut fhi = self.clearance(hi_t, tan, &lo, &hi);
        if flo <= 0.0 || fhi <= 0.0 {
            return true;
        }
        let mut c = hi_t - INV_PHI * (hi_t - lo_t);
        let mut d = lo_t + INV_PHI * (hi_t - lo_t);
        let mut fc = self.clearance(c, tan, &lo, &hi);
        let mut fd = self.clearance(d, tan, &lo, &hi);
        for _ in 0..OVERLAP_STEPS {
            if fc <= 0.0 || fd <= 0.0 {
                return true;
            }
            if convex_min_lower_bound(lo_t, flo, c, fc, d, fd, hi_t, fhi) > 0.0 {
                return false;
            }
            if fc < fd {
                hi_t = d;
                fhi = fd;
                d = c;
                fd = fc;
                c = hi_t - INV_PHI * (hi_t - lo_t);
                fc = self.clearance(c, tan, &lo, &hi);
            } else {
                lo_t = c;
                flo = fc;
                c = d;
                fc = fd;
                d = lo_t + INV_PHI * (hi_t - lo_t);
                fd = self.clearance(d, tan, &lo, &hi);
            }
        }
        fc.min(fd) <= 0.0 || self.clearance((lo_t + hi_t) * 0.5, tan, &lo, &hi) <= 0.0
    }

    /// Whether all of `bx` lies inside the solid cone.
    ///
    /// Exact: the solid cone is convex, and a convex set contains a box iff it
    /// contains the box's corners, so eight corner tests decide it.
    pub fn contains_box(&self, bx: Box3D) -> bool {
        let corners = [
            [bx.min_x, bx.min_y, bx.min_z],
            [bx.max_x, bx.min_y, bx.min_z],
            [bx.min_x, bx.max_y, bx.min_z],
            [bx.max_x, bx.max_y, bx.min_z],
            [bx.min_x, bx.min_y, bx.max_z],
            [bx.max_x, bx.min_y, bx.max_z],
            [bx.min_x, bx.max_y, bx.max_z],
            [bx.max_x, bx.max_y, bx.max_z],
        ];
        corners.into_iter().all(|c| self.contains_point(c))
    }

    /// Whether the point lies in the closed solid cone.
    #[inline]
    fn contains_point(&self, [x, y, z]: [f64; 3]) -> bool {
        let [ax, ay, az] = self.apex;
        let (vx, vy, vz) = (x - ax, y - ay, z - az);
        let len = (vx * vx + vy * vy + vz * vz).sqrt();
        // Inside the height band: the projection on the axis is in
        // [0, height] (the caps).
        let along = vx * self.axis[0] + vy * self.axis[1] + vz * self.axis[2];
        if !(0.0..=self.height).contains(&along) {
            return false;
        }
        // Inside the wedge: the angle to the axis is at most half_angle. The
        // apex itself is in the closed cone.
        len == 0.0 || along / len >= self.cos_half
    }
}

/// A lower bound on the minimum of a convex function over `[a, b]` from its
/// values at `a < c < d < b` (degenerate equal points allowed).
///
/// Convexity means secant slopes are nondecreasing left to right, so on each
/// of the three pieces the function lies above the line through its right
/// (or left) neighbours' secant: on `[a, c]` above the `cd` secant continued
/// leftwards, on `[d, b]` above it continued rightwards, and on `[c, d]` above
/// both the `ac` secant continued right and the `db` secant continued left —
/// their crossing is the bound there.
#[allow(clippy::too_many_arguments)]
fn convex_min_lower_bound(
    a: f64,
    fa: f64,
    c: f64,
    fc: f64,
    d: f64,
    fd: f64,
    b: f64,
    fb: f64,
) -> f64 {
    let s_cd = if d > c { (fd - fc) / (d - c) } else { 0.0 };
    let lb_left = if c > a {
        fc - s_cd.max(0.0) * (c - a)
    } else {
        fc
    };
    let lb_right = if b > d {
        fd + s_cd.min(0.0) * (b - d)
    } else {
        fd
    };
    let lb_mid = if d > c {
        let s_ac = if c > a {
            (fc - fa) / (c - a)
        } else {
            f64::NEG_INFINITY
        };
        let s_db = if b > d {
            (fb - fd) / (b - d)
        } else {
            f64::INFINITY
        };
        if s_ac >= 0.0 {
            fc
        } else if s_db <= 0.0 {
            fd
        } else {
            // `fc + s_ac (t - c)` falls, `fd + s_db (t - d)` rises; the
            // minimum of their maximum is at the crossing, clamped to the
            // piece. An unknown outer slope drops its line.
            let t = if s_ac.is_finite() && s_db.is_finite() {
                ((fd - fc + s_ac * c - s_db * d) / (s_ac - s_db)).clamp(c, d)
            } else if s_ac.is_finite() {
                d
            } else {
                c
            };
            let l1 = if s_ac.is_finite() {
                fc + s_ac * (t - c)
            } else {
                f64::NEG_INFINITY
            };
            let l2 = if s_db.is_finite() {
                fd + s_db * (t - d)
            } else {
                f64::NEG_INFINITY
            };
            l1.max(l2)
        }
    } else {
        fc.min(fd)
    };
    lb_left.min(lb_mid).min(lb_right)
}

impl Overlaps3D for Cone3D {
    #[inline]
    fn overlaps_box(&self, bx: Box3D) -> bool {
        self.overlaps_box(bx)
    }

    #[inline]
    fn contains_box(&self, bx: Box3D) -> bool {
        self.contains_box(bx)
    }

    #[inline]
    fn bounding_box_hint(&self) -> Option<Box3D> {
        // The apex and the base disc, the disc bounded by its radius on every
        // axis: loose, but only the traversal form reads it.
        let r = self.height * self.tan_half;
        let base = [0, 1, 2].map(|d| self.apex[d] + self.axis[d] * self.height);
        let lo = [0, 1, 2].map(|d| self.apex[d].min(base[d] - r));
        let hi = [0, 1, 2].map(|d| self.apex[d].max(base[d] + r));
        Some(Box3D::new(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Index3DBuilder;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Whether some point of a dense grid over the box (corners, edges and
    /// faces included) lies in the cone: a certificate of overlap that cannot
    /// be wrong, only incomplete.
    fn grid_hits(cone: &Cone3D, bx: Box3D, steps: usize) -> bool {
        let at = |i: usize, lo: f64, hi: f64| lo + (hi - lo) * i as f64 / steps as f64;
        (0..=steps).any(|i| {
            (0..=steps).any(|j| {
                (0..=steps).any(|k| {
                    cone.contains_point([
                        at(i, bx.min_x, bx.max_x),
                        at(j, bx.min_y, bx.max_y),
                        at(k, bx.min_z, bx.max_z),
                    ])
                })
            })
        })
    }

    fn random_cone(rng: &mut StdRng) -> Cone3D {
        Cone3D::try_new(
            [
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
            ],
            [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ],
            rng.random_range(0.05..1.4),
            rng.random_range(5.0..30.0),
        )
        .unwrap()
    }

    #[test]
    fn overlap_is_certified_by_grid_points_and_refuted_by_an_inflated_cone() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut thin = 0;
        for _ in 0..1500 {
            let cone = random_cone(&mut rng);
            let (x, y, z) = (
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
            );
            let bx = Box3D::new(x, y, z, x + 3.0, y + 3.0, z + 3.0);
            let got = cone.overlaps_box(bx);
            // A grid point inside the cone is proof of overlap: never missed.
            if grid_hits(&cone, bx, 24) {
                assert!(got, "missed overlap: cone {cone:?} box {bx:?}");
                continue;
            }
            if got {
                // Claimed overlap without a grid witness: the intersection is
                // thinner than the grid step. A cone widened by 0.05 rad and
                // lengthened by 0.5 units must then catch a grid point; and a
                // box shrunk by a grid step must lose the overlap.
                let fat = Cone3D::try_new(
                    [
                        cone.apex[0] - 0.5 * cone.axis[0],
                        cone.apex[1] - 0.5 * cone.axis[1],
                        cone.apex[2] - 0.5 * cone.axis[2],
                    ],
                    cone.axis,
                    (cone.half_angle + 0.05).min(1.5),
                    cone.height + 1.0,
                )
                .unwrap();
                assert!(
                    grid_hits(&fat, bx, 48),
                    "overlap far from the surface: cone {cone:?} box {bx:?}"
                );
                thin += 1;
            }
        }
        assert!(thin < 80, "too many thin cases: {thin}");
    }

    #[test]
    fn overlap_agrees_with_corner_containment() {
        // A box inside the cone overlaps it; the two tests must never disagree
        // in that direction.
        let mut rng = StdRng::seed_from_u64(13);
        let mut inside = 0;
        for _ in 0..3000 {
            let cone = random_cone(&mut rng);
            let t: f64 = rng.random_range(0.0..cone.height);
            let c = [
                cone.apex[0] + t * cone.axis[0] + rng.random_range(-2.0..2.0),
                cone.apex[1] + t * cone.axis[1] + rng.random_range(-2.0..2.0),
                cone.apex[2] + t * cone.axis[2] + rng.random_range(-2.0..2.0),
            ];
            let s: f64 = rng.random_range(0.01..1.0);
            let bx = Box3D::new(c[0], c[1], c[2], c[0] + s, c[1] + s, c[2] + s);
            if cone.contains_box(bx) {
                inside += 1;
                assert!(cone.overlaps_box(bx), "cone {cone:?} box {bx:?}");
            }
        }
        assert!(inside > 100, "the sample never landed inside: {inside}");
    }

    #[test]
    fn a_box_inside_the_axis_ball_but_outside_the_cone_is_rejected() {
        // The naive test compares the axis point's distance to the box with
        // the disk radius, i.e. against a *ball*; this box sits inside that
        // ball (radius 5·tan 0.5 ≈ 2.73 around height 5) but outside the cone
        // (at height 3 the cone's radius is ≈ 1.64, the box is 1.9 out).
        let cone = Cone3D::try_new([0.0; 3], [0.0, 0.0, 1.0], 0.5, 10.0).unwrap();
        let bx = Box3D::new(1.89, -0.01, 2.99, 1.91, 0.01, 3.01);
        assert!(!cone.overlaps_box(bx));
        // Moving it onto the cone's radius makes it a hit.
        let on = Box3D::new(1.60, -0.01, 2.99, 1.65, 0.01, 3.01);
        assert!(cone.overlaps_box(on));
    }

    #[test]
    fn boxes_behind_the_apex_or_beyond_the_base_are_rejected() {
        let cone = Cone3D::try_new([0.0; 3], [0.0, 0.0, 1.0], 0.8, 10.0).unwrap();
        assert!(!cone.overlaps_box(Box3D::new(-1.0, -1.0, -3.0, 1.0, 1.0, -0.1)));
        assert!(!cone.overlaps_box(Box3D::new(-1.0, -1.0, 10.1, 1.0, 1.0, 12.0)));
        // Touching the apex from behind counts (closed cone).
        assert!(cone.overlaps_box(Box3D::new(-1.0, -1.0, -3.0, 1.0, 1.0, 0.0)));
        // The base cap is inside too.
        assert!(cone.overlaps_box(Box3D::new(-1.0, -1.0, 10.0, 1.0, 1.0, 12.0)));
        // Axis-aligned slab straddling the cone at mid height.
        assert!(cone.overlaps_box(Box3D::new(-50.0, -50.0, 4.0, 50.0, 50.0, 6.0)));
        // Same slab, but shifted sideways out of reach: at height 6 the radius
        // is 6·tan 0.8 ≈ 6.18.
        assert!(!cone.overlaps_box(Box3D::new(6.3, -50.0, 4.0, 50.0, 50.0, 6.0)));
        assert!(cone.overlaps_box(Box3D::new(6.1, -50.0, 4.0, 50.0, 50.0, 6.0)));
    }

    #[test]
    fn search_agrees_with_a_bruteforce_filter() {
        let mut rng = StdRng::seed_from_u64(12);
        let items: Vec<Box3D> = (0..300)
            .map(|_| {
                let (x, y, z) = (
                    rng.random_range(-30.0..30.0),
                    rng.random_range(-30.0..30.0),
                    rng.random_range(-30.0..30.0),
                );
                Box3D::new(x, y, z, x + 2.0, y + 2.0, z + 2.0)
            })
            .collect();
        let mut b = Index3DBuilder::new(items.len());
        for &bx in &items {
            b.add(bx);
        }
        let index = b.finish().unwrap();
        for _ in 0..30 {
            let cone = Cone3D::try_new(
                [rng.random_range(-20.0..20.0); 3],
                [rng.random_range(-1.0..1.0); 3],
                rng.random_range(0.1..1.0),
                rng.random_range(10.0..50.0),
            )
            .unwrap();
            let want: Vec<usize> = (0..items.len())
                .filter(|&i| cone.overlaps_box(items[i]))
                .collect();
            let mut got = index.search(&cone);
            got.sort_unstable();
            assert_eq!(got, want, "cone {cone:?}");
        }
    }

    #[test]
    fn convex_lower_bound_never_exceeds_the_sampled_minimum() {
        // Random convex functions: sums of |t - k|·w and a quadratic, on random
        // brackets with a golden-section split; the bound must sit at or
        // below a dense sample minimum, and be tight when the function is
        // linear (equal secants).
        let mut rng = StdRng::seed_from_u64(21);
        for _ in 0..5000 {
            let q: f64 = rng.random_range(0.0..2.0);
            let kinks: Vec<(f64, f64)> = (0..3)
                .map(|_| (rng.random_range(-5.0..5.0), rng.random_range(0.0..3.0)))
                .collect();
            let lin: f64 = rng.random_range(-4.0..4.0);
            let f = |t: f64| {
                q * t * t + lin * t + kinks.iter().map(|&(k, w)| w * (t - k).abs()).sum::<f64>()
            };
            let a: f64 = rng.random_range(-6.0..4.0);
            let b: f64 = a + rng.random_range(0.0..6.0);
            let c = b - INV_PHI * (b - a);
            let d = a + INV_PHI * (b - a);
            let lb = convex_min_lower_bound(a, f(a), c, f(c), d, f(d), b, f(b));
            let sampled = (0..=2000)
                .map(|i| f(a + (b - a) * i as f64 / 2000.0))
                .fold(f64::INFINITY, f64::min);
            assert!(lb <= sampled + 1e-9, "lb {lb} above sampled {sampled}");
        }
        // Linear: the bound is exact.
        let f = |t: f64| 3.0 * t + 1.0;
        let (a, b) = (0.0, 1.0);
        let (c, d) = (b - INV_PHI, INV_PHI);
        let lb = convex_min_lower_bound(a, f(a), c, f(c), d, f(d), b, f(b));
        assert!((lb - 1.0).abs() < 1e-12, "{lb}");
    }

    #[test]
    fn validation_rejects_degenerate_cones() {
        assert_eq!(
            Cone3D::try_new([0.0; 3], [0.0, 0.0, 0.0], 0.5, 10.0),
            Err(Cone3DError::ZeroDirection)
        );
        assert_eq!(
            Cone3D::try_new([0.0; 3], [0.0, 1.0, 0.0], 0.0, 10.0),
            Err(Cone3DError::HalfAngle(0.0))
        );
        assert_eq!(
            Cone3D::try_new([0.0; 3], [0.0, 1.0, 0.0], 0.5, -1.0),
            Err(Cone3DError::Height(-1.0))
        );
        assert!(Cone3D::try_new([0.0; 3], [0.0, 2.0, 0.0], 0.5, 10.0).is_ok());
    }
}
