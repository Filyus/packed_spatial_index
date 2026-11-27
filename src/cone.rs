//! A solid cone in 3D: apex, axis direction, half-angle and height — the
//! sensor-FOV / spotlight query shape.
//!
//! The box-vs-cone overlap test reduces to a closed-form piecewise-quadratic
//! problem along the axis. Slice the box by the plane at height `t` above the
//! apex: the closest point of `box ∩ plane(t)` to `apex + t·axis` satisfies the
//! KKT condition `q_i(s) = clamp(apex_i + s·axis_i, lo_i, hi_i)` for the unique
//! scalar `s` where `t(s) = axis · (q(s) - apex) = t`. Because `q(s)` is
//! piecewise linear with at most six kinks (where `apex_i + s·axis_i` crosses
//! `lo_i` or `hi_i`), both `t(s)` and `‖q(s) - apex‖²` are explicit polynomials
//! of degree ≤ 2 on each piece, and `‖q(s) - apex‖² · cos²(half_angle) - t(s)²
//! <= 0` is minimized in closed form on each clipped piece — no iteration.
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

    /// Whether the KKT point `q(s) = clamp(apex + s·axis, lo, hi)` lies inside
    /// the cone's disk at its height `t(s) = axis · (q(s) - apex)`.
    #[inline]
    fn kkt_point_in_cone(&self, s: f64, lo: &[f64; 3], hi: &[f64; 3]) -> bool {
        let a = self.axis;
        let q = [
            (self.apex[0] + s * a[0]).clamp(lo[0], hi[0]),
            (self.apex[1] + s * a[1]).clamp(lo[1], hi[1]),
            (self.apex[2] + s * a[2]).clamp(lo[2], hi[2]),
        ];
        let (dx, dy, dz) = (
            q[0] - self.apex[0],
            q[1] - self.apex[1],
            q[2] - self.apex[2],
        );
        let along = dx * a[0] + dy * a[1] + dz * a[2];
        // `s` is already clipped to `t(s) in [lo_t, hi_t] ⊆ [0, height]`, so
        // clamp tiny f64 rounding overshoot at the caps rather than dropping a
        // cap-tangent hit.
        let t = along.clamp(0.0, self.height);
        let (px, py, pz) = (dx - t * a[0], dy - t * a[1], dz - t * a[2]);
        let perp2 = px * px + py * py + pz * pz;
        let r = t * self.tan_half;
        // Derived rounding margin (`8u = 4 * f64::EPSILON`): covers `tan(half_angle)`
        // (1u), `t * tan` (2u), squaring (3u), and `perp2` (3u) so an exact
        // tangent such as `half_angle = pi/4` is never dropped by 1 ULP.
        const RADIUS_SQ_UPPER: f64 = 1.0 + 4.0 * f64::EPSILON;
        perp2 <= (r * r) * RADIUS_SQ_UPPER
    }

    /// Whether any part of `bx` intersects the solid cone.
    pub fn overlaps_box(&self, bx: Box3D) -> bool {
        let lo = [bx.min_x, bx.min_y, bx.min_z];
        let hi = [bx.max_x, bx.max_y, bx.max_z];
        let (t_min, t_max) = self.height_range(&lo, &hi);
        let (lo_t, hi_t) = (t_min.max(0.0), t_max.min(self.height));
        if lo_t > hi_t {
            // The box lies wholly behind the apex or beyond the base.
            return false;
        }
        let tan = self.tan_half;
        // Cheap decisions first; most boxes a traversal tests are nowhere near
        // the cone. A corner inside the cone settles it.
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

        // Exact closed-form piecewise-quadratic solution along the KKT curve
        // `q_i(s) = clamp(apex_i + s·a_i, lo_i, hi_i)`. Its kinks are sorted
        // once per box; on each piece `(s_a, s_b)`, free axes contribute
        // `alpha = sum a_i²` and clamped axes contribute `beta = sum a_i·(c_i - apex_i)`
        // so `t(s) = alpha·s + beta`, and `‖q(s) - apex‖²·cos²θ - t(s)²` is a
        // quadratic with leading coefficient `alpha·(cos²θ - alpha)` and linear
        // coefficient `-2·alpha·beta`, minimized analytically at `s_lo`, `s_hi`,
        // or (when convex) `s* = beta / (cos²θ - alpha)`.
        let a = self.axis;
        let mut kinks = [0.0f64; 6];
        let mut n = 0usize;
        for i in 0..3 {
            if a[i] != 0.0 {
                kinks[n] = (lo[i] - self.apex[i]) / a[i];
                kinks[n + 1] = (hi[i] - self.apex[i]) / a[i];
                n += 2;
            }
        }
        kinks[..n].sort_by(|x, y| x.total_cmp(y));

        // Flat box perpendicular to the axis: every non-zero axis has
        // `lo[i] == hi[i]`, so all kinks collapse to one `s` and the box lies
        // wholly in the plane at height `lo_t == hi_t`.
        if n > 0 && kinks[0] == kinks[n - 1] {
            return self.kkt_point_in_cone(kinks[0], &lo, &hi);
        }

        let c2 = self.cos_half * self.cos_half;
        for k in 0..n.saturating_sub(1) {
            let (s_a, s_b) = (kinks[k], kinks[k + 1]);
            if s_b <= s_a {
                continue;
            }
            let s_mid = 0.5 * (s_a + s_b);
            let mut alpha = 0.0;
            let mut beta = 0.0;
            for i in 0..3 {
                if a[i] == 0.0 {
                    let c = self.apex[i].clamp(lo[i], hi[i]);
                    // `a[i] == 0` contributes 0 to both `alpha` and `beta`;
                    // `kkt_point_in_cone` evaluates the exact perpendicular gap.
                    let _ = c;
                } else {
                    let v = self.apex[i] + s_mid * a[i];
                    if v < lo[i] {
                        beta += a[i] * (lo[i] - self.apex[i]);
                    } else if v > hi[i] {
                        beta += a[i] * (hi[i] - self.apex[i]);
                    } else {
                        alpha += a[i] * a[i];
                    }
                }
            }
            if alpha == 0.0 {
                continue;
            }
            let s_lo = s_a.max((lo_t - beta) / alpha);
            let s_hi = s_b.min((hi_t - beta) / alpha);
            if s_lo > s_hi {
                continue;
            }
            if self.kkt_point_in_cone(s_lo, &lo, &hi) || self.kkt_point_in_cone(s_hi, &lo, &hi) {
                return true;
            }
            // When `c2 > alpha`, the quadratic is strictly convex in `s` with
            // unique interior stationary point `s* = beta / (c2 - alpha)`.
            if c2 > alpha {
                let s_star = beta / (c2 - alpha);
                if s_star > s_lo && s_star < s_hi && self.kkt_point_in_cone(s_star, &lo, &hi) {
                    return true;
                }
            }
        }
        false
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
    fn oblique_cone_tangent_edge_hit_without_corners_inside() {
        // Box edge grazing the cone's lateral surface while all 8 corners lie
        // outside the cone and the axis misses the box (exercises the interior
        // quadratic vertex `s_star`).
        let cone =
            Cone3D::try_new([0.0; 3], [0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_4, 10.0).unwrap();
        // At z = 5, radius = 5. Line x = 5, y in [-10, 10], z = 5 grazes at (5, 0, 5),
        // while corners at y = ±10 are at distance sqrt(125) > 5 (outside).
        let grazing = Box3D::new(5.0, -10.0, 5.0, 6.0, 10.0, 5.0);
        assert!(cone.overlaps_box(grazing));
        let outside = Box3D::new(5.01, -10.0, 5.0, 6.0, 10.0, 5.0);
        assert!(!cone.overlaps_box(outside));

        // Oblique cone (all 3 axis components non-zero) grazing a box edge with
        // all 8 corners outside and axis missing the box.
        let oblique = Cone3D::try_new([0.0; 3], [1.0, 1.0, 1.0], 0.35, 20.0).unwrap();
        // Unit axis a = (1,1,1)/sqrt(3). At t = 6, axis point is (2*sqrt(3)) * (1,1,1).
        // Perpendicular unit vector u = (1,-1,0)/sqrt(2), radius = 6 * tan(0.35).
        let s3 = 3.0_f64.sqrt();
        let s2 = 2.0_f64.sqrt();
        let r = 6.0 * 0.35_f64.tan();
        let p = [6.0 / s3 + r / s2, 6.0 / s3 - r / s2, 6.0 / s3];
        // A thin box along the third orthogonal direction v = (1,1,-2)/sqrt(6)
        // centered just inside `p` (all corners far away along z).
        let hit_box = Box3D::new(
            p[0] - 0.05,
            p[1] - 0.05,
            p[2] - 4.0,
            p[0] + 0.05,
            p[1] + 0.05,
            p[2] + 4.0,
        );
        assert!(oblique.overlaps_box(hit_box));
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
