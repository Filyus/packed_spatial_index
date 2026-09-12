//! Capsule regions: a segment thickened by a radius — a thick ray, or picking
//! with a tolerance in world units instead of the pixel frustum.
//!
//! The capsule is the set of points within `radius` of a segment, so the
//! broad-phase question "could the box touch the capsule?" is an exact
//! **segment-to-box distance**: zero when the segment enters the box, else the
//! true clearance. The distance is computed exactly, by the convexity of
//! `dist(P(t), box)` along the segment: the point `P(t)` is affine in `t`, the
//! distance to an axis-aligned box is convex, so the minimum over `t` in
//! `[0, 1]` is found by splitting the parameter at the (few) slab crossings
//! and minimizing the resulting quadratic piece analytically. No iteration,
//! no tolerance.
//!
//! The containment test is the exact mirror: the farthest point of an
//! axis-aligned box from the segment lies at a corner (distance to the segment
//! is convex, so its max over a box is at a vertex), which is what lets a
//! subtree fully inside the capsule be accepted whole.

use crate::geometry::{Box2D, Box3D, Overlaps2D, Overlaps3D};

/// Squared distance from point `p` to the axis-aligned box, per axis gaps.
#[cfg(test)]
#[inline]
fn point_box_gap_squared_2d(px: f64, py: f64, bx: Box2D) -> f64 {
    let dx = (bx.min_x - px).max(px - bx.max_x).max(0.0);
    let dy = (bx.min_y - py).max(py - bx.max_y).max(0.0);
    dx * dx + dy * dy
}

#[cfg(test)]
#[inline]
fn point_box_gap_squared_3d(px: f64, py: f64, pz: f64, bx: Box3D) -> f64 {
    let dx = (bx.min_x - px).max(px - bx.max_x).max(0.0);
    let dy = (bx.min_y - py).max(py - bx.max_y).max(0.0);
    let dz = (bx.min_z - pz).max(pz - bx.max_z).max(0.0);
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from point `p` to segment `ab`.
#[inline]
fn point_segment_gap_squared_2d(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (ax + t * dx, ay + t * dy);
    (px - qx) * (px - qx) + (py - qy) * (py - qy)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn point_segment_gap_squared_3d(
    px: f64,
    py: f64,
    pz: f64,
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
) -> f64 {
    let (dx, dy, dz) = (bx - ax, by - ay, bz - az);
    let len2 = dx * dx + dy * dy + dz * dz;
    let t = if len2 > 0.0 {
        (((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy, qz) = (ax + t * dx, ay + t * dy, az + t * dz);
    (px - qx) * (px - qx) + (py - qy) * (py - qy) + (pz - qz) * (pz - qz)
}

/// Exact minimum of `gap(P(t), box)²` for `t` in `[t0, t1]`, where `P(t)` is
/// the affine segment. Per axis the gap is convex piecewise linear with one
/// kink where the segment crosses a slab boundary, so splitting at the kinks
/// leaves pieces on which every axis gap is a fixed-sign affine form and the
/// squared distance is a plain quadratic — minimized analytically at its
/// vertex, clamped to the piece.
pub(crate) fn min_segment_box_gap_squared(
    a: &[f64; 3],
    b: &[f64; 3],
    lo: &[f64; 3],
    hi: &[f64; 3],
    dims: usize,
    t0: f64,
    t1: f64,
) -> f64 {
    let mut cuts = [t0, t1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut cut_len = 2;
    for i in 0..dims {
        let w = b[i] - a[i];
        if w != 0.0 {
            for bound in [lo[i], hi[i]] {
                let t = (bound - a[i]) / w;
                if t > t0 && t < t1 {
                    cuts[cut_len] = t;
                    cut_len += 1;
                }
            }
        }
    }
    cuts[..cut_len].sort_by(|x, y| x.total_cmp(y));

    let mut best = f64::INFINITY;
    for piece in cuts[..cut_len].windows(2) {
        let (t_a, t_b) = (piece[0], piece[1]);
        // On this piece each axis gap is `g0 + g1*t` with a fixed sign:
        // `g0 = sign*(a_i - bound)`, `g1 = sign*w`, `sign` in `{-1, +1}`; an
        // axis inside its slab contributes nothing. The squared distance is
        // then the quadratic `q*t² + l*t + c`.
        let (mut q, mut l) = (0.0f64, 0.0f64);
        let mut inside_axes = 0usize;
        for i in 0..dims {
            let w = b[i] - a[i];
            let p_mid = a[i] + (t_a + t_b) * 0.5 * w;
            let (bound, sign) = if p_mid >= lo[i] && p_mid <= hi[i] {
                inside_axes += 1;
                continue;
            } else if p_mid < lo[i] {
                (lo[i], 1.0)
            } else {
                (hi[i], -1.0)
            };
            let g0 = sign * (a[i] - bound);
            let g1 = sign * w;
            q += g1 * g1;
            l += 2.0 * g0 * g1;
        }
        if inside_axes == dims {
            // Every axis inside its slab on this piece: the segment passes
            // through the box.
            return 0.0;
        }
        // Minimize `q*t² + l*t + c` on `[t_a, t_b]`. `q` is zero only when
        // every outside axis has a stationary segment (a degenerate point
        // capsule); then the gap does not depend on `t` at all.
        let mut candidates = [t_a, t_b, t_b];
        if q > 0.0 {
            candidates[1] = (-l / (2.0 * q)).clamp(t_a, t_b);
        }
        for &t in &candidates {
            let mut f = 0.0;
            for i in 0..dims {
                let p = a[i] + t * (b[i] - a[i]);
                f += (lo[i] - p).max(p - hi[i]).max(0.0).powi(2);
            }
            best = best.min(f);
        }
    }
    best
}

/// A thick segment in 2D: every point within `radius` of `ab`.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box2D, Capsule2D, Index2DBuilder};
///
/// let mut b = Index2DBuilder::new(2);
/// b.add(Box2D::new(0.0, 4.9, 10.0, 5.1));
/// b.add(Box2D::new(50.0, 50.0, 51.0, 51.0));
/// let index = b.finish().unwrap();
///
/// // A thick ray along the y axis, tolerance 0.5 in world units.
/// let ray = Capsule2D::new([0.0, 0.0], [0.0, 10.0], 0.5);
/// assert_eq!(index.search(&ray), vec![0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule2D {
    /// Segment start.
    pub a: [f64; 2],
    /// Segment end.
    pub b: [f64; 2],
    /// Thickness, in coordinate units.
    pub radius: f64,
}

impl Capsule2D {
    /// A capsule around segment `a`-`b` with the given radius.
    pub fn new(a: [f64; 2], b: [f64; 2], radius: f64) -> Self {
        Self { a, b, radius }
    }

    /// Exact segment-to-box distance, squared; zero when they intersect.
    pub fn segment_box_gap_squared(&self, bx: Box2D) -> f64 {
        min_segment_box_gap_squared(
            &[self.a[0], self.a[1], 0.0],
            &[self.b[0], self.b[1], 0.0],
            &[bx.min_x, bx.min_y, 0.0],
            &[bx.max_x, bx.max_y, 0.0],
            2,
            0.0,
            1.0,
        )
    }

    /// Whether any part of `bx` lies within `radius` of the segment.
    pub fn overlaps_box(&self, bx: Box2D) -> bool {
        self.segment_box_gap_squared(bx) <= self.radius * self.radius
    }

    /// Whether all of `bx` lies within `radius` of the segment: the farthest
    /// box point from the segment is a corner.
    pub fn contains_box(&self, bx: Box2D) -> bool {
        let r2 = self.radius * self.radius;
        let (ax, ay, bxx, by) = (self.a[0], self.a[1], self.b[0], self.b[1]);
        [
            [bx.min_x, bx.min_y],
            [bx.max_x, bx.min_y],
            [bx.min_x, bx.max_y],
            [bx.max_x, bx.max_y],
        ]
        .into_iter()
        .all(|[x, y]| point_segment_gap_squared_2d(x, y, ax, ay, bxx, by) <= r2)
    }
}

impl Overlaps2D for Capsule2D {
    #[inline]
    fn overlaps_box(&self, bx: Box2D) -> bool {
        self.overlaps_box(bx)
    }

    #[inline]
    fn contains_box(&self, bx: Box2D) -> bool {
        self.contains_box(bx)
    }
}

/// A thick segment in 3D — the world-unit-tolerance picking shape.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, Capsule3D, Index3DBuilder};
///
/// let mut b = Index3DBuilder::new(2);
/// b.add(Box3D::new(0.0, 4.8, 0.0, 1.0, 5.2, 1.0));
/// b.add(Box3D::new(50.0, 50.0, 50.0, 51.0, 51.0, 51.0));
/// let index = b.finish().unwrap();
///
/// let ray = Capsule3D::new([0.0, 0.0, 0.0], [0.0, 10.0, 0.0], 0.5);
/// assert_eq!(index.search(&ray), vec![0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule3D {
    /// Segment start.
    pub a: [f64; 3],
    /// Segment end.
    pub b: [f64; 3],
    /// Thickness, in coordinate units.
    pub radius: f64,
}

impl Capsule3D {
    /// A capsule around segment `a`-`b` with the given radius.
    pub fn new(a: [f64; 3], b: [f64; 3], radius: f64) -> Self {
        Self { a, b, radius }
    }

    /// Exact segment-to-box distance, squared; zero when they intersect.
    pub fn segment_box_gap_squared(&self, bx: Box3D) -> f64 {
        min_segment_box_gap_squared(
            &self.a,
            &self.b,
            &[bx.min_x, bx.min_y, bx.min_z],
            &[bx.max_x, bx.max_y, bx.max_z],
            3,
            0.0,
            1.0,
        )
    }

    /// Whether any part of `bx` lies within `radius` of the segment.
    pub fn overlaps_box(&self, bx: Box3D) -> bool {
        self.segment_box_gap_squared(bx) <= self.radius * self.radius
    }

    /// Whether all of `bx` lies within `radius` of the segment.
    pub fn contains_box(&self, bx: Box3D) -> bool {
        let r2 = self.radius * self.radius;
        let [ax, ay, az] = self.a;
        let [bxx, by, bz] = self.b;
        [
            [bx.min_x, bx.min_y, bx.min_z],
            [bx.max_x, bx.min_y, bx.min_z],
            [bx.min_x, bx.max_y, bx.min_z],
            [bx.max_x, bx.max_y, bx.min_z],
            [bx.min_x, bx.min_y, bx.max_z],
            [bx.max_x, bx.min_y, bx.max_z],
            [bx.min_x, bx.max_y, bx.max_z],
            [bx.max_x, bx.max_y, bx.max_z],
        ]
        .into_iter()
        .all(|[x, y, z]| point_segment_gap_squared_3d(x, y, z, ax, ay, az, bxx, by, bz) <= r2)
    }
}

impl Overlaps3D for Capsule3D {
    #[inline]
    fn overlaps_box(&self, bx: Box3D) -> bool {
        self.overlaps_box(bx)
    }

    #[inline]
    fn contains_box(&self, bx: Box3D) -> bool {
        self.contains_box(bx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Index2DBuilder, Index3DBuilder};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Brute-force segment-box distance: sample the segment densely and take
    /// the smallest point-to-box gap. The analytic minimum must never sit
    /// above the sample (it must be a true minimum) and tracks it closely.
    fn brute_gap_2d(cap: &Capsule2D, bx: Box2D, steps: usize) -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let px = cap.a[0] + t * (cap.b[0] - cap.a[0]);
            let py = cap.a[1] + t * (cap.b[1] - cap.a[1]);
            best = best.min(point_box_gap_squared_2d(px, py, bx));
        }
        best
    }

    fn brute_gap_3d(cap: &Capsule3D, bx: Box3D, steps: usize) -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let px = cap.a[0] + t * (cap.b[0] - cap.a[0]);
            let py = cap.a[1] + t * (cap.b[1] - cap.a[1]);
            let pz = cap.a[2] + t * (cap.b[2] - cap.a[2]);
            best = best.min(point_box_gap_squared_3d(px, py, pz, bx));
        }
        best
    }

    #[test]
    fn analytic_gap_tracks_a_dense_sample_and_is_never_above_it() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..2000 {
            let cap = Capsule2D::new(
                [rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)],
                [rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)],
                rng.random_range(0.0..3.0),
            );
            let x: f64 = rng.random_range(-12.0..12.0);
            let y: f64 = rng.random_range(-12.0..12.0);
            let bx = Box2D::new(x, y, x + 4.0, y + 4.0);
            let analytic = cap.segment_box_gap_squared(bx);
            let sampled = brute_gap_2d(&cap, bx, 400);
            assert!(
                analytic <= sampled + 1e-9,
                "analytic {analytic} above sampled {sampled}"
            );
            assert!(
                analytic >= sampled - 1.0,
                "analytic {analytic} far below sampled {sampled}"
            );
        }
    }

    #[test]
    fn analytic_gap_3d_tracks_a_dense_sample_and_is_never_above_it() {
        let mut rng = StdRng::seed_from_u64(10);
        for _ in 0..1000 {
            let cap = Capsule3D::new(
                [
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                ],
                [
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                ],
                rng.random_range(0.0..3.0),
            );
            let (x, y, z) = (
                rng.random_range(-12.0..12.0),
                rng.random_range(-12.0..12.0),
                rng.random_range(-12.0..12.0),
            );
            let bx = Box3D::new(x, y, z, x + 3.0, y + 3.0, z + 3.0);
            let analytic = cap.segment_box_gap_squared(bx);
            let sampled = brute_gap_3d(&cap, bx, 200);
            assert!(
                analytic <= sampled + 1e-9,
                "analytic {analytic} above sampled {sampled}"
            );
            assert!(
                analytic >= sampled - 1.5,
                "analytic {analytic} far below sampled {sampled}"
            );
        }
    }

    #[test]
    fn capsule_search_agrees_with_a_bruteforce_filter() {
        let mut rng = StdRng::seed_from_u64(8);
        let items: Vec<Box2D> = (0..500)
            .map(|_| {
                let x: f64 = rng.random_range(-50.0..50.0);
                let y: f64 = rng.random_range(-50.0..50.0);
                Box2D::new(x, y, x + 3.0, y + 3.0)
            })
            .collect();
        let mut b = Index2DBuilder::new(items.len());
        for &bx in &items {
            b.add(bx);
        }
        let index = b.finish().unwrap();
        for _ in 0..50 {
            let cap = Capsule2D::new(
                [rng.random_range(-40.0..40.0), rng.random_range(-40.0..40.0)],
                [rng.random_range(-40.0..40.0), rng.random_range(-40.0..40.0)],
                rng.random_range(0.0..8.0),
            );
            let want: Vec<usize> = (0..items.len())
                .filter(|&i| cap.overlaps_box(items[i]))
                .collect();
            let mut got = index.search(&cap);
            got.sort_unstable();
            assert_eq!(got, want, "capsule {cap:?}");
        }
    }

    #[test]
    fn capsule_3d_search_agrees_with_a_bruteforce_filter() {
        let mut rng = StdRng::seed_from_u64(9);
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
            let cap = Capsule3D::new(
                [
                    rng.random_range(-25.0..25.0),
                    rng.random_range(-25.0..25.0),
                    rng.random_range(-25.0..25.0),
                ],
                [
                    rng.random_range(-25.0..25.0),
                    rng.random_range(-25.0..25.0),
                    rng.random_range(-25.0..25.0),
                ],
                rng.random_range(0.0..5.0),
            );
            let want: Vec<usize> = (0..items.len())
                .filter(|&i| cap.overlaps_box(items[i]))
                .collect();
            let mut got = index.search(&cap);
            got.sort_unstable();
            assert_eq!(got, want, "capsule {cap:?}");
        }
    }

    #[test]
    fn degenerate_point_capsule_is_a_point_query() {
        let cap = Capsule2D::new([5.0, 5.0], [5.0, 5.0], 0.0);
        assert!(cap.overlaps_box(Box2D::new(5.0, 5.0, 6.0, 6.0)));
        assert!(cap.overlaps_box(Box2D::new(4.0, 4.0, 5.0, 5.0)));
        assert!(!cap.overlaps_box(Box2D::new(5.1, 5.0, 6.0, 6.0)));
    }
}
