//! A 3D view frustum: six inward-pointing planes for conservative culling — and,
//! narrowed to the pixels around a cursor, for picking and rubber-band selection
//! (see [`Frustum3D`]).
//!
//! [`Frustum3D`] can be queried with [`Index3D::search`](crate::Index3D::search). The query is *conservative*: it returns every item whose box
//! overlaps the frustum, and may include a few boxes that lie just outside an
//! edge or corner (the standard frustum-culling p-vertex test). It never drops a
//! box that is actually visible, which is what culling needs — an extra box is
//! cheap to reject downstream; a missing one is a hole in the frame.
//!
//! [`Frustum3D::bounding_box`] computes the frustum's axis-aligned bounding box
//! from its eight corner points, for callers that want a coarse candidate box
//! before applying the tighter frustum test.

use crate::geometry::{Box3D, Overlaps3D};
use crate::ray::Ray3D;

/// The normalized-device-coordinate depth range a projection matrix targets, for
/// [`Frustum3D::from_view_projection`]. D3D12, Vulkan, Metal and WebGPU clip `z`
/// to `[0, 1]` (the modern default); OpenGL and WebGL clip it to `[-1, 1]`. Only
/// the near plane differs between the two conventions. There is deliberately no
/// silent default on the constructor — the convention is not recoverable from the
/// matrix, so every caller states it — but [`ClipSpaceZ::default()`] is the
/// modern `ZeroToOne` if you need one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClipSpaceZ {
    /// D3D12 / Vulkan / Metal / WebGPU clip space: `0 <= z <= w`. The modern
    /// majority, so the [`Default`].
    #[default]
    ZeroToOne,
    /// OpenGL / WebGL clip space: `-w <= z <= w`.
    NegOneToOne,
}

/// A 3D frustum as six inward-pointing half-space planes.
///
/// Each plane is `[a, b, c, d]`; a point `p` is *inside* that plane when
/// `a*p.x + b*p.y + c*p.z + d >= 0`, and inside the frustum when it is inside all
/// six. The planes need not be normalized — only the sign of the plane equation
/// is used.
///
/// # Picking a click, not just culling a camera
///
/// A frustum does not have to be the camera's. Narrow one to the few pixels
/// around the cursor and the same query becomes 3D picking: every object whose
/// box could be under the click, found by tree traversal instead of a scan over
/// the scene. A dragged rectangle (rubber-band selection) is the same
/// construction with a bigger rectangle.
///
/// Scale the view-projection so the clicked NDC rectangle fills the clip cube —
/// the classic pick matrix — then read the planes off that:
///
/// ```
/// use packed_spatial_index::{Box3D, ClipSpaceZ, Frustum3D, Index3DBuilder};
///
/// # let mut b = Index3DBuilder::new(2);
/// # b.add(Box3D::new(-0.1, -0.1, -0.5, 0.1, 0.1, 0.5)); // under the cursor
/// # b.add(Box3D::new(5.0, 5.0, 0.0, 6.0, 6.0, 1.0)); // far off to one side
/// # let index = b.finish().unwrap();
/// // The camera's view-projection, row-major (transpose a column-major one).
/// let vp = [
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
/// ];
///
/// // The clicked rectangle in NDC — here a small tolerance around the cursor.
/// let (x0, x1, y0, y1) = (-0.05, 0.05, -0.05, 0.05);
/// let (sx, sy) = (2.0 / (x1 - x0), 2.0 / (y1 - y0));
/// let (tx, ty) = (-(x1 + x0) / (x1 - x0), -(y1 + y0) / (y1 - y0));
///
/// // pick = S * vp, where S maps that rectangle onto the whole clip cube.
/// let blend = |r: [f64; 4], s: f64, t: f64| {
///     let w = vp[3];
///     [r[0] * s + w[0] * t, r[1] * s + w[1] * t, r[2] * s + w[2] * t, r[3] * s + w[3] * t]
/// };
/// let pick = [blend(vp[0], sx, tx), blend(vp[1], sy, ty), vp[2], vp[3]];
///
/// let under_cursor = Frustum3D::from_view_projection(pick, ClipSpaceZ::NegOneToOne);
/// assert_eq!(index.search(&under_cursor), vec![0]);
/// ```
///
/// Two things this gives you and one it does not. It gives the **candidate
/// set** — conservatively, so an object just outside the tolerance may be in it,
/// and none that should be there is missing — and it gives it in
/// insertion-order ids, cheaply, without touching the rest of the scene. It does
/// **not** give you the winner: results are bounding boxes in unspecified order,
/// so "nearest to the camera" is a second step. Resolve it with
/// [`Index3D::raycast_closest`](crate::Index3D::raycast_closest) along the
/// cursor ray, which returns the first box the ray enters and the distance to
/// it, or by testing the candidates against your real geometry — the boxes are
/// a broad phase, and a click that lands inside an object's box but beside the
/// object itself is a hit here and a miss there.
///
/// For a single-pixel pick with no tolerance, the ray alone is the more direct
/// tool; reach for the frustum when the click has a radius, when the selection
/// is a dragged rectangle, or when you want every candidate rather than the
/// nearest one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum3D {
    planes: [[f64; 4]; 6],
}

impl Frustum3D {
    /// Build from six explicit inward-pointing planes (`[a, b, c, d]` each).
    #[inline]
    pub const fn from_planes(planes: [[f64; 4]; 6]) -> Self {
        Self { planes }
    }

    /// Extract the six frustum planes from a **row-major** view-projection matrix
    /// `vp` via the Gribb-Hartmann method.
    ///
    /// `vp[i][j]` is row `i`, column `j`; a world point `[x, y, z]` maps to clip
    /// space as `clip_i = vp[i][0]*x + vp[i][1]*y + vp[i][2]*z + vp[i][3]`. `clip`
    /// is the NDC depth range your projection targets — pass
    /// [`ClipSpaceZ::NegOneToOne`] for OpenGL / WebGL or [`ClipSpaceZ::ZeroToOne`]
    /// for D3D12 / Vulkan / Metal / WebGPU; it changes only the near plane.
    /// Engines that store the matrix column-major (e.g. `glam`, `cgmath`) should
    /// pass the transpose.
    pub fn from_view_projection(vp: [[f64; 4]; 4], clip: ClipSpaceZ) -> Self {
        let row = |i: usize| vp[i];
        let add = |a: [f64; 4], b: [f64; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f64; 4], b: [f64; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let near = match clip {
            ClipSpaceZ::ZeroToOne => r2, // D3D/Vulkan/Metal/WebGPU: clip_z >= 0
            ClipSpaceZ::NegOneToOne => add(r3, r2), // OpenGL: clip_w + clip_z >= 0
        };
        Self {
            planes: [
                add(r3, r0), // left
                sub(r3, r0), // right
                add(r3, r1), // bottom
                sub(r3, r1), // top
                near,        // near
                sub(r3, r2), // far
            ],
        }
    }

    /// The pixel frustum of a pick ray: four side planes through the ray
    /// origin at `half_angle`, a near plane at `near`, and the far plane at the
    /// ray's own `max_distance`.
    ///
    /// This is the region of [`Index3D::search_pick`](crate::Index3D::search_pick):
    /// a click in a viewport turns the screen pixel into a truncated pyramid
    /// through the camera, and this constructor builds it from the pixel's
    /// central ray plus the click's angular tolerance. The direction need not
    /// be normalized — the pyramid's shape is what matters, not the ray's
    /// parameterization (the hit keys stay in the ray's direction-length units).
    ///
    /// `half_angle` is the **half** of the cone's opening, the angular radius
    /// the pyramid spans on each side of the ray: a box at depth `t` is inside
    /// when its perpendicular offset is at most `t * tan(half_angle)`. From the
    /// pixel's full angular size `delta` (what one pixel of the viewport
    /// subtends), pass `half_angle = delta / 2`; from a perspective projection
    /// with vertical FOV `fov` over a viewport `h` pixels tall,
    /// `delta = 2 * atan(2 * tan(fov / 2) / h)`.
    ///
    /// Fails when the ray's direction is zero or non-finite, when `half_angle`
    /// is not in `(0°, 90°)`, or when `near` is negative, non-finite, or at or
    /// beyond the ray's `max_distance` (the pyramid would be empty).
    pub fn try_from_ray(ray: Ray3D, half_angle: f64, near: f64) -> Result<Self, FrustumRayError> {
        let dir = [ray.dir_x, ray.dir_y, ray.dir_z];
        let mag = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if mag <= 0.0 || !mag.is_finite() {
            return Err(FrustumRayError::ZeroDirection);
        }
        if !(half_angle > 0.0 && half_angle < std::f64::consts::FRAC_PI_2 && half_angle.is_finite())
        {
            return Err(FrustumRayError::HalfAngle(half_angle));
        }
        if !(near >= 0.0 && near.is_finite() && near < ray.max_distance) {
            return Err(FrustumRayError::Near(near));
        }
        let u = cross3(dir, [0.0, 0.0, 1.0]);
        // A ray parallel to +z: the z-up reference gives no right vector, so
        // fall back to x — any frame perpendicular to the direction is fine.
        let u = if u[0] * u[0] + u[1] * u[1] + u[2] * u[2] > 0.0 {
            u
        } else {
            cross3(dir, [1.0, 0.0, 0.0])
        };
        let u = norm3(u);
        let v = norm3(cross3(u, dir));
        let o = [ray.origin.x, ray.origin.y, ray.origin.z];
        let dot = |p: [f64; 3], q: [f64; 3]| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
        let (sa, ca) = (half_angle.sin(), half_angle.cos());
        let side = |e: [f64; 3]| -> [f64; 4] {
            let n = norm3([
                sa * dir[0] - ca * e[0],
                sa * dir[1] - ca * e[1],
                sa * dir[2] - ca * e[2],
            ]);
            [n[0], n[1], n[2], -dot(n, o)]
        };
        let near_plane = [dir[0], dir[1], dir[2], -dot(dir, o) - near];
        let far_plane = [-dir[0], -dir[1], -dir[2], dot(dir, o) + ray.max_distance];
        Ok(Self {
            planes: [
                side(u),
                side([-u[0], -u[1], -u[2]]),
                side(v),
                side([-v[0], -v[1], -v[2]]),
                near_plane,
                far_plane,
            ],
        })
    }

    /// The six planes, in `[left, right, bottom, top, near, far]` order when built
    /// by [`from_view_projection`](Self::from_view_projection).
    #[inline]
    pub fn planes(&self) -> &[[f64; 4]; 6] {
        &self.planes
    }

    /// Conservative overlap: `false` only when the box lies entirely outside some
    /// plane. Uses the p-vertex shortcut (the box corner most positive along each
    /// plane normal), so it may return `true` for a box just outside a frustum
    /// edge or corner — never `false` for a box that truly overlaps.
    #[inline]
    pub fn overlaps_box(&self, b: Box3D) -> bool {
        for p in &self.planes {
            let px = if p[0] >= 0.0 { b.max_x } else { b.min_x };
            let py = if p[1] >= 0.0 { b.max_y } else { b.min_y };
            let pz = if p[2] >= 0.0 { b.max_z } else { b.min_z };
            if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0 {
                return false;
            }
        }
        true
    }

    /// Whether the box lies entirely inside the frustum (every corner inside every
    /// plane, via the n-vertex shortcut). Used to accept a whole subtree without
    /// testing its leaves. Exact (no false positives).
    #[inline]
    pub fn contains_box(&self, b: Box3D) -> bool {
        for p in &self.planes {
            let nx = if p[0] >= 0.0 { b.min_x } else { b.max_x };
            let ny = if p[1] >= 0.0 { b.min_y } else { b.max_y };
            let nz = if p[2] >= 0.0 { b.min_z } else { b.max_z };
            if p[0] * nx + p[1] * ny + p[2] * nz + p[3] < 0.0 {
                return false;
            }
        }
        true
    }

    /// The frustum's axis-aligned bounding box, computed from its eight corner
    /// points.
    ///
    /// Each corner is the intersection of one plane from `{planes()[0],
    /// planes()[1]}`, one from `{planes()[2], planes()[3]}`, and one from
    /// `{planes()[4], planes()[5]}` — the pairing [`from_view_projection`]
    /// produces (`[left, right, bottom, top, near, far]`). This is only guaranteed
    /// to be a meaningful frustum shape for that pairing; a [`from_planes`]
    /// frustum built from six arbitrary inward planes has no guaranteed
    /// left/right/bottom/top/near/far structure, so the eight "corners" computed
    /// here may not form the frustum's actual convex hull in that case.
    ///
    /// Returns `None` if any corner's three planes are near-parallel or otherwise
    /// degenerate (the 3-plane intersection is singular), rather than returning a
    /// silently-wrong box. The degeneracy test is scale-invariant — it compares
    /// the normalized triple product of the plane normals, so a valid frustum
    /// whose planes are uniformly scaled (planes need not be normalized) is not
    /// falsely reported degenerate.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Frustum3D};
    ///
    /// let f = Frustum3D::from_planes([
    ///     [1.0, 0.0, 0.0, -0.0],  // x >= 0
    ///     [-1.0, 0.0, 0.0, 1.0],  // x <= 1
    ///     [0.0, 1.0, 0.0, -0.0],  // y >= 0
    ///     [0.0, -1.0, 0.0, 1.0],  // y <= 1
    ///     [0.0, 0.0, 1.0, -0.0],  // z >= 0
    ///     [0.0, 0.0, -1.0, 1.0],  // z <= 1
    /// ]);
    /// assert_eq!(f.bounding_box(), Some(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)));
    /// ```
    ///
    /// [`from_planes`]: Self::from_planes
    /// [`from_view_projection`]: Self::from_view_projection
    pub fn bounding_box(&self) -> Option<Box3D> {
        // Relative threshold: `det` is the scalar triple product of the three
        // plane normals, which scales with the product of their magnitudes.
        // Comparing `|det|` against `EPS * |n0| * |n1| * |n2|` tests the triple
        // product of the *unit* normals (|sin| of the solid angle they span), so
        // it is invariant to how the planes are scaled.
        const EPS: f64 = 1e-9;

        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let norm = |a: [f64; 3]| dot(a, a).sqrt();

        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];

        for &i0 in &[0usize, 1] {
            for &i1 in &[2usize, 3] {
                for &i2 in &[4usize, 5] {
                    let p0 = self.planes[i0];
                    let p1 = self.planes[i1];
                    let p2 = self.planes[i2];
                    let n0 = [p0[0], p0[1], p0[2]];
                    let n1 = [p1[0], p1[1], p1[2]];
                    let n2 = [p2[0], p2[1], p2[2]];
                    let (d0, d1, d2) = (p0[3], p1[3], p2[3]);

                    let n1xn2 = cross(n1, n2);
                    let det = dot(n0, n1xn2);
                    let scale = norm(n0) * norm(n1) * norm(n2);
                    if scale == 0.0 || det.abs() < EPS * scale {
                        return None;
                    }

                    let n2xn0 = cross(n2, n0);
                    let n0xn1 = cross(n0, n1);
                    let corner = [
                        -(d0 * n1xn2[0] + d1 * n2xn0[0] + d2 * n0xn1[0]) / det,
                        -(d0 * n1xn2[1] + d1 * n2xn0[1] + d2 * n0xn1[1]) / det,
                        -(d0 * n1xn2[2] + d1 * n2xn0[2] + d2 * n0xn1[2]) / det,
                    ];

                    for axis in 0..3 {
                        min[axis] = min[axis].min(corner[axis]);
                        max[axis] = max[axis].max(corner[axis]);
                    }
                }
            }
        }

        Some(Box3D::new(min[0], min[1], min[2], max[0], max[1], max[2]))
    }
}

impl Overlaps3D for Frustum3D {
    #[inline]
    fn overlaps_box(&self, bx: Box3D) -> bool {
        self.overlaps_box(bx)
    }

    #[inline]
    fn contains_box(&self, bx: Box3D) -> bool {
        self.contains_box(bx)
    }
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

/// Why [`Frustum3D::try_from_ray`] refused its inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrustumRayError {
    /// The ray's direction is zero or non-finite, so no perpendicular frame
    /// exists.
    ZeroDirection,
    /// The half-angle is not in `(0°, 90°)` (or not finite).
    HalfAngle(f64),
    /// The near distance is negative, non-finite, or at or beyond the ray's
    /// `max_distance`, which would make the pyramid empty.
    Near(f64),
}

impl std::fmt::Display for FrustumRayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDirection => write!(f, "the ray direction is zero or non-finite"),
            Self::HalfAngle(a) => write!(
                f,
                "half-angle must be in (0, pi/2) degrees included, got {a}"
            ),
            Self::Near(n) => write!(
                f,
                "near must be finite, non-negative and below the ray's max_distance, got {n}"
            ),
        }
    }
}

impl std::error::Error for FrustumRayError {}

#[cfg(test)]
mod ray_ctor_tests {
    use super::*;
    use crate::geometry::Point3D;
    use crate::ray::Ray3D;

    #[test]
    fn planes_contain_the_axis_and_exclude_off_axis_points() {
        let ray = Ray3D::new(
            Point3D {
                x: -1.0,
                y: 2.0,
                z: 3.0,
            },
            0.0,
            1.0,
            0.0,
            100.0,
        );
        let fr = Frustum3D::try_from_ray(ray, 0.1_f64.to_radians(), 1.0).unwrap();
        // Points along the axis, between near and far, are inside.
        for t in [1.0f64, 50.0, 99.5] {
            let p = [ray.origin.x, ray.origin.y + t, ray.origin.z];
            for plane in fr.planes() {
                assert!(plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3] >= 0.0);
            }
        }
        // Far past max_distance: outside the far plane; behind near: outside.
        for t in [100.5f64, 0.5] {
            let p = [ray.origin.x, ray.origin.y + t, ray.origin.z];
            assert!(!fr.overlaps_box(Box3D::new(
                p[0] - 0.01,
                p[1] - 0.01,
                p[2] - 0.01,
                p[0] + 0.01,
                p[1] + 0.01,
                p[2] + 0.01
            )));
        }
        // Half the tolerance away at half the far distance: inside the side
        // planes (10 world units sideways at t=50, half-angle 0.1 rad).
        let fr = Frustum3D::try_from_ray(ray, 0.1, 1.0).unwrap();
        let side = 50.0 * 0.1f64.tan() * 0.9;
        assert!(fr.overlaps_box(Box3D::new(
            ray.origin.x - side - 1.0,
            51.0,
            ray.origin.z - 1.0,
            ray.origin.x - side + 1.0,
            52.0,
            ray.origin.z + 1.0
        )));
        // Twice the tolerance away: excluded by a side plane.
        assert!(!fr.overlaps_box(Box3D::new(
            ray.origin.x - 2.0 * side - 1.0,
            51.0,
            ray.origin.z - 1.0,
            ray.origin.x - 2.0 * side + 1.0,
            52.0,
            ray.origin.z + 1.0
        )));
    }

    #[test]
    fn rejects_zero_direction_bad_angle_and_bad_near() {
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
        assert_eq!(
            Frustum3D::try_from_ray(
                Ray3D::new(
                    Point3D {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0
                    },
                    0.0,
                    0.0,
                    0.0,
                    100.0
                ),
                0.1,
                1.0
            ),
            Err(FrustumRayError::ZeroDirection)
        );
        assert_eq!(
            Frustum3D::try_from_ray(ray, 0.0, 1.0),
            Err(FrustumRayError::HalfAngle(0.0))
        );
        assert_eq!(
            Frustum3D::try_from_ray(ray, 1.0_f64.to_radians(), 100.0),
            Err(FrustumRayError::Near(100.0))
        );
        assert_eq!(
            Frustum3D::try_from_ray(ray, 1.0_f64.to_radians(), -1.0),
            Err(FrustumRayError::Near(-1.0))
        );
    }

    #[test]
    fn a_ray_parallel_to_z_still_gets_a_frame() {
        let ray = Ray3D::new(
            Point3D {
                x: 0.0,
                y: 0.0,
                z: -10.0,
            },
            0.0,
            0.0,
            1.0,
            100.0,
        );
        let fr = Frustum3D::try_from_ray(ray, 0.2_f64.to_radians(), 1.0).unwrap();
        assert!(fr.overlaps_box(Box3D::new(-1.0, -1.0, 49.0, 1.0, 1.0, 51.0)));
    }
}
