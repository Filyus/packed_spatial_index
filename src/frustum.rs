//! A 3D view frustum: six inward-pointing planes for conservative culling.
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
