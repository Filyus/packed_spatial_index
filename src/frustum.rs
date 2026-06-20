//! A 3D view frustum: six inward-pointing planes for conservative culling.
//!
//! [`Frustum3D`] backs [`Index3D::search_frustum`](crate::Index3D::search_frustum)
//! and friends. The query is *conservative*: it returns every item whose box
//! overlaps the frustum, and may include a few boxes that lie just outside an
//! edge or corner (the standard frustum-culling p-vertex test). It never drops a
//! box that is actually visible, which is what culling needs — an extra box is
//! cheap to reject downstream; a missing one is a hole in the frame.

use crate::geometry::Box3D;

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
}
