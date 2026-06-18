//! Fixed-width triangle payload records, in `f64` (default) or `f32` (compact).
//!
//! A triangle is a natural **fixed-width** payload: 6 (2D) or 9 (3D) coordinates,
//! the same size for every item. Stored through the serializer's fixed-width
//! path (`.triangles(..)` / `.records(..)`) it needs no offset table, so the file
//! is smaller and a view can borrow the records as a zero-copy typed slice. Pair
//! a triangle payload with an index built over the triangles' bounding boxes
//! ([`Index3D::from_triangles`](crate::Index3D::from_triangles)) to get a
//! streamable bounding-volume hierarchy over a mesh.
//!
//! Following the crate convention ([`Box3D`] / [`SimdIndex3D`](crate::SimdIndex3D)
//! are `f64`, the `F32` suffix is the compact variant), [`Triangle3D`] stores
//! `f64` vertices and [`Triangle3DF32`] stores `f32`. The `f32` ray-triangle path
//! uses an explicit SIMD kernel with the `simd` feature; the `f64` path is scalar.

use crate::geometry::{Box2D, Box3D};

mod sealed {
    pub trait Sealed {}
}

/// A ray-triangle intersection: the `index` of the hit triangle in the queried
/// slice, and the ray parameter `t` in direction-length units (the same `t`
/// convention as [`crate::Ray3D`]). Returned by [`crate::Ray3D::closest_triangle`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleHit {
    /// Index of the hit triangle in the slice passed to the query.
    pub index: usize,
    /// Ray parameter at the hit, in direction-length units.
    pub t: f64,
}

/// A 2D triangle record (`f64` or `f32` vertices). Implemented by [`Triangle2D`]
/// and [`Triangle2DF32`]; sealed, so the set of record types is fixed.
pub trait Triangle2: Copy + sealed::Sealed {
    /// Byte stride of one record.
    const STRIDE: usize;
    /// The triangle's axis-aligned bounding box (always `f64`, for the index).
    fn aabb(&self) -> Box2D;
    #[doc(hidden)]
    fn read_le(bytes: &[u8]) -> Self;
}

/// A 3D triangle record (`f64` or `f32` vertices). Implemented by [`Triangle3D`]
/// and [`Triangle3DF32`]; sealed, so the set of record types is fixed.
pub trait Triangle3: Copy + sealed::Sealed {
    /// Byte stride of one record.
    const STRIDE: usize;
    /// The triangle's axis-aligned bounding box (always `f64`, for the index).
    fn aabb(&self) -> Box3D;
    #[doc(hidden)]
    fn read_le(bytes: &[u8]) -> Self;
    /// Closest hit of the ray segment `(o, d, max_t)` over `tris`, computed in the
    /// record's own precision (`f32` uses the SIMD kernel under the `simd`
    /// feature). Crate-internal; drive it through [`crate::Ray3D::closest_triangle`].
    #[doc(hidden)]
    fn closest_hit(o: [f64; 3], d: [f64; 3], max_t: f64, tris: &[Self]) -> Option<TriangleHit>
    where
        Self: Sized;
}

macro_rules! tri2 {
    ($name:ident, $t:ty, $stride:expr) => {
        /// A 2D triangle: three vertices `[x, y]`. `repr(C)`, unpadded, so a slice
        /// casts to and from the on-disk fixed-width payload with no copy.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            /// First vertex `[x, y]`.
            pub a: [$t; 2],
            /// Second vertex `[x, y]`.
            pub b: [$t; 2],
            /// Third vertex `[x, y]`.
            pub c: [$t; 2],
        }

        impl $name {
            /// Create a triangle from three vertices.
            #[inline]
            pub const fn new(a: [$t; 2], b: [$t; 2], c: [$t; 2]) -> Self {
                Self { a, b, c }
            }
        }

        impl sealed::Sealed for $name {}
        impl Triangle2 for $name {
            const STRIDE: usize = $stride;
            #[inline]
            fn aabb(&self) -> Box2D {
                let min_x = self.a[0].min(self.b[0]).min(self.c[0]);
                let min_y = self.a[1].min(self.b[1]).min(self.c[1]);
                let max_x = self.a[0].max(self.b[0]).max(self.c[0]);
                let max_y = self.a[1].max(self.b[1]).max(self.c[1]);
                Box2D::new(min_x as f64, min_y as f64, max_x as f64, max_y as f64)
            }
            #[inline]
            fn read_le(b: &[u8]) -> Self {
                let mut v = [0 as $t; 6];
                read_le_into::<$t>(b, &mut v);
                Self {
                    a: [v[0], v[1]],
                    b: [v[2], v[3]],
                    c: [v[4], v[5]],
                }
            }
        }
    };
}

tri2!(Triangle2D, f64, 48);
tri2!(Triangle2DF32, f32, 24);

impl Triangle2D {
    /// Whether this triangle's filled area overlaps the axis-aligned box `bx`.
    ///
    /// The 2D separating-axis test: the box's two axes plus the triangle's three
    /// edge normals. This is the predicate behind `Index2D::search_triangle` /
    /// `any_triangle` / `visit_triangle`.
    #[inline]
    pub fn overlaps_box(&self, bx: Box2D) -> bool {
        let v = [self.a, self.b, self.c];
        // Box axes.
        let lo = v[0][0].min(v[1][0]).min(v[2][0]);
        let hi = v[0][0].max(v[1][0]).max(v[2][0]);
        if hi < bx.min_x || lo > bx.max_x {
            return false;
        }
        let lo = v[0][1].min(v[1][1]).min(v[2][1]);
        let hi = v[0][1].max(v[1][1]).max(v[2][1]);
        if hi < bx.min_y || lo > bx.max_y {
            return false;
        }
        // Triangle edge normals.
        let corners = [
            [bx.min_x, bx.min_y],
            [bx.max_x, bx.min_y],
            [bx.min_x, bx.max_y],
            [bx.max_x, bx.max_y],
        ];
        for e in 0..3 {
            let p = v[e];
            let q = v[(e + 1) % 3];
            let (ax, ay) = (-(q[1] - p[1]), q[0] - p[0]);
            let mut tlo = f64::INFINITY;
            let mut thi = f64::NEG_INFINITY;
            for w in &v {
                let d = w[0] * ax + w[1] * ay;
                tlo = tlo.min(d);
                thi = thi.max(d);
            }
            let mut blo = f64::INFINITY;
            let mut bhi = f64::NEG_INFINITY;
            for c in &corners {
                let d = c[0] * ax + c[1] * ay;
                blo = blo.min(d);
                bhi = bhi.max(d);
            }
            if thi < blo || tlo > bhi {
                return false;
            }
        }
        true
    }

    /// Whether the box `bx` lies entirely inside this triangle (all four corners
    /// inside the convex region). Used to accept a whole subtree without testing
    /// its leaves. Always `false` for a degenerate (zero-area) triangle.
    #[inline]
    pub fn contains_box(&self, bx: Box2D) -> bool {
        let area2 = (self.b[0] - self.a[0]) * (self.c[1] - self.a[1])
            - (self.c[0] - self.a[0]) * (self.b[1] - self.a[1]);
        if area2 == 0.0 {
            return false;
        }
        self.contains_point([bx.min_x, bx.min_y])
            && self.contains_point([bx.max_x, bx.min_y])
            && self.contains_point([bx.min_x, bx.max_y])
            && self.contains_point([bx.max_x, bx.max_y])
    }

    #[inline]
    fn contains_point(&self, p: [f64; 2]) -> bool {
        let edge = |a: [f64; 2], b: [f64; 2]| {
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        };
        let d1 = edge(self.a, self.b);
        let d2 = edge(self.b, self.c);
        let d3 = edge(self.c, self.a);
        let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(has_neg && has_pos)
    }
}

macro_rules! tri3 {
    ($name:ident, $t:ty, $stride:expr, $closest:path) => {
        /// A 3D triangle: three vertices `[x, y, z]`. `repr(C)`, unpadded, so a
        /// slice casts to and from the on-disk fixed-width payload with no copy.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            /// First vertex `[x, y, z]`.
            pub a: [$t; 3],
            /// Second vertex `[x, y, z]`.
            pub b: [$t; 3],
            /// Third vertex `[x, y, z]`.
            pub c: [$t; 3],
        }

        impl $name {
            /// Create a triangle from three vertices.
            #[inline]
            pub const fn new(a: [$t; 3], b: [$t; 3], c: [$t; 3]) -> Self {
                Self { a, b, c }
            }
        }

        impl sealed::Sealed for $name {}
        impl Triangle3 for $name {
            const STRIDE: usize = $stride;
            #[inline]
            fn aabb(&self) -> Box3D {
                let min_x = self.a[0].min(self.b[0]).min(self.c[0]);
                let min_y = self.a[1].min(self.b[1]).min(self.c[1]);
                let min_z = self.a[2].min(self.b[2]).min(self.c[2]);
                let max_x = self.a[0].max(self.b[0]).max(self.c[0]);
                let max_y = self.a[1].max(self.b[1]).max(self.c[1]);
                let max_z = self.a[2].max(self.b[2]).max(self.c[2]);
                Box3D::new(
                    min_x as f64,
                    min_y as f64,
                    min_z as f64,
                    max_x as f64,
                    max_y as f64,
                    max_z as f64,
                )
            }
            #[inline]
            fn read_le(b: &[u8]) -> Self {
                let mut v = [0 as $t; 9];
                read_le_into::<$t>(b, &mut v);
                Self {
                    a: [v[0], v[1], v[2]],
                    b: [v[3], v[4], v[5]],
                    c: [v[6], v[7], v[8]],
                }
            }
            #[inline]
            fn closest_hit(
                o: [f64; 3],
                d: [f64; 3],
                max_t: f64,
                tris: &[Self],
            ) -> Option<TriangleHit> {
                $closest(o, d, max_t, tris)
            }
        }
    };
}

tri3!(Triangle3D, f64, 72, closest_f64);
tri3!(Triangle3DF32, f32, 36, closest_f32);

/// Decode little-endian floats from `bytes` into `out`. `T` is `f32` or `f64`.
trait LeFloat: Copy {
    const SIZE: usize;
    fn from_le(b: &[u8]) -> Self;
}
impl LeFloat for f32 {
    const SIZE: usize = 4;
    #[inline]
    fn from_le(b: &[u8]) -> Self {
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}
impl LeFloat for f64 {
    const SIZE: usize = 8;
    #[inline]
    fn from_le(b: &[u8]) -> Self {
        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
}
#[inline]
fn read_le_into<T: LeFloat>(bytes: &[u8], out: &mut [T]) {
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = T::from_le(&bytes[i * T::SIZE..]);
    }
}

/// Reinterpret a triangle slice as its little-endian record bytes (no copy).
///
/// Sound because every triangle type is `repr(C)` over `f32`/`f64` only (no
/// padding, every bit pattern valid) and call sites bound `T` to the sealed
/// [`Triangle2`] / [`Triangle3`]. Produces the same bytes on little-endian
/// targets, which the format already assumes throughout.
#[inline]
pub(crate) fn records_as_bytes<T: Copy>(records: &[T]) -> &[u8] {
    // SAFETY: see the doc comment; `T` is a `repr(C)` all-float record type.
    unsafe {
        core::slice::from_raw_parts(
            records.as_ptr() as *const u8,
            std::mem::size_of_val(records),
        )
    }
}

/// Borrow a fixed-width blob region as a typed record slice, zero-copy, when the
/// bytes are aligned (e.g. an mmap). `None` if `blobs` is not suitably aligned.
#[inline]
pub(crate) fn blobs_as_records<T: Copy>(blobs: &[u8]) -> Option<&[T]> {
    // SAFETY: same record-type argument as `records_as_bytes`; `align_to` splits
    // any unaligned prefix/suffix and we accept only a clean whole-slice cast.
    let (prefix, mid, suffix) = unsafe { blobs.align_to::<T>() };
    (prefix.is_empty() && suffix.is_empty()).then_some(mid)
}

// ---- ray-triangle closest hit ----

/// Branchless Moller-Trumbore in `f64`: ray parameter `t`, or `+inf` on a miss.
#[inline(always)]
#[allow(clippy::manual_range_contains)] // branchless miss mask, kept for autovec
fn mt_f64(t: &Triangle3D, o: [f64; 3], d: [f64; 3], max_t: f64) -> f64 {
    let e1 = [t.b[0] - t.a[0], t.b[1] - t.a[1], t.b[2] - t.a[2]];
    let e2 = [t.c[0] - t.a[0], t.c[1] - t.a[1], t.c[2] - t.a[2]];
    let p = [
        d[1] * e2[2] - d[2] * e2[1],
        d[2] * e2[0] - d[0] * e2[2],
        d[0] * e2[1] - d[1] * e2[0],
    ];
    let det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    let inv = 1.0 / det;
    let s = [o[0] - t.a[0], o[1] - t.a[1], o[2] - t.a[2]];
    let u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
    let q = [
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0],
    ];
    let v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
    let t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
    let miss = (det.abs() < 1e-12)
        | (u < 0.0)
        | (u > 1.0)
        | (v < 0.0)
        | (u + v > 1.0)
        | (t < 0.0)
        | (t > max_t);
    if miss { f64::INFINITY } else { t }
}

fn closest_f64(o: [f64; 3], d: [f64; 3], max_t: f64, tris: &[Triangle3D]) -> Option<TriangleHit> {
    let mut best_t = f64::INFINITY;
    let mut best_i = usize::MAX;
    for (i, tri) in tris.iter().enumerate() {
        let t = mt_f64(tri, o, d, max_t);
        if t < best_t {
            best_t = t;
            best_i = i;
        }
    }
    (best_i != usize::MAX).then_some(TriangleHit {
        index: best_i,
        t: best_t,
    })
}

/// Branchless Moller-Trumbore in `f32`: ray parameter `t`, or `+inf` on a miss.
#[inline(always)]
#[cfg_attr(feature = "simd", allow(dead_code))]
#[allow(clippy::manual_range_contains)] // branchless miss mask, kept for autovec
fn mt_f32(t: &Triangle3DF32, o: [f32; 3], d: [f32; 3], max_t: f32) -> f32 {
    let e1 = [t.b[0] - t.a[0], t.b[1] - t.a[1], t.b[2] - t.a[2]];
    let e2 = [t.c[0] - t.a[0], t.c[1] - t.a[1], t.c[2] - t.a[2]];
    let p = [
        d[1] * e2[2] - d[2] * e2[1],
        d[2] * e2[0] - d[0] * e2[2],
        d[0] * e2[1] - d[1] * e2[0],
    ];
    let det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    let inv = 1.0 / det;
    let s = [o[0] - t.a[0], o[1] - t.a[1], o[2] - t.a[2]];
    let u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
    let q = [
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0],
    ];
    let v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
    let t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
    let miss = (det.abs() < 1e-8)
        | (u < 0.0)
        | (u > 1.0)
        | (v < 0.0)
        | (u + v > 1.0)
        | (t < 0.0)
        | (t > max_t);
    if miss { f32::INFINITY } else { t }
}

fn closest_f32(
    o: [f64; 3],
    d: [f64; 3],
    max_t: f64,
    tris: &[Triangle3DF32],
) -> Option<TriangleHit> {
    let o = [o[0] as f32, o[1] as f32, o[2] as f32];
    let d = [d[0] as f32, d[1] as f32, d[2] as f32];
    let max_t = max_t as f32;
    #[cfg(feature = "simd")]
    {
        closest_f32_simd(o, d, max_t, tris)
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut best_t = f32::INFINITY;
        let mut best_i = usize::MAX;
        for (i, tri) in tris.iter().enumerate() {
            let t = mt_f32(tri, o, d, max_t);
            if t < best_t {
                best_t = t;
                best_i = i;
            }
        }
        (best_i != usize::MAX).then_some(TriangleHit {
            index: best_i,
            t: best_t as f64,
        })
    }
}

/// SIMD `f32` closest hit: 8 triangles per iteration through `wide::f32x8`. The
/// AoS records are gathered into SoA lanes; tail lanes stay degenerate (miss).
#[cfg(feature = "simd")]
fn closest_f32_simd(
    o: [f32; 3],
    d: [f32; 3],
    max_t: f32,
    tris: &[Triangle3DF32],
) -> Option<TriangleHit> {
    use wide::f32x8;
    let inf = f32x8::splat(f32::INFINITY);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let eps = f32x8::splat(1e-8);
    let maxv = f32x8::splat(max_t);
    let ox = f32x8::splat(o[0]);
    let oy = f32x8::splat(o[1]);
    let oz = f32x8::splat(o[2]);
    let dx = f32x8::splat(d[0]);
    let dy = f32x8::splat(d[1]);
    let dz = f32x8::splat(d[2]);

    let mut best_t = f32::INFINITY;
    let mut best_i = usize::MAX;
    for (chunk_i, chunk) in tris.chunks(8).enumerate() {
        let mut g = [[0.0f32; 8]; 9];
        for (l, t) in chunk.iter().enumerate() {
            g[0][l] = t.a[0];
            g[1][l] = t.a[1];
            g[2][l] = t.a[2];
            g[3][l] = t.b[0];
            g[4][l] = t.b[1];
            g[5][l] = t.b[2];
            g[6][l] = t.c[0];
            g[7][l] = t.c[1];
            g[8][l] = t.c[2];
        }
        let ax = f32x8::from(g[0]);
        let ay = f32x8::from(g[1]);
        let az = f32x8::from(g[2]);
        let e1x = f32x8::from(g[3]) - ax;
        let e1y = f32x8::from(g[4]) - ay;
        let e1z = f32x8::from(g[5]) - az;
        let e2x = f32x8::from(g[6]) - ax;
        let e2y = f32x8::from(g[7]) - ay;
        let e2z = f32x8::from(g[8]) - az;

        let px = dy * e2z - dz * e2y;
        let py = dz * e2x - dx * e2z;
        let pz = dx * e2y - dy * e2x;
        let det = e1x * px + e1y * py + e1z * pz;
        let inv = one / det;
        let sx = ox - ax;
        let sy = oy - ay;
        let sz = oz - az;
        let u = (sx * px + sy * py + sz * pz) * inv;
        let qx = sy * e1z - sz * e1y;
        let qy = sz * e1x - sx * e1z;
        let qz = sx * e1y - sy * e1x;
        let v = (dx * qx + dy * qy + dz * qz) * inv;
        let t = (e2x * qx + e2y * qy + e2z * qz) * inv;

        let miss = det.abs().simd_lt(eps)
            | u.simd_lt(zero)
            | u.simd_gt(one)
            | v.simd_lt(zero)
            | (u + v).simd_gt(one)
            | t.simd_lt(zero)
            | t.simd_gt(maxv);
        let t = miss.blend(inf, t);

        let base = chunk_i * 8;
        for (l, &tl) in t.to_array().iter().enumerate() {
            if tl < best_t {
                best_t = tl;
                best_i = base + l;
            }
        }
    }
    (best_i != usize::MAX).then_some(TriangleHit {
        index: best_i,
        t: best_t as f64,
    })
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    #[test]
    fn f32_simd_matches_scalar() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0x7711);
        let tris: Vec<Triangle3DF32> = (0..1500)
            .map(|_| {
                let c = [
                    rng.random_range(0.0..10.0f32),
                    rng.random_range(0.0..10.0),
                    rng.random_range(0.0..10.0),
                ];
                let mut v = || {
                    [
                        c[0] + rng.random_range(-1.5..1.5f32),
                        c[1] + rng.random_range(-1.5..1.5),
                        c[2] + rng.random_range(-1.5..1.5),
                    ]
                };
                Triangle3DF32::new(v(), v(), v())
            })
            .collect();
        for _ in 0..300 {
            let o = [
                rng.random_range(0.0..10.0f32),
                rng.random_range(0.0..10.0),
                rng.random_range(0.0..10.0),
            ];
            let d = [
                rng.random_range(-1.0..1.0f32),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ];
            let simd = closest_f32_simd(o, d, 100.0, &tris);
            let mut best_t = f32::INFINITY;
            let mut best_i = usize::MAX;
            for (i, tri) in tris.iter().enumerate() {
                let t = mt_f32(tri, o, d, 100.0);
                if t < best_t {
                    best_t = t;
                    best_i = i;
                }
            }
            let scalar = (best_i != usize::MAX).then_some(best_i);
            // When both hit, indices/params agree; a rare grazing-edge split is OK.
            if let (Some(s), Some(v)) = (scalar, simd) {
                assert!(
                    s == v.index || (best_t as f64 - v.t).abs() < 1e-3,
                    "scalar {s} vs simd {v:?}"
                );
            }
        }
    }
}
