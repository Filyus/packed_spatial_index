//! A 2D convex polygon query region (the half-plane generalization of
//! [`Triangle2D`](crate::Triangle2D)): N vertices, exact box-vs-polygon SAT.

use crate::geometry::{Box2D, Overlaps2D};

/// A convex polygon in 2D, given as vertices in order (CW or CCW). The query
/// predicates assume convexity; a non-convex polygon yields unspecified results.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexPolygon2D {
    verts: Vec<[f64; 2]>,
}

impl ConvexPolygon2D {
    /// Build from vertices in boundary order (CW or CCW).
    pub fn new(verts: Vec<[f64; 2]>) -> Self {
        Self { verts }
    }

    /// The vertices.
    pub fn vertices(&self) -> &[[f64; 2]] {
        &self.verts
    }

    /// Twice the signed area (shoelace); zero ⇒ degenerate (collinear/empty).
    fn signed_area2(&self) -> f64 {
        let v = &self.verts;
        let mut a = 0.0;
        for i in 0..v.len() {
            let p = v[i];
            let q = v[(i + 1) % v.len()];
            a += p[0] * q[1] - q[0] * p[1];
        }
        a
    }

    /// Whether this polygon's filled area overlaps the axis-aligned box `bx`
    /// (exact separating-axis test: the box's two axes plus each edge normal).
    pub fn overlaps_box(&self, bx: Box2D) -> bool {
        let v = &self.verts;
        if v.len() < 3 {
            return false;
        }
        // Box axes.
        let (mut lox, mut hix) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut loy, mut hiy) = (f64::INFINITY, f64::NEG_INFINITY);
        for w in v {
            lox = lox.min(w[0]);
            hix = hix.max(w[0]);
            loy = loy.min(w[1]);
            hiy = hiy.max(w[1]);
        }
        if hix < bx.min_x || lox > bx.max_x || hiy < bx.min_y || loy > bx.max_y {
            return false;
        }
        // Derived f64 rounding margin for edge-normal dot products
        // (`8 * f64::EPSILON * (sx*|nx| + sy*|ny|)`), preventing false misses
        // on oblique touching edges.
        let sx = lox
            .abs()
            .max(hix.abs())
            .max(bx.min_x.abs())
            .max(bx.max_x.abs());
        let sy = loy
            .abs()
            .max(hiy.abs())
            .max(bx.min_y.abs())
            .max(bx.max_y.abs());

        // For a non-degenerate convex polygon, each oriented edge defines a
        // supporting half-plane containing the whole polygon, so SAT reduces to
        // O(N) p-vertex half-plane tests (one per edge, no inner vertex loop).
        let area2 = self.signed_area2();
        if area2 != 0.0 {
            let sign = if area2 > 0.0 { 1.0 } else { -1.0 };
            for i in 0..v.len() {
                let p = v[i];
                let q = v[(i + 1) % v.len()];
                // Outward normal: points away from the polygon interior.
                let nx = sign * (q[1] - p[1]);
                let ny = sign * (p[0] - q[0]);
                let bx_lo = if nx >= 0.0 { bx.min_x } else { bx.max_x };
                let by_lo = if ny >= 0.0 { bx.min_y } else { bx.max_y };
                let blo = bx_lo * nx + by_lo * ny;
                let edge_d = (p[0] * nx + p[1] * ny).max(q[0] * nx + q[1] * ny);
                let pad = 8.0 * f64::EPSILON * (sx * nx.abs() + sy * ny.abs());
                if blo > edge_d + pad {
                    return false;
                }
            }
            return true;
        }

        // Degenerate (zero-area / collinear) polygon: test the single line
        // direction against the box in O(N).
        for i in 0..v.len() {
            let p = v[i];
            let q = v[(i + 1) % v.len()];
            let (ax, ay) = (-(q[1] - p[1]), q[0] - p[0]);
            if ax == 0.0 && ay == 0.0 {
                continue;
            }
            let (bx_lo, bx_hi) = if ax >= 0.0 {
                (bx.min_x, bx.max_x)
            } else {
                (bx.max_x, bx.min_x)
            };
            let (by_lo, by_hi) = if ay >= 0.0 {
                (bx.min_y, bx.max_y)
            } else {
                (bx.max_y, bx.min_y)
            };
            let blo = bx_lo * ax + by_lo * ay;
            let bhi = bx_hi * ax + by_hi * ay;
            let line_d = p[0] * ax + p[1] * ay;
            let pad = 8.0 * f64::EPSILON * (sx * ax.abs() + sy * ay.abs());
            if line_d + pad < blo || line_d - pad > bhi {
                return false;
            }
            break;
        }
        true
    }

    /// Whether the box `bx` lies entirely inside the polygon (all four corners
    /// inside). Always `false` for a degenerate (zero-area) polygon.
    pub fn contains_box(&self, bx: Box2D) -> bool {
        if self.verts.len() < 3 || self.signed_area2() == 0.0 {
            return false;
        }
        self.contains_point([bx.min_x, bx.min_y])
            && self.contains_point([bx.max_x, bx.min_y])
            && self.contains_point([bx.min_x, bx.max_y])
            && self.contains_point([bx.max_x, bx.max_y])
    }

    fn contains_point(&self, p: [f64; 2]) -> bool {
        let v = &self.verts;
        let (mut has_neg, mut has_pos) = (false, false);
        for i in 0..v.len() {
            let a = v[i];
            let b = v[(i + 1) % v.len()];
            let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            if cross < 0.0 {
                has_neg = true;
            } else if cross > 0.0 {
                has_pos = true;
            }
            if has_neg && has_pos {
                return false;
            }
        }
        true
    }
}

impl Overlaps2D for ConvexPolygon2D {
    #[inline]
    fn overlaps_box(&self, bx: Box2D) -> bool {
        self.overlaps_box(bx)
    }

    #[inline]
    fn contains_box(&self, bx: Box2D) -> bool {
        self.contains_box(bx)
    }

    #[inline]
    fn bounding_box_hint(&self) -> Option<Box2D> {
        let mut out = crate::geometry::empty_box2d();
        for &[x, y] in &self.verts {
            out = Box2D::new(
                out.min_x.min(x),
                out.min_y.min(y),
                out.max_x.max(x),
                out.max_y.max(y),
            );
        }
        Some(out)
    }
}
