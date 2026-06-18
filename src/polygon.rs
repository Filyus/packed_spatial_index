//! A 2D convex polygon query region (the half-plane generalization of
//! [`Triangle2D`](crate::Triangle2D)): N vertices, exact box-vs-polygon SAT.

use crate::geometry::Box2D;

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
        // Polygon edge normals.
        let corners = [
            [bx.min_x, bx.min_y],
            [bx.max_x, bx.min_y],
            [bx.min_x, bx.max_y],
            [bx.max_x, bx.max_y],
        ];
        for i in 0..v.len() {
            let p = v[i];
            let q = v[(i + 1) % v.len()];
            let (ax, ay) = (-(q[1] - p[1]), q[0] - p[0]);
            let mut tlo = f64::INFINITY;
            let mut thi = f64::NEG_INFINITY;
            for w in v {
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
