//! Half-space regions: everything on one side of a line (2D) or plane (3D).
//!
//! A half-space is the predicate the other region shapes cannot express: it is
//! **unbounded**, so there is no bounding box to pre-filter with and no
//! grown-window workaround to fall back on — the tree traversal itself is the
//! whole answer. That is the BIM / medical / geology cross-section case: cut
//! everything on one side of a plane, without first bounding an infinite
//! region.
//!
//! The query rides the ordinary region machinery through
//! [`Overlaps2D`](crate::Overlaps2D) / [`Overlaps3D`](crate::Overlaps3D): the
//! node prune is the two-corner sign test (the extreme corner along the
//! normal), and its mirror gives the whole-subtree containment test, so a
//! subtree fully inside the half-space is accepted in O(1).

use crate::geometry::{Box2D, Box3D, Overlaps2D, Overlaps3D};

/// Everything on one side of a line in 2D.
///
/// A point `p` is *inside* when `nx*p.x + ny*p.y + d >= 0`. The normal need
/// not be normalized — only the sign of the plane function is used, at the
/// box corners that are extreme along it.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box2D, HalfSpace2D, Index2DBuilder};
///
/// let mut b = Index2DBuilder::new(2);
/// b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
/// let index = b.finish().unwrap();
///
/// // Everything right of x = 3.
/// let right_of_3 = HalfSpace2D::new(1.0, 0.0, -3.0);
/// assert_eq!(index.search(&right_of_3), vec![1]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfSpace2D {
    /// Normal x component.
    pub nx: f64,
    /// Normal y component.
    pub ny: f64,
    /// Plane offset.
    pub d: f64,
}

impl HalfSpace2D {
    /// A half-space bounded by the line `nx*x + ny*y + d = 0`, keeping the
    /// side where the plane function is non-negative.
    pub fn new(nx: f64, ny: f64, d: f64) -> Self {
        Self { nx, ny, d }
    }

    /// The plane function's maximum over the box's corners: the extreme
    /// corner along the normal.
    #[inline]
    fn plane_max(&self, bx: Box2D) -> f64 {
        self.d
            + self.nx * if self.nx >= 0.0 { bx.max_x } else { bx.min_x }
            + self.ny * if self.ny >= 0.0 { bx.max_y } else { bx.min_y }
    }

    /// The plane function's minimum over the box's corners.
    #[inline]
    fn plane_min(&self, bx: Box2D) -> f64 {
        self.d
            + self.nx * if self.nx >= 0.0 { bx.min_x } else { bx.max_x }
            + self.ny * if self.ny >= 0.0 { bx.min_y } else { bx.max_y }
    }

    /// Whether any part of `bx` is inside the half-space: the two-corner sign
    /// test. A box the line does not cross is decided by one extreme corner.
    pub fn overlaps_box(&self, bx: Box2D) -> bool {
        self.plane_max(bx) >= 0.0
    }

    /// Whether all of `bx` is inside the half-space: the mirror corner test,
    /// which is what lets a subtree fully inside the region be accepted whole.
    pub fn contains_box(&self, bx: Box2D) -> bool {
        self.plane_min(bx) >= 0.0
    }
}

impl Overlaps2D for HalfSpace2D {
    #[inline]
    fn overlaps_box(&self, bx: Box2D) -> bool {
        self.overlaps_box(bx)
    }

    #[inline]
    fn contains_box(&self, bx: Box2D) -> bool {
        self.contains_box(bx)
    }
}

/// Everything on one side of a plane in 3D.
///
/// A point `p` is *inside* when `nx*p.x + ny*p.y + nz*p.z + d >= 0`. The
/// normal need not be normalized. See
/// [`HalfSpace2D`](crate::HalfSpace2D) for the 2D form.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, HalfSpace3D, Index3DBuilder};
///
/// let mut b = Index3DBuilder::new(2);
/// b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
/// b.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
/// let index = b.finish().unwrap();
///
/// // Everything above z = 3: the geological cut plane.
/// let above = HalfSpace3D::new(0.0, 0.0, 1.0, -3.0);
/// assert_eq!(index.search(&above), vec![1]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfSpace3D {
    /// Normal x component.
    pub nx: f64,
    /// Normal y component.
    pub ny: f64,
    /// Normal z component.
    pub nz: f64,
    /// Plane offset.
    pub d: f64,
}

impl HalfSpace3D {
    /// A half-space bounded by the plane `nx*x + ny*y + nz*z + d = 0`,
    /// keeping the side where the plane function is non-negative.
    pub fn new(nx: f64, ny: f64, nz: f64, d: f64) -> Self {
        Self { nx, ny, nz, d }
    }

    /// The plane function's maximum over the box's corners.
    #[inline]
    fn plane_max(&self, bx: Box3D) -> f64 {
        self.d
            + self.nx * if self.nx >= 0.0 { bx.max_x } else { bx.min_x }
            + self.ny * if self.ny >= 0.0 { bx.max_y } else { bx.min_y }
            + self.nz * if self.nz >= 0.0 { bx.max_z } else { bx.min_z }
    }

    /// The plane function's minimum over the box's corners.
    #[inline]
    fn plane_min(&self, bx: Box3D) -> f64 {
        self.d
            + self.nx * if self.nx >= 0.0 { bx.min_x } else { bx.max_x }
            + self.ny * if self.ny >= 0.0 { bx.min_y } else { bx.max_y }
            + self.nz * if self.nz >= 0.0 { bx.min_z } else { bx.max_z }
    }

    /// Whether any part of `bx` is inside the half-space: the two-corner sign
    /// test.
    pub fn overlaps_box(&self, bx: Box3D) -> bool {
        self.plane_max(bx) >= 0.0
    }

    /// Whether all of `bx` is inside the half-space.
    pub fn contains_box(&self, bx: Box3D) -> bool {
        self.plane_min(bx) >= 0.0
    }
}

impl Overlaps3D for HalfSpace3D {
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

    #[test]
    fn two_corner_sign_test_matches_corner_bruteforce() {
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..2000 {
            let x: f64 = rng.random_range(-10.0..10.0);
            let y: f64 = rng.random_range(-10.0..10.0);
            let bx = Box2D::new(x, y, x + 5.0, y + 5.0);
            let (nx, ny, d) = (
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-5.0..5.0),
            );
            let hs = HalfSpace2D::new(nx, ny, d);
            let f = |x: f64, y: f64| nx * x + ny * y + d;
            let brute_max = f(bx.min_x, bx.min_y)
                .max(f(bx.max_x, bx.min_y))
                .max(f(bx.min_x, bx.max_y))
                .max(f(bx.max_x, bx.max_y));
            let brute_min = f(bx.min_x, bx.min_y)
                .min(f(bx.max_x, bx.min_y))
                .min(f(bx.min_x, bx.max_y))
                .min(f(bx.max_x, bx.max_y));
            assert_eq!(hs.overlaps_box(bx), brute_max >= 0.0);
            assert_eq!(hs.contains_box(bx), brute_min >= 0.0);
        }
    }

    #[test]
    fn search_agrees_with_a_bruteforce_filter() {
        let mut rng = StdRng::seed_from_u64(5);
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
            let hs = HalfSpace2D::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-20.0..20.0),
            );
            let want: Vec<usize> = (0..items.len())
                .filter(|&i| hs.overlaps_box(items[i]))
                .collect();
            let mut got = index.search(&hs);
            got.sort_unstable();
            assert_eq!(got, want, "half-space {hs:?}");
        }
    }

    #[test]
    fn search_3d_agrees_with_a_bruteforce_filter() {
        let mut rng = StdRng::seed_from_u64(6);
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
            let hs = HalfSpace3D::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-15.0..15.0),
            );
            let want: Vec<usize> = (0..items.len())
                .filter(|&i| hs.overlaps_box(items[i]))
                .collect();
            let mut got = index.search(&hs);
            got.sort_unstable();
            assert_eq!(got, want, "half-space {hs:?}");
        }
    }
}
