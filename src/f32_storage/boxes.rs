use crate::geometry::{Box2D, Box3D};
#[cfg(feature = "simd")]
use crate::persistence::read_f32_le_unchecked;

/// Round `x` down to the nearest `f32` that is `<= x`.
#[inline]
pub(crate) fn round_down(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) > x { r.next_down() } else { r }
}

/// Round `x` up to the nearest `f32` that is `>= x`.
#[inline]
pub(crate) fn round_up(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) < x { r.next_up() } else { r }
}

/// 2D box stored as four `f32` (`min_x, min_y, max_x, max_y`).
#[derive(Clone, Copy)]
pub(crate) struct Box2DF32 {
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) max_x: f32,
    pub(crate) max_y: f32,
}

impl Box2DF32 {
    /// Superset of `b` with bounds rounded outward (min down, max up).
    #[inline]
    pub(crate) fn from_box2d_outward(b: Box2D) -> Self {
        Self {
            min_x: round_down(b.min_x),
            min_y: round_down(b.min_y),
            max_x: round_up(b.max_x),
            max_y: round_up(b.max_y),
        }
    }

    /// `b` rounded inward (min up, max down) onto the f32 grid.
    #[inline]
    pub(crate) fn from_box2d_inward(b: Box2D) -> Self {
        Self {
            min_x: round_up(b.min_x),
            min_y: round_up(b.min_y),
            max_x: round_down(b.max_x),
            max_y: round_down(b.max_y),
        }
    }

    /// Read a 2D f32 box from SoA columns.
    #[inline]
    pub(crate) fn from_soa(
        min_xs: &[f32],
        min_ys: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        pos: usize,
    ) -> Self {
        Self {
            min_x: min_xs[pos],
            min_y: min_ys[pos],
            max_x: max_xs[pos],
            max_y: max_ys[pos],
        }
    }

    /// Read a 2D f32 box record from TREE bytes.
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn read_tree(entries: &[u8], pos: usize) -> Self {
        let off = pos * 16;
        Self {
            min_x: read_f32_le_unchecked(entries, off),
            min_y: read_f32_le_unchecked(entries, off + 4),
            max_x: read_f32_le_unchecked(entries, off + 8),
            max_y: read_f32_le_unchecked(entries, off + 12),
        }
    }

    /// Widen losslessly to an f64 box.
    #[inline]
    pub(crate) fn widen(self) -> Box2D {
        Box2D::new(
            self.min_x as f64,
            self.min_y as f64,
            self.max_x as f64,
            self.max_y as f64,
        )
    }

    #[inline]
    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    #[inline]
    pub(crate) fn definitely_overlaps_exact(self, query: Box2D) -> bool {
        (self.min_x.next_up() as f64 <= query.max_x)
            && (self.max_x.next_down() as f64 >= query.min_x)
            && (self.min_y.next_up() as f64 <= query.max_y)
            && (self.max_y.next_down() as f64 >= query.min_y)
    }

    #[inline]
    pub(crate) fn overlaps_exact_or_refined(
        self,
        query: Box2D,
        exact_box: impl FnOnce() -> Box2D,
    ) -> bool {
        self.definitely_overlaps_exact(query) || exact_box().overlaps(query)
    }

    /// True when `self` fully contains `other` (both already rounded).
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.min_x <= other.min_x
            && other.max_x <= self.max_x
            && self.min_y <= other.min_y
            && other.max_y <= self.max_y
    }
}

/// 3D box stored as six `f32`.
#[derive(Clone, Copy)]
pub(crate) struct Box3DF32 {
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) min_z: f32,
    pub(crate) max_x: f32,
    pub(crate) max_y: f32,
    pub(crate) max_z: f32,
}

impl Box3DF32 {
    /// Superset of `b` with bounds rounded outward (min down, max up).
    #[inline]
    pub(crate) fn from_box3d_outward(b: Box3D) -> Self {
        Self {
            min_x: round_down(b.min_x),
            min_y: round_down(b.min_y),
            min_z: round_down(b.min_z),
            max_x: round_up(b.max_x),
            max_y: round_up(b.max_y),
            max_z: round_up(b.max_z),
        }
    }

    /// `b` rounded inward (min up, max down) onto the f32 grid.
    #[inline]
    pub(crate) fn from_box3d_inward(b: Box3D) -> Self {
        Self {
            min_x: round_up(b.min_x),
            min_y: round_up(b.min_y),
            min_z: round_up(b.min_z),
            max_x: round_down(b.max_x),
            max_y: round_down(b.max_y),
            max_z: round_down(b.max_z),
        }
    }

    /// Read a 3D f32 box from SoA columns.
    #[inline]
    pub(crate) fn from_soa(
        min_xs: &[f32],
        min_ys: &[f32],
        min_zs: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        max_zs: &[f32],
        pos: usize,
    ) -> Self {
        Self {
            min_x: min_xs[pos],
            min_y: min_ys[pos],
            min_z: min_zs[pos],
            max_x: max_xs[pos],
            max_y: max_ys[pos],
            max_z: max_zs[pos],
        }
    }

    /// Read a 3D f32 box record from TREE bytes.
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn read_tree(entries: &[u8], pos: usize) -> Self {
        let off = pos * 24;
        Self {
            min_x: read_f32_le_unchecked(entries, off),
            min_y: read_f32_le_unchecked(entries, off + 4),
            min_z: read_f32_le_unchecked(entries, off + 8),
            max_x: read_f32_le_unchecked(entries, off + 12),
            max_y: read_f32_le_unchecked(entries, off + 16),
            max_z: read_f32_le_unchecked(entries, off + 20),
        }
    }

    /// Widen losslessly to an f64 box.
    #[inline]
    pub(crate) fn widen(self) -> Box3D {
        Box3D::new(
            self.min_x as f64,
            self.min_y as f64,
            self.min_z as f64,
            self.max_x as f64,
            self.max_y as f64,
            self.max_z as f64,
        )
    }

    #[inline]
    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }

    #[inline]
    pub(crate) fn definitely_overlaps_exact(self, query: Box3D) -> bool {
        (self.min_x.next_up() as f64 <= query.max_x)
            && (self.max_x.next_down() as f64 >= query.min_x)
            && (self.min_y.next_up() as f64 <= query.max_y)
            && (self.max_y.next_down() as f64 >= query.min_y)
            && (self.min_z.next_up() as f64 <= query.max_z)
            && (self.max_z.next_down() as f64 >= query.min_z)
    }

    #[inline]
    pub(crate) fn overlaps_exact_or_refined(
        self,
        query: Box3D,
        exact_box: impl FnOnce() -> Box3D,
    ) -> bool {
        self.definitely_overlaps_exact(query) || exact_box().overlaps(query)
    }

    /// True when `self` fully contains `other` (both already rounded).
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.min_x <= other.min_x
            && other.max_x <= self.max_x
            && self.min_y <= other.min_y
            && other.max_y <= self.max_y
            && self.min_z <= other.min_z
            && other.max_z <= self.max_z
    }
}
