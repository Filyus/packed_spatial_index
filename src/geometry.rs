use std::{error::Error, fmt};

/// Spatial coordinates are `f64`, matching the reference default.
pub(crate) type Num = f64;

/// Error returned by [`Box2D::try_new`] and [`Box3D::try_new`] for invalid coordinate bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundsError {
    /// Bounds do not satisfy `min_x <= max_x` and `min_y <= max_y`.
    ///
    /// This also covers `NaN`, because `NaN` is unordered and fails those
    /// comparisons.
    InvalidBounds {
        /// Minimum x coordinate.
        min_x: f64,
        /// Minimum y coordinate.
        min_y: f64,
        /// Maximum x coordinate.
        max_x: f64,
        /// Maximum y coordinate.
        max_y: f64,
    },
    /// 3D bounds do not satisfy `min <= max` on every axis.
    ///
    /// This also covers `NaN`, because `NaN` is unordered and fails those
    /// comparisons.
    InvalidBounds3D {
        /// Minimum x coordinate.
        min_x: f64,
        /// Minimum y coordinate.
        min_y: f64,
        /// Minimum z coordinate.
        min_z: f64,
        /// Maximum x coordinate.
        max_x: f64,
        /// Maximum y coordinate.
        max_y: f64,
        /// Maximum z coordinate.
        max_z: f64,
    },
}

impl fmt::Display for BoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundsError::InvalidBounds { .. } => {
                write!(f, "bounds must satisfy min_x <= max_x and min_y <= max_y")
            }
            BoundsError::InvalidBounds3D { .. } => write!(
                f,
                "bounds must satisfy min_x <= max_x, min_y <= max_y, and min_z <= max_z"
            ),
        }
    }
}

impl Error for BoundsError {}

/// Axis-aligned 2D box stored as `(min_x, min_y, max_x, max_y)`.
///
/// Boxes are inclusive: boxes that touch at an edge or corner overlap.
/// [`Box2D::new`] is a cheap constructor and does not validate or reorder
/// coordinate bounds; use [`Box2D::try_new`] when accepting unchecked input.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Point2D, Box2D, BoundsError};
///
/// let a = Box2D::new(0.0, 0.0, 1.0, 1.0);
/// let b = Box2D::try_new(1.0, 1.0, 2.0, 2.0)?;
///
/// assert!(a.overlaps(b));
/// assert!(a.contains_point(Point2D::new(0.5, 0.5)));
/// assert!(!a.contains(b));
/// # Ok::<(), BoundsError>(())
/// ```
// `repr(C)` guarantees the field layout is exactly `[min_x, min_y, max_x, max_y]` as
// four contiguous, unpadded `f64`. This matches the on-disk box record, so on
// little-endian targets the box array can be serialized with a single bulk memcpy.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box2D {
    /// Minimum x coordinate.
    pub min_x: f64,
    /// Minimum y coordinate.
    pub min_y: f64,
    /// Maximum x coordinate.
    pub max_x: f64,
    /// Maximum y coordinate.
    pub max_y: f64,
}

impl Box2D {
    /// Create a box from `[min_x, min_y, max_x, max_y]`.
    ///
    /// This constructor does not validate or reorder coordinates. Prefer
    /// [`Box2D::try_new`] for data that may contain inverted coordinate bounds or `NaN`.
    #[inline]
    pub const fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Create a zero-size box at `point`.
    ///
    /// This is useful for point containment queries:
    /// `index.search(Box2D::from_point(point))` returns boxes that contain the
    /// point, because box overlap is inclusive.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Point2D};
    ///
    /// let point = Point2D::new(2.0, 3.0);
    /// assert_eq!(Box2D::from_point(point), Box2D::new(2.0, 3.0, 2.0, 3.0));
    /// ```
    #[inline]
    pub const fn from_point(point: Point2D) -> Self {
        Self::new(point.x, point.y, point.x, point.y)
    }

    /// Try to create a validated box.
    ///
    /// Returns [`BoundsError::InvalidBounds`] when `min_x > max_x`, `min_y > max_y`,
    /// or any bound is `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, BoundsError};
    ///
    /// let box2d = Box2D::try_new(0.0, 0.0, 1.0, 1.0)?;
    /// assert_eq!(box2d, Box2D::new(0.0, 0.0, 1.0, 1.0));
    ///
    /// assert!(matches!(
    ///     Box2D::try_new(2.0, 0.0, 1.0, 1.0),
    ///     Err(BoundsError::InvalidBounds { .. })
    /// ));
    /// # Ok::<(), BoundsError>(())
    /// ```
    #[inline]
    pub const fn try_new(
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<Self, BoundsError> {
        if min_x <= max_x && min_y <= max_y {
            Ok(Self::new(min_x, min_y, max_x, max_y))
        } else {
            Err(BoundsError::InvalidBounds {
                min_x,
                min_y,
                max_x,
                max_y,
            })
        }
    }

    /// Return `true` when this box overlaps `other`.
    ///
    /// Edges are inclusive: boxes that only touch at an edge or corner
    /// are considered overlapping.
    #[inline]
    pub fn overlaps(&self, other: Box2D) -> bool {
        // Branchless: compute all four comparisons and combine them with bitwise `&`
        // to remove hard-to-predict floating-point branches from the traversal loop.
        (self.min_x <= other.max_x)
            & (self.max_x >= other.min_x)
            & (self.min_y <= other.max_y)
            & (self.max_y >= other.min_y)
    }

    /// Return `true` when this box fully contains `other`.
    ///
    /// Edges are inclusive.
    #[inline]
    pub fn contains(&self, other: Box2D) -> bool {
        (self.min_x <= other.min_x)
            & (self.min_y <= other.min_y)
            & (self.max_x >= other.max_x)
            & (self.max_y >= other.max_y)
    }

    /// Return `true` when this box contains `point`.
    ///
    /// Edges are inclusive.
    #[inline]
    pub fn contains_point(&self, point: Point2D) -> bool {
        (self.min_x <= point.x)
            & (self.max_x >= point.x)
            & (self.min_y <= point.y)
            & (self.max_y >= point.y)
    }

    #[inline]
    pub(crate) fn distance_squared_to(&self, point: Point2D) -> f64 {
        let dx = axis_distance(point.x, self.min_x, self.max_x);
        let dy = axis_distance(point.y, self.min_y, self.max_y);
        dx * dx + dy * dy
    }
}

/// Axis-aligned 3D box stored as `(min_x, min_y, min_z, max_x, max_y, max_z)`.
///
/// Boxes are inclusive: boxes that touch at a face, edge, or corner overlap.
/// [`Box3D::new`] is a cheap constructor and does not validate or reorder
/// coordinate bounds; use [`Box3D::try_new`] when accepting unchecked input.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, Point3D, BoundsError};
///
/// let a = Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
/// let b = Box3D::try_new(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)?;
///
/// assert!(a.overlaps(b));
/// assert!(a.contains_point(Point3D::new(0.5, 0.5, 0.5)));
/// assert!(!a.contains(b));
/// # Ok::<(), BoundsError>(())
/// ```
// `repr(C)` guarantees `[min_x, min_y, min_z, max_x, max_y, max_z]` as six contiguous,
// unpadded `f64`, matching the on-disk box record for single-memcpy serialization.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box3D {
    /// Minimum x coordinate.
    pub min_x: f64,
    /// Minimum y coordinate.
    pub min_y: f64,
    /// Minimum z coordinate.
    pub min_z: f64,
    /// Maximum x coordinate.
    pub max_x: f64,
    /// Maximum y coordinate.
    pub max_y: f64,
    /// Maximum z coordinate.
    pub max_z: f64,
}

impl Box3D {
    /// Create a box from `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    ///
    /// This constructor does not validate or reorder coordinates. Prefer
    /// [`Box3D::try_new`] for data that may contain inverted coordinate bounds or `NaN`.
    #[inline]
    pub const fn new(
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Self {
        Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    /// Create a zero-size box at `point`.
    ///
    /// This is useful for point containment queries:
    /// `index.search(Box3D::from_point(point))` returns boxes that contain the
    /// point, because box overlap is inclusive.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box3D, Point3D};
    ///
    /// let point = Point3D::new(2.0, 3.0, 4.0);
    /// assert_eq!(
    ///     Box3D::from_point(point),
    ///     Box3D::new(2.0, 3.0, 4.0, 2.0, 3.0, 4.0)
    /// );
    /// ```
    #[inline]
    pub const fn from_point(point: Point3D) -> Self {
        Self::new(point.x, point.y, point.z, point.x, point.y, point.z)
    }

    /// Try to create a validated 3D box.
    ///
    /// Returns [`BoundsError::InvalidBounds3D`] when any axis is inverted or
    /// any bound is `NaN`.
    #[inline]
    pub const fn try_new(
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Result<Self, BoundsError> {
        if min_x <= max_x && min_y <= max_y && min_z <= max_z {
            Ok(Self::new(min_x, min_y, min_z, max_x, max_y, max_z))
        } else {
            Err(BoundsError::InvalidBounds3D {
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
            })
        }
    }

    /// Return `true` when this box overlaps `other`.
    ///
    /// Edges are inclusive: boxes that only touch at a face, edge, or corner
    /// are considered overlapping.
    #[inline]
    pub fn overlaps(&self, other: Box3D) -> bool {
        (self.min_x <= other.max_x)
            & (self.max_x >= other.min_x)
            & (self.min_y <= other.max_y)
            & (self.max_y >= other.min_y)
            & (self.min_z <= other.max_z)
            & (self.max_z >= other.min_z)
    }

    /// Return `true` when this box fully contains `other`.
    ///
    /// Edges are inclusive.
    #[inline]
    pub fn contains(&self, other: Box3D) -> bool {
        (self.min_x <= other.min_x)
            & (self.min_y <= other.min_y)
            & (self.min_z <= other.min_z)
            & (self.max_x >= other.max_x)
            & (self.max_y >= other.max_y)
            & (self.max_z >= other.max_z)
    }

    /// Return `true` when this box contains `point`.
    ///
    /// Edges are inclusive.
    #[inline]
    pub fn contains_point(&self, point: Point3D) -> bool {
        (self.min_x <= point.x)
            & (self.max_x >= point.x)
            & (self.min_y <= point.y)
            & (self.max_y >= point.y)
            & (self.min_z <= point.z)
            & (self.max_z >= point.z)
    }

    #[inline]
    pub(crate) fn distance_squared_to(&self, point: Point3D) -> f64 {
        let dx = axis_distance(point.x, self.min_x, self.max_x);
        let dy = axis_distance(point.y, self.min_y, self.max_y);
        let dz = axis_distance(point.z, self.min_z, self.max_z);
        dx * dx + dy * dy + dz * dz
    }
}

/// 2D point used by nearest-neighbor searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::Point2D;
///
/// let point = Point2D::new(10.0, 20.0);
/// assert_eq!(point.x, 10.0);
/// assert_eq!(point.y, 20.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point2D {
    /// Create a point from `x, y`.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// 3D point used by nearest-neighbor searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::Point3D;
///
/// let point = Point3D::new(10.0, 20.0, 30.0);
/// assert_eq!(point.z, 30.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

impl Point3D {
    /// Create a point from `x, y, z`.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

#[inline(always)]
pub(crate) const fn empty_box2d() -> Box2D {
    Box2D::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    )
}

#[inline(always)]
pub(crate) fn extend_box2d(bounds: &mut Box2D, other: Box2D) {
    bounds.min_x = bounds.min_x.min(other.min_x);
    bounds.min_y = bounds.min_y.min(other.min_y);
    bounds.max_x = bounds.max_x.max(other.max_x);
    bounds.max_y = bounds.max_y.max(other.max_y);
}

#[inline(always)]
pub(crate) const fn empty_box3d() -> Box3D {
    Box3D::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    )
}

#[inline(always)]
pub(crate) fn extend_box3d(bounds: &mut Box3D, other: Box3D) {
    bounds.min_x = bounds.min_x.min(other.min_x);
    bounds.min_y = bounds.min_y.min(other.min_y);
    bounds.min_z = bounds.min_z.min(other.min_z);
    bounds.max_x = bounds.max_x.max(other.max_x);
    bounds.max_y = bounds.max_y.max(other.max_y);
    bounds.max_z = bounds.max_z.max(other.max_z);
}

#[inline]
fn axis_distance(point: f64, min: f64, max: f64) -> f64 {
    if point < min {
        min - point
    } else if point > max {
        point - max
    } else {
        0.0
    }
}
