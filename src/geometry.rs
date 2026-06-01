use std::{error::Error, fmt};

/// Index coordinates are `f64`, matching the reference default.
pub(crate) type Num = f64;

/// Error returned by [`Rect::try_new`] for inverted or unordered bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RectError {
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
}

impl fmt::Display for RectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RectError::InvalidBounds { .. } => {
                write!(
                    f,
                    "rectangle bounds must satisfy min_x <= max_x and min_y <= max_y"
                )
            }
        }
    }
}

impl Error for RectError {}

/// Axis-aligned rectangle bounds.
///
/// Bounds are inclusive: rectangles that touch at an edge or corner overlap.
/// [`Rect::new`] is a cheap constructor and does not validate or reorder
/// bounds; use [`Rect::try_new`] when accepting unchecked input.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Point, Rect, RectError};
///
/// let a = Rect::new(0.0, 0.0, 1.0, 1.0);
/// let b = Rect::try_new(1.0, 1.0, 2.0, 2.0)?;
///
/// assert!(a.overlaps(b));
/// assert!(a.contains_point(Point::new(0.5, 0.5)));
/// assert!(!a.contains(b));
/// # Ok::<(), RectError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Minimum x coordinate.
    pub min_x: f64,
    /// Minimum y coordinate.
    pub min_y: f64,
    /// Maximum x coordinate.
    pub max_x: f64,
    /// Maximum y coordinate.
    pub max_y: f64,
}

impl Rect {
    /// Create a rectangle from `[min_x, min_y, max_x, max_y]` bounds.
    ///
    /// This constructor does not validate or reorder bounds. Prefer
    /// [`Rect::try_new`] for data that may contain inverted bounds or `NaN`.
    #[inline]
    pub const fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Try to create a rectangle with validated bounds.
    ///
    /// Returns [`RectError::InvalidBounds`] when `min_x > max_x`, `min_y > max_y`,
    /// or any bound is `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Rect, RectError};
    ///
    /// let rect = Rect::try_new(0.0, 0.0, 1.0, 1.0)?;
    /// assert_eq!(rect, Rect::new(0.0, 0.0, 1.0, 1.0));
    ///
    /// assert!(matches!(
    ///     Rect::try_new(2.0, 0.0, 1.0, 1.0),
    ///     Err(RectError::InvalidBounds { .. })
    /// ));
    /// # Ok::<(), RectError>(())
    /// ```
    #[inline]
    pub const fn try_new(
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<Self, RectError> {
        if min_x <= max_x && min_y <= max_y {
            Ok(Self::new(min_x, min_y, max_x, max_y))
        } else {
            Err(RectError::InvalidBounds {
                min_x,
                min_y,
                max_x,
                max_y,
            })
        }
    }

    /// Return `true` when this rectangle overlaps `other`.
    ///
    /// Edges are inclusive: rectangles that only touch at an edge or corner
    /// are considered overlapping.
    #[inline]
    pub fn overlaps(&self, other: Rect) -> bool {
        // Branchless: compute all four comparisons and combine them with bitwise `&`
        // to remove hard-to-predict floating-point branches from the traversal loop.
        (self.min_x <= other.max_x)
            & (self.max_x >= other.min_x)
            & (self.min_y <= other.max_y)
            & (self.max_y >= other.min_y)
    }

    /// Return `true` when this rectangle fully contains `other`.
    ///
    /// Edges are inclusive.
    #[inline]
    pub fn contains(&self, other: Rect) -> bool {
        (self.min_x <= other.min_x)
            & (self.min_y <= other.min_y)
            & (self.max_x >= other.max_x)
            & (self.max_y >= other.max_y)
    }

    /// Return `true` when this rectangle contains `point`.
    ///
    /// Edges are inclusive.
    #[inline]
    pub fn contains_point(&self, point: Point) -> bool {
        (self.min_x <= point.x)
            & (self.max_x >= point.x)
            & (self.min_y <= point.y)
            & (self.max_y >= point.y)
    }

    #[inline]
    pub(crate) fn distance_squared_to(&self, point: Point) -> f64 {
        let dx = axis_distance(point.x, self.min_x, self.max_x);
        let dy = axis_distance(point.y, self.min_y, self.max_y);
        dx * dx + dy * dy
    }
}

/// 2D point used by nearest-neighbor searches.
///
/// # Example
///
/// ```
/// use packed_spatial_index::Point;
///
/// let point = Point::new(10.0, 20.0);
/// assert_eq!(point.x, 10.0);
/// assert_eq!(point.y, 20.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point {
    /// Create a point from `x, y`.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
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
