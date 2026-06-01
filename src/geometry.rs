/// Index coordinates are `f64`, matching the reference default.
pub(crate) type Num = f64;

/// Axis-aligned rectangle bounds.
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
    #[inline]
    pub const fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
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
