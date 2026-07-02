use geo::BoundingRect;
use geo_types::{Coord, LineString, MultiPolygon, Polygon};
use packed_spatial_index::{Box2D, Box3D, Frustum3D};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::GeoError;

/// 2D geospatial query.
///
/// # Example
///
/// ```rust
/// use packed_spatial_index_geo::{Box2D, GeoQuery2D};
/// use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};
///
/// let query = GeoQuery2D::box2d(Box2D::new(-10.0, 35.0, 20.0, 60.0));
/// assert!(matches!(query, GeoQuery2D::Box2D(_)));
/// let triangle = Polygon::new(
///     LineString::new(vec![
///         Coord { x: 0.0, y: 0.0 },
///         Coord { x: 4.0, y: 0.0 },
///         Coord { x: 0.0, y: 4.0 },
///         Coord { x: 0.0, y: 0.0 },
///     ]),
///     vec![],
/// );
/// let poly = GeoQuery2D::polygon(triangle);
/// assert!(matches!(poly, GeoQuery2D::Polygon(_)));
/// let radius = GeoQuery2D::spherical_radius(-73.9857, 40.7484, 500.0);
/// assert!(matches!(radius, GeoQuery2D::SphericalRadius { .. }));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum GeoQuery2D {
    /// Query rectangle in source XY coordinates.
    Box2D(Box2D),
    /// Arbitrary planar polygon / multipolygon query in source XY coordinates.
    ///
    /// The index narrows candidates by this geometry's bounding box; exact
    /// filtering then keeps only candidates whose geometry actually intersects
    /// the polygon, removing the bbox false-positives over holes and concavities.
    Polygon(MultiPolygon<f64>),
    /// Spherical point-radius query in longitude/latitude coordinates.
    SphericalRadius {
        /// Query longitude in degrees.
        lon: f64,
        /// Query latitude in degrees.
        lat: f64,
        /// Query radius in metres on a spherical Earth.
        radius_metres: f64,
    },
}

impl GeoQuery2D {
    /// Create a 2D box query in source XY coordinates.
    pub fn box2d(bbox: Box2D) -> Self {
        Self::Box2D(bbox)
    }

    /// Create a planar polygon query in source XY coordinates.
    pub fn polygon(polygon: Polygon<f64>) -> Self {
        Self::Polygon(MultiPolygon::new(vec![polygon]))
    }

    /// Create a planar multipolygon query in source XY coordinates.
    pub fn multi_polygon(multi_polygon: MultiPolygon<f64>) -> Self {
        Self::Polygon(multi_polygon)
    }

    /// Create a spherical point-radius query in longitude/latitude coordinates.
    pub fn spherical_radius(lon: f64, lat: f64, radius_metres: f64) -> Self {
        Self::SphericalRadius {
            lon,
            lat,
            radius_metres,
        }
    }

    /// Return 2D lon/lat candidate boxes for this query geometry.
    ///
    /// `Box2D` returns itself. `SphericalRadius` returns one or two longitude /
    /// latitude boxes, splitting at the antimeridian when needed.
    pub fn candidate_boxes_2d(&self) -> Result<Vec<Box2D>, GeoError> {
        match self {
            GeoQuery2D::Box2D(bbox) => Ok(vec![*bbox]),
            GeoQuery2D::Polygon(multi_polygon) => {
                let rect = multi_polygon
                    .bounding_rect()
                    .ok_or(GeoError::EmptyQueryPolygon)?;
                Ok(vec![Box2D::new(
                    rect.min().x,
                    rect.min().y,
                    rect.max().x,
                    rect.max().y,
                )])
            }
            GeoQuery2D::SphericalRadius {
                lon,
                lat,
                radius_metres,
            } => Ok(
                crate::geodetic::SphericalRadius::new(*lon, *lat, *radius_metres)?
                    .candidate_boxes(),
            ),
        }
    }
}

impl From<Box2D> for GeoQuery2D {
    fn from(value: Box2D) -> Self {
        Self::Box2D(value)
    }
}

impl From<Polygon<f64>> for GeoQuery2D {
    fn from(value: Polygon<f64>) -> Self {
        Self::polygon(value)
    }
}

impl From<MultiPolygon<f64>> for GeoQuery2D {
    fn from(value: MultiPolygon<f64>) -> Self {
        Self::Polygon(value)
    }
}

impl Serialize for GeoQuery2D {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            GeoQuery2D::Box2D(bbox) => {
                GeoQuery2DSerde::Box2D([bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y])
                    .serialize(serializer)
            }
            GeoQuery2D::Polygon(multi_polygon) => {
                GeoQuery2DSerde::Polygon(multi_polygon_to_rings(multi_polygon))
                    .serialize(serializer)
            }
            GeoQuery2D::SphericalRadius {
                lon,
                lat,
                radius_metres,
            } => GeoQuery2DSerde::SphericalRadius {
                lon: *lon,
                lat: *lat,
                radius_metres: *radius_metres,
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for GeoQuery2D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match GeoQuery2DSerde::deserialize(deserializer)? {
            GeoQuery2DSerde::Box2D([min_x, min_y, max_x, max_y]) => {
                GeoQuery2D::Box2D(Box2D::new(min_x, min_y, max_x, max_y))
            }
            GeoQuery2DSerde::Polygon(rings) => GeoQuery2D::Polygon(rings_to_multi_polygon(rings)),
            GeoQuery2DSerde::SphericalRadius {
                lon,
                lat,
                radius_metres,
            } => GeoQuery2D::SphericalRadius {
                lon,
                lat,
                radius_metres,
            },
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum GeoQuery2DSerde {
    Box2D([f64; 4]),
    /// Nested `[polygon][ring][x, y]` coordinate arrays; ring 0 is the exterior.
    Polygon(Vec<Vec<Vec<[f64; 2]>>>),
    SphericalRadius {
        lon: f64,
        lat: f64,
        radius_metres: f64,
    },
}

/// Flatten a [`MultiPolygon`] into serde-friendly `[polygon][ring][x, y]` arrays.
fn multi_polygon_to_rings(multi_polygon: &MultiPolygon<f64>) -> Vec<Vec<Vec<[f64; 2]>>> {
    multi_polygon
        .iter()
        .map(|polygon| {
            let mut rings = Vec::with_capacity(1 + polygon.interiors().len());
            rings.push(ring_to_coords(polygon.exterior()));
            rings.extend(polygon.interiors().iter().map(ring_to_coords));
            rings
        })
        .collect()
}

fn ring_to_coords(ring: &LineString<f64>) -> Vec<[f64; 2]> {
    ring.coords().map(|coord| [coord.x, coord.y]).collect()
}

/// Rebuild a [`MultiPolygon`] from `[polygon][ring][x, y]` arrays (ring 0 = exterior).
fn rings_to_multi_polygon(polygons: Vec<Vec<Vec<[f64; 2]>>>) -> MultiPolygon<f64> {
    MultiPolygon::new(
        polygons
            .into_iter()
            .map(|rings| {
                let mut rings = rings.into_iter().map(coords_to_ring);
                let exterior = rings.next().unwrap_or_else(|| LineString::new(Vec::new()));
                Polygon::new(exterior, rings.collect())
            })
            .collect(),
    )
}

fn coords_to_ring(coords: Vec<[f64; 2]>) -> LineString<f64> {
    LineString::new(coords.into_iter().map(|[x, y]| Coord { x, y }).collect())
}

/// 3D geospatial query.
///
/// # Example
///
/// ```rust
/// use packed_spatial_index_geo::{Box3D, GeoQuery3D};
///
/// let query = GeoQuery3D::box3d(Box3D::new(-10.0, 35.0, 0.0, 20.0, 60.0, 100.0));
/// assert!(matches!(query, GeoQuery3D::Box3D(_)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeoQuery3D {
    /// Query box in source XYZ coordinates.
    Box3D(Box3D),
    /// View-frustum candidate query in source XYZ coordinates.
    ///
    /// Coarse only: like [`Frustum3D::overlaps_box`], the search may include
    /// boxes that only partly overlap the frustum (the standard
    /// frustum-culling p-vertex test). There is no exact narrow-phase filter
    /// for frustum queries in this crate — do your own test on the returned
    /// candidates, the same pattern `packed_spatial_index`'s own raycast
    /// establishes (see its `examples/raycast_mesh.rs`).
    ///
    /// Only supported against `f64`-precision in-memory indexes
    /// (`GeoIndex3D`) and artifacts of either precision;
    /// an `f32`-precision in-memory index
    /// (`GeoIndex3DF32`) rejects this variant, since
    /// its underlying core index only implements a box-based search.
    Frustum3D(Frustum3D),
}

impl GeoQuery3D {
    /// Create a 3D box query in source XYZ coordinates.
    pub fn box3d(bbox: Box3D) -> Self {
        Self::Box3D(bbox)
    }

    /// Create a 3D view-frustum candidate query in source XYZ coordinates.
    ///
    /// # Example
    ///
    /// ```rust
    /// use packed_spatial_index_geo::{ClipSpaceZ, Frustum3D, GeoQuery3D};
    ///
    /// let identity = [
    ///     [1.0, 0.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0, 0.0],
    ///     [0.0, 0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 0.0, 1.0],
    /// ];
    /// let frustum = Frustum3D::from_view_projection(identity, ClipSpaceZ::NegOneToOne);
    /// let query = GeoQuery3D::frustum3d(frustum);
    /// assert!(matches!(query, GeoQuery3D::Frustum3D(_)));
    /// ```
    pub fn frustum3d(frustum: Frustum3D) -> Self {
        Self::Frustum3D(frustum)
    }

    /// Return a coarse 3D covering box for this query.
    ///
    /// For [`GeoQuery3D::Box3D`] this is the query box itself. For
    /// [`GeoQuery3D::Frustum3D`] this is the frustum's own bounding box
    /// ([`Frustum3D::bounding_box`]) — a covering superset, not the frustum
    /// shape itself, and `Err` when the frustum's planes are degenerate.
    ///
    /// This is a metadata/diagnostics helper, not what index search uses: a
    /// `Frustum3D` search dispatches on the query variant directly, so it
    /// keeps the frustum's own tighter overlap test instead of degrading to
    /// this box.
    pub fn candidate_box_3d(self) -> Result<Box3D, GeoError> {
        match self {
            GeoQuery3D::Box3D(bbox) => Ok(bbox),
            GeoQuery3D::Frustum3D(frustum) => frustum.bounding_box().ok_or_else(|| {
                GeoError::UnsupportedArtifact(
                    "frustum bounding box is undefined: its planes are degenerate or \
                     near-parallel"
                        .to_string(),
                )
            }),
        }
    }
}

impl From<Box3D> for GeoQuery3D {
    fn from(value: Box3D) -> Self {
        Self::Box3D(value)
    }
}

impl From<Frustum3D> for GeoQuery3D {
    fn from(value: Frustum3D) -> Self {
        Self::Frustum3D(value)
    }
}

impl Serialize for GeoQuery3D {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            GeoQuery3D::Box3D(bbox) => GeoQuery3DSerde::Box3D([
                bbox.min_x, bbox.min_y, bbox.min_z, bbox.max_x, bbox.max_y, bbox.max_z,
            ])
            .serialize(serializer),
            GeoQuery3D::Frustum3D(frustum) => {
                GeoQuery3DSerde::Frustum3D(*frustum.planes()).serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for GeoQuery3D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match GeoQuery3DSerde::deserialize(deserializer)? {
            GeoQuery3DSerde::Box3D([min_x, min_y, min_z, max_x, max_y, max_z]) => {
                GeoQuery3D::Box3D(Box3D::new(min_x, min_y, min_z, max_x, max_y, max_z))
            }
            GeoQuery3DSerde::Frustum3D(planes) => {
                GeoQuery3D::Frustum3D(Frustum3D::from_planes(planes))
            }
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum GeoQuery3DSerde {
    Box3D([f64; 6]),
    Frustum3D([[f64; 4]; 6]),
}

/// Exact source-filtering predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialPredicate {
    /// Keep features whose geometry intersects the query geometry.
    Intersects,
}

/// How exact filtering handles non-planar geography/edge metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonPlanarExactPolicy {
    /// Reject exact filtering when the selected column is not planar.
    Reject,
    /// Treat stored coordinates as planar XY for the predicate.
    TreatAsPlanar,
}
