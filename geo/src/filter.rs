use std::io::Cursor;

use geo::Intersects;
use geozero::wkb::{FromWkb, WkbDialect};
use packed_spatial_index::Box2D;
use serde::{Deserialize, Serialize};

use crate::geodetic::SphericalRadius;
use crate::{
    EdgeAlgorithm, EdgeModel, FeatureRef, GeoError, GeoQuery2D, GeometryEncoding, GeometrySelector,
    NonPlanarExactPolicy, SpatialPredicate, wkb,
};

fn reject_non_planar_exact(
    encoding: &GeometryEncoding,
    edges: EdgeModel,
    column: &str,
    policy: NonPlanarExactPolicy,
) -> Result<(), GeoError> {
    if matches!(policy, NonPlanarExactPolicy::TreatAsPlanar) {
        return Ok(());
    }
    if matches!(encoding, GeometryEncoding::ParquetGeography { .. })
        || !matches!(edges, EdgeModel::Planar)
    {
        return Err(GeoError::NonPlanarExactPredicate {
            column: column.to_string(),
            edges,
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) enum PreparedFilterQuery {
    Box2D(Box2D),
    Polygon(geo_types::MultiPolygon<f64>),
    SphericalRadius(SphericalRadius),
}

/// Validate a query against the column's edge/encoding model and lower it to a
/// `PreparedFilterQuery`. Takes the column facts directly so both the source
/// dataset (from `ColumnState`) and the artifact path (from the `geoM` manifest)
/// can share it.
pub(crate) fn prepare_filter_query(
    encoding: &GeometryEncoding,
    edges: EdgeModel,
    column: &str,
    query: GeoQuery2D,
    non_planar: NonPlanarExactPolicy,
) -> Result<PreparedFilterQuery, GeoError> {
    match query {
        GeoQuery2D::Box2D(bbox) => {
            reject_non_planar_exact(encoding, edges, column, non_planar)?;
            Ok(PreparedFilterQuery::Box2D(bbox))
        }
        GeoQuery2D::Polygon(multi_polygon) => {
            reject_non_planar_exact(encoding, edges, column, non_planar)?;
            Ok(PreparedFilterQuery::Polygon(multi_polygon))
        }
        GeoQuery2D::SphericalRadius {
            lon,
            lat,
            radius_metres,
        } => {
            let compatible_native = !matches!(encoding, GeometryEncoding::ParquetGeography { .. })
                || matches!(
                    encoding,
                    GeometryEncoding::ParquetGeography {
                        algorithm: EdgeAlgorithm::Spherical
                    }
                );
            if !matches!(edges, EdgeModel::Spherical) || !compatible_native {
                return Err(GeoError::NonSphericalExactPredicate {
                    column: column.to_string(),
                    edges,
                });
            }
            Ok(PreparedFilterQuery::SphericalRadius(SphericalRadius::new(
                lon,
                lat,
                radius_metres,
            )?))
        }
    }
}

pub(crate) fn decode_geo_geometry(
    bytes: &[u8],
) -> Result<Option<geo_types::Geometry<f64>>, GeoError> {
    let mut cursor = Cursor::new(bytes);
    match geo_types::Geometry::<f64>::from_wkb(&mut cursor, WkbDialect::Wkb) {
        Ok(geometry) => Ok(Some(geometry)),
        Err(err) => {
            if wkb::is_empty_point_wkb(bytes) {
                return Ok(None);
            }
            let msg = err.to_string();
            Err(GeoError::Wkb(msg))
        }
    }
}

pub(crate) fn exact_predicate_matches(
    geometry: &geo_types::Geometry<f64>,
    query: &PreparedFilterQuery,
    predicate: SpatialPredicate,
) -> Result<bool, GeoError> {
    match (query, predicate) {
        (PreparedFilterQuery::Box2D(bbox), SpatialPredicate::Intersects) => {
            let rect = geo_types::Rect::new(
                geo_types::Coord {
                    x: bbox.min_x,
                    y: bbox.min_y,
                },
                geo_types::Coord {
                    x: bbox.max_x,
                    y: bbox.max_y,
                },
            );
            Ok(geometry.intersects(&rect))
        }
        (PreparedFilterQuery::Polygon(multi_polygon), SpatialPredicate::Intersects) => {
            Ok(geometry.intersects(multi_polygon))
        }
        (PreparedFilterQuery::SphericalRadius(query), SpatialPredicate::Intersects) => {
            spherical_radius_matches(geometry, *query)
        }
    }
}

fn spherical_radius_matches(
    geometry: &geo_types::Geometry<f64>,
    query: SphericalRadius,
) -> Result<bool, GeoError> {
    match geometry {
        geo_types::Geometry::Point(point) => Ok(query.contains_point(point.x(), point.y())),
        geo_types::Geometry::MultiPoint(points) => Ok(points
            .iter()
            .any(|point| query.contains_point(point.x(), point.y()))),
        geo_types::Geometry::Line(_) => {
            Err(GeoError::UnsupportedGeodeticGeometry("Line".to_string()))
        }
        geo_types::Geometry::LineString(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "LineString".to_string(),
        )),
        geo_types::Geometry::Polygon(_) => {
            Err(GeoError::UnsupportedGeodeticGeometry("Polygon".to_string()))
        }
        geo_types::Geometry::MultiLineString(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "MultiLineString".to_string(),
        )),
        geo_types::Geometry::MultiPolygon(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "MultiPolygon".to_string(),
        )),
        geo_types::Geometry::GeometryCollection(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "GeometryCollection".to_string(),
        )),
        geo_types::Geometry::Rect(_) => {
            Err(GeoError::UnsupportedGeodeticGeometry("Rect".to_string()))
        }
        geo_types::Geometry::Triangle(_) => Err(GeoError::UnsupportedGeodeticGeometry(
            "Triangle".to_string(),
        )),
    }
}

/// Request for `GeoDataset::filter_features` (`parquet` feature).
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{
///     open_geoparquet, Box2D, FeatureFilterRequest, FeatureRef,
/// };
///
/// let mut source = open_geoparquet(File::open("cities.parquet")?)?;
/// let exact = source.filter_features(FeatureFilterRequest::intersects(
///     vec![FeatureRef::row_number(42)],
///     Box2D::new(-10.0, 35.0, 20.0, 60.0),
/// ))?;
/// println!("{} exact hits", exact.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureFilterRequest {
    /// Candidate feature refs to filter against source geometry.
    pub features: Vec<FeatureRef>,
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Query geometry.
    pub query: GeoQuery2D,
    /// Predicate to evaluate.
    pub predicate: SpatialPredicate,
    /// Non-planar edge handling.
    pub non_planar: NonPlanarExactPolicy,
    /// Optional source fingerprint expected by the caller or artifact manifest.
    pub expected_source_fingerprint: Option<String>,
}

impl FeatureFilterRequest {
    /// Create an `intersects` request from candidate feature refs and a 2D query.
    pub fn intersects<Q: Into<GeoQuery2D>>(features: Vec<FeatureRef>, query: Q) -> Self {
        Self {
            features,
            selector: GeometrySelector::Default,
            query: query.into(),
            predicate: SpatialPredicate::Intersects,
            non_planar: NonPlanarExactPolicy::Reject,
            expected_source_fingerprint: None,
        }
    }

    /// Create an `intersects` request from artifact hits and a 2D query.
    pub fn intersects_from_hits<Q: Into<GeoQuery2D>>(hits: Vec<crate::GeoHit>, query: Q) -> Self {
        Self::intersects(hits.into_iter().map(|hit| hit.feature).collect(), query)
    }
}
