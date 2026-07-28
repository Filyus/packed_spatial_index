use crate::{Box2D, GeoError};
use serde::{Deserialize, Serialize};

/// Handling for null or empty geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullPolicy {
    /// Return an error.
    Error,
    /// Skip the geometry and preserve source row numbers in `FeatureRef`.
    Skip,
}

/// How to handle envelopes crossing the antimeridian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AntimeridianPolicy {
    /// Return an error for antimeridian-crossing envelopes.
    Reject,
    /// Split the feature into two index entries.
    Split,
    /// Expand the longitude interval to the whole world.
    ExpandToWorld,
}

/// Envelope interpretation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvelopePolicy {
    /// Treat coordinates as ordinary planar axes.
    Planar,
    /// Treat x as longitude and apply an antimeridian policy.
    Geographic {
        /// Antimeridian handling.
        antimeridian: AntimeridianPolicy,
    },
}

pub(crate) const EARTH_RADIUS_METRES: f64 = 6_371_008.8;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SphericalRadius {
    pub lon: f64,
    pub lat: f64,
    pub radius_metres: f64,
}

impl SphericalRadius {
    pub(crate) fn new(lon: f64, lat: f64, radius_metres: f64) -> Result<Self, GeoError> {
        if !lon.is_finite() || !(-180.0..=180.0).contains(&lon) {
            return Err(GeoError::InvalidSphericalQuery(
                "longitude must be finite and in [-180, 180]".to_string(),
            ));
        }
        if !lat.is_finite() || !(-90.0..=90.0).contains(&lat) {
            return Err(GeoError::InvalidSphericalQuery(
                "latitude must be finite and in [-90, 90]".to_string(),
            ));
        }
        if !radius_metres.is_finite() || radius_metres < 0.0 {
            return Err(GeoError::InvalidSphericalQuery(
                "radius must be finite and non-negative".to_string(),
            ));
        }
        Ok(Self {
            lon,
            lat,
            radius_metres,
        })
    }

    pub(crate) fn candidate_boxes(self) -> Vec<Box2D> {
        let angular = self.radius_metres / EARTH_RADIUS_METRES;
        if angular >= std::f64::consts::PI {
            return vec![world_box()];
        }

        let lat = self.lat.to_radians();
        let min_lat = lat - angular;
        let max_lat = lat + angular;
        let min_lat_deg = min_lat.max(-std::f64::consts::FRAC_PI_2).to_degrees();
        let max_lat_deg = max_lat.min(std::f64::consts::FRAC_PI_2).to_degrees();
        if min_lat <= -std::f64::consts::FRAC_PI_2 || max_lat >= std::f64::consts::FRAC_PI_2 {
            return vec![Box2D::new(-180.0, min_lat_deg, 180.0, max_lat_deg)];
        }

        let arg = (angular.sin() / lat.cos()).clamp(-1.0, 1.0);
        let delta_lon = arg.asin().to_degrees();
        let west = normalize_lon(self.lon - delta_lon);
        let east = normalize_lon(self.lon + delta_lon);
        if west <= east {
            vec![Box2D::new(west, min_lat_deg, east, max_lat_deg)]
        } else {
            vec![
                Box2D::new(west, min_lat_deg, 180.0, max_lat_deg),
                Box2D::new(-180.0, min_lat_deg, east, max_lat_deg),
            ]
        }
    }

    pub(crate) fn contains_point(self, lon: f64, lat: f64) -> bool {
        if !lon.is_finite() || !lat.is_finite() {
            return false;
        }
        // Longitude is periodic, so one outside the principal range names a
        // real place — a dataset stored in [0, 360) is the usual reason. Wrap
        // it rather than dropping the point, matching what `candidate_boxes`
        // already does to the query. Latitude has no such reading: beyond the
        // poles is nowhere.
        let lon = normalize_lon(lon);
        if !(-90.0..=90.0).contains(&lat) {
            return false;
        }
        haversine_metres(self.lon, self.lat, lon, lat) <= self.radius_metres
    }
}

fn world_box() -> Box2D {
    Box2D::new(-180.0, -90.0, 180.0, 90.0)
}

/// Wrap a longitude into `[-180, 180]`.
///
/// Values already in range are returned untouched, which keeps `180.0` as
/// `180.0` rather than folding it onto `-180.0`. Everything else is wrapped
/// arithmetically: stepping by 360 in a loop is fine for a query longitude
/// plus a bounded delta, but source coordinates are arbitrary finite floats
/// and a value like `1e30` would take on the order of `1e27` steps.
fn normalize_lon(lon: f64) -> f64 {
    if (-180.0..=180.0).contains(&lon) {
        return lon;
    }
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

fn haversine_metres(a_lon: f64, a_lat: f64, b_lon: f64, b_lat: f64) -> f64 {
    let lat1 = a_lat.to_radians();
    let lat2 = b_lat.to_radians();
    let dlat = (b_lat - a_lat).to_radians();
    let dlon = (b_lon - a_lon).to_radians();
    let inner = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    EARTH_RADIUS_METRES * 2.0 * inner.sqrt().min(1.0).asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_point_accepts_longitudes_outside_the_principal_range() {
        let query = SphericalRadius::new(-170.0, 0.0, 200_000.0).unwrap();
        assert!(query.contains_point(-169.0, 0.0));
        // The same meridian written as degrees east of Greenwich, which is how
        // a dataset stored in [0, 360) expresses it.
        assert!(query.contains_point(191.0, 0.0));
        assert!(!query.contains_point(0.0, 0.0));
    }

    #[test]
    fn contains_point_still_rejects_impossible_latitudes() {
        let query = SphericalRadius::new(0.0, 0.0, 200_000.0).unwrap();
        assert!(!query.contains_point(0.0, 91.0));
        assert!(!query.contains_point(f64::NAN, 0.0));
        assert!(!query.contains_point(0.0, f64::INFINITY));
    }

    #[test]
    fn normalize_lon_terminates_for_far_out_of_range_values() {
        // Reaching the principal range by repeated addition would take on the
        // order of 1e27 steps; completing this test at all is the assertion.
        assert!(normalize_lon(1.0e30).is_finite());
        assert!(normalize_lon(-1.0e30).is_finite());
    }

    #[test]
    fn normalize_lon_leaves_the_principal_range_alone() {
        for lon in [-180.0, -0.5, 0.0, 179.999, 180.0] {
            assert_eq!(normalize_lon(lon), lon);
        }
        assert_eq!(normalize_lon(181.0), -179.0);
        assert_eq!(normalize_lon(-181.0), 179.0);
        // Both ends name the antimeridian; an out-of-range value lands on the
        // negative one, an in-range one is left as written.
        assert_eq!(normalize_lon(540.0), -180.0);
        // The reachable inputs from `candidate_boxes` are a query longitude
        // plus a bounded delta, so the wrap must agree with stepping by 360
        // across that whole span.
        for lon in [-360.0, -270.0, 0.0, 270.0, 360.0] {
            assert_eq!(normalize_lon(lon), step_by_360(lon));
        }
    }

    fn step_by_360(mut lon: f64) -> f64 {
        while lon < -180.0 {
            lon += 360.0;
        }
        while lon > 180.0 {
            lon -= 360.0;
        }
        lon
    }
}
