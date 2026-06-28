use crate::{Box2D, GeoError};

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
        if !(-180.0..=180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
            return false;
        }
        haversine_metres(self.lon, self.lat, lon, lat) <= self.radius_metres
    }
}

fn world_box() -> Box2D {
    Box2D::new(-180.0, -90.0, 180.0, 90.0)
}

fn normalize_lon(mut lon: f64) -> f64 {
    while lon < -180.0 {
        lon += 360.0;
    }
    while lon > 180.0 {
        lon -= 360.0;
    }
    lon
}

fn haversine_metres(a_lon: f64, a_lat: f64, b_lon: f64, b_lat: f64) -> f64 {
    let lat1 = a_lat.to_radians();
    let lat2 = b_lat.to_radians();
    let dlat = (b_lat - a_lat).to_radians();
    let dlon = (b_lon - a_lon).to_radians();
    let inner = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    EARTH_RADIUS_METRES * 2.0 * inner.sqrt().min(1.0).asin()
}
