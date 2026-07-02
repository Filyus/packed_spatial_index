use geozero::GeomProcessor;
use geozero::geojson::GeoJsonString;
use geozero::wkb::{FromWkb, WkbDialect};

use crate::{CoordinateDims, GeoError, GeometryKind};

#[derive(Debug, Clone)]
pub(crate) struct Coord {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
    pub m: Option<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct GeometryBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub dims: CoordinateDims,
    pub any: bool,
    pub lon_values: Vec<f64>,
    pub from_covering: bool,
}

impl GeometryBounds {
    pub(crate) fn new(_collect_lons: bool) -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
            dims: CoordinateDims::Unknown,
            any: false,
            lon_values: Vec::new(),
            from_covering: false,
        }
    }

    pub(crate) fn add_coord(&mut self, coord: &Coord, collect_lons: bool) {
        self.min[0] = self.min[0].min(coord.x);
        self.min[1] = self.min[1].min(coord.y);
        self.max[0] = self.max[0].max(coord.x);
        self.max[1] = self.max[1].max(coord.y);
        if let Some(z) = coord.z {
            self.min[2] = self.min[2].min(z);
            self.max[2] = self.max[2].max(z);
        }
        let coord_dims = match (coord.z.is_some(), coord.m.is_some()) {
            (true, true) => CoordinateDims::Xyzm,
            (true, false) => CoordinateDims::Xyz,
            (false, true) => CoordinateDims::Xym,
            (false, false) => CoordinateDims::Xy,
        };
        self.dims = self.dims.merge(coord_dims);
        self.any = true;
        if collect_lons {
            self.lon_values.push(coord.x);
        }
    }

    pub(crate) fn as_3d(&self) -> [f64; 6] {
        [
            self.min[0],
            self.min[1],
            if self.min[2].is_finite() {
                self.min[2]
            } else {
                0.0
            },
            self.max[0],
            self.max[1],
            if self.max[2].is_finite() {
                self.max[2]
            } else {
                0.0
            },
        ]
    }
}

struct BoundsProcessor {
    bounds: GeometryBounds,
    collect_lons: bool,
    non_finite: bool,
}

impl BoundsProcessor {
    fn new(collect_lons: bool) -> Self {
        Self {
            bounds: GeometryBounds::new(collect_lons),
            collect_lons,
            non_finite: false,
        }
    }

    fn add_coord(&mut self, coord: Coord) {
        if !coord.x.is_finite()
            || !coord.y.is_finite()
            || coord.z.is_some_and(|z| !z.is_finite())
            || coord.m.is_some_and(|m| !m.is_finite())
        {
            self.non_finite = true;
            return;
        }
        self.bounds.add_coord(&coord, self.collect_lons);
    }
}

impl GeomProcessor for BoundsProcessor {
    fn multi_dim(&self) -> bool {
        true
    }

    fn xy(&mut self, x: f64, y: f64, _idx: usize) -> geozero::error::Result<()> {
        self.add_coord(Coord {
            x,
            y,
            z: None,
                m: None,
        });
        Ok(())
    }

    fn empty_point(&mut self, _idx: usize) -> geozero::error::Result<()> {
        Ok(())
    }

    fn coordinate(
        &mut self,
        x: f64,
        y: f64,
        z: Option<f64>,
        m: Option<f64>,
        _t: Option<f64>,
        _tm: Option<u64>,
        _idx: usize,
    ) -> geozero::error::Result<()> {
        self.add_coord(Coord { x, y, z, m });
        Ok(())
    }
}

pub(crate) fn bounds(bytes: &[u8], collect_lons: bool) -> Result<Option<GeometryBounds>, GeoError> {
    let mut processor = BoundsProcessor::new(collect_lons);
    let mut cursor = std::io::Cursor::new(bytes);
    if let Err(err) = geozero::wkb::process_wkb_geom(&mut cursor, &mut processor) {
        return Err(GeoError::Wkb(err.to_string()));
    }
    if processor.non_finite {
        return Err(GeoError::Wkb(
            "geometry contains a non-finite coordinate".to_string(),
        ));
    }
    let mut out = processor.bounds;
    if let Some(header_dims) = dims_from_wkb(bytes) {
        out.dims = out.dims.merge(header_dims);
    }
    Ok(out.any.then_some(out))
}

pub(crate) fn geometry_json(bytes: &[u8]) -> Result<serde_json::Value, GeoError> {
    let mut cursor = std::io::Cursor::new(bytes);
    let json = GeoJsonString::from_wkb(&mut cursor, WkbDialect::Wkb)
        .map_err(|e| GeoError::Wkb(e.to_string()))?;
    serde_json::from_str(&json.0).map_err(|e| GeoError::Wkb(e.to_string()))
}

pub(crate) fn dims_from_wkb(bytes: &[u8]) -> Option<CoordinateDims> {
    if bytes.len() < 5 {
        return None;
    }
    let little = match bytes[0] {
        0 => false,
        1 => true,
        _ => return None,
    };
    let raw = if little {
        u32::from_le_bytes(bytes[1..5].try_into().ok()?)
    } else {
        u32::from_be_bytes(bytes[1..5].try_into().ok()?)
    };
    let ewkb_z = raw & 0x8000_0000 != 0;
    let ewkb_m = raw & 0x4000_0000 != 0;
    if ewkb_z || ewkb_m {
        return Some(match (ewkb_z, ewkb_m) {
            (true, true) => CoordinateDims::Xyzm,
            (true, false) => CoordinateDims::Xyz,
            (false, true) => CoordinateDims::Xym,
            (false, false) => CoordinateDims::Xy,
        });
    }
    let code = raw % 1000;
    if raw >= 3000 && code > 0 {
        Some(CoordinateDims::Xyzm)
    } else if raw >= 2000 && code > 0 {
        Some(CoordinateDims::Xym)
    } else if raw >= 1000 && code > 0 {
        Some(CoordinateDims::Xyz)
    } else {
        Some(CoordinateDims::Xy)
    }
}

pub(crate) fn is_empty_point_wkb(bytes: &[u8]) -> bool {
    if bytes.len() < 5 {
        return false;
    }
    let little = match bytes[0] {
        0 => false,
        1 => true,
        _ => return false,
    };
    let raw = read_u32_endian(&bytes[1..5], little);
    let ewkb_z = raw & 0x8000_0000 != 0;
    let ewkb_m = raw & 0x4000_0000 != 0;
    let ewkb_srid = raw & 0x2000_0000 != 0;
    let (base_type, has_z, has_m) = if ewkb_z || ewkb_m || ewkb_srid {
        (raw & 0x0000_FFFF, ewkb_z, ewkb_m)
    } else {
        let base = raw % 1000;
        (
            base,
            (1000..2000).contains(&raw) || raw >= 3000,
            raw >= 2000,
        )
    };
    if base_type != 1 {
        return false;
    }
    let coord_count = 2 + usize::from(has_z) + usize::from(has_m);
    let offset: usize = 5 + if ewkb_srid { 4 } else { 0 };
    let Some(coord_bytes) = coord_count.checked_mul(8) else {
        return false;
    };
    let Some(end) = offset.checked_add(coord_bytes) else {
        return false;
    };
    if bytes.len() < end {
        return false;
    }
    (0..coord_count).all(|i| read_f64_endian(&bytes[offset + i * 8..offset + (i + 1) * 8], little).is_nan())
}

fn read_u32_endian(bytes: &[u8], little: bool) -> u32 {
    let mut value = [0u8; 4];
    value.copy_from_slice(&bytes[..4]);
    if little {
        u32::from_le_bytes(value)
    } else {
        u32::from_be_bytes(value)
    }
}

fn read_f64_endian(bytes: &[u8], little: bool) -> f64 {
    let mut value = [0u8; 8];
    value.copy_from_slice(&bytes[..8]);
    let bits = if little {
        u64::from_le_bytes(value)
    } else {
        u64::from_be_bytes(value)
    };
    f64::from_bits(bits)
}

pub(crate) fn write_geometry(
    kind: GeometryKind,
    dims: CoordinateDims,
    parts: GeometryParts,
) -> Vec<u8> {
    let mut out = Vec::new();
    match (kind, parts) {
        (GeometryKind::Point, GeometryParts::Point(point)) => {
            write_header(&mut out, 1, dims);
            write_coord(&mut out, &point, dims);
        }
        (GeometryKind::LineString, GeometryParts::LineString(line)) => {
            write_line_string(&mut out, &line, dims);
        }
        (GeometryKind::Polygon, GeometryParts::Polygon(rings)) => {
            write_polygon(&mut out, &rings, dims);
        }
        (GeometryKind::MultiPoint, GeometryParts::LineString(points)) => {
            write_header(&mut out, 4, dims);
            write_u32(&mut out, points.len());
            for point in points {
                write_header(&mut out, 1, dims);
                write_coord(&mut out, &point, dims);
            }
        }
        (GeometryKind::MultiLineString, GeometryParts::Polygon(lines)) => {
            write_header(&mut out, 5, dims);
            write_u32(&mut out, lines.len());
            for line in lines {
                write_line_string(&mut out, &line, dims);
            }
        }
        (GeometryKind::MultiPolygon, GeometryParts::MultiPolygon(polygons)) => {
            write_header(&mut out, 6, dims);
            write_u32(&mut out, polygons.len());
            for polygon in polygons {
                write_polygon(&mut out, &polygon, dims);
            }
        }
        _ => {
            write_header(&mut out, 7, dims);
            write_u32(&mut out, 0);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub(crate) enum GeometryParts {
    Point(Coord),
    LineString(Vec<Coord>),
    Polygon(Vec<Vec<Coord>>),
    MultiPolygon(Vec<Vec<Vec<Coord>>>),
}

fn write_line_string(out: &mut Vec<u8>, line: &[Coord], dims: CoordinateDims) {
    write_header(out, 2, dims);
    write_u32(out, line.len());
    for coord in line {
        write_coord(out, coord, dims);
    }
}

fn write_polygon(out: &mut Vec<u8>, rings: &[Vec<Coord>], dims: CoordinateDims) {
    write_header(out, 3, dims);
    write_u32(out, rings.len());
    for ring in rings {
        write_u32(out, ring.len());
        for coord in ring {
            write_coord(out, coord, dims);
        }
    }
}

fn write_header(out: &mut Vec<u8>, base_type: u32, dims: CoordinateDims) {
    out.push(1);
    let code = match dims {
        CoordinateDims::Xy | CoordinateDims::Unknown => base_type,
        CoordinateDims::Xyz => base_type + 1000,
        CoordinateDims::Xym => base_type + 2000,
        CoordinateDims::Xyzm => base_type + 3000,
    };
    out.extend_from_slice(&code.to_le_bytes());
}

fn write_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}

fn write_coord(out: &mut Vec<u8>, coord: &Coord, dims: CoordinateDims) {
    out.extend_from_slice(&coord.x.to_le_bytes());
    out.extend_from_slice(&coord.y.to_le_bytes());
    if dims.has_z() {
        out.extend_from_slice(&coord.z.unwrap_or(0.0).to_le_bytes());
    }
    if dims.has_m() {
        out.extend_from_slice(&coord.m.unwrap_or(0.0).to_le_bytes());
    }
}
