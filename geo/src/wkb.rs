#[cfg(feature = "_source")]
use geozero::GeomProcessor;
#[cfg(feature = "parquet")]
use geozero::geojson::GeoJsonString;
#[cfg(feature = "parquet")]
use geozero::wkb::{FromWkb, WkbDialect};

#[cfg(feature = "_source")]
use crate::CoordinateDims;
#[cfg(feature = "parquet")]
use crate::{GeoError, GeometryKind};

#[cfg(feature = "_source")]
#[derive(Debug, Clone)]
pub(crate) struct Coord {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
    pub m: Option<f64>,
}

#[cfg(feature = "_source")]
#[derive(Debug, Clone)]
pub(crate) struct GeometryBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub dims: CoordinateDims,
    pub any: bool,
    pub lon_values: Vec<f64>,
    pub from_covering: bool,
}

#[cfg(feature = "_source")]
impl GeometryBounds {
    #[cfg(feature = "_source")]
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

    #[cfg(feature = "_source")]
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

    #[cfg(feature = "_source")]
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

#[cfg(feature = "_source")]
pub(crate) struct BoundsProcessor {
    bounds: GeometryBounds,
    collect_lons: bool,
    non_finite: bool,
}

#[cfg(feature = "_source")]
impl BoundsProcessor {
    #[cfg(feature = "_source")]
    fn new(collect_lons: bool) -> Self {
        Self {
            bounds: GeometryBounds::new(collect_lons),
            collect_lons,
            non_finite: false,
        }
    }

    #[cfg(feature = "_source")]
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

    #[cfg(feature = "_source")]
    fn finish(self) -> Result<Option<GeometryBounds>, String> {
        if self.non_finite {
            return Err("geometry contains a non-finite coordinate".to_string());
        }
        Ok(self.bounds.any.then_some(self.bounds))
    }
}

#[cfg(feature = "_source")]
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

#[cfg(feature = "parquet")]
pub(crate) fn bounds(bytes: &[u8], collect_lons: bool) -> Result<Option<GeometryBounds>, GeoError> {
    WkbBoundsReader::new(bytes, collect_lons)
        .read()
        .map_err(GeoError::Wkb)
}

/// Run any geozero geometry-processing closure through the shared
/// [`BoundsProcessor`] accumulator: same non-finite rejection, dimension
/// detection, and antimeridian longitude collection for every source format.
/// Errors are returned as plain strings so each format can wrap them in its
/// own [`GeoError`] variant.
#[cfg(feature = "_source")]
pub(crate) fn bounds_from_geozero(
    process: impl FnOnce(&mut BoundsProcessor) -> geozero::error::Result<()>,
    collect_lons: bool,
) -> Result<Option<GeometryBounds>, String> {
    let mut processor = BoundsProcessor::new(collect_lons);
    if let Err(err) = process(&mut processor) {
        return Err(err.to_string());
    }
    processor.finish()
}

#[cfg(feature = "parquet")]
const WKB_MAX_NESTING_DEPTH: usize = 128;

#[cfg(feature = "parquet")]
const WKB_HEADER_BYTES: usize = 5;

#[cfg(feature = "parquet")]
#[derive(Clone, Copy)]
struct WkbHeader {
    little: bool,
    base_type: u32,
    dims: CoordinateDims,
}

#[cfg(feature = "parquet")]
impl WkbHeader {
    fn coord_count(self) -> usize {
        2 + usize::from(self.dims.has_z()) + usize::from(self.dims.has_m())
    }

    fn coord_bytes(self) -> usize {
        self.coord_count() * 8
    }
}

#[cfg(feature = "parquet")]
struct WkbBoundsReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    processor: BoundsProcessor,
}

#[cfg(feature = "parquet")]
impl<'a> WkbBoundsReader<'a> {
    fn new(bytes: &'a [u8], collect_lons: bool) -> Self {
        Self {
            bytes,
            pos: 0,
            processor: BoundsProcessor::new(collect_lons),
        }
    }

    fn read(mut self) -> Result<Option<GeometryBounds>, String> {
        self.read_geometry(0)?;
        self.processor.finish()
    }

    fn read_geometry(&mut self, depth: usize) -> Result<(), String> {
        if depth > WKB_MAX_NESTING_DEPTH {
            return Err(format!("WKB nesting depth exceeds {WKB_MAX_NESTING_DEPTH}"));
        }
        let header = self.read_header()?;
        self.read_geometry_body(header, depth)
    }

    fn read_typed_geometry(
        &mut self,
        depth: usize,
        expected: &[u32],
        context: &str,
    ) -> Result<(), String> {
        if depth > WKB_MAX_NESTING_DEPTH {
            return Err(format!("WKB nesting depth exceeds {WKB_MAX_NESTING_DEPTH}"));
        }
        let header = self.read_header()?;
        if !expected.contains(&header.base_type) {
            return Err(format!(
                "expected {context} in WKB collection, found type {}",
                header.base_type
            ));
        }
        self.read_geometry_body(header, depth)
    }

    fn read_geometry_body(&mut self, header: WkbHeader, depth: usize) -> Result<(), String> {
        match header.base_type {
            1 => self.read_point(header),
            2 | 8 => self.read_coord_sequence(header),
            3 | 17 => self.read_polygon(header),
            4 => self.read_child_collection(header, depth, &[1], "Point"),
            5 => self.read_child_collection(header, depth, &[2], "LineString"),
            6 => self.read_child_collection(header, depth, &[3], "Polygon"),
            7 => self.read_any_collection(header, depth),
            9 => self.read_child_collection(header, depth, &[2, 8], "curve"),
            10 => self.read_child_collection(header, depth, &[2, 8, 9], "curve"),
            11 => self.read_child_collection(header, depth, &[2, 8, 9], "curve"),
            12 => self.read_child_collection(header, depth, &[3, 10], "surface"),
            15 => self.read_child_collection(header, depth, &[3], "Polygon"),
            16 => self.read_child_collection(header, depth, &[17], "Triangle"),
            other => Err(format!("unsupported WKB geometry type {other}")),
        }
    }

    fn read_header(&mut self) -> Result<WkbHeader, String> {
        let little = match self.read_u8()? {
            0 => false,
            1 => true,
            other => return Err(format!("invalid WKB byte order {other}")),
        };
        let raw = self.read_u32(little)?;
        let ewkb_z = raw & 0x8000_0000 != 0;
        let ewkb_m = raw & 0x4000_0000 != 0;
        let ewkb_srid = raw & 0x2000_0000 != 0;
        let (base_type, dims) = if ewkb_z || ewkb_m || ewkb_srid {
            if ewkb_srid {
                self.skip(4)?;
            }
            (
                raw & 0x0000_FFFF,
                match (ewkb_z, ewkb_m) {
                    (true, true) => CoordinateDims::Xyzm,
                    (true, false) => CoordinateDims::Xyz,
                    (false, true) => CoordinateDims::Xym,
                    (false, false) => CoordinateDims::Xy,
                },
            )
        } else {
            let dim_code = raw / 1000;
            if dim_code > 3 {
                return Err(format!("unsupported WKB geometry type {raw}"));
            }
            (
                raw % 1000,
                match dim_code {
                    1 => CoordinateDims::Xyz,
                    2 => CoordinateDims::Xym,
                    3 => CoordinateDims::Xyzm,
                    _ => CoordinateDims::Xy,
                },
            )
        };
        if !(1..=17).contains(&base_type) || matches!(base_type, 13 | 14) {
            return Err(format!("unsupported WKB geometry type {base_type}"));
        }
        Ok(WkbHeader {
            little,
            base_type,
            dims,
        })
    }

    fn read_point(&mut self, header: WkbHeader) -> Result<(), String> {
        let coord = self.read_coord(header)?;
        if is_empty_point_coord(&coord) {
            return Ok(());
        }
        self.processor.add_coord(coord);
        Ok(())
    }

    fn read_coord_sequence(&mut self, header: WkbHeader) -> Result<(), String> {
        let len = self.read_bounded_count(header, header.coord_bytes(), "coordinate sequence")?;
        for _ in 0..len {
            let coord = self.read_coord(header)?;
            self.processor.add_coord(coord);
        }
        Ok(())
    }

    fn read_polygon(&mut self, header: WkbHeader) -> Result<(), String> {
        let rings = self.read_bounded_count(header, 4, "polygon ring list")?;
        for _ in 0..rings {
            self.read_coord_sequence(header)?;
        }
        Ok(())
    }

    fn read_any_collection(&mut self, header: WkbHeader, depth: usize) -> Result<(), String> {
        let count = self.read_bounded_count(header, WKB_HEADER_BYTES, "geometry collection")?;
        for _ in 0..count {
            self.read_geometry(depth + 1)?;
        }
        Ok(())
    }

    fn read_child_collection(
        &mut self,
        header: WkbHeader,
        depth: usize,
        expected: &[u32],
        context: &str,
    ) -> Result<(), String> {
        let count = self.read_bounded_count(header, WKB_HEADER_BYTES, context)?;
        for _ in 0..count {
            self.read_typed_geometry(depth + 1, expected, context)?;
        }
        Ok(())
    }

    fn read_bounded_count(
        &mut self,
        header: WkbHeader,
        min_item_bytes: usize,
        context: &str,
    ) -> Result<usize, String> {
        let count = self.read_u32(header.little)? as usize;
        let min_bytes = count
            .checked_mul(min_item_bytes)
            .ok_or_else(|| format!("WKB {context} count overflows byte size"))?;
        let remaining = self.remaining();
        if min_bytes > remaining {
            return Err(format!(
                "WKB {context} declares {count} items but only {remaining} bytes remain"
            ));
        }
        Ok(count)
    }

    fn read_coord(&mut self, header: WkbHeader) -> Result<Coord, String> {
        Ok(Coord {
            x: self.read_f64(header.little)?,
            y: self.read_f64(header.little)?,
            z: header
                .dims
                .has_z()
                .then(|| self.read_f64(header.little))
                .transpose()?,
            m: header
                .dims
                .has_m()
                .then(|| self.read_f64(header.little))
                .transpose()?,
        })
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let byte = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| "unexpected end of WKB".to_string())?;
        self.pos += 1;
        Ok(byte)
    }

    fn read_u32(&mut self, little: bool) -> Result<u32, String> {
        let bytes = self.read_array::<4>()?;
        Ok(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    }

    fn read_f64(&mut self, little: bool) -> Result<f64, String> {
        let bytes = self.read_array::<8>()?;
        let bits = if little {
            u64::from_le_bytes(bytes)
        } else {
            u64::from_be_bytes(bytes)
        };
        Ok(f64::from_bits(bits))
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or_else(|| "WKB offset overflow".to_string())?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| "unexpected end of WKB".to_string())?;
        self.pos = end;
        Ok(bytes.try_into().expect("slice length is fixed"))
    }

    fn skip(&mut self, len: usize) -> Result<(), String> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| "WKB offset overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected end of WKB".to_string());
        }
        self.pos = end;
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}

#[cfg(feature = "parquet")]
fn is_empty_point_coord(coord: &Coord) -> bool {
    coord.x.is_nan()
        && coord.y.is_nan()
        && coord.z.is_none_or(f64::is_nan)
        && coord.m.is_none_or(f64::is_nan)
}

#[cfg(feature = "parquet")]
pub(crate) fn geometry_json(bytes: &[u8]) -> Result<serde_json::Value, GeoError> {
    let mut cursor = std::io::Cursor::new(bytes);
    let json = GeoJsonString::from_wkb(&mut cursor, WkbDialect::Wkb)
        .map_err(|e| GeoError::Wkb(e.to_string()))?;
    serde_json::from_str(&json.0).map_err(|e| GeoError::Wkb(e.to_string()))
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
    (0..coord_count)
        .all(|i| read_f64_endian(&bytes[offset + i * 8..offset + (i + 1) * 8], little).is_nan())
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
#[derive(Debug, Clone)]
pub(crate) enum GeometryParts {
    Point(Coord),
    LineString(Vec<Coord>),
    Polygon(Vec<Vec<Coord>>),
    MultiPolygon(Vec<Vec<Vec<Coord>>>),
}

#[cfg(feature = "parquet")]
fn write_line_string(out: &mut Vec<u8>, line: &[Coord], dims: CoordinateDims) {
    write_header(out, 2, dims);
    write_u32(out, line.len());
    for coord in line {
        write_coord(out, coord, dims);
    }
}

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
fn write_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}

#[cfg(feature = "parquet")]
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
