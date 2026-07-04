#[cfg(any(feature = "parquet", feature = "flatgeobuf"))]
use geozero::GeomProcessor;
#[cfg(feature = "parquet")]
use geozero::geojson::GeoJsonString;
#[cfg(feature = "parquet")]
use geozero::wkb::{FromWkb, WkbDialect};

#[cfg(feature = "_source")]
use crate::CoordinateDims;
use crate::GeoError;
#[cfg(feature = "_source")]
use crate::GeometryKind;

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

#[cfg(any(feature = "parquet", feature = "flatgeobuf"))]
pub(crate) struct BoundsProcessor {
    bounds: GeometryBounds,
    collect_lons: bool,
    non_finite: bool,
}

#[cfg(any(feature = "parquet", feature = "flatgeobuf"))]
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

#[cfg(any(feature = "parquet", feature = "flatgeobuf"))]
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
#[cfg(any(feature = "parquet", feature = "flatgeobuf"))]
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
    if let Some(value) = geometry_json_direct(bytes).map_err(GeoError::Wkb)? {
        return Ok(value);
    }
    let mut cursor = std::io::Cursor::new(bytes);
    let json = GeoJsonString::from_wkb(&mut cursor, WkbDialect::Wkb)
        .map_err(|e| GeoError::Wkb(e.to_string()))?;
    serde_json::from_str(&json.0).map_err(|e| GeoError::Wkb(e.to_string()))
}

#[cfg(feature = "parquet")]
fn geometry_json_direct(bytes: &[u8]) -> Result<Option<serde_json::Value>, String> {
    let mut reader = SimpleWkbReader::new(bytes);
    reader.read_json_geometry(0)
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

pub(crate) fn bbox_intersects_point_or_multipoint(
    bytes: &[u8],
    bbox: packed_spatial_index::Box2D,
) -> Result<Option<bool>, GeoError> {
    let mut reader = SimpleWkbReader::new(bytes);
    let header = reader.read_header().map_err(GeoError::Wkb)?;
    match header.base_type {
        1 => reader
            .read_point_intersects(header, bbox)
            .map(Some)
            .map_err(GeoError::Wkb),
        4 => reader
            .read_multipoint_intersects(header, bbox)
            .map(Some)
            .map_err(GeoError::Wkb),
        _ => Ok(None),
    }
}

#[derive(Clone, Copy)]
struct SimpleWkbHeader {
    little: bool,
    base_type: u32,
    #[cfg(feature = "parquet")]
    has_z: bool,
    #[cfg(feature = "parquet")]
    has_m: bool,
    coord_count: usize,
}

struct SimpleWkbReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SimpleWkbReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_header(&mut self) -> Result<SimpleWkbHeader, String> {
        let little = match self.read_u8()? {
            0 => false,
            1 => true,
            other => return Err(format!("invalid WKB byte order {other}")),
        };
        let raw = self.read_u32(little)?;
        let ewkb_z = raw & 0x8000_0000 != 0;
        let ewkb_m = raw & 0x4000_0000 != 0;
        let ewkb_srid = raw & 0x2000_0000 != 0;
        let (base_type, has_z, has_m) = if ewkb_z || ewkb_m || ewkb_srid {
            if ewkb_srid {
                self.skip(4)?;
            }
            (raw & 0x0000_FFFF, ewkb_z, ewkb_m)
        } else {
            let dim_code = raw / 1000;
            if dim_code > 3 {
                return Err(format!("unsupported WKB geometry type {raw}"));
            }
            (
                raw % 1000,
                dim_code == 1 || dim_code == 3,
                dim_code == 2 || dim_code == 3,
            )
        };
        Ok(SimpleWkbHeader {
            little,
            base_type,
            #[cfg(feature = "parquet")]
            has_z,
            #[cfg(feature = "parquet")]
            has_m,
            coord_count: 2 + usize::from(has_z) + usize::from(has_m),
        })
    }

    #[cfg(feature = "parquet")]
    fn read_json_geometry(&mut self, depth: usize) -> Result<Option<serde_json::Value>, String> {
        if depth > 128 {
            return Err("WKB nesting depth exceeds 128".to_string());
        }
        let header = self.read_header()?;
        self.read_json_geometry_body(header, depth)
    }

    #[cfg(feature = "parquet")]
    fn read_json_geometry_body(
        &mut self,
        header: SimpleWkbHeader,
        depth: usize,
    ) -> Result<Option<serde_json::Value>, String> {
        let value = match header.base_type {
            1 => serde_json::json!({
                "type": "Point",
                "coordinates": self.read_json_coord(header)?,
            }),
            2 => serde_json::json!({
                "type": "LineString",
                "coordinates": self.read_json_coord_sequence(header)?,
            }),
            3 => serde_json::json!({
                "type": "Polygon",
                "coordinates": self.read_json_polygon(header)?,
            }),
            4 => serde_json::json!({
                "type": "MultiPoint",
                "coordinates": self.read_json_child_points(header)?,
            }),
            5 => serde_json::json!({
                "type": "MultiLineString",
                "coordinates": self.read_json_child_lines(header)?,
            }),
            6 => serde_json::json!({
                "type": "MultiPolygon",
                "coordinates": self.read_json_child_polygons(header)?,
            }),
            7 => serde_json::json!({
                "type": "GeometryCollection",
                "geometries": match self.read_json_collection(header, depth)? {
                    Some(geometries) => geometries,
                    None => return Ok(None),
                },
            }),
            _ => return Ok(None),
        };
        Ok(Some(value))
    }

    #[cfg(feature = "parquet")]
    fn read_json_coord(&mut self, header: SimpleWkbHeader) -> Result<serde_json::Value, String> {
        let x = self.read_f64(header.little)?;
        let y = self.read_f64(header.little)?;
        let z = header
            .has_z
            .then(|| self.read_f64(header.little))
            .transpose()?;
        if header.has_m {
            let _ = self.read_f64(header.little)?;
        }
        if x.is_nan() && y.is_nan() && z.is_none_or(f64::is_nan) {
            return Ok(serde_json::Value::Array(Vec::new()));
        }
        let mut values = vec![serde_json::json!(x), serde_json::json!(y)];
        if let Some(z) = z {
            values.push(serde_json::json!(z));
        }
        Ok(serde_json::Value::Array(values))
    }

    #[cfg(feature = "parquet")]
    fn read_json_coord_sequence(
        &mut self,
        header: SimpleWkbHeader,
    ) -> Result<Vec<serde_json::Value>, String> {
        let count = self.read_u32(header.little)? as usize;
        let min_bytes = count
            .checked_mul(header.coord_count * 8)
            .ok_or_else(|| "WKB coordinate sequence count overflows byte size".to_string())?;
        if min_bytes > self.remaining() {
            return Err(format!(
                "WKB coordinate sequence declares {count} items but only {} bytes remain",
                self.remaining()
            ));
        }
        let mut coords = Vec::with_capacity(count);
        for _ in 0..count {
            coords.push(self.read_json_coord(header)?);
        }
        Ok(coords)
    }

    #[cfg(feature = "parquet")]
    fn read_json_polygon(
        &mut self,
        header: SimpleWkbHeader,
    ) -> Result<Vec<Vec<serde_json::Value>>, String> {
        let count = self.read_u32(header.little)? as usize;
        let mut rings = Vec::with_capacity(count);
        for _ in 0..count {
            rings.push(self.read_json_coord_sequence(header)?);
        }
        Ok(rings)
    }

    #[cfg(feature = "parquet")]
    fn read_json_child_points(
        &mut self,
        header: SimpleWkbHeader,
    ) -> Result<Vec<serde_json::Value>, String> {
        let count = self.read_u32(header.little)? as usize;
        let mut points = Vec::with_capacity(count);
        for _ in 0..count {
            let child = self.read_header()?;
            if child.base_type != 1 {
                return Err(format!(
                    "expected Point in WKB MultiPoint, found type {}",
                    child.base_type
                ));
            }
            points.push(self.read_json_coord(child)?);
        }
        Ok(points)
    }

    #[cfg(feature = "parquet")]
    fn read_json_child_lines(
        &mut self,
        header: SimpleWkbHeader,
    ) -> Result<Vec<Vec<serde_json::Value>>, String> {
        let count = self.read_u32(header.little)? as usize;
        let mut lines = Vec::with_capacity(count);
        for _ in 0..count {
            let child = self.read_header()?;
            if child.base_type != 2 {
                return Err(format!(
                    "expected LineString in WKB MultiLineString, found type {}",
                    child.base_type
                ));
            }
            lines.push(self.read_json_coord_sequence(child)?);
        }
        Ok(lines)
    }

    #[cfg(feature = "parquet")]
    fn read_json_child_polygons(
        &mut self,
        header: SimpleWkbHeader,
    ) -> Result<Vec<Vec<Vec<serde_json::Value>>>, String> {
        let count = self.read_u32(header.little)? as usize;
        let mut polygons = Vec::with_capacity(count);
        for _ in 0..count {
            let child = self.read_header()?;
            if child.base_type != 3 {
                return Err(format!(
                    "expected Polygon in WKB MultiPolygon, found type {}",
                    child.base_type
                ));
            }
            polygons.push(self.read_json_polygon(child)?);
        }
        Ok(polygons)
    }

    #[cfg(feature = "parquet")]
    fn read_json_collection(
        &mut self,
        header: SimpleWkbHeader,
        depth: usize,
    ) -> Result<Option<Vec<serde_json::Value>>, String> {
        let count = self.read_u32(header.little)? as usize;
        let mut geometries = Vec::with_capacity(count);
        for _ in 0..count {
            let Some(geometry) = self.read_json_geometry(depth + 1)? else {
                return Ok(None);
            };
            geometries.push(geometry);
        }
        Ok(Some(geometries))
    }

    fn read_point_intersects(
        &mut self,
        header: SimpleWkbHeader,
        bbox: packed_spatial_index::Box2D,
    ) -> Result<bool, String> {
        let x = self.read_f64(header.little)?;
        let y = self.read_f64(header.little)?;
        for _ in 2..header.coord_count {
            let _ = self.read_f64(header.little)?;
        }
        if x.is_nan() && y.is_nan() {
            return Ok(false);
        }
        Ok(x >= bbox.min_x && x <= bbox.max_x && y >= bbox.min_y && y <= bbox.max_y)
    }

    fn read_multipoint_intersects(
        &mut self,
        header: SimpleWkbHeader,
        bbox: packed_spatial_index::Box2D,
    ) -> Result<bool, String> {
        let count = self.read_u32(header.little)? as usize;
        let min_bytes = count
            .checked_mul(5)
            .ok_or_else(|| "WKB MultiPoint count overflows byte size".to_string())?;
        if min_bytes > self.remaining() {
            return Err(format!(
                "WKB MultiPoint declares {count} points but only {} bytes remain",
                self.remaining()
            ));
        }
        for _ in 0..count {
            let point_header = self.read_header()?;
            if point_header.base_type != 1 {
                return Err(format!(
                    "expected Point in WKB MultiPoint, found type {}",
                    point_header.base_type
                ));
            }
            if self.read_point_intersects(point_header, bbox)? {
                return Ok(true);
            }
        }
        Ok(false)
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

#[cfg(feature = "_source")]
pub(crate) fn write_geometry(
    kind: GeometryKind,
    dims: CoordinateDims,
    parts: GeometryParts,
) -> Vec<u8> {
    let mut out = Vec::new();
    write_geometry_into(&mut out, kind, dims, parts);
    out
}

#[cfg(feature = "_source")]
fn write_geometry_into(
    out: &mut Vec<u8>,
    kind: GeometryKind,
    dims: CoordinateDims,
    parts: GeometryParts,
) {
    match (kind, parts) {
        (GeometryKind::Point, GeometryParts::Point(point)) => {
            write_header(out, 1, dims);
            write_coord(out, &point, dims);
        }
        (GeometryKind::LineString, GeometryParts::LineString(line)) => {
            write_line_string(out, &line, dims);
        }
        (GeometryKind::Polygon, GeometryParts::Polygon(rings)) => {
            write_polygon(out, &rings, dims);
        }
        (GeometryKind::MultiPoint, GeometryParts::LineString(points)) => {
            write_header(out, 4, dims);
            write_u32(out, points.len());
            for point in points {
                write_header(out, 1, dims);
                write_coord(out, &point, dims);
            }
        }
        (GeometryKind::MultiLineString, GeometryParts::Polygon(lines)) => {
            write_header(out, 5, dims);
            write_u32(out, lines.len());
            for line in lines {
                write_line_string(out, &line, dims);
            }
        }
        (GeometryKind::MultiPolygon, GeometryParts::MultiPolygon(polygons)) => {
            write_header(out, 6, dims);
            write_u32(out, polygons.len());
            for polygon in polygons {
                write_polygon(out, &polygon, dims);
            }
        }
        (GeometryKind::Unknown, GeometryParts::GeometryCollection(children)) => {
            write_header(out, 7, dims);
            write_u32(out, children.len());
            for (kind, child) in children {
                write_geometry_into(out, kind, dims, child);
            }
        }
        _ => {
            write_header(out, 7, dims);
            write_u32(out, 0);
        }
    }
}

#[cfg(feature = "_source")]
#[derive(Debug, Clone)]
pub(crate) enum GeometryParts {
    Point(Coord),
    LineString(Vec<Coord>),
    Polygon(Vec<Vec<Coord>>),
    MultiPolygon(Vec<Vec<Vec<Coord>>>),
    GeometryCollection(Vec<(GeometryKind, GeometryParts)>),
}

#[cfg(feature = "_source")]
fn write_line_string(out: &mut Vec<u8>, line: &[Coord], dims: CoordinateDims) {
    write_header(out, 2, dims);
    write_u32(out, line.len());
    for coord in line {
        write_coord(out, coord, dims);
    }
}

#[cfg(feature = "_source")]
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

#[cfg(feature = "_source")]
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

#[cfg(feature = "_source")]
fn write_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}

#[cfg(feature = "_source")]
fn write_coord(out: &mut Vec<u8>, coord: &Coord, dims: CoordinateDims) {
    out.extend_from_slice(&coord.x.to_le_bytes());
    out.extend_from_slice(&coord.y.to_le_bytes());
    if dims.has_z() {
        out.extend_from_slice(&coord.z.unwrap_or(0.0).to_le_bytes());
    }
    if matches!(dims, CoordinateDims::Xym | CoordinateDims::Xyzm) {
        out.extend_from_slice(&coord.m.unwrap_or(0.0).to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packed_spatial_index::Box2D;

    #[test]
    fn point_bbox_fast_path_matches_and_misses() {
        let bbox = Box2D::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(
            bbox_intersects_point_or_multipoint(&point_wkb(5.0, 5.0), bbox).unwrap(),
            Some(true)
        );
        assert_eq!(
            bbox_intersects_point_or_multipoint(&point_wkb(11.0, 5.0), bbox).unwrap(),
            Some(false)
        );
    }

    #[test]
    fn multipoint_bbox_fast_path_matches_any_child() {
        let bbox = Box2D::new(0.0, 0.0, 10.0, 10.0);
        let mut wkb = Vec::new();
        wkb.push(1);
        wkb.extend_from_slice(&4u32.to_le_bytes());
        wkb.extend_from_slice(&2u32.to_le_bytes());
        wkb.extend_from_slice(&point_wkb(20.0, 20.0));
        wkb.extend_from_slice(&point_wkb(3.0, 4.0));
        assert_eq!(
            bbox_intersects_point_or_multipoint(&wkb, bbox).unwrap(),
            Some(true)
        );
    }

    #[test]
    fn bbox_fast_path_falls_back_for_non_point_types() {
        let bbox = Box2D::new(0.0, 0.0, 10.0, 10.0);
        let mut line = Vec::new();
        line.push(1);
        line.extend_from_slice(&2u32.to_le_bytes());
        line.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            bbox_intersects_point_or_multipoint(&line, bbox).unwrap(),
            None
        );
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn geometry_json_direct_handles_core_point() {
        let json = geometry_json(&point_wkb(5.0, 6.0)).unwrap();
        assert_eq!(json["type"], "Point");
        assert_eq!(json["coordinates"], serde_json::json!([5.0, 6.0]));
    }

    fn point_wkb(x: f64, y: f64) -> Vec<u8> {
        let mut wkb = Vec::new();
        wkb.push(1);
        wkb.extend_from_slice(&1u32.to_le_bytes());
        wkb.extend_from_slice(&x.to_le_bytes());
        wkb.extend_from_slice(&y.to_le_bytes());
        wkb
    }
}
