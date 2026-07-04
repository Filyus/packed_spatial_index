use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
    StructArray,
};

use crate::wkb::{self, Coord, GeometryBounds, GeometryParts};
use crate::{
    CoordinateDims, CoordinateLayout, GeoError, GeometryEncoding, GeometryKind, WkbFlavor,
};

#[derive(Debug, Clone)]
pub(crate) struct GeoArrowRow {
    pub bounds: GeometryBounds,
    pub wkb: Vec<u8>,
}

pub(crate) fn encoding_from_geoparquet(value: &str) -> GeometryEncoding {
    if value.eq_ignore_ascii_case("wkb") {
        GeometryEncoding::Wkb {
            flavor: WkbFlavor::Iso,
        }
    } else {
        let kind = GeometryKind::from_geoarrow_encoding(value);
        if kind == GeometryKind::Unknown {
            GeometryEncoding::Unknown(value.to_string())
        } else {
            GeometryEncoding::GeoArrow {
                kind,
                layout: CoordinateLayout::Unknown,
            }
        }
    }
}

pub(crate) fn is_supported_encoding(value: &GeometryEncoding) -> bool {
    match value {
        GeometryEncoding::Wkb { .. }
        | GeometryEncoding::ParquetGeometry
        | GeometryEncoding::ParquetGeography { .. } => true,
        GeometryEncoding::GeoArrow { kind, .. } => *kind != GeometryKind::Unknown,
        // FlatGeobuf/GeoJson encodings belong to their own sources and never
        // appear in a Parquet column.
        _ => false,
    }
}

pub(crate) fn dims_from_arrow(array: &ArrayRef, kind: GeometryKind) -> CoordinateDims {
    let Some(depth) = kind.list_depth() else {
        return CoordinateDims::Unknown;
    };
    dims_at_depth(array.as_ref(), depth)
}

pub(crate) fn scan_row(
    array: &ArrayRef,
    kind: GeometryKind,
    declared_dims: CoordinateDims,
    row: usize,
    collect_lons: bool,
) -> Result<Option<GeoArrowRow>, GeoError> {
    if array.is_null(row) {
        return Ok(None);
    }
    let dims = if declared_dims == CoordinateDims::Unknown {
        dims_from_arrow(array, kind)
    } else {
        declared_dims
    };
    let Some(parts) = parts_for_row(array.as_ref(), kind, row)? else {
        return Ok(None);
    };
    let mut bounds = GeometryBounds::new(collect_lons);
    visit_parts(&parts, |coord| bounds.add_coord(coord, collect_lons));
    if !bounds.any {
        return Ok(None);
    }
    let wkb = wkb::write_geometry(kind, dims, parts);
    Ok(Some(GeoArrowRow { bounds, wkb }))
}

fn dims_at_depth(array: &dyn Array, depth: usize) -> CoordinateDims {
    dims_kind_at_depth(array, depth)
        .map(|(dims, _)| dims)
        .unwrap_or(CoordinateDims::Unknown)
}

fn dims_kind_at_depth(
    array: &dyn Array,
    depth: usize,
) -> Option<(CoordinateDims, CoordinateLayout)> {
    if depth == 0 {
        return coordinate_dims(array);
    }
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        return dims_kind_at_depth(list.values().as_ref(), depth - 1);
    }
    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        return dims_kind_at_depth(list.values().as_ref(), depth - 1);
    }
    None
}

fn coordinate_dims(array: &dyn Array) -> Option<(CoordinateDims, CoordinateLayout)> {
    if let Some(st) = array.as_any().downcast_ref::<StructArray>() {
        let has_z = st.column_by_name("z").is_some();
        let has_m = st.column_by_name("m").is_some();
        let dims = match (has_z, has_m) {
            (true, true) => CoordinateDims::Xyzm,
            (true, false) => CoordinateDims::Xyz,
            (false, true) => CoordinateDims::Xym,
            (false, false) => CoordinateDims::Xy,
        };
        return Some((dims, CoordinateLayout::Struct));
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let dims = match list.value_length() {
            2 => CoordinateDims::Xy,
            3 => CoordinateDims::Xyz,
            4 => CoordinateDims::Xyzm,
            _ => CoordinateDims::Unknown,
        };
        return Some((dims, CoordinateLayout::Interleaved));
    }
    None
}

fn parts_for_row(
    array: &dyn Array,
    kind: GeometryKind,
    row: usize,
) -> Result<Option<GeometryParts>, GeoError> {
    Ok(match kind {
        GeometryKind::Point => coordinate_at(array, row)?.map(GeometryParts::Point),
        GeometryKind::LineString | GeometryKind::MultiPoint => {
            let Some(coords) = coords_list_at(array, row)? else {
                return Ok(None);
            };
            Some(GeometryParts::LineString(coords))
        }
        GeometryKind::Polygon | GeometryKind::MultiLineString => {
            let Some(rings) = lines_list_at(array, row)? else {
                return Ok(None);
            };
            Some(GeometryParts::Polygon(rings))
        }
        GeometryKind::MultiPolygon => {
            let Some(polygons) = polygons_list_at(array, row)? else {
                return Ok(None);
            };
            Some(GeometryParts::MultiPolygon(polygons))
        }
        GeometryKind::Unknown => None,
    })
}

fn coords_list_at(array: &dyn Array, row: usize) -> Result<Option<Vec<Coord>>, GeoError> {
    let Some(values) = list_value(array, row)? else {
        return Ok(None);
    };
    let mut coords = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if let Some(coord) = coordinate_at(values.as_ref(), i)? {
            coords.push(coord);
        }
    }
    Ok(Some(coords))
}

fn lines_list_at(array: &dyn Array, row: usize) -> Result<Option<Vec<Vec<Coord>>>, GeoError> {
    let Some(values) = list_value(array, row)? else {
        return Ok(None);
    };
    let mut lines = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if let Some(line) = coords_list_at(values.as_ref(), i)? {
            lines.push(line);
        }
    }
    Ok(Some(lines))
}

fn polygons_list_at(
    array: &dyn Array,
    row: usize,
) -> Result<Option<Vec<Vec<Vec<Coord>>>>, GeoError> {
    let Some(values) = list_value(array, row)? else {
        return Ok(None);
    };
    let mut polygons = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if let Some(polygon) = lines_list_at(values.as_ref(), i)? {
            polygons.push(polygon);
        }
    }
    Ok(Some(polygons))
}

fn list_value(array: &dyn Array, row: usize) -> Result<Option<ArrayRef>, GeoError> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        return Ok(Some(list.value(row)));
    }
    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        return Ok(Some(list.value(row)));
    }
    Err(GeoError::UnsupportedEncoding(format!(
        "expected List/LargeList geoarrow nesting, got {:?}",
        array.data_type()
    )))
}

fn coordinate_at(array: &dyn Array, row: usize) -> Result<Option<Coord>, GeoError> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(st) = array.as_any().downcast_ref::<StructArray>() {
        let x = number_at(
            st.column_by_name("x")
                .ok_or_else(|| GeoError::Metadata("GeoArrow coordinate missing x".to_string()))?,
            row,
        )?;
        let y = number_at(
            st.column_by_name("y")
                .ok_or_else(|| GeoError::Metadata("GeoArrow coordinate missing y".to_string()))?,
            row,
        )?;
        let z = st
            .column_by_name("z")
            .map(|a| number_at(a, row))
            .transpose()?;
        let m = st
            .column_by_name("m")
            .map(|a| number_at(a, row))
            .transpose()?;
        return Ok(Some(Coord { x, y, z, m }));
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let value = list.value(row);
        let len = value.len();
        if len < 2 {
            return Ok(None);
        }
        let x = number_at(&value, 0)?;
        let y = number_at(&value, 1)?;
        let z = (len >= 3).then(|| number_at(&value, 2)).transpose()?;
        let m = (len >= 4).then(|| number_at(&value, 3)).transpose()?;
        return Ok(Some(Coord { x, y, z, m }));
    }
    Err(GeoError::UnsupportedEncoding(format!(
        "expected GeoArrow coordinate struct/fixed-size-list, got {:?}",
        array.data_type()
    )))
}

fn number_at(array: &ArrayRef, row: usize) -> Result<f64, GeoError> {
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(values.value(row) as f64);
    }
    Err(GeoError::UnsupportedEncoding(format!(
        "GeoArrow coordinate component is not float ({:?})",
        array.data_type()
    )))
}

fn visit_parts<F: FnMut(&Coord)>(parts: &GeometryParts, mut f: F) {
    match parts {
        GeometryParts::Point(coord) => f(coord),
        GeometryParts::LineString(line) => {
            for coord in line {
                f(coord);
            }
        }
        GeometryParts::Polygon(lines) => {
            for line in lines {
                for coord in line {
                    f(coord);
                }
            }
        }
        GeometryParts::MultiPolygon(polygons) => {
            for polygon in polygons {
                for line in polygon {
                    for coord in line {
                        f(coord);
                    }
                }
            }
        }
    }
}
