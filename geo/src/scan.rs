use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, LargeBinaryArray,
    StructArray,
};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::file::reader::ChunkReader;

use crate::dataset::GeoDataset;
use crate::discovery::{self, ColumnState, GeoParquetBboxCovering};
use crate::feature_read;
use crate::geoarrow;
use crate::payload::{self, FeatureRef};
use crate::scan_core::{self, ScanEntry};
use crate::wkb::{self, GeometryBounds};
use crate::{
    CoordinateDims, EnvelopePolicy, GeoError, GeometryEncoding, GeometryScan, IndexDimsRequest,
    InspectRequest, NullPolicy, PayloadPlan, ScanRequest,
};

/// Run a scan request against the selected column, materializing one envelope
/// per row into a [`GeometryScan`]. Consumes the dataset's reader, mirroring
/// [`GeoDataset::read_features`](crate::dataset::GeoDataset::read_features)'s
/// take-then-use pattern rather than peeking the builder before consuming it.
pub(crate) fn scan_selected<R: ChunkReader + 'static>(
    dataset: &mut GeoDataset<R>,
    state: &ColumnState,
    req: ScanRequest,
) -> Result<GeometryScan, GeoError> {
    if matches!(req.envelope, EnvelopePolicy::Geographic { .. })
        && state.info.crs.is_known_projected()
    {
        return Err(GeoError::Metadata(format!(
            "column `{}` has a projected CRS; geographic antimeridian handling is only valid for lon/lat coordinates",
            state.info.name
        )));
    }
    let builder = dataset.take_builder()?;
    let row_groups = feature_read::row_group_spans(builder.metadata());
    let need_geometry_payload = matches!(
        req.payload,
        PayloadPlan::RowWkb | PayloadPlan::FeatureJson { .. }
    );
    let scan_roots = scan_projection_roots(builder.schema().as_ref(), state, &req)?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), scan_roots.iter().copied());
    let batches = builder
        .with_projection(projection)
        .build()
        .map_err(GeoError::from)?;
    let mut entries = Vec::new();
    let mut detected_dims = state.info.coordinate_dims;
    let collect_lons = matches!(req.envelope, EnvelopePolicy::Geographic { .. });
    let mut row_base = 0u64;
    let mut row_group_cursor = 0usize;

    for batch in batches {
        let batch = batch?;
        let geom = batch
            .column_by_name(&state.info.name)
            .ok_or_else(|| GeoError::GeometryColumnNotFound(state.info.name.clone()))?
            .clone();
        let binary = needs_binary(&state.info.encoding)
            .then(|| binary_column(&batch, &state.info.name))
            .transpose()?;
        let covering = covering_arrays(&batch, state.covering.as_ref())?;
        let property_columns =
            feature_read::projection_columns(&batch, &state.info.name, &req.payload)?;
        let property_rows = match &req.payload {
            PayloadPlan::FeatureJson { properties } => {
                feature_read::feature_json_property_rows(&batch, properties, &property_columns)?
            }
            _ => None,
        };
        for row in 0..batch.num_rows() {
            let row_number = row_base + row as u64;
            let (row_group, row_in_group) =
                feature_read::row_group_for_row(row_number, &row_groups, &mut row_group_cursor)?;
            let scanned = scan_one_row(
                state,
                &geom,
                binary.as_ref(),
                covering.as_ref(),
                row,
                collect_lons,
                need_geometry_payload,
            )?;
            let Some((bounds, wkb)) = scanned else {
                match req.nulls {
                    NullPolicy::Skip => continue,
                    NullPolicy::Error => {
                        return Err(GeoError::NullGeometry {
                            row: row_number as usize,
                        });
                    }
                }
            };
            detected_dims = detected_dims.merge(bounds.dims);
            let feature = FeatureRef::row_in_group(row_number, row_group, row_in_group);
            let property_json = match &req.payload {
                PayloadPlan::FeatureJson { .. } => Some(feature_read::feature_json_payload(
                    &feature,
                    wkb.as_deref(),
                    property_rows.as_ref().map(|rows| rows[row].clone()),
                )?),
                PayloadPlan::RowRef => Some(payload::encode_feature_ref(&feature)),
                PayloadPlan::RowWkb => Some(payload::encode_feature_wkb(
                    &feature,
                    wkb.as_deref().ok_or_else(|| {
                        GeoError::UnsupportedEncoding(format!(
                            "{} cannot emit WKB payload",
                            state.info.encoding
                        ))
                    })?,
                )),
                PayloadPlan::None => None,
            };
            entries.push(ScanEntry {
                bounds,
                feature,
                payload: property_json,
            });
        }
        row_base += batch.num_rows() as u64;
    }

    let profile = discovery::profile_from_state(state, dataset.discovery().num_rows);
    scan_core::assemble_scan(entries, &req, profile, detected_dims)
}

fn scan_projection_roots(
    schema: &Schema,
    state: &ColumnState,
    req: &ScanRequest,
) -> Result<Vec<usize>, GeoError> {
    let mut roots = Vec::new();
    roots.push(feature_read::root_column_index(schema, &state.info.name)?);
    if let Some(covering) = state.covering.as_ref() {
        push_covering_root(schema, &mut roots, &covering.xmin)?;
        push_covering_root(schema, &mut roots, &covering.ymin)?;
        push_covering_root(schema, &mut roots, &covering.xmax)?;
        push_covering_root(schema, &mut roots, &covering.ymax)?;
        if let Some(path) = &covering.zmin {
            push_covering_root(schema, &mut roots, path)?;
        }
        if let Some(path) = &covering.zmax {
            push_covering_root(schema, &mut roots, path)?;
        }
    }
    if let PayloadPlan::FeatureJson { properties } = &req.payload {
        roots.extend(feature_read::property_root_indices(
            schema,
            &state.info.name,
            properties,
        )?);
    }
    roots.sort_unstable();
    roots.dedup();
    Ok(roots)
}

fn push_covering_root(
    schema: &Schema,
    roots: &mut Vec<usize>,
    path: &[String],
) -> Result<(), GeoError> {
    let root = path
        .first()
        .ok_or_else(|| GeoError::Metadata("empty bbox covering path".to_string()))?;
    roots.push(feature_read::root_column_index(schema, root)?);
    Ok(())
}

pub(crate) struct ScanRequestForInspect;

impl ScanRequestForInspect {
    pub(crate) fn from_request(req: InspectRequest) -> ScanRequest {
        ScanRequest {
            selector: req.selector,
            dims: IndexDimsRequest::Auto,
            nulls: NullPolicy::Skip,
            envelope: EnvelopePolicy::Planar,
            payload: PayloadPlan::None,
        }
    }
}

pub(crate) enum WkbCol<'a> {
    Bin(&'a BinaryArray),
    Large(&'a LargeBinaryArray),
    View(&'a BinaryViewArray),
}

impl WkbCol<'_> {
    pub(crate) fn is_null(&self, row: usize) -> bool {
        match self {
            WkbCol::Bin(array) => array.is_null(row),
            WkbCol::Large(array) => array.is_null(row),
            WkbCol::View(array) => array.is_null(row),
        }
    }

    pub(crate) fn value(&self, row: usize) -> &[u8] {
        match self {
            WkbCol::Bin(array) => array.value(row),
            WkbCol::Large(array) => array.value(row),
            WkbCol::View(array) => array.value(row),
        }
    }
}

pub(crate) fn needs_binary(encoding: &GeometryEncoding) -> bool {
    encoding.is_wkb_payload()
}

pub(crate) fn binary_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<WkbCol<'a>, GeoError> {
    let arr = batch
        .column_by_name(name)
        .ok_or_else(|| GeoError::GeometryColumnNotFound(name.to_string()))?;
    if let Some(a) = arr.as_any().downcast_ref::<BinaryArray>() {
        Ok(WkbCol::Bin(a))
    } else if let Some(a) = arr.as_any().downcast_ref::<LargeBinaryArray>() {
        Ok(WkbCol::Large(a))
    } else if let Some(a) = arr.as_any().downcast_ref::<BinaryViewArray>() {
        Ok(WkbCol::View(a))
    } else {
        Err(GeoError::UnsupportedEncoding(format!(
            "geometry column `{name}` is not binary WKB ({:?})",
            arr.data_type()
        )))
    }
}

struct CoveringArrays {
    xmin: Vec<f64>,
    ymin: Vec<f64>,
    zmin: Option<Vec<f64>>,
    xmax: Vec<f64>,
    ymax: Vec<f64>,
    zmax: Option<Vec<f64>>,
}

fn covering_arrays(
    batch: &RecordBatch,
    covering: Option<&GeoParquetBboxCovering>,
) -> Result<Option<CoveringArrays>, GeoError> {
    let Some(covering) = covering else {
        return Ok(None);
    };
    Ok(Some(CoveringArrays {
        xmin: f64_path(batch, &covering.xmin)?,
        ymin: f64_path(batch, &covering.ymin)?,
        zmin: covering
            .zmin
            .as_ref()
            .map(|path| f64_path(batch, path))
            .transpose()?,
        xmax: f64_path(batch, &covering.xmax)?,
        ymax: f64_path(batch, &covering.ymax)?,
        zmax: covering
            .zmax
            .as_ref()
            .map(|path| f64_path(batch, path))
            .transpose()?,
    }))
}

fn f64_path(batch: &RecordBatch, path: &[String]) -> Result<Vec<f64>, GeoError> {
    let arr = descend(batch, path)?;
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        Ok((0..a.len()).map(|i| a.value(i)).collect())
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        Ok((0..a.len()).map(|i| a.value(i) as f64).collect())
    } else {
        Err(GeoError::Metadata(format!(
            "bbox covering path {path:?} is not a float column ({:?})",
            arr.data_type()
        )))
    }
}

fn descend<'a>(batch: &'a RecordBatch, path: &[String]) -> Result<&'a ArrayRef, GeoError> {
    let first = path
        .first()
        .ok_or_else(|| GeoError::Metadata("empty bbox covering path".to_string()))?;
    let mut cur = batch
        .column_by_name(first)
        .ok_or_else(|| GeoError::Metadata(format!("bbox covering column `{first}` not found")))?;
    for field in &path[1..] {
        let st = cur.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
            GeoError::Metadata(format!(
                "bbox covering path segment `{field}` is not a struct"
            ))
        })?;
        cur = st.column_by_name(field).ok_or_else(|| {
            GeoError::Metadata(format!("bbox covering field `{field}` not found"))
        })?;
    }
    Ok(cur)
}

type RowScanResult = Option<(GeometryBounds, Option<Vec<u8>>)>;

fn scan_one_row(
    state: &ColumnState,
    geom: &ArrayRef,
    binary: Option<&WkbCol<'_>>,
    covering: Option<&CoveringArrays>,
    row: usize,
    collect_lons: bool,
    need_geometry_payload: bool,
) -> Result<RowScanResult, GeoError> {
    if geom.is_null(row) {
        return Ok(None);
    }
    if let Some(binary) = binary {
        if binary.is_null(row) {
            return Ok(None);
        }
        let wkb = binary.value(row);
        let bounds = if let Some(covering) = covering.filter(|_| !need_geometry_payload) {
            bounds_from_covering(covering, row, collect_lons)?
        } else {
            let Some(bounds) = wkb::bounds(wkb, collect_lons)? else {
                return Ok(None);
            };
            bounds
        };
        return Ok(Some((bounds, need_geometry_payload.then(|| wkb.to_vec()))));
    }
    let GeometryEncoding::GeoArrow { kind, .. } = state.info.encoding else {
        return Err(GeoError::UnsupportedEncoding(
            state.info.encoding.to_string(),
        ));
    };
    if let Some(covering) = covering.filter(|_| !need_geometry_payload) {
        return Ok(Some((
            bounds_from_covering(covering, row, collect_lons)?,
            None,
        )));
    }
    let Some(row) = geoarrow::scan_row(geom, kind, state.info.coordinate_dims, row, collect_lons)?
    else {
        return Ok(None);
    };
    Ok(Some((row.bounds, need_geometry_payload.then_some(row.wkb))))
}

fn bounds_from_covering(
    covering: &CoveringArrays,
    row: usize,
    collect_lons: bool,
) -> Result<GeometryBounds, GeoError> {
    let mut bounds = GeometryBounds::new(collect_lons);
    bounds.min[0] = covering.xmin[row];
    bounds.min[1] = covering.ymin[row];
    bounds.max[0] = covering.xmax[row];
    bounds.max[1] = covering.ymax[row];
    if !bounds.min[0].is_finite()
        || !bounds.min[1].is_finite()
        || !bounds.max[0].is_finite()
        || !bounds.max[1].is_finite()
    {
        return Err(GeoError::Metadata(
            "bbox covering contains a non-finite coordinate".to_string(),
        ));
    }
    if let (Some(zmin), Some(zmax)) = (&covering.zmin, &covering.zmax) {
        bounds.min[2] = zmin[row];
        bounds.max[2] = zmax[row];
        if !bounds.min[2].is_finite() || !bounds.max[2].is_finite() {
            return Err(GeoError::Metadata(
                "bbox covering contains a non-finite coordinate".to_string(),
            ));
        }
        bounds.dims = CoordinateDims::Xyz;
    } else {
        bounds.dims = CoordinateDims::Xy;
    }
    bounds.any = true;
    bounds.from_covering = true;
    if collect_lons {
        bounds.lon_values.push(bounds.min[0]);
        bounds.lon_values.push(bounds.max[0]);
    }
    Ok(bounds)
}
