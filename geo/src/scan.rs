use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, LargeBinaryArray,
    StructArray,
};
use arrow::record_batch::RecordBatch;
use packed_spatial_index::{Box2D, Box3D};
use parquet::file::reader::ChunkReader;

use crate::dataset::GeoDataset;
use crate::discovery::{self, ColumnState, GeoParquetBboxCovering};
use crate::feature_read;
use crate::geoarrow;
use crate::payload::{self, FeatureRef};
use crate::wkb::{self, GeometryBounds};
use crate::{
    AntimeridianPolicy, CoordinateDims, EnvelopePolicy, GeoError, GeometryEncoding,
    GeometryProfile, GeometrySelector, IndexDimsRequest, InspectRequest, NullPolicy, PayloadPlan,
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
    let builder = dataset.take_builder()?;
    let row_groups = feature_read::row_group_spans(builder.metadata());
    let batches = builder.build().map_err(GeoError::from)?;
    let mut entries = Vec::new();
    let mut detected_dims = state.info.coordinate_dims;
    let collect_lons = matches!(req.envelope, EnvelopePolicy::Geographic { .. });
    let want_payload = !matches!(req.payload, PayloadPlan::None);
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
                want_payload,
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
                PayloadPlan::FeatureJson { properties } => {
                    Some(feature_read::feature_json_payload(
                        &feature,
                        wkb.as_deref(),
                        &batch,
                        row,
                        properties,
                        &property_columns,
                    )?)
                }
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

    let dims = resolve_scan_dims(req.dims, detected_dims, &entries)?;
    let mut profile = discovery::profile_from_state(state, dataset.discovery().num_rows);
    profile.coordinate_dims = detected_dims;

    match dims {
        ResolvedDims::D2 => {
            let mut boxes = Vec::new();
            let mut features = Vec::new();
            let mut payloads = req.payload_payloads();
            for entry in entries {
                let parts = split_2d(&entry.bounds, req.envelope, entry.feature.row_number)?;
                let has_parts = parts.len() > 1;
                for (part_index, bbox) in parts.into_iter().enumerate() {
                    let mut feature = entry.feature.clone();
                    if has_parts {
                        feature.part = Some(part_index as u16);
                    }
                    boxes.push(bbox);
                    features.push(feature);
                    if let Some(payloads) = payloads.as_mut() {
                        payloads.push(entry.payload.clone().unwrap_or_default());
                    }
                }
            }
            Ok(GeometryScan::D2(GeometryScan2D {
                boxes,
                features,
                payloads,
                profile,
            }))
        }
        ResolvedDims::D3 => {
            let mut boxes = Vec::new();
            let mut features = Vec::new();
            let mut payloads = req.payload_payloads();
            for entry in entries {
                let parts = split_3d(&entry.bounds, req.envelope, entry.feature.row_number)?;
                let has_parts = parts.len() > 1;
                for (part_index, bbox) in parts.into_iter().enumerate() {
                    let mut feature = entry.feature.clone();
                    if has_parts {
                        feature.part = Some(part_index as u16);
                    }
                    boxes.push(bbox);
                    features.push(feature);
                    if let Some(payloads) = payloads.as_mut() {
                        payloads.push(entry.payload.clone().unwrap_or_default());
                    }
                }
            }
            Ok(GeometryScan::D3(GeometryScan3D {
                boxes,
                features,
                payloads,
                profile,
            }))
        }
    }
}

trait PayloadVec {
    fn payload_payloads(&self) -> Option<Vec<Vec<u8>>>;
}

impl PayloadVec for ScanRequest {
    fn payload_payloads(&self) -> Option<Vec<Vec<u8>>> {
        (!matches!(self.payload, PayloadPlan::None)).then(Vec::new)
    }
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
    want_payload: bool,
) -> Result<RowScanResult, GeoError> {
    if geom.is_null(row) {
        return Ok(None);
    }
    if let Some(binary) = binary {
        if binary.is_null(row) {
            return Ok(None);
        }
        let wkb = binary.value(row);
        let bounds = if let Some(covering) = covering.filter(|_| !want_payload) {
            bounds_from_covering(covering, row, collect_lons)
        } else {
            let Some(bounds) = wkb::bounds(wkb, collect_lons)? else {
                return Ok(None);
            };
            bounds
        };
        return Ok(Some((bounds, want_payload.then(|| wkb.to_vec()))));
    }
    let GeometryEncoding::GeoArrow { kind, .. } = state.info.encoding else {
        return Err(GeoError::UnsupportedEncoding(
            state.info.encoding.to_string(),
        ));
    };
    if let Some(covering) = covering.filter(|_| !want_payload) {
        return Ok(Some((
            bounds_from_covering(covering, row, collect_lons),
            None,
        )));
    }
    let Some(row) = geoarrow::scan_row(geom, kind, state.info.coordinate_dims, row, collect_lons)?
    else {
        return Ok(None);
    };
    Ok(Some((row.bounds, want_payload.then_some(row.wkb))))
}

fn bounds_from_covering(
    covering: &CoveringArrays,
    row: usize,
    collect_lons: bool,
) -> GeometryBounds {
    let mut bounds = GeometryBounds::new(collect_lons);
    bounds.min[0] = covering.xmin[row];
    bounds.min[1] = covering.ymin[row];
    bounds.max[0] = covering.xmax[row];
    bounds.max[1] = covering.ymax[row];
    if let (Some(zmin), Some(zmax)) = (&covering.zmin, &covering.zmax) {
        bounds.min[2] = zmin[row];
        bounds.max[2] = zmax[row];
        bounds.dims = CoordinateDims::Xyz;
    } else {
        bounds.dims = CoordinateDims::Xy;
    }
    bounds.any = true;
    if collect_lons {
        bounds.lon_values.push(bounds.min[0]);
        bounds.lon_values.push(bounds.max[0]);
    }
    bounds
}

#[derive(Debug)]
struct ScanEntry {
    bounds: GeometryBounds,
    feature: FeatureRef,
    payload: Option<Vec<u8>>,
}

enum ResolvedDims {
    D2,
    D3,
}

fn resolve_scan_dims(
    requested: IndexDimsRequest,
    detected: CoordinateDims,
    entries: &[ScanEntry],
) -> Result<ResolvedDims, GeoError> {
    let has_z = detected.has_z() || entries.iter().any(|entry| entry.bounds.dims.has_z());
    match requested {
        IndexDimsRequest::D2 if has_z => Err(GeoError::DimMismatch {
            expected: 2,
            found: 3,
        }),
        IndexDimsRequest::D2 => Ok(ResolvedDims::D2),
        IndexDimsRequest::D3 if !has_z => Err(GeoError::DimMismatch {
            expected: 3,
            found: 2,
        }),
        IndexDimsRequest::D3 => Ok(ResolvedDims::D3),
        IndexDimsRequest::Auto if has_z => Ok(ResolvedDims::D3),
        IndexDimsRequest::Auto => Ok(ResolvedDims::D2),
    }
}

fn split_2d(
    bounds: &GeometryBounds,
    policy: EnvelopePolicy,
    row: u64,
) -> Result<Vec<Box2D>, GeoError> {
    match policy {
        EnvelopePolicy::Planar => Ok(vec![Box2D::new(
            bounds.min[0],
            bounds.min[1],
            bounds.max[0],
            bounds.max[1],
        )]),
        EnvelopePolicy::Geographic { antimeridian } => {
            split_lon(bounds, antimeridian, row).map(|parts| {
                parts
                    .into_iter()
                    .map(|(xmin, xmax)| Box2D::new(xmin, bounds.min[1], xmax, bounds.max[1]))
                    .collect()
            })
        }
    }
}

fn split_3d(
    bounds: &GeometryBounds,
    policy: EnvelopePolicy,
    row: u64,
) -> Result<Vec<Box3D>, GeoError> {
    match policy {
        EnvelopePolicy::Planar => {
            let b = bounds.as_3d();
            Ok(vec![Box3D::new(b[0], b[1], b[2], b[3], b[4], b[5])])
        }
        EnvelopePolicy::Geographic { antimeridian } => {
            let zmin = if bounds.min[2].is_finite() {
                bounds.min[2]
            } else {
                0.0
            };
            let zmax = if bounds.max[2].is_finite() {
                bounds.max[2]
            } else {
                0.0
            };
            split_lon(bounds, antimeridian, row).map(|parts| {
                parts
                    .into_iter()
                    .map(|(xmin, xmax)| {
                        Box3D::new(xmin, bounds.min[1], zmin, xmax, bounds.max[1], zmax)
                    })
                    .collect()
            })
        }
    }
}

fn split_lon(
    bounds: &GeometryBounds,
    policy: AntimeridianPolicy,
    row: u64,
) -> Result<Vec<(f64, f64)>, GeoError> {
    let (start, end, crosses) = if bounds.min[0] > bounds.max[0] {
        (bounds.min[0], bounds.max[0], true)
    } else if bounds.lon_values.len() > 1 {
        minimal_lon_interval(&bounds.lon_values)
    } else {
        (bounds.min[0], bounds.max[0], false)
    };
    if !crosses {
        return Ok(vec![(start, end)]);
    }
    match policy {
        AntimeridianPolicy::Reject => Err(GeoError::Antimeridian { row }),
        AntimeridianPolicy::Split => Ok(vec![(start, 180.0), (-180.0, end)]),
        AntimeridianPolicy::ExpandToWorld => Ok(vec![(-180.0, 180.0)]),
    }
}

fn minimal_lon_interval(values: &[f64]) -> (f64, f64, bool) {
    let mut lons: Vec<f64> = values.iter().copied().map(normalize_lon).collect();
    lons.sort_by(|a, b| a.total_cmp(b));
    lons.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    if lons.len() <= 1 {
        let one = lons.first().copied().unwrap_or(0.0);
        return (one, one, false);
    }
    let mut max_gap = -1.0;
    let mut gap_index = 0usize;
    for i in 0..lons.len() {
        let next = if i + 1 == lons.len() {
            lons[0] + 360.0
        } else {
            lons[i + 1]
        };
        let gap = next - lons[i];
        if gap > max_gap {
            max_gap = gap;
            gap_index = i;
        }
    }
    let start = normalize_lon(lons[(gap_index + 1) % lons.len()]);
    let end = normalize_lon(lons[gap_index]);
    (start, end, start > end)
}

fn normalize_lon(value: f64) -> f64 {
    let mut v = value;
    while v < -180.0 {
        v += 360.0;
    }
    while v > 180.0 {
        v -= 360.0;
    }
    v
}

/// Request for [`GeoDataset::scan`](crate::dataset::GeoDataset::scan).
#[derive(Debug, Clone)]
pub struct ScanRequest {
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Requested envelope dimensionality.
    pub dims: IndexDimsRequest,
    /// Null/empty geometry policy.
    pub nulls: NullPolicy,
    /// Envelope interpretation policy.
    pub envelope: EnvelopePolicy,
    /// Payloads to emit for each scanned entry.
    pub payload: PayloadPlan,
}

impl Default for ScanRequest {
    fn default() -> Self {
        Self {
            selector: GeometrySelector::Default,
            dims: IndexDimsRequest::Auto,
            nulls: NullPolicy::Error,
            envelope: EnvelopePolicy::Planar,
            payload: PayloadPlan::None,
        }
    }
}

/// Result of scanning feature envelopes.
#[derive(Debug, Clone)]
pub enum GeometryScan {
    /// 2D scan result.
    D2(GeometryScan2D),
    /// 3D scan result.
    D3(GeometryScan3D),
}

/// 2D scan result.
#[derive(Debug, Clone)]
pub struct GeometryScan2D {
    /// One bounding box per index entry.
    pub boxes: Vec<Box2D>,
    /// Feature reference for each box.
    pub features: Vec<FeatureRef>,
    /// Optional payload for each box.
    pub payloads: Option<Vec<Vec<u8>>>,
    /// Profile of the scanned column.
    pub profile: GeometryProfile,
}

/// 3D scan result.
#[derive(Debug, Clone)]
pub struct GeometryScan3D {
    /// One bounding box per index entry.
    pub boxes: Vec<Box3D>,
    /// Feature reference for each box.
    pub features: Vec<FeatureRef>,
    /// Optional payload for each box.
    pub payloads: Option<Vec<Vec<u8>>>,
    /// Profile of the scanned column.
    pub profile: GeometryProfile,
}
