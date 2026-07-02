use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryBuilder, UInt32Array, new_empty_array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use arrow_json::LineDelimitedWriter;
use arrow_select::take::take;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::file::metadata::ParquetMetaData;
use serde::{Deserialize, Serialize};

use crate::discovery::ColumnState;
use crate::geoarrow;
use crate::payload::FeatureRef;
use crate::scan::{self, WkbCol};
use crate::wkb;
use crate::{GeoError, GeometryEncoding, GeometrySelector, PayloadPlan};

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowGroupSpan {
    index: u32,
    start: u64,
    len: u64,
}

impl RowGroupSpan {
    fn end(self) -> u64 {
        self.start + self.len
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedFeature {
    feature: FeatureRef,
    row_group: u32,
    row_in_group: u32,
    original_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SourceRow {
    row_number: u64,
    row_group: u32,
    row_in_group: u32,
}

pub(crate) struct SourceReadPlan {
    pub(crate) row_groups: Vec<usize>,
    pub(crate) selection: RowSelection,
}

pub(crate) fn row_group_spans(meta: &ParquetMetaData) -> Vec<RowGroupSpan> {
    let mut start = 0u64;
    meta.row_groups()
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let len = group.num_rows().max(0) as u64;
            let span = RowGroupSpan {
                index: index as u32,
                start,
                len,
            };
            start += len;
            span
        })
        .collect()
}

pub(crate) fn row_group_for_row(
    row_number: u64,
    spans: &[RowGroupSpan],
    cursor: &mut usize,
) -> Result<(u32, u32), GeoError> {
    while *cursor + 1 < spans.len() && row_number >= spans[*cursor].end() {
        *cursor += 1;
    }
    let Some(span) = spans.get(*cursor).copied() else {
        return Err(GeoError::FeatureRowOutOfBounds {
            row_number,
            num_rows: 0,
        });
    };
    if row_number >= span.start && row_number < span.end() {
        return Ok((span.index, (row_number - span.start) as u32));
    }
    Err(GeoError::FeatureRowOutOfBounds {
        row_number,
        num_rows: spans.last().map(|span| span.end()).unwrap_or(0),
    })
}

pub(crate) fn resolve_feature_refs(
    features: &[FeatureRef],
    spans: &[RowGroupSpan],
    num_rows: u64,
) -> Result<Vec<ResolvedFeature>, GeoError> {
    features
        .iter()
        .enumerate()
        .map(|(original_index, feature)| {
            if feature.row_number >= num_rows {
                return Err(GeoError::FeatureRowOutOfBounds {
                    row_number: feature.row_number,
                    num_rows,
                });
            }
            let (row_group, row_in_group) = position_for_row(feature.row_number, spans)?;
            if let (Some(given_group), Some(given_row)) = (feature.row_group, feature.row_in_group)
            {
                let Some(span) = spans.get(given_group as usize).copied() else {
                    return Err(GeoError::FeatureRowPositionMismatch {
                        row_number: feature.row_number,
                        row_group: given_group,
                        row_in_group: given_row,
                    });
                };
                if given_row as u64 >= span.len
                    || span.start + given_row as u64 != feature.row_number
                {
                    return Err(GeoError::FeatureRowPositionMismatch {
                        row_number: feature.row_number,
                        row_group: given_group,
                        row_in_group: given_row,
                    });
                }
            }
            let mut resolved = feature.clone();
            resolved.row_group = Some(row_group);
            resolved.row_in_group = Some(row_in_group);
            Ok(ResolvedFeature {
                feature: resolved,
                row_group,
                row_in_group,
                original_index,
            })
        })
        .collect()
}

fn position_for_row(row_number: u64, spans: &[RowGroupSpan]) -> Result<(u32, u32), GeoError> {
    let mut lo = 0usize;
    let mut hi = spans.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let span = spans[mid];
        if row_number < span.start {
            hi = mid;
        } else if row_number >= span.end() {
            lo = mid + 1;
        } else {
            return Ok((span.index, (row_number - span.start) as u32));
        }
    }
    Err(GeoError::FeatureRowOutOfBounds {
        row_number,
        num_rows: spans.last().map(|span| span.end()).unwrap_or(0),
    })
}

pub(crate) fn output_feature_order(
    resolved: &[ResolvedFeature],
    order: FeatureReadOrder,
    duplicates: DuplicateFeatureRows,
) -> Vec<FeatureRef> {
    let mut selected: Vec<&ResolvedFeature> = Vec::new();
    let mut seen = HashSet::new();
    for feature in resolved {
        if matches!(duplicates, DuplicateFeatureRows::DedupRows)
            && !seen.insert(feature.feature.row_number)
        {
            continue;
        }
        selected.push(feature);
    }
    match order {
        FeatureReadOrder::SourceOrder => selected.sort_by_key(|feature| {
            (
                feature.feature.row_number,
                feature.row_group,
                feature.row_in_group,
                feature.original_index,
            )
        }),
        FeatureReadOrder::RequestOrder => selected.sort_by_key(|feature| feature.original_index),
    }
    selected
        .into_iter()
        .map(|feature| feature.feature.clone())
        .collect()
}

pub(crate) fn unique_source_rows(features: &[FeatureRef]) -> Vec<SourceRow> {
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for feature in features {
        if seen.insert(feature.row_number) {
            rows.push(SourceRow {
                row_number: feature.row_number,
                row_group: feature.row_group.expect("resolved feature row group"),
                row_in_group: feature.row_in_group.expect("resolved feature row offset"),
            });
        }
    }
    rows.sort_by_key(|row| (row.row_number, row.row_group, row.row_in_group));
    rows
}

pub(crate) fn source_read_plan(
    rows: &[SourceRow],
    spans: &[RowGroupSpan],
) -> Result<SourceReadPlan, GeoError> {
    let mut row_groups: Vec<usize> = rows.iter().map(|row| row.row_group as usize).collect();
    row_groups.sort_unstable();
    row_groups.dedup();

    let mut selected_offsets = HashMap::new();
    let mut total_rows = 0u64;
    for &group in &row_groups {
        let span = spans
            .get(group)
            .ok_or_else(|| GeoError::Metadata(format!("row group {group} not found")))?;
        selected_offsets.insert(span.index, total_rows);
        total_rows += span.len;
    }

    let mut offsets = Vec::with_capacity(rows.len());
    for row in rows {
        let base = *selected_offsets.get(&row.row_group).ok_or_else(|| {
            GeoError::Metadata(format!("row group {} not selected", row.row_group))
        })?;
        offsets.push(
            usize::try_from(base + row.row_in_group as u64).map_err(|_| {
                GeoError::Metadata("row selection offset does not fit usize".to_string())
            })?,
        );
    }
    offsets.sort_unstable();
    offsets.dedup();

    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for offset in offsets {
        if let Some(last) = ranges.last_mut()
            && last.end == offset
        {
            last.end += 1;
            continue;
        }
        ranges.push(offset..offset + 1);
    }
    let total_rows = usize::try_from(total_rows)
        .map_err(|_| GeoError::Metadata("row selection length does not fit usize".to_string()))?;
    Ok(SourceReadPlan {
        row_groups,
        selection: RowSelection::from_consecutive_ranges(ranges.into_iter(), total_rows),
    })
}

pub(crate) fn property_root_indices(
    schema: &Schema,
    geometry_column: &str,
    properties: &PropertyProjection,
) -> Result<Vec<usize>, GeoError> {
    let names: Vec<_> = schema.fields().iter().map(|field| field.name()).collect();
    match properties {
        PropertyProjection::None => Ok(Vec::new()),
        PropertyProjection::AllNonGeometry => Ok(names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| (*name != geometry_column).then_some(idx))
            .collect()),
        PropertyProjection::Include(include) => include
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|candidate| *candidate == name)
                    .ok_or_else(|| GeoError::PropertyColumnNotFound(name.clone()))
            })
            .collect(),
        PropertyProjection::Exclude(exclude) => Ok(names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                (*name != geometry_column && !exclude.iter().any(|excluded| excluded == *name))
                    .then_some(idx)
            })
            .collect()),
    }
}

pub(crate) fn root_column_index(schema: &Schema, name: &str) -> Result<usize, GeoError> {
    schema
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .ok_or_else(|| GeoError::GeometryColumnNotFound(name.to_string()))
}

pub(crate) fn projected_schema(schema: &Schema, roots: &[usize]) -> Arc<Schema> {
    Arc::new(Schema::new(
        roots
            .iter()
            .map(|&idx| schema.field(idx).clone())
            .collect::<Vec<_>>(),
    ))
}

pub(crate) fn empty_read_batch(
    source_schema: &Schema,
    roots: &[usize],
    row_count: usize,
) -> Result<RecordBatch, GeoError> {
    let schema = projected_schema(source_schema, roots);
    let columns = schema
        .fields()
        .iter()
        .map(|field| new_empty_array(field.data_type()))
        .collect();
    record_batch_with_len(schema, columns, row_count)
}

pub(crate) fn take_indices_for_features(
    features: &[FeatureRef],
    read_rows: &[SourceRow],
) -> Result<Vec<u32>, GeoError> {
    let positions: HashMap<_, _> = read_rows
        .iter()
        .enumerate()
        .map(|(idx, row)| (row.row_number, idx as u32))
        .collect();
    features
        .iter()
        .map(|feature| {
            positions.get(&feature.row_number).copied().ok_or_else(|| {
                GeoError::Metadata(format!(
                    "feature row {} was not read from source",
                    feature.row_number
                ))
            })
        })
        .collect()
}

pub(crate) fn needs_take(indices: &[u32]) -> bool {
    indices
        .iter()
        .enumerate()
        .any(|(idx, &value)| value as usize != idx)
}

pub(crate) fn take_batch(batch: &RecordBatch, indices: &[u32]) -> Result<RecordBatch, GeoError> {
    if batch.num_columns() == 0 {
        return record_batch_with_len(batch.schema(), Vec::new(), indices.len());
    }
    let indices = UInt32Array::from(indices.to_vec());
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(batch.schema(), columns).map_err(GeoError::from)
}

pub(crate) fn finish_feature_batch(
    state: &ColumnState,
    source_schema: &Schema,
    property_roots: &[usize],
    geometry: GeometryReadMode,
    read_batch: RecordBatch,
    features: &[FeatureRef],
) -> Result<RecordBatch, GeoError> {
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for &root in property_roots {
        let field = source_schema.field(root).clone();
        let name = field.name().clone();
        let column = read_batch
            .column_by_name(&name)
            .ok_or_else(|| GeoError::Metadata(format!("projected column `{name}` was not read")))?;
        fields.push(field);
        columns.push(column.clone());
    }
    if matches!(geometry, GeometryReadMode::Wkb) {
        fields.push(Field::new("geometry_wkb", DataType::Binary, false));
        columns.push(geometry_wkb_array(state, &read_batch, features)?);
    }
    record_batch_with_len(Arc::new(Schema::new(fields)), columns, features.len())
}

fn record_batch_with_len(
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    row_count: usize,
) -> Result<RecordBatch, GeoError> {
    if columns.is_empty() {
        RecordBatch::try_new_with_options(
            schema,
            columns,
            &RecordBatchOptions::new().with_row_count(Some(row_count)),
        )
        .map_err(GeoError::from)
    } else {
        RecordBatch::try_new(schema, columns).map_err(GeoError::from)
    }
}

fn geometry_wkb_array(
    state: &ColumnState,
    batch: &RecordBatch,
    features: &[FeatureRef],
) -> Result<ArrayRef, GeoError> {
    let geom = batch
        .column_by_name(&state.info.name)
        .ok_or_else(|| GeoError::GeometryColumnNotFound(state.info.name.clone()))?
        .clone();
    let binary = scan::needs_binary(&state.info.encoding)
        .then(|| scan::binary_column(batch, &state.info.name))
        .transpose()?;
    let mut builder = BinaryBuilder::new();
    for (row, feature) in features.iter().enumerate() {
        let wkb = wkb_payload_one_row(state, &geom, binary.as_ref(), row)?.ok_or_else(|| {
            GeoError::NullGeometry {
                row: feature.row_number as usize,
            }
        })?;
        builder.append_value(wkb);
    }
    Ok(Arc::new(builder.finish()))
}

fn wkb_payload_one_row(
    state: &ColumnState,
    geom: &ArrayRef,
    binary: Option<&WkbCol<'_>>,
    row: usize,
) -> Result<Option<Vec<u8>>, GeoError> {
    if geom.is_null(row) {
        return Ok(None);
    }
    if let Some(binary) = binary {
        if binary.is_null(row) {
            return Ok(None);
        }
        return Ok(Some(binary.value(row).to_vec()));
    }
    let GeometryEncoding::GeoArrow { kind, .. } = state.info.encoding else {
        return Err(GeoError::UnsupportedEncoding(
            state.info.encoding.to_string(),
        ));
    };
    Ok(geoarrow::scan_row(geom, kind, state.info.coordinate_dims, row, false)?.map(|row| row.wkb))
}

pub(crate) fn projection_columns(
    batch: &RecordBatch,
    geometry_column: &str,
    payload: &PayloadPlan,
) -> Result<Vec<usize>, GeoError> {
    let properties = match payload {
        PayloadPlan::FeatureJson { properties } => properties,
        _ => return Ok(Vec::new()),
    };
    let fields = batch.schema().fields().clone();
    let names: Vec<_> = fields.iter().map(|field| field.name().clone()).collect();
    let selected: Vec<usize> = match properties {
        PropertyProjection::None => Vec::new(),
        PropertyProjection::AllNonGeometry => names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| (name != geometry_column).then_some(idx))
            .collect(),
        PropertyProjection::Include(include) => include
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| GeoError::PropertyColumnNotFound(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?,
        PropertyProjection::Exclude(exclude) => names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                (name != geometry_column && !exclude.iter().any(|excluded| excluded == name))
                    .then_some(idx)
            })
            .collect(),
    };
    Ok(selected)
}

pub(crate) fn feature_json_payload(
    feature: &FeatureRef,
    wkb: Option<&[u8]>,
    properties: Option<serde_json::Value>,
) -> Result<Vec<u8>, GeoError> {
    let geometry = wkb
        .map(wkb::geometry_json)
        .transpose()?
        .unwrap_or(serde_json::Value::Null);
    let properties = properties.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    let feature = serde_json::json!({
        "type": "Feature",
        "id": feature.feature_id.as_deref().unwrap_or(""),
        "feature_ref": feature,
        "geometry": geometry,
        "properties": properties,
    });
    serde_json::to_vec(&feature).map_err(|e| GeoError::Wkb(e.to_string()))
}

pub(crate) fn feature_json_property_rows(
    batch: &RecordBatch,
    properties: &PropertyProjection,
    property_columns: &[usize],
) -> Result<Option<Vec<serde_json::Value>>, GeoError> {
    if matches!(properties, PropertyProjection::None) || property_columns.is_empty() {
        return Ok(None);
    }
    let mut fields = Vec::with_capacity(property_columns.len());
    let mut arrays = Vec::with_capacity(property_columns.len());
    for &idx in property_columns {
        fields.push(batch.schema().field(idx).clone());
        arrays.push(batch.column(idx).clone());
    }
    let schema = Arc::new(Schema::new(fields));
    let projected = RecordBatch::try_new(schema, arrays)?;
    let mut buf = Vec::new();
    let mut writer = LineDelimitedWriter::new(&mut buf);
    writer.write(&projected)?;
    writer.finish()?;
    let trimmed = trim_ascii(buf.as_slice());
    if trimmed.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut rows = Vec::with_capacity(batch.num_rows());
    for line in trimmed.split(|&byte| byte == b'\n') {
        let value: serde_json::Value =
            serde_json::from_slice(trim_ascii(line)).map_err(|e| GeoError::Wkb(e.to_string()))?;
        rows.push(value);
    }
    if rows.len() != batch.num_rows() {
        return Err(GeoError::Metadata(format!(
            "property JSON writer emitted {} rows for a {}-row batch",
            rows.len(),
            batch.num_rows()
        )));
    }
    Ok(Some(rows))
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = bytes.len();
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &bytes[start..end]
}

/// Property projection for `FeatureJson` payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "columns", rename_all = "snake_case")]
pub enum PropertyProjection {
    /// Emit an empty properties object.
    None,
    /// Emit all non-geometry columns.
    AllNonGeometry,
    /// Emit only these property columns.
    Include(Vec<String>),
    /// Emit all non-geometry columns except these.
    Exclude(Vec<String>),
}

/// Geometry materialization mode for [`GeoDataset::read_features`](crate::dataset::GeoDataset::read_features).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryReadMode {
    /// Do not include geometry in the returned rows.
    Omit,
    /// Append a `geometry_wkb` binary column.
    Wkb,
}

/// Output order for [`GeoDataset::read_features`](crate::dataset::GeoDataset::read_features).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureReadOrder {
    /// Return rows sorted by source row number.
    SourceOrder,
    /// Return rows in the requested hit/feature order.
    RequestOrder,
}

/// Duplicate handling for feature refs that point at the same source row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DuplicateFeatureRows {
    /// Return each source row once, keeping the first feature ref for that row.
    DedupRows,
    /// Return one output row per requested feature ref, including split parts.
    KeepParts,
}

/// Request for [`GeoDataset::read_features`](crate::dataset::GeoDataset::read_features).
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{
///     open, Box2D, BuildRequest, FeatureReadRequest, GeoIndex,
/// };
///
/// let mut indexed = open(File::open("cities.parquet")?)?;
/// let GeoIndex::D2(index) = indexed.build(BuildRequest::default())? else {
///     panic!("expected a 2D index");
/// };
/// let hits = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
///
/// let mut source = open(File::open("cities.parquet")?)?;
/// let rows = source.read_features(FeatureReadRequest::from_features(hits))?;
/// println!("{} source rows", rows.features.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureReadRequest {
    /// Feature refs to read from the source Parquet file.
    pub features: Vec<FeatureRef>,
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Property columns to project into the returned batch.
    pub properties: PropertyProjection,
    /// Optional geometry materialization.
    pub geometry: GeometryReadMode,
    /// Output row order.
    pub order: FeatureReadOrder,
    /// Duplicate source-row handling.
    pub duplicates: DuplicateFeatureRows,
    /// Optional source fingerprint expected by the caller or artifact manifest.
    pub expected_source_fingerprint: Option<String>,
}

impl FeatureReadRequest {
    /// Create a default read request from feature refs.
    pub fn from_features(features: Vec<FeatureRef>) -> Self {
        Self {
            features,
            ..Self::default()
        }
    }

    /// Create a default read request from artifact hits.
    pub fn from_hits(hits: Vec<crate::GeoHit>) -> Self {
        Self {
            features: hits.into_iter().map(|hit| hit.feature).collect(),
            ..Self::default()
        }
    }
}

impl Default for FeatureReadRequest {
    fn default() -> Self {
        Self {
            features: Vec::new(),
            selector: GeometrySelector::Default,
            properties: PropertyProjection::AllNonGeometry,
            geometry: GeometryReadMode::Omit,
            order: FeatureReadOrder::SourceOrder,
            duplicates: DuplicateFeatureRows::DedupRows,
            expected_source_fingerprint: None,
        }
    }
}

/// Rows fetched from a Parquet source for feature refs.
///
/// `features[i]` describes the source feature represented by row `i` in
/// `batch`. The batch contains the requested property columns and, when
/// requested, a `geometry_wkb` binary column.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{open, FeatureReadRequest, FeatureRef};
///
/// let mut source = open(File::open("cities.parquet")?)?;
/// let rows = source.read_features(FeatureReadRequest::from_features(vec![
///     FeatureRef {
///         row_number: 42,
///         row_group: None,
///         row_in_group: None,
///         part: None,
///         feature_id: None,
///     },
/// ]))?;
/// assert_eq!(rows.features.len(), rows.batch.num_rows());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct FeatureRows {
    /// Feature refs aligned with returned batch rows.
    pub features: Vec<FeatureRef>,
    /// Projected source rows.
    pub batch: RecordBatch,
}
