//! Format-agnostic scan core: the per-entry intermediate representation and
//! the envelope/dimension/payload assembly shared by every source format.
//!
//! A source module (Parquet in `scan`, and any future format) produces one
//! [`ScanEntry`] per feature — bounds from a [`crate::wkb::GeometryBounds`]
//! accumulator, a [`FeatureRef`] back-reference, and optional payload bytes —
//! then hands the batch to [`assemble_scan`], which applies the envelope
//! policy (antimeridian handling), resolves 2D/3D, and packages a
//! [`GeometryScan`]. Nothing here touches arrow or parquet.

use packed_spatial_index::{Box2D, Box3D};
use serde::{Deserialize, Serialize};

use crate::payload::{FeatureRef, PropertyProjection};
use crate::wkb::GeometryBounds;
use crate::{
    AntimeridianPolicy, CoordinateDims, EnvelopePolicy, GeoError, GeometryProfile,
    GeometrySelector, NullPolicy, PayloadPlan,
};

/// Requested index dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexDimsRequest {
    /// Infer dimensions.
    Auto,
    /// Force 2D envelopes.
    D2,
    /// Force 3D envelopes.
    D3,
}

/// One scanned feature: envelope bounds, source back-reference, and optional
/// payload bytes. The format-specific scanners produce these; everything
/// downstream is shared.
#[derive(Debug)]
pub(crate) struct ScanEntry {
    pub(crate) bounds: GeometryBounds,
    pub(crate) feature: FeatureRef,
    pub(crate) payload: Option<Vec<u8>>,
}

pub(crate) enum ResolvedDims {
    D2,
    D3,
}

pub(crate) fn resolve_scan_dims(
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

pub(crate) fn split_2d(
    bounds: &GeometryBounds,
    policy: EnvelopePolicy,
    row: u64,
) -> Result<Vec<Box2D>, GeoError> {
    match policy {
        EnvelopePolicy::Planar => {
            reject_wrapped_covering_under_planar(bounds, row)?;
            Ok(vec![Box2D::new(
                bounds.min[0],
                bounds.min[1],
                bounds.max[0],
                bounds.max[1],
            )])
        }
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

pub(crate) fn split_3d(
    bounds: &GeometryBounds,
    policy: EnvelopePolicy,
    row: u64,
) -> Result<Vec<Box3D>, GeoError> {
    match policy {
        EnvelopePolicy::Planar => {
            reject_wrapped_covering_under_planar(bounds, row)?;
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

fn reject_wrapped_covering_under_planar(bounds: &GeometryBounds, row: u64) -> Result<(), GeoError> {
    if bounds.from_covering && bounds.min[0] > bounds.max[0] {
        return Err(GeoError::Antimeridian { row });
    }
    Ok(())
}

fn split_lon(
    bounds: &GeometryBounds,
    policy: AntimeridianPolicy,
    row: u64,
) -> Result<Vec<(f64, f64)>, GeoError> {
    let (start, end, crosses) = if bounds.min[0] > bounds.max[0] {
        (bounds.min[0], bounds.max[0], true)
    } else if bounds.from_covering {
        (bounds.min[0], bounds.max[0], false)
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

/// Geometry materialization mode for source read-back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryReadMode {
    /// Do not include WKB geometry in the returned rows.
    Omit,
    /// Materialize source geometry as WKB.
    Wkb,
}

/// Output order for source read-back.
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

/// Request for source read-back by [`FeatureRef`].
///
/// The same request type is accepted by Parquet, GeoJSON, and FlatGeobuf
/// sources. Parquet returns `FeatureRows`, while non-Arrow sources return
/// [`FeatureRecord`] values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureReadRequest {
    /// Feature refs to read from the source dataset.
    pub features: Vec<FeatureRef>,
    /// Geometry column selector.
    pub selector: GeometrySelector,
    /// Properties to project into the returned rows.
    pub properties: PropertyProjection,
    /// Optional WKB geometry materialization.
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

/// A source feature materialized without Arrow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureRecord {
    /// Feature ref aligned with this output record.
    pub feature: FeatureRef,
    /// WKB geometry when requested by [`FeatureReadRequest::geometry`].
    pub geometry_wkb: Option<Vec<u8>>,
    /// GeoJSON geometry, when the source row has a non-null geometry.
    pub geometry_json: Option<serde_json::Value>,
    /// Projected source properties as a JSON object.
    pub properties: serde_json::Value,
}

pub(crate) fn ordered_feature_refs(
    features: &[FeatureRef],
    num_rows: Option<u64>,
    order: FeatureReadOrder,
    duplicates: DuplicateFeatureRows,
) -> Result<Vec<FeatureRef>, GeoError> {
    let mut selected: Vec<(usize, FeatureRef)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (original_index, feature) in features.iter().enumerate() {
        if let Some(num_rows) = num_rows
            && feature.row_number >= num_rows
        {
            return Err(GeoError::FeatureRowOutOfBounds {
                row_number: feature.row_number,
                num_rows,
            });
        }
        if matches!(duplicates, DuplicateFeatureRows::DedupRows) && !seen.insert(feature.row_number)
        {
            continue;
        }
        selected.push((original_index, feature.clone()));
    }
    match order {
        FeatureReadOrder::SourceOrder => {
            selected.sort_by_key(|(original_index, feature)| {
                (feature.row_number, feature.part, *original_index)
            });
        }
        FeatureReadOrder::RequestOrder => {
            selected.sort_by_key(|(original_index, _)| *original_index);
        }
    }
    Ok(selected.into_iter().map(|(_, feature)| feature).collect())
}

/// Request for a source scan such as `GeoDataset::scan`.
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

pub(crate) trait PayloadVec {
    fn payload_payloads(&self) -> Option<Vec<Vec<u8>>>;
}

impl PayloadVec for ScanRequest {
    fn payload_payloads(&self) -> Option<Vec<Vec<u8>>> {
        (!matches!(self.payload, PayloadPlan::None)).then(Vec::new)
    }
}

/// Assemble scanned entries into a [`GeometryScan`]: stamp the detected
/// dimensions into the profile, resolve 2D vs 3D, apply the envelope policy
/// (splitting antimeridian-crossing features into parts), and pair payload
/// bytes with each resulting index entry.
pub(crate) fn assemble_scan(
    entries: Vec<ScanEntry>,
    req: &ScanRequest,
    mut profile: GeometryProfile,
    detected_dims: CoordinateDims,
) -> Result<GeometryScan, GeoError> {
    let dims = resolve_scan_dims(req.dims, detected_dims, &entries)?;
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
                payload: req.payload.clone(),
                nulls: req.nulls,
                envelope: req.envelope,
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
                payload: req.payload.clone(),
                nulls: req.nulls,
                envelope: req.envelope,
            }))
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
///
/// Obtain one from a source scan such as `GeoDataset::scan`: it cannot be
/// constructed outside this crate, and the payload/provenance fields are
/// read-only through accessors ([`payload`](Self::payload),
/// [`payloads`](Self::payloads), [`nulls`](Self::nulls),
/// [`envelope`](Self::envelope)). That keeps the recorded payload plan and the
/// payload bytes paired as the scan produced them, so
/// [`GeoArtifact::from_scan`](crate::GeoArtifact::from_scan) can trust the
/// pairing when writing the manifest — external code can neither forge the
/// pair at construction nor mutate it afterward.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GeometryScan2D {
    /// One bounding box per index entry.
    pub boxes: Vec<Box2D>,
    /// Feature reference for each box.
    pub features: Vec<FeatureRef>,
    /// Profile of the scanned column.
    pub profile: GeometryProfile,
    /// Optional payload bytes for each box. Read via [`payloads`](Self::payloads).
    pub(crate) payloads: Option<Vec<Vec<u8>>>,
    /// Payload plan that produced `payloads`. Read via [`payload`](Self::payload).
    pub(crate) payload: PayloadPlan,
    /// Null/empty policy applied during the scan. Read via [`nulls`](Self::nulls).
    pub(crate) nulls: NullPolicy,
    /// Envelope policy applied during the scan. Read via [`envelope`](Self::envelope).
    pub(crate) envelope: EnvelopePolicy,
}

impl GeometryScan2D {
    /// The payload bytes recorded for each index entry, if any.
    pub fn payloads(&self) -> Option<&[Vec<u8>]> {
        self.payloads.as_deref()
    }

    /// The payload plan that produced [`payloads`](Self::payloads).
    pub fn payload(&self) -> &PayloadPlan {
        &self.payload
    }

    /// The null/empty policy applied during the scan.
    pub fn nulls(&self) -> NullPolicy {
        self.nulls
    }

    /// The envelope policy applied during the scan.
    pub fn envelope(&self) -> EnvelopePolicy {
        self.envelope
    }
}

/// 3D scan result.
///
/// Obtain one from a source scan such as `GeoDataset::scan`: it cannot be
/// constructed outside this crate, and the payload/provenance fields are
/// read-only through accessors ([`payload`](Self::payload),
/// [`payloads`](Self::payloads), [`nulls`](Self::nulls),
/// [`envelope`](Self::envelope)). That keeps the recorded payload plan and the
/// payload bytes paired as the scan produced them, so
/// [`GeoArtifact::from_scan`](crate::GeoArtifact::from_scan) can trust the
/// pairing when writing the manifest — external code can neither forge the
/// pair at construction nor mutate it afterward.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GeometryScan3D {
    /// One bounding box per index entry.
    pub boxes: Vec<Box3D>,
    /// Feature reference for each box.
    pub features: Vec<FeatureRef>,
    /// Profile of the scanned column.
    pub profile: GeometryProfile,
    /// Optional payload bytes for each box. Read via [`payloads`](Self::payloads).
    pub(crate) payloads: Option<Vec<Vec<u8>>>,
    /// Payload plan that produced `payloads`. Read via [`payload`](Self::payload).
    pub(crate) payload: PayloadPlan,
    /// Null/empty policy applied during the scan. Read via [`nulls`](Self::nulls).
    pub(crate) nulls: NullPolicy,
    /// Envelope policy applied during the scan. Read via [`envelope`](Self::envelope).
    pub(crate) envelope: EnvelopePolicy,
}

impl GeometryScan3D {
    /// The payload bytes recorded for each index entry, if any.
    pub fn payloads(&self) -> Option<&[Vec<u8>]> {
        self.payloads.as_deref()
    }

    /// The payload plan that produced [`payloads`](Self::payloads).
    pub fn payload(&self) -> &PayloadPlan {
        &self.payload
    }

    /// The null/empty policy applied during the scan.
    pub fn nulls(&self) -> NullPolicy {
        self.nulls
    }

    /// The envelope policy applied during the scan.
    pub fn envelope(&self) -> EnvelopePolicy {
        self.envelope
    }
}
