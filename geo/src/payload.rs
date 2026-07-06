#[cfg(feature = "_source")]
use std::collections::HashSet;

#[cfg(feature = "parquet")]
use parquet::file::metadata::ParquetMetaData;
use serde::{Deserialize, Serialize};
#[cfg(feature = "_source")]
use serde_json::value::RawValue;

#[cfg(feature = "_source")]
use crate::GeoError;

/// Content type used for [`PayloadPlan::RowRef`] payload sections.
pub const FEATURE_REF_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.feature-ref";
/// Content type used for [`PayloadPlan::RowWkb`] payload sections.
pub const FEATURE_WKB_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.feature-wkb";
/// Content type used for [`PayloadPlan::FeatureJson`] payload sections.
pub const FEATURE_JSON_CONTENT_TYPE: &str = "application/geo+json";
/// Byte length of the fixed-width [`FeatureRef`] payload record.
pub const FEATURE_REF_RECORD_LEN: usize = 24;

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

#[cfg(feature = "_source")]
pub(crate) fn encode_feature_ref(feature: &FeatureRef) -> Vec<u8> {
    let mut out = Vec::with_capacity(FEATURE_REF_RECORD_LEN);
    out.extend_from_slice(&feature.row_number.to_le_bytes());
    out.extend_from_slice(&feature.row_group.unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(&feature.row_in_group.unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(&feature.part.unwrap_or(u16::MAX).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

#[cfg(feature = "_source")]
pub(crate) fn encode_feature_wkb(feature: &FeatureRef, wkb: &[u8]) -> Vec<u8> {
    let mut out = encode_feature_ref(feature);
    out.extend_from_slice(wkb);
    out
}

/// Stamp a split part number into an already-encoded payload.
///
/// Scan encodes payloads once per source feature, before envelope splitting
/// duplicates entries; each duplicated payload must be re-stamped so the
/// decoded [`FeatureRef::part`] matches the entry it describes. Empty
/// payloads are left untouched, mirroring the scan path's tolerance for
/// missing payload bytes.
#[cfg(feature = "_source")]
pub(crate) fn stamp_payload_part(
    plan: &PayloadPlan,
    payload: &mut Vec<u8>,
    part: u16,
) -> Result<(), GeoError> {
    if payload.is_empty() {
        return Ok(());
    }
    match plan {
        PayloadPlan::None => Ok(()),
        PayloadPlan::RowRef | PayloadPlan::RowWkb => {
            if payload.len() < FEATURE_REF_RECORD_LEN {
                return Err(GeoError::PayloadDecode(format!(
                    "payload of {} bytes is too short for a feature-ref record",
                    payload.len()
                )));
            }
            payload[16..18].copy_from_slice(&part.to_le_bytes());
            Ok(())
        }
        PayloadPlan::FeatureJson { .. } => {
            let mut value: serde_json::Value = serde_json::from_slice(payload)
                .map_err(|e| GeoError::PayloadDecode(e.to_string()))?;
            let Some(feature_ref) = value.get_mut("feature_ref") else {
                return Err(GeoError::PayloadDecode(
                    "FeatureJson payload is missing the feature_ref member".to_string(),
                ));
            };
            feature_ref["part"] = serde_json::Value::from(part);
            *payload =
                serde_json::to_vec(&value).map_err(|e| GeoError::PayloadDecode(e.to_string()))?;
            Ok(())
        }
    }
}

/// Serialize a GeoJSON `Feature` payload from already-materialized geometry
/// and properties JSON. Format-specific callers supply the geometry however
/// they hold it — decoded from WKB (Parquet) or taken straight from the
/// source (GeoJSON) — so this stays free of arrow and WKB concerns.
#[cfg(feature = "_source")]
#[allow(dead_code)]
pub(crate) fn feature_json_from_parts(
    feature: &FeatureRef,
    geometry: serde_json::Value,
    properties: Option<serde_json::Value>,
) -> Result<Vec<u8>, GeoError> {
    let properties =
        properties.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    let feature = serde_json::json!({
        "type": "Feature",
        "id": feature.feature_id.as_deref().unwrap_or(""),
        "feature_ref": feature,
        "geometry": geometry,
        "properties": properties,
    });
    serde_json::to_vec(&feature).map_err(|e| GeoError::Wkb(e.to_string()))
}

/// Serialize a GeoJSON `Feature` payload while borrowing an already-valid raw
/// GeoJSON geometry string.
#[cfg(feature = "_source")]
pub(crate) fn feature_json_from_raw_parts(
    feature: &FeatureRef,
    geometry: &RawValue,
    properties: Option<serde_json::Value>,
) -> Result<Vec<u8>, GeoError> {
    #[derive(Serialize)]
    struct RawFeatureJson<'a> {
        #[serde(rename = "type")]
        kind: &'static str,
        id: &'a str,
        feature_ref: &'a FeatureRef,
        geometry: &'a RawValue,
        properties: serde_json::Value,
    }

    let payload = RawFeatureJson {
        kind: "Feature",
        id: feature.feature_id.as_deref().unwrap_or(""),
        feature_ref: feature,
        geometry,
        properties: properties.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
    };
    serde_json::to_vec(&payload).map_err(|e| GeoError::Wkb(e.to_string()))
}

/// Decode a fixed-width [`FeatureRef`] payload.
///
/// Returns `None` if the payload is shorter than [`FEATURE_REF_RECORD_LEN`].
pub fn decode_feature_ref_payload(payload: &[u8]) -> Option<FeatureRef> {
    if payload.len() < FEATURE_REF_RECORD_LEN {
        return None;
    }
    let row_number = u64::from_le_bytes(payload[0..8].try_into().ok()?);
    let row_group = decode_u32_option(payload[8..12].try_into().ok()?);
    let row_in_group = decode_u32_option(payload[12..16].try_into().ok()?);
    let part = decode_u16_option(payload[16..18].try_into().ok()?);
    Some(FeatureRef {
        row_number,
        row_group,
        row_in_group,
        part,
        feature_id: None,
    })
}

/// Decode a [`FeatureRef`] followed by WKB bytes.
///
/// This is the payload shape generated by [`PayloadPlan::RowWkb`]. Returns
/// `None` when the fixed feature-ref prefix is truncated.
pub fn decode_feature_wkb_payload(payload: &[u8]) -> Option<(FeatureRef, &[u8])> {
    let feature = decode_feature_ref_payload(payload)?;
    Some((feature, &payload[FEATURE_REF_RECORD_LEN..]))
}

fn decode_u32_option(bytes: [u8; 4]) -> Option<u32> {
    match u32::from_le_bytes(bytes) {
        u32::MAX => None,
        value => Some(value),
    }
}

fn decode_u16_option(bytes: [u8; 2]) -> Option<u16> {
    match u16::from_le_bytes(bytes) {
        u16::MAX => None,
        value => Some(value),
    }
}

#[cfg(feature = "_source")]
pub(crate) fn unique_feature_count(features: &[FeatureRef]) -> usize {
    features
        .iter()
        .map(|feature| feature.row_number)
        .collect::<HashSet<_>>()
        .len()
}

#[cfg(feature = "_source")]
pub(crate) fn entries_may_duplicate_rows(features: &[FeatureRef]) -> bool {
    let mut seen = HashSet::new();
    features
        .iter()
        .any(|feature| !seen.insert(feature.row_number))
}

#[cfg(feature = "parquet")]
pub(crate) fn source_fingerprint(meta: &ParquetMetaData) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash = fnv(hash, &meta.file_metadata().num_rows().to_le_bytes());
    for col in meta.file_metadata().schema_descr().columns() {
        hash = fnv(hash, col.path().string().as_bytes());
        hash = fnv(hash, format!("{:?}", col.logical_type_ref()).as_bytes());
    }
    format!("fnv64:{hash:016x}")
}

#[cfg(feature = "_source")]
pub(crate) fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// Stable reference back to a source feature.
///
/// # Example
///
/// ```rust
/// use packed_spatial_index_geo::FeatureRef;
///
/// let feature = FeatureRef {
///     row_number: 42,
///     row_group: None,
///     row_in_group: None,
///     part: Some(1),
///     feature_id: None,
/// };
/// assert_eq!(feature.row_number, 42);
/// assert_eq!(feature.part, Some(1));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureRef {
    /// Absolute source row number.
    pub row_number: u64,
    /// Source row group when known.
    pub row_group: Option<u32>,
    /// Row offset within the row group when known.
    pub row_in_group: Option<u32>,
    /// Split part for duplicated index entries.
    pub part: Option<u16>,
    /// Optional feature identifier.
    pub feature_id: Option<String>,
}

impl FeatureRef {
    /// Create a feature ref from an absolute source row number.
    pub fn row_number(row_number: u64) -> Self {
        Self {
            row_number,
            row_group: None,
            row_in_group: None,
            part: None,
            feature_id: None,
        }
    }

    #[cfg(feature = "parquet")]
    pub(crate) fn row_in_group(row_number: u64, row_group: u32, row_in_group: u32) -> Self {
        Self {
            row_number,
            row_group: Some(row_group),
            row_in_group: Some(row_in_group),
            part: None,
            feature_id: None,
        }
    }

    /// Whether both refs point at the same source feature.
    ///
    /// `part` is ignored: split index entries (for example antimeridian
    /// parts) of one feature compare equal.
    pub fn same_feature(&self, other: &FeatureRef) -> bool {
        self.cmp_feature(other) == std::cmp::Ordering::Equal
    }

    /// Order by source feature identity: `row_number`, `row_group`,
    /// `row_in_group`, `feature_id`. `part` is ignored.
    pub fn cmp_feature(&self, other: &FeatureRef) -> std::cmp::Ordering {
        self.row_number
            .cmp(&other.row_number)
            .then_with(|| self.row_group.cmp(&other.row_group))
            .then_with(|| self.row_in_group.cmp(&other.row_in_group))
            .then_with(|| self.feature_id.as_deref().cmp(&other.feature_id.as_deref()))
    }

    /// [`cmp_feature`](Self::cmp_feature), then `part` — the deterministic
    /// entry-level order.
    pub fn cmp_entry(&self, other: &FeatureRef) -> std::cmp::Ordering {
        self.cmp_feature(other)
            .then_with(|| self.part.cmp(&other.part))
    }
}

/// Payload to attach to converted artifact entries or scan results.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{open_geoparquet, ConvertRequest, PayloadPlan, PropertyProjection};
///
/// let mut dataset = open_geoparquet(File::open("cities.parquet")?)?;
/// let bytes = dataset.convert(ConvertRequest {
///     payload: PayloadPlan::FeatureJson {
///         properties: PropertyProjection::AllNonGeometry,
///     },
///     ..ConvertRequest::default()
/// })?;
/// println!("{} bytes", bytes.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PayloadPlan {
    /// Emit no payloads.
    None,
    /// Emit only fixed-width `FeatureRef` records.
    RowRef,
    /// Emit fixed-width `FeatureRef` records followed by WKB bytes.
    RowWkb,
    /// Emit GeoJSON Feature bytes with projected properties.
    FeatureJson {
        /// Property projection.
        properties: PropertyProjection,
    },
}
