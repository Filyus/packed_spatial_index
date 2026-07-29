#[cfg(feature = "_source")]
use std::collections::HashSet;

#[cfg(feature = "parquet")]
use parquet::file::metadata::ParquetMetaData;
use serde::{Deserialize, Serialize};
#[cfg(feature = "_source")]
use serde_json::value::RawValue;

#[cfg(feature = "_source")]
use crate::GeoError;

/// Project-defined vendor media type used for [`PayloadPlan::RowRef`]
/// payload sections. It is not currently registered with IANA.
pub const FEATURE_REF_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.feature-ref";
/// Project-defined vendor media type used for [`PayloadPlan::RowWkb`]
/// payload sections. It is not currently registered with IANA.
pub const FEATURE_WKB_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.feature-wkb";
/// Project-defined, unregistered vendor media type used for feature-ref-prefixed
/// [`PayloadPlan::FeatureJson`] payload sections.
///
/// This identifies the complete binary `FeatureRef` + GeoJSON record, not a
/// `.psindex` file or a standalone JSON representation. [`feature_json_body`]
/// returns the embedded GeoJSON bytes. It intentionally has no `+json` suffix,
/// because a generic JSON parser cannot consume the binary prefix.
pub const FEATURE_JSON_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.feature-json";
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

#[cfg(feature = "_source")]
fn encode_feature_json(feature: &FeatureRef, json: &[u8]) -> Vec<u8> {
    let mut out = encode_feature_ref(feature);
    out.extend_from_slice(json);
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
            let stamped_prefix = has_feature_ref_prefix(payload);
            if stamped_prefix {
                payload[16..18].copy_from_slice(&part.to_le_bytes());
            }
            let body = feature_json_body(payload);
            let mut value: serde_json::Value =
                serde_json::from_slice(body).map_err(|e| GeoError::PayloadDecode(e.to_string()))?;
            let Some(feature_ref) = value.get_mut("feature_ref") else {
                // The JSON member is written only where it carries a source id
                // the fixed record has no room for, so its absence is normal —
                // as long as the record that replaced it took the stamp.
                if stamped_prefix {
                    return Ok(());
                }
                return Err(GeoError::PayloadDecode(
                    "FeatureJson payload is missing the feature_ref member".to_string(),
                ));
            };
            // Indexing a `Value` by name panics unless it is an object or
            // null. Every producer in this crate writes an object, but this is
            // the one place a malformed payload would abort the process rather
            // than report.
            let Some(fields) = feature_ref.as_object_mut() else {
                return Err(GeoError::PayloadDecode(format!(
                    "FeatureJson payload has a non-object feature_ref member: {feature_ref}"
                )));
            };
            fields.insert("part".to_string(), serde_json::Value::from(part));
            let json =
                serde_json::to_vec(&value).map_err(|e| GeoError::PayloadDecode(e.to_string()))?;
            *payload = encode_feature_json_prefix_compatible(payload, &json);
            Ok(())
        }
    }
}

/// Serialize a GeoJSON `Feature` payload from already-materialized geometry
/// and properties JSON. Format-specific callers supply the geometry however
/// they hold it — decoded from WKB (Parquet) or taken straight from the
/// source (GeoJSON) — so this stays free of arrow and WKB concerns.
#[cfg(feature = "parquet")]
pub(crate) fn feature_json_from_parts(
    feature: &FeatureRef,
    geometry: serde_json::Value,
    properties: Option<serde_json::Value>,
) -> Result<Vec<u8>, GeoError> {
    let properties =
        properties.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    let mut value = serde_json::Map::new();
    value.insert("type".to_string(), serde_json::json!("Feature"));
    // Both members hinge on the same thing, and only GeoJSON sources ever
    // supply one: RFC 7946 makes `id` optional, and writing `""` would assert
    // an identity the feature does not have; `feature_ref` re-encodes the
    // fixed record this payload already carries, so the id is the only part of
    // it the record has no room for. Without an id it is pure duplication.
    if let Some(id) = &feature.feature_id {
        value.insert("id".to_string(), serde_json::json!(id));
        value.insert("feature_ref".to_string(), serde_json::json!(feature));
    }
    value.insert("geometry".to_string(), geometry);
    value.insert("properties".to_string(), properties);
    let json = serde_json::to_vec(&value).map_err(|e| GeoError::Wkb(e.to_string()))?;
    Ok(encode_feature_json(feature, &json))
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
        // Both are written only for a feature that has a source id; see
        // `feature_json_from_parts` for why they travel together.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        feature_ref: Option<&'a FeatureRef>,
        geometry: &'a RawValue,
        properties: serde_json::Value,
    }

    let payload = RawFeatureJson {
        kind: "Feature",
        id: feature.feature_id.as_deref(),
        feature_ref: feature.feature_id.is_some().then_some(feature),
        geometry,
        properties: properties.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
    };
    let json = serde_json::to_vec(&payload).map_err(|e| GeoError::Wkb(e.to_string()))?;
    Ok(encode_feature_json(feature, &json))
}

/// Return the JSON body of a FeatureJson payload. New artifacts prefix the
/// JSON with a fixed-width FeatureRef record so header scans can page without
/// reading bodies; older artifacts contain only JSON and remain readable.
pub fn feature_json_body(payload: &[u8]) -> &[u8] {
    if has_feature_ref_prefix(payload) {
        &payload[FEATURE_REF_RECORD_LEN..]
    } else {
        payload
    }
}

#[cfg(feature = "_source")]
fn encode_feature_json_prefix_compatible(old_payload: &[u8], json: &[u8]) -> Vec<u8> {
    if has_feature_ref_prefix(old_payload) {
        let mut out = old_payload[..FEATURE_REF_RECORD_LEN].to_vec();
        out.extend_from_slice(json);
        out
    } else {
        json.to_vec()
    }
}

fn has_feature_ref_prefix(payload: &[u8]) -> bool {
    decode_feature_ref_payload(payload).is_some()
}

/// Decode a fixed-width [`FeatureRef`] payload.
///
/// Returns `None` if the payload is shorter than [`FEATURE_REF_RECORD_LEN`] or
/// the reserved bytes do not match the fixed-width record format.
pub fn decode_feature_ref_payload(payload: &[u8]) -> Option<FeatureRef> {
    if payload.len() < FEATURE_REF_RECORD_LEN {
        return None;
    }
    if payload[18..FEATURE_REF_RECORD_LEN].iter().any(|&b| b != 0) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_json_body_detects_prefix_structurally() {
        let json = br#"{"type":"Feature"}"#;
        assert_eq!(feature_json_body(json), json);

        let mut payload = Vec::new();
        payload.extend_from_slice(&123u64.to_le_bytes());
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        payload.extend_from_slice(&u16::MAX.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(json);

        assert_eq!(payload[0], b'{');
        assert_eq!(
            decode_feature_ref_payload(&payload).unwrap().row_number,
            123
        );
        assert_eq!(feature_json_body(&payload), json);
    }

    /// No producer in this crate writes a non-object `feature_ref`, but
    /// indexing a `Value` by name panics on one, and aborting the process is
    /// not how this crate reports a malformed payload.
    #[cfg(feature = "_source")]
    #[test]
    fn stamping_a_part_rejects_a_non_object_feature_ref() {
        let plan = PayloadPlan::FeatureJson {
            properties: PropertyProjection::None,
        };
        for body in [
            br#"{"type":"Feature","feature_ref":7}"#.as_slice(),
            br#"{"type":"Feature","feature_ref":"row-7"}"#.as_slice(),
            br#"{"type":"Feature","feature_ref":[]}"#.as_slice(),
        ] {
            let mut payload = body.to_vec();
            let err = stamp_payload_part(&plan, &mut payload, 1).unwrap_err();
            assert!(
                matches!(&err, GeoError::PayloadDecode(message) if message.contains("feature_ref")),
                "{err}"
            );
        }

        // `null` is the one non-object serde_json would have accepted, by
        // promoting it to an object. A payload whose feature ref is null is
        // just as malformed as the others, so it is refused too.
        let mut payload = br#"{"type":"Feature","feature_ref":null}"#.to_vec();
        assert!(stamp_payload_part(&plan, &mut payload, 1).is_err());
    }

    /// Only GeoJSON sources supply a feature id; Parquet and FlatGeobuf never
    /// do. An `"id": ""` written for those would be indistinguishable from a
    /// source id that really is the empty string.
    #[cfg(feature = "_source")]
    #[test]
    fn a_feature_without_a_source_id_carries_no_id_member() {
        let geometry =
            RawValue::from_string(r#"{"type":"Point","coordinates":[1.0,2.0]}"#.to_string())
                .unwrap();
        let stored = |feature: &FeatureRef| -> serde_json::Value {
            let payload = feature_json_from_raw_parts(feature, &geometry, None).unwrap();
            serde_json::from_slice(feature_json_body(&payload)).unwrap()
        };

        let anonymous = stored(&FeatureRef::row_number(7));
        assert!(anonymous.get("id").is_none(), "{anonymous}");

        let mut named = FeatureRef::row_number(7);
        named.feature_id = Some("west".to_string());
        assert_eq!(stored(&named)["id"], "west");
    }

    /// The JSON `feature_ref` re-encodes the fixed record the payload already
    /// carries in front of it. The id is the only field that record has no
    /// room for, so without one the member is pure duplication.
    #[cfg(feature = "_source")]
    #[test]
    fn the_json_feature_ref_is_written_only_to_carry_an_id() {
        let geometry =
            RawValue::from_string(r#"{"type":"Point","coordinates":[1.0,2.0]}"#.to_string())
                .unwrap();
        let stored = |feature: &FeatureRef| -> (Vec<u8>, serde_json::Value) {
            let payload = feature_json_from_raw_parts(feature, &geometry, None).unwrap();
            let body = serde_json::from_slice(feature_json_body(&payload)).unwrap();
            (payload, body)
        };

        let (payload, anonymous) = stored(&FeatureRef::row_number(7));
        assert!(anonymous.get("feature_ref").is_none(), "{anonymous}");
        // Nothing is lost: the fixed record in front of the JSON says the same.
        assert_eq!(decode_feature_ref_payload(&payload).unwrap().row_number, 7);

        let mut named = FeatureRef::row_number(7);
        named.feature_id = Some("west".to_string());
        let (_, named) = stored(&named);
        assert_eq!(named["feature_ref"]["feature_id"], "west");
    }

    /// Stamping a split part must still work on a payload whose only feature
    /// ref is the fixed record — that is now the common shape, not a defect.
    #[cfg(feature = "_source")]
    #[test]
    fn stamping_a_part_uses_the_fixed_record_when_the_member_is_gone() {
        let plan = PayloadPlan::FeatureJson {
            properties: PropertyProjection::None,
        };
        let geometry =
            RawValue::from_string(r#"{"type":"Point","coordinates":[1.0,2.0]}"#.to_string())
                .unwrap();
        let mut payload =
            feature_json_from_raw_parts(&FeatureRef::row_number(7), &geometry, None).unwrap();
        let before = payload.clone();

        stamp_payload_part(&plan, &mut payload, 3).unwrap();

        assert_eq!(decode_feature_ref_payload(&payload).unwrap().part, Some(3));
        // Only the record changed; the JSON body was left byte-for-byte alone.
        assert_eq!(feature_json_body(&payload), feature_json_body(&before));

        // A payload with neither a record nor a member is still a defect.
        let mut orphan = br#"{"type":"Feature"}"#.to_vec();
        assert!(stamp_payload_part(&plan, &mut orphan, 1).is_err());
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn the_parquet_serializer_omits_the_id_the_same_way() {
        let geometry = serde_json::json!({"type": "Point", "coordinates": [1.0, 2.0]});
        let stored = |feature: &FeatureRef| -> serde_json::Value {
            let payload = feature_json_from_parts(feature, geometry.clone(), None).unwrap();
            serde_json::from_slice(feature_json_body(&payload)).unwrap()
        };

        let anonymous = stored(&FeatureRef::row_in_group(7, 0, 7));
        assert!(anonymous.get("id").is_none(), "{anonymous}");

        let mut named = FeatureRef::row_in_group(7, 0, 7);
        named.feature_id = Some("west".to_string());
        assert_eq!(stored(&named)["id"], "west");
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

/// Whether the written payload bodies carry any source feature id.
///
/// Only a `FeatureJson` body has room for one; the fixed-width record the
/// other plans write has no id field at all, so a scan that collected ids
/// still produces an artifact without them.
#[cfg(feature = "_source")]
pub(crate) fn stores_feature_ids(plan: &PayloadPlan, features: &[FeatureRef]) -> bool {
    matches!(plan, PayloadPlan::FeatureJson { .. })
        && features.iter().any(|feature| feature.feature_id.is_some())
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
    ///
    /// `feature_id` participates, so refs decoded from payload bodies and refs
    /// decoded from fixed payload prefixes — which carry no id — sort together
    /// only because `row_number` already identifies a source feature.
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
    /// Emit a fixed-width [`FeatureRef`] followed by GeoJSON Feature bytes with
    /// projected properties.
    FeatureJson {
        /// Property projection.
        properties: PropertyProjection,
    },
}
