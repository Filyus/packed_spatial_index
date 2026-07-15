//! Geo-first WASM entry for the Cloudflare Worker + R2 demo.
//!
//! The Worker passes in a JS `read_range(offset, length) -> Promise<Uint8Array>`
//! callback backed by R2 range reads. This module wraps it as an
//! [`AsyncRangeReader`], caches the parsed [`GeoArtifactDirectory`] for warm
//! isolates, and returns API-shaped JSON for a single FeatureJson-backed
//! collection.

use std::cell::RefCell;
use std::io;

use js_sys::{Function, Promise, Uint8Array};
use packed_spatial_index_geo::{
    AsyncRangeReader, Box2D, FeatureRef, GeoArtifactDirectory, GeoArtifactIndex,
    GeoArtifactIndex2D, GeoError, GeoMatch, GeoMatchHeader, GeoPayload, GeoPayloadHeader,
    PayloadPlan, StreamLimits, open_geo_index_with_limits_async,
};
use serde_json::{Map, Value, json};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

const COLLECTION_ID: &str = "synthetic-points";
const COLLECTION_TITLE: &str = "Synthetic clustered points";
const COLLECTION_DESCRIPTION: &str =
    "Deterministic synthetic GeoParquet seed served directly from a GeoPSINDEX object in R2";

thread_local! {
    static DIRECTORY: RefCell<Option<CachedDirectory>> = const { RefCell::new(None) };
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObjectIdentity {
    etag: String,
    file_len: u64,
}

#[derive(Clone)]
struct CachedDirectory {
    identity: ObjectIdentity,
    directory: GeoArtifactDirectory,
}

struct R2Reader {
    read_range: Function,
    len: Option<u64>,
}

impl AsyncRangeReader for R2Reader {
    async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let promise = self
            .read_range
            .call2(
                &JsValue::NULL,
                &JsValue::from_f64(offset as f64),
                &JsValue::from_f64(buf.len() as f64),
            )
            .map_err(js_io)?;
        let promise: Promise = promise
            .dyn_into()
            .map_err(|_| io_err("read_range must return a Promise"))?;
        let value = JsFuture::from(promise).await.map_err(js_io)?;
        let arr: Uint8Array = value
            .dyn_into()
            .map_err(|_| io_err("range result must be a Uint8Array"))?;
        if arr.length() as usize != buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short range read",
            ));
        }
        arr.copy_to(buf);
        Ok(())
    }

    fn len(&self) -> Option<u64> {
        self.len
    }
}

#[wasm_bindgen]
pub async fn collection(
    read_range: Function,
    file_len: f64,
    object_etag: String,
    max_reads: f64,
    detail: bool,
) -> Result<String, JsValue> {
    let index = open_2d(read_range, file_len, object_etag, max_reads).await?;
    let (dir, _reader) = index.into_directory();
    let manifest = dir.manifest();
    let mut out = collection_summary(manifest, dir.num_entries(), dir.node_size());
    if detail {
        let obj = out
            .as_object_mut()
            .ok_or_else(|| JsValue::from_str("collection summary was not an object"))?;
        obj.insert("nodeSize".to_string(), json!(dir.node_size()));
        obj.insert("sourceFormat".to_string(), json!(manifest.source_format));
        obj.insert(
            "sourceFingerprint".to_string(),
            json!(manifest.source_fingerprint),
        );
        obj.insert(
            "selectedColumn".to_string(),
            json!(manifest.selected_column),
        );
        obj.insert("crs".to_string(), json!(manifest.crs));
        obj.insert("edges".to_string(), json!(manifest.edges));
        obj.insert("encoding".to_string(), json!(manifest.encoding));
    }
    Ok(out.to_string())
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn search(
    read_range: Function,
    file_len: f64,
    object_etag: String,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    limit: f64,
    offset: f64,
    payload: String,
    level: String,
    max_reads: f64,
) -> Result<String, JsValue> {
    let index = open_2d(read_range, file_len, object_etag, max_reads).await?;
    let bbox = Box2D::new(min_x, min_y, max_x, max_y);
    let limit = bounded_usize(limit, 100, 1_000);
    let offset = bounded_usize(offset, 0, usize::MAX);
    let payload_mode = parse_payload_mode(&payload)?;
    let result_level = parse_level(&level)?;

    let records: Vec<Value>;
    let number_matched;
    if !index.manifest().entries_may_duplicate_rows {
        let mut headers = index
            .search_payload_headers_async(bbox)
            .await
            .map_err(geo_err)?;
        GeoPayloadHeader::sort_by_entry(&mut headers);
        number_matched = headers.len();
        let page_headers = page(&headers, offset, limit);
        records = if payload_mode == PayloadMode::Full {
            index
                .fetch_payload_header_matches_async(&page_headers)
                .await
                .map_err(geo_err)?
                .into_iter()
                .map(|m| match_record(m, payload_mode, result_level))
                .collect()
        } else {
            page_headers
                .into_iter()
                .map(|h| payload_header_record(h, payload_mode, &index.manifest().payload_plan))
                .collect()
        };
    } else {
        let mut headers = index
            .search_match_headers_async(bbox)
            .await
            .map_err(geo_err)?;
        GeoMatchHeader::sort_by_entry(&mut headers);
        if result_level == ResultLevel::Feature {
            GeoMatchHeader::dedupe_by_feature(&mut headers);
        }
        number_matched = headers.len();
        let page_headers = page(&headers, offset, limit);
        records = if payload_mode == PayloadMode::Full {
            index
                .fetch_matches_async(&page_headers)
                .await
                .map_err(geo_err)?
                .into_iter()
                .map(|m| match_record(m, payload_mode, result_level))
                .collect()
        } else {
            page_headers
                .into_iter()
                .map(|h| header_record(h, payload_mode, &index.manifest().payload_plan))
                .collect()
        };
    }

    let body = json!({
        "collectionId": COLLECTION_ID,
        "query": query_json([min_x, min_y, max_x, max_y], limit, offset, payload_mode, result_level),
        "payloadKind": payload_kind(&index.manifest().payload_plan),
        "numberMatched": number_matched,
        "numberReturned": records.len(),
        "matches": records,
    });

    Ok(body.to_string())
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn items(
    read_range: Function,
    file_len: f64,
    object_etag: String,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    limit: f64,
    offset: f64,
    max_reads: f64,
) -> Result<String, JsValue> {
    let index = open_2d(read_range, file_len, object_etag, max_reads).await?;
    if !matches!(
        index.manifest().payload_plan,
        PayloadPlan::FeatureJson { .. }
    ) {
        return Err(JsValue::from_str(
            "/items requires an artifact built with --payload feature-json",
        ));
    }

    let bbox = Box2D::new(min_x, min_y, max_x, max_y);
    let limit = bounded_usize(limit, 100, 1_000);
    let offset = bounded_usize(offset, 0, usize::MAX);
    if !index.manifest().entries_may_duplicate_rows {
        let mut headers = index
            .search_payload_headers_async(bbox)
            .await
            .map_err(geo_err)?;
        GeoPayloadHeader::sort_by_entry(&mut headers);
        let number_matched = headers.len();
        let page_headers = page(&headers, offset, limit);
        let matches = index
            .fetch_payload_header_matches_async(&page_headers)
            .await
            .map_err(geo_err)?;
        return items_response(
            matches,
            number_matched,
            [min_x, min_y, max_x, max_y],
            limit,
            offset,
        );
    }

    let mut headers = index
        .search_match_headers_async(bbox)
        .await
        .map_err(geo_err)?;
    GeoMatchHeader::sort_by_entry(&mut headers);
    GeoMatchHeader::dedupe_by_feature(&mut headers);
    let number_matched = headers.len();
    let page_headers = page(&headers, offset, limit);
    let matches = index
        .fetch_matches_async(&page_headers)
        .await
        .map_err(geo_err)?;
    items_response(
        matches,
        number_matched,
        [min_x, min_y, max_x, max_y],
        limit,
        offset,
    )
}

fn items_response(
    matches: Vec<GeoMatch>,
    number_matched: usize,
    bbox: [f64; 4],
    limit: usize,
    offset: usize,
) -> Result<String, JsValue> {
    let features: Vec<Value> = matches
        .into_iter()
        .filter_map(|m| match m.payload {
            GeoPayload::FeatureJson(feature) => Some(public_feature_json(feature)),
            _ => None,
        })
        .collect();

    Ok(json!({
        "type": "FeatureCollection",
        "features": features,
        "numberMatched": number_matched,
        "numberReturned": features.len(),
        "query": {
            "bbox": bbox,
            "predicate": "bbox",
            "limit": limit,
            "offset": offset,
        },
    })
    .to_string())
}

async fn open_2d(
    read_range: Function,
    file_len: f64,
    object_etag: String,
    max_reads: f64,
) -> Result<GeoArtifactIndex2D<R2Reader>, JsValue> {
    let identity = object_identity(file_len, object_etag).map_err(JsValue::from_str)?;
    let reader = R2Reader {
        read_range,
        len: Some(identity.file_len),
    };
    let limits = StreamLimits {
        max_reads: (max_reads > 0.0).then_some(max_reads as usize),
        max_read_bytes: Some(16 * 1024 * 1024),
        max_items: Some(1_000_000),
        directory_budget_bytes: Some(16 * 1024 * 1024),
        coalesce_gap_bytes: Some(256 * 1024),
    };

    let cached = DIRECTORY.with(|d| {
        d.borrow()
            .as_ref()
            .filter(|cached| cached.identity == identity)
            .map(|cached| cached.directory.clone())
    });
    let index = match cached {
        Some(dir) => {
            GeoArtifactIndex::from_directory_with_limits(&dir, reader, limits).map_err(geo_err)?
        }
        None => {
            let opened = open_geo_index_with_limits_async(reader, limits)
                .await
                .map_err(geo_err)?;
            let (dir, reader) = opened.into_directory();
            DIRECTORY.with(|d| {
                *d.borrow_mut() = Some(CachedDirectory {
                    identity,
                    directory: dir.clone(),
                });
            });
            GeoArtifactIndex::from_directory_with_limits(&dir, reader, limits).map_err(geo_err)?
        }
    };

    match index {
        GeoArtifactIndex::D2(index) => Ok(index),
        GeoArtifactIndex::D3(_) => Err(JsValue::from_str(
            "this demo Worker serves 2D bbox artifacts only",
        )),
    }
}

fn object_identity(file_len: f64, etag: String) -> Result<ObjectIdentity, &'static str> {
    if !file_len.is_finite()
        || file_len <= 0.0
        || file_len.fract() != 0.0
        || file_len > u64::MAX as f64
    {
        return Err("R2 object length must be a positive integer");
    }
    if etag.is_empty() {
        return Err("R2 object ETag is missing");
    }
    Ok(ObjectIdentity {
        etag,
        file_len: file_len as u64,
    })
}

fn collection_summary(
    manifest: &packed_spatial_index_geo::GeoArtifactManifest,
    entry_count: usize,
    node_size: usize,
) -> Value {
    let payload_kind = payload_kind(&manifest.payload_plan);
    json!({
        "id": COLLECTION_ID,
        "title": COLLECTION_TITLE,
        "description": COLLECTION_DESCRIPTION,
        "featureCount": manifest.feature_count,
        "entryCount": entry_count,
        "dims": manifest.dims,
        "storagePrecision": manifest.storage_precision,
        "payloadKind": payload_kind,
        "nodeSize": node_size,
        "capabilities": {
            "items": payload_kind == "feature_json",
            "predicates": ["bbox"],
            "levels": ["feature", "entry"],
            "payloadModes": ["none", "summary", "full"],
        },
    })
}

fn query_json(
    bbox: [f64; 4],
    limit: usize,
    offset: usize,
    payload: PayloadMode,
    level: ResultLevel,
) -> Value {
    json!({
        "bbox": bbox,
        "predicate": "bbox",
        "level": level.as_str(),
        "payload": payload.as_str(),
        "limit": limit,
        "offset": offset,
    })
}

fn match_record(m: GeoMatch, payload_mode: PayloadMode, level: ResultLevel) -> Value {
    let mut feature_ref = m.feature;
    if level == ResultLevel::Feature {
        feature_ref.part = None;
    }
    let payload = match (payload_mode, m.payload) {
        (PayloadMode::None, _) => Value::Null,
        (_, GeoPayload::RowRef) => json!({ "kind": "row_ref" }),
        (mode, GeoPayload::RowWkb(wkb)) => json!({
            "kind": "row_wkb",
            "byteLength": wkb.len(),
            "wkbBase64": (mode == PayloadMode::Full).then(|| base64(&wkb)),
        }),
        (mode, GeoPayload::FeatureJson(feature)) => json!({
            "kind": "feature_json",
            "feature": (mode == PayloadMode::Full).then(|| public_feature_json(feature)),
        }),
    };
    let mut record = Map::new();
    record.insert("entryId".to_string(), json!(m.entry_id));
    record.insert("featureRef".to_string(), feature_ref_json(feature_ref));
    if payload_mode != PayloadMode::None {
        record.insert("payload".to_string(), strip_null_object_fields(payload));
    }
    Value::Object(record)
}

fn header_record(header: GeoMatchHeader, payload_mode: PayloadMode, plan: &PayloadPlan) -> Value {
    let body_byte_len = header.body_byte_len().unwrap_or(0);
    let mut record = Map::new();
    record.insert("entryId".to_string(), json!(header.entry_id));
    record.insert("featureRef".to_string(), feature_ref_json(header.feature));
    if payload_mode != PayloadMode::None {
        record.insert(
            "payload".to_string(),
            match plan {
                PayloadPlan::RowRef => json!({ "kind": "row_ref" }),
                PayloadPlan::RowWkb => json!({
                    "kind": "row_wkb",
                    "byteLength": body_byte_len,
                }),
                PayloadPlan::FeatureJson { .. } => json!({ "kind": "feature_json" }),
                PayloadPlan::None => json!({ "kind": "none" }),
            },
        );
    }
    Value::Object(record)
}

fn payload_header_record(
    header: GeoPayloadHeader,
    payload_mode: PayloadMode,
    plan: &PayloadPlan,
) -> Value {
    let mut record = Map::new();
    record.insert("entryId".to_string(), json!(header.entry_id));
    if payload_mode != PayloadMode::None {
        record.insert(
            "payload".to_string(),
            match plan {
                PayloadPlan::RowRef => json!({ "kind": "row_ref" }),
                PayloadPlan::RowWkb => json!({
                    "kind": "row_wkb",
                    "byteLength": header.body_byte_len().unwrap_or(0),
                }),
                PayloadPlan::FeatureJson { .. } => json!({ "kind": "feature_json" }),
                PayloadPlan::None => json!({ "kind": "none" }),
            },
        );
    }
    Value::Object(record)
}

fn feature_ref_json(feature: FeatureRef) -> Value {
    let mut out = Map::new();
    out.insert("rowNumber".to_string(), json!(feature.row_number));
    if let Some(row_group) = feature.row_group {
        out.insert("rowGroup".to_string(), json!(row_group));
    }
    if let Some(row_in_group) = feature.row_in_group {
        out.insert("rowInGroup".to_string(), json!(row_in_group));
    }
    if let Some(part) = feature.part {
        out.insert("part".to_string(), json!(part));
    }
    if let Some(feature_id) = feature.feature_id {
        out.insert("featureId".to_string(), json!(feature_id));
    }
    Value::Object(out)
}

fn public_feature_json(mut feature: Value) -> Value {
    if let Some(object) = feature.as_object_mut() {
        object.remove("feature_ref");
    }
    feature
}

fn payload_kind(plan: &PayloadPlan) -> &'static str {
    match plan {
        PayloadPlan::None => "none",
        PayloadPlan::RowRef => "row_ref",
        PayloadPlan::RowWkb => "row_wkb",
        PayloadPlan::FeatureJson { .. } => "feature_json",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PayloadMode {
    None,
    Summary,
    Full,
}

impl PayloadMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Summary => "summary",
            Self::Full => "full",
        }
    }
}

fn parse_payload_mode(value: &str) -> Result<PayloadMode, JsValue> {
    match value {
        "none" => Ok(PayloadMode::None),
        "summary" => Ok(PayloadMode::Summary),
        "full" => Ok(PayloadMode::Full),
        _ => Err(JsValue::from_str(
            "payload must be one of none, summary, full",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultLevel {
    Entry,
    Feature,
}

impl ResultLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Entry => "entry",
            Self::Feature => "feature",
        }
    }
}

fn parse_level(value: &str) -> Result<ResultLevel, JsValue> {
    match value {
        "entry" => Ok(ResultLevel::Entry),
        "feature" => Ok(ResultLevel::Feature),
        _ => Err(JsValue::from_str("level must be one of entry, feature")),
    }
}

fn bounded_usize(value: f64, default: usize, max: usize) -> usize {
    if !value.is_finite() || value < 0.0 {
        return default;
    }
    (value as usize).min(max)
}

fn page<T: Clone>(values: &[T], offset: usize, limit: usize) -> Vec<T> {
    values.iter().skip(offset).take(limit).cloned().collect()
}

fn strip_null_object_fields(value: Value) -> Value {
    let Value::Object(mut map) = value else {
        return value;
    };
    map.retain(|_, v| !v.is_null());
    Value::Object(map)
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        s.push(T[(n >> 18 & 63) as usize] as char);
        s.push(T[(n >> 12 & 63) as usize] as char);
        s.push(if chunk.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        s.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    s
}

fn io_err(msg: &str) -> io::Error {
    io::Error::other(msg)
}

fn js_io(v: JsValue) -> io::Error {
    io::Error::other(
        v.as_string()
            .unwrap_or_else(|| "js error in read_range".to_string()),
    )
}

fn geo_err(e: GeoError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[cfg(test)]
mod tests {
    use super::object_identity;

    #[test]
    fn object_identity_changes_with_etag_or_length() {
        let original = object_identity(1024.0, "etag-a".to_string()).unwrap();
        let replaced = object_identity(1024.0, "etag-b".to_string()).unwrap();
        let resized = object_identity(2048.0, "etag-a".to_string()).unwrap();

        assert_ne!(original, replaced);
        assert_ne!(original, resized);
    }

    #[test]
    fn object_identity_rejects_missing_or_invalid_metadata() {
        assert!(object_identity(0.0, "etag".to_string()).is_err());
        assert!(object_identity(1.5, "etag".to_string()).is_err());
        assert!(object_identity(1.0, String::new()).is_err());
    }
}
