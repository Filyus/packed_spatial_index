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
    GeoArtifactIndex2D, GeoError, GeoMatch, GeoPayload, PayloadPlan, StreamLimits,
    open_geo_index_with_limits_async,
};
use serde_json::{Map, Value, json};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

const COLLECTION_ID: &str = "cities";
const COLLECTION_TITLE: &str = "Cities";
const COLLECTION_DESCRIPTION: &str =
    "Deterministic GeoParquet seed served directly from a GeoPSINDEX object in R2";

thread_local! {
    static DIRECTORY: RefCell<Option<GeoArtifactDirectory>> = const { RefCell::new(None) };
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
    max_reads: f64,
    detail: bool,
) -> Result<String, JsValue> {
    let index = open_2d(read_range, file_len, max_reads).await?;
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
    let index = open_2d(read_range, file_len, max_reads).await?;
    let bbox = Box2D::new(min_x, min_y, max_x, max_y);
    let limit = bounded_usize(limit, 100, 1_000);
    let offset = bounded_usize(offset, 0, usize::MAX);
    let payload_mode = parse_payload_mode(&payload)?;
    let result_level = parse_level(&level)?;

    let body = if payload_mode == PayloadMode::None && result_level == ResultLevel::Entry {
        let mut ids = index.search_entry_ids_async(bbox).await.map_err(geo_err)?;
        ids.sort_unstable();
        let number_matched = ids.len();
        let matches: Vec<Value> = page(&ids, offset, limit)
            .into_iter()
            .map(|entry_id| json!({ "entryId": entry_id }))
            .collect();
        json!({
            "collectionId": COLLECTION_ID,
            "query": query_json([min_x, min_y, max_x, max_y], limit, offset, payload_mode, result_level),
            "payloadKind": payload_kind(&index.manifest().payload_plan),
            "numberMatched": number_matched,
            "numberReturned": matches.len(),
            "matches": matches,
        })
    } else {
        let mut matches = match result_level {
            ResultLevel::Feature => index
                .search_feature_matches_async(bbox)
                .await
                .map_err(geo_err)?,
            ResultLevel::Entry => {
                let mut matches = index.search_matches_async(bbox).await.map_err(geo_err)?;
                GeoMatch::sort_by_entry(&mut matches);
                matches
            }
        };
        if result_level == ResultLevel::Feature {
            GeoMatch::sort_by_entry(&mut matches);
        }
        let number_matched = matches.len();
        let records: Vec<Value> = page(&matches, offset, limit)
            .into_iter()
            .map(|m| match_record(m, payload_mode, result_level))
            .collect();
        json!({
            "collectionId": COLLECTION_ID,
            "query": query_json([min_x, min_y, max_x, max_y], limit, offset, payload_mode, result_level),
            "payloadKind": payload_kind(&index.manifest().payload_plan),
            "numberMatched": number_matched,
            "numberReturned": records.len(),
            "matches": records,
        })
    };

    Ok(body.to_string())
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn items(
    read_range: Function,
    file_len: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    limit: f64,
    offset: f64,
    max_reads: f64,
) -> Result<String, JsValue> {
    let index = open_2d(read_range, file_len, max_reads).await?;
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
    let matches = index
        .search_feature_matches_async(bbox)
        .await
        .map_err(geo_err)?;
    let number_matched = matches.len();
    let features: Vec<Value> = page(&matches, offset, limit)
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
            "bbox": [min_x, min_y, max_x, max_y],
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
    max_reads: f64,
) -> Result<GeoArtifactIndex2D<R2Reader>, JsValue> {
    let reader = R2Reader {
        read_range,
        len: (file_len > 0.0).then_some(file_len as u64),
    };
    let limits = StreamLimits {
        max_reads: (max_reads > 0.0).then_some(max_reads as usize),
        max_read_bytes: Some(16 * 1024 * 1024),
        max_items: Some(1_000_000),
        directory_budget_bytes: Some(16 * 1024 * 1024),
        coalesce_gap_bytes: Some(256 * 1024),
    };

    let cached = DIRECTORY.with(|d| d.borrow().clone());
    let index = match cached {
        Some(dir) => {
            GeoArtifactIndex::from_directory_with_limits(&dir, reader, limits).map_err(geo_err)?
        }
        None => {
            let opened = open_geo_index_with_limits_async(reader, limits)
                .await
                .map_err(geo_err)?;
            let (dir, reader) = opened.into_directory();
            DIRECTORY.with(|d| *d.borrow_mut() = Some(dir.clone()));
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
