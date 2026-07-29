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
    AsyncRangeReader, Box2D, Box3D, FeatureRef, GeoArtifactDirectory, GeoArtifactIndex,
    GeoArtifactIndex2D, GeoArtifactIndex3D, GeoArtifactManifest, GeoError, GeoMatch,
    GeoMatchHeader, GeoMatchHeaderPage, GeoPayload, GeoPayloadHeader, GeoPayloadHeaderPage,
    PayloadPlan, StreamError, StreamLimits, open_geo_index_with_limits_async,
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
    let index = open_index(read_range, file_len, object_etag, max_reads).await?;
    let (dir, _reader) = index.into_directory();
    let manifest = dir.manifest();
    let mut out = collection_summary(manifest, dir.num_entries(), dir.node_size());
    if detail {
        let obj = out.as_object_mut().ok_or_else(|| {
            worker_err(
                500,
                "artifact_error",
                "collection summary was not an object",
            )
        })?;
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
    bbox: Vec<f64>,
    limit: f64,
    offset: f64,
    payload: String,
    level: String,
    max_reads: f64,
) -> Result<String, JsValue> {
    let index = open_index(read_range, file_len, object_etag, max_reads).await?;
    let limit = bounded_usize(limit, 100, 1_000);
    let offset = bounded_usize(offset, 0, usize::MAX);
    let payload_mode = parse_payload_mode(&payload)?;
    let result_level = parse_level(&level)?;

    match (index, bbox.len()) {
        (GeoArtifactIndex::D2(index), 4) => {
            search_impl(
                index,
                box_2d(&bbox),
                &bbox,
                limit,
                offset,
                payload_mode,
                result_level,
            )
            .await
        }
        (GeoArtifactIndex::D3(index), 6) => {
            search_impl(
                index,
                box_3d(&bbox),
                &bbox,
                limit,
                offset,
                payload_mode,
                result_level,
            )
            .await
        }
        (index, len) => Err(bbox_arity_error(artifact_dims(&index), len)),
    }
}

async fn search_impl<I: AsyncGeoIndex>(
    index: I,
    query: I::Query,
    bbox: &[f64],
    limit: usize,
    offset: usize,
    payload_mode: PayloadMode,
    result_level: ResultLevel,
) -> Result<String, JsValue> {
    let records: Vec<Value>;
    let number_matched;
    if !index.manifest().entries_may_duplicate_rows {
        let page = index
            .search_payload_headers_page(query, offset, limit)
            .await
            .map_err(geo_err)?;
        number_matched = page.number_matched;
        let page_headers = page.headers;
        records = if payload_mode == PayloadMode::Full {
            index
                .fetch_payload_header_matches(&page_headers)
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
        let (matched, page_headers) = if result_level == ResultLevel::Entry {
            let page = index
                .search_match_headers_page(query, offset, limit)
                .await
                .map_err(geo_err)?;
            (page.number_matched, page.headers)
        } else {
            let mut headers = index.search_match_headers(query).await.map_err(geo_err)?;
            GeoMatchHeader::dedupe_by_feature(&mut headers);
            (headers.len(), page(&headers, offset, limit))
        };
        number_matched = matched;
        records = if payload_mode == PayloadMode::Full {
            index
                .fetch_matches(&page_headers)
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
        "query": query_json(bbox, limit, offset, payload_mode, result_level),
        "payloadKind": payload_kind(&index.manifest().payload_plan),
        "numberMatched": number_matched,
        "numberReturned": records.len(),
        "matches": records,
    });

    Ok(body.to_string())
}

#[wasm_bindgen]
pub async fn items(
    read_range: Function,
    file_len: f64,
    object_etag: String,
    bbox: Vec<f64>,
    limit: f64,
    offset: f64,
    max_reads: f64,
) -> Result<String, JsValue> {
    let index = open_index(read_range, file_len, object_etag, max_reads).await?;
    if !matches!(
        index.manifest().payload_plan,
        PayloadPlan::FeatureJson { .. }
    ) {
        return Err(worker_err(
            422,
            "unsupported_payload",
            "/items requires an artifact built with --payload feature-json",
        ));
    }

    let limit = bounded_usize(limit, 100, 1_000);
    let offset = bounded_usize(offset, 0, usize::MAX);

    match (index, bbox.len()) {
        (GeoArtifactIndex::D2(index), 4) => {
            items_impl(index, box_2d(&bbox), &bbox, limit, offset).await
        }
        (GeoArtifactIndex::D3(index), 6) => {
            items_impl(index, box_3d(&bbox), &bbox, limit, offset).await
        }
        (index, len) => Err(bbox_arity_error(artifact_dims(&index), len)),
    }
}

async fn items_impl<I: AsyncGeoIndex>(
    index: I,
    query: I::Query,
    bbox: &[f64],
    limit: usize,
    offset: usize,
) -> Result<String, JsValue> {
    if !index.manifest().entries_may_duplicate_rows {
        let page = index
            .search_payload_headers_page(query, offset, limit)
            .await
            .map_err(geo_err)?;
        let number_matched = page.number_matched;
        let page_headers = page.headers;
        let matches = index
            .fetch_payload_header_matches(&page_headers)
            .await
            .map_err(geo_err)?;
        return items_response(matches, number_matched, bbox, limit, offset);
    }

    let mut headers = index.search_match_headers(query).await.map_err(geo_err)?;
    GeoMatchHeader::dedupe_by_feature(&mut headers);
    let number_matched = headers.len();
    let page_headers = page(&headers, offset, limit);
    let matches = index.fetch_matches(&page_headers).await.map_err(geo_err)?;
    items_response(matches, number_matched, bbox, limit, offset)
}

fn items_response(
    matches: Vec<GeoMatch>,
    number_matched: usize,
    bbox: &[f64],
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

/// The async artifact methods this Worker needs, in a shape that is the same
/// for both dimensions.
///
/// The two query bodies differ only in the type of the region they search, so
/// they are written once against this trait rather than duplicated per
/// dimension. It stays private and is only ever used through the generic
/// `search_impl`/`items_impl`, so `async fn` in a trait costs nothing here.
trait AsyncGeoIndex {
    type Query;

    fn manifest(&self) -> &GeoArtifactManifest;

    async fn search_payload_headers_page(
        &self,
        query: Self::Query,
        offset: usize,
        limit: usize,
    ) -> Result<GeoPayloadHeaderPage, GeoError>;

    async fn fetch_payload_header_matches(
        &self,
        headers: &[GeoPayloadHeader],
    ) -> Result<Vec<GeoMatch>, GeoError>;

    async fn search_match_headers_page(
        &self,
        query: Self::Query,
        offset: usize,
        limit: usize,
    ) -> Result<GeoMatchHeaderPage, GeoError>;

    async fn search_match_headers(
        &self,
        query: Self::Query,
    ) -> Result<Vec<GeoMatchHeader>, GeoError>;

    async fn fetch_matches(&self, headers: &[GeoMatchHeader]) -> Result<Vec<GeoMatch>, GeoError>;
}

macro_rules! impl_async_geo_index {
    ($index:ty, $query:ty) => {
        impl AsyncGeoIndex for $index {
            type Query = $query;

            fn manifest(&self) -> &GeoArtifactManifest {
                <$index>::manifest(self)
            }

            async fn search_payload_headers_page(
                &self,
                query: Self::Query,
                offset: usize,
                limit: usize,
            ) -> Result<GeoPayloadHeaderPage, GeoError> {
                self.search_payload_headers_page_async(query, offset, limit)
                    .await
            }

            async fn fetch_payload_header_matches(
                &self,
                headers: &[GeoPayloadHeader],
            ) -> Result<Vec<GeoMatch>, GeoError> {
                self.fetch_payload_header_matches_async(headers).await
            }

            async fn search_match_headers_page(
                &self,
                query: Self::Query,
                offset: usize,
                limit: usize,
            ) -> Result<GeoMatchHeaderPage, GeoError> {
                self.search_match_headers_page_async(query, offset, limit)
                    .await
            }

            async fn search_match_headers(
                &self,
                query: Self::Query,
            ) -> Result<Vec<GeoMatchHeader>, GeoError> {
                self.search_match_headers_async(query).await
            }

            async fn fetch_matches(
                &self,
                headers: &[GeoMatchHeader],
            ) -> Result<Vec<GeoMatch>, GeoError> {
                self.fetch_matches_async(headers).await
            }
        }
    };
}

impl_async_geo_index!(GeoArtifactIndex2D<R2Reader>, Box2D);
impl_async_geo_index!(GeoArtifactIndex3D<R2Reader>, Box3D);

fn box_2d(bbox: &[f64]) -> Box2D {
    Box2D::new(bbox[0], bbox[1], bbox[2], bbox[3])
}

fn box_3d(bbox: &[f64]) -> Box3D {
    Box3D::new(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5])
}

/// Index dimensions of an opened artifact.
///
/// The manifest carries the *source* coordinate dimensions, which are not the
/// same question -- an `xym` column builds a 2D index -- and the mapping
/// between them is private to the geo crate. The opened variant answers it
/// directly.
fn artifact_dims<R>(index: &GeoArtifactIndex<R>) -> u8 {
    match index {
        GeoArtifactIndex::D2(_) => 2,
        GeoArtifactIndex::D3(_) => 3,
    }
}

/// Report a bbox whose length does not match the artifact.
///
/// The Worker cannot know the dimensions until the object is open, so the
/// arity check lives here rather than in the request parser, and the message
/// names what the artifact actually is.
fn bbox_arity_error(dims: u8, len: usize) -> JsValue {
    worker_err(400, "invalid_bbox", bbox_arity_message(dims, len))
}

fn bbox_arity_message(dims: u8, len: usize) -> String {
    format!(
        "this artifact is {dims}D, so bbox must contain {} numbers, not {len}",
        u32::from(dims) * 2
    )
}

async fn open_index(
    read_range: Function,
    file_len: f64,
    object_etag: String,
    max_reads: f64,
) -> Result<GeoArtifactIndex<R2Reader>, JsValue> {
    let identity = object_identity(file_len, object_etag)
        .map_err(|message| worker_err(500, "artifact_error", message))?;
    let reader = R2Reader {
        read_range,
        len: Some(identity.file_len),
    };
    let mut limits = StreamLimits::default();
    limits.max_reads = (max_reads > 0.0).then_some(max_reads as usize);
    limits.max_read_bytes = Some(16 * 1024 * 1024);
    limits.max_items = Some(1_000_000);
    limits.directory_budget_bytes = Some(16 * 1024 * 1024);
    limits.coalesce_gap_bytes = Some(256 * 1024);
    // Every read here is an R2 round trip and a billed operation, so buy them
    // back with bytes: a strided prefix scan would otherwise issue one request
    // per match.
    limits.prefix_coalesce_gap_bytes = Some(256 * 1024);

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

    Ok(index)
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
    manifest: &GeoArtifactManifest,
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
    bbox: &[f64],
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
        _ => Err(worker_err(
            400,
            "invalid_payload",
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
        _ => Err(worker_err(
            400,
            "invalid_level",
            "level must be one of entry, feature",
        )),
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
    let message = v
        .dyn_ref::<js_sys::Error>()
        .map(|error| error.message().into())
        .or_else(|| v.as_string())
        .unwrap_or_else(|| "js error in read_range".to_string());
    io::Error::other(message)
}

/// One failure, in the shape the Worker's HTTP layer speaks.
///
/// Serialized as JSON across the wasm boundary so the classification survives
/// the trip: `wasm_bindgen` can only reject with a `JsValue`, and a bare string
/// would leave the TypeScript side unable to tell "the bbox is the wrong
/// length" from "the artifact is corrupt". The codes and statuses are the
/// native server's (`server/src/error.rs`).
struct WorkerError {
    status: u16,
    code: &'static str,
    message: String,
}

impl From<WorkerError> for JsValue {
    fn from(err: WorkerError) -> Self {
        JsValue::from_str(
            &json!({
                "status": err.status,
                "code": err.code,
                "message": err.message,
            })
            .to_string(),
        )
    }
}

fn worker_err(status: u16, code: &'static str, message: impl Into<String>) -> JsValue {
    WorkerError {
        status,
        code,
        message: message.into(),
    }
    .into()
}

/// Classify a [`GeoError`], mirroring the intercepts the native server applies
/// before its catch-all (`server/src/error.rs`).
///
/// The default is a 500: a geo error that is not one of the cases below
/// describes the artifact, not the request, so blaming the client would be
/// wrong. The exceptions are the ones the client *can* act on — narrow the
/// bbox, or ask for something the artifact supports.
fn geo_err(e: GeoError) -> JsValue {
    let (status, code) = geo_error_class(&e);
    worker_err(status, code, e.to_string())
}

fn geo_error_class(e: &GeoError) -> (u16, &'static str) {
    match e {
        GeoError::Stream(StreamError::LimitExceeded) => (422, "query_too_large"),
        GeoError::NonPlanarExactPredicate { .. } | GeoError::NonSphericalExactPredicate { .. } => {
            (422, "unsupported_query")
        }
        GeoError::InvalidSphericalQuery(_) | GeoError::EmptyQueryPolygon => (400, "invalid_bbox"),
        _ => (500, "artifact_error"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GeoError, PayloadMode, ResultLevel, StreamError, bbox_arity_message, geo_error_class,
        object_identity, query_json,
    };

    #[test]
    fn geo_errors_are_classified_like_the_server() {
        // The client can act on these two, so they are 4xx.
        assert_eq!(
            geo_error_class(&GeoError::Stream(StreamError::LimitExceeded)),
            (422, "query_too_large")
        );
        assert_eq!(
            geo_error_class(&GeoError::InvalidSphericalQuery("bad".into())),
            (400, "invalid_bbox")
        );
        // Everything else describes the artifact rather than the request, so
        // blaming the caller would be wrong.
        assert_eq!(
            geo_error_class(&GeoError::MissingGeoManifest),
            (500, "artifact_error")
        );
        assert_eq!(
            geo_error_class(&GeoError::PayloadDecode("truncated".into())),
            (500, "artifact_error")
        );
        assert_eq!(
            geo_error_class(&GeoError::Stream(StreamError::NoPayload)),
            (500, "artifact_error")
        );
    }

    #[test]
    fn bbox_arity_message_names_the_artifact_dimensions() {
        assert_eq!(
            bbox_arity_message(3, 4),
            "this artifact is 3D, so bbox must contain 6 numbers, not 4"
        );
        assert_eq!(
            bbox_arity_message(2, 6),
            "this artifact is 2D, so bbox must contain 4 numbers, not 6"
        );
    }

    #[test]
    fn query_json_echoes_the_bbox_it_was_given() {
        let two = query_json(
            &[1.0, 2.0, 3.0, 4.0],
            10,
            0,
            PayloadMode::Summary,
            ResultLevel::Feature,
        );
        assert_eq!(two["bbox"], serde_json::json!([1.0, 2.0, 3.0, 4.0]));

        let three = query_json(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            10,
            0,
            PayloadMode::Summary,
            ResultLevel::Feature,
        );
        assert_eq!(
            three["bbox"],
            serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
    }

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
