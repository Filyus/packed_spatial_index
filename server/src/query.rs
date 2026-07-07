use base64::{Engine as _, engine::general_purpose::STANDARD};
use packed_spatial_index_geo::{
    Box2D, Box3D, CoordinateDims, CrsInfo, EdgeModel, FeatureRef, GeoArtifactIndex, GeoMatch,
    GeoMatchHeader, GeoPayload, GeoQuery2D, GeometryEncoding, NonPlanarExactPolicy, PayloadPlan,
    SpatialPredicate, StoragePrecision,
};
use serde::{Deserialize, Serialize};

use crate::{Collection, ServerError};

const DEFAULT_LIMIT: usize = 100;
const MAX_LIMIT: usize = 10_000;

/// Query parameters accepted by `/search` and `/items`.
///
/// `/items` rejects `level` and `payload`; both endpoints share the rest.
#[derive(Debug, Deserialize)]
pub struct SearchParams {
    /// Query bbox as comma-separated numbers.
    #[serde(default)]
    pub bbox: Option<String>,
    /// Maximum returned records.
    #[serde(default)]
    pub limit: Option<String>,
    /// Number of matched records to skip.
    #[serde(default)]
    pub offset: Option<String>,
    /// Spatial predicate: `bbox` or `intersects`.
    #[serde(default)]
    pub predicate: Option<String>,
    /// Result level for `/search`: `feature` or `entry`.
    #[serde(default)]
    pub level: Option<String>,
    /// Payload materialization mode for `/search`.
    #[serde(default)]
    pub payload: Option<String>,
}

/// Payload materialization mode for `/search`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadMode {
    /// Omit the payload object from each match.
    None,
    /// Return payload kind and cheap metadata only.
    #[default]
    Summary,
    /// Return full payload values where the artifact stores them.
    Full,
}

/// Spatial predicate applied by a search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryPredicate {
    /// Envelope intersection against the packed index only.
    #[default]
    Bbox,
    /// Exact geometry intersection refined from artifact payloads.
    Intersects,
}

/// Result granularity for `/search`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultLevel {
    /// One record per source feature; split index entries are deduplicated.
    Feature,
    /// One record per index entry, including split parts.
    Entry,
}

/// Artifact payload kind in server wire vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadKind {
    /// Artifact stores no payload section.
    None,
    /// Artifact stores fixed-width feature refs.
    RowRef,
    /// Artifact stores WKB geometry bytes.
    RowWkb,
    /// Artifact stores GeoJSON features.
    FeatureJson,
}

impl From<&PayloadPlan> for PayloadKind {
    fn from(plan: &PayloadPlan) -> Self {
        match plan {
            PayloadPlan::None => Self::None,
            PayloadPlan::RowRef => Self::RowRef,
            PayloadPlan::RowWkb => Self::RowWkb,
            PayloadPlan::FeatureJson { .. } => Self::FeatureJson,
        }
    }
}

/// Effective query echoed back in search responses.
///
/// Field names match the query parameters exactly; values reflect applied
/// defaults, so clients can see how an omitted parameter was resolved.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryInfo {
    /// Parsed query bbox.
    pub bbox: Vec<f64>,
    /// Applied spatial predicate.
    pub predicate: QueryPredicate,
    /// Applied result level; `/items` responses omit it (always feature).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<ResultLevel>,
    /// Applied payload mode; `/items` responses omit it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<PayloadMode>,
    /// Applied limit.
    pub limit: usize,
    /// Applied offset.
    pub offset: usize,
}

/// Per-collection query capabilities exposed through the HTTP API.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    /// Whether `/items` can serve GeoJSON from this artifact.
    pub items: bool,
    /// Spatial predicates accepted by `/search` and `/items`.
    pub predicates: Vec<QueryPredicate>,
    /// Result levels accepted by `/search`.
    pub levels: Vec<ResultLevel>,
    /// Payload modes accepted by `/search`.
    pub payload_modes: Vec<PayloadMode>,
}

/// Collection summary returned by list/detail endpoints.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionSummary {
    /// Collection id.
    pub id: String,
    /// Optional title.
    pub title: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Number of unique source features represented in the artifact.
    pub feature_count: usize,
    /// Number of index entries.
    pub entry_count: usize,
    /// Artifact coordinate dimensions.
    pub dims: CoordinateDims,
    /// Artifact coordinate precision.
    pub storage_precision: StoragePrecision,
    /// Artifact payload kind.
    pub payload_kind: PayloadKind,
    /// Server capabilities for this collection.
    pub capabilities: Capabilities,
}

/// Collection detail returned by `/collections/{id}`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionDetail {
    /// Collection summary.
    #[serde(flatten)]
    pub summary: CollectionSummary,
    /// Packed node size in the artifact.
    pub node_size: usize,
    /// Source format label from `geoM`.
    pub source_format: String,
    /// Stable source metadata fingerprint.
    pub source_fingerprint: String,
    /// Selected geometry column.
    pub selected_column: String,
    /// CRS metadata from `geoM`.
    pub crs: CrsInfo,
    /// Edge model from `geoM`.
    pub edges: EdgeModel,
    /// Geometry encoding from `geoM`.
    pub encoding: GeometryEncoding,
}

/// `/search` response envelope.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResponse {
    /// Collection id.
    pub collection_id: String,
    /// Effective query after defaults were applied.
    pub query: QueryInfo,
    /// Artifact payload kind.
    pub payload_kind: PayloadKind,
    /// Total matched records before pagination.
    pub number_matched: usize,
    /// Returned records after pagination.
    pub number_returned: usize,
    /// Returned records.
    pub matches: Vec<MatchRecord>,
}

/// One `/search` record.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchRecord {
    /// Index entry ordinal in the artifact. Stable for one artifact build,
    /// not across rebuilds. At feature level this is the representative
    /// (lowest-part) entry of the source feature.
    pub entry_id: usize,
    /// Source feature ref when the payload contains one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_ref: Option<FeatureRefRecord>,
    /// Payload summary or value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<MatchPayload>,
}

/// Payload object for a match.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MatchPayload {
    /// Artifact has no payload section.
    None,
    /// Payload stores only a feature ref.
    RowRef,
    /// Payload stores WKB bytes.
    RowWkb {
        /// WKB payload byte length.
        #[serde(rename = "byteLength")]
        byte_length: usize,
        /// Base64 WKB bytes, present only when `payload=full`.
        #[serde(rename = "wkbBase64")]
        #[serde(skip_serializing_if = "Option::is_none")]
        wkb_base64: Option<String>,
    },
    /// Payload stores a GeoJSON Feature.
    FeatureJson {
        /// GeoJSON Feature, present only when `payload=full`.
        #[serde(skip_serializing_if = "Option::is_none")]
        feature: Option<serde_json::Value>,
    },
}

/// Source feature reference in HTTP response casing.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeatureRefRecord {
    /// Source-level row number.
    pub row_number: u64,
    /// GeoParquet row group when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_group: Option<u32>,
    /// Row within the row group when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_in_group: Option<u32>,
    /// Geometry part for entry-level records of split features; omitted at
    /// feature level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part: Option<u16>,
    /// Source feature id when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_id: Option<String>,
}

impl From<FeatureRef> for FeatureRefRecord {
    fn from(value: FeatureRef) -> Self {
        Self {
            row_number: value.row_number,
            row_group: value.row_group,
            row_in_group: value.row_in_group,
            part: value.part,
            feature_id: value.feature_id,
        }
    }
}

/// GeoJSON FeatureCollection response from `/items`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeatureCollectionResponse {
    /// GeoJSON type.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Returned GeoJSON features.
    pub features: Vec<serde_json::Value>,
    /// Total matched features before pagination.
    pub number_matched: usize,
    /// Returned features after pagination.
    pub number_returned: usize,
    /// Effective query after defaults were applied.
    pub query: QueryInfo,
}

impl CollectionSummary {
    /// Build a collection summary.
    pub fn new(collection: &Collection) -> Self {
        let manifest = collection.manifest();
        Self {
            id: collection.id().to_owned(),
            title: collection.title().map(str::to_owned),
            description: collection.description().map(str::to_owned),
            feature_count: manifest.feature_count,
            entry_count: collection.entry_count(),
            dims: manifest.dims,
            storage_precision: manifest.storage_precision,
            payload_kind: PayloadKind::from(&manifest.payload_plan),
            capabilities: capabilities(collection),
        }
    }
}

impl CollectionDetail {
    /// Build collection detail.
    pub fn new(collection: &Collection) -> Self {
        let manifest = collection.manifest();
        Self {
            summary: CollectionSummary::new(collection),
            node_size: collection.node_size(),
            source_format: manifest.source_format.clone(),
            source_fingerprint: manifest.source_fingerprint.clone(),
            selected_column: manifest.selected_column.clone(),
            crs: manifest.crs.clone(),
            edges: manifest.edges,
            encoding: manifest.encoding.clone(),
        }
    }
}

/// Return query capabilities for a collection.
pub fn capabilities(collection: &Collection) -> Capabilities {
    let payload_kind = PayloadKind::from(&collection.manifest().payload_plan);
    let mut predicates = vec![QueryPredicate::Bbox];
    if collection.supports_intersects_predicate() {
        predicates.push(QueryPredicate::Intersects);
    }
    let levels = if payload_kind == PayloadKind::None {
        vec![ResultLevel::Entry]
    } else {
        vec![ResultLevel::Feature, ResultLevel::Entry]
    };
    Capabilities {
        items: payload_kind == PayloadKind::FeatureJson,
        predicates,
        levels,
        payload_modes: vec![PayloadMode::None, PayloadMode::Summary, PayloadMode::Full],
    }
}

/// Search `/search`.
pub fn search_response(
    collection: &Collection,
    params: SearchParams,
) -> Result<SearchResponse, ServerError> {
    let options = SearchOptions::from_params(&params)?;
    let payload_mode = parse_payload_mode(params.payload.as_deref())?;
    let level = resolve_level(collection, parse_level(params.level.as_deref())?)?;
    let payload_kind = PayloadKind::from(&collection.manifest().payload_plan);
    let outcome = search_records(
        collection,
        &options.bbox,
        options.predicate,
        payload_mode,
        level,
        options.offset,
        options.limit,
    )?;
    Ok(SearchResponse {
        collection_id: collection.id().to_owned(),
        query: QueryInfo {
            bbox: options.bbox,
            predicate: options.predicate,
            level: Some(level),
            payload: Some(payload_mode),
            limit: options.limit,
            offset: options.offset,
        },
        payload_kind,
        number_matched: outcome.number_matched,
        number_returned: outcome.records.len(),
        matches: outcome.records,
    })
}

/// Search `/items`.
pub fn items_response(
    collection: &Collection,
    params: SearchParams,
) -> Result<FeatureCollectionResponse, ServerError> {
    if params.payload.is_some() {
        return Err(ServerError::UnsupportedQuery(
            "payload is only supported on /search".to_string(),
        ));
    }
    if params.level.is_some() {
        return Err(ServerError::UnsupportedQuery(
            "level is only supported on /search".to_string(),
        ));
    }
    if !matches!(
        collection.manifest().payload_plan,
        PayloadPlan::FeatureJson { .. }
    ) {
        return Err(ServerError::UnsupportedPayload(format!(
            "collection `{}` cannot serve /items because its artifact payload is not feature_json; use /search",
            collection.id()
        )));
    }
    let options = SearchOptions::from_params(&params)?;
    let outcome = search_records(
        collection,
        &options.bbox,
        options.predicate,
        PayloadMode::Full,
        ResultLevel::Feature,
        options.offset,
        options.limit,
    )?;
    let number_matched = outcome.number_matched;
    let features = outcome
        .records
        .into_iter()
        .filter_map(|record| match record.payload {
            Some(MatchPayload::FeatureJson { feature }) => feature,
            _ => None,
        })
        .collect::<Vec<_>>();
    Ok(FeatureCollectionResponse {
        kind: "FeatureCollection",
        number_matched,
        number_returned: features.len(),
        query: QueryInfo {
            bbox: options.bbox,
            predicate: options.predicate,
            level: None,
            payload: None,
            limit: options.limit,
            offset: options.offset,
        },
        features,
    })
}

struct SearchOptions {
    bbox: Vec<f64>,
    limit: usize,
    offset: usize,
    predicate: QueryPredicate,
}

impl SearchOptions {
    fn from_params(params: &SearchParams) -> Result<Self, ServerError> {
        let bbox = parse_bbox(params.bbox.as_deref())?;
        let (limit, offset) = limit_offset(params.limit.as_deref(), params.offset.as_deref())?;
        let predicate = parse_predicate(params.predicate.as_deref())?;
        Ok(Self {
            bbox,
            limit,
            offset,
            predicate,
        })
    }
}

fn resolve_level(
    collection: &Collection,
    requested: Option<ResultLevel>,
) -> Result<ResultLevel, ServerError> {
    let has_feature_refs = !matches!(collection.manifest().payload_plan, PayloadPlan::None);
    match requested {
        None => Ok(if has_feature_refs {
            ResultLevel::Feature
        } else {
            ResultLevel::Entry
        }),
        Some(ResultLevel::Feature) if !has_feature_refs => {
            Err(ServerError::UnsupportedLevel(format!(
                "collection `{}` stores no feature references; use level=entry",
                collection.id()
            )))
        }
        Some(level) => Ok(level),
    }
}

/// Matched-and-paged search result: the pre-pagination match count plus the
/// records of the requested page only.
///
/// `numberMatched` still comes from the materialized header/id list because
/// every record needs its identity anyway. A future count-only query
/// parameter (for example `count=only`) would skip that list entirely via
/// geo's `count_entries`.
struct SearchOutcome {
    number_matched: usize,
    records: Vec<MatchRecord>,
}

#[allow(clippy::too_many_arguments)]
fn search_records(
    collection: &Collection,
    bbox: &[f64],
    predicate: QueryPredicate,
    payload_mode: PayloadMode,
    level: ResultLevel,
    offset: usize,
    limit: usize,
) -> Result<SearchOutcome, ServerError> {
    let exact = predicate == QueryPredicate::Intersects;
    let index = collection.open_local_index()?;
    match index {
        GeoArtifactIndex::D2(index) => {
            if bbox.len() != 4 {
                return Err(ServerError::InvalidBbox(format!(
                    "2D collection `{}` expects bbox=minx,miny,maxx,maxy",
                    collection.id()
                )));
            }
            let query = Box2D::new(bbox[0], bbox[1], bbox[2], bbox[3]);
            let payload_plan = &collection.manifest().payload_plan;
            if matches!(payload_plan, PayloadPlan::None) {
                if exact {
                    return Err(ServerError::UnsupportedPredicate(format!(
                        "collection `{}` cannot apply predicate=intersects because its artifact has no geometry payload",
                        collection.id()
                    )));
                }
                return Ok(id_outcome(
                    index.search_entry_ids(query)?,
                    payload_mode,
                    offset,
                    limit,
                ));
            }
            if exact && !collection.supports_intersects_predicate() {
                return Err(ServerError::UnsupportedPredicate(format!(
                    "collection `{}` cannot apply predicate=intersects from its artifact payload",
                    collection.id()
                )));
            }
            // Header path: RowRef/RowWkb identity lives in the fixed payload
            // prefix, so a bbox search sorts, dedupes, and pages without
            // reading payload bodies — bodies are fetched for the page only.
            // predicate=intersects needs every match's geometry up front, and
            // FeatureJson stores identity inside the JSON body; both keep the
            // full-decode path (payload work is still page-only below).
            if !exact && matches!(payload_plan, PayloadPlan::RowRef | PayloadPlan::RowWkb) {
                let headers = index.search_match_headers(query)?;
                return header_outcome(
                    headers,
                    payload_mode,
                    level,
                    offset,
                    limit,
                    payload_plan,
                    |page| index.fetch_matches(page),
                );
            }
            let mut matches = index.search_matches(query)?;
            if exact {
                matches = index
                    .filter_matches(
                        matches,
                        GeoQuery2D::box2d(query),
                        SpatialPredicate::Intersects,
                        NonPlanarExactPolicy::Reject,
                    )
                    .map_err(ServerError::from_geo)?;
            }
            Ok(match_outcome(matches, payload_mode, level, offset, limit))
        }
        GeoArtifactIndex::D3(index) => {
            if bbox.len() != 6 {
                return Err(ServerError::InvalidBbox(format!(
                    "3D collection `{}` expects bbox=minx,miny,minz,maxx,maxy,maxz",
                    collection.id()
                )));
            }
            if exact {
                return Err(ServerError::UnsupportedPredicate(format!(
                    "collection `{}` is 3D; predicate=intersects is only supported for 2D artifacts in this server",
                    collection.id()
                )));
            }
            let query = Box3D::new(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
            let payload_plan = &collection.manifest().payload_plan;
            if matches!(payload_plan, PayloadPlan::None) {
                return Ok(id_outcome(
                    index.search_entry_ids(query)?,
                    payload_mode,
                    offset,
                    limit,
                ));
            }
            if matches!(payload_plan, PayloadPlan::RowRef | PayloadPlan::RowWkb) {
                let headers = index.search_match_headers(query)?;
                return header_outcome(
                    headers,
                    payload_mode,
                    level,
                    offset,
                    limit,
                    payload_plan,
                    |page| index.fetch_matches(page),
                );
            }
            let matches = index.search_matches(query)?;
            Ok(match_outcome(matches, payload_mode, level, offset, limit))
        }
    }
}

/// Page an id-only (payload-less) result set.
fn id_outcome(
    mut ids: Vec<usize>,
    payload_mode: PayloadMode,
    offset: usize,
    limit: usize,
) -> SearchOutcome {
    ids.sort_unstable();
    ids.dedup();
    let number_matched = ids.len();
    let records = paginate(&mut ids, offset, limit)
        .into_iter()
        .map(|id| MatchRecord {
            entry_id: id,
            feature_ref: None,
            payload: match_payload_none(payload_mode),
        })
        .collect();
    SearchOutcome {
        number_matched,
        records,
    }
}

/// Sort, dedupe, and page fully-decoded matches; record mapping (base64/JSON
/// serialization) runs for the page only.
fn match_outcome(
    mut matches: Vec<GeoMatch>,
    payload_mode: PayloadMode,
    level: ResultLevel,
    offset: usize,
    limit: usize,
) -> SearchOutcome {
    GeoMatch::sort_by_entry(&mut matches);
    if matches!(level, ResultLevel::Feature) {
        GeoMatch::dedupe_by_feature(&mut matches);
    }
    let number_matched = matches.len();
    let records = paginate(&mut matches, offset, limit)
        .into_iter()
        .map(|m| match_record(m.entry_id, Some(m.feature), m.payload, payload_mode, level))
        .collect();
    SearchOutcome {
        number_matched,
        records,
    }
}

/// Sort, dedupe, and page match headers; payload bodies are fetched only for
/// the page, and only when `payload=full` needs them.
fn header_outcome(
    mut headers: Vec<GeoMatchHeader>,
    payload_mode: PayloadMode,
    level: ResultLevel,
    offset: usize,
    limit: usize,
    plan: &PayloadPlan,
    fetch: impl FnOnce(&[GeoMatchHeader]) -> Result<Vec<GeoMatch>, packed_spatial_index_geo::GeoError>,
) -> Result<SearchOutcome, ServerError> {
    GeoMatchHeader::sort_by_entry(&mut headers);
    if matches!(level, ResultLevel::Feature) {
        GeoMatchHeader::dedupe_by_feature(&mut headers);
    }
    let number_matched = headers.len();
    let page = paginate(&mut headers, offset, limit);
    let records = if payload_mode == PayloadMode::Full {
        fetch(&page)?
            .into_iter()
            .map(|m| match_record(m.entry_id, Some(m.feature), m.payload, payload_mode, level))
            .collect()
    } else {
        page.into_iter()
            .map(|header| header_record(header, payload_mode, plan))
            .collect()
    };
    Ok(SearchOutcome {
        number_matched,
        records,
    })
}

/// Build a record straight from a header — no payload body was read, so
/// summary mode derives `byteLength` from the header's payload length.
fn header_record(
    header: GeoMatchHeader,
    payload_mode: PayloadMode,
    plan: &PayloadPlan,
) -> MatchRecord {
    let payload = match (payload_mode, plan) {
        (PayloadMode::None, _) => None,
        (_, PayloadPlan::RowRef) => Some(MatchPayload::RowRef),
        (_, PayloadPlan::RowWkb) => Some(MatchPayload::RowWkb {
            byte_length: header.body_byte_len().unwrap_or(0),
            wkb_base64: None,
        }),
        // The header search rejects every other plan up front.
        _ => None,
    };
    MatchRecord {
        entry_id: header.entry_id,
        feature_ref: Some(header.feature.into()),
        payload,
    }
}

fn match_record(
    entry_id: usize,
    feature_ref: Option<FeatureRef>,
    payload: GeoPayload,
    payload_mode: PayloadMode,
    level: ResultLevel,
) -> MatchRecord {
    let payload = match (payload_mode, payload) {
        (PayloadMode::None, _) => None,
        (_, GeoPayload::RowRef) => Some(MatchPayload::RowRef),
        (mode, GeoPayload::RowWkb(wkb)) => Some(MatchPayload::RowWkb {
            byte_length: wkb.len(),
            wkb_base64: (mode == PayloadMode::Full).then(|| STANDARD.encode(wkb)),
        }),
        (mode, GeoPayload::FeatureJson(feature)) => Some(MatchPayload::FeatureJson {
            feature: (mode == PayloadMode::Full).then(|| public_feature_json(feature)),
        }),
    };
    let feature_ref = feature_ref.map(|mut feature| {
        // A representative part number is meaningless once split entries
        // collapse into one feature-level record.
        if matches!(level, ResultLevel::Feature) {
            feature.part = None;
        }
        FeatureRefRecord::from(feature)
    });
    MatchRecord {
        entry_id,
        feature_ref,
        payload,
    }
}

fn public_feature_json(mut feature: serde_json::Value) -> serde_json::Value {
    if let Some(object) = feature.as_object_mut() {
        object.remove("feature_ref");
    }
    feature
}

fn match_payload_none(payload_mode: PayloadMode) -> Option<MatchPayload> {
    (payload_mode != PayloadMode::None).then_some(MatchPayload::None)
}

fn parse_bbox(raw: Option<&str>) -> Result<Vec<f64>, ServerError> {
    let raw = raw.ok_or_else(|| ServerError::InvalidBbox("bbox is required".to_string()))?;
    let mut values = Vec::new();
    for part in raw.split(',') {
        let value = part.trim().parse::<f64>().map_err(|_| {
            ServerError::InvalidBbox(format!("bbox value `{}` is not a number", part.trim()))
        })?;
        if !value.is_finite() {
            return Err(ServerError::InvalidBbox(format!(
                "bbox value `{}` is not finite",
                part.trim()
            )));
        }
        values.push(value);
    }
    if !matches!(values.len(), 4 | 6) {
        return Err(ServerError::InvalidBbox(
            "bbox must contain either 4 numbers (2D) or 6 numbers (3D)".to_string(),
        ));
    }
    if values.len() == 4 && (values[0] > values[2] || values[1] > values[3]) {
        return Err(ServerError::InvalidBbox(
            "2D bbox minimums must be <= maximums".to_string(),
        ));
    }
    if values.len() == 6
        && (values[0] > values[3] || values[1] > values[4] || values[2] > values[5])
    {
        return Err(ServerError::InvalidBbox(
            "3D bbox minimums must be <= maximums".to_string(),
        ));
    }
    Ok(values)
}

fn limit_offset(limit: Option<&str>, offset: Option<&str>) -> Result<(usize, usize), ServerError> {
    let limit = match limit {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|_| ServerError::InvalidLimit("limit must be an integer".to_string()))?,
        None => DEFAULT_LIMIT,
    };
    if limit == 0 || limit > MAX_LIMIT {
        return Err(ServerError::InvalidLimit(format!(
            "limit must be between 1 and {MAX_LIMIT}"
        )));
    }
    let offset = match offset {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|_| ServerError::InvalidOffset("offset must be an integer".to_string()))?,
        None => 0,
    };
    Ok((limit, offset))
}

fn parse_predicate(raw: Option<&str>) -> Result<QueryPredicate, ServerError> {
    match raw {
        None | Some("") | Some("bbox") => Ok(QueryPredicate::Bbox),
        Some("intersects") => Ok(QueryPredicate::Intersects),
        Some(_) => Err(ServerError::InvalidPredicate(
            "predicate must be bbox or intersects".to_string(),
        )),
    }
}

fn parse_level(raw: Option<&str>) -> Result<Option<ResultLevel>, ServerError> {
    match raw {
        None | Some("") => Ok(None),
        Some("feature") => Ok(Some(ResultLevel::Feature)),
        Some("entry") => Ok(Some(ResultLevel::Entry)),
        Some(_) => Err(ServerError::InvalidLevel(
            "level must be feature or entry".to_string(),
        )),
    }
}

fn parse_payload_mode(raw: Option<&str>) -> Result<PayloadMode, ServerError> {
    match raw {
        None | Some("") | Some("summary") => Ok(PayloadMode::Summary),
        Some("none") => Ok(PayloadMode::None),
        Some("full") => Ok(PayloadMode::Full),
        Some(_) => Err(ServerError::InvalidPayload(
            "payload must be none, summary, or full".to_string(),
        )),
    }
}

fn paginate<T>(records: &mut Vec<T>, offset: usize, limit: usize) -> Vec<T> {
    if offset >= records.len() {
        return Vec::new();
    }
    let end = records.len().min(offset.saturating_add(limit));
    records.drain(offset..end).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bbox_rejects_wrong_arity() {
        assert!(parse_bbox(Some("1,2,3")).is_err());
    }

    #[test]
    fn parse_bbox_rejects_inverted_ranges() {
        assert!(parse_bbox(Some("10,0,0,2")).is_err());
        assert!(parse_bbox(Some("0,0,5,1,1,4")).is_err());
    }

    #[test]
    fn limit_defaults_and_caps() {
        assert_eq!(limit_offset(None, None).unwrap(), (DEFAULT_LIMIT, 0));
        assert!(limit_offset(Some("0"), None).is_err());
        assert!(limit_offset(Some(&(MAX_LIMIT + 1).to_string()), None).is_err());
    }

    #[test]
    fn payload_mode_defaults_to_summary() {
        assert_eq!(parse_payload_mode(None).unwrap(), PayloadMode::Summary);
        assert_eq!(parse_payload_mode(Some("none")).unwrap(), PayloadMode::None);
        assert!(parse_payload_mode(Some("yes")).is_err());
    }

    #[test]
    fn predicate_defaults_to_bbox() {
        assert_eq!(parse_predicate(None).unwrap(), QueryPredicate::Bbox);
        assert_eq!(
            parse_predicate(Some("intersects")).unwrap(),
            QueryPredicate::Intersects
        );
        assert!(parse_predicate(Some("exact")).is_err());
    }

    #[test]
    fn level_is_optional_until_resolved() {
        assert_eq!(parse_level(None).unwrap(), None);
        assert_eq!(
            parse_level(Some("entry")).unwrap(),
            Some(ResultLevel::Entry)
        );
        assert!(parse_level(Some("item")).is_err());
    }
}
