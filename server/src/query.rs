use std::collections::HashSet;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use packed_spatial_index_geo::{
    Box2D, Box3D, CoordinateDims, CrsInfo, EdgeModel, FeatureRef, GeoArtifactIndex, GeoPayload,
    GeoQuery2D, GeometryEncoding, NonPlanarExactPolicy, PayloadPlan, SpatialPredicate,
    StoragePrecision,
};
use serde::{Deserialize, Serialize};

use crate::{Collection, ServerError};

const DEFAULT_LIMIT: usize = 100;
const MAX_LIMIT: usize = 10_000;

/// Query parameters accepted by `/items`.
#[derive(Debug, Deserialize)]
pub struct ItemsParams {
    /// Query bbox as comma-separated numbers.
    #[serde(default)]
    pub bbox: Option<String>,
    /// Maximum returned features.
    #[serde(default)]
    pub limit: Option<String>,
    /// Number of matched features to skip.
    #[serde(default)]
    pub offset: Option<String>,
    /// Whether to apply exact post-filtering when supported.
    #[serde(default)]
    pub exact: Option<String>,
    /// Payload materialization is only accepted on `/hits`.
    #[serde(default)]
    pub payload: Option<String>,
}

/// Query parameters accepted by `/hits`.
#[derive(Debug, Deserialize)]
pub struct HitsParams {
    /// Query bbox as comma-separated numbers.
    #[serde(default)]
    pub bbox: Option<String>,
    /// Maximum returned hits.
    #[serde(default)]
    pub limit: Option<String>,
    /// Number of matched hits to skip.
    #[serde(default)]
    pub offset: Option<String>,
    /// Whether to apply exact post-filtering when supported.
    #[serde(default)]
    pub exact: Option<String>,
    /// Payload materialization mode.
    #[serde(default)]
    pub payload: Option<String>,
}

/// Payload materialization mode for `/hits`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadMode {
    /// Omit the payload object from each hit.
    None,
    /// Return payload kind and cheap metadata only.
    #[default]
    Summary,
    /// Return full payload values where the artifact stores them.
    Full,
}

/// Normalized query information included in search responses.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryInfo {
    /// Parsed query bbox.
    pub bbox: Vec<f64>,
    /// Whether exact filtering was requested.
    pub exact: bool,
    /// Whether exact filtering was actually applied.
    pub exact_applied: bool,
}

/// Collection capabilities exposed through the HTTP API.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    /// Spatial bbox search is available.
    pub bbox_search: bool,
    /// `/items` can return GeoJSON FeatureCollection without source read-back.
    pub feature_json_items: bool,
    /// `/hits` can return artifact hits.
    pub hits: bool,
    /// Exact 2D filtering can run from artifact payloads.
    pub exact_filter: bool,
    /// Original source read-back is wired.
    pub source_read_back: bool,
    /// Artifact has WKB geometry payloads.
    pub row_wkb_payload: bool,
    /// Artifact has row-reference payloads.
    pub row_ref_payload: bool,
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
    /// Artifact payload plan.
    pub payload_plan: PayloadPlan,
    /// Packed node size.
    pub node_size: usize,
    /// Whether the artifact has a payload section.
    pub has_payload: bool,
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

/// `/hits` response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HitsResponse {
    /// Collection id.
    pub collection_id: String,
    /// Normalized query information.
    pub query: QueryInfo,
    /// Total matched hits before pagination.
    #[serde(rename = "numberMatched")]
    pub number_matched: usize,
    /// Returned hits after pagination.
    #[serde(rename = "numberReturned")]
    pub number_returned: usize,
    /// Applied offset.
    pub offset: usize,
    /// Applied limit.
    pub limit: usize,
    /// Artifact payload plan.
    pub payload_plan: PayloadPlan,
    /// Requested payload materialization mode.
    pub payload_mode: PayloadMode,
    /// Returned hits.
    pub hits: Vec<HitRecord>,
}

/// One `/hits` record.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HitRecord {
    /// Compact index entry id in the artifact.
    pub entry_id: usize,
    /// Source feature ref when the payload contains one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_ref: Option<FeatureRefRecord>,
    /// Payload summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<HitPayload>,
}

/// Payload summary for a hit.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HitPayload {
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
    /// Geometry part when a source feature expands into multiple index entries.
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
    #[serde(rename = "numberMatched")]
    pub number_matched: usize,
    /// Returned features after pagination.
    #[serde(rename = "numberReturned")]
    pub number_returned: usize,
    /// Applied offset.
    pub offset: usize,
    /// Applied limit.
    pub limit: usize,
    /// Normalized query information.
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
            payload_plan: manifest.payload_plan.clone(),
            node_size: collection.node_size(),
            has_payload: collection.has_payload(),
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
            source_format: manifest.source_format.clone(),
            source_fingerprint: manifest.source_fingerprint.clone(),
            selected_column: manifest.selected_column.clone(),
            crs: manifest.crs.clone(),
            edges: manifest.edges,
            encoding: manifest.encoding.clone(),
        }
    }
}

/// Return capability flags for a collection.
pub fn capabilities(collection: &Collection) -> Capabilities {
    let payload = &collection.manifest().payload_plan;
    Capabilities {
        bbox_search: true,
        feature_json_items: matches!(payload, PayloadPlan::FeatureJson { .. }),
        hits: true,
        exact_filter: collection.supports_exact_filter(),
        source_read_back: false,
        row_wkb_payload: matches!(payload, PayloadPlan::RowWkb),
        row_ref_payload: matches!(payload, PayloadPlan::RowRef),
    }
}

/// Search `/hits`.
pub fn hits_response(
    collection: &Collection,
    params: HitsParams,
) -> Result<HitsResponse, ServerError> {
    let options = SearchOptions::from_parts(
        params.bbox.as_deref(),
        params.limit.as_deref(),
        params.offset.as_deref(),
        params.exact.as_deref(),
    )?;
    let payload_mode = parse_payload_mode(params.payload.as_deref())?;
    let payload_plan = collection.manifest().payload_plan.clone();
    let SearchOutcome {
        mut records,
        exact_applied,
    } = search_records(
        collection,
        &options.bbox,
        options.exact,
        payload_mode,
        SearchGranularity::IndexEntries,
    )?;
    let number_matched = records.len();
    let records = paginate(&mut records, options.offset, options.limit);
    Ok(HitsResponse {
        collection_id: collection.id().to_owned(),
        query: QueryInfo {
            bbox: options.bbox,
            exact: options.exact,
            exact_applied,
        },
        number_matched,
        number_returned: records.len(),
        offset: options.offset,
        limit: options.limit,
        payload_plan,
        payload_mode,
        hits: records,
    })
}

/// Search `/items`.
pub fn items_response(
    collection: &Collection,
    params: ItemsParams,
) -> Result<FeatureCollectionResponse, ServerError> {
    if params.payload.is_some() {
        return Err(ServerError::InvalidPayload(
            "payload is only supported on /hits".to_string(),
        ));
    }
    if !matches!(
        collection.manifest().payload_plan,
        PayloadPlan::FeatureJson { .. }
    ) {
        return Err(ServerError::UnsupportedPayload(format!(
            "collection `{}` cannot serve /items because its artifact payload is not FeatureJson; use /hits",
            collection.id()
        )));
    }
    let options = SearchOptions::from_parts(
        params.bbox.as_deref(),
        params.limit.as_deref(),
        params.offset.as_deref(),
        params.exact.as_deref(),
    )?;
    let SearchOutcome {
        mut records,
        exact_applied,
    } = search_records(
        collection,
        &options.bbox,
        options.exact,
        PayloadMode::Full,
        SearchGranularity::SourceFeatures,
    )?;
    let number_matched = records.len();
    let records = paginate(&mut records, options.offset, options.limit);
    let features = records
        .into_iter()
        .filter_map(|record| match record.payload {
            Some(HitPayload::FeatureJson { feature }) => feature,
            _ => None,
        })
        .collect::<Vec<_>>();
    Ok(FeatureCollectionResponse {
        kind: "FeatureCollection",
        number_matched,
        number_returned: features.len(),
        offset: options.offset,
        limit: options.limit,
        query: QueryInfo {
            bbox: options.bbox,
            exact: options.exact,
            exact_applied,
        },
        features,
    })
}

struct SearchOutcome {
    records: Vec<HitRecord>,
    exact_applied: bool,
}

struct SearchOptions {
    bbox: Vec<f64>,
    limit: usize,
    offset: usize,
    exact: bool,
}

impl SearchOptions {
    fn from_parts(
        bbox: Option<&str>,
        limit: Option<&str>,
        offset: Option<&str>,
        exact: Option<&str>,
    ) -> Result<Self, ServerError> {
        let bbox = parse_bbox(bbox)?;
        let (limit, offset) = limit_offset(limit, offset)?;
        let exact = parse_exact(exact)?;
        Ok(Self {
            bbox,
            limit,
            offset,
            exact,
        })
    }
}

#[derive(Clone, Copy)]
enum SearchGranularity {
    IndexEntries,
    SourceFeatures,
}

fn search_records(
    collection: &Collection,
    bbox: &[f64],
    exact: bool,
    payload_mode: PayloadMode,
    granularity: SearchGranularity,
) -> Result<SearchOutcome, ServerError> {
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
                    return Err(ServerError::ExactFilterUnavailable(format!(
                        "collection `{}` cannot exact-filter because its artifact has no geometry payload",
                        collection.id()
                    )));
                }
                let mut items = index.search_items(query)?;
                items.sort_unstable();
                items.dedup();
                return Ok(SearchOutcome {
                    records: items
                        .into_iter()
                        .map(|item| HitRecord {
                            entry_id: item,
                            feature_ref: None,
                            payload: hit_payload_none(payload_mode),
                        })
                        .collect(),
                    exact_applied: false,
                });
            }
            if exact && !collection.supports_exact_filter() {
                return Err(ServerError::ExactFilterUnavailable(format!(
                    "collection `{}` cannot exact-filter from its artifact payload",
                    collection.id()
                )));
            }
            let mut hits = index.search_hits(query)?;
            if exact {
                hits = index.filter_hits(
                    hits,
                    GeoQuery2D::box2d(query),
                    SpatialPredicate::Intersects,
                    NonPlanarExactPolicy::Reject,
                )?;
            }
            sort_hits(&mut hits);
            if matches!(granularity, SearchGranularity::SourceFeatures) {
                dedupe_feature_hits(&mut hits);
            }
            Ok(SearchOutcome {
                records: hits
                    .into_iter()
                    .map(|hit| hit_record(hit.item, Some(hit.feature), hit.payload, payload_mode))
                    .collect(),
                exact_applied: exact,
            })
        }
        GeoArtifactIndex::D3(index) => {
            if bbox.len() != 6 {
                return Err(ServerError::InvalidBbox(format!(
                    "3D collection `{}` expects bbox=minx,miny,minz,maxx,maxy,maxz",
                    collection.id()
                )));
            }
            if exact {
                return Err(ServerError::ExactFilterUnavailable(format!(
                    "collection `{}` is 3D; exact filtering is only supported for 2D artifacts in this server",
                    collection.id()
                )));
            }
            let query = Box3D::new(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
            if matches!(collection.manifest().payload_plan, PayloadPlan::None) {
                let mut items = index.search_items(query)?;
                items.sort_unstable();
                items.dedup();
                return Ok(SearchOutcome {
                    records: items
                        .into_iter()
                        .map(|item| HitRecord {
                            entry_id: item,
                            feature_ref: None,
                            payload: hit_payload_none(payload_mode),
                        })
                        .collect(),
                    exact_applied: false,
                });
            }
            let mut hits = index.search_hits(query)?;
            sort_hits(&mut hits);
            if matches!(granularity, SearchGranularity::SourceFeatures) {
                dedupe_feature_hits(&mut hits);
            }
            Ok(SearchOutcome {
                records: hits
                    .into_iter()
                    .map(|hit| hit_record(hit.item, Some(hit.feature), hit.payload, payload_mode))
                    .collect(),
                exact_applied: false,
            })
        }
    }
}

fn hit_record(
    item: usize,
    feature_ref: Option<FeatureRef>,
    payload: GeoPayload,
    payload_mode: PayloadMode,
) -> HitRecord {
    let payload = match (payload_mode, payload) {
        (PayloadMode::None, _) => None,
        (_, GeoPayload::RowRef) => Some(HitPayload::RowRef),
        (mode, GeoPayload::RowWkb(wkb)) => Some(HitPayload::RowWkb {
            byte_length: wkb.len(),
            wkb_base64: (mode == PayloadMode::Full).then(|| STANDARD.encode(wkb)),
        }),
        (mode, GeoPayload::FeatureJson(feature)) => Some(HitPayload::FeatureJson {
            feature: (mode == PayloadMode::Full).then(|| public_feature_json(feature)),
        }),
    };
    HitRecord {
        entry_id: item,
        feature_ref: feature_ref.map(Into::into),
        payload,
    }
}

fn public_feature_json(mut feature: serde_json::Value) -> serde_json::Value {
    if let Some(object) = feature.as_object_mut() {
        object.remove("feature_ref");
    }
    feature
}

fn hit_payload_none(payload_mode: PayloadMode) -> Option<HitPayload> {
    (payload_mode != PayloadMode::None).then_some(HitPayload::None)
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

fn parse_exact(raw: Option<&str>) -> Result<bool, ServerError> {
    match raw {
        None => Ok(false),
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(_) => Err(ServerError::InvalidExact(
            "exact must be true or false".to_string(),
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

fn paginate(records: &mut Vec<HitRecord>, offset: usize, limit: usize) -> Vec<HitRecord> {
    if offset >= records.len() {
        return Vec::new();
    }
    let end = records.len().min(offset.saturating_add(limit));
    records.drain(offset..end).collect()
}

fn sort_hits(hits: &mut [packed_spatial_index_geo::GeoHit]) {
    hits.sort_by(|a, b| {
        feature_sort_key(&a.feature)
            .cmp(&feature_sort_key(&b.feature))
            .then_with(|| a.item.cmp(&b.item))
    });
}

fn dedupe_feature_hits(hits: &mut Vec<packed_spatial_index_geo::GeoHit>) {
    let mut seen = HashSet::new();
    hits.retain(|hit| seen.insert(feature_identity(&hit.feature)));
}

fn feature_identity(feature: &FeatureRef) -> (u64, Option<u32>, Option<u32>, Option<String>) {
    (
        feature.row_number,
        feature.row_group,
        feature.row_in_group,
        feature.feature_id.clone(),
    )
}

fn feature_sort_key(
    feature: &FeatureRef,
) -> (u64, Option<u32>, Option<u32>, Option<u16>, Option<String>) {
    (
        feature.row_number,
        feature.row_group,
        feature.row_in_group,
        feature.part,
        feature.feature_id.clone(),
    )
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
}
