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

/// Query parameters shared by `/items` and `/hits`.
#[derive(Debug, Deserialize)]
pub struct SearchParams {
    /// Query bbox as comma-separated numbers.
    pub bbox: String,
    /// Maximum returned features.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Number of matched features to skip.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Whether to apply exact post-filtering when supported.
    #[serde(default)]
    pub exact: bool,
    /// Whether `/hits` should include payload bytes/values.
    #[serde(default)]
    pub include_payload: bool,
}

/// Collection capabilities exposed through the HTTP API.
#[derive(Debug, Serialize)]
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
pub struct CollectionSummary {
    /// Collection id.
    pub id: String,
    /// Optional title.
    pub title: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Number of indexed items.
    pub item_count: usize,
    /// Number of unique source features represented in the artifact.
    pub feature_count: usize,
    /// Number of index entries.
    pub index_entry_count: usize,
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
pub struct HitsResponse {
    /// Collection id.
    pub collection_id: String,
    /// Query bbox.
    pub bbox: Vec<f64>,
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
    /// Returned hits.
    pub hits: Vec<HitRecord>,
}

/// One `/hits` record.
#[derive(Debug, Serialize)]
pub struct HitRecord {
    /// Compact item id in the artifact.
    pub item: usize,
    /// Source feature ref when the payload contains one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_ref: Option<FeatureRef>,
    /// Payload summary.
    pub payload: HitPayload,
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
        /// Base64 WKB bytes, present only when `include_payload=true`.
        #[serde(skip_serializing_if = "Option::is_none")]
        wkb_base64: Option<String>,
    },
    /// Payload stores a GeoJSON Feature.
    FeatureJson {
        /// GeoJSON Feature, present only when `include_payload=true`.
        #[serde(skip_serializing_if = "Option::is_none")]
        feature: Option<serde_json::Value>,
    },
}

/// GeoJSON FeatureCollection response from `/items`.
#[derive(Debug, Serialize)]
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
}

impl CollectionSummary {
    /// Build a collection summary.
    pub fn new(collection: &Collection) -> Self {
        let manifest = collection.manifest();
        Self {
            id: collection.id.clone(),
            title: collection.title.clone(),
            description: collection.description.clone(),
            item_count: collection.num_items(),
            feature_count: manifest.feature_count,
            index_entry_count: manifest.index_entry_count,
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
    params: SearchParams,
) -> Result<HitsResponse, ServerError> {
    let bbox = parse_bbox(&params.bbox)?;
    let (limit, offset) = limit_offset(params.limit, params.offset)?;
    let payload_plan = collection.manifest().payload_plan.clone();
    let mut records = search_records(collection, &bbox, params.exact, params.include_payload)?;
    let number_matched = records.len();
    let records = paginate(&mut records, offset, limit);
    Ok(HitsResponse {
        collection_id: collection.id.clone(),
        bbox,
        number_matched,
        number_returned: records.len(),
        offset,
        limit,
        payload_plan,
        hits: records,
    })
}

/// Search `/items`.
pub fn items_response(
    collection: &Collection,
    params: SearchParams,
) -> Result<FeatureCollectionResponse, ServerError> {
    if !matches!(
        collection.manifest().payload_plan,
        PayloadPlan::FeatureJson { .. }
    ) {
        return Err(ServerError::Unsupported(format!(
            "collection `{}` cannot serve /items because its artifact payload is not FeatureJson; use /hits",
            collection.id
        )));
    }
    let bbox = parse_bbox(&params.bbox)?;
    let (limit, offset) = limit_offset(params.limit, params.offset)?;
    let mut records = search_records(collection, &bbox, params.exact, true)?;
    let number_matched = records.len();
    let records = paginate(&mut records, offset, limit);
    let features = records
        .into_iter()
        .filter_map(|record| match record.payload {
            HitPayload::FeatureJson { feature } => feature,
            _ => None,
        })
        .collect::<Vec<_>>();
    Ok(FeatureCollectionResponse {
        kind: "FeatureCollection",
        number_matched,
        number_returned: features.len(),
        offset,
        limit,
        features,
    })
}

fn search_records(
    collection: &Collection,
    bbox: &[f64],
    exact: bool,
    include_payload: bool,
) -> Result<Vec<HitRecord>, ServerError> {
    let index = collection.open_index()?;
    match index {
        GeoArtifactIndex::D2(index) => {
            if bbox.len() != 4 {
                return Err(ServerError::BadRequest(format!(
                    "2D collection `{}` expects bbox=minx,miny,maxx,maxy",
                    collection.id
                )));
            }
            let query = Box2D::new(bbox[0], bbox[1], bbox[2], bbox[3]);
            let payload_plan = &collection.manifest().payload_plan;
            if matches!(payload_plan, PayloadPlan::None) {
                if exact {
                    return Err(ServerError::Unsupported(format!(
                        "collection `{}` cannot exact-filter because its artifact has no geometry payload",
                        collection.id
                    )));
                }
                let mut items = index.search_items(query)?;
                items.sort_unstable();
                items.dedup();
                return Ok(items
                    .into_iter()
                    .map(|item| HitRecord {
                        item,
                        feature_ref: None,
                        payload: HitPayload::None,
                    })
                    .collect());
            }
            if exact && !collection.supports_exact_filter() {
                return Err(ServerError::Unsupported(format!(
                    "collection `{}` cannot exact-filter from its artifact payload",
                    collection.id
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
            dedupe_sort_hits(&mut hits);
            Ok(hits
                .into_iter()
                .map(|hit| hit_record(hit.item, Some(hit.feature), hit.payload, include_payload))
                .collect())
        }
        GeoArtifactIndex::D3(index) => {
            if bbox.len() != 6 {
                return Err(ServerError::BadRequest(format!(
                    "3D collection `{}` expects bbox=minx,miny,minz,maxx,maxy,maxz",
                    collection.id
                )));
            }
            if exact {
                return Err(ServerError::Unsupported(format!(
                    "collection `{}` is 3D; exact filtering is only supported for 2D artifacts in this server",
                    collection.id
                )));
            }
            let query = Box3D::new(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
            if matches!(collection.manifest().payload_plan, PayloadPlan::None) {
                let mut items = index.search_items(query)?;
                items.sort_unstable();
                items.dedup();
                return Ok(items
                    .into_iter()
                    .map(|item| HitRecord {
                        item,
                        feature_ref: None,
                        payload: HitPayload::None,
                    })
                    .collect());
            }
            let mut hits = index.search_hits(query)?;
            dedupe_sort_hits(&mut hits);
            Ok(hits
                .into_iter()
                .map(|hit| hit_record(hit.item, Some(hit.feature), hit.payload, include_payload))
                .collect())
        }
    }
}

fn hit_record(
    item: usize,
    feature_ref: Option<FeatureRef>,
    payload: GeoPayload,
    include_payload: bool,
) -> HitRecord {
    let payload = match payload {
        GeoPayload::RowRef => HitPayload::RowRef,
        GeoPayload::RowWkb(wkb) => HitPayload::RowWkb {
            wkb_base64: include_payload.then(|| STANDARD.encode(wkb)),
        },
        GeoPayload::FeatureJson(feature) => HitPayload::FeatureJson {
            feature: include_payload.then_some(feature),
        },
    };
    HitRecord {
        item,
        feature_ref,
        payload,
    }
}

fn parse_bbox(raw: &str) -> Result<Vec<f64>, ServerError> {
    let mut values = Vec::new();
    for part in raw.split(',') {
        let value = part.trim().parse::<f64>().map_err(|_| {
            ServerError::BadRequest(format!("bbox value `{}` is not a number", part.trim()))
        })?;
        if !value.is_finite() {
            return Err(ServerError::BadRequest(format!(
                "bbox value `{}` is not finite",
                part.trim()
            )));
        }
        values.push(value);
    }
    if !matches!(values.len(), 4 | 6) {
        return Err(ServerError::BadRequest(
            "bbox must contain either 4 numbers (2D) or 6 numbers (3D)".to_string(),
        ));
    }
    if values.len() == 4 && (values[0] > values[2] || values[1] > values[3]) {
        return Err(ServerError::BadRequest(
            "2D bbox minimums must be <= maximums".to_string(),
        ));
    }
    if values.len() == 6
        && (values[0] > values[3] || values[1] > values[4] || values[2] > values[5])
    {
        return Err(ServerError::BadRequest(
            "3D bbox minimums must be <= maximums".to_string(),
        ));
    }
    Ok(values)
}

fn limit_offset(
    limit: Option<usize>,
    offset: Option<usize>,
) -> Result<(usize, usize), ServerError> {
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    if limit == 0 || limit > MAX_LIMIT {
        return Err(ServerError::BadRequest(format!(
            "limit must be between 1 and {MAX_LIMIT}"
        )));
    }
    Ok((limit, offset.unwrap_or(0)))
}

fn paginate(records: &mut Vec<HitRecord>, offset: usize, limit: usize) -> Vec<HitRecord> {
    if offset >= records.len() {
        return Vec::new();
    }
    let end = records.len().min(offset.saturating_add(limit));
    records.drain(offset..end).collect()
}

fn dedupe_sort_hits(hits: &mut Vec<packed_spatial_index_geo::GeoHit>) {
    hits.sort_by(|a, b| {
        feature_sort_key(&a.feature)
            .cmp(&feature_sort_key(&b.feature))
            .then_with(|| a.item.cmp(&b.item))
    });
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
        assert!(parse_bbox("1,2,3").is_err());
    }

    #[test]
    fn parse_bbox_rejects_inverted_ranges() {
        assert!(parse_bbox("10,0,0,2").is_err());
        assert!(parse_bbox("0,0,5,1,1,4").is_err());
    }

    #[test]
    fn limit_defaults_and_caps() {
        assert_eq!(limit_offset(None, None).unwrap(), (DEFAULT_LIMIT, 0));
        assert!(limit_offset(Some(0), None).is_err());
        assert!(limit_offset(Some(MAX_LIMIT + 1), None).is_err());
    }
}
