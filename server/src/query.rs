use base64::{Engine as _, engine::general_purpose::STANDARD};
use packed_spatial_index_geo::{
    Box2D, Box3D, CoordinateDims, CrsInfo, EdgeModel, FeatureRef, Frustum3D, GeoArtifactIndex,
    GeoArtifactIndex2D, GeoArtifactIndex3D, GeoError, GeoMatch, GeoMatchHeader, GeoMatchHeaderPage,
    GeoPayload, GeoQuery2D, GeoQuery3D, GeometryEncoding, IdentityMode, LevelError,
    NonPlanarExactPolicy, PayloadMode, PayloadPlan, RangeReader, ResultLevel, SpatialPredicate,
    StoragePrecision, needs_payload_bodies, public_feature_json, resolve_identity,
    stores_feature_ids,
};
use serde::{Deserialize, Serialize};

use crate::{Collection, ServerError};

const DEFAULT_LIMIT: usize = 100;
const MAX_LIMIT: usize = 10_000;

/// Query parameters accepted by `/search` and `/items`.
///
/// `/items` rejects `level`, `payload`, and `identity`; both endpoints share
/// the rest. Unknown parameters are rejected rather than ignored: a misspelled
/// name would otherwise resolve to a default and silently answer a different
/// question.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchParams {
    /// Query bbox as comma-separated numbers.
    #[serde(default)]
    pub bbox: Option<String>,
    /// Query view frustum as 24 comma-separated numbers: six inward-pointing
    /// planes `a,b,c,d`. Mutually exclusive with `bbox`.
    #[serde(default)]
    pub frustum: Option<String>,
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
    /// Source-identity mode for `/search`.
    #[serde(default)]
    pub identity: Option<String>,
    /// Set to `only` to return `numberMatched` without any records.
    #[serde(default)]
    pub count: Option<String>,
}

/// How much of a search a caller wants back.
///
/// `Records` is the default: match the query, page it, return the records.
/// `Only` answers just `numberMatched`, which the index can count without
/// materializing a single record — the shape a "how many are in this bbox"
/// caller actually wants, and the one that previously cost a full header list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CountMode {
    /// Return the matched page, with `numberMatched` alongside it.
    #[default]
    Records,
    /// Return `numberMatched` only, with an empty `matches` array.
    Only,
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
    /// Parsed query bbox; absent when the query was a frustum.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Vec<f64>>,
    /// Parsed frustum planes; absent when the query was a bbox.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frustum: Option<[[f64; 4]; 6]>,
    /// Applied spatial predicate.
    pub predicate: QueryPredicate,
    /// Applied result level; `/items` responses omit it (always feature).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<ResultLevel>,
    /// Applied payload mode; `/items` responses omit it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<PayloadMode>,
    /// Applied identity mode; `/items` responses omit it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<IdentityMode>,
    /// Applied count mode; `/items` responses omit it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<CountMode>,
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
    /// Identity modes accepted by `/search`.
    pub identity_modes: Vec<IdentityMode>,
    /// Count modes accepted by `/search`. `only` is absent when this
    /// collection cannot answer a count without materializing matches.
    pub count_modes: Vec<CountMode>,
    /// Query shapes accepted by `/search`. `frustum` needs a 3D artifact.
    pub query_shapes: Vec<QueryShapeKind>,
}

/// A query shape a collection accepts, for [`Capabilities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryShapeKind {
    /// An axis-aligned window, 4 numbers for 2D and 6 for 3D.
    Bbox,
    /// Six inward-pointing planes. 3D artifacts only.
    Frustum,
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
    // `full` is still accepted everywhere — a client querying a mixed catalog
    // should not have to vary its request per collection — but advertising it
    // would promise detail this collection cannot produce.
    let identity_modes = if stores_feature_ids(collection.manifest()) {
        vec![IdentityMode::Ref, IdentityMode::Full]
    } else {
        vec![IdentityMode::Ref]
    };
    // The index counts entries. Where an entry can duplicate a source row --
    // a geometry split across the antimeridian, a multi-part feature -- that
    // is not the feature count, and collapsing to features means materializing
    // the matches after all. Such a collection advertises `only` for
    // `level=entry` and refuses it for `level=feature`; advertising it
    // unconditionally would promise a number the server has to reject.
    let count_modes = if collection.entries_may_duplicate_rows() {
        vec![CountMode::Records]
    } else {
        vec![CountMode::Records, CountMode::Only]
    };
    // A frustum is a 3D region; a 2D artifact has no z to test it against.
    let query_shapes = if matches!(collection.manifest().dims, CoordinateDims::Xyz) {
        vec![QueryShapeKind::Bbox, QueryShapeKind::Frustum]
    } else {
        vec![QueryShapeKind::Bbox]
    };
    Capabilities {
        items: payload_kind == PayloadKind::FeatureJson,
        predicates,
        levels,
        payload_modes: vec![PayloadMode::None, PayloadMode::Summary, PayloadMode::Full],
        identity_modes,
        count_modes,
        query_shapes,
    }
}

/// Search `/search`.
pub fn search_response(
    collection: &Collection,
    params: SearchParams,
) -> Result<SearchResponse, ServerError> {
    let options = SearchOptions::from_params(&params)?;
    let payload = parse_payload_mode(params.payload.as_deref())?;
    let shape = RecordShape {
        payload,
        level: resolve_level(collection, parse_level(params.level.as_deref())?)?,
        identity: resolve_identity(
            stores_feature_ids(collection.manifest()),
            payload,
            parse_identity_mode(params.identity.as_deref())?,
        ),
    };
    let payload_kind = PayloadKind::from(&collection.manifest().payload_plan);
    let count_mode = parse_count_mode(params.count.as_deref())?;
    let outcome = if count_mode == CountMode::Only {
        count_only_outcome(collection, &options.shape, options.predicate, shape)?
    } else {
        search_records(
            collection,
            &options.shape,
            options.predicate,
            shape,
            options.offset,
            options.limit,
        )?
    };
    Ok(SearchResponse {
        collection_id: collection.id().to_owned(),
        query: QueryInfo {
            bbox: options.bbox(),
            frustum: options.frustum(),
            predicate: options.predicate,
            level: Some(shape.level),
            payload: Some(shape.payload),
            identity: Some(shape.identity),
            count: Some(count_mode),
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
    if params.identity.is_some() {
        return Err(ServerError::UnsupportedQuery(
            "identity is only supported on /search".to_string(),
        ));
    }
    if params.count.is_some() {
        return Err(ServerError::UnsupportedQuery(
            "count is only supported on /search".to_string(),
        ));
    }
    if params.frustum.is_some() {
        return Err(ServerError::UnsupportedQuery(
            "frustum is only supported on /search".to_string(),
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
        &options.shape,
        options.predicate,
        RecordShape {
            payload: PayloadMode::Full,
            level: ResultLevel::Feature,
            // GeoJSON features carry their own `id`, so the server never emits
            // a separate `featureRef` here.
            identity: IdentityMode::Ref,
        },
        options.offset,
        options.limit,
    )?;
    let number_matched = outcome.number_matched;
    let page_len = outcome.records.len();
    let features = outcome
        .records
        .into_iter()
        .filter_map(|record| match record.payload {
            Some(MatchPayload::FeatureJson { feature }) => feature,
            _ => None,
        })
        .collect::<Vec<_>>();
    // Nothing here can be dropped today: the payload plan is checked to be
    // `FeatureJson` above, `PayloadMode::Full` is passed unconditionally, and
    // that combination always decodes a body. Those three facts now sit three
    // layers apart, so pin the consequence rather than trusting the reader to
    // re-derive it.
    debug_assert_eq!(
        features.len(),
        page_len,
        "an /items page silently lost a feature"
    );
    Ok(FeatureCollectionResponse {
        kind: "FeatureCollection",
        number_matched,
        number_returned: features.len(),
        query: QueryInfo {
            bbox: options.bbox(),
            frustum: options.frustum(),
            predicate: options.predicate,
            level: None,
            payload: None,
            identity: None,
            count: None,
            limit: options.limit,
            offset: options.offset,
        },
        features,
    })
}

/// The region a search was asked for.
///
/// A frustum arrives as six planes rather than as a view-projection matrix on
/// purpose. A matrix carries two conventions the wire cannot recover -- the
/// clip-space depth range (`ClipSpaceZ`, which this project refuses to default
/// silently because it is not derivable from the matrix) and row- versus
/// column-major storage -- and getting either wrong moves the near plane
/// without failing. Planes have no conventions left in them: the client
/// resolves both locally, where it knows the answer, and sends the result.
/// `Frustum3D::from_view_projection` is right there for it to use.
enum QueryShape {
    Bbox(Vec<f64>),
    Frustum(Frustum3D),
}

struct SearchOptions {
    shape: QueryShape,
    limit: usize,
    offset: usize,
    predicate: QueryPredicate,
}

impl SearchOptions {
    fn from_params(params: &SearchParams) -> Result<Self, ServerError> {
        let shape = match (params.bbox.as_deref(), params.frustum.as_deref()) {
            (Some(_), Some(_)) => {
                return Err(ServerError::InvalidQuery(
                    "bbox and frustum are mutually exclusive".to_string(),
                ));
            }
            (_, Some(raw)) => QueryShape::Frustum(parse_frustum(raw)?),
            (raw, None) => QueryShape::Bbox(parse_bbox(raw)?),
        };
        let (limit, offset) = limit_offset(params.limit.as_deref(), params.offset.as_deref())?;
        let predicate = parse_predicate(params.predicate.as_deref())?;
        Ok(Self {
            shape,
            limit,
            offset,
            predicate,
        })
    }

    /// The bbox to echo back, if the query was one.
    fn bbox(&self) -> Option<Vec<f64>> {
        match &self.shape {
            QueryShape::Bbox(bbox) => Some(bbox.clone()),
            QueryShape::Frustum(_) => None,
        }
    }

    /// The frustum planes to echo back, if the query was one.
    fn frustum(&self) -> Option<[[f64; 4]; 6]> {
        match &self.shape {
            QueryShape::Bbox(_) => None,
            QueryShape::Frustum(frustum) => Some(*frustum.planes()),
        }
    }
}

/// Resolve `level` for a collection, naming the collection in the error when
/// it asked for a level the artifact cannot serve.
///
/// The decision itself lives in [`packed_spatial_index_geo::resolve_level`],
/// shared with every other artifact query frontend; this wrapper only adds
/// the collection id to the message.
fn resolve_level(
    collection: &Collection,
    requested: Option<ResultLevel>,
) -> Result<ResultLevel, ServerError> {
    // `LevelError` is `#[non_exhaustive]`, so it cannot be destructured
    // irrefutably here even though it has one variant today; only the
    // collection id in the message is server-specific.
    packed_spatial_index_geo::resolve_level(requested, &collection.manifest().payload_plan).map_err(
        |_: LevelError| {
            ServerError::UnsupportedLevel(format!(
                "collection `{}` stores no feature references; use level=entry",
                collection.id()
            ))
        },
    )
}

/// Matched-and-paged search result: the pre-pagination match count plus the
/// records of the requested page only.
///
/// `numberMatched` comes from the materialized header/id list because every
/// returned record needs its identity anyway. A caller that wants only the
/// number asks for `count=only`, which skips the list entirely through geo's
/// `count_entries`.
struct SearchOutcome {
    number_matched: usize,
    records: Vec<MatchRecord>,
}

/// What a returned record contains, independent of the path that produced it.
#[derive(Debug, Clone, Copy)]
struct RecordShape {
    payload: PayloadMode,
    level: ResultLevel,
    identity: IdentityMode,
}

impl RecordShape {
    /// Whether the requested shape needs payload bodies for the returned page.
    fn needs_payload_bodies(&self) -> bool {
        needs_payload_bodies(self.payload, self.identity)
    }

    /// Whether the artifact's paged header search can answer this shape.
    ///
    /// Paging happens over index entries. Feature-level results collapse split
    /// entries first, and that regrouping has to see the whole match set before
    /// anything can be counted or sliced — unless the artifact records that its
    /// entries never duplicate a source row, in which case the collapse is a
    /// no-op and entry order is already feature order.
    fn can_page_entries(&self, collection: &Collection) -> bool {
        matches!(self.level, ResultLevel::Entry) || !collection.entries_may_duplicate_rows()
    }
}

/// The synchronous artifact methods a bbox search needs, in a shape that is
/// the same for both dimensions.
///
/// The two search bodies differ only in the region type they search, so they
/// are written once against this trait rather than duplicated per dimension —
/// the shape the Worker demo's `AsyncGeoIndex` already takes for the async
/// path. Exact filtering deliberately stays off it: only 2D has a query
/// vocabulary to refine candidates against, so `GeoArtifactIndex2D` keeps that
/// path to itself instead of every implementor carrying a method one of them
/// cannot answer.
trait SyncGeoIndex {
    /// Region type this dimension searches.
    type Query: Copy;

    fn search_entry_ids(&self, query: Self::Query) -> Result<Vec<usize>, GeoError>;

    fn search_match_headers(&self, query: Self::Query) -> Result<Vec<GeoMatchHeader>, GeoError>;

    fn search_match_headers_page(
        &self,
        query: Self::Query,
        offset: usize,
        limit: usize,
    ) -> Result<GeoMatchHeaderPage, GeoError>;

    fn fetch_matches(&self, headers: &[GeoMatchHeader]) -> Result<Vec<GeoMatch>, GeoError>;
}

macro_rules! impl_sync_geo_index {
    ($index:ident, $query:ty) => {
        impl<R: RangeReader> SyncGeoIndex for $index<R> {
            type Query = $query;

            fn search_entry_ids(&self, query: Self::Query) -> Result<Vec<usize>, GeoError> {
                <$index<R>>::search_entry_ids(self, query)
            }

            fn search_match_headers(
                &self,
                query: Self::Query,
            ) -> Result<Vec<GeoMatchHeader>, GeoError> {
                <$index<R>>::search_match_headers(self, query)
            }

            fn search_match_headers_page(
                &self,
                query: Self::Query,
                offset: usize,
                limit: usize,
            ) -> Result<GeoMatchHeaderPage, GeoError> {
                <$index<R>>::search_match_headers_page(self, query, offset, limit)
            }

            fn fetch_matches(&self, headers: &[GeoMatchHeader]) -> Result<Vec<GeoMatch>, GeoError> {
                <$index<R>>::fetch_matches(self, headers)
            }
        }
    };
}

impl_sync_geo_index!(GeoArtifactIndex2D, Box2D);
impl_sync_geo_index!(GeoArtifactIndex3D, GeoQuery3D);

/// Answer `count=only`: how many index entries match, without materializing
/// one of them.
///
/// Refused rather than approximated in the two cases where the index's own
/// count is not the number the caller asked for:
///
/// * `predicate=intersects` narrows candidates by exact geometry *after* the
///   index answers, so the index count is an upper bound, not the answer.
/// * `level=feature` on a collection whose entries can duplicate a source row
///   needs those rows collapsed, which means reading the matches -- exactly
///   the work this mode exists to skip. Entry level counts fine there.
fn count_only_outcome(
    collection: &Collection,
    query_shape: &QueryShape,
    predicate: QueryPredicate,
    shape: RecordShape,
) -> Result<SearchOutcome, ServerError> {
    if predicate == QueryPredicate::Intersects {
        return Err(ServerError::UnsupportedQuery(
            "count=only counts index matches, which predicate=intersects narrows afterwards;              drop one of them"
                .to_string(),
        ));
    }
    if shape.level == ResultLevel::Feature && collection.entries_may_duplicate_rows() {
        return Err(ServerError::UnsupportedQuery(format!(
            "collection `{}` can store one source feature as several index entries, so a              feature-level count has to read the matches; use count=only with level=entry,              or drop count=only",
            collection.id()
        )));
    }
    let number_matched = match collection.open_local_index()? {
        GeoArtifactIndex::D2(index) => index
            .count_entries(query_2d(query_shape, collection)?)
            .map_err(ServerError::from_geo)?,
        GeoArtifactIndex::D3(index) => index
            .count_entries(query_3d(query_shape, collection)?)
            .map_err(ServerError::from_geo)?,
    };
    Ok(SearchOutcome {
        number_matched,
        records: Vec::new(),
    })
}

fn search_records(
    collection: &Collection,
    query_shape: &QueryShape,
    predicate: QueryPredicate,
    shape: RecordShape,
    offset: usize,
    limit: usize,
) -> Result<SearchOutcome, ServerError> {
    let exact = predicate == QueryPredicate::Intersects;
    match collection.open_local_index()? {
        GeoArtifactIndex::D2(index) => {
            let query = query_2d(query_shape, collection)?;
            if exact {
                return exact_records(&index, query, collection, shape, offset, limit);
            }
            bbox_records(&index, query, collection, shape, offset, limit)
        }
        GeoArtifactIndex::D3(index) => {
            // The only thing 3D answers differently: `GeoQuery3D` has no
            // polygon variant, so there is no exact phase to refine candidates
            // with -- and a frustum has no narrow phase in this crate at all.
            // Everything below this line is shared with 2D.
            if exact {
                return Err(ServerError::UnsupportedPredicate(format!(
                    "collection `{}` is 3D; predicate=intersects is only supported for 2D artifacts in this server",
                    collection.id()
                )));
            }
            bbox_records(
                &index,
                query_3d(query_shape, collection)?,
                collection,
                shape,
                offset,
                limit,
            )
        }
    }
}

/// The 2D query a shape resolves to, or why it cannot be one.
fn query_2d(shape: &QueryShape, collection: &Collection) -> Result<Box2D, ServerError> {
    match shape {
        QueryShape::Bbox(bbox) if bbox.len() == 4 => {
            Ok(Box2D::new(bbox[0], bbox[1], bbox[2], bbox[3]))
        }
        QueryShape::Bbox(_) => Err(ServerError::InvalidBbox(format!(
            "2D collection `{}` expects bbox=minx,miny,maxx,maxy",
            collection.id()
        ))),
        QueryShape::Frustum(_) => Err(ServerError::UnsupportedQuery(format!(
            "collection `{}` is 2D; a frustum query needs a 3D artifact -- use bbox",
            collection.id()
        ))),
    }
}

/// The 3D query a shape resolves to, or why it cannot be one.
fn query_3d(shape: &QueryShape, collection: &Collection) -> Result<GeoQuery3D, ServerError> {
    match shape {
        QueryShape::Bbox(bbox) if bbox.len() == 6 => Ok(GeoQuery3D::from(Box3D::new(
            bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5],
        ))),
        QueryShape::Bbox(_) => Err(ServerError::InvalidBbox(format!(
            "3D collection `{}` expects bbox=minx,miny,minz,maxx,maxy,maxz",
            collection.id()
        ))),
        QueryShape::Frustum(frustum) => Ok(GeoQuery3D::from(*frustum)),
    }
}

/// Answer a bbox search — the path both dimensions take.
///
/// Feature identity lives in the fixed payload prefix, so this sorts, dedupes,
/// and pages without reading payload bodies; bodies are fetched for the
/// returned page only, and only when the requested shape needs them.
fn bbox_records<I: SyncGeoIndex>(
    index: &I,
    query: I::Query,
    collection: &Collection,
    shape: RecordShape,
    offset: usize,
    limit: usize,
) -> Result<SearchOutcome, ServerError> {
    let payload_plan = &collection.manifest().payload_plan;
    // No payload section means no feature refs to sort or group by, so entry
    // ids are the whole answer.
    if matches!(payload_plan, PayloadPlan::None) {
        return Ok(id_outcome(
            index
                .search_entry_ids(query)
                .map_err(ServerError::from_geo)?,
            shape,
            offset,
            limit,
        ));
    }
    if shape.can_page_entries(collection) {
        let page = index
            .search_match_headers_page(query, offset, limit)
            .map_err(ServerError::from_geo)?;
        return page_outcome(page, shape, payload_plan, |page| index.fetch_matches(page));
    }
    let headers = index
        .search_match_headers(query)
        .map_err(ServerError::from_geo)?;
    header_outcome(headers, shape, offset, limit, payload_plan, |page| {
        index.fetch_matches(page)
    })
}

/// Answer `predicate=intersects`, which only 2D can serve.
///
/// The exact phase needs every match's geometry up front, so this materializes
/// the whole match set instead of paging headers the way [`bbox_records`] can.
fn exact_records<R: RangeReader>(
    index: &GeoArtifactIndex2D<R>,
    query: Box2D,
    collection: &Collection,
    shape: RecordShape,
    offset: usize,
    limit: usize,
) -> Result<SearchOutcome, ServerError> {
    if matches!(collection.manifest().payload_plan, PayloadPlan::None) {
        return Err(ServerError::UnsupportedPredicate(format!(
            "collection `{}` cannot apply predicate=intersects because its artifact has no geometry payload",
            collection.id()
        )));
    }
    if !collection.supports_intersects_predicate() {
        return Err(ServerError::UnsupportedPredicate(format!(
            "collection `{}` cannot apply predicate=intersects from its artifact payload",
            collection.id()
        )));
    }
    let matches = index.search_matches(query).map_err(ServerError::from_geo)?;
    let matches = index
        .filter_matches(
            matches,
            GeoQuery2D::box2d(query),
            SpatialPredicate::Intersects,
            NonPlanarExactPolicy::Reject,
        )
        .map_err(ServerError::from_geo)?;
    Ok(match_outcome(matches, shape, offset, limit))
}

/// Page an id-only (payload-less) result set.
fn id_outcome(
    mut ids: Vec<usize>,
    shape: RecordShape,
    offset: usize,
    limit: usize,
) -> SearchOutcome {
    ids.sort_unstable();
    ids.dedup();
    let number_matched = ids.len();
    let records = paginate(&ids, offset, limit)
        .into_iter()
        .map(|id| MatchRecord {
            entry_id: id,
            feature_ref: None,
            payload: match_payload_none(shape.payload),
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
    shape: RecordShape,
    offset: usize,
    limit: usize,
) -> SearchOutcome {
    GeoMatch::sort_by_entry(&mut matches);
    if matches!(shape.level, ResultLevel::Feature) {
        GeoMatch::dedupe_by_feature(&mut matches);
    }
    let number_matched = matches.len();
    let records = paginate(&matches, offset, limit)
        .into_iter()
        .map(|m| match_record(m.entry_id, Some(m.feature), m.payload, shape))
        .collect();
    SearchOutcome {
        number_matched,
        records,
    }
}

/// Sort, dedupe, and page match headers; payload bodies are fetched only for
/// the page, and only when the requested shape needs them.
///
/// Used when feature-level grouping has to happen before counting, which needs
/// the whole match set in memory. [`page_outcome`] is the bounded path.
fn header_outcome(
    mut headers: Vec<GeoMatchHeader>,
    shape: RecordShape,
    offset: usize,
    limit: usize,
    plan: &PayloadPlan,
    fetch: impl FnOnce(&[GeoMatchHeader]) -> Result<Vec<GeoMatch>, packed_spatial_index_geo::GeoError>,
) -> Result<SearchOutcome, ServerError> {
    GeoMatchHeader::sort_by_entry(&mut headers);
    if matches!(shape.level, ResultLevel::Feature) {
        GeoMatchHeader::dedupe_by_feature(&mut headers);
    }
    let number_matched = headers.len();
    let page = paginate(&headers, offset, limit);
    Ok(SearchOutcome {
        number_matched,
        records: page_records(page, shape, plan, fetch)?,
    })
}

/// Build records from a page geo already counted and sliced.
fn page_outcome(
    page: GeoMatchHeaderPage,
    shape: RecordShape,
    plan: &PayloadPlan,
    fetch: impl FnOnce(&[GeoMatchHeader]) -> Result<Vec<GeoMatch>, packed_spatial_index_geo::GeoError>,
) -> Result<SearchOutcome, ServerError> {
    Ok(SearchOutcome {
        number_matched: page.number_matched,
        records: page_records(page.headers, shape, plan, fetch)?,
    })
}

fn page_records(
    page: Vec<GeoMatchHeader>,
    shape: RecordShape,
    plan: &PayloadPlan,
    fetch: impl FnOnce(&[GeoMatchHeader]) -> Result<Vec<GeoMatch>, packed_spatial_index_geo::GeoError>,
) -> Result<Vec<MatchRecord>, ServerError> {
    if shape.needs_payload_bodies() {
        Ok(fetch(&page)
            .map_err(ServerError::from_geo)?
            .into_iter()
            .map(|m| match_record(m.entry_id, Some(m.feature), m.payload, shape))
            .collect())
    } else {
        Ok(page
            .into_iter()
            .map(|header| header_record(header, shape, plan))
            .collect())
    }
}

/// Build a record straight from a header — no payload body was read, so
/// summary mode derives `byteLength` from the header's payload length.
fn header_record(header: GeoMatchHeader, shape: RecordShape, plan: &PayloadPlan) -> MatchRecord {
    let payload = match (shape.payload, plan) {
        (PayloadMode::None, _) => None,
        (_, PayloadPlan::RowRef) => Some(MatchPayload::RowRef),
        (_, PayloadPlan::RowWkb) => Some(MatchPayload::RowWkb {
            byte_length: header.body_byte_len().unwrap_or(0),
            wkb_base64: None,
        }),
        // No `byteLength` for FeatureJson: the intersects path decodes bodies
        // instead of headers and has no cheap equivalent, so reporting it here
        // would make the field appear only on one of the two paths.
        (_, PayloadPlan::FeatureJson { .. }) => Some(MatchPayload::FeatureJson { feature: None }),
        // The header search rejects payload-less artifacts up front.
        (_, PayloadPlan::None) => None,
    };
    let mut feature = header.feature;
    // A representative part number is meaningless once split entries collapse
    // into one feature-level record. Feature-level dedupe already clears it,
    // but the paged path skips dedupe when the artifact cannot split entries.
    if matches!(shape.level, ResultLevel::Feature) {
        feature.part = None;
    }
    MatchRecord {
        entry_id: header.entry_id,
        feature_ref: Some(feature.into()),
        payload,
    }
}

fn match_record(
    entry_id: usize,
    feature_ref: Option<FeatureRef>,
    payload: GeoPayload,
    shape: RecordShape,
) -> MatchRecord {
    let payload = match (shape.payload, payload) {
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
        if matches!(shape.level, ResultLevel::Feature) {
            feature.part = None;
        }
        // Source feature ids live in payload bodies, so a body-decoding path
        // can produce one where the header path cannot. Emit it only when the
        // client asked for it, so a record describes the artifact rather than
        // the path that found it.
        if shape.identity == IdentityMode::Ref {
            feature.feature_id = None;
        }
        FeatureRefRecord::from(feature)
    });
    MatchRecord {
        entry_id,
        feature_ref,
        payload,
    }
}

fn match_payload_none(payload_mode: PayloadMode) -> Option<MatchPayload> {
    (payload_mode != PayloadMode::None).then_some(MatchPayload::None)
}

/// Parse six inward-pointing planes from 24 comma-separated numbers.
///
/// A plane is `a,b,c,d` with a point inside when `a*x + b*y + c*z + d >= 0`;
/// the planes need not be normalized, since only the sign is used. Order is
/// the frustum's own (left, right, bottom, top, near, far), which only matters
/// to a reader -- the test is against all six.
fn parse_frustum(raw: &str) -> Result<Frustum3D, ServerError> {
    let mut values = Vec::with_capacity(24);
    for part in raw.split(',') {
        let value = part.trim().parse::<f64>().map_err(|_| {
            ServerError::InvalidFrustum(format!("frustum value `{}` is not a number", part.trim()))
        })?;
        if !value.is_finite() {
            return Err(ServerError::InvalidFrustum(format!(
                "frustum value `{}` is not finite",
                part.trim()
            )));
        }
        values.push(value);
    }
    if values.len() != 24 {
        return Err(ServerError::InvalidFrustum(format!(
            "frustum must contain 24 numbers (six planes of a,b,c,d), got {}",
            values.len()
        )));
    }
    let mut planes = [[0.0f64; 4]; 6];
    for (plane, chunk) in planes.iter_mut().zip(values.as_chunks::<4>().0) {
        plane.copy_from_slice(chunk);
    }
    // A plane whose normal is zero tests nothing: `0*x + 0*y + 0*z + d` is a
    // constant, so the frustum silently becomes a half-open region instead of
    // failing. Cheaper to refuse than to explain.
    if let Some(index) = planes
        .iter()
        .position(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0)
    {
        return Err(ServerError::InvalidFrustum(format!(
            "frustum plane {index} has a zero normal, so it constrains nothing"
        )));
    }
    Ok(Frustum3D::from_planes(planes))
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

fn parse_count_mode(raw: Option<&str>) -> Result<CountMode, ServerError> {
    match raw {
        None | Some("") | Some("records") => Ok(CountMode::Records),
        Some("only") => Ok(CountMode::Only),
        Some(_) => Err(ServerError::InvalidCount(
            "count must be records or only".to_string(),
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

fn parse_identity_mode(raw: Option<&str>) -> Result<IdentityMode, ServerError> {
    match raw {
        None | Some("") | Some("ref") => Ok(IdentityMode::Ref),
        Some("full") => Ok(IdentityMode::Full),
        Some(_) => Err(ServerError::InvalidIdentity(
            "identity must be ref or full".to_string(),
        )),
    }
}

fn paginate<T: Clone>(records: &[T], offset: usize, limit: usize) -> Vec<T> {
    if offset >= records.len() {
        return Vec::new();
    }
    let end = records.len().min(offset.saturating_add(limit));
    records[offset..end].to_vec()
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
    fn identity_defaults_to_ref() {
        assert_eq!(parse_identity_mode(None).unwrap(), IdentityMode::Ref);
        assert_eq!(
            parse_identity_mode(Some("full")).unwrap(),
            IdentityMode::Full
        );
        assert!(parse_identity_mode(Some("id")).is_err());
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
