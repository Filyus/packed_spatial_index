use axum::{
    Json, Router,
    extract::{FromRequestParts, Path, Query, RawQuery, State, rejection::JsonRejection},
    http::{HeaderValue, Method, Uri, request::Parts},
    routing::get,
};
use serde::{Serialize, de::DeserializeOwned};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tracing::Level;

use crate::{
    ServerError, ServerState,
    query::{
        AntiJoinParams, ClosestPairParams, CollectionDetail, CollectionSummary, ComponentsParams,
        JoinParams, NearestParams, SearchParams, anti_join_response, closest_pair_response,
        components_response, items_response, join_response, nearest_response, search_response,
    },
};

/// Build the HTTP router.
///
/// Both fallbacks are registered after the routes on purpose:
/// `method_not_allowed_fallback` attaches to the method routers registered so
/// far, so an earlier call would leave later routes answering 405 in axum's
/// default plain-text shape instead of this server's JSON error envelope.
pub fn router(state: ServerState) -> Router {
    router_with_cors(state, &[]).expect("an empty origin list is always valid")
}

/// Build the HTTP router, allowing the listed browser origins to read it.
///
/// An empty list adds no CORS headers at all, which is the default: a browser
/// page from another origin cannot read the responses. Origins are exact —
/// there is no wildcard — because this server has no authentication and
/// opening it to every origin should be a per-deployment decision.
pub fn router_with_cors(state: ServerState, origins: &[String]) -> Result<Router, ServerError> {
    let router = Router::new()
        .route("/health", get(health))
        // These three take no query parameters, so unlike the two below they
        // do not use `ValidQuery`: there is nothing a typo could silently
        // become, and refusing an incidental `?f=json` would only be rude.
        .route("/collections", get(collections))
        .route("/collections/{id}", get(collection))
        .route("/collections/{id}/items", get(items))
        // POST carries the same search as a JSON body, for a polygon too
        // large for a URL. See `search_post` for why the query string must
        // be empty there.
        .route("/collections/{id}/search", get(search).post(search_post))
        // Distance join: every pair of entries between two collections whose
        // boxes lie within an `max_distance` distance. Joining a collection with
        // itself is the self join: each unordered pair once.
        .route("/collections/{id}/join/{other}", get(join))
        // The noise side of the same graph: entries of `id` with no entry of
        // `other` within `max_distance`. `other` may not equal `id` — see
        // `anti_join_response`.
        .route("/collections/{id}/anti-join/{other}", get(anti_join))
        // The single nearest pair between two collections, or within one
        // when `other` equals `id`: the join with no bound to guess.
        .route("/collections/{id}/closest-pair/{other}", get(closest_pair))
        // Connected components of one collection's own proximity graph. No
        // `{other}` segment: a component is a property of one graph.
        .route("/collections/{id}/components", get(components))
        // The k entries nearest a point, nearest first, under a planar or
        // spherical metric. See `nearest_response` for how the metric is
        // chosen.
        .route("/collections/{id}/nearest", get(nearest))
        .method_not_allowed_fallback(method_not_allowed)
        .fallback(route_not_found)
        // Layered outside the fallbacks so a 404 or 405 is logged too.
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .with_state(state);

    if origins.is_empty() {
        return Ok(router);
    }
    let allowed = origins
        .iter()
        .map(|origin| {
            origin
                .parse::<HeaderValue>()
                .map_err(|_| ServerError::Config(format!("`{origin}` is not a valid CORS origin")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(router.layer(
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(allowed))
            .allow_methods([Method::GET, Method::HEAD]),
    ))
}

/// Query-string extractor that reports rejections in the JSON error envelope.
///
/// [`Query`] answers a malformed or unknown parameter with plain text, which
/// would be the one failure shape a client cannot parse. It also means a
/// misspelled parameter reaches the handler as a silent default, so the
/// deserialized type is expected to carry `deny_unknown_fields`.
struct ValidQuery<T>(T);

impl<T, S> FromRequestParts<S> for ValidQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ServerError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        Query::<T>::from_request_parts(parts, state)
            .await
            .map(|Query(params)| Self(params))
            .map_err(|rejection| ServerError::InvalidQuery(rejection.body_text()))
    }
}

async fn route_not_found(uri: Uri) -> ServerError {
    ServerError::RouteNotFound(uri.path().to_owned())
}

async fn method_not_allowed(method: Method, uri: Uri) -> ServerError {
    ServerError::MethodNotAllowed(format!("{method} {}", uri.path()))
}

/// Serve the router on an already-bound listener until a shutdown signal.
///
/// In-flight requests finish before the process exits, which matters here
/// because a query holds an open artifact file and can be mid-read.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: ServerState,
) -> Result<(), std::io::Error> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
}

/// Serve with the catalog's allowed browser origins applied.
pub async fn serve_with_cors(
    listener: tokio::net::TcpListener,
    state: ServerState,
    origins: &[String],
) -> Result<(), ServerError> {
    let router = router_with_cors(state, origins)?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|err| ServerError::io("listener", err))
}

/// Resolve when the process is asked to stop.
///
/// Ctrl-C everywhere, plus the platform's "please stop" signal: SIGTERM on
/// Unix, which is what a container runtime or service manager sends first, and
/// Ctrl-Break on Windows, which is the one a console process can be sent
/// individually.
async fn shutdown_signal() {
    // A missing handler leaves nothing to wait for. Hold the future open so
    // the server keeps serving instead of exiting the moment it starts.
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(windows)]
    let terminate = async {
        match tokio::signal::windows::ctrl_break() {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(any(unix, windows)))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("interrupted; finishing in-flight requests"),
        () = terminate => tracing::info!("termination requested; finishing in-flight requests"),
    }
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn collections(State(state): State<ServerState>) -> Json<Vec<CollectionSummary>> {
    let summaries = state
        .collections()
        .into_iter()
        .map(|collection| CollectionSummary::new(&collection))
        .collect();
    Json(summaries)
}

async fn collection(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Json<CollectionDetail>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    Ok(Json(CollectionDetail::new(&collection)))
}

async fn items(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    ValidQuery(params): ValidQuery<SearchParams>,
) -> Result<Json<crate::query::FeatureCollectionResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    Ok(Json(
        query_blocking(move || items_response(&collection, params)).await?,
    ))
}

async fn search(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    ValidQuery(params): ValidQuery<SearchParams>,
) -> Result<Json<crate::query::SearchResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    Ok(Json(
        query_blocking(move || search_response(&collection, params)).await?,
    ))
}

/// Distance join between two collections: every pair of entries whose boxes
/// lie within `max_distance`. `other` may equal `id` — the self join reports each
/// unordered pair once. See `join_response` for the semantics.
async fn join(
    State(state): State<ServerState>,
    Path((id, other)): Path<(String, String)>,
    ValidQuery(params): ValidQuery<JoinParams>,
) -> Result<Json<crate::query::JoinResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    let other = state
        .collection(&other)
        .ok_or(ServerError::CollectionNotFound(other))?;
    Ok(Json(
        query_blocking(move || join_response(&collection, &other, params)).await?,
    ))
}

/// The single nearest pair between `id` and `other`, or within `id` when
/// they are the same. See `closest_pair_response`.
async fn closest_pair(
    State(state): State<ServerState>,
    Path((id, other)): Path<(String, String)>,
    ValidQuery(params): ValidQuery<ClosestPairParams>,
) -> Result<Json<crate::query::ClosestPairResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    let other = state
        .collection(&other)
        .ok_or(ServerError::CollectionNotFound(other))?;
    Ok(Json(
        query_blocking(move || closest_pair_response(&collection, &other, params)).await?,
    ))
}

/// Distance anti-join: entries of `id` with no entry of `other` within
/// `max_distance`. `other` may not equal `id`. See `anti_join_response`.
async fn anti_join(
    State(state): State<ServerState>,
    Path((id, other)): Path<(String, String)>,
    ValidQuery(params): ValidQuery<AntiJoinParams>,
) -> Result<Json<crate::query::AntiJoinResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    let other = state
        .collection(&other)
        .ok_or(ServerError::CollectionNotFound(other))?;
    Ok(Json(
        query_blocking(move || anti_join_response(&collection, &other, params)).await?,
    ))
}

/// Components of one collection's `max_distance`-proximity graph. See
/// `components_response`.
async fn components(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    ValidQuery(params): ValidQuery<ComponentsParams>,
) -> Result<Json<crate::query::ComponentsResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    Ok(Json(
        query_blocking(move || components_response(&collection, params)).await?,
    ))
}

/// The k entries nearest a point. See `nearest_response`.
async fn nearest(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    ValidQuery(params): ValidQuery<NearestParams>,
) -> Result<Json<crate::query::NearestResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    Ok(Json(
        query_blocking(move || nearest_response(&collection, params)).await?,
    ))
}

/// `POST /collections/{id}/search` -- the same search, sent as a body.
///
/// The body is the whole request and the query string must be empty. Merging
/// the two would mean deciding which wins per parameter, and every future
/// parameter would inherit that question; refusing costs one check and leaves
/// exactly one source of truth per request.
async fn search_post(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    RawQuery(query): RawQuery,
    body: Result<Json<crate::query::SearchBody>, JsonRejection>,
) -> Result<Json<crate::query::SearchResponse>, ServerError> {
    if query.is_some_and(|query| !query.is_empty()) {
        return Err(ServerError::InvalidQuery(
            "a POST search is described by its body; send no query string".to_string(),
        ));
    }
    let Json(body) = body.map_err(|rejection| ServerError::InvalidQuery(rejection.body_text()))?;
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    let params = SearchParams::from(body);
    Ok(Json(
        query_blocking(move || search_response(&collection, params)).await?,
    ))
}

/// Run an artifact query off the async runtime.
///
/// Querying an artifact is synchronous file I/O plus decoding. Running it
/// directly inside a handler parks a runtime worker for the duration, so a
/// handful of broad queries can stall every other request the runtime is
/// driving, including `/health`.
async fn query_blocking<T, F>(query: F) -> Result<T, ServerError>
where
    F: FnOnce() -> Result<T, ServerError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(query)
        .await
        .map_err(|err| ServerError::QueryTask(err.to_string()))?
}
