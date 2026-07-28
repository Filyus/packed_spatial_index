use axum::{
    Json, Router,
    extract::{FromRequestParts, Path, Query, State},
    http::{Method, Uri, request::Parts},
    routing::get,
};
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    ServerError, ServerState,
    query::{CollectionDetail, CollectionSummary, SearchParams, items_response, search_response},
};

/// Build the HTTP router.
///
/// Both fallbacks are registered after the routes on purpose:
/// `method_not_allowed_fallback` attaches to the method routers registered so
/// far, so an earlier call would leave later routes answering 405 in axum's
/// default plain-text shape instead of this server's JSON error envelope.
pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/collections", get(collections))
        .route("/collections/{id}", get(collection))
        .route("/collections/{id}/items", get(items))
        .route("/collections/{id}/search", get(search))
        .method_not_allowed_fallback(method_not_allowed)
        .fallback(route_not_found)
        .with_state(state)
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

/// Serve the router on an already-bound listener.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: ServerState,
) -> Result<(), std::io::Error> {
    axum::serve(listener, router(state)).await
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
