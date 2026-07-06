use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::get,
};
use serde::Serialize;

use crate::{
    ServerError, ServerState,
    query::{CollectionDetail, CollectionSummary, SearchParams, items_response, search_response},
};

/// Build the HTTP router.
pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/collections", get(collections))
        .route("/collections/{id}", get(collection))
        .route("/collections/{id}/items", get(items))
        .route("/collections/{id}/search", get(search))
        .with_state(state)
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
    Query(params): Query<SearchParams>,
) -> Result<Json<crate::query::FeatureCollectionResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    Ok(Json(items_response(&collection, params)?))
}

async fn search(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    Query(params): Query<SearchParams>,
) -> Result<Json<crate::query::SearchResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::CollectionNotFound(id.clone()))?;
    Ok(Json(search_response(&collection, params)?))
}
