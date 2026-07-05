use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::get,
};
use serde::Serialize;

use crate::{
    AppState, ServerError,
    query::{CollectionDetail, CollectionSummary, SearchParams, hits_response, items_response},
};

/// Build the HTTP router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/collections", get(collections))
        .route("/collections/{id}", get(collection))
        .route("/collections/{id}/items", get(items))
        .route("/collections/{id}/hits", get(hits))
        .with_state(state)
}

/// Serve the router on an already-bound listener.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: AppState,
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

async fn collections(State(state): State<AppState>) -> Json<Vec<CollectionSummary>> {
    let summaries = state
        .collections()
        .into_iter()
        .map(|collection| CollectionSummary::new(&collection))
        .collect();
    Json(summaries)
}

async fn collection(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<CollectionDetail>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::NotFound(id.clone()))?;
    Ok(Json(CollectionDetail::new(&collection)))
}

async fn items(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<SearchParams>,
) -> Result<Json<crate::query::FeatureCollectionResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::NotFound(id.clone()))?;
    Ok(Json(items_response(&collection, params)?))
}

async fn hits(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<SearchParams>,
) -> Result<Json<crate::query::HitsResponse>, ServerError> {
    let collection = state
        .collection(&id)
        .ok_or_else(|| ServerError::NotFound(id.clone()))?;
    Ok(Json(hits_response(&collection, params)?))
}
