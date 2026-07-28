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
        // These three take no query parameters, so unlike the two below they
        // do not use `ValidQuery`: there is nothing a typo could silently
        // become, and refusing an incidental `?f=json` would only be rude.
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
