//! `/collections/{id}/closest-pair/{other}`: the join family's member with
//! no bound to guess.

use std::{fs, path::Path};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use packed_spatial_index_geo::{ConvertRequest, PayloadPlan, open_geojson_slice};
use packed_spatial_index_server::{Catalog, ServerState, router};
use serde_json::Value;
use tempfile::tempdir;
use tower::ServiceExt;

/// a0 (0,0), a1 (10,0), a2 (100,100).
fn grid_points_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "a0", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "a1", "geometry": {"type": "Point", "coordinates": [10.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "a2", "geometry": {"type": "Point", "coordinates": [100.0, 100.0]}, "properties": {}}
        ]
    }"#
}

/// b0 (50,0), b1 (13,0): b1 is 3.0 from a1, everything else is farther.
fn other_points_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "b0", "geometry": {"type": "Point", "coordinates": [50.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "b1", "geometry": {"type": "Point", "coordinates": [13.0, 0.0]}, "properties": {}}
        ]
    }"#
}

fn one_point_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "only", "geometry": {"type": "Point", "coordinates": [1.0, 1.0]}, "properties": {}}
        ]
    }"#
}

fn one_point_3d_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "only", "geometry": {"type": "Point", "coordinates": [1.0, 1.0, 1.0]}, "properties": {}}
        ]
    }"#
}

fn write_artifact(path: &Path, doc: &[u8]) {
    let mut source = open_geojson_slice(doc).unwrap();
    let bytes = source
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();
    fs::write(path, bytes).unwrap();
}

fn two_collection_state(first: &[u8], second: &[u8]) -> ServerState {
    let dir = tempdir().unwrap().keep();
    let data_dir = dir.join("data");
    fs::create_dir(&data_dir).unwrap();
    write_artifact(&data_dir.join("places.psindex"), first);
    write_artifact(&data_dir.join("other.psindex"), second);
    let catalog = Catalog::from_toml_str(
        r#"
        [[collections]]
        id = "places"
        artifact = "data/places.psindex"

        [[collections]]
        id = "roads"
        artifact = "data/other.psindex"
    "#,
        &dir,
    )
    .unwrap();
    ServerState::from_catalog(catalog).unwrap()
}

async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, Value) {
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn closest_pair_between_two_collections() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        other_points_geojson(),
    ));
    let (status, json) = get_json(app.clone(), "/collections/places/closest-pair/roads").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["collectionId"], "places");
    assert_eq!(json["joinCollectionId"], "roads");
    assert_eq!(json["pair"]["a"], 1);
    assert_eq!(json["pair"]["b"], 1);
    assert_eq!(json["pair"]["distance"], 3.0);

    // The other way round swaps the sides.
    let (status, json) = get_json(app, "/collections/roads/closest-pair/places").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["pair"]["a"], 1);
    assert_eq!(json["pair"]["b"], 1);
    assert_eq!(json["pair"]["distance"], 3.0);
}

#[tokio::test]
async fn self_form_reports_distinct_entries() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        other_points_geojson(),
    ));
    let (status, json) = get_json(app, "/collections/places/closest-pair/places").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["joinCollectionId"], "places");
    let (a, b) = (
        json["pair"]["a"].as_u64().unwrap(),
        json["pair"]["b"].as_u64().unwrap(),
    );
    assert_eq!((a.min(b), a.max(b)), (0, 1), "{json}");
    assert_eq!(json["pair"]["distance"], 10.0);
}

#[tokio::test]
async fn null_when_there_is_no_pair() {
    let app = router(two_collection_state(
        one_point_geojson(),
        other_points_geojson(),
    ));
    // One entry has no distinct partner within its own collection.
    let (status, json) = get_json(app.clone(), "/collections/places/closest-pair/places").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert!(json["pair"].is_null(), "{json}");
    // But it pairs across collections.
    let (status, json) = get_json(app, "/collections/places/closest-pair/roads").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["pair"]["a"], 0);
    assert_eq!(json["pair"]["b"], 1);
}

#[tokio::test]
async fn rejects_mismatched_dimensions_unknown_collections_and_parameters() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        one_point_3d_geojson(),
    ));
    let (status, json) = get_json(app.clone(), "/collections/places/closest-pair/roads").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{json}");
    assert_eq!(json["error"]["code"], "unsupported_query");

    let (status, _) = get_json(app.clone(), "/collections/places/closest-pair/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = get_json(app.clone(), "/collections/nope/closest-pair/places").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, json) = get_json(app, "/collections/places/closest-pair/places?within=1").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["code"], "invalid_query");
}
