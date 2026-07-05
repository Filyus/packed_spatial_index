use std::{fs, path::Path};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use packed_spatial_index_geo::{
    ConvertRequest, PayloadPlan, PropertyProjection, open_geojson_slice,
};
use packed_spatial_index_server::{AppState, Catalog, router};
use serde_json::Value;
use tempfile::tempdir;
use tower::ServiceExt;

fn sample_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "west",
                "geometry": {"type": "Point", "coordinates": [-5.0, 1.0]},
                "properties": {"name": "west"}
            },
            {
                "type": "Feature",
                "id": "east",
                "geometry": {"type": "Point", "coordinates": [25.0, 3.0]},
                "properties": {"name": "east"}
            }
        ]
    }"#
}

fn write_artifact(path: &Path, payload: PayloadPlan) {
    let mut source = open_geojson_slice(sample_geojson()).unwrap();
    let bytes = source
        .convert(ConvertRequest {
            payload,
            ..ConvertRequest::default()
        })
        .unwrap();
    fs::write(path, bytes).unwrap();
}

fn state_with_payload(payload: PayloadPlan) -> AppState {
    let dir = tempdir().unwrap().keep();
    let data_dir = dir.join("data");
    fs::create_dir(&data_dir).unwrap();
    let artifact = data_dir.join("places.psindex");
    write_artifact(&artifact, payload);
    let catalog_text = r#"
        [[collections]]
        id = "places"
        title = "Places"
        description = "Local places index"
        artifact = "data/places.psindex"
    "#;
    let catalog = Catalog::from_toml_str(catalog_text, &dir).unwrap();
    AppState::from_catalog(catalog).unwrap()
}

async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, Value) {
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

#[tokio::test]
async fn health_and_collections_work() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app.clone(), "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "ok");

    let (status, json) = get_json(app.clone(), "/collections").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json[0]["id"], "places");
    assert_eq!(json[0]["capabilities"]["hits"], true);

    let (status, json) = get_json(app, "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["id"], "places");
    assert_eq!(json["source_format"], "geojson");
}

#[tokio::test]
async fn items_returns_geojson_for_feature_json_payload() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(app, "/collections/places/items?bbox=-10,0,0,2&limit=10").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["type"], "FeatureCollection");
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(json["features"][0]["properties"]["name"], "west");
}

#[tokio::test]
async fn items_rejects_non_feature_json_payload() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));
    let (status, json) = get_json(app, "/collections/places/items?bbox=-10,0,0,2&limit=10").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(json["error"]["message"].as_str().unwrap().contains("/hits"));
}

#[tokio::test]
async fn hits_returns_row_refs_and_paginates() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(
        app,
        "/collections/places/hits?bbox=-10,0,30,5&limit=1&offset=1",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(json["hits"][0]["feature_ref"]["row_number"], 1);
    assert_eq!(json["hits"][0]["payload"]["kind"], "row_ref");
}

#[tokio::test]
async fn hits_can_include_wkb_payload() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));
    let (status, json) = get_json(
        app,
        "/collections/places/hits?bbox=-10,0,0,2&include_payload=true",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["hits"][0]["payload"]["kind"], "row_wkb");
    assert!(
        json["hits"][0]["payload"]["wkb_base64"]
            .as_str()
            .unwrap()
            .len()
            > 8
    );
}

#[tokio::test]
async fn hits_can_include_feature_json_payload() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app,
        "/collections/places/hits?bbox=-10,0,0,2&include_payload=true",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["hits"][0]["payload"]["kind"], "feature_json");
    assert_eq!(
        json["hits"][0]["payload"]["feature"]["properties"]["name"],
        "west"
    );
}

#[tokio::test]
async fn route_errors_are_json() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app.clone(), "/collections/missing").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "not_found");

    let (status, json) = get_json(app.clone(), "/collections/places/hits?bbox=1,2,3").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "bad_request");

    let (status, json) = get_json(app, "/collections/places/hits?bbox=-10,0,0,2&exact=true").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported");
}
