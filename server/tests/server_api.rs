use std::{fs, path::Path};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use packed_spatial_index_geo::{
    AntimeridianPolicy, ConvertRequest, EnvelopePolicy, PayloadPlan, PropertyProjection,
    open_geojson_slice,
};
use packed_spatial_index_server::{Catalog, ServerState, router};
use serde_json::{Value, json};
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

fn antimeridian_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "crossing",
                "geometry": {"type": "LineString", "coordinates": [[170.0, 0.0], [-170.0, 1.0]]},
                "properties": {"name": "crossing"}
            }
        ]
    }"#
}

fn write_artifact_with_request(path: &Path, req: ConvertRequest, doc: &[u8]) {
    let mut source = open_geojson_slice(doc).unwrap();
    let bytes = source.convert(req).unwrap();
    fs::write(path, bytes).unwrap();
}

fn state_with_payload(payload: PayloadPlan) -> ServerState {
    state_with_geojson(payload, sample_geojson())
}

fn state_with_geojson(payload: PayloadPlan, doc: &[u8]) -> ServerState {
    state_with_geojson_request(
        ConvertRequest {
            payload,
            ..ConvertRequest::default()
        },
        doc,
    )
}

fn state_with_geojson_request(req: ConvertRequest, doc: &[u8]) -> ServerState {
    let dir = tempdir().unwrap().keep();
    let data_dir = dir.join("data");
    fs::create_dir(&data_dir).unwrap();
    let artifact = data_dir.join("places.psindex");
    write_artifact_with_request(&artifact, req, doc);
    let catalog_text = r#"
        [[collections]]
        id = "places"
        title = "Places"
        description = "Local places index"
        artifact = "data/places.psindex"
    "#;
    let catalog = Catalog::from_toml_str(catalog_text, &dir).unwrap();
    ServerState::from_catalog(catalog).unwrap()
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

fn assert_contract(actual: &Value, expected: Value) {
    assert_eq!(
        actual,
        &expected,
        "actual:\n{}\n\nexpected:\n{}",
        serde_json::to_string_pretty(actual).unwrap(),
        serde_json::to_string_pretty(&expected).unwrap()
    );
}

#[tokio::test]
async fn contract_collections_summary_shape() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app, "/collections").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json,
        json!([
            {
                "id": "places",
                "title": "Places",
                "description": "Local places index",
                "featureCount": 2,
                "entryCount": 2,
                "dims": "xy",
                "storagePrecision": "f64",
                "payloadPlan": {"kind": "row_ref"},
                "nodeSize": 16,
                "hasPayload": true,
                "capabilities": {
                    "bboxSearch": true,
                    "featureJsonItems": false,
                    "hits": true,
                    "exactFilter": false,
                    "sourceReadBack": false,
                    "rowWkbPayload": false,
                    "rowRefPayload": true
                }
            }
        ]),
    );
}

#[tokio::test]
async fn contract_hits_summary_shape() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app, "/collections/places/hits?bbox=-10,0,0,2").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json,
        json!({
            "collectionId": "places",
            "query": {
                "bbox": [-10.0, 0.0, 0.0, 2.0],
                "exact": false,
                "exactApplied": false
            },
            "numberMatched": 1,
            "numberReturned": 1,
            "offset": 0,
            "limit": 100,
            "payloadPlan": {"kind": "row_ref"},
            "payloadMode": "summary",
            "hits": [
                {
                    "entryId": 0,
                    "featureRef": {
                        "rowNumber": 0
                    },
                    "payload": {"kind": "row_ref"}
                }
            ]
        }),
    );
}

#[tokio::test]
async fn contract_hits_feature_json_full_shape() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) =
        get_json(app, "/collections/places/hits?bbox=-10,0,0,2&payload=full").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json,
        json!({
            "collectionId": "places",
            "query": {
                "bbox": [-10.0, 0.0, 0.0, 2.0],
                "exact": false,
                "exactApplied": false
            },
            "numberMatched": 1,
            "numberReturned": 1,
            "offset": 0,
            "limit": 100,
            "payloadPlan": {
                "kind": "feature_json",
                "properties": {"kind": "all_non_geometry"}
            },
            "payloadMode": "full",
            "hits": [
                {
                    "entryId": 0,
                    "featureRef": {
                        "rowNumber": 0,
                        "featureId": "west"
                    },
                    "payload": {
                        "kind": "feature_json",
                        "feature": {
                            "type": "Feature",
                            "id": "west",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [-5.0, 1.0]
                            },
                            "properties": {"name": "west"}
                        }
                    }
                }
            ]
        }),
    );
}

#[tokio::test]
async fn contract_items_feature_collection_shape() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(app, "/collections/places/items?bbox=-10,0,0,2").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json,
        json!({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "west",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-5.0, 1.0]
                    },
                    "properties": {"name": "west"}
                }
            ],
            "numberMatched": 1,
            "numberReturned": 1,
            "offset": 0,
            "limit": 100,
            "query": {
                "bbox": [-10.0, 0.0, 0.0, 2.0],
                "exact": false,
                "exactApplied": false
            }
        }),
    );
}

#[tokio::test]
async fn contract_error_shape() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app, "/collections/places/hits?bbox=-10,0,0,2&payload=yes").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_contract(
        &json,
        json!({
            "error": {
                "code": "invalid_payload",
                "message": "invalid payload mode: payload must be none, summary, or full"
            }
        }),
    );
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
    assert_eq!(json[0]["featureCount"], 2);
    assert_eq!(json[0]["entryCount"], 2);
    assert!(json[0].get("itemCount").is_none());
    assert!(json[0].get("indexEntryCount").is_none());

    let (status, json) = get_json(app, "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["id"], "places");
    assert_eq!(json["sourceFormat"], "geojson");
}

#[tokio::test]
async fn hits_are_entry_level_items_are_feature_level() {
    let app = router(state_with_geojson_request(
        ConvertRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::AllNonGeometry,
            },
            ..ConvertRequest::default()
        },
        antimeridian_geojson(),
    ));

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/hits?bbox=-180,-10,180,10&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);
    assert_eq!(json["numberReturned"], 2);
    assert_eq!(json["hits"][0]["featureRef"]["rowNumber"], 0);
    assert_eq!(json["hits"][1]["featureRef"]["rowNumber"], 0);
    assert_ne!(json["hits"][0]["entryId"], json["hits"][1]["entryId"]);

    let (status, json) = get_json(app, "/collections/places/items?bbox=-180,-10,180,10").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(json["features"][0]["properties"]["name"], "crossing");
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
    assert_eq!(
        json["query"]["bbox"],
        serde_json::json!([-10.0, 0.0, 0.0, 2.0])
    );
    assert_eq!(json["query"]["exact"], false);
    assert_eq!(json["query"]["exactApplied"], false);
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
    assert_eq!(json["payloadMode"], "summary");
    assert_eq!(
        json["query"]["bbox"],
        serde_json::json!([-10.0, 0.0, 30.0, 5.0])
    );
    assert_eq!(json["query"]["exact"], false);
    assert_eq!(json["query"]["exactApplied"], false);
    assert_eq!(json["hits"][0]["entryId"], 1);
    assert_eq!(json["hits"][0]["featureRef"]["rowNumber"], 1);
    assert_eq!(json["hits"][0]["payload"]["kind"], "row_ref");
}

#[tokio::test]
async fn hits_can_include_wkb_payload() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));
    let (status, json) =
        get_json(app, "/collections/places/hits?bbox=-10,0,0,2&payload=full").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["payloadMode"], "full");
    assert_eq!(json["hits"][0]["payload"]["kind"], "row_wkb");
    assert!(json["hits"][0]["payload"]["byteLength"].as_u64().unwrap() > 8);
    assert!(
        json["hits"][0]["payload"]["wkbBase64"]
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
    let (status, json) =
        get_json(app, "/collections/places/hits?bbox=-10,0,0,2&payload=full").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["hits"][0]["payload"]["kind"], "feature_json");
    assert_eq!(
        json["hits"][0]["payload"]["feature"]["properties"]["name"],
        "west"
    );
}

#[tokio::test]
async fn hits_can_omit_payload_objects() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));
    let (status, json) =
        get_json(app, "/collections/places/hits?bbox=-10,0,0,2&payload=none").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["payloadMode"], "none");
    assert!(json["hits"][0].get("payload").is_none());
}

#[tokio::test]
async fn route_errors_are_json() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app.clone(), "/collections/missing").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "collection_not_found");

    let (status, json) = get_json(app.clone(), "/collections/places/hits?bbox=1,2,3").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_bbox");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/hits?bbox=-10,0,0,2&limit=0",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_limit");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/hits?bbox=-10,0,0,2&payload=yes",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_payload");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/items?bbox=-10,0,0,2&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_payload");

    let (status, json) = get_json(app, "/collections/places/hits?bbox=-10,0,0,2&exact=true").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "exact_filter_unavailable");
}
