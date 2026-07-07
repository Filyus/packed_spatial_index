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
                "payloadKind": "row_ref",
                "capabilities": {
                    "items": false,
                    "predicates": ["bbox"],
                    "levels": ["feature", "entry"],
                    "payloadModes": ["none", "summary", "full"]
                }
            }
        ]),
    );
}

#[tokio::test]
async fn contract_search_summary_shape() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app, "/collections/places/search?bbox=-10,0,0,2").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json,
        json!({
            "collectionId": "places",
            "query": {
                "bbox": [-10.0, 0.0, 0.0, 2.0],
                "predicate": "bbox",
                "level": "feature",
                "payload": "summary",
                "limit": 100,
                "offset": 0
            },
            "payloadKind": "row_ref",
            "numberMatched": 1,
            "numberReturned": 1,
            "matches": [
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
async fn contract_search_feature_json_full_shape() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json,
        json!({
            "collectionId": "places",
            "query": {
                "bbox": [-10.0, 0.0, 0.0, 2.0],
                "predicate": "bbox",
                "level": "feature",
                "payload": "full",
                "limit": 100,
                "offset": 0
            },
            "payloadKind": "feature_json",
            "numberMatched": 1,
            "numberReturned": 1,
            "matches": [
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
            "query": {
                "bbox": [-10.0, 0.0, 0.0, 2.0],
                "predicate": "bbox",
                "limit": 100,
                "offset": 0
            }
        }),
    );
}

#[tokio::test]
async fn contract_error_shape() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) =
        get_json(app, "/collections/places/search?bbox=-10,0,0,2&payload=yes").await;
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
    assert_eq!(json[0]["payloadKind"], "row_ref");
    assert_eq!(json[0]["capabilities"]["items"], false);
    assert_eq!(json[0]["featureCount"], 2);
    assert_eq!(json[0]["entryCount"], 2);
    assert!(json[0].get("payloadPlan").is_none());
    assert!(json[0].get("hasPayload").is_none());
    assert!(json[0].get("nodeSize").is_none());

    let (status, json) = get_json(app, "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["id"], "places");
    assert_eq!(json["sourceFormat"], "geojson");
    assert_eq!(json["nodeSize"], 16);
}

#[tokio::test]
async fn search_levels_control_split_entry_grouping() {
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
        "/collections/places/search?bbox=-180,-10,180,10&level=entry&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["level"], "entry");
    assert_eq!(json["numberMatched"], 2);
    assert_eq!(json["numberReturned"], 2);
    assert_eq!(json["matches"][0]["featureRef"]["rowNumber"], 0);
    assert_eq!(json["matches"][1]["featureRef"]["rowNumber"], 0);
    assert_ne!(json["matches"][0]["entryId"], json["matches"][1]["entryId"]);

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-180,-10,180,10",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["level"], "feature");
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["numberReturned"], 1);
    assert!(json["matches"][0]["featureRef"].get("part").is_none());

    let (status, json) = get_json(app, "/collections/places/items?bbox=-180,-10,180,10").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(json["features"][0]["properties"]["name"], "crossing");
}

#[tokio::test]
async fn payloadless_artifact_falls_back_to_entry_level() {
    let app = router(state_with_payload(PayloadPlan::None));

    let (status, json) = get_json(app.clone(), "/collections").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json[0]["payloadKind"], "none");
    assert_eq!(json[0]["capabilities"]["levels"], json!(["entry"]));
    assert_eq!(json[0]["capabilities"]["predicates"], json!(["bbox"]));

    let (status, json) = get_json(app.clone(), "/collections/places/search?bbox=-10,0,0,2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["level"], "entry");
    assert_eq!(json["numberMatched"], 1);
    assert!(json["matches"][0].get("featureRef").is_none());
    assert_eq!(json["matches"][0]["payload"]["kind"], "none");

    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&level=feature",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_level");
}

#[tokio::test]
async fn intersects_predicate_filters_from_wkb_payload() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));

    let (status, json) = get_json(app.clone(), "/collections").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json[0]["capabilities"]["predicates"],
        json!(["bbox", "intersects"])
    );

    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&predicate=intersects",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["predicate"], "intersects");
    assert_eq!(json["numberMatched"], 1);
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
    assert_eq!(json["query"]["predicate"], "bbox");
    assert_eq!(json["features"][0]["properties"]["name"], "west");
}

#[tokio::test]
async fn items_rejects_non_feature_json_payload() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));
    let (status, json) = get_json(app, "/collections/places/items?bbox=-10,0,0,2&limit=10").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("/search")
    );
}

#[tokio::test]
async fn search_returns_row_refs_and_paginates() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,30,5&limit=1&offset=1",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(json["query"]["payload"], "summary");
    assert_eq!(
        json["query"]["bbox"],
        serde_json::json!([-10.0, 0.0, 30.0, 5.0])
    );
    assert_eq!(json["query"]["predicate"], "bbox");
    assert_eq!(json["query"]["limit"], 1);
    assert_eq!(json["query"]["offset"], 1);
    assert_eq!(json["matches"][0]["entryId"], 1);
    assert_eq!(json["matches"][0]["featureRef"]["rowNumber"], 1);
    assert_eq!(json["matches"][0]["payload"]["kind"], "row_ref");
}

#[tokio::test]
async fn search_can_include_wkb_payload() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["payload"], "full");
    assert_eq!(json["matches"][0]["payload"]["kind"], "row_wkb");
    assert!(
        json["matches"][0]["payload"]["byteLength"]
            .as_u64()
            .unwrap()
            > 8
    );
    assert!(
        json["matches"][0]["payload"]["wkbBase64"]
            .as_str()
            .unwrap()
            .len()
            > 8
    );
}

#[tokio::test]
async fn search_can_include_feature_json_payload() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["matches"][0]["payload"]["kind"], "feature_json");
    assert_eq!(
        json["matches"][0]["payload"]["feature"]["properties"]["name"],
        "west"
    );
}

#[tokio::test]
async fn search_can_omit_payload_objects() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&payload=none",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["payload"], "none");
    assert!(json["matches"][0].get("payload").is_none());
}

#[tokio::test]
async fn row_wkb_pages_at_both_levels() {
    let app = router(state_with_payload(PayloadPlan::RowWkb));

    // Summary page at entry level: byteLength without payload bodies.
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,30,5&limit=1&offset=1&level=entry",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(json["matches"][0]["entryId"], 1);
    assert_eq!(json["matches"][0]["featureRef"]["rowNumber"], 1);
    assert_eq!(json["matches"][0]["payload"]["kind"], "row_wkb");
    let summary_len = json["matches"][0]["payload"]["byteLength"]
        .as_u64()
        .unwrap();
    assert!(summary_len > 8);
    assert!(json["matches"][0]["payload"].get("wkbBase64").is_none());

    // Full page at feature level: body fetched for the page only; byteLength
    // must agree with the summary derived from the header.
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,30,5&limit=1&offset=1&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["level"], "feature");
    assert_eq!(json["numberMatched"], 2);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(
        json["matches"][0]["payload"]["byteLength"]
            .as_u64()
            .unwrap(),
        summary_len
    );
    assert!(
        json["matches"][0]["payload"]["wkbBase64"]
            .as_str()
            .unwrap()
            .len()
            > 8
    );
}

#[tokio::test]
async fn route_errors_are_json() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(app.clone(), "/collections/missing").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "collection_not_found");

    let (status, json) = get_json(app.clone(), "/collections/places/search?bbox=1,2,3").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_bbox");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&limit=0",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_limit");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&predicate=exact",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_predicate");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&level=item",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_level");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/items?bbox=-10,0,0,2&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/items?bbox=-10,0,0,2&level=entry",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&predicate=intersects",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_predicate");
}
