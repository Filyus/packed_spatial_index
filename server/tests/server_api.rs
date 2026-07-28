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

fn elevations_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "low",
                "geometry": {"type": "Point", "coordinates": [1.0, 1.0, 10.0]},
                "properties": {"name": "low"}
            },
            {
                "type": "Feature",
                "id": "high",
                "geometry": {"type": "Point", "coordinates": [2.0, 2.0, 90.0]},
                "properties": {"name": "high"}
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

fn state_with_catalog(payload: PayloadPlan, catalog_text: &str) -> ServerState {
    let dir = tempdir().unwrap().keep();
    let data_dir = dir.join("data");
    fs::create_dir(&data_dir).unwrap();
    let artifact = data_dir.join("places.psindex");
    write_artifact_with_request(
        &artifact,
        ConvertRequest {
            payload,
            ..ConvertRequest::default()
        },
        sample_geojson(),
    );
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

async fn request_json(
    app: axum::Router,
    method: &str,
    uri: &str,
) -> (StatusCode, Option<String>, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let allow = response
        .headers()
        .get("allow")
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, allow, serde_json::from_slice(&bytes).unwrap())
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
                    "payloadModes": ["none", "summary", "full"],
                    "identityModes": ["ref", "full"]
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
                "identity": "ref",
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
                "identity": "ref",
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
                        "rowNumber": 0
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
async fn contract_search_feature_json_summary_shape() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
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
                "identity": "ref",
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
                        "rowNumber": 0
                    },
                    "payload": {"kind": "feature_json"}
                }
            ]
        }),
    );
}

/// A match record describes the artifact, not the code path that found it, so
/// swapping the predicate must not change its shape.
#[tokio::test]
async fn search_records_do_not_depend_on_the_predicate() {
    for payload in [
        PayloadPlan::RowWkb,
        PayloadPlan::FeatureJson {
            properties: PropertyProjection::AllNonGeometry,
        },
    ] {
        let app = router(state_with_payload(payload));
        let (bbox_status, bbox) =
            get_json(app.clone(), "/collections/places/search?bbox=-10,0,0,2").await;
        let (exact_status, exact) = get_json(
            app,
            "/collections/places/search?bbox=-10,0,0,2&predicate=intersects",
        )
        .await;
        assert_eq!(bbox_status, StatusCode::OK);
        assert_eq!(exact_status, StatusCode::OK);
        assert_contract(&exact["matches"], bbox["matches"].clone());
    }
}

#[tokio::test]
async fn identity_full_adds_the_source_feature_id() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&identity=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["identity"], "full");
    assert_eq!(json["matches"][0]["featureRef"]["featureId"], "west");
    // Identity is orthogonal to payload: asking for an id does not smuggle the
    // payload body into the response.
    assert_eq!(json["matches"][0]["payload"]["kind"], "feature_json");
    assert!(json["matches"][0]["payload"].get("feature").is_none());

    // The default stays cheap, and `payload=full` alone does not opt in.
    let (_, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&payload=full",
    )
    .await;
    assert_eq!(json["query"]["identity"], "ref");
    assert!(json["matches"][0]["featureRef"].get("featureId").is_none());
}

#[tokio::test]
async fn identity_full_resolves_ids_for_the_requested_page() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,30,5&identity=full&limit=1&offset=1",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);
    assert_eq!(json["numberReturned"], 1);
    assert_eq!(json["matches"][0]["featureRef"]["featureId"], "east");
}

#[tokio::test]
async fn identity_full_records_do_not_depend_on_the_predicate() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (_, bbox) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&identity=full",
    )
    .await;
    let (_, exact) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&identity=full&predicate=intersects",
    )
    .await;
    assert_contract(&exact["matches"], bbox["matches"].clone());
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

/// Paging happens inside the artifact when entries never duplicate a source
/// row, and in the server when feature-level grouping has to run first. Both
/// must produce the same page and the same `numberMatched`.
#[tokio::test]
async fn paged_and_grouped_searches_agree() {
    let split = state_with_geojson_request(
        ConvertRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            payload: PayloadPlan::RowWkb,
            ..ConvertRequest::default()
        },
        antimeridian_geojson(),
    );
    let plain = state_with_payload(PayloadPlan::RowWkb);

    for (state, expected_matched) in [(split, 1), (plain, 2)] {
        let app = router(state);
        let bbox = "bbox=-180,-90,180,90";
        let (status, whole) =
            get_json(app.clone(), &format!("/collections/places/search?{bbox}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(whole["numberMatched"], expected_matched);

        for offset in 0..=expected_matched {
            let (status, page) = get_json(
                app.clone(),
                &format!("/collections/places/search?{bbox}&limit=1&offset={offset}"),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(page["numberMatched"], expected_matched);
            assert_contract(
                &page["matches"],
                json!(
                    whole["matches"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .skip(offset)
                        .take(1)
                        .cloned()
                        .collect::<Vec<_>>()
                ),
            );
        }
    }
}

/// `search_records` has a whole 3D branch that no test reached: the server
/// suite contained no six-number bbox at all.
#[tokio::test]
async fn three_dimensional_collections_are_served() {
    let app = router(state_with_geojson_request(
        ConvertRequest {
            payload: PayloadPlan::RowWkb,
            ..ConvertRequest::default()
        },
        elevations_geojson(),
    ));

    let (status, json) = get_json(app.clone(), "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["dims"], "xyz");
    // Exact filtering is 2D-only, so a 3D collection must not advertise it.
    assert_contract(&json["capabilities"]["predicates"], json!(["bbox"]));

    // The z range is what separates the two features.
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=0,0,0,3,3,50&payload=full",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["matches"][0]["featureRef"]["rowNumber"], 0);

    let (status, json) =
        get_json(app.clone(), "/collections/places/search?bbox=0,0,0,3,3,100").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);

    // A 2D bbox cannot address a 3D collection.
    let (status, json) = get_json(app.clone(), "/collections/places/search?bbox=0,0,3,3").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_bbox");
    assert!(
        json["error"]["message"].as_str().unwrap().contains("minz"),
        "{json}"
    );

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=0,0,0,3,3,100&predicate=intersects",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_predicate");

    // /items is GeoJSON-only, and this artifact stores WKB.
    let (status, json) = get_json(app, "/collections/places/items?bbox=0,0,0,3,3,100").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_payload");
}

#[tokio::test]
async fn three_dimensional_collections_page_and_group() {
    let app = router(state_with_geojson_request(
        ConvertRequest {
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::AllNonGeometry,
            },
            ..ConvertRequest::default()
        },
        elevations_geojson(),
    ));
    let bbox = "bbox=0,0,0,3,3,100";

    let (status, whole) =
        get_json(app.clone(), &format!("/collections/places/search?{bbox}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(whole["numberMatched"], 2);

    for offset in 0..2 {
        let (status, page) = get_json(
            app.clone(),
            &format!("/collections/places/search?{bbox}&limit=1&offset={offset}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(page["numberMatched"], 2);
        assert_contract(&page["matches"][0], whole["matches"][offset].clone());
    }

    // `identity=full` reads bodies on the 3D path too.
    let (status, json) = get_json(
        app,
        &format!("/collections/places/search?{bbox}&identity=full&limit=1"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["matches"][0]["featureRef"]["featureId"], "low");
}

#[tokio::test]
async fn unknown_query_parameters_are_rejected() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&limitt=1",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_query");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("limitt"),
        "{json}"
    );

    // The rejection is about the name, not the value: a known parameter still
    // reaches the handler and gets the domain-specific error.
    let (status, json) = get_json(app, "/collections/places/search?bbox=-10,0,0,2&limit=x").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_limit");
}

/// A 500 says the server failed, not where it keeps its files. The artifact is
/// reopened per request, so a catalog entry whose file disappears after startup
/// reaches a client.
#[tokio::test]
async fn server_faults_do_not_name_the_artifact_path() {
    let dir = tempdir().unwrap().keep();
    let data_dir = dir.join("data");
    fs::create_dir(&data_dir).unwrap();
    let artifact = data_dir.join("places.psindex");
    write_artifact_with_request(
        &artifact,
        ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        },
        sample_geojson(),
    );
    let catalog = Catalog::from_toml_str(
        r#"
        [[collections]]
        id = "places"
        artifact = "data/places.psindex"
        "#,
        &dir,
    )
    .unwrap();
    let state = ServerState::from_catalog(catalog).unwrap();
    // The directory is cached at startup; the file itself is opened per query.
    fs::remove_file(&artifact).unwrap();

    let (status, json) = get_json(router(state), "/collections/places/search?bbox=-10,0,0,2").await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(json["error"]["code"], "io");
    let message = json["error"]["message"].as_str().unwrap();
    assert!(!message.contains("places.psindex"), "{message}");
    assert!(!message.contains(&dir.display().to_string()), "{message}");
    assert!(
        !message.contains(&artifact.display().to_string()),
        "{message}"
    );
}

#[tokio::test]
async fn unknown_routes_and_methods_use_the_error_envelope() {
    let app = router(state_with_payload(PayloadPlan::RowRef));

    let (status, _, json) = request_json(app.clone(), "GET", "/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_contract(
        &json,
        json!({"error": {"code": "not_found", "message": "no route for `/nope`"}}),
    );

    let (status, allow, json) = request_json(app.clone(), "POST", "/collections").await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    // The envelope replaces axum's empty body; it must not cost the `Allow`
    // header a client needs to learn what the route accepts.
    assert_eq!(allow.as_deref(), Some("GET,HEAD"));
    assert_contract(
        &json,
        json!({
            "error": {
                "code": "method_not_allowed",
                "message": "method not allowed: POST /collections"
            }
        }),
    );

    // The 405 fallback attaches to the routes registered so far, so it has to
    // cover the last one as well as the first.
    let (status, _, json) =
        request_json(app, "DELETE", "/collections/places/search?bbox=0,0,1,1").await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(json["error"]["code"], "method_not_allowed");
}

#[tokio::test]
async fn queries_over_the_catalog_limits_are_rejected() {
    let app = router(state_with_catalog(
        PayloadPlan::RowRef,
        r#"
        [server.limits]
        max_items = 1

        [[collections]]
        id = "places"
        artifact = "data/places.psindex"
        "#,
    ));
    // One match stays inside the budget.
    let (status, _) = get_json(app.clone(), "/collections/places/search?bbox=-10,0,0,2").await;
    assert_eq!(status, StatusCode::OK);

    // Two matches exceed it, and that is the client's problem, not a 500.
    let (status, json) = get_json(app, "/collections/places/search?bbox=-10,0,30,5").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "query_too_large");
}

#[tokio::test]
async fn zero_lifts_a_catalog_limit() {
    let app = router(state_with_catalog(
        PayloadPlan::RowRef,
        r#"
        [server.limits]
        max_items = 0

        [[collections]]
        id = "places"
        artifact = "data/places.psindex"
        "#,
    ));
    let (status, json) = get_json(app, "/collections/places/search?bbox=-10,0,30,5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);
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
        app.clone(),
        "/collections/places/items?bbox=-10,0,0,2&identity=full",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&identity=id",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_identity");

    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&predicate=intersects",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_predicate");
}
