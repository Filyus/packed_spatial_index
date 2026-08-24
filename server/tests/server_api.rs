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

/// The same places without ids — what every Parquet and FlatGeobuf source
/// looks like, since neither scan ever assigns one.
fn anonymous_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-5.0, 1.0]},
                "properties": {"name": "west"}
            },
            {
                "type": "Feature",
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

/// A 5x5x5 grid of 3D points at x,y,z in {0, 10, 20, 30, 40}.
///
/// `elevations_geojson` has two features, which cannot show a frustum being
/// more selective than the box around it -- and that selectivity is the whole
/// reason to send a frustum over the wire instead of its corner bbox.
fn grid_3d_geojson() -> String {
    let mut features = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                let (fx, fy, fz) = (x as f64 * 10.0, y as f64 * 10.0, z as f64 * 10.0);
                features.push(format!(
                    r#"{{"type":"Feature","id":"p{x}{y}{z}","geometry":{{"type":"Point","coordinates":[{fx},{fy},{fz}]}},"properties":{{}}}}"#
                ));
            }
        }
    }
    format!(
        r#"{{"type":"FeatureCollection","features":[{}]}}"#,
        features.join(",")
    )
}

/// Six inward-pointing planes describing an axis-aligned box.
///
/// A frustum this shape must answer exactly like the same box sent as `bbox`,
/// which is the cleanest correctness check the wire format has.
fn box_planes(min: [f64; 3], max: [f64; 3]) -> String {
    let planes = [
        [1.0, 0.0, 0.0, -min[0]],
        [-1.0, 0.0, 0.0, max[0]],
        [0.0, 1.0, 0.0, -min[1]],
        [0.0, -1.0, 0.0, max[1]],
        [0.0, 0.0, 1.0, -min[2]],
        [0.0, 0.0, -1.0, max[2]],
    ];
    planes
        .iter()
        .flat_map(|p| p.iter())
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
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
                    "identityModes": ["ref"],
                    "countModes": ["records", "only"],
                    "queryShapes": ["bbox"]
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
                "count": "records",
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
                // `payload=full` reads the body, which is where the id is, so
                // withholding it from `featureRef` would hide nothing.
                "identity": "full",
                "count": "records",
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
                "count": "records",
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
    // Identity does not work the other way round: asking for an id does not
    // smuggle the payload body into the response.
    assert_eq!(json["matches"][0]["payload"]["kind"], "feature_json");
    assert!(json["matches"][0]["payload"].get("feature").is_none());

    // The default stays cheap.
    let (_, json) = get_json(app, "/collections/places/search?bbox=-10,0,0,2").await;
    assert_eq!(json["query"]["identity"], "ref");
    assert!(json["matches"][0]["featureRef"].get("featureId").is_none());
}

/// The returned GeoJSON feature carries its own `id`, so `payload=full` puts
/// the source id in the response whatever `identity` says. Withholding it from
/// `featureRef` there would hide nothing while costing a second parameter, so
/// the mode resolves up — and the echo says so.
#[tokio::test]
async fn payload_full_resolves_identity_up_because_it_reads_the_body_anyway() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&payload=full&identity=ref",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["identity"], "full");
    assert_eq!(json["matches"][0]["featureRef"]["featureId"], "west");
    // The same id, in the place a GeoJSON reader looks for it.
    assert_eq!(json["matches"][0]["payload"]["feature"]["id"], "west");
}

/// `identity=full` is accepted everywhere so a client need not vary its request
/// per collection, but a collection with no source id to give must not sell
/// one: `full` there would buy a page of body reads for a byte-identical
/// answer. It resolves down to `ref`, and the echo reports what applied.
#[tokio::test]
async fn identity_full_resolves_down_where_no_id_is_stored() {
    let plans = [
        // No room for an id: the fixed-width record has no such field.
        (PayloadPlan::RowRef, sample_geojson()),
        (PayloadPlan::RowWkb, sample_geojson()),
        (PayloadPlan::None, sample_geojson()),
        // Room for one, but the source supplied none. This is every artifact
        // built from Parquet or FlatGeobuf, whose scans never assign an id.
        (
            PayloadPlan::FeatureJson {
                properties: PropertyProjection::AllNonGeometry,
            },
            anonymous_geojson(),
        ),
    ];
    for (payload, doc) in plans {
        let app = router(state_with_geojson(payload.clone(), doc));

        let (status, listed) = get_json(app.clone(), "/collections").await;
        assert_eq!(status, StatusCode::OK);
        assert_contract(&listed[0]["capabilities"]["identityModes"], json!(["ref"]));

        let (status, plain) =
            get_json(app.clone(), "/collections/places/search?bbox=-10,0,30,5").await;
        assert_eq!(status, StatusCode::OK);
        let (status, asked) = get_json(
            app,
            "/collections/places/search?bbox=-10,0,30,5&identity=full",
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Accepted, but reported as the mode that actually applied...
        assert_eq!(asked["query"]["identity"], "ref", "{payload:?}");
        // ...and the records are the ones the cheap path already produced.
        assert_contract(&asked["matches"], plain["matches"].clone());
    }
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

/// `count=only` must return the number a full search would have reported —
/// that is the whole contract, and the cheap path is only worth having if it
/// cannot disagree with the expensive one.
#[tokio::test]
async fn count_only_agrees_with_a_full_search() {
    let app = router(state_with_payload(PayloadPlan::RowRef));
    for bbox in ["-10,0,0,2", "-180,-90,180,90", "100,100,101,101"] {
        let (status, full) = get_json(
            app.clone(),
            &format!("/collections/places/search?bbox={bbox}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, counted) = get_json(
            app.clone(),
            &format!("/collections/places/search?bbox={bbox}&count=only"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        assert_eq!(
            counted["numberMatched"], full["numberMatched"],
            "bbox {bbox}"
        );
        // The point of the mode: nothing is materialized.
        assert_eq!(counted["numberReturned"], 0);
        assert_contract(&counted["matches"], json!([]));
        assert_eq!(counted["query"]["count"], "only");
    }
}

/// A 3D collection counts through the same path with a six-number bbox.
#[tokio::test]
async fn count_only_serves_three_dimensional_collections() {
    let app = router(state_with_geojson_request(
        ConvertRequest {
            payload: PayloadPlan::RowWkb,
            ..ConvertRequest::default()
        },
        elevations_geojson(),
    ));
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=0,0,0,3,3,50&count=only",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["numberReturned"], 0);

    // Still a 3D collection: a four-number bbox is refused before counting.
    let (status, json) = get_json(app, "/collections/places/search?bbox=0,0,3,3&count=only").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_bbox");
}

/// The two cases where the index's own count is not the number asked for are
/// refused rather than answered approximately.
#[tokio::test]
async fn count_only_refuses_what_it_cannot_answer_from_the_index() {
    // Exact filtering narrows the match set after the index answers.
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&predicate=intersects&count=only",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    // A split feature is several entries, so an entry count is not a feature
    // count. Entry level still counts; feature level is refused, and the
    // capabilities say so before the request is made.
    let split = router(state_with_geojson_request(
        ConvertRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            payload: PayloadPlan::RowWkb,
            ..ConvertRequest::default()
        },
        antimeridian_geojson(),
    ));
    let (status, json) = get_json(split.clone(), "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(&json["capabilities"]["countModes"], json!(["records"]));

    let bbox = "bbox=-180,-90,180,90";
    let (status, json) = get_json(
        split.clone(),
        &format!("/collections/places/search?{bbox}&count=only"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    let (status, counted) = get_json(
        split.clone(),
        &format!("/collections/places/search?{bbox}&level=entry&count=only"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, full) = get_json(
        split,
        &format!("/collections/places/search?{bbox}&level=entry"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(counted["numberMatched"], full["numberMatched"]);
    // Two index entries for one source feature is exactly the case the
    // feature-level refusal exists for.
    assert_eq!(counted["numberMatched"], 2);
}

/// `count` is a /search knob, like payload, level and identity.
#[tokio::test]
async fn count_is_rejected_on_items() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/items?bbox=-10,0,0,2&count=only",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    // An unknown value is a malformed request, named like its siblings
    // (`invalid_level`, `invalid_payload`) rather than a shape this collection
    // cannot serve -- that distinction is what the 422s above are for.
    let (status, json) = get_json(app, "/collections/places/search?bbox=-10,0,0,2&count=all").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_count");
}

/// The sample places are ~3300 km apart (lon -5 and lon 25 near the equator),
/// so a radius can tell them apart without the test depending on a precise
/// geodesic.
#[tokio::test]
async fn a_radius_query_selects_by_distance() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));

    // The sample source declares planar edges, as GeoJSON without an `edges`
    // member does -- which is most real data -- so a spherical radius has to be
    // opted in. The refusal without it is asserted below.
    let near_west = "radius=-5,1,100000&nonplanar=treat_as_planar";
    let both = "radius=-5,1,5000000&nonplanar=treat_as_planar";
    let nowhere = "radius=100,50,1000&nonplanar=treat_as_planar";

    let (status, json) = get_json(
        app.clone(),
        &format!("/collections/places/search?{near_west}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["matches"][0]["featureRef"]["rowNumber"], 0);
    // The echoed query names the shape that applied, and only that one.
    assert_contract(&json["query"]["radius"], json!([-5.0, 1.0, 100_000.0]));
    assert!(json["query"]["bbox"].is_null());

    let (status, json) = get_json(app.clone(), &format!("/collections/places/search?{both}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 2);

    let (status, json) = get_json(
        app.clone(),
        &format!("/collections/places/search?{nowhere}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 0);

    // It really narrows: a box covering both features keeps both, the small cap
    // around one of them keeps one.
    let (status, boxed) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-180,-90,180,90",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(boxed["numberMatched"], 2);

    // /items is the GeoJSON view of the same search, and a radius is exactly
    // the kind of query it should answer.
    let (status, json) = get_json(app, &format!("/collections/places/items?{near_west}")).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(json["features"][0]["properties"]["name"], "west");
}

/// A radius narrows its candidates against source geometry whatever the
/// predicate says, so the shapes that describe only index matches must refuse
/// it -- the rule `gp2psindex query` already enforces for `--count --radius`.
#[tokio::test]
async fn a_radius_query_is_always_exact() {
    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?radius=-5,1,100000&count=only",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    // A spherical query against a column declaring planar edges is a different
    // question, so it is refused rather than quietly answered -- unless the
    // caller says to read the coordinates as planar.
    let (status, json) =
        get_json(app.clone(), "/collections/places/search?radius=-5,1,100000").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("spherical"),
        "{json}"
    );
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?radius=-5,1,100000&nonplanar=treat_as_planar",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["query"]["nonplanar"], "treat_as_planar");

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?radius=-5,1,100000&nonplanar=maybe",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_query");

    // An artifact with no geometry to filter against cannot answer one at all,
    // and its capabilities say so rather than letting the request find out.
    let ref_only = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(ref_only.clone(), "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(&json["capabilities"]["queryShapes"], json!(["bbox"]));
    let (status, json) = get_json(ref_only, "/collections/places/search?radius=-5,1,100000").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_predicate");

    let geometry = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(geometry, "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json["capabilities"]["queryShapes"],
        json!(["bbox", "radius"]),
    );
}

#[tokio::test]
async fn radius_queries_are_refused_where_they_cannot_apply() {
    // A spherical cap is a 2D query; a 3D artifact takes bbox or frustum.
    let three_d = router(grid_3d_state());
    let (status, json) = get_json(three_d, "/collections/places/search?radius=1,1,1000").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    let app = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));

    // One shape per query.
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&radius=-5,1,1000",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_query");

    for raw in [
        "1,2",         // wrong arity
        "1,2,3,4",     // wrong arity
        "x,1,1000",    // not a number
        "inf,1,1000",  // not finite
        "-200,1,1000", // lon out of range
        "1,100,1000",  // lat out of range
        "1,1,0",       // a zero radius selects nothing by construction
        "1,1,-5",      // negative
    ] {
        let (status, json) = get_json(
            app.clone(),
            &format!("/collections/places/search?radius={raw}"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "raw {raw}: {json}");
        assert_eq!(json["error"]["code"], "invalid_radius", "raw {raw}");
    }
}

fn grid_3d_state() -> ServerState {
    state_with_geojson_request(
        ConvertRequest {
            payload: PayloadPlan::RowWkb,
            ..ConvertRequest::default()
        },
        grid_3d_geojson().as_bytes(),
    )
}

/// A frustum whose six planes describe an axis-aligned box must answer exactly
/// like that box sent as `bbox`. Anything else means the planes are being read
/// with the wrong sign, order, or arity.
#[tokio::test]
async fn an_axis_aligned_frustum_answers_like_the_same_box() {
    let app = router(grid_3d_state());
    for (min, max) in [
        ([-1.0, -1.0, -1.0], [41.0, 41.0, 41.0]),
        ([-1.0, -1.0, -1.0], [11.0, 11.0, 11.0]),
        ([9.0, 9.0, 9.0], [21.0, 21.0, 21.0]),
        ([100.0, 100.0, 100.0], [200.0, 200.0, 200.0]),
    ] {
        let bbox = format!(
            "bbox={},{},{},{},{},{}",
            min[0], min[1], min[2], max[0], max[1], max[2]
        );
        let frustum = format!("frustum={}", box_planes(min, max));
        let (status, by_box) = get_json(
            app.clone(),
            &format!("/collections/places/search?{bbox}&limit=1000"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, by_frustum) = get_json(
            app.clone(),
            &format!("/collections/places/search?{frustum}&limit=1000"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{by_frustum}");

        assert_eq!(
            by_frustum["numberMatched"], by_box["numberMatched"],
            "planes {min:?}..{max:?}"
        );
        assert_contract(&by_frustum["matches"], by_box["matches"].clone());
        // The echoed query reports the shape that applied, and only that one.
        assert!(by_frustum["query"]["bbox"].is_null());
        assert_eq!(by_frustum["query"]["frustum"].as_array().unwrap().len(), 6);
        assert!(by_box["query"]["frustum"].is_null());
    }
}

/// The reason a frustum travels over the wire at all: a tilted view selects far
/// less than the box around it, so a client that can only send a bbox pays for
/// everything it will never draw.
#[tokio::test]
async fn a_tilted_frustum_selects_less_than_its_corner_box() {
    let app = router(grid_3d_state());
    // A diagonal slab: |x - y| <= 5, clipped to the grid. Its corner bbox is
    // the whole grid, but it keeps only the diagonal.
    let planes = [
        [1.0, -1.0, 0.0, 5.0],
        [-1.0, 1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 41.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, -1.0, 41.0],
    ];
    let frustum = planes
        .iter()
        .flat_map(|p| p.iter())
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let (status, tilted) = get_json(
        app.clone(),
        &format!("/collections/places/search?frustum={frustum}&count=only"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{tilted}");
    let (status, corner_box) = get_json(
        app,
        "/collections/places/search?bbox=-1,-1,-1,41,41,41&count=only",
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let tilted = tilted["numberMatched"].as_u64().unwrap();
    let whole = corner_box["numberMatched"].as_u64().unwrap();
    assert_eq!(whole, 125, "the corner box is the whole grid");
    assert!(
        tilted > 0 && tilted < whole / 2,
        "frustum matched {tilted} of {whole}: it is not pruning"
    );
}

/// `count=only` over a frustum is what closes the loop with the core's
/// `count_region`: counting a shape query without materializing it.
#[tokio::test]
async fn count_only_works_over_a_frustum() {
    let app = router(grid_3d_state());
    let frustum = box_planes([9.0, 9.0, 9.0], [21.0, 21.0, 21.0]);
    let (status, counted) = get_json(
        app.clone(),
        &format!("/collections/places/search?frustum={frustum}&count=only"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, full) = get_json(
        app,
        &format!("/collections/places/search?frustum={frustum}&limit=1000"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(counted["numberMatched"], full["numberMatched"]);
    assert_eq!(counted["numberReturned"], 0);
}

#[tokio::test]
async fn frustum_queries_are_refused_where_they_cannot_apply() {
    let three_d = router(grid_3d_state());
    let planes = box_planes([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);

    // A 2D artifact has no z for a frustum to test against, and its
    // capabilities say so before the request is made.
    let two_d = router(state_with_payload(PayloadPlan::RowRef));
    let (status, json) = get_json(two_d.clone(), "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(&json["capabilities"]["queryShapes"], json!(["bbox"]));
    let (status, json) = get_json(
        two_d,
        &format!("/collections/places/search?frustum={planes}"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");

    let (status, json) = get_json(three_d.clone(), "/collections/places").await;
    assert_eq!(status, StatusCode::OK);
    assert_contract(
        &json["capabilities"]["queryShapes"],
        json!(["bbox", "frustum"]),
    );

    // No narrow phase exists for a frustum, so exact filtering cannot apply.
    let (status, json) = get_json(
        three_d.clone(),
        &format!("/collections/places/search?frustum={planes}&predicate=intersects"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_predicate");

    // One shape per query.
    let (status, json) = get_json(
        three_d.clone(),
        &format!("/collections/places/search?bbox=0,0,0,1,1,1&frustum={planes}"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_query");

    // /items is the GeoJSON view; the frustum belongs to /search.
    let geojson = router(state_with_payload(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }));
    let (status, json) = get_json(
        geojson,
        &format!("/collections/places/items?frustum={planes}"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");
}

/// A malformed frustum is named as such rather than folded into invalid_query.
#[tokio::test]
async fn malformed_frustum_planes_are_rejected() {
    let app = router(grid_3d_state());
    let twenty_three = vec!["1"; 23].join(",");
    for raw in [
        // wrong arity
        "1,2,3",
        &twenty_three,
        // not a number, then not finite
        "a,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
        "inf,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
        // A zero normal constrains nothing, which would silently widen the
        // frustum instead of failing.
        "0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1",
    ] {
        let (status, json) = get_json(
            app.clone(),
            &format!("/collections/places/search?frustum={raw}"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "raw {raw}: {json}");
        assert_eq!(json["error"]["code"], "invalid_frustum", "raw {raw}");
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

/// CORS is opt-in per deployment: no configured origin means no headers, so a
/// browser page from elsewhere cannot read the responses.
#[tokio::test]
async fn cross_origin_reads_are_opt_in() {
    let state = state_with_payload(PayloadPlan::RowRef);
    let closed = router(state.clone());
    let response = closed
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("origin", "https://example.org")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );

    let open = packed_spatial_index_server::router_with_cors(
        state.clone(),
        &["https://example.org".to_string()],
    )
    .unwrap();
    let response = open
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("origin", "https://example.org")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .map(|v| v.to_str().unwrap()),
        Some("https://example.org")
    );

    // A different origin is not on the list, so it gets nothing.
    let open =
        packed_spatial_index_server::router_with_cors(state, &["https://example.org".to_string()])
            .unwrap();
    let response = open
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("origin", "https://evil.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );

    assert!(
        packed_spatial_index_server::router_with_cors(
            state_with_payload(PayloadPlan::RowRef),
            &["not a header value\n".to_string()]
        )
        .is_err()
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
        app.clone(),
        "/collections/places/search?bbox=-10,0,0,2&level=feature",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_level");

    // An exact predicate is refused for the reason that actually applies:
    // there is no stored geometry to refine against, which is a different
    // failure from an artifact whose payload simply cannot answer one.
    let (status, json) = get_json(
        app,
        "/collections/places/search?bbox=-10,0,0,2&predicate=intersects",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_predicate");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no geometry payload"),
        "{json}"
    );
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
