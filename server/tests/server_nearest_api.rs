//! `/collections/{id}/nearest`: k nearest entries under a planar or spherical
//! metric, over the same cached owned index the join family uses.

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

/// Four lon/lat points: Berlin, Paris, Madrid and a point in the Atlantic.
/// Chosen so that planar and spherical orderings from a high-latitude query
/// point disagree — see `spherical_and_planar_orderings_differ`.
fn cities_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "berlin", "geometry": {"type": "Point", "coordinates": [13.40, 52.52]}, "properties": {}},
            {"type": "Feature", "id": "paris", "geometry": {"type": "Point", "coordinates": [2.35, 48.86]}, "properties": {}},
            {"type": "Feature", "id": "madrid", "geometry": {"type": "Point", "coordinates": [-3.70, 40.42]}, "properties": {}},
            {"type": "Feature", "id": "atlantic", "geometry": {"type": "Point", "coordinates": [-30.0, 60.0]}, "properties": {}}
        ]
    }"#
}

fn grid_3d_geojson() -> String {
    let mut features = Vec::new();
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
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

fn state_over(doc: &[u8]) -> ServerState {
    let dir = tempdir().unwrap().keep();
    let data_dir = dir.join("data");
    fs::create_dir(&data_dir).unwrap();
    write_artifact(&data_dir.join("places.psindex"), doc);
    let catalog = Catalog::from_toml_str(
        r#"
        [[collections]]
        id = "places"
        artifact = "data/places.psindex"
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

fn entries(json: &Value) -> Vec<u64> {
    json["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["entry"].as_u64().unwrap())
        .collect()
}

fn distances(json: &Value) -> Vec<f64> {
    json["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["distance"].as_f64().unwrap())
        .collect()
}

#[tokio::test]
async fn planar_is_the_default_on_geojson_and_orders_by_euclidean_degrees() {
    let app = router(state_over(cities_geojson()));
    // From Lisbon (-9.14, 38.72). In degrees Madrid is closest, then Paris.
    let (status, json) = get_json(app, "/collections/places/nearest?point=-9.14,38.72&k=2").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["metric"], "planar");
    assert_eq!(json["k"], 2);
    assert_eq!(json["numberReturned"], 2);
    assert_eq!(json["point"], serde_json::json!([-9.14, 38.72]));
    assert!(json.get("within").is_none(), "no cutoff was given: {json}");
    assert_eq!(entries(&json), vec![2, 1]);
    let d = distances(&json);
    assert!(d[0] < d[1], "nearest first: {d:?}");
    // Point entries: box distance is the exact point distance.
    let madrid = ((-3.70f64 + 9.14).powi(2) + (40.42f64 - 38.72).powi(2)).sqrt();
    assert!((d[0] - madrid).abs() < 1e-9, "{} vs {madrid}", d[0]);
}

#[tokio::test]
async fn spherical_needs_the_nonplanar_opt_in_on_planar_edges() {
    let app = router(state_over(cities_geojson()));
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/nearest?point=-9.14,38.72&k=1&metric=spherical",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{json}");
    assert_eq!(json["error"]["code"], "unsupported_query");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nonplanar=treat_as_planar"),
        "{json}"
    );

    let (status, json) = get_json(
        app,
        "/collections/places/nearest?point=-9.14,38.72&k=1&metric=spherical&nonplanar=treat_as_planar",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["metric"], "spherical");
    assert_eq!(entries(&json), vec![2]);
    // Lisbon-Madrid great-circle distance is about 503 km.
    let d = distances(&json)[0];
    assert!((490_000.0..520_000.0).contains(&d), "{d}");
}

#[tokio::test]
async fn spherical_and_planar_orderings_differ() {
    // From (-40, 70), high latitude, where a degree of longitude is a third
    // of a degree of latitude. In degrees: Atlantic 14.1, Madrid 46.8, Paris
    // 47.3, Berlin 56.2. On the sphere: Atlantic 1 204 km, Paris 3 233 km,
    // Berlin 3 288 km, Madrid 3 892 km. The two metrics agree on the first
    // answer and disagree on the rest.
    let app = router(state_over(cities_geojson()));
    let (_, planar) = get_json(app.clone(), "/collections/places/nearest?point=-40,70&k=4").await;
    let (_, spherical) = get_json(
        app,
        "/collections/places/nearest?point=-40,70&k=4&metric=spherical&nonplanar=treat_as_planar",
    )
    .await;
    // Both put the Atlantic point first and return everything.
    assert_eq!(entries(&planar)[0], 3);
    assert_eq!(entries(&spherical)[0], 3);
    assert_eq!(entries(&planar).len(), 4);
    assert_eq!(entries(&spherical).len(), 4);
    assert_eq!(entries(&planar), vec![3, 2, 1, 0], "{planar}");
    assert_eq!(entries(&spherical), vec![3, 1, 0, 2], "{spherical}");
    let d = distances(&spherical);
    assert!(d.windows(2).all(|w| w[0] <= w[1]), "nearest first: {d:?}");
}

#[tokio::test]
async fn within_caps_the_answer_and_is_echoed() {
    let app = router(state_over(cities_geojson()));
    // Only Madrid lies within 8 degrees of Lisbon.
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/nearest?point=-9.14,38.72&k=10&within=8",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["within"], 8.0);
    assert_eq!(entries(&json), vec![2]);
    assert_eq!(json["numberReturned"], 1);

    // Spherical cutoff is in metres.
    let (status, json) = get_json(
        app,
        "/collections/places/nearest?point=-9.14,38.72&k=10&within=600000&metric=spherical&nonplanar=treat_as_planar",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(entries(&json), vec![2]);
}

#[tokio::test]
async fn three_d_collections_take_three_coordinates_and_are_planar_only() {
    let doc = grid_3d_geojson();
    let app = router(state_over(doc.as_bytes()));
    let (status, json) = get_json(app.clone(), "/collections/places/nearest?point=1,1,1&k=2").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["metric"], "planar");
    let d = distances(&json);
    assert!((d[0] - 3f64.sqrt()).abs() < 1e-9, "{d:?}");
    assert!((d[1] - (1.0f64 + 1.0 + 81.0).sqrt()).abs() < 1e-9, "{d:?}");

    let (status, json) = get_json(app.clone(), "/collections/places/nearest?point=1,1&k=2").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["code"], "invalid_point");

    let (status, json) = get_json(
        app,
        "/collections/places/nearest?point=1,1,1&k=2&metric=spherical",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{json}");
    assert_eq!(json["error"]["code"], "unsupported_query");
}

#[tokio::test]
async fn rejects_bad_inputs_with_the_sibling_codes() {
    let app = router(state_over(cities_geojson()));
    for (uri, code) in [
        ("/collections/places/nearest?k=1", "invalid_point"),
        ("/collections/places/nearest?point=1&k=1", "invalid_point"),
        ("/collections/places/nearest?point=1,x&k=1", "invalid_point"),
        (
            "/collections/places/nearest?point=1,NaN&k=1",
            "invalid_point",
        ),
        ("/collections/places/nearest?point=1,2", "invalid_k"),
        ("/collections/places/nearest?point=1,2&k=0", "invalid_k"),
        ("/collections/places/nearest?point=1,2&k=10001", "invalid_k"),
        (
            "/collections/places/nearest?point=1,2&k=1&metric=taxicab",
            "invalid_metric",
        ),
        (
            "/collections/places/nearest?point=1,2&k=1&within=-1",
            "invalid_within",
        ),
        (
            "/collections/places/nearest?point=1,2&k=1&limit=3",
            "invalid_query",
        ),
        (
            "/collections/places/nearest?point=200,2&k=1&metric=spherical&nonplanar=treat_as_planar",
            "invalid_point",
        ),
    ] {
        let (status, json) = get_json(app.clone(), uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {json}");
        assert_eq!(json["error"]["code"], code, "{uri}: {json}");
    }

    let (status, json) = get_json(app, "/collections/nope/nearest?point=1,2&k=1").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{json}");
}

#[tokio::test]
async fn capabilities_advertise_the_metrics() {
    let app = router(state_over(cities_geojson()));
    let (status, json) = get_json(app, "/collections/places").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(
        json["capabilities"]["nearestMetrics"],
        serde_json::json!(["planar", "spherical"])
    );

    let doc = grid_3d_geojson();
    let app = router(state_over(doc.as_bytes()));
    let (_, json) = get_json(app, "/collections/places").await;
    assert_eq!(
        json["capabilities"]["nearestMetrics"],
        serde_json::json!(["planar"])
    );
}
