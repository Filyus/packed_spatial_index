//! Integration tests for `GET /collections/{id}/join/{other}` — the distance
//! join endpoint. Self-contained, like the other API test files.

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

/// Three well-separated points: a0-a1 = 10 apart, a2 far from both.
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

/// The same grid shifted: b0 sits 1.0 from a0, b1 2.0 from a1, b2 1.0 from
/// a2, and b0 lies 9.0 from a1 — every other cross pair is much farther.
fn grid_points_shifted_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "b0", "geometry": {"type": "Point", "coordinates": [1.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "b1", "geometry": {"type": "Point", "coordinates": [12.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "b2", "geometry": {"type": "Point", "coordinates": [100.0, 101.0]}, "properties": {}}
        ]
    }"#
}

fn big_polygon_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "yard", "geometry": {"type": "Polygon", "coordinates": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]]]}, "properties": {}}
        ]
    }"#
}

/// A 5x5x5 grid of 3D points at x,y,z in {0, 10, 20, 30, 40}.
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

/// Two collections over two artifacts: `places` and `other` under `second_id`.
fn two_collection_state(first: &[u8], second: &[u8], second_id: &str) -> ServerState {
    let dir = tempdir().unwrap().keep();
    let data_dir = dir.join("data");
    fs::create_dir(&data_dir).unwrap();
    write_artifact(&data_dir.join("places.psindex"), first);
    write_artifact(&data_dir.join("other.psindex"), second);
    let catalog_text = format!(
        r#"
        [[collections]]
        id = "places"
        artifact = "data/places.psindex"

        [[collections]]
        id = "{second_id}"
        artifact = "data/other.psindex"
    "#
    );
    let catalog = Catalog::from_toml_str(&catalog_text, &dir).unwrap();
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

fn sorted_pairs(json: &Value) -> Vec<(u64, u64)> {
    let mut pairs: Vec<(u64, u64)> = json["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p["a"].as_u64().unwrap(), p["b"].as_u64().unwrap()))
        .collect();
    pairs.sort_unstable();
    pairs
}

#[tokio::test]
async fn join_pairs_match_distances() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));

    // a0-b0 = 1, a1-b1 = 2, a2-b2 = 1, a1-b0 = 9, everything else farther.
    // Pair order is traversal order and not part of the API, so the envelope
    // is checked field-by-field and the pairs as a set.
    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?epsilon=2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["collectionId"], "places");
    assert_eq!(json["joinCollectionId"], "roads");
    assert_eq!(json["epsilon"], 2.0);
    assert_eq!(json["count"], "records");
    assert_eq!(json["numberMatched"], 3);
    assert_eq!(json["numberReturned"], 3);
    assert_eq!(sorted_pairs(&json), vec![(0, 0), (1, 1), (2, 2)]);

    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?epsilon=9.5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 4);
    assert_eq!(sorted_pairs(&json), vec![(0, 0), (1, 0), (1, 1), (2, 2)]);

    // The bound is inclusive: a1-b0 is exactly 9.0 away.
    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?epsilon=9").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 4);

    // Distinct points never touch, so epsilon = 0 is empty here.
    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?epsilon=0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 0);

    // A large enough bound pairs everything: 3 x 3.
    let (status, json) = get_json(app, "/collections/places/join/roads?epsilon=200").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 9);
    assert_eq!(json["numberReturned"], 9);
}

#[tokio::test]
async fn join_epsilon_zero_answers_overlaps() {
    // One big polygon over the origin, probed by points inside and outside it.
    let points = br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "in", "geometry": {"type": "Point", "coordinates": [10.0, 10.0]}, "properties": {}},
            {"type": "Feature", "id": "out", "geometry": {"type": "Point", "coordinates": [30.0, 30.0]}, "properties": {}}
        ]
    }"#;
    let app = router(two_collection_state(big_polygon_geojson(), points, "roads"));

    // The polygon's box overlaps the inside point at distance 0; the outside
    // point is ~14.1 away.
    let (status, json) = get_json(app, "/collections/places/join/roads?epsilon=0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(sorted_pairs(&json), vec![(0, 0)]);
}

#[tokio::test]
async fn join_count_only_reports_total_without_pairs() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));
    let (status, json) = get_json(
        app,
        "/collections/places/join/roads?epsilon=9.5&count=only",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["count"], "only");
    assert_eq!(json["numberMatched"], 4);
    assert_eq!(json["numberReturned"], 0);
    assert_eq!(json["pairs"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn join_limit_truncates_but_counts_through() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));
    let (status, json) = get_json(
        app,
        "/collections/places/join/roads?epsilon=9.5&limit=2",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 4, "limit must not change the total");
    assert_eq!(json["numberReturned"], 2);
    assert_eq!(json["pairs"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn join_with_self_reports_each_unordered_pair_once() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));

    // Within `places`: a0-a1 = 10 apart, a2 far from both. The unordered
    // pair (0, 1) appears exactly once, with `a` below `b`.
    let (status, json) =
        get_json(app.clone(), "/collections/places/join/places?epsilon=10.5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["collectionId"], "places");
    assert_eq!(json["joinCollectionId"], "places");
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(sorted_pairs(&json), vec![(0, 1)]);

    // The cross join against the shifted copy still reports both directions.
    let (status, json) = get_json(app, "/collections/places/join/roads?epsilon=2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 3);
}

#[tokio::test]
async fn join_rejects_bad_requests() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));

    // Unknown right-hand collection.
    let (status, json) = get_json(app.clone(), "/collections/places/join/nope?epsilon=1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "collection_not_found");

    // Missing, non-numeric, negative, and NaN epsilons.
    for (uri, code) in [
        ("/collections/places/join/roads", "invalid_epsilon"),
        ("/collections/places/join/roads?epsilon=abc", "invalid_epsilon"),
        ("/collections/places/join/roads?epsilon=-1", "invalid_epsilon"),
        ("/collections/places/join/roads?epsilon=NaN", "invalid_epsilon"),
    ] {
        let (status, json) = get_json(app.clone(), uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "uri: {uri}");
        assert_eq!(json["error"]["code"], code, "uri: {uri}");
    }

    // Bad limit and unknown parameters.
    let (status, json) =
        get_json(app.clone(), "/collections/places/join/roads?epsilon=1&limit=0").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_limit");

    let (status, json) = get_json(app, "/collections/places/join/roads?epsilon=1&offset=4").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_query");
}

#[tokio::test]
async fn join_cross_dimension_rejected() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_3d_geojson().as_bytes(),
        "grid3d",
    ));
    let (status, json) = get_json(app, "/collections/places/join/grid3d?epsilon=1").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");
}

#[tokio::test]
async fn join_3d_collections() {
    // A 5x5x5 grid against an identical copy: epsilon = 5 pairs only the
    // co-located entries, one per grid point.
    let app = router(two_collection_state(
        grid_3d_geojson().as_bytes(),
        grid_3d_geojson().as_bytes(),
        "gridcopy",
    ));
    let (status, json) = get_json(app, "/collections/places/join/gridcopy?epsilon=5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 125);
    assert_eq!(sorted_pairs(&json).len(), 125);
}
