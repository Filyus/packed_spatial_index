//! Integration tests for `GET /collections/{id}/join/{other}` — the distance
//! join endpoint. Self-contained, like the other API test files.

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
    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?within=2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["collectionId"], "places");
    assert_eq!(json["joinCollectionId"], "roads");
    assert_eq!(json["within"], 2.0);
    assert_eq!(json["count"], "records");
    assert_eq!(json["numberMatched"], 3);
    assert_eq!(json["numberReturned"], 3);
    assert_eq!(sorted_pairs(&json), vec![(0, 0), (1, 1), (2, 2)]);

    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?within=9.5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 4);
    assert_eq!(sorted_pairs(&json), vec![(0, 0), (1, 0), (1, 1), (2, 2)]);

    // The bound is inclusive: a1-b0 is exactly 9.0 away.
    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?within=9").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 4);

    // Distinct points never touch, so max_distance = 0 is empty here.
    let (status, json) = get_json(app.clone(), "/collections/places/join/roads?within=0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 0);

    // A large enough bound pairs everything: 3 x 3.
    let (status, json) = get_json(app, "/collections/places/join/roads?within=200").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 9);
    assert_eq!(json["numberReturned"], 9);
}

#[tokio::test]
async fn join_within_zero_answers_overlaps() {
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
    let (status, json) = get_json(app, "/collections/places/join/roads?within=0").await;
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
    let (status, json) =
        get_json(app, "/collections/places/join/roads?within=9.5&count=only").await;
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
    let (status, json) = get_json(app, "/collections/places/join/roads?within=9.5&limit=2").await;
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
    let (status, json) = get_json(app.clone(), "/collections/places/join/places?within=10.5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["collectionId"], "places");
    assert_eq!(json["joinCollectionId"], "places");
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(sorted_pairs(&json), vec![(0, 1)]);

    // The cross join against the shifted copy still reports both directions.
    let (status, json) = get_json(app, "/collections/places/join/roads?within=2").await;
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
    let (status, json) = get_json(app.clone(), "/collections/places/join/nope?within=1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "collection_not_found");

    // Missing, non-numeric, negative, and NaN max_distances.
    for (uri, code) in [
        ("/collections/places/join/roads", "invalid_within"),
        (
            "/collections/places/join/roads?within=abc",
            "invalid_within",
        ),
        ("/collections/places/join/roads?within=-1", "invalid_within"),
        (
            "/collections/places/join/roads?within=NaN",
            "invalid_within",
        ),
    ] {
        let (status, json) = get_json(app.clone(), uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "uri: {uri}");
        assert_eq!(json["error"]["code"], code, "uri: {uri}");
    }

    // Bad limit and unknown parameters.
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/join/roads?within=1&limit=0",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_limit");

    let (status, json) = get_json(app, "/collections/places/join/roads?within=1&offset=4").await;
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
    let (status, json) = get_json(app, "/collections/places/join/grid3d?within=1").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");
}

#[tokio::test]
async fn join_3d_collections() {
    // A 5x5x5 grid against an identical copy: max_distance = 5 pairs only the
    // co-located entries, one per grid point.
    let app = router(two_collection_state(
        grid_3d_geojson().as_bytes(),
        grid_3d_geojson().as_bytes(),
        "gridcopy",
    ));
    let (status, json) = get_json(app, "/collections/places/join/gridcopy?within=5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 125);
    assert_eq!(sorted_pairs(&json).len(), 125);
}

// ---------------------------------------------------------------------------
// `GET /collections/{id}/anti-join/{other}` — the noise side of the same graph
// ---------------------------------------------------------------------------

fn items(json: &Value) -> Vec<u64> {
    let mut items: Vec<u64> = json["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap())
        .collect();
    items.sort_unstable();
    items
}

#[tokio::test]
async fn anti_join_reports_entries_with_no_partner() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));

    // a0-b0 = 1, a1-b1 = 2, a2-b2 = 1: at max_distance = 2 every entry is paired.
    let (status, json) =
        get_json(app.clone(), "/collections/places/anti-join/roads?within=2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["collectionId"], "places");
    assert_eq!(json["joinCollectionId"], "roads");
    assert_eq!(json["within"], 2.0);
    assert_eq!(json["count"], "records");
    assert_eq!(json["numberMatched"], 0);
    assert_eq!(items(&json), Vec::<u64>::new());

    // Below a1's nearest partner (2.0) it stands alone; the bound is
    // inclusive, so 2.0 itself already pairs it (checked above).
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/anti-join/roads?within=1.9",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 1);
    assert_eq!(items(&json), vec![1]);

    // Nothing within zero: distinct points never touch, so every entry is
    // unpaired.
    let (status, json) =
        get_json(app.clone(), "/collections/places/anti-join/roads?within=0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 3);
    assert_eq!(items(&json), vec![0, 1, 2]);

    // count=only reports the total and ships nothing.
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/anti-join/roads?within=0&count=only",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["count"], "only");
    assert_eq!(json["numberMatched"], 3);
    assert_eq!(json["numberReturned"], 0);
    assert_eq!(items(&json), Vec::<u64>::new());

    // limit truncates the array; numberMatched keeps counting through.
    let (status, json) =
        get_json(app, "/collections/places/anti-join/roads?within=0&limit=2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["numberMatched"], 3);
    assert_eq!(json["numberReturned"], 2);
    assert_eq!(items(&json).len(), 2);
}

#[tokio::test]
async fn anti_join_with_self_is_refused() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));
    // Every entry is at distance zero from itself, so the literal answer is
    // always empty. The endpoint says so and points at /components instead of
    // quietly answering a different question.
    let (status, json) = get_json(app, "/collections/places/anti-join/places?within=1").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("/components"),
        "{json}"
    );
}

#[tokio::test]
async fn anti_join_rejects_bad_requests() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));

    let (status, json) = get_json(app.clone(), "/collections/places/anti-join/nope?within=1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "collection_not_found");

    for uri in [
        "/collections/places/anti-join/roads",
        "/collections/places/anti-join/roads?within=abc",
        "/collections/places/anti-join/roads?within=-1",
        "/collections/places/anti-join/roads?within=NaN",
    ] {
        let (status, json) = get_json(app.clone(), uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "uri: {uri}");
        assert_eq!(json["error"]["code"], "invalid_within", "uri: {uri}");
    }

    let (status, json) = get_json(
        app.clone(),
        "/collections/places/anti-join/roads?within=1&limit=0",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_limit");

    let (status, json) =
        get_json(app, "/collections/places/anti-join/roads?within=1&offset=4").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_query");
}

#[tokio::test]
async fn anti_join_cross_dimension_rejected() {
    let app = router(two_collection_state(
        grid_points_geojson(),
        grid_3d_geojson().as_bytes(),
        "grid3d",
    ));
    let (status, json) = get_json(app, "/collections/places/anti-join/grid3d?within=1").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "unsupported_query");
}

// ---------------------------------------------------------------------------
// `GET /collections/{id}/components` — components of one proximity graph
// ---------------------------------------------------------------------------

/// Four points: a chain 0-1-2 each 3.0 from the next, and one far away.
/// At max_distance = 3 the chain is one component even though its ends are 6.0
/// apart — proximity is not transitive, and this is exactly the case that
/// shows why the labels are not clusters.
fn chain_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "c0", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "c1", "geometry": {"type": "Point", "coordinates": [3.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "c2", "geometry": {"type": "Point", "coordinates": [6.0, 0.0]}, "properties": {}},
            {"type": "Feature", "id": "lone", "geometry": {"type": "Point", "coordinates": [500.0, 500.0]}, "properties": {}}
        ]
    }"#
}

fn labels(json: &Value) -> Vec<u64> {
    json["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l.as_u64().unwrap())
        .collect()
}

#[tokio::test]
async fn components_label_the_chain_and_the_isolated_entry() {
    let app = router(two_collection_state(
        chain_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));

    let (status, json) = get_json(app.clone(), "/collections/places/components?within=3").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["collectionId"], "places");
    assert_eq!(json["within"], 3.0);
    assert_eq!(json["count"], "records");
    assert_eq!(json["itemCount"], 4);
    // The chain collapses into one component labelled with its minimum id;
    // the far entry is its own label. Two components, not one cluster each.
    assert_eq!(json["componentCount"], 2);
    assert_eq!(labels(&json), vec![0, 0, 0, 3]);

    // Below the chain's link length every entry stands alone.
    let (status, json) = get_json(app.clone(), "/collections/places/components?within=2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["componentCount"], 4);
    assert_eq!(labels(&json), vec![0, 1, 2, 3]);

    // Distinct points never touch, so max_distance = 0 is all singletons too.
    let (status, json) = get_json(app.clone(), "/collections/places/components?within=0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["componentCount"], 4);

    // A bound large enough to reach the far entry merges everything.
    let (status, json) = get_json(app, "/collections/places/components?within=1000").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["componentCount"], 1);
    assert_eq!(labels(&json), vec![0, 0, 0, 0]);
}

#[tokio::test]
async fn components_count_only_omits_the_labels() {
    let app = router(two_collection_state(
        chain_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));
    let (status, json) = get_json(app, "/collections/places/components?within=3&count=only").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["count"], "only");
    assert_eq!(json["itemCount"], 4);
    assert_eq!(json["componentCount"], 2);
    assert_eq!(labels(&json), Vec::<u64>::new());
}

#[tokio::test]
async fn components_work_in_3d() {
    // The 5x5x5 grid at spacing 10: below the spacing every point is its own
    // component, above it the whole grid is one.
    let app = router(two_collection_state(
        grid_3d_geojson().as_bytes(),
        grid_points_shifted_geojson(),
        "roads",
    ));
    let (status, json) = get_json(app.clone(), "/collections/places/components?within=9").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["itemCount"], 125);
    assert_eq!(json["componentCount"], 125);

    let (status, json) = get_json(app, "/collections/places/components?within=10").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["componentCount"], 1);
}

#[tokio::test]
async fn components_reject_bad_requests() {
    let app = router(two_collection_state(
        chain_geojson(),
        grid_points_shifted_geojson(),
        "roads",
    ));

    let (status, json) = get_json(app.clone(), "/collections/nope/components?within=1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "collection_not_found");

    for uri in [
        "/collections/places/components",
        "/collections/places/components?within=abc",
        "/collections/places/components?within=-1",
        "/collections/places/components?within=NaN",
    ] {
        let (status, json) = get_json(app.clone(), uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "uri: {uri}");
        assert_eq!(json["error"]["code"], "invalid_within", "uri: {uri}");
    }

    // `labels` is one entry per index entry by definition, so there is no
    // limit to accept.
    let (status, json) = get_json(app, "/collections/places/components?within=1&limit=2").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_query");
}
