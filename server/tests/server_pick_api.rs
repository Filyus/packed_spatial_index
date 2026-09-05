//! `/collections/{id}/pick`: the click's ordered broad phase over a 3D
//! collection — a ray plus a pixel frustum, candidates on-ray-first,
//! near-to-far.

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

/// A 3x3x3 grid of points spaced 10 apart, centered at (10,10,10).
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

fn cities_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "berlin", "geometry": {"type": "Point", "coordinates": [13.40, 52.52]}, "properties": {}}
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

#[tokio::test]
async fn orders_on_ray_boxes_first_near_to_far() {
    let app = router(state_over(grid_3d_geojson().as_bytes()));
    // A ray down the +x axis from far outside the grid: it pierces the three
    // points on the x axis, in increasing x, and grazes nothing at this
    // tolerance.
    let (status, json) = get_json(
        app,
        "/collections/places/pick?origin=-100,0,0&dir=1,0,0&halfAngle=1&limit=10",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["limit"], 10);
    assert_eq!(json["numberReturned"], 3);
    // The three pierced boxes, near-to-far: (0,0,0), (10,0,0), (20,0,0).
    assert_eq!(entries(&json), vec![0, 9, 18]);
    let items = json["items"].as_array().unwrap();
    for item in items {
        assert_eq!(item["distanceSquared"], 0.0, "{item}");
    }
    let ts: Vec<f64> = items
        .iter()
        .map(|item| item["entryT"].as_f64().unwrap())
        .collect();
    assert!((ts[0] - 100.0).abs() < 1e-9, "{ts:?}"); // entry at x=0 from x=-100
    assert!(ts.windows(2).all(|w| w[0] < w[1]), "near-to-far: {ts:?}");
}

#[tokio::test]
async fn limit_defaults_to_one_and_truncates_in_order() {
    let app = router(state_over(grid_3d_geojson().as_bytes()));
    let app2 = app.clone();
    let (status, json) =
        get_json(app, "/collections/places/pick?origin=-100,0,0&dir=1,0,0&halfAngle=1").await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["limit"], 1);
    assert_eq!(entries(&json), vec![0]);
    assert_eq!(json["numberReturned"], 1);

    let (status, json) = get_json(
        app2,
        "/collections/places/pick?origin=-100,0,0&dir=1,0,0&halfAngle=1&limit=2",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(entries(&json), vec![0, 9]);
}

#[tokio::test]
async fn off_axis_points_are_qualified_by_the_key() {
    let app = router(state_over(grid_3d_geojson().as_bytes()));
    // The ray runs down the x axis through three points; a wide tolerance
    // admits off-axis points as grazes. They must all come after the pierced
    // ones, ordered by perpendicular distance, each carrying an infinite
    // entry_t (serialized as JSON null).
    let (status, json) = get_json(
        app,
        "/collections/places/pick?origin=-100,0,0&dir=1,0,0&halfAngle=5&limit=10",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    let items = json["items"].as_array().unwrap();
    // 3 pierced + the 3 off-axis points the 5-degree cone still reaches at
    // the far column (radius ~10.5 world units at t=120, grid pitch 10).
    assert_eq!(items.len(), 6);
    // First the three pierced boxes, near-to-far, distance 0.
    assert_eq!(&entries(&json)[..3], vec![0, 9, 18]);
    for item in &items[..3] {
        assert_eq!(item["distanceSquared"], 0.0, "{item}");
        assert!(!item["entryT"].is_null(), "{item}");
    }
    // Then the grazes: no pierced candidate remains, so every entry_t is
    // infinite and the perpendicular distance takes over, nondecreasing.
    let ds: Vec<f64> = items[3..]
        .iter()
        .map(|item| item["distanceSquared"].as_f64().unwrap())
        .collect();
    assert!(ds.windows(2).all(|w| w[0] <= w[1]), "{ds:?}");
    for item in &items[3..] {
        assert!(item["entryT"].is_null(), "{item}");
        assert!(item["distanceSquared"].as_f64().unwrap() > 0.0, "{item}");
    }
}

#[tokio::test]
async fn a_2d_collection_is_refused() {
    let app = router(state_over(cities_geojson()));
    let (status, json) =
        get_json(app, "/collections/places/pick?origin=0,0,0&dir=1,0,0&halfAngle=1").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{json}");
    assert_eq!(json["error"]["code"], "unsupported_query");
    assert!(json["error"]["message"].as_str().unwrap().contains("2D"));
}

#[tokio::test]
async fn bad_ray_and_angle_are_rejected() {
    let app = router(state_over(grid_3d_geojson().as_bytes()));

    // zero direction
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/pick?origin=0,0,0&dir=0,0,0&halfAngle=1",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");

    // missing halfAngle
    let (status, json) = get_json(app.clone(), "/collections/places/pick?origin=0,0,0&dir=1,0,0").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");

    // halfAngle out of range
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/pick?origin=0,0,0&dir=1,0,0&halfAngle=90",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["code"], "invalid_frustum");

    // negative near
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/pick?origin=0,0,0&dir=1,0,0&halfAngle=1&near=-1",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["code"], "invalid_frustum");

    // limit out of range
    let (status, json) = get_json(
        app.clone(),
        "/collections/places/pick?origin=0,0,0&dir=1,0,0&halfAngle=1&limit=0",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["code"], "invalid_limit");
}
