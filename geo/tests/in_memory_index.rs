//! Coverage for the in-memory index queries, which the artifact path has no
//! counterpart for: nearest-neighbour, raycast, and the escape hatch to the
//! underlying core index.
#![cfg(feature = "geojson")]

use packed_spatial_index_geo::{
    BuildRequest, GeoIndex, IndexBuildOptions, Point2D, Point3D, Ray2D, Ray3D, StoragePrecision,
    open_geojson_slice,
};

/// Three points a degree apart along the equator, plus one far east.
fn places() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","id":"origin","geometry":{"type":"Point","coordinates":[0.0,0.0]},"properties":{}},
            {"type":"Feature","id":"east","geometry":{"type":"Point","coordinates":[1.0,0.0]},"properties":{}},
            {"type":"Feature","id":"north","geometry":{"type":"Point","coordinates":[0.0,1.0]},"properties":{}},
            {"type":"Feature","id":"far","geometry":{"type":"Point","coordinates":[40.0,0.0]},"properties":{}}
        ]
    }"#
}

fn towers() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","id":"low","geometry":{"type":"Point","coordinates":[0.0,0.0,1.0]},"properties":{}},
            {"type":"Feature","id":"mid","geometry":{"type":"Point","coordinates":[0.0,0.0,5.0]},"properties":{}},
            {"type":"Feature","id":"aside","geometry":{"type":"Point","coordinates":[9.0,9.0,9.0]},"properties":{}}
        ]
    }"#
}

fn build(doc: &[u8], precision: StoragePrecision) -> GeoIndex {
    let mut source = open_geojson_slice(doc).unwrap();
    source
        .build(BuildRequest {
            build: IndexBuildOptions {
                precision,
                ..IndexBuildOptions::default()
            },
            ..BuildRequest::default()
        })
        .unwrap()
}

#[test]
fn nearest_returns_features_in_distance_order() {
    let GeoIndex::D2(index) = build(places(), StoragePrecision::F64) else {
        panic!("expected a 2D index");
    };

    let nearest = index.nearest_feature_refs(Point2D::new(0.1, 0.0), 3);
    assert_eq!(nearest.len(), 3);
    assert_eq!(
        nearest
            .iter()
            .map(|(f, _)| f.row_number)
            .collect::<Vec<_>>(),
        vec![0, 1, 2],
        "origin, then east, then north"
    );
    assert!(
        nearest.windows(2).all(|pair| pair[0].1 <= pair[1].1),
        "distances are non-decreasing"
    );

    // Asking for more than exists returns what exists, not an error.
    assert_eq!(
        index.nearest_feature_refs(Point2D::new(0.0, 0.0), 99).len(),
        4
    );
    assert!(
        index
            .nearest_feature_refs(Point2D::new(0.0, 0.0), 0)
            .is_empty()
    );
}

/// Planar distance and great-circle distance disagree about which neighbour is
/// closer once longitude degrees stop being the same length as latitude ones.
#[test]
fn haversine_nearest_bounds_by_metres() {
    let GeoIndex::D2(index) = build(places(), StoragePrecision::F64) else {
        panic!("expected a 2D index");
    };

    // A degree of longitude at the equator is about 111 km.
    let near = index.nearest_feature_refs_haversine(0.0, 0.0, 10, 150_000.0);
    assert_eq!(
        near.iter().map(|(f, _)| f.row_number).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "the point 40 degrees east is far outside the radius"
    );
    assert!(near.iter().all(|(_, metres)| *metres <= 150_000.0));

    let tight = index.nearest_feature_refs_haversine(0.0, 0.0, 10, 1_000.0);
    assert_eq!(
        tight.len(),
        1,
        "only the point at the origin is within 1 km"
    );
}

#[test]
fn raycast_returns_crossed_features_and_the_closest_one() {
    let GeoIndex::D2(index) = build(places(), StoragePrecision::F64) else {
        panic!("expected a 2D index");
    };

    // Due east along the equator: crosses the origin and the point at x = 1.
    let ray = Ray2D::new(Point2D::new(-1.0, 0.0), 1.0, 0.0, 10.0);
    let mut crossed = index
        .raycast_feature_refs(ray)
        .iter()
        .map(|f| f.row_number)
        .collect::<Vec<_>>();
    crossed.sort_unstable();
    assert_eq!(crossed, vec![0, 1]);

    let (closest, t) = index
        .raycast_closest_feature_ref(ray)
        .expect("the ray crosses something");
    assert_eq!(closest.row_number, 0);
    assert!((0.0..=2.0).contains(&t), "t was {t}");

    // A ray pointing away from everything hits nothing.
    let away = Ray2D::new(Point2D::new(-1.0, 50.0), 1.0, 0.0, 10.0);
    assert!(index.raycast_feature_refs(away).is_empty());
    assert!(index.raycast_closest_feature_ref(away).is_none());
}

#[test]
fn three_dimensional_nearest_and_raycast() {
    let GeoIndex::D3(index) = build(towers(), StoragePrecision::F64) else {
        panic!("expected a 3D index");
    };

    let nearest = index.nearest_feature_refs(Point3D::new(0.0, 0.0, 0.0), 2);
    assert_eq!(
        nearest
            .iter()
            .map(|(f, _)| f.row_number)
            .collect::<Vec<_>>(),
        vec![0, 1],
        "the tower at z = 1 is closer than the one at z = 5"
    );

    // Straight up through both towers.
    let ray = Ray3D::new(Point3D::new(0.0, 0.0, -1.0), 0.0, 0.0, 1.0, 20.0);
    let mut crossed = index
        .raycast_feature_refs(ray)
        .iter()
        .map(|f| f.row_number)
        .collect::<Vec<_>>();
    crossed.sort_unstable();
    assert_eq!(crossed, vec![0, 1]);

    let (closest, _) = index
        .raycast_closest_feature_ref(ray)
        .expect("the ray crosses both towers");
    assert_eq!(closest.row_number, 0, "the lower tower is hit first");
}

/// `f32` storage rounds envelopes outward, so its candidate set is a superset.
/// It must still answer, and still agree on what is nearest.
#[test]
fn f32_indexes_answer_the_same_queries() {
    let GeoIndex::D2F32(index) = build(places(), StoragePrecision::F32) else {
        panic!("expected an f32 2D index");
    };
    let nearest = index.nearest_feature_refs(Point2D::new(0.1, 0.0), 1);
    assert_eq!(nearest[0].0.row_number, 0);

    let ray = Ray2D::new(Point2D::new(-1.0, 0.0), 1.0, 0.0, 10.0);
    assert!(!index.raycast_feature_refs(ray).is_empty());

    let GeoIndex::D3F32(index) = build(towers(), StoragePrecision::F32) else {
        panic!("expected an f32 3D index");
    };
    assert_eq!(
        index
            .nearest_feature_refs(Point3D::new(0.0, 0.0, 0.0), 1)
            .len(),
        1
    );
}

/// `raw_index` hands out the core index for callers that need an API this
/// crate does not wrap. It had no test at all.
#[test]
fn raw_index_exposes_the_core_index() {
    let GeoIndex::D2(index) = build(places(), StoragePrecision::F64) else {
        panic!("expected a 2D index");
    };
    assert_eq!(
        index.raw_index().num_items(),
        index.metadata.index_entry_count
    );
    assert_eq!(index.metadata.feature_count, 4);

    let GeoIndex::D3(index) = build(towers(), StoragePrecision::F64) else {
        panic!("expected a 3D index");
    };
    assert_eq!(
        index.raw_index().num_items(),
        index.metadata.index_entry_count
    );
}
