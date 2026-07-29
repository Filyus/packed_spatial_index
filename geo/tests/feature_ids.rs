#![cfg(feature = "geojson")]
//! What `stores_feature_ids` in the `geoM` manifest promises.
//!
//! It answers "can this artifact produce a source id at all", which the payload
//! plan alone cannot: the plan says where an id *would* live, and a reader that
//! keys on it pays a page of body reads on every `FeatureJson` artifact whose
//! source never supplied one.

use packed_spatial_index_geo::{
    ConvertRequest, GeoArtifactManifest, PayloadPlan, PropertyProjection, open_geojson_slice,
    read_geo_manifest,
};

fn geojson(ids: bool) -> Vec<u8> {
    let features: Vec<String> = ["west", "east"]
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let id = if ids {
                format!(r#""id":"{name}","#)
            } else {
                String::new()
            };
            format!(
                r#"{{"type":"Feature",{id}"geometry":{{"type":"Point","coordinates":[{i}.0,1.0]}},"properties":{{}}}}"#
            )
        })
        .collect();
    format!(
        r#"{{"type":"FeatureCollection","features":[{}]}}"#,
        features.join(",")
    )
    .into_bytes()
}

fn manifest(ids: bool, payload: PayloadPlan) -> GeoArtifactManifest {
    let source = geojson(ids);
    let mut dataset = open_geojson_slice(&source).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            payload,
            ..ConvertRequest::default()
        })
        .unwrap();
    read_geo_manifest(&bytes).unwrap().unwrap()
}

fn feature_json() -> PayloadPlan {
    PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }
}

#[test]
fn a_feature_json_artifact_reports_whether_its_source_had_ids() {
    assert_eq!(
        manifest(true, feature_json()).stores_feature_ids,
        Some(true)
    );
    // The same plan over a source with no ids. This is also the Parquet and
    // FlatGeobuf case: neither scan ever assigns `feature_id`, so every
    // artifact they produce lands here.
    assert_eq!(
        manifest(false, feature_json()).stores_feature_ids,
        Some(false)
    );
}

#[test]
fn the_other_plans_report_false_however_rich_the_source() {
    // The fixed-width record has no id field, so ids the scan collected are
    // simply not written. A reader keying on the payload plan alone would get
    // this one right; the `FeatureJson` case above is where it goes wrong.
    for payload in [PayloadPlan::RowRef, PayloadPlan::RowWkb, PayloadPlan::None] {
        let manifest = manifest(true, payload.clone());
        assert_eq!(manifest.stores_feature_ids, Some(false), "{payload:?}");
    }
}

/// Artifacts written before the field existed must stay readable, and must not
/// be read as "no ids" — an older `feature_json` artifact may well carry them.
#[test]
fn an_older_manifest_reports_unknown_rather_than_false() {
    let current = manifest(true, feature_json());
    let mut json = serde_json::to_value(&current).unwrap();
    json.as_object_mut().unwrap().remove("stores_feature_ids");

    let older: GeoArtifactManifest = serde_json::from_value(json).unwrap();
    assert_eq!(older.stores_feature_ids, None);
    assert_eq!(
        GeoArtifactManifest {
            stores_feature_ids: current.stores_feature_ids,
            ..older
        },
        current
    );
}
