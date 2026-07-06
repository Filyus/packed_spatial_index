#![cfg(feature = "geojson")]

use std::cmp::Ordering;

use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, ConvertRequest, EnvelopePolicy, FeatureRef, GeoArtifactIndex,
    GeoError, GeoMatch, PayloadPlan, PropertyProjection, SliceReader, open_geo_index,
    open_geojson_slice,
};

fn split_and_points_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "crossing",
                "geometry": {"type": "LineString", "coordinates": [[170.0, 0.0], [-170.0, 1.0]]},
                "properties": {"name": "crossing"}
            },
            {
                "type": "Feature",
                "id": "west",
                "geometry": {"type": "Point", "coordinates": [-5.0, 1.0]},
                "properties": {"name": "west"}
            }
        ]
    }"#
}

fn split_artifact(payload: PayloadPlan) -> GeoArtifactIndex<SliceReader<Vec<u8>>> {
    let mut source = open_geojson_slice(split_and_points_geojson()).unwrap();
    let bytes = source
        .convert(ConvertRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            payload,
            ..ConvertRequest::default()
        })
        .unwrap();
    open_geo_index(SliceReader::new(bytes)).unwrap()
}

fn world() -> Box2D {
    Box2D::new(-180.0, -10.0, 180.0, 10.0)
}

#[test]
fn feature_level_collapses_split_entries() {
    let GeoArtifactIndex::D2(index) = split_artifact(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }) else {
        panic!("expected 2D artifact");
    };

    let matches = index.search_matches(world()).unwrap();
    assert_eq!(matches.len(), 3, "split line contributes two entries");

    let features = index.search_features(world()).unwrap();
    assert_eq!(features.len(), 2);
    assert!(features.iter().all(|f| f.part.is_none()));
    assert_eq!(features[0].row_number, 0);
    assert_eq!(features[1].row_number, 1);

    let feature_matches = index.search_feature_matches(world()).unwrap();
    assert_eq!(feature_matches.len(), 2);

    // The representative of the split feature is its lowest-part entry:
    // same entry_id and payload as the part-0 match from the raw search.
    let part0 = matches
        .iter()
        .find(|m| m.feature.row_number == 0 && m.feature.part == Some(0))
        .expect("part-0 entry present at entry level");
    let collapsed = &feature_matches[0];
    assert_eq!(collapsed.entry_id, part0.entry_id);
    assert_eq!(collapsed.payload, part0.payload);
    assert_eq!(collapsed.feature.part, None);
}

#[test]
fn dedupe_by_feature_runs_after_external_filtering() {
    let GeoArtifactIndex::D2(index) = split_artifact(PayloadPlan::RowRef) else {
        panic!("expected 2D artifact");
    };
    // Simulate a caller-side filter between search and dedupe: drop part 0,
    // keep part 1 — the survivor must still collapse into one feature record.
    let mut matches = index.search_matches(world()).unwrap();
    matches.retain(|m| m.feature.row_number != 0 || m.feature.part == Some(1));
    GeoMatch::dedupe_by_feature(&mut matches);
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].feature.row_number, 0);
    assert_eq!(matches[0].feature.part, None);
}

#[test]
fn feature_level_rejects_payload_less_artifacts() {
    let GeoArtifactIndex::D2(index) = split_artifact(PayloadPlan::None) else {
        panic!("expected 2D artifact");
    };
    assert!(matches!(
        index.search_features(world()),
        Err(GeoError::Stream(_))
    ));
    assert!(matches!(
        index.search_feature_matches(world()),
        Err(GeoError::Stream(_))
    ));
}

#[test]
fn feature_ref_ordering_and_identity() {
    let base = FeatureRef::row_number(7);
    let mut split_a = base.clone();
    split_a.part = Some(0);
    let mut split_b = base.clone();
    split_b.part = Some(1);
    assert!(split_a.same_feature(&split_b));
    assert_eq!(split_a.cmp_feature(&split_b), Ordering::Equal);
    assert_eq!(split_a.cmp_entry(&split_b), Ordering::Less);

    let mut named = base.clone();
    named.feature_id = Some("id".to_string());
    assert!(!base.same_feature(&named));
    // None feature_id orders before Some, mirroring Option ordering.
    assert_eq!(base.cmp_feature(&named), Ordering::Less);

    let other_row = FeatureRef::row_number(8);
    assert!(!base.same_feature(&other_row));
    assert_eq!(base.cmp_entry(&other_row), Ordering::Less);
}
