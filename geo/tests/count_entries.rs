#![cfg(feature = "geojson")]

use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, ConvertRequest, EnvelopePolicy, GeoArtifactIndex, GeoQuery2D,
    PayloadPlan, SliceReader, open_geo_index, open_geojson_slice,
};

fn sample_geojson() -> &'static [u8] {
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

fn artifact(payload: PayloadPlan) -> GeoArtifactIndex<SliceReader<Vec<u8>>> {
    let mut source = open_geojson_slice(sample_geojson()).unwrap();
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

#[test]
fn count_matches_search_entry_ids() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::RowRef) else {
        panic!("expected 2D artifact");
    };

    // Plain box, empty box, and a polygon query all agree with id search.
    for query in [
        GeoQuery2D::Box2D(Box2D::new(-180.0, -10.0, 180.0, 10.0)),
        GeoQuery2D::Box2D(Box2D::new(50.0, 50.0, 60.0, 60.0)),
        GeoQuery2D::Box2D(Box2D::new(-10.0, 0.0, 30.0, 5.0)),
    ] {
        assert_eq!(
            index.count_entries(query.clone()).unwrap(),
            index.search_entry_ids(query).unwrap().len()
        );
    }

    use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};
    let triangle = Polygon::new(
        LineString::new(vec![
            Coord { x: -10.0, y: -1.0 },
            Coord { x: 0.0, y: 5.0 },
            Coord { x: 10.0, y: -1.0 },
            Coord { x: -10.0, y: -1.0 },
        ]),
        vec![],
    );
    let query = GeoQuery2D::polygon(triangle);
    assert_eq!(
        index.count_entries(query.clone()).unwrap(),
        index.search_entry_ids(query).unwrap().len()
    );
}

#[test]
fn count_works_on_payload_less_artifacts() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::None) else {
        panic!("expected 2D artifact");
    };
    let world = Box2D::new(-180.0, -10.0, 180.0, 10.0);
    assert_eq!(
        index.count_entries(world).unwrap(),
        index.search_entry_ids(world).unwrap().len()
    );
    // The split line contributes one count per entry, not per feature.
    assert_eq!(index.count_entries(world).unwrap(), 4);
}
