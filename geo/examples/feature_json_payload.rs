//! Store projected properties next to geometry as GeoJSON Feature payloads.

use bytes::Bytes;
use packed_spatial_index_geo::{
    Box2D, ConvertRequest, GeoArtifactIndex, GeoPayload, PayloadPlan, PropertyProjection,
    SliceReader, open, open_geo_index,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Bytes::from_static(include_bytes!(
        "../tests/fixtures/parquet-geospatial/crs-srid.parquet"
    ));
    let mut dataset = open(data)?;
    let artifact_bytes = dataset.convert(ConvertRequest {
        payload: PayloadPlan::FeatureJson {
            properties: PropertyProjection::AllNonGeometry,
        },
        ..ConvertRequest::default()
    })?;

    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(artifact_bytes))? else {
        panic!("sample fixture is 2D");
    };

    let query = Box2D::new(-1.0e9, -1.0e9, 1.0e9, 1.0e9);
    let hits = index.search_hits(query)?;
    let Some(hit) = hits.first() else {
        panic!("sample fixture should match the broad query");
    };
    let GeoPayload::FeatureJson(feature) = &hit.payload else {
        panic!("expected GeoJSON Feature payload");
    };

    let property_count = feature["properties"]
        .as_object()
        .map_or(0, serde_json::Map::len);
    println!(
        "row {}: {} with {property_count} projected properties",
        hit.feature.row_number, feature["geometry"]["type"]
    );
    assert_eq!(feature["feature_ref"]["row_number"], hit.feature.row_number);
    Ok(())
}
