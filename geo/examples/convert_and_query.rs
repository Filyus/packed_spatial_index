//! Convert Parquet geometry to PSINDEX, then query the converted artifact.

use bytes::Bytes;
use packed_spatial_index_geo::{
    Box2D, ConvertRequest, GeoArtifactIndex, GeoPayload, SliceReader, open, open_geo_index,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Bytes::from_static(include_bytes!(
        "../tests/fixtures/parquet-geospatial/crs-geography.parquet"
    ));
    let mut dataset = open(data)?;
    let artifact_bytes = dataset.convert(ConvertRequest::default())?;

    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(artifact_bytes))? else {
        panic!("sample fixture is 2D");
    };

    let query = Box2D::new(-1.0e9, -1.0e9, 1.0e9, 1.0e9);
    let hits = index.search_hits(query)?;
    println!("matched {} artifact entries", hits.len());

    if let Some(hit) = hits.first() {
        println!("first source row: {}", hit.feature.row_number);
        if let GeoPayload::RowWkb(wkb) = &hit.payload {
            println!("first WKB payload: {} bytes", wkb.len());
        }
    }

    assert!(!hits.is_empty());
    Ok(())
}
