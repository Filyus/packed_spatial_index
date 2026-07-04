//! Build an in-memory spatial index from a geospatial Parquet file.

use bytes::Bytes;
use packed_spatial_index_geo::{Box2D, BuildRequest, GeoIndex, open_geoparquet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Bytes::from_static(include_bytes!(
        "../tests/fixtures/parquet-geospatial/crs-geography.parquet"
    ));
    let mut dataset = open_geoparquet(data)?;
    let index = dataset.build(BuildRequest::default())?;

    let GeoIndex::D2(index) = index else {
        panic!("sample fixture is 2D");
    };

    let query = Box2D::new(-1.0e9, -1.0e9, 1.0e9, 1.0e9);
    let features = index.search_features(query)?;
    println!("matched {} source features", features.len());

    assert!(!features.is_empty());
    Ok(())
}
