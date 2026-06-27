//! List usable geometry columns before choosing what to index.

use bytes::Bytes;
use packed_spatial_index_geo::{SelectionStatus, open};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Bytes::from_static(include_bytes!(
        "../tests/fixtures/parquet-geospatial/crs-srid.parquet"
    ));
    let dataset = open(data)?;
    let discovery = dataset.discovery();

    println!("rows: {}", discovery.num_rows);
    match &discovery.default_selection {
        SelectionStatus::Selected { column, reason } => {
            println!("default geometry column: {column} ({reason:?})");
        }
        other => println!("no default geometry column: {other:?}"),
    }

    for column in &discovery.columns {
        println!(
            "{}: {:?}, {:?}, dims={:?}, row_wkb={}",
            column.name,
            column.source,
            column.encoding,
            column.coordinate_dims,
            column.capabilities.can_emit_row_wkb
        );
    }

    assert!(!discovery.columns.is_empty());
    Ok(())
}
