#![cfg(all(feature = "parquet", feature = "geojson", feature = "flatgeobuf"))]

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use flatgeobuf::{ColumnType, FgbCrs, FgbWriter, FgbWriterOptions, GeometryType};
use geozero::geojson::GeoJson;
use geozero::{ColumnValue, PropertyProcessor};
use packed_spatial_index_geo::{
    Box2D, BuildRequest, GeoIndex, GeoSource, GeometryScan, ScanRequest, open_flatgeobuf,
    open_geojson_slice, open_geoparquet,
};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

fn wkb_point_2d(x: f64, y: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(21);
    v.push(1);
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&x.to_le_bytes());
    v.extend_from_slice(&y.to_le_bytes());
    v
}

fn sample_geojson() -> &'static [u8] {
    br#"{"type":"FeatureCollection","features":[
        {"type":"Feature","geometry":{"type":"Point","coordinates":[-5.0,1.0]},"properties":{"name":"west"}},
        {"type":"Feature","geometry":{"type":"Point","coordinates":[25.0,3.0]},"properties":{"name":"east"}}
    ]}"#
}

fn sample_fgb() -> Vec<u8> {
    let mut fgb = FgbWriter::create_with_options(
        "points",
        GeometryType::Point,
        FgbWriterOptions {
            write_index: false,
            crs: FgbCrs {
                code: 4326,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    fgb.add_column("name", ColumnType::String, |_, _| {});
    fgb.add_feature_geom(
        GeoJson(r#"{"type":"Point","coordinates":[-5.0,1.0]}"#),
        |feat| {
            feat.property(0, "name", &ColumnValue::String("west"))
                .unwrap();
        },
    )
    .unwrap();
    fgb.add_feature_geom(
        GeoJson(r#"{"type":"Point","coordinates":[25.0,3.0]}"#),
        |feat| {
            feat.property(0, "name", &ColumnValue::String("east"))
                .unwrap();
        },
    )
    .unwrap();

    let mut bytes = Vec::new();
    fgb.write(&mut bytes).unwrap();
    bytes
}

fn sample_parquet() -> Bytes {
    let wkbs = [wkb_point_2d(-5.0, 1.0), wkb_point_2d(25.0, 3.0)];
    let wkb_refs: Vec<&[u8]> = wkbs.iter().map(Vec::as_slice).collect();
    let geometry: ArrayRef = Arc::new(BinaryArray::from(wkb_refs));
    let names: ArrayRef = Arc::new(StringArray::from(vec!["west", "east"]));
    let batch = RecordBatch::try_from_iter(vec![("geometry", geometry), ("name", names)]).unwrap();
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "geo".to_string(),
            r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"]}}}"#
                .to_string(),
        )]))
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}

fn scan_signature(scan: GeometryScan) -> Vec<(u64, [f64; 4])> {
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    scan.features
        .iter()
        .zip(scan.boxes.iter())
        .map(|(feature, bbox)| {
            (
                feature.row_number,
                [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y],
            )
        })
        .collect()
}

fn query_west(index: GeoIndex) -> Vec<u64> {
    let GeoIndex::D2(index) = index else {
        panic!("expected 2D index");
    };
    index
        .search_features(Box2D::new(-10.0, 0.0, 0.0, 2.0))
        .unwrap()
        .into_iter()
        .map(|feature| feature.row_number)
        .collect()
}

#[test]
fn parquet_geojson_and_flatgeobuf_scan_and_query_match() {
    let mut parquet = open_geoparquet(sample_parquet()).unwrap();
    let parquet_scan = scan_signature(parquet.scan(ScanRequest::default()).unwrap());
    let parquet_query = query_west(
        open_geoparquet(sample_parquet())
            .unwrap()
            .build(BuildRequest::default())
            .unwrap(),
    );

    let mut geojson = open_geojson_slice(sample_geojson()).unwrap();
    let geojson_scan = scan_signature(geojson.scan(ScanRequest::default()).unwrap());
    let geojson_query = query_west(
        open_geojson_slice(sample_geojson())
            .unwrap()
            .build(BuildRequest::default())
            .unwrap(),
    );

    let mut fgb = open_flatgeobuf(std::io::Cursor::new(sample_fgb())).unwrap();
    let fgb_scan = scan_signature(fgb.scan(ScanRequest::default()).unwrap());
    let fgb_query = query_west(
        open_flatgeobuf(std::io::Cursor::new(sample_fgb()))
            .unwrap()
            .build(BuildRequest::default())
            .unwrap(),
    );

    assert_eq!(geojson_scan, parquet_scan);
    assert_eq!(fgb_scan, parquet_scan);
    assert_eq!(geojson_query, vec![0]);
    assert_eq!(fgb_query, vec![0]);
    assert_eq!(parquet_query, vec![0]);
}

/// Build and query through the format-agnostic `GeoSource` trait: one generic
/// function drives all three sources, proving the trait unifies the build path.
fn query_west_via_trait<S: GeoSource>(mut source: S) -> Vec<u64> {
    query_west(source.build(BuildRequest::default()).unwrap())
}

#[test]
fn geo_source_trait_drives_every_format() {
    let parquet = query_west_via_trait(open_geoparquet(sample_parquet()).unwrap());
    let geojson = query_west_via_trait(open_geojson_slice(sample_geojson()).unwrap());
    let fgb = query_west_via_trait(open_flatgeobuf(std::io::Cursor::new(sample_fgb())).unwrap());

    assert_eq!(parquet, vec![0]);
    assert_eq!(geojson, parquet);
    assert_eq!(fgb, parquet);
}
