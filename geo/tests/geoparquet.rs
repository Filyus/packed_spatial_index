//! End-to-end tests over in-memory GeoParquet fixtures.
//!
//! Each fixture is a real Parquet file (built with `arrow` + `parquet`) carrying
//! a WKB geometry column, an optional `bbox` covering struct column, and the
//! injected `geo` key-value metadata. Correctness is checked against a
//! brute-force overlap oracle over the very boxes the reader produced — the same
//! self-consistent style as the core crate's property tests (it exercises the
//! reader + traversal, not the predicate math).

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Float64Array, StructArray};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use packed_spatial_index::{
    Box2D, Box3D, SliceReader, StreamIndex2D, StreamIndex2DF32, read_metadata,
};
use packed_spatial_index_geo::{
    BuildOpts, ConvertOpts, GeoError, build_index_2d, build_index_3d, convert_2d, detect_dims,
    read_bboxes_2d, read_bboxes_3d,
};

// --- WKB encoders (little-endian, ISO) --------------------------------------

fn wkb_point_2d(x: f64, y: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(21);
    v.push(1); // little-endian
    v.extend_from_slice(&1u32.to_le_bytes()); // Point
    v.extend_from_slice(&x.to_le_bytes());
    v.extend_from_slice(&y.to_le_bytes());
    v
}

fn wkb_point_3d(x: f64, y: f64, z: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(29);
    v.push(1);
    v.extend_from_slice(&1001u32.to_le_bytes()); // ISO Point Z
    v.extend_from_slice(&x.to_le_bytes());
    v.extend_from_slice(&y.to_le_bytes());
    v.extend_from_slice(&z.to_le_bytes());
    v
}

// --- Fixture builders -------------------------------------------------------

fn write_parquet(cols: Vec<(&str, ArrayRef)>, geo_json: String) -> Bytes {
    let batch = RecordBatch::try_from_iter(cols).expect("record batch");
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new("geo".to_string(), geo_json)]))
        .build();
    let mut buf = Vec::new();
    let mut w = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).expect("writer");
    w.write(&batch).expect("write");
    w.close().expect("close");
    Bytes::from(buf)
}

fn binary_col(wkbs: &[Option<Vec<u8>>]) -> ArrayRef {
    let values: Vec<Option<&[u8]>> = wkbs.iter().map(|o| o.as_deref()).collect();
    Arc::new(BinaryArray::from(values))
}

fn bbox_struct_2d(boxes: &[[f64; 4]]) -> ArrayRef {
    let col = |k: usize| Float64Array::from(boxes.iter().map(|b| b[k]).collect::<Vec<_>>());
    Arc::new(StructArray::from(vec![
        (
            Arc::new(Field::new("xmin", DataType::Float64, false)),
            Arc::new(col(0)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("ymin", DataType::Float64, false)),
            Arc::new(col(1)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("xmax", DataType::Float64, false)),
            Arc::new(col(2)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("ymax", DataType::Float64, false)),
            Arc::new(col(3)) as ArrayRef,
        ),
    ]))
}

fn bbox_struct_3d(boxes: &[[f64; 6]]) -> ArrayRef {
    let col = |k: usize| Float64Array::from(boxes.iter().map(|b| b[k]).collect::<Vec<_>>());
    Arc::new(StructArray::from(vec![
        (
            Arc::new(Field::new("xmin", DataType::Float64, false)),
            Arc::new(col(0)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("ymin", DataType::Float64, false)),
            Arc::new(col(1)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("zmin", DataType::Float64, false)),
            Arc::new(col(2)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("xmax", DataType::Float64, false)),
            Arc::new(col(3)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("ymax", DataType::Float64, false)),
            Arc::new(col(4)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("zmax", DataType::Float64, false)),
            Arc::new(col(5)) as ArrayRef,
        ),
    ]))
}

const CRS_JSON: &str = r#"{"id":{"authority":"EPSG","code":4326}}"#;

fn geo_meta_2d(covering: bool, crs: bool) -> String {
    let crs_field = if crs {
        format!(r#","crs":{CRS_JSON}"#)
    } else {
        String::new()
    };
    let cover = if covering {
        r#","covering":{"bbox":{"xmin":["bbox","xmin"],"ymin":["bbox","ymin"],"xmax":["bbox","xmax"],"ymax":["bbox","ymax"]}}"#
    } else {
        ""
    };
    format!(
        r#"{{"version":"1.1.0","primary_column":"geometry","columns":{{"geometry":{{"encoding":"WKB","geometry_types":["Point"]{crs_field}{cover}}}}}}}"#
    )
}

fn geo_meta_3d(covering: bool) -> String {
    let cover = if covering {
        r#","covering":{"bbox":{"xmin":["bbox","xmin"],"ymin":["bbox","ymin"],"zmin":["bbox","zmin"],"xmax":["bbox","xmax"],"ymax":["bbox","ymax"],"zmax":["bbox","zmax"]}}"#
    } else {
        ""
    };
    format!(
        r#"{{"version":"1.1.0","primary_column":"geometry","columns":{{"geometry":{{"encoding":"WKB","geometry_types":["Point Z"]{cover}}}}}}}"#
    )
}

// --- Oracles ----------------------------------------------------------------

fn brute_2d(boxes: &[Box2D], q: Box2D) -> Vec<usize> {
    let mut v: Vec<usize> = boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| b.overlaps(q))
        .map(|(i, _)| i)
        .collect();
    v.sort_unstable();
    v
}

fn brute_3d(boxes: &[Box3D], q: Box3D) -> Vec<usize> {
    let mut v: Vec<usize> = boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| b.overlaps(q))
        .map(|(i, _)| i)
        .collect();
    v.sort_unstable();
    v
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

// --- Tests ------------------------------------------------------------------

#[test]
fn read_and_build_2d_wkb_envelope_match_bruteforce() {
    let pts = [(0.0, 0.0), (10.0, 10.0), (5.0, 5.0), (3.0, 8.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts.iter().map(|&(x, y)| Some(wkb_point_2d(x, y))).collect();
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, false),
    );

    let boxes = read_bboxes_2d(data.clone()).unwrap();
    assert_eq!(boxes.len(), 4);

    let index = build_index_2d(data, BuildOpts::default()).unwrap();
    for q in [
        Box2D::new(-1.0, -1.0, 6.0, 6.0),
        Box2D::new(9.0, 9.0, 11.0, 11.0),
        Box2D::new(100.0, 100.0, 200.0, 200.0),
    ] {
        assert_eq!(sorted(index.search(q)), brute_2d(&boxes, q), "query {q:?}");
    }
}

#[test]
fn covering_column_is_used_over_geometry_envelope() {
    // Points sit at (5,5) but each covering box is a wide rectangle. A query that
    // misses the points yet hits the covering box proves the covering path is used.
    let pts = [(5.0, 5.0), (50.0, 50.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts.iter().map(|&(x, y)| Some(wkb_point_2d(x, y))).collect();
    let boxes_cov = [[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 60.0, 60.0]];
    let data = write_parquet(
        vec![
            ("geometry", binary_col(&wkbs)),
            ("bbox", bbox_struct_2d(&boxes_cov)),
        ],
        geo_meta_2d(true, false),
    );

    let boxes = read_bboxes_2d(data.clone()).unwrap();
    assert!(
        boxes[0].overlaps(Box2D::new(1.0, 1.0, 2.0, 2.0)),
        "covering box read"
    );

    let index = build_index_2d(data, BuildOpts::default()).unwrap();
    assert_eq!(index.search(Box2D::new(1.0, 1.0, 2.0, 2.0)), vec![0]);
}

#[test]
fn null_geometry_is_rejected() {
    let wkbs = vec![
        Some(wkb_point_2d(0.0, 0.0)),
        None,
        Some(wkb_point_2d(2.0, 2.0)),
    ];
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, false),
    );
    match read_bboxes_2d(data) {
        Err(GeoError::NullGeometry { row }) => assert_eq!(row, 1),
        other => panic!("expected NullGeometry, got {other:?}"),
    }
}

#[test]
fn convert_2d_roundtrips_payloads_and_metadata() {
    let pts = [(0.0, 0.0), (10.0, 10.0), (5.0, 5.0), (3.0, 8.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts.iter().map(|&(x, y)| Some(wkb_point_2d(x, y))).collect();
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, true),
    );

    let boxes = read_bboxes_2d(data.clone()).unwrap();
    let psindex = convert_2d(data, ConvertOpts::default()).unwrap();

    let meta = read_metadata(&psindex).unwrap();
    assert_eq!(meta.content_type.as_deref(), Some("application/geo+wkb"));
    assert_eq!(meta.crs.as_deref(), Some(CRS_JSON));

    let index = StreamIndex2D::open(SliceReader::new(psindex.as_slice())).unwrap();
    let q = Box2D::new(-1.0, -1.0, 6.0, 6.0);
    let hits = index.search_payloads(q).unwrap();
    assert_eq!(
        sorted(hits.iter().map(|(i, _)| *i).collect()),
        brute_2d(&boxes, q)
    );
    // Each returned payload is exactly the original row's WKB.
    for (id, payload) in &hits {
        assert_eq!(payload, wkbs[*id].as_ref().unwrap(), "payload for row {id}");
    }
}

#[test]
fn convert_2d_compact_f32_is_a_conservative_superset() {
    let pts = [(0.0, 0.0), (10.0, 10.0), (5.0, 5.0), (3.0, 8.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts.iter().map(|&(x, y)| Some(wkb_point_2d(x, y))).collect();
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, false),
    );

    let boxes = read_bboxes_2d(data.clone()).unwrap();
    let opts = ConvertOpts {
        compact_f32: true,
        ..Default::default()
    };
    let psindex = convert_2d(data, opts).unwrap();

    let index = StreamIndex2DF32::open(SliceReader::new(psindex.as_slice())).unwrap();
    let q = Box2D::new(-1.0, -1.0, 6.0, 6.0);
    let got = sorted(index.search(q).unwrap());
    for expected in brute_2d(&boxes, q) {
        assert!(
            got.contains(&expected),
            "f32 result must include exact hit {expected}"
        );
    }
}

#[test]
fn detect_dims_reports_2d_and_3d() {
    let p2 = [Some(wkb_point_2d(0.0, 0.0))];
    let d2 = write_parquet(
        vec![("geometry", binary_col(&p2))],
        geo_meta_2d(false, false),
    );
    assert_eq!(detect_dims(d2).unwrap(), 2);

    let p3 = [Some(wkb_point_3d(0.0, 0.0, 0.0))];
    let d3 = write_parquet(vec![("geometry", binary_col(&p3))], geo_meta_3d(false));
    assert_eq!(detect_dims(d3).unwrap(), 3);
}

#[test]
fn read_and_build_3d_wkb_envelope_match_bruteforce() {
    let pts = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (5.0, 5.0, 5.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts
        .iter()
        .map(|&(x, y, z)| Some(wkb_point_3d(x, y, z)))
        .collect();
    let data = write_parquet(vec![("geometry", binary_col(&wkbs))], geo_meta_3d(false));

    let boxes = read_bboxes_3d(data.clone()).unwrap();
    assert_eq!(boxes.len(), 3);

    let index = build_index_3d(data, BuildOpts::default()).unwrap();
    for q in [
        Box3D::new(-1.0, -1.0, -1.0, 6.0, 6.0, 6.0),
        Box3D::new(9.0, 9.0, 9.0, 11.0, 11.0, 11.0),
    ] {
        assert_eq!(sorted(index.search(q)), brute_3d(&boxes, q), "query {q:?}");
    }
}

#[test]
fn read_and_build_3d_covering_match_bruteforce() {
    let pts = [(5.0, 5.0, 5.0), (50.0, 50.0, 50.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts
        .iter()
        .map(|&(x, y, z)| Some(wkb_point_3d(x, y, z)))
        .collect();
    let cov = [
        [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        [40.0, 40.0, 40.0, 60.0, 60.0, 60.0],
    ];
    let data = write_parquet(
        vec![
            ("geometry", binary_col(&wkbs)),
            ("bbox", bbox_struct_3d(&cov)),
        ],
        geo_meta_3d(true),
    );

    let boxes = read_bboxes_3d(data.clone()).unwrap();
    let index = build_index_3d(data, BuildOpts::default()).unwrap();
    let q = Box3D::new(1.0, 1.0, 1.0, 2.0, 2.0, 2.0);
    assert_eq!(sorted(index.search(q)), brute_3d(&boxes, q));
    assert_eq!(index.search(q), vec![0]);
}
