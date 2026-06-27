//! End-to-end tests over in-memory GeoParquet fixtures.
//!
//! Each fixture is a real Parquet file (built with `arrow` + `parquet`) carrying
//! a WKB geometry column, an optional `bbox` covering struct column, and the
//! injected `geo` key-value metadata. Correctness is checked against a
//! brute-force overlap oracle over the very boxes the reader produced — the same
//! self-consistent style as the core crate's property tests (it exercises the
//! reader + traversal, not the predicate math).

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Float32Array, Float64Array, StructArray};
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
    BuildOpts, ConvertOpts, ConvertPayload, GeoError, GeometryColumnSelection,
    GeometryMetadataSource, GeometrySelectionReason, ROW_ID_CONTENT_TYPE, ROW_WKB_CONTENT_TYPE,
    build_index_2d, build_index_3d, convert_2d, convert_2d_into, decode_row_id_payload,
    decode_row_wkb_payload, detect_dims, discover, inspect, read_bboxes_2d, read_bboxes_3d,
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

/// A closed-ring rectangle polygon, so the WKB envelope must track min/max over
/// several coordinates (not a single point).
fn wkb_polygon_2d(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Vec<u8> {
    let mut v = Vec::new();
    v.push(1);
    v.extend_from_slice(&3u32.to_le_bytes()); // Polygon
    v.extend_from_slice(&1u32.to_le_bytes()); // 1 ring
    v.extend_from_slice(&5u32.to_le_bytes()); // 5 points (closed)
    for (x, y) in [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ] {
        v.extend_from_slice(&x.to_le_bytes());
        v.extend_from_slice(&y.to_le_bytes());
    }
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

/// Same 2D covering struct but with `Float32` children, to exercise the f32 read
/// path.
fn bbox_struct_2d_f32(boxes: &[[f64; 4]]) -> ArrayRef {
    let col = |k: usize| Float32Array::from(boxes.iter().map(|b| b[k] as f32).collect::<Vec<_>>());
    Arc::new(StructArray::from(vec![
        (
            Arc::new(Field::new("xmin", DataType::Float32, false)),
            Arc::new(col(0)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("ymin", DataType::Float32, false)),
            Arc::new(col(1)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("xmax", DataType::Float32, false)),
            Arc::new(col(2)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("ymax", DataType::Float32, false)),
            Arc::new(col(3)) as ArrayRef,
        ),
    ]))
}

/// A geoarrow-native point column: `struct { x: f64, y: f64 }`. Not WKB, so the
/// reader must lean on the covering column for boxes.
fn native_point_struct(pts: &[(f64, f64)]) -> ArrayRef {
    let xs = Float64Array::from(pts.iter().map(|p| p.0).collect::<Vec<_>>());
    let ys = Float64Array::from(pts.iter().map(|p| p.1).collect::<Vec<_>>());
    Arc::new(StructArray::from(vec![
        (
            Arc::new(Field::new("x", DataType::Float64, false)),
            Arc::new(xs) as ArrayRef,
        ),
        (
            Arc::new(Field::new("y", DataType::Float64, false)),
            Arc::new(ys) as ArrayRef,
        ),
    ]))
}

/// 2D `geo` metadata with the native `point` encoding and a bbox covering.
fn geo_meta_2d_native_with_covering() -> String {
    r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"point","geometry_types":["Point"],"covering":{"bbox":{"xmin":["bbox","xmin"],"ymin":["bbox","ymin"],"xmax":["bbox","xmax"],"ymax":["bbox","ymax"]}}}}}"#
        .to_string()
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
    assert_eq!(meta.content_type.as_deref(), Some(ROW_WKB_CONTENT_TYPE));
    assert_eq!(meta.crs.as_deref(), Some(CRS_JSON));

    let index = StreamIndex2D::open(SliceReader::new(psindex.as_slice())).unwrap();
    let q = Box2D::new(-1.0, -1.0, 6.0, 6.0);
    let hits = index.search_payloads(q).unwrap();
    assert_eq!(
        sorted(hits.iter().map(|(i, _)| *i).collect()),
        brute_2d(&boxes, q)
    );
    // Each returned payload preserves the source row id before the WKB bytes.
    for (id, payload) in &hits {
        let (row, wkb) = decode_row_wkb_payload(payload).unwrap();
        assert_eq!(row as usize, *id);
        assert_eq!(wkb, wkbs[row as usize].as_ref().unwrap());
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

#[test]
fn polygon_wkb_envelope_tracks_all_coordinates() {
    // Envelope of each rectangle must be the rectangle itself, not one corner.
    let rects = [[0.0, 0.0, 4.0, 2.0], [10.0, 10.0, 12.0, 18.0]];
    let wkbs: Vec<Option<Vec<u8>>> = rects
        .iter()
        .map(|r| Some(wkb_polygon_2d(r[0], r[1], r[2], r[3])))
        .collect();
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, false),
    );

    let boxes = read_bboxes_2d(data.clone()).unwrap();
    // A query touching only the far corner of the first rectangle must still hit.
    assert!(boxes[0].overlaps(Box2D::new(3.5, 1.5, 3.9, 1.9)));
    let index = build_index_2d(data, BuildOpts::default()).unwrap();
    let q = Box2D::new(3.5, 1.5, 11.0, 11.0);
    assert_eq!(sorted(index.search(q)), brute_2d(&boxes, q));
}

#[test]
fn f32_covering_column_is_read() {
    let pts = [(5.0, 5.0), (50.0, 50.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts.iter().map(|&(x, y)| Some(wkb_point_2d(x, y))).collect();
    let cov = [[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 60.0, 60.0]];
    let data = write_parquet(
        vec![
            ("geometry", binary_col(&wkbs)),
            ("bbox", bbox_struct_2d_f32(&cov)),
        ],
        geo_meta_2d(true, false),
    );

    let boxes = read_bboxes_2d(data.clone()).unwrap();
    assert!(
        boxes[0].overlaps(Box2D::new(1.0, 1.0, 2.0, 2.0)),
        "f32 covering box read"
    );
    let index = build_index_2d(data, BuildOpts::default()).unwrap();
    assert_eq!(index.search(Box2D::new(1.0, 1.0, 2.0, 2.0)), vec![0]);
}

#[test]
fn native_encoding_uses_covering_and_rejects_payload() {
    let pts = [(5.0, 5.0), (50.0, 50.0)];
    let cov = [[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 60.0, 60.0]];
    let make = || {
        write_parquet(
            vec![
                ("geometry", native_point_struct(&pts)),
                ("bbox", bbox_struct_2d(&cov)),
            ],
            geo_meta_2d_native_with_covering(),
        )
    };

    // Accelerator works on native encoding: boxes come from the covering column,
    // geometry is never decoded.
    let index = build_index_2d(make(), BuildOpts::default()).unwrap();
    assert_eq!(index.search(Box2D::new(1.0, 1.0, 2.0, 2.0)), vec![0]);

    // The converter needs the WKB payload, which native encoding cannot provide.
    match convert_2d(make(), ConvertOpts::default()) {
        Err(GeoError::UnsupportedEncoding(_)) => {}
        other => panic!("expected UnsupportedEncoding, got {other:?}"),
    }
    // ...unless the payload is turned off, in which case it succeeds.
    let opts = ConvertOpts {
        include_payload: false,
        ..Default::default()
    };
    assert!(convert_2d(make(), opts).is_ok());

    // Row-id payloads are also compatible with native encoding because they do
    // not require decoding the geometry column.
    let opts = ConvertOpts {
        payload: ConvertPayload::RowIds,
        ..Default::default()
    };
    let psindex = convert_2d(make(), opts).unwrap();
    let meta = read_metadata(&psindex).unwrap();
    assert_eq!(meta.content_type.as_deref(), Some(ROW_ID_CONTENT_TYPE));
    let index = StreamIndex2D::open(SliceReader::new(psindex.as_slice())).unwrap();
    let hits = index
        .search_payloads(Box2D::new(1.0, 1.0, 2.0, 2.0))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(decode_row_id_payload(&hits[0].1), Some(0));
}

#[test]
fn multi_batch_row_indices_are_correct() {
    // > one default record batch (1024 rows) so the cross-batch `row_base`
    // bookkeeping and per-batch column reads are exercised. Points walk a
    // diagonal; row i sits at (i, i).
    const N: usize = 2500;
    let wkbs: Vec<Option<Vec<u8>>> = (0..N)
        .map(|i| Some(wkb_point_2d(i as f64, i as f64)))
        .collect();
    let cov: Vec<[f64; 4]> = (0..N)
        .map(|i| [i as f64, i as f64, i as f64, i as f64])
        .collect();

    // WKB-envelope path across batches.
    let wkb_data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, false),
    );
    let index = build_index_2d(wkb_data.clone(), BuildOpts::default()).unwrap();
    assert_eq!(read_bboxes_2d(wkb_data).unwrap().len(), N);
    // A late, narrow query must return exactly that row's index.
    assert_eq!(
        index.search(Box2D::new(2000.0, 2000.0, 2000.0, 2000.0)),
        vec![2000]
    );
    let mut hits = index.search(Box2D::new(1500.0, 1500.0, 1502.0, 1502.0));
    hits.sort_unstable();
    assert_eq!(hits, vec![1500, 1501, 1502]);

    // Covering path across batches.
    let cov_data = write_parquet(
        vec![
            ("geometry", binary_col(&wkbs)),
            ("bbox", bbox_struct_2d(&cov)),
        ],
        geo_meta_2d(true, false),
    );
    let index = build_index_2d(cov_data, BuildOpts::default()).unwrap();
    assert_eq!(
        index.search(Box2D::new(2300.0, 2300.0, 2300.0, 2300.0)),
        vec![2300]
    );
}

#[test]
fn inspect_reports_metadata_without_reading_rows() {
    let pts = [(0.0, 0.0), (10.0, 10.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts.iter().map(|&(x, y)| Some(wkb_point_2d(x, y))).collect();
    let cov = [[0.0, 0.0, 1.0, 1.0], [9.0, 9.0, 11.0, 11.0]];
    let data = write_parquet(
        vec![
            ("geometry", binary_col(&wkbs)),
            ("bbox", bbox_struct_2d(&cov)),
        ],
        geo_meta_2d(true, true),
    );

    let info = inspect(data).unwrap();
    assert_eq!(info.version, "1.1.0");
    assert_eq!(info.geometry_column, "geometry");
    assert_eq!(info.dims, 2);
    assert_eq!(info.encoding, "WKB");
    assert!(info.has_covering);
    assert_eq!(info.crs.as_deref(), Some(CRS_JSON));
    assert_eq!(info.bounds, None);
    assert_eq!(info.num_rows, 2);

    // A file that records its overall extent surfaces it via `bounds`.
    let with_bounds = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"],"bbox":[0.0,0.0,10.0,10.0]}}}"#
            .to_string(),
    );
    assert_eq!(
        inspect(with_bounds).unwrap().bounds,
        Some(vec![0.0, 0.0, 10.0, 10.0])
    );
}

#[test]
fn discover_selects_geoparquet_primary_column() {
    let wkbs = [Some(wkb_point_2d(0.0, 0.0)), Some(wkb_point_2d(10.0, 10.0))];
    let cov = [[0.0, 0.0, 1.0, 1.0], [9.0, 9.0, 11.0, 11.0]];
    let data = write_parquet(
        vec![
            ("geometry", binary_col(&wkbs)),
            ("bbox", bbox_struct_2d(&cov)),
        ],
        geo_meta_2d(true, true),
    );

    let discovery = discover(data).unwrap();
    assert_eq!(discovery.num_rows, 2);
    assert_eq!(discovery.geo_metadata_version.as_deref(), Some("1.1.0"));
    assert_eq!(discovery.geo_primary_column.as_deref(), Some("geometry"));
    assert_eq!(
        discovery.selection,
        GeometryColumnSelection::Selected {
            column: "geometry".to_string(),
            reason: GeometrySelectionReason::GeoParquetPrimary,
        }
    );
    assert_eq!(discovery.columns.len(), 1);

    let column = &discovery.columns[0];
    assert_eq!(column.name, "geometry");
    assert_eq!(column.metadata_source, GeometryMetadataSource::GeoParquet);
    assert_eq!(column.encoding, "WKB");
    assert_eq!(column.dims, Some(2));
    assert!(column.has_covering);
    assert_eq!(column.crs.as_deref(), Some(CRS_JSON));
    assert!(column.can_build_index);
    assert!(column.can_emit_row_wkb);
}

#[test]
fn discover_lists_all_geoparquet_columns() {
    let wkbs = [Some(wkb_point_2d(0.0, 0.0))];
    let data = write_parquet(
        vec![
            ("geom_a", binary_col(&wkbs)),
            ("geom_b", binary_col(&wkbs)),
        ],
        r#"{"version":"1.1.0","primary_column":"geom_a","columns":{"geom_a":{"encoding":"WKB","geometry_types":["Point"]},"geom_b":{"encoding":"WKB","geometry_types":["Point"]}}}"#
            .to_string(),
    );

    let discovery = discover(data).unwrap();
    let names: Vec<_> = discovery
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    assert_eq!(names, vec!["geom_a", "geom_b"]);
    assert_eq!(
        discovery.selection,
        GeometryColumnSelection::Selected {
            column: "geom_a".to_string(),
            reason: GeometrySelectionReason::GeoParquetPrimary,
        }
    );
}

#[test]
fn malformed_geo_metadata_is_reported() {
    let wkbs = [Some(wkb_point_2d(0.0, 0.0))];
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        r#"{"version":"1.1.0","primary_column":"geometry","columns":"bad"}"#.to_string(),
    );

    assert!(matches!(discover(data), Err(GeoError::Metadata(_))));
}

#[test]
fn discover_reports_native_geoarrow_covering_capabilities() {
    let pts = [(5.0, 5.0), (50.0, 50.0)];
    let cov = [[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 60.0, 60.0]];
    let data = write_parquet(
        vec![
            ("geometry", native_point_struct(&pts)),
            ("bbox", bbox_struct_2d(&cov)),
        ],
        geo_meta_2d_native_with_covering(),
    );

    let discovery = discover(data).unwrap();
    assert_eq!(discovery.columns.len(), 1);
    let column = &discovery.columns[0];
    assert_eq!(column.name, "geometry");
    assert_eq!(column.metadata_source, GeometryMetadataSource::GeoParquet);
    assert_eq!(column.encoding, "point");
    assert_eq!(column.dims, Some(2));
    assert!(column.has_covering);
    assert!(column.can_build_index);
    assert!(!column.can_emit_row_wkb);
}

#[test]
fn convert_into_matches_convert() {
    let pts = [(0.0, 0.0), (10.0, 10.0), (5.0, 5.0)];
    let wkbs: Vec<Option<Vec<u8>>> = pts.iter().map(|&(x, y)| Some(wkb_point_2d(x, y))).collect();
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, true),
    );

    let owned = convert_2d(data.clone(), ConvertOpts::default()).unwrap();
    let mut buf = Vec::new();
    convert_2d_into(data, ConvertOpts::default(), &mut buf).unwrap();
    assert_eq!(owned, buf);
}

#[test]
fn convert_skip_null_drops_null_rows() {
    // Row 1 is null; with skip_null the converter keeps the two real geometries.
    let wkbs = vec![
        Some(wkb_point_2d(0.0, 0.0)),
        None,
        Some(wkb_point_2d(10.0, 10.0)),
    ];
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, false),
    );

    // Default policy still errors.
    assert!(matches!(
        convert_2d(data.clone(), ConvertOpts::default()),
        Err(GeoError::NullGeometry { row: 1 })
    ));

    let opts = ConvertOpts {
        skip_null: true,
        ..Default::default()
    };
    let psindex = convert_2d(data, opts).unwrap();
    let index = StreamIndex2D::open(SliceReader::new(psindex.as_slice())).unwrap();
    // Two surviving rows; ids are compacted, payload row ids point to the source.
    let hits = index
        .search_payloads(Box2D::new(-1.0, -1.0, 11.0, 11.0))
        .unwrap();
    assert_eq!(hits.len(), 2);
    let mut decoded: Vec<(usize, u64, Vec<u8>)> = hits
        .iter()
        .map(|(id, payload)| {
            let (row, wkb) = decode_row_wkb_payload(payload).unwrap();
            (*id, row, wkb.to_vec())
        })
        .collect();
    decoded.sort_by_key(|(_, row, _)| *row);
    assert_eq!(decoded[0].0, 0);
    assert_eq!(decoded[0].1, 0);
    assert_eq!(decoded[0].2, *wkbs[0].as_ref().unwrap());
    assert_eq!(decoded[1].0, 1);
    assert_eq!(decoded[1].1, 2);
    assert_eq!(decoded[1].2, *wkbs[2].as_ref().unwrap());
}

#[test]
fn convert_row_id_payload_is_compact_sidecar_mode() {
    let wkbs = vec![
        Some(wkb_point_2d(0.0, 0.0)),
        None,
        Some(wkb_point_2d(10.0, 10.0)),
    ];
    let data = write_parquet(
        vec![("geometry", binary_col(&wkbs))],
        geo_meta_2d(false, false),
    );
    let opts = ConvertOpts {
        payload: ConvertPayload::RowIds,
        skip_null: true,
        ..Default::default()
    };

    let psindex = convert_2d(data, opts).unwrap();

    let meta = read_metadata(&psindex).unwrap();
    assert_eq!(meta.content_type.as_deref(), Some(ROW_ID_CONTENT_TYPE));
    let index = StreamIndex2D::open(SliceReader::new(psindex.as_slice())).unwrap();
    let hits = index
        .search_payloads(Box2D::new(-1.0, -1.0, 11.0, 11.0))
        .unwrap();
    let mut rows: Vec<u64> = hits
        .iter()
        .map(|(_, payload)| decode_row_id_payload(payload).unwrap())
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![0, 2]);
}
