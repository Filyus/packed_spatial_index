use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;
use std::sync::Arc;
use std::{env, fs};

use arrow::array::{ArrayRef, BinaryArray, Float64Array, ListArray, StringArray, StructArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use base64::Engine as _;
use bytes::Bytes;
use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, Box3D, ConvertRequest, CoordinateDims, DuplicateFeatureRows,
    EnvelopePolicy, FEATURE_REF_RECORD_LEN, FeatureFilterRequest, FeatureReadOrder,
    FeatureReadRequest, FeatureRef, GeoArtifactIndex, GeoError, GeoIndex, GeoPayload, GeoQuery2D,
    GeometryEncoding, GeometryMetadataSource, GeometryReadMode, GeometryScan, GeometrySelector,
    IndexDimsRequest, InspectRequest, NonPlanarExactPolicy, NullPolicy, PayloadPlan,
    PropertyProjection, RangeReader, SliceReader, StoragePrecision, StreamIndex2D,
    decode_feature_ref_payload, decode_feature_wkb_payload, open, open_geo_index,
    read_geo_manifest,
};
use parquet::arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions};
use parquet::basic::{GeometryType, LogicalType, Repetition, Type as ParquetPhysicalType};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};

fn geometry_fixture() -> Bytes {
    Bytes::from_static(include_bytes!(
        "fixtures/parquet-geospatial/geospatial.parquet"
    ))
}

fn srid_fixture() -> Bytes {
    Bytes::from_static(include_bytes!(
        "fixtures/parquet-geospatial/crs-srid.parquet"
    ))
}

fn wkb_point_2d(x: f64, y: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(21);
    v.push(1);
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&x.to_le_bytes());
    v.extend_from_slice(&y.to_le_bytes());
    v
}

fn wkb_point_3d(x: f64, y: f64, z: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(29);
    v.push(1);
    v.extend_from_slice(&1001u32.to_le_bytes());
    v.extend_from_slice(&x.to_le_bytes());
    v.extend_from_slice(&y.to_le_bytes());
    v.extend_from_slice(&z.to_le_bytes());
    v
}

fn wkb_line_2d(coords: &[(f64, f64)]) -> Vec<u8> {
    let mut v = Vec::new();
    v.push(1);
    v.extend_from_slice(&2u32.to_le_bytes());
    v.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for (x, y) in coords {
        v.extend_from_slice(&x.to_le_bytes());
        v.extend_from_slice(&y.to_le_bytes());
    }
    v
}

fn wkb_multipoint_2d(coords: &[(f64, f64)]) -> Vec<u8> {
    let mut v = Vec::new();
    v.push(1);
    v.extend_from_slice(&4u32.to_le_bytes());
    v.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for (x, y) in coords {
        v.extend_from_slice(&wkb_point_2d(*x, *y));
    }
    v
}

fn binary_col(values: &[Option<Vec<u8>>]) -> ArrayRef {
    let values: Vec<Option<&[u8]>> = values.iter().map(|value| value.as_deref()).collect();
    Arc::new(BinaryArray::from(values))
}

fn write_geoparquet(cols: Vec<(&str, ArrayRef)>, geo_json: String) -> Bytes {
    let batch = RecordBatch::try_from_iter(cols).unwrap();
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new("geo".to_string(), geo_json)]))
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}

fn write_geoparquet_with_row_group_size(
    cols: Vec<(&str, ArrayRef)>,
    geo_json: String,
    row_group_rows: usize,
) -> Bytes {
    let batch = RecordBatch::try_from_iter(cols).unwrap();
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new("geo".to_string(), geo_json)]))
        .set_max_row_group_row_count(Some(row_group_rows))
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}

fn geo_meta_wkb(geometry_types: &[&str]) -> String {
    let types = geometry_types
        .iter()
        .map(|ty| format!(r#""{ty}""#))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"version":"1.1.0","primary_column":"geometry","columns":{{"geometry":{{"encoding":"WKB","geometry_types":[{types}]}}}}}}"#
    )
}

fn geo_meta_wkb_edges(geometry_types: &[&str], edges: &str) -> String {
    let types = geometry_types
        .iter()
        .map(|ty| format!(r#""{ty}""#))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"version":"1.1.0","primary_column":"geometry","columns":{{"geometry":{{"encoding":"WKB","geometry_types":[{types}],"edges":"{edges}"}}}}}}"#
    )
}

fn geo_meta_arrow(encoding: &str, geometry_type: &str) -> String {
    format!(
        r#"{{"version":"1.1.0","primary_column":"geometry","columns":{{"geometry":{{"encoding":"{encoding}","geometry_types":["{geometry_type}"]}}}}}}"#
    )
}

fn geoarrow_points(points: &[(f64, f64)]) -> ArrayRef {
    let xs = Float64Array::from(points.iter().map(|p| p.0).collect::<Vec<_>>());
    let ys = Float64Array::from(points.iter().map(|p| p.1).collect::<Vec<_>>());
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

fn list(values: ArrayRef, lengths: &[usize]) -> ArrayRef {
    let field = Arc::new(Field::new("item", values.data_type().clone(), false));
    Arc::new(ListArray::new(
        field,
        OffsetBuffer::<i32>::from_lengths(lengths.iter().copied()),
        values,
        None,
    ))
}

fn native_geometry_schema(names: &[&str]) -> SchemaDescriptor {
    let fields = names
        .iter()
        .map(|name| {
            Arc::new(
                ParquetType::primitive_type_builder(name, ParquetPhysicalType::BYTE_ARRAY)
                    .with_repetition(Repetition::REQUIRED)
                    .with_logical_type(Some(LogicalType::Geometry(GeometryType { crs: None })))
                    .build()
                    .unwrap(),
            )
        })
        .collect();
    let root = ParquetType::group_type_builder("schema")
        .with_fields(fields)
        .build()
        .unwrap();
    SchemaDescriptor::new(Arc::new(root))
}

fn native_parquet(names: &[&str], values: Vec<Vec<Vec<u8>>>) -> Bytes {
    let cols: Vec<_> = names
        .iter()
        .zip(values)
        .map(|(name, wkbs)| {
            let refs: Vec<&[u8]> = wkbs.iter().map(Vec::as_slice).collect();
            (*name, Arc::new(BinaryArray::from(refs)) as ArrayRef)
        })
        .collect();
    let batch = RecordBatch::try_from_iter(cols).unwrap();
    let options = ArrowWriterOptions::new().with_parquet_schema(native_geometry_schema(names));
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new_with_options(&mut buf, batch.schema(), options).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}

fn assert_no_panic<T>(label: &str, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| panic!("{label} panicked"))
}

#[test]
fn geoparquet_primary_discovery_inspect_scan_and_build() {
    let data = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[Some(wkb_point_2d(0.0, 0.0)), Some(wkb_point_2d(10.0, 10.0))]),
        )],
        geo_meta_wkb(&["Point"]),
    );

    let dataset = open(data.clone()).unwrap();
    assert_eq!(dataset.discovery().num_rows, 2);
    assert_eq!(
        dataset.discovery().default_selection,
        packed_spatial_index_geo::SelectionStatus::Selected {
            column: "geometry".to_string(),
            reason: packed_spatial_index_geo::GeometrySelectionReason::GeoParquetPrimary,
        }
    );

    let mut dataset = open(data.clone()).unwrap();
    let profile = dataset.inspect(InspectRequest::default()).unwrap();
    assert_eq!(profile.source, GeometryMetadataSource::GeoParquet);
    assert_eq!(profile.coordinate_dims, CoordinateDims::Xy);

    let mut dataset = open(data.clone()).unwrap();
    let scan = dataset.scan(Default::default()).unwrap();
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    assert_eq!(scan.boxes.len(), 2);
    assert_eq!(scan.features[1].row_number, 1);

    let mut dataset = open(data).unwrap();
    let index = dataset.build(Default::default()).unwrap();
    let GeoIndex::D2(index) = index else {
        panic!("expected 2D index");
    };
    let hits = index
        .search_features(Box2D::new(-1.0, -1.0, 1.0, 1.0))
        .unwrap();
    assert_eq!(hits[0].row_number, 0);
}

#[test]
fn feature_refs_include_row_group_positions() {
    let data = write_geoparquet_with_row_group_size(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_point_2d(0.0, 0.0)),
                    Some(wkb_point_2d(1.0, 1.0)),
                    Some(wkb_point_2d(2.0, 2.0)),
                    Some(wkb_point_2d(3.0, 3.0)),
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
        2,
    );
    let mut dataset = open(data).unwrap();
    let GeometryScan::D2(scan) = dataset.scan(Default::default()).unwrap() else {
        panic!("expected 2D scan");
    };
    assert_eq!(scan.features[0].row_group, Some(0));
    assert_eq!(scan.features[0].row_in_group, Some(0));
    assert_eq!(scan.features[2].row_group, Some(1));
    assert_eq!(scan.features[2].row_in_group, Some(0));
}

#[test]
fn read_features_returns_projected_rows_and_wkb() {
    let data = write_geoparquet_with_row_group_size(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_point_2d(0.0, 0.0)),
                    Some(wkb_point_2d(10.0, 10.0)),
                    Some(wkb_point_2d(20.0, 20.0)),
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
        2,
    );

    let mut indexed = open(data.clone()).unwrap();
    let GeoIndex::D2(index) = indexed.build(Default::default()).unwrap() else {
        panic!("expected 2D index");
    };
    let features = index
        .search_features(Box2D::new(5.0, 5.0, 25.0, 25.0))
        .unwrap();

    let mut source = open(data).unwrap();
    let rows = source
        .read_features(FeatureReadRequest {
            features,
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            geometry: GeometryReadMode::Wkb,
            ..FeatureReadRequest::default()
        })
        .unwrap();

    assert_eq!(
        rows.features
            .iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    let names = rows
        .batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "b");
    assert_eq!(names.value(1), "c");
    let wkbs = rows
        .batch
        .column_by_name("geometry_wkb")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    assert_eq!(wkbs.value(0), wkb_point_2d(10.0, 10.0));
    assert_eq!(wkbs.value(1), wkb_point_2d(20.0, 20.0));
}

#[test]
fn read_features_empty_request_keeps_requested_schema() {
    let data = write_geoparquet(
        vec![
            ("geometry", binary_col(&[Some(wkb_point_2d(0.0, 0.0))])),
            ("name", Arc::new(StringArray::from(vec!["a"])) as ArrayRef),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let mut source = open(data).unwrap();
    let rows = source
        .read_features(FeatureReadRequest {
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            geometry: GeometryReadMode::Wkb,
            ..FeatureReadRequest::default()
        })
        .unwrap();
    assert_eq!(rows.features.len(), 0);
    assert_eq!(rows.batch.num_rows(), 0);
    assert!(rows.batch.column_by_name("name").is_some());
    assert!(rows.batch.column_by_name("geometry_wkb").is_some());
}

#[test]
fn read_features_can_preserve_request_order_and_duplicate_parts() {
    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_point_2d(0.0, 0.0)),
                    Some(wkb_point_2d(10.0, 10.0)),
                    Some(wkb_point_2d(20.0, 20.0)),
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let mut source = open(data).unwrap();
    let rows = source
        .read_features(FeatureReadRequest {
            features: vec![
                FeatureRef {
                    row_number: 2,
                    row_group: None,
                    row_in_group: None,
                    part: Some(0),
                    feature_id: None,
                },
                FeatureRef {
                    row_number: 1,
                    row_group: None,
                    row_in_group: None,
                    part: None,
                    feature_id: None,
                },
                FeatureRef {
                    row_number: 2,
                    row_group: None,
                    row_in_group: None,
                    part: Some(1),
                    feature_id: None,
                },
            ],
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            order: FeatureReadOrder::RequestOrder,
            duplicates: DuplicateFeatureRows::KeepParts,
            ..FeatureReadRequest::default()
        })
        .unwrap();
    assert_eq!(
        rows.features
            .iter()
            .map(|feature| (feature.row_number, feature.part))
            .collect::<Vec<_>>(),
        vec![(2, Some(0)), (1, None), (2, Some(1))]
    );
    let names = rows
        .batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "c");
    assert_eq!(names.value(1), "b");
    assert_eq!(names.value(2), "c");
}

#[test]
fn read_features_reports_fingerprint_and_bounds_errors() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(0.0, 0.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut source = open(data.clone()).unwrap();
    let mismatch = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef {
                row_number: 0,
                row_group: None,
                row_in_group: None,
                part: None,
                feature_id: None,
            }],
            expected_source_fingerprint: Some("fnv64:0000000000000000".to_string()),
            ..FeatureReadRequest::default()
        })
        .unwrap_err();
    assert!(matches!(
        mismatch,
        GeoError::SourceFingerprintMismatch { .. }
    ));

    let mut source = open(data).unwrap();
    let out_of_bounds = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef {
                row_number: 9,
                row_group: None,
                row_in_group: None,
                part: None,
                feature_id: None,
            }],
            ..FeatureReadRequest::default()
        })
        .unwrap_err();
    assert!(matches!(
        out_of_bounds,
        GeoError::FeatureRowOutOfBounds { row_number: 9, .. }
    ));
}

#[test]
fn filter_features_removes_bbox_false_positive_and_keeps_points() {
    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_line_2d(&[(0.0, 0.0), (10.0, 10.0)])),
                    Some(wkb_point_2d(0.5, 9.5)),
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["line", "point"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["LineString", "Point"]),
    );
    let query = Box2D::new(0.0, 9.0, 1.0, 10.0);
    let mut indexed = open(data.clone()).unwrap();
    let GeoIndex::D2(index) = indexed.build(Default::default()).unwrap() else {
        panic!("expected 2D index");
    };
    let candidates = index.search_features(query).unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );

    let mut source = open(data.clone()).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(candidates, query))
        .unwrap();
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].row_number, 1);

    let mut source = open(data).unwrap();
    let rows = source
        .read_features(FeatureReadRequest {
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            ..FeatureReadRequest::from_features(exact)
        })
        .unwrap();
    let names = rows
        .batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "point");
}

#[test]
fn filter_features_supports_polygon_query() {
    use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};

    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_point_2d(1.0, 1.0)),   // 0: inside the triangle
                    Some(wkb_point_2d(8.0, 8.0)),   // 1: in bbox, outside the triangle
                    Some(wkb_point_2d(2.0, 2.0)),   // 2: inside the triangle
                    Some(wkb_point_2d(20.0, 20.0)), // 3: outside the bbox
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );

    // Right triangle (0,0)-(10,0)-(0,10): bbox is [0,10]^2, but x+y <= 10 inside.
    let triangle = Polygon::new(
        LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    );

    // A polygon query narrows index candidates by its bounding box: rows 0,1,2.
    let mut indexed = open(data.clone()).unwrap();
    let GeoIndex::D2(index) = indexed.build(Default::default()).unwrap() else {
        panic!("expected 2D index");
    };
    let candidates = index
        .search_features(GeoQuery2D::polygon(triangle.clone()))
        .unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );

    // Exact filtering drops the bbox false-positive (row 1 is outside the triangle).
    let mut source = open(data).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            candidates,
            GeoQuery2D::polygon(triangle),
        ))
        .unwrap();
    assert_eq!(
        exact
            .iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
}

#[test]
fn filter_hits_filters_artifact_search_by_polygon() {
    use packed_spatial_index_geo::SpatialPredicate;
    use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};

    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_point_2d(1.0, 1.0)),   // 0: inside the triangle
                    Some(wkb_point_2d(8.0, 8.0)),   // 1: in bbox, outside the triangle
                    Some(wkb_point_2d(2.0, 2.0)),   // 2: inside the triangle
                    Some(wkb_point_2d(20.0, 20.0)), // 3: outside the bbox
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );

    // Convert to a PSINDEX artifact carrying WKB geometry payloads, then open it.
    let mut dataset = open(data).unwrap();
    let artifact = dataset.convert(ConvertRequest::default()).unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(artifact)).unwrap() else {
        panic!("expected 2D artifact");
    };

    let triangle = Polygon::new(
        LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    );

    // Streaming search narrows by bbox (rows 0,1,2); order is leaf-based, compare as sets.
    let hits = index
        .search_hits(GeoQuery2D::polygon(triangle.clone()))
        .unwrap();
    let mut candidate_rows = hits
        .iter()
        .map(|hit| hit.feature.row_number)
        .collect::<Vec<_>>();
    candidate_rows.sort_unstable();
    assert_eq!(candidate_rows, vec![0, 1, 2]);

    // filter_hits removes the bbox false-positive (row 1) using the payload geometry.
    let exact = index
        .filter_hits(
            hits,
            GeoQuery2D::polygon(triangle),
            SpatialPredicate::Intersects,
            NonPlanarExactPolicy::Reject,
        )
        .unwrap();
    let mut exact_rows = exact
        .iter()
        .map(|hit| hit.feature.row_number)
        .collect::<Vec<_>>();
    exact_rows.sort_unstable();
    assert_eq!(exact_rows, vec![0, 2]);
}

#[test]
fn filter_hits_supports_feature_json_payload() {
    use packed_spatial_index_geo::SpatialPredicate;
    use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};

    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_point_2d(1.0, 1.0)), // 0: inside the triangle
                    Some(wkb_point_2d(8.0, 8.0)), // 1: in bbox, outside the triangle
                    Some(wkb_point_2d(2.0, 2.0)), // 2: inside the triangle
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );

    // Artifact with GeoJSON Feature payloads (geometry embedded as GeoJSON, not WKB).
    let mut dataset = open(data).unwrap();
    let artifact = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::None,
            },
            ..ConvertRequest::default()
        })
        .unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(artifact)).unwrap() else {
        panic!("expected 2D artifact");
    };

    let triangle = Polygon::new(
        LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    );

    // filter_hits decodes geometry from the GeoJSON payload, no source re-read.
    let hits = index
        .search_hits(GeoQuery2D::polygon(triangle.clone()))
        .unwrap();
    let exact = index
        .filter_hits(
            hits,
            GeoQuery2D::polygon(triangle),
            SpatialPredicate::Intersects,
            NonPlanarExactPolicy::Reject,
        )
        .unwrap();
    let mut rows = exact
        .iter()
        .map(|hit| hit.feature.row_number)
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, vec![0, 2]);
}

#[test]
fn filter_features_supports_native_parquet_and_geoarrow_sources() {
    let query = Box2D::new(4.0, 4.0, 6.0, 6.0);

    let native = native_parquet(
        &["geometry"],
        vec![vec![wkb_point_2d(5.0, 5.0), wkb_point_2d(10.0, 10.0)]],
    );
    let mut source = open(native).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0), FeatureRef::row_number(1)],
            query,
        ))
        .unwrap();
    assert_eq!(
        exact.iter().map(|f| f.row_number).collect::<Vec<_>>(),
        vec![0]
    );

    let geoarrow = write_geoparquet(
        vec![("geometry", geoarrow_points(&[(5.0, 5.0), (10.0, 10.0)]))],
        geo_meta_arrow("point", "Point"),
    );
    let mut source = open(geoarrow).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0), FeatureRef::row_number(1)],
            query,
        ))
        .unwrap();
    assert_eq!(
        exact.iter().map(|f| f.row_number).collect::<Vec<_>>(),
        vec![0]
    );
}

#[test]
fn filter_features_handles_duplicates_malformed_wkb_and_fingerprint() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(5.0, 5.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut source = open(data.clone()).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![
                FeatureRef {
                    row_number: 0,
                    row_group: None,
                    row_in_group: None,
                    part: Some(0),
                    feature_id: None,
                },
                FeatureRef {
                    row_number: 0,
                    row_group: None,
                    row_in_group: None,
                    part: Some(1),
                    feature_id: None,
                },
            ],
            Box2D::new(4.0, 4.0, 6.0, 6.0),
        ))
        .unwrap();
    assert_eq!(
        exact
            .iter()
            .map(|feature| (feature.row_number, feature.part))
            .collect::<Vec<_>>(),
        vec![(0, Some(0)), (0, Some(1))]
    );

    let malformed = write_geoparquet(
        vec![("geometry", binary_col(&[Some(vec![1, 2, 3])]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut source = open(malformed).unwrap();
    assert!(matches!(
        source.filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            Box2D::new(0.0, 0.0, 1.0, 1.0),
        )),
        Err(GeoError::Wkb(_))
    ));

    let empty = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[Some(wkb_point_2d(f64::NAN, f64::NAN))]),
        )],
        geo_meta_wkb(&["Point"]),
    );
    let mut source = open(empty).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            Box2D::new(0.0, 0.0, 10.0, 10.0),
        ))
        .unwrap();
    assert!(exact.is_empty());

    let mut source = open(data).unwrap();
    let mismatch = source
        .filter_features(FeatureFilterRequest {
            expected_source_fingerprint: Some("fnv64:0000000000000000".to_string()),
            ..FeatureFilterRequest::intersects(
                vec![FeatureRef::row_number(0)],
                Box2D::new(4.0, 4.0, 6.0, 6.0),
            )
        })
        .unwrap_err();
    assert!(matches!(
        mismatch,
        GeoError::SourceFingerprintMismatch { .. }
    ));
}

#[test]
fn filter_features_spherical_radius_matches_points_and_multipoints() {
    let data = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[
                Some(wkb_point_2d(2.3522, 48.8566)),
                Some(wkb_point_2d(13.4050, 52.5200)),
            ]),
        )],
        geo_meta_wkb_edges(&["Point"], "spherical"),
    );
    let mut source = open(data).unwrap();
    let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(2.35, 48.85, 2_000.0);
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0), FeatureRef::row_number(1)],
            query,
        ))
        .unwrap();
    assert_eq!(
        exact
            .iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![0]
    );

    let data = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[
                Some(wkb_multipoint_2d(&[(13.4050, 52.5200), (2.3522, 48.8566)])),
                Some(wkb_multipoint_2d(&[(13.4050, 52.5200)])),
            ]),
        )],
        geo_meta_wkb_edges(&["MultiPoint"], "spherical"),
    );
    let mut source = open(data).unwrap();
    let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(2.35, 48.85, 2_000.0);
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0), FeatureRef::row_number(1)],
            query,
        ))
        .unwrap();
    assert_eq!(
        exact
            .iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![0]
    );
}

#[test]
fn filter_features_spherical_radius_rejects_wrong_edges_and_unsupported_geometry() {
    let planar = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(2.0, 49.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut source = open(planar).unwrap();
    let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(2.0, 49.0, 1_000.0);
    assert!(matches!(
        source.filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            query,
        )),
        Err(GeoError::NonSphericalExactPredicate { .. })
    ));

    let unknown_edges = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(2.0, 49.0))]))],
        geo_meta_wkb_edges(&["Point"], "karney"),
    );
    let mut source = open(unknown_edges).unwrap();
    let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(2.0, 49.0, 1_000.0);
    assert!(matches!(
        source.filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            query,
        )),
        Err(GeoError::NonSphericalExactPredicate { .. })
    ));

    let line = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[Some(wkb_line_2d(&[(2.0, 49.0), (3.0, 49.0)]))]),
        )],
        geo_meta_wkb_edges(&["LineString"], "spherical"),
    );
    let mut source = open(line).unwrap();
    let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(2.0, 49.0, 1_000.0);
    assert!(matches!(
        source.filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            query,
        )),
        Err(GeoError::UnsupportedGeodeticGeometry(kind)) if kind == "LineString"
    ));
}

#[test]
fn filter_features_spherical_radius_handles_empty_malformed_and_candidate_boxes() {
    let empty = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[Some(wkb_point_2d(f64::NAN, f64::NAN))]),
        )],
        geo_meta_wkb_edges(&["Point"], "spherical"),
    );
    let mut source = open(empty).unwrap();
    let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(2.0, 49.0, 1_000.0);
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            query,
        ))
        .unwrap();
    assert!(exact.is_empty());

    let malformed = write_geoparquet(
        vec![("geometry", binary_col(&[Some(vec![1, 2, 3])]))],
        geo_meta_wkb_edges(&["Point"], "spherical"),
    );
    let mut source = open(malformed).unwrap();
    let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(2.0, 49.0, 1_000.0);
    assert!(matches!(
        source.filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            query,
        )),
        Err(GeoError::Wkb(_))
    ));

    let antimeridian =
        packed_spatial_index_geo::GeoQuery2D::spherical_radius(179.5, 0.0, 200_000.0)
            .candidate_boxes_2d()
            .unwrap();
    assert_eq!(antimeridian.len(), 2);

    let pole = packed_spatial_index_geo::GeoQuery2D::spherical_radius(0.0, 89.0, 300_000.0)
        .candidate_boxes_2d()
        .unwrap();
    assert_eq!(pole.len(), 1);
    assert_eq!(pole[0].min_x, -180.0);
    assert_eq!(pole[0].max_x, 180.0);
}

#[test]
fn filter_features_rejects_non_planar_edges_unless_opted_in() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(5.0, 5.0))]))],
        geo_meta_wkb_edges(&["Point"], "spherical"),
    );
    let mut source = open(data.clone()).unwrap();
    let err = source
        .filter_features(FeatureFilterRequest::intersects(
            vec![FeatureRef::row_number(0)],
            Box2D::new(4.0, 4.0, 6.0, 6.0),
        ))
        .unwrap_err();
    assert!(matches!(err, GeoError::NonPlanarExactPredicate { .. }));

    let mut source = open(data).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest {
            non_planar: NonPlanarExactPolicy::TreatAsPlanar,
            ..FeatureFilterRequest::intersects(
                vec![FeatureRef::row_number(0)],
                Box2D::new(4.0, 4.0, 6.0, 6.0),
            )
        })
        .unwrap();
    assert_eq!(exact[0].row_number, 0);
}

#[test]
fn row_ref_artifact_hits_feed_read_features() {
    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[Some(wkb_point_2d(1.0, 1.0)), Some(wkb_point_2d(9.0, 9.0))]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["near", "far"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data.clone()).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };
    let manifest = index.manifest().clone();
    let hits = index.search_hits(Box2D::new(9.0, 9.0, 9.0, 9.0)).unwrap();

    let mut source = open(data).unwrap();
    let rows = source
        .read_features(FeatureReadRequest {
            selector: GeometrySelector::Name(manifest.selected_column),
            expected_source_fingerprint: Some(manifest.source_fingerprint),
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            ..FeatureReadRequest::from_hits(hits)
        })
        .unwrap();
    let names = rows
        .batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(rows.features[0].row_number, 1);
    assert_eq!(names.value(0), "far");
}

#[test]
fn row_ref_artifact_hits_feed_exact_filter_then_read_features() {
    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_line_2d(&[(0.0, 0.0), (10.0, 10.0)])),
                    Some(wkb_point_2d(0.5, 9.5)),
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["line", "point"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["LineString", "Point"]),
    );
    let mut dataset = open(data.clone()).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };
    let query = Box2D::new(0.0, 9.0, 1.0, 10.0);
    let manifest = index.manifest().clone();
    let hits = index.search_hits(query).unwrap();
    assert_eq!(hits.len(), 2);

    let mut source = open(data.clone()).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest {
            selector: GeometrySelector::Name(manifest.selected_column.clone()),
            expected_source_fingerprint: Some(manifest.source_fingerprint.clone()),
            ..FeatureFilterRequest::intersects_from_hits(hits, query)
        })
        .unwrap();
    assert_eq!(
        exact.iter().map(|f| f.row_number).collect::<Vec<_>>(),
        vec![1]
    );

    let mut source = open(data).unwrap();
    let rows = source
        .read_features(FeatureReadRequest {
            selector: GeometrySelector::Name(manifest.selected_column),
            expected_source_fingerprint: Some(manifest.source_fingerprint),
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            ..FeatureReadRequest::from_features(exact)
        })
        .unwrap();
    let names = rows
        .batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "point");
}

#[test]
fn read_features_emits_wkb_for_geoarrow_geometry() {
    let data = write_geoparquet(
        vec![
            ("geometry", geoarrow_points(&[(2.0, 3.0)])),
            (
                "name",
                Arc::new(StringArray::from(vec!["geoarrow"])) as ArrayRef,
            ),
        ],
        geo_meta_arrow("point", "Point"),
    );
    let mut source = open(data).unwrap();
    let rows = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef {
                row_number: 0,
                row_group: None,
                row_in_group: None,
                part: None,
                feature_id: None,
            }],
            geometry: GeometryReadMode::Wkb,
            ..FeatureReadRequest::default()
        })
        .unwrap();
    let wkbs = rows
        .batch
        .column_by_name("geometry_wkb")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    assert_eq!(wkbs.value(0), wkb_point_2d(2.0, 3.0));
}

#[test]
fn native_parquet_single_and_ambiguous_selection() {
    let dataset = open(srid_fixture()).unwrap();
    assert!(matches!(
        dataset.discovery().columns[0].encoding,
        GeometryEncoding::ParquetGeometry
    ));
    assert_eq!(
        dataset.discovery().columns[0].source,
        GeometryMetadataSource::ParquetGeospatial
    );

    let data = native_parquet(
        &["geom_a", "geom_b"],
        vec![vec![wkb_point_2d(0.0, 0.0)], vec![wkb_point_2d(10.0, 10.0)]],
    );
    let dataset = open(data.clone()).unwrap();
    assert!(matches!(
        dataset.discovery().default_selection,
        packed_spatial_index_geo::SelectionStatus::Ambiguous { .. }
    ));
    let mut dataset = open(data.clone()).unwrap();
    assert!(matches!(
        dataset.scan(Default::default()),
        Err(GeoError::AmbiguousGeometryColumn { .. })
    ));

    let mut dataset = open(data).unwrap();
    let scan = dataset
        .scan(packed_spatial_index_geo::ScanRequest {
            selector: GeometrySelector::Name("geom_b".to_string()),
            ..Default::default()
        })
        .unwrap();
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    assert_eq!(scan.boxes[0], Box2D::new(10.0, 10.0, 10.0, 10.0));
}

#[test]
fn explicit_missing_column_is_clear_error() {
    let dataset = open(srid_fixture()).unwrap();
    assert!(matches!(
        dataset.select(GeometrySelector::Name("missing".to_string())),
        Err(GeoError::GeometryColumnNotFound(name)) if name == "missing"
    ));
}

#[test]
fn geoarrow_point_and_nested_encodings_scan_without_covering() {
    let cases = [
        ("point", "Point", geoarrow_points(&[(1.0, 2.0)])),
        (
            "linestring",
            "LineString",
            list(geoarrow_points(&[(0.0, 0.0), (2.0, 3.0)]), &[2]),
        ),
        (
            "polygon",
            "Polygon",
            list(
                list(
                    geoarrow_points(&[(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 0.0)]),
                    &[4],
                ),
                &[1],
            ),
        ),
        (
            "multipolygon",
            "MultiPolygon",
            list(
                list(
                    list(
                        geoarrow_points(&[(5.0, 6.0), (7.0, 6.0), (7.0, 8.0), (5.0, 6.0)]),
                        &[4],
                    ),
                    &[1],
                ),
                &[1],
            ),
        ),
    ];

    for (encoding, geometry_type, array) in cases {
        let data = write_geoparquet(
            vec![("geometry", array)],
            geo_meta_arrow(encoding, geometry_type),
        );
        let mut dataset = open(data).unwrap();
        let scan = dataset.scan(Default::default()).unwrap();
        let GeometryScan::D2(scan) = scan else {
            panic!("expected 2D scan for {encoding}");
        };
        assert_eq!(scan.boxes.len(), 1, "{encoding}");
    }
}

#[test]
fn geoarrow_row_wkb_payload_and_manifest_roundtrip() {
    let data = write_geoparquet(
        vec![("geometry", geoarrow_points(&[(1.0, 2.0)]))],
        geo_meta_arrow("point", "Point"),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset.convert(ConvertRequest::default()).unwrap();
    let manifest = read_geo_manifest(&bytes).unwrap().unwrap();
    assert_eq!(manifest.selected_column, "geometry");
    assert_eq!(manifest.index_entry_count, 1);

    let stream = StreamIndex2D::open(SliceReader::new(bytes)).unwrap();
    let hits = stream
        .search_payloads(Box2D::new(1.0, 2.0, 1.0, 2.0))
        .unwrap();
    let (feature, wkb) = decode_feature_wkb_payload(&hits[0].1).unwrap();
    assert_eq!(feature.row_number, 0);
    assert!(!wkb.is_empty());
}

#[test]
fn geo_artifact_reader_searches_default_row_wkb_payloads() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(1.0, 2.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset.convert(ConvertRequest::default()).unwrap();

    let artifact = open_geo_index(SliceReader::new(bytes)).unwrap();
    assert_eq!(artifact.manifest().selected_column, "geometry");
    let GeoArtifactIndex::D2(index) = artifact else {
        panic!("expected 2D artifact");
    };
    let hits = index.search_hits(Box2D::new(1.0, 2.0, 1.0, 2.0)).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].feature.row_number, 0);
    let GeoPayload::RowWkb(wkb) = &hits[0].payload else {
        panic!("expected RowWkb payload");
    };
    assert_eq!(wkb, &wkb_point_2d(1.0, 2.0));
    assert_eq!(
        index
            .search_features(Box2D::new(1.0, 2.0, 1.0, 2.0))
            .unwrap()[0]
            .row_number,
        0
    );
}

#[test]
fn geo_artifact_reader_searches_row_ref_payloads() {
    let data = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[None, Some(wkb_point_2d(5.0, 5.0))]),
        )],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            nulls: NullPolicy::Skip,
            ..ConvertRequest::default()
        })
        .unwrap();

    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };
    let hits = index.search_hits(Box2D::new(5.0, 5.0, 5.0, 5.0)).unwrap();
    assert_eq!(hits[0].feature.row_number, 1);
    assert_eq!(hits[0].payload, GeoPayload::RowRef);
}

#[test]
fn feature_json_includes_projected_properties() {
    let data = write_geoparquet(
        vec![
            ("geometry", binary_col(&[Some(wkb_point_2d(3.0, 4.0))])),
            (
                "name",
                Arc::new(StringArray::from(vec!["alpha"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::AllNonGeometry,
            },
            ..ConvertRequest::default()
        })
        .unwrap();
    let stream = StreamIndex2D::open(SliceReader::new(bytes)).unwrap();
    let hits = stream
        .search_payloads(Box2D::new(3.0, 4.0, 3.0, 4.0))
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&hits[0].1).unwrap();
    assert_eq!(json["type"], "Feature");
    assert_eq!(json["feature_ref"]["row_number"], 0);
    assert_eq!(json["properties"]["name"], "alpha");
    assert_eq!(json["geometry"]["type"], "Point");
}

#[test]
fn geo_artifact_reader_searches_feature_json_payloads() {
    let data = write_geoparquet(
        vec![
            ("geometry", binary_col(&[Some(wkb_point_2d(3.0, 4.0))])),
            (
                "name",
                Arc::new(StringArray::from(vec!["alpha"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::AllNonGeometry,
            },
            ..ConvertRequest::default()
        })
        .unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };
    let hits = index.search_hits(Box2D::new(3.0, 4.0, 3.0, 4.0)).unwrap();
    assert_eq!(hits[0].feature.row_number, 0);
    let GeoPayload::FeatureJson(json) = &hits[0].payload else {
        panic!("expected FeatureJson payload");
    };
    assert_eq!(json["properties"]["name"], "alpha");
}

#[test]
fn geo_artifact_reader_uses_manifest_precision_for_f32_artifacts() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(9.0, 10.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            precision: StoragePrecision::F32,
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };
    assert_eq!(index.manifest().storage_precision, StoragePrecision::F32);
    let hits = index.search_hits(Box2D::new(9.0, 10.0, 9.0, 10.0)).unwrap();
    assert_eq!(hits[0].feature.row_number, 0);
}

#[test]
fn geo_artifact_reader_searches_3d_artifacts() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_3d(1.0, 2.0, 3.0))]))],
        geo_meta_wkb(&["Point Z"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            dims: IndexDimsRequest::D3,
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();

    let GeoArtifactIndex::D3(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 3D artifact");
    };
    let hits = index
        .search_hits(Box3D::new(1.0, 2.0, 3.0, 1.0, 2.0, 3.0))
        .unwrap();
    assert_eq!(hits[0].feature.row_number, 0);
    assert_eq!(hits[0].payload, GeoPayload::RowRef);
}

#[test]
fn geo_artifact_reader_does_not_require_known_length() {
    struct NoLenReader(SliceReader<Vec<u8>>);

    impl RangeReader for NoLenReader {
        fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.0.read_exact_at(offset, buf)
        }
    }

    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(6.0, 7.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset.convert(ConvertRequest::default()).unwrap();

    let GeoArtifactIndex::D2(index) = open_geo_index(NoLenReader(SliceReader::new(bytes))).unwrap()
    else {
        panic!("expected 2D artifact");
    };
    let hits = index
        .search_features(Box2D::new(6.0, 7.0, 6.0, 7.0))
        .unwrap();
    assert_eq!(hits[0].row_number, 0);
}

#[test]
fn geo_artifact_reader_requires_geo_manifest() {
    let mut builder = packed_spatial_index::Index2DBuilder::new(1);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    let bytes = builder.finish().unwrap().to_bytes();
    assert!(matches!(
        open_geo_index(SliceReader::new(bytes)),
        Err(GeoError::MissingGeoManifest)
    ));
}

#[test]
fn antimeridian_split_duplicates_feature_ref_parts() {
    let data = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[Some(wkb_line_2d(&[(170.0, 0.0), (-170.0, 1.0)]))]),
        )],
        geo_meta_wkb(&["LineString"]),
    );
    let mut dataset = open(data.clone()).unwrap();
    let scan = dataset
        .scan(packed_spatial_index_geo::ScanRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            ..Default::default()
        })
        .unwrap();
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    assert_eq!(scan.boxes.len(), 2);
    assert_eq!(scan.features[0].row_number, scan.features[1].row_number);
    assert_eq!(scan.features[0].part, Some(0));
    assert_eq!(scan.features[1].part, Some(1));
    for bbox in &scan.boxes {
        assert!(bbox.min_x >= -180.0);
        assert!(bbox.max_x <= 180.0);
        assert!(bbox.min_x <= bbox.max_x);
    }

    let mut source = open(data).unwrap();
    let exact = source
        .filter_features(FeatureFilterRequest::intersects(
            scan.features,
            Box2D::new(-180.0, -1.0, 180.0, 2.0),
        ))
        .unwrap();
    assert_eq!(
        exact.iter().map(|feature| feature.part).collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
}

#[test]
fn null_skip_preserves_source_row_number_and_row_ref_payload() {
    let data = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[None, Some(wkb_point_2d(5.0, 5.0))]),
        )],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data.clone()).unwrap();
    assert!(matches!(
        dataset.scan(Default::default()),
        Err(GeoError::NullGeometry { row: 0 })
    ));

    let mut dataset = open(data).unwrap();
    let bytes = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            nulls: NullPolicy::Skip,
            ..ConvertRequest::default()
        })
        .unwrap();
    let stream = StreamIndex2D::open(SliceReader::new(bytes)).unwrap();
    let hits = stream
        .search_payloads(Box2D::new(5.0, 5.0, 5.0, 5.0))
        .unwrap();
    assert_eq!(hits[0].1.len(), FEATURE_REF_RECORD_LEN);
    let feature = decode_feature_ref_payload(&hits[0].1).unwrap();
    assert_eq!(feature.row_number, 1);
}

#[test]
fn native_3d_fixture_scans_as_3d() {
    let data = native_parquet(&["geometry"], vec![vec![wkb_point_3d(1.0, 2.0, 3.0)]]);
    let mut dataset = open(data).unwrap();
    let scan = dataset
        .scan(packed_spatial_index_geo::ScanRequest {
            dims: IndexDimsRequest::D3,
            ..Default::default()
        })
        .unwrap();
    let GeometryScan::D3(scan) = scan else {
        panic!("expected 3D scan");
    };
    assert_eq!(
        scan.boxes[0],
        packed_spatial_index_geo::Box3D::new(1.0, 2.0, 3.0, 1.0, 2.0, 3.0)
    );
}

#[test]
fn apache_native_fixture_currently_opens() {
    let mut dataset = open(geometry_fixture()).unwrap();
    let profile = dataset
        .inspect(InspectRequest {
            exact: true,
            ..InspectRequest::default()
        })
        .unwrap();
    assert_eq!(profile.source, GeometryMetadataSource::ParquetGeospatial);
}

#[test]
fn cli_discover_json_smoke() {
    let output = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("discover")
        .arg("tests/fixtures/parquet-geospatial/crs-srid.parquet")
        .arg("--json")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["columns"][0]["source"], "parquet_geospatial");
}

#[test]
fn cli_query_json_smoke() {
    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[Some(wkb_point_2d(1.0, 1.0)), Some(wkb_point_2d(8.0, 8.0))]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["near", "far"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data.clone()).unwrap();
    let psindex = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();

    let dir = env::temp_dir().join(format!(
        "psi_geo_query_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&dir).unwrap();
    let source_path = dir.join("source.parquet");
    let index_path = dir.join("source.psi");
    fs::write(&source_path, &data).unwrap();
    fs::write(&index_path, &psindex).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--bbox")
        .arg("8,8,8,8")
        .arg("--properties")
        .arg("include:name")
        .arg("--geometry")
        .arg("wkb")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 1);
    assert_eq!(json[0]["feature"]["row_number"], 1);
    assert_eq!(json[0]["properties"]["name"], "far");
    assert_eq!(
        json[0]["geometry_wkb"],
        base64::engine::general_purpose::STANDARD.encode(wkb_point_2d(8.0, 8.0))
    );
}

#[test]
fn cli_query_exact_filters_candidates_and_handles_non_planar_policy() {
    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_line_2d(&[(0.0, 0.0), (10.0, 10.0)])),
                    Some(wkb_point_2d(0.5, 9.5)),
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["line", "point"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["LineString", "Point"]),
    );
    let mut dataset = open(data.clone()).unwrap();
    let psindex = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();
    let dir = env::temp_dir().join(format!(
        "psi_geo_query_exact_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&dir).unwrap();
    let source_path = dir.join("source.parquet");
    let index_path = dir.join("source.psi");
    fs::write(&source_path, &data).unwrap();
    fs::write(&index_path, &psindex).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--bbox")
        .arg("0,9,1,10")
        .arg("--properties")
        .arg("include:name")
        .arg("--exact")
        .arg("--predicate")
        .arg("intersects")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 1);
    assert_eq!(json[0]["feature"]["row_number"], 1);
    assert_eq!(json[0]["properties"]["name"], "point");

    let ndjson = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--bbox")
        .arg("0,9,1,10")
        .arg("--exact")
        .arg("--ndjson")
        .output()
        .unwrap();
    assert!(
        ndjson.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&ndjson.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&ndjson.stdout).lines().count(), 1);

    let bad_predicate = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--bbox")
        .arg("0,9,1,10")
        .arg("--exact")
        .arg("--predicate")
        .arg("contains")
        .output()
        .unwrap();
    assert!(!bad_predicate.status.success());

    let non_planar = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(5.0, 5.0))]))],
        geo_meta_wkb_edges(&["Point"], "spherical"),
    );
    let mut dataset = open(non_planar.clone()).unwrap();
    let non_planar_index = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();
    let non_planar_source = dir.join("non_planar.parquet");
    let non_planar_psi = dir.join("non_planar.psi");
    fs::write(&non_planar_source, &non_planar).unwrap();
    fs::write(&non_planar_psi, &non_planar_index).unwrap();

    let rejected = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&non_planar_source)
        .arg(&non_planar_psi)
        .arg("--bbox")
        .arg("4,4,6,6")
        .arg("--exact")
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("non-planar"),
        "stderr: {}",
        String::from_utf8_lossy(&rejected.stderr)
    );

    let opted_in = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&non_planar_source)
        .arg(&non_planar_psi)
        .arg("--bbox")
        .arg("4,4,6,6")
        .arg("--exact")
        .arg("--treat-nonplanar-as-planar")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        opted_in.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&opted_in.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&opted_in.stdout).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 1);
}

#[test]
fn cli_query_spherical_radius_filters_geography_points() {
    let data = write_geoparquet(
        vec![
            (
                "geometry",
                binary_col(&[
                    Some(wkb_point_2d(179.8, 0.0)),
                    Some(wkb_point_2d(-179.8, 0.0)),
                    Some(wkb_point_2d(170.0, 0.0)),
                ]),
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["west", "east", "far"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb_edges(&["Point"], "spherical"),
    );
    let mut dataset = open(data.clone()).unwrap();
    let psindex = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            ..ConvertRequest::default()
        })
        .unwrap();

    let dir = env::temp_dir().join(format!(
        "psi_geo_query_radius_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&dir).unwrap();
    let source_path = dir.join("source.parquet");
    let index_path = dir.join("source.psi");
    fs::write(&source_path, &data).unwrap();
    fs::write(&index_path, &psindex).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--radius")
        .arg("180,0,60000")
        .arg("--properties")
        .arg("include:name")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut names = json
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["properties"]["name"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["east".to_string(), "west".to_string()]);

    let ndjson = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--radius")
        .arg("180,0,60000")
        .arg("--ndjson")
        .output()
        .unwrap();
    assert!(
        ndjson.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&ndjson.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&ndjson.stdout).lines().count(), 2);

    let mutually_exclusive = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--bbox")
        .arg("-180,-1,180,1")
        .arg("--radius")
        .arg("180,0,60000")
        .output()
        .unwrap();
    assert!(!mutually_exclusive.status.success());

    let invalid = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&source_path)
        .arg(&index_path)
        .arg("--radius")
        .arg("200,0,60000")
        .output()
        .unwrap();
    assert!(!invalid.status.success());

    let planar = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(179.8, 0.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(planar.clone()).unwrap();
    let planar_index = dataset
        .convert(ConvertRequest {
            payload: PayloadPlan::RowRef,
            ..ConvertRequest::default()
        })
        .unwrap();
    let planar_source = dir.join("planar.parquet");
    let planar_psi = dir.join("planar.psi");
    fs::write(&planar_source, &planar).unwrap();
    fs::write(&planar_psi, &planar_index).unwrap();

    let planar_rejected = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("query")
        .arg(&planar_source)
        .arg(&planar_psi)
        .arg("--radius")
        .arg("180,0,60000")
        .output()
        .unwrap();
    assert!(!planar_rejected.status.success());
    assert!(
        String::from_utf8_lossy(&planar_rejected.stderr).contains("spherical"),
        "stderr: {}",
        String::from_utf8_lossy(&planar_rejected.stderr)
    );
}

#[test]
fn geo_manifest_reader_handles_corrupt_bytes_without_panic() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(1.0, 2.0))]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let bytes = dataset.convert(ConvertRequest::default()).unwrap();
    assert!(read_geo_manifest(&bytes).unwrap().is_some());

    let mut huge_directory = vec![0; 32];
    huge_directory[..8].copy_from_slice(b"PSINDEX\0");
    huge_directory[8..16].copy_from_slice(&2u64.to_le_bytes());
    huge_directory[16..20].copy_from_slice(&u32::MAX.to_le_bytes());

    let mut cases = vec![Vec::new(), b"PSINDEX".to_vec(), huge_directory];
    for i in 0..bytes.len().min(160) {
        let mut mutated = bytes.clone();
        mutated[i] ^= 0xA5;
        cases.push(mutated);
    }

    for (i, case) in cases.iter().enumerate() {
        let _ = assert_no_panic(&format!("manifest case {i}"), || read_geo_manifest(case));
    }
}

#[test]
fn payload_decoders_handle_short_and_arbitrary_bytes_without_panic() {
    for len in 0..FEATURE_REF_RECORD_LEN {
        let payload = vec![0xAB; len];
        assert_no_panic(&format!("short row-ref len {len}"), || {
            assert!(decode_feature_ref_payload(&payload).is_none());
        });
        assert_no_panic(&format!("short row-wkb len {len}"), || {
            assert!(decode_feature_wkb_payload(&payload).is_none());
        });
    }

    let mut payload = vec![0xCD; FEATURE_REF_RECORD_LEN + 5];
    let (feature, wkb) = assert_no_panic("arbitrary full row-wkb payload", || {
        decode_feature_wkb_payload(&payload).unwrap()
    });
    assert_eq!(feature.row_number, u64::from_le_bytes([0xCD; 8]));
    assert_eq!(wkb, &[0xCD; 5]);

    payload.truncate(FEATURE_REF_RECORD_LEN);
    let (_feature, wkb) = assert_no_panic("empty row-wkb suffix", || {
        decode_feature_wkb_payload(&payload).unwrap()
    });
    assert!(wkb.is_empty());
}

#[test]
fn malformed_wkb_scan_returns_error_without_panic() {
    let cases = [
        Vec::new(),
        vec![2, 1, 0, 0, 0],
        vec![1, 1, 0, 0],
        vec![1, 2, 0, 0, 0, 3, 0],
    ];

    for (i, wkb) in cases.into_iter().enumerate() {
        let data = write_geoparquet(
            vec![("geometry", binary_col(&[Some(wkb)]))],
            geo_meta_wkb(&["Point"]),
        );
        let result = assert_no_panic(&format!("malformed WKB case {i}"), || {
            let mut dataset = open(data).unwrap();
            dataset.scan(Default::default())
        });
        assert!(
            result.is_err(),
            "malformed WKB case {i} unexpectedly scanned"
        );
    }
}

#[test]
fn malformed_geoparquet_metadata_errors_without_panic() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(wkb_point_2d(1.0, 2.0))]))],
        "{ not json".to_string(),
    );
    let result = assert_no_panic("malformed GeoParquet metadata", || open(data));
    assert!(matches!(result, Err(GeoError::Metadata(_))));
}

#[test]
fn feature_json_missing_property_projection_errors_without_panic() {
    let data = write_geoparquet(
        vec![
            ("geometry", binary_col(&[Some(wkb_point_2d(3.0, 4.0))])),
            (
                "name",
                Arc::new(StringArray::from(vec!["alpha"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let result = assert_no_panic("missing FeatureJson property projection", || {
        let mut dataset = open(data).unwrap();
        dataset.convert(ConvertRequest {
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::Include(vec!["missing".to_string()]),
            },
            ..ConvertRequest::default()
        })
    });
    assert!(matches!(
        result,
        Err(GeoError::PropertyColumnNotFound(name)) if name == "missing"
    ));
}
