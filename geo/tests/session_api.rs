use std::process::Command;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Float64Array, ListArray, StringArray, StructArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, ConvertRequest, CoordinateDims, EnvelopePolicy,
    FEATURE_REF_RECORD_LEN, GeoError, GeoIndex, GeometryEncoding, GeometryMetadataSource,
    GeometryScan, GeometrySelector, IndexDimsRequest, InspectRequest, NullPolicy, PayloadPlan,
    PropertyProjection, SliceReader, StreamIndex2D, decode_feature_ref_payload,
    decode_feature_wkb_payload, open, read_geo_manifest,
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
    let hits = index.search_features(Box2D::new(-1.0, -1.0, 1.0, 1.0));
    assert_eq!(hits[0].row_number, 0);
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
    assert_eq!(json["properties"]["name"], "alpha");
    assert_eq!(json["geometry"]["type"], "Point");
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
    let mut dataset = open(data).unwrap();
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
