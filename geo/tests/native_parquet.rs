//! Native Apache Parquet GEOMETRY / GEOGRAPHY logical type tests.

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use packed_spatial_index::read_metadata;
use packed_spatial_index_geo::{
    Box2D, BuildOpts, ConvertOpts, GeoError, GeometryMetadataSource, ReadOpts, SliceReader,
    StreamIndex2D, build_index_2d, convert_2d, decode_row_wkb_payload, inspect, read_bboxes_2d,
    read_bboxes_2d_with_opts, read_bboxes_3d,
};
use parquet::arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions};
use parquet::basic::{LogicalType, Repetition, Type as ParquetPhysicalType};
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};

fn geometry_fixture() -> Bytes {
    Bytes::from_static(include_bytes!(
        "fixtures/parquet-geospatial/geospatial.parquet"
    ))
}

fn geography_fixture() -> Bytes {
    Bytes::from_static(include_bytes!(
        "fixtures/parquet-geospatial/crs-geography.parquet"
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

fn binary_col(wkbs: &[Vec<u8>]) -> ArrayRef {
    let values: Vec<&[u8]> = wkbs.iter().map(Vec::as_slice).collect();
    Arc::new(BinaryArray::from(values))
}

fn native_geometry_schema(names: &[&str]) -> SchemaDescriptor {
    let fields = names
        .iter()
        .map(|name| {
            Arc::new(
                ParquetType::primitive_type_builder(name, ParquetPhysicalType::BYTE_ARRAY)
                    .with_repetition(Repetition::REQUIRED)
                    .with_logical_type(Some(LogicalType::Geometry { crs: None }))
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

fn two_native_geometry_columns() -> Bytes {
    let batch = RecordBatch::try_from_iter(vec![
        ("geom_a", binary_col(&[wkb_point_2d(0.0, 0.0)])),
        ("geom_b", binary_col(&[wkb_point_2d(10.0, 10.0)])),
    ])
    .unwrap();
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Binary);

    let options = ArrowWriterOptions::new()
        .with_parquet_schema(native_geometry_schema(&["geom_a", "geom_b"]));
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new_with_options(&mut buf, batch.schema(), options).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}

fn one_native_3d_geometry_column() -> Bytes {
    let batch = RecordBatch::try_from_iter(vec![(
        "geometry",
        binary_col(&[wkb_point_3d(1.0, 2.0, 3.0)]),
    )])
    .unwrap();
    let options =
        ArrowWriterOptions::new().with_parquet_schema(native_geometry_schema(&["geometry"]));
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new_with_options(&mut buf, batch.schema(), options).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}

#[test]
fn native_geometry_without_geo_metadata_reads_builds_and_converts() {
    let info = inspect(srid_fixture()).unwrap();
    assert_eq!(
        info.metadata_source,
        GeometryMetadataSource::ParquetGeospatial
    );
    assert_eq!(info.version, "parquet-geospatial");
    assert_eq!(info.geometry_column, "geometry");
    assert_eq!(info.dims, 2);
    assert_eq!(info.encoding, "GEOMETRY");
    assert_eq!(info.crs.as_deref(), Some("srid:5070"));
    assert!(!info.has_covering);
    assert_eq!(info.num_rows, 1);

    let boxes = read_bboxes_2d(srid_fixture()).unwrap();
    assert_eq!(boxes.len(), 1);

    let index = build_index_2d(srid_fixture(), BuildOpts::default()).unwrap();
    assert_eq!(
        index.search(Box2D::new(
            -10_000_000.0,
            -10_000_000.0,
            10_000_000.0,
            10_000_000.0
        )),
        vec![0]
    );

    let psindex = convert_2d(srid_fixture(), ConvertOpts::default()).unwrap();
    let meta = read_metadata(&psindex).unwrap();
    assert_eq!(meta.crs.as_deref(), Some("srid:5070"));

    let stream = StreamIndex2D::open(SliceReader::new(psindex.as_slice())).unwrap();
    let hits = stream
        .search_payloads(Box2D::new(
            -10_000_000.0,
            -10_000_000.0,
            10_000_000.0,
            10_000_000.0,
        ))
        .unwrap();
    assert_eq!(hits.len(), 1);
    let (row, wkb) = decode_row_wkb_payload(&hits[0].1).unwrap();
    assert_eq!(row, 0);
    assert!(!wkb.is_empty());
}

#[test]
fn native_geography_indexes_coordinate_aabbs() {
    let info = inspect(geography_fixture()).unwrap();
    assert_eq!(
        info.metadata_source,
        GeometryMetadataSource::ParquetGeospatial
    );
    assert_eq!(info.geometry_column, "geography");
    assert_eq!(info.dims, 2);
    assert_eq!(info.encoding, "GEOGRAPHY(SPHERICAL)");
    assert_eq!(info.crs, None);

    let boxes = read_bboxes_2d(geography_fixture()).unwrap();
    assert_eq!(boxes.len(), 1);
}

#[test]
fn native_geometry_fixture_detects_3d_and_supports_3d_scan() {
    let info = inspect(geometry_fixture()).unwrap();
    assert_eq!(
        info.metadata_source,
        GeometryMetadataSource::ParquetGeospatial
    );
    assert_eq!(info.geometry_column, "geometry");
    assert_eq!(info.encoding, "GEOMETRY");
    assert_eq!(info.dims, 3);

    assert!(matches!(
        read_bboxes_2d(geometry_fixture()),
        Err(GeoError::DimMismatch {
            expected: 2,
            found: 3
        })
    ));

    let boxes = read_bboxes_3d(one_native_3d_geometry_column()).unwrap();
    assert_eq!(boxes.len(), 1);
    assert!(boxes[0].overlaps(packed_spatial_index_geo::Box3D::new(
        1.0, 2.0, 3.0, 1.0, 2.0, 3.0
    )));
}

#[test]
fn explicit_geometry_column_disambiguates_native_columns() {
    let data = two_native_geometry_columns();

    match read_bboxes_2d(data.clone()) {
        Err(GeoError::AmbiguousGeometryColumn { columns }) => {
            assert_eq!(columns, vec!["geom_a".to_string(), "geom_b".to_string()]);
        }
        other => panic!("expected ambiguous geometry column, got {other:?}"),
    }

    let read_opts = ReadOpts {
        geometry_column: Some("geom_b".to_string()),
    };
    let boxes = read_bboxes_2d_with_opts(data.clone(), read_opts).unwrap();
    assert_eq!(boxes.len(), 1);
    assert!(boxes[0].overlaps(Box2D::new(9.0, 9.0, 11.0, 11.0)));

    let build_opts = BuildOpts {
        geometry_column: Some("geom_b".to_string()),
        ..Default::default()
    };
    let index = build_index_2d(data.clone(), build_opts).unwrap();
    assert_eq!(index.search(Box2D::new(9.0, 9.0, 11.0, 11.0)), vec![0]);

    let convert_opts = ConvertOpts {
        geometry_column: Some("geom_b".to_string()),
        ..Default::default()
    };
    let psindex = convert_2d(data, convert_opts).unwrap();
    let stream = StreamIndex2D::open(SliceReader::new(psindex.as_slice())).unwrap();
    assert_eq!(
        stream.search(Box2D::new(9.0, 9.0, 11.0, 11.0)).unwrap(),
        vec![0]
    );
}

#[test]
fn explicit_missing_geometry_column_errors_clearly() {
    let err = read_bboxes_2d_with_opts(
        srid_fixture(),
        ReadOpts {
            geometry_column: Some("missing".to_string()),
        },
    )
    .unwrap_err();

    assert!(matches!(err, GeoError::GeometryColumnNotFound(name) if name == "missing"));
}
