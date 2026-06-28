use std::process::Command;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use packed_spatial_index_geo::{
    AntimeridianPolicy, CoordinateDims, EnvelopePolicy, GeometrySelectionReason, GeometrySelector,
    PayloadPlan, PropertyProjection, SelectionStatus, ValidateRequest, ValidationCode, open,
};
use parquet::arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions};
use parquet::basic::{GeometryType, LogicalType, Repetition, Type as ParquetPhysicalType};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};

fn compat_point_wkb_fixture() -> Bytes {
    Bytes::from_static(include_bytes!(
        "fixtures/geoparquet-compat/data-point-encoding_wkb.parquet"
    ))
}

fn compat_native_crs_fixture() -> Bytes {
    Bytes::from_static(include_bytes!(
        "fixtures/geoparquet-compat/example-crs_vermont-custom.parquet"
    ))
}

fn geography_fixture() -> Bytes {
    Bytes::from_static(include_bytes!(
        "fixtures/parquet-geospatial/crs-geography.parquet"
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

fn has_issue(report: &packed_spatial_index_geo::ValidationReport, code: ValidationCode) -> bool {
    report.issues.iter().any(|issue| issue.code == code)
}

#[test]
fn validation_metadata_only_succeeds_on_official_wkb_fixture() {
    let mut dataset = open(compat_point_wkb_fixture()).unwrap();
    let report = dataset.validate(ValidateRequest::default()).unwrap();
    assert!(report.ok);
    assert!(report.issues.is_empty());
    assert_eq!(report.discovery.num_rows, 4);
    assert_eq!(
        report.selected,
        SelectionStatus::Selected {
            column: "geometry".to_string(),
            reason: GeometrySelectionReason::GeoParquetPrimary,
        }
    );
    assert_eq!(report.profile.unwrap().coordinate_dims, CoordinateDims::Xy);
}

#[test]
fn validation_reports_native_parquet_geospatial_stats() {
    let mut dataset = open(compat_native_crs_fixture()).unwrap();
    let report = dataset.validate(ValidateRequest::default()).unwrap();
    assert!(report.ok);
    assert_eq!(report.native_stats.len(), 1);
    let stats = &report.native_stats[0];
    assert_eq!(stats.column, "geometry");
    assert_eq!(stats.row_group_count, 1);
    assert_eq!(stats.groups_with_bbox, 1);
    assert_eq!(stats.groups_with_types, 1);
    assert_eq!(stats.inferred_dims, CoordinateDims::Xy);
    assert!(stats.row_groups[0].bbox.is_some());
}

#[test]
fn validation_reports_explicit_missing_and_ambiguous_selection() {
    let mut dataset = open(compat_point_wkb_fixture()).unwrap();
    let missing = dataset
        .validate(ValidateRequest {
            selector: GeometrySelector::Name("missing".to_string()),
            ..ValidateRequest::default()
        })
        .unwrap();
    assert!(!missing.ok);
    assert!(has_issue(&missing, ValidationCode::GeometryColumnNotFound));

    let data = native_parquet(
        &["geom_a", "geom_b"],
        vec![vec![wkb_point_2d(0.0, 0.0)], vec![wkb_point_2d(1.0, 1.0)]],
    );
    let mut dataset = open(data).unwrap();
    let report = dataset.validate(ValidateRequest::default()).unwrap();
    assert!(!report.ok);
    assert!(has_issue(&report, ValidationCode::AmbiguousGeometryColumn));
}

#[test]
fn validation_warns_for_geography_coordinate_aabb() {
    let mut dataset = open(geography_fixture()).unwrap();
    let report = dataset.validate(ValidateRequest::default()).unwrap();
    assert!(report.ok);
    assert!(has_issue(&report, ValidationCode::GeographyCoordinateAabb));
}

#[test]
fn validation_exact_antimeridian_rejects_or_splits() {
    let data = write_geoparquet(
        vec![(
            "geometry",
            binary_col(&[Some(wkb_line_2d(&[(170.0, 0.0), (-170.0, 1.0)]))]),
        )],
        geo_meta_wkb(&["LineString"]),
    );
    let mut dataset = open(data.clone()).unwrap();
    let rejected = dataset
        .validate(ValidateRequest {
            exact: true,
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Reject,
            },
            ..ValidateRequest::default()
        })
        .unwrap();
    assert!(!rejected.ok);
    assert!(has_issue(&rejected, ValidationCode::AntimeridianWrap));

    let mut dataset = open(data).unwrap();
    let split = dataset
        .validate(ValidateRequest {
            exact: true,
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            ..ValidateRequest::default()
        })
        .unwrap();
    assert!(split.ok);
    assert!(!has_issue(&split, ValidationCode::AntimeridianWrap));
}

#[test]
fn validation_exact_malformed_wkb_is_report_error_without_panic() {
    let data = write_geoparquet(
        vec![("geometry", binary_col(&[Some(vec![1, 1, 0])]))],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let report = dataset
        .validate(ValidateRequest {
            exact: true,
            ..ValidateRequest::default()
        })
        .unwrap();
    assert!(!report.ok);
    assert!(has_issue(&report, ValidationCode::ExactScanFailed));
}

#[test]
fn validation_feature_json_missing_projection_is_report_error() {
    let data = write_geoparquet(
        vec![
            ("geometry", binary_col(&[Some(wkb_point_2d(1.0, 2.0))])),
            (
                "name",
                Arc::new(StringArray::from(vec!["alpha"])) as ArrayRef,
            ),
        ],
        geo_meta_wkb(&["Point"]),
    );
    let mut dataset = open(data).unwrap();
    let report = dataset
        .validate(ValidateRequest {
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::Include(vec!["missing".to_string()]),
            },
            ..ValidateRequest::default()
        })
        .unwrap();
    assert!(!report.ok);
    assert!(has_issue(&report, ValidationCode::ProjectedPropertyMissing));
}

#[test]
fn cli_validate_json_and_strict_smoke() {
    let output = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("validate")
        .arg("tests/fixtures/geoparquet-compat/data-point-encoding_wkb.parquet")
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
    assert_eq!(json["ok"], true);

    let strict = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("validate")
        .arg("tests/fixtures/parquet-geospatial/crs-geography.parquet")
        .arg("--strict")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(!strict.status.success());
    assert!(
        String::from_utf8_lossy(&strict.stdout).contains("GeographyCoordinateAabb"),
        "stdout: {}",
        String::from_utf8_lossy(&strict.stdout)
    );

    let invalid_properties = Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .arg("validate")
        .arg("tests/fixtures/geoparquet-compat/data-point-encoding_wkb.parquet")
        .arg("--properties")
        .arg("include:name")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(!invalid_properties.status.success());
    assert!(
        String::from_utf8_lossy(&invalid_properties.stderr)
            .contains("--properties can only be used with --payload feature-json"),
        "stderr: {}",
        String::from_utf8_lossy(&invalid_properties.stderr)
    );
}
