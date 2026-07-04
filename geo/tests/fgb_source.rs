#![cfg(feature = "flatgeobuf")]

use std::io::Cursor;

use flatgeobuf::{ColumnType, FgbCrs, FgbWriter, FgbWriterOptions, GeometryType};
use geozero::geojson::GeoJson;
use geozero::{ColumnValue, PropertyProcessor};
use packed_spatial_index_geo::{
    ConvertRequest, DuplicateFeatureRows, FeatureReadOrder, FeatureReadRequest, FeatureRef,
    GeoArtifactIndex, GeoError, GeometryMetadataSource, GeometryReadMode, PayloadPlan,
    PropertyProjection, SliceReader, open_flatgeobuf, open_geo_index, read_geo_manifest,
};

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
        GeoJson(r#"{"type":"Point","coordinates":[-5,1]}"#),
        |feat| {
            feat.property(0, "name", &ColumnValue::String("west"))
                .unwrap();
        },
    )
    .unwrap();
    fgb.add_feature_geom(
        GeoJson(r#"{"type":"Point","coordinates":[25,3]}"#),
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

#[test]
fn flatgeobuf_convert_manifest_and_query_round_trip() {
    let bytes = sample_fgb();
    let mut source = open_flatgeobuf(Cursor::new(bytes)).unwrap();
    let profile = source.profile().unwrap();
    assert_eq!(profile.source, GeometryMetadataSource::FlatGeobuf);
    assert_eq!(profile.num_rows, 2);

    let artifact_bytes = source
        .convert(ConvertRequest {
            payload: PayloadPlan::RowWkb,
            ..ConvertRequest::default()
        })
        .unwrap();
    let manifest = read_geo_manifest(&artifact_bytes).unwrap().unwrap();
    assert_eq!(manifest.source_format, "flatgeobuf");
    assert_eq!(manifest.feature_count, 2);
    assert_eq!(manifest.index_entry_count, 2);

    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(artifact_bytes)).unwrap()
    else {
        panic!("expected 2D artifact");
    };
    let hits = index
        .search_hits(packed_spatial_index_geo::Box2D::new(20.0, 0.0, 30.0, 5.0))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].feature.row_number, 1);
}

#[test]
fn flatgeobuf_read_features_materializes_properties_and_geometry() {
    let bytes = sample_fgb();
    let mut source = open_flatgeobuf(Cursor::new(bytes)).unwrap();
    let records = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef::row_number(1)],
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            geometry: GeometryReadMode::Wkb,
            geometry_json: true,
            expected_source_fingerprint: Some(source.source_fingerprint().to_string()),
            ..FeatureReadRequest::default()
        })
        .unwrap();

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].feature.row_number, 1);
    assert_eq!(
        records[0]
            .properties
            .get("name")
            .and_then(serde_json::Value::as_str),
        Some("east")
    );
    assert!(records[0].geometry_wkb.is_some());
    assert!(records[0].geometry_json.is_some());
}

#[test]
fn flatgeobuf_read_features_defaults_geometry_json_off() {
    let bytes = sample_fgb();
    let mut source = open_flatgeobuf(Cursor::new(bytes)).unwrap();
    let records = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef::row_number(0)],
            geometry: GeometryReadMode::Wkb,
            ..FeatureReadRequest::default()
        })
        .unwrap();

    assert!(records[0].geometry_wkb.is_some());
    assert!(records[0].geometry_json.is_none());
}

#[test]
fn flatgeobuf_read_features_keeps_duplicate_rows_after_move_out() {
    let bytes = sample_fgb();
    let mut source = open_flatgeobuf(Cursor::new(bytes)).unwrap();
    let records = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef::row_number(1), FeatureRef::row_number(1)],
            geometry: GeometryReadMode::Wkb,
            order: FeatureReadOrder::RequestOrder,
            duplicates: DuplicateFeatureRows::KeepParts,
            ..FeatureReadRequest::default()
        })
        .unwrap();

    assert_eq!(records.len(), 2);
    assert_eq!(records[0].feature.row_number, 1);
    assert_eq!(records[1].feature.row_number, 1);
    assert!(records.iter().all(|record| record.geometry_wkb.is_some()));
    assert!(records.iter().all(|record| record.geometry_json.is_none()));
}

#[test]
fn flatgeobuf_read_features_consumes_reader() {
    let bytes = sample_fgb();
    let mut source = open_flatgeobuf(Cursor::new(bytes)).unwrap();
    source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef::row_number(0)],
            ..FeatureReadRequest::default()
        })
        .unwrap();
    let err = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef::row_number(1)],
            ..FeatureReadRequest::default()
        })
        .unwrap_err();
    assert!(matches!(err, GeoError::DatasetConsumed));
}

#[cfg(feature = "parquet")]
#[test]
fn flatgeobuf_cli_build_and_query_detects_format() {
    let dir = std::env::temp_dir().join(format!(
        "psi_fgb_cli_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("points.fgb");
    let output = dir.join("points.psi");
    std::fs::write(&input, sample_fgb()).unwrap();

    let bin = env!("CARGO_BIN_EXE_gp2psindex");
    let build = std::process::Command::new(bin)
        .arg("build")
        .arg(&input)
        .arg(&output)
        .arg("--payload")
        .arg("row-wkb")
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "build failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );

    let query = std::process::Command::new(bin)
        .arg("query")
        .arg(&input)
        .arg(&output)
        .arg("--bbox")
        .arg("20,0,30,5")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        query.status.success(),
        "query failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&query.stdout),
        String::from_utf8_lossy(&query.stderr)
    );
    let stdout = String::from_utf8(query.stdout).unwrap();
    assert!(stdout.contains("\"name\": \"east\""), "{stdout}");
}
