#![cfg(feature = "geojson")]

use std::io::Cursor;

use packed_spatial_index_geo::{
    AntimeridianPolicy, BuildRequest, ConvertRequest, EnvelopePolicy, FeatureReadOrder,
    FeatureReadRequest, FeatureRef, GeoArtifactIndex, GeoError, GeoIndex, GeometryMetadataSource,
    GeometryReadMode, GeometryScan, IndexDimsRequest, NullPolicy, PayloadPlan, PropertyProjection,
    ScanRequest, SliceReader, build_geojson_stream, convert_geojson_stream, open_geo_index,
    open_geojson_slice, read_geo_manifest,
};

fn sample_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "bbox": [-10.0, 0.0, 30.0, 5.0],
        "features": [
            {
                "type": "Feature",
                "id": "west",
                "geometry": {"type": "Point", "coordinates": [-5.0, 1.0]},
                "properties": {"name": "west", "rank": 1}
            },
            {
                "type": "Feature",
                "geometry": null,
                "properties": {"name": "empty"}
            },
            {
                "type": "Feature",
                "id": 42,
                "geometry": {"type": "Point", "coordinates": [25.0, 3.0]},
                "properties": {"name": "east", "kind": "city"}
            }
        ]
    }"#
}

#[test]
fn geojson_convert_manifest_and_query_round_trip() {
    let mut source = open_geojson_slice(sample_geojson()).unwrap();
    let profile = source.profile().unwrap();
    assert_eq!(profile.source, GeometryMetadataSource::GeoJson);
    assert_eq!(profile.num_rows, 3);
    assert_eq!(profile.extent.unwrap().values, vec![-10.0, 0.0, 30.0, 5.0]);

    let bytes = source
        .convert(ConvertRequest {
            payload: PayloadPlan::RowWkb,
            ..ConvertRequest::default()
        })
        .unwrap();
    let manifest = read_geo_manifest(&bytes).unwrap().unwrap();
    assert_eq!(manifest.source_format, "geojson");
    assert_eq!(manifest.selected_column, "geometry");
    assert_eq!(manifest.feature_count, 2);
    assert_eq!(manifest.index_entry_count, 2);

    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };
    let hits = index
        .search_matches(packed_spatial_index_geo::Box2D::new(-10.0, 0.0, 0.0, 2.0))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].feature.row_number, 0);
    let wkb = match &hits[0].payload {
        packed_spatial_index_geo::GeoPayload::RowWkb(wkb) => wkb,
        other => panic!("expected row-wkb payload, got {other:?}"),
    };
    assert!(!wkb.is_empty());
}

#[test]
fn geojson_read_features_preserves_order_duplicates_and_properties() {
    let source = open_geojson_slice(sample_geojson()).unwrap();
    let records = source
        .read_features(FeatureReadRequest {
            features: vec![
                FeatureRef::row_number(2),
                FeatureRef::row_number(0),
                FeatureRef::row_number(0),
            ],
            properties: PropertyProjection::Include(vec!["name".to_string()]),
            geometry: GeometryReadMode::Wkb,
            geometry_json: true,
            order: FeatureReadOrder::RequestOrder,
            duplicates: packed_spatial_index_geo::DuplicateFeatureRows::KeepParts,
            expected_source_fingerprint: Some(source.source_fingerprint().to_string()),
            ..FeatureReadRequest::default()
        })
        .unwrap();

    assert_eq!(
        records
            .iter()
            .map(|record| record.feature.row_number)
            .collect::<Vec<_>>(),
        vec![2, 0, 0]
    );
    assert_eq!(records[0].feature.feature_id.as_deref(), Some("42"));
    assert_eq!(
        records[0]
            .properties
            .get("name")
            .and_then(serde_json::Value::as_str),
        Some("east")
    );
    assert!(records.iter().all(|record| record.geometry_wkb.is_some()));
    assert!(records.iter().all(|record| record.geometry_json.is_some()));
}

#[test]
fn geojson_read_features_defaults_geometry_json_off() {
    let source = open_geojson_slice(sample_geojson()).unwrap();
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
fn geojson_read_features_reports_missing_include_column() {
    let source = open_geojson_slice(sample_geojson()).unwrap();
    let err = source
        .read_features(FeatureReadRequest {
            features: vec![FeatureRef::row_number(0)],
            properties: PropertyProjection::Include(vec!["missing".to_string()]),
            ..FeatureReadRequest::default()
        })
        .unwrap_err();
    assert!(err.to_string().contains("missing column `missing`"));
}

#[test]
fn geojson_null_policy_preserves_source_row_numbers() {
    let mut source = open_geojson_slice(sample_geojson()).unwrap();
    let err = source.scan(ScanRequest::default()).unwrap_err();
    assert!(matches!(err, GeoError::NullGeometry { row: 1 }));

    let scan = source
        .scan(ScanRequest {
            nulls: NullPolicy::Skip,
            ..ScanRequest::default()
        })
        .unwrap();
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    assert_eq!(
        scan.features
            .iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
}

#[test]
fn geojson_detects_3d_and_rejects_forced_2d() {
    let doc = br#"{"type":"FeatureCollection","features":[
        {"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0,3.0]},"properties":{}}
    ]}"#;
    let mut source = open_geojson_slice(doc).unwrap();
    assert!(matches!(
        source.scan(ScanRequest::default()).unwrap(),
        GeometryScan::D3(_)
    ));

    let err = source
        .scan(ScanRequest {
            dims: IndexDimsRequest::D2,
            ..ScanRequest::default()
        })
        .unwrap_err();
    assert!(matches!(
        err,
        GeoError::DimMismatch {
            expected: 2,
            found: 3
        }
    ));
}

#[test]
fn geojson_antimeridian_split_and_reject() {
    let doc = br#"{"type":"FeatureCollection","features":[
        {"type":"Feature","geometry":{"type":"LineString","coordinates":[[170.0,0.0],[-170.0,1.0]]},"properties":{}}
    ]}"#;
    let mut source = open_geojson_slice(doc).unwrap();
    let scan = source
        .scan(ScanRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            ..ScanRequest::default()
        })
        .unwrap();
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    assert_eq!(scan.boxes.len(), 2);
    assert_eq!(
        scan.features
            .iter()
            .map(|feature| feature.part)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );

    let err = source
        .scan(ScanRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Reject,
            },
            ..ScanRequest::default()
        })
        .unwrap_err();
    assert!(matches!(err, GeoError::Antimeridian { row: 0 }));
}

#[test]
fn geojson_direct_walker_covers_geometry_types_and_wkb_payloads() {
    let doc = br#"{"type":"FeatureCollection","features":[
        {"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":{}},
        {"type":"Feature","geometry":{"type":"MultiPoint","coordinates":[[2.0,3.0],[3.0,4.0]]},"properties":{}},
        {"type":"Feature","geometry":{"type":"LineString","coordinates":[[4.0,5.0],[5.0,6.0]]},"properties":{}},
        {"type":"Feature","geometry":{"type":"MultiLineString","coordinates":[[[6.0,7.0],[7.0,8.0]]]},"properties":{}},
        {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[8.0,9.0],[9.0,9.0],[9.0,10.0],[8.0,9.0]]]},"properties":{}},
        {"type":"Feature","geometry":{"type":"MultiPolygon","coordinates":[[[[10.0,11.0],[11.0,11.0],[11.0,12.0],[10.0,11.0]]]]},"properties":{}},
        {"type":"Feature","geometry":{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[12.0,13.0]},{"type":"LineString","coordinates":[[13.0,14.0],[14.0,15.0]]}]},"properties":{}}
    ]}"#;
    let mut source = open_geojson_slice(doc).unwrap();
    let scan = source
        .scan(ScanRequest {
            payload: PayloadPlan::RowWkb,
            ..ScanRequest::default()
        })
        .unwrap();
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    assert_eq!(scan.features.len(), 7);
    assert!(
        scan.payloads()
            .unwrap()
            .iter()
            .all(|payload| !payload.is_empty())
    );
    assert!(
        scan.profile
            .geometry_types
            .types
            .iter()
            .any(|kind| kind == "GeometryCollection")
    );
}

#[test]
fn geojson_direct_walker_handles_empty_and_invalid_coordinates() {
    let empty = br#"{"type":"FeatureCollection","features":[
        {"type":"Feature","geometry":{"type":"LineString","coordinates":[]},"properties":{}}
    ]}"#;
    let mut source = open_geojson_slice(empty).unwrap();
    let err = source.scan(ScanRequest::default()).unwrap_err();
    assert!(matches!(err, GeoError::NullGeometry { row: 0 }));
    let scan = source
        .scan(ScanRequest {
            nulls: NullPolicy::Skip,
            ..ScanRequest::default()
        })
        .unwrap();
    let GeometryScan::D2(scan) = scan else {
        panic!("expected 2D scan");
    };
    assert!(scan.features.is_empty());

    let invalid = br#"{"type":"FeatureCollection","features":[
        {"type":"Feature","geometry":{"type":"Point","coordinates":["x",2.0]},"properties":{}}
    ]}"#;
    let mut source = open_geojson_slice(invalid).unwrap();
    let err = source.scan(ScanRequest::default()).unwrap_err();
    assert!(err.to_string().contains("coordinate x is not a number"));
}

#[test]
fn geojson_stream_convert_and_build_match_eager_source_identity() {
    let eager = open_geojson_slice(sample_geojson()).unwrap();
    let mut bytes = Vec::new();
    let artifact = convert_geojson_stream(
        Cursor::new(sample_geojson()),
        ConvertRequest {
            payload: PayloadPlan::RowRef,
            nulls: NullPolicy::Skip,
            ..ConvertRequest::default()
        },
        &mut bytes,
    )
    .unwrap();
    assert_eq!(
        artifact.manifest.source_fingerprint,
        eager.source_fingerprint()
    );
    assert_eq!(artifact.manifest.feature_count, 2);
    assert_eq!(artifact.manifest.index_entry_count, 2);

    let GeoIndex::D2(index) = build_geojson_stream(
        Cursor::new(sample_geojson()),
        BuildRequest {
            nulls: NullPolicy::Skip,
            ..BuildRequest::default()
        },
    )
    .unwrap() else {
        panic!("expected 2D stream-built index");
    };
    let hits = index
        .search_feature_refs(packed_spatial_index_geo::Box2D::new(20.0, 0.0, 30.0, 5.0))
        .unwrap();
    assert_eq!(
        hits.iter()
            .map(|feature| feature.row_number)
            .collect::<Vec<_>>(),
        vec![2]
    );
}

#[cfg(feature = "parquet")]
#[test]
fn geojson_cli_build_and_query_detects_format() {
    let dir = std::env::temp_dir().join(format!(
        "psi_geojson_cli_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("points.geojson");
    let output = dir.join("points.psi");
    std::fs::write(&input, sample_geojson()).unwrap();

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
        .arg("-10,0,0,2")
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
    assert!(stdout.contains("\"name\": \"west\""), "{stdout}");
}

#[cfg(feature = "parquet")]
#[test]
fn geojson_cli_build_single_feature_uses_eager_fallback() {
    let dir = std::env::temp_dir().join(format!(
        "psi_geojson_cli_single_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("single.geojson");
    let output = dir.join("single.psi");
    std::fs::write(
        &input,
        br#"{"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":{"name":"solo"}}"#,
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_gp2psindex");
    let build = std::process::Command::new(bin)
        .arg("build")
        .arg(&input)
        .arg(&output)
        .arg("--payload")
        .arg("row-ref")
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "build failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    let manifest = read_geo_manifest(&std::fs::read(output).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(manifest.feature_count, 1);
}
