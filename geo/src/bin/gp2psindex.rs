//! Command-line converter and inspector for geospatial Parquet inputs.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use arrow_json::LineDelimitedWriter;
use base64::Engine as _;
#[cfg(feature = "flatgeobuf")]
use packed_spatial_index_geo::open_flatgeobuf;
#[cfg(feature = "geojson")]
use packed_spatial_index_geo::open_geojson;
use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, Box3D, ConvertRequest, DuplicateFeatureRows, EnvelopePolicy,
    FeatureFilterRequest, FeatureReadOrder, FeatureReadRequest, FeatureRecord, FeatureRef,
    FeatureRows, GeoArtifact, GeoArtifactIndex, GeoArtifactIndex2D, GeoArtifactIndex3D,
    GeoArtifactManifest, GeoDiscovery, GeoError, GeoQuery2D, GeometryProfile, GeometryReadMode,
    GeometryScan, GeometrySelector, IndexDimsRequest, InspectRequest, NonPlanarExactPolicy,
    NullPolicy, PayloadPlan, PropertyProjection, RangeReader, ScanRequest, SliceReader,
    SpatialPredicate, StoragePrecision, ValidateRequest, ValidationReport, ValidationSeverity,
    open, open_geo_index,
};

const USAGE: &str = "\
usage:
  gp2psindex discover <input> [--format parquet|flatgeobuf|geojson] [--json]
  gp2psindex inspect <input> [--format parquet|flatgeobuf|geojson] [--geometry-column name] [--exact] [--json]
  gp2psindex build <input> <output.psi>
      [--format parquet|flatgeobuf|geojson]
      [--geometry-column name]
      [--dims auto|2d|3d]
      [--precision f64|f32]
      [--nulls error|skip]
      [--payload none|row-ref|row-wkb|feature-json]
      [--properties none|all|include:a,b|exclude:a,b]
      [--antimeridian reject|split|world]
      [--no-interleave]
  gp2psindex validate <input>
      [--format parquet|flatgeobuf|geojson]
      [--geometry-column name]
      [--exact]
      [--json]
      [--strict]
      [--dims auto|2d|3d]
      [--nulls error|skip]
      [--payload none|row-ref|row-wkb|feature-json]
      [--properties none|all|include:a,b|exclude:a,b]
      [--antimeridian reject|split|world]
  gp2psindex query <source> <index.psi>
      [--format parquet|flatgeobuf|geojson]
      (--bbox xmin,ymin,xmax,ymax | --radius lon,lat,metres)
      [--exact]
      [--predicate intersects]
      [--treat-nonplanar-as-planar]
      [--geometry none|wkb]
      [--properties none|all|include:a,b|exclude:a,b]
      [--order source|hit]
      [--duplicates dedup|parts]
      [--json|--ndjson]
      [--allow-source-mismatch]
      (against a 3D index: --bbox takes xmin,ymin,zmin,xmax,ymax,zmax;
       --radius/--exact/--predicate/--treat-nonplanar-as-planar are 2D-only)";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err("missing command".into());
    };
    match command {
        "discover" => discover_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        "inspect" => inspect_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        "build" => build_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        "validate" => validate_cmd(&args[1..]),
        "query" => query_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        _ => Err(format!("unknown command `{command}`").into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Parquet,
    FlatGeobuf,
    GeoJson,
}

fn source_kind(
    path: &str,
    format: Option<String>,
) -> Result<SourceKind, Box<dyn std::error::Error>> {
    if let Some(format) = format {
        return match format.as_str() {
            "parquet" => Ok(SourceKind::Parquet),
            "flatgeobuf" | "fgb" => Ok(SourceKind::FlatGeobuf),
            "geojson" | "json" => Ok(SourceKind::GeoJson),
            _ => Err(format!("invalid --format `{format}`").into()),
        };
    }
    if let Some(kind) = source_kind_from_extension(path) {
        return Ok(kind);
    }
    source_kind_from_signature(path)
}

fn source_kind_from_extension(path: &str) -> Option<SourceKind> {
    let ext = Path::new(path)
        .extension()
        .and_then(|value| value.to_str())?
        .to_ascii_lowercase();
    match ext.as_str() {
        "parquet" | "pq" => Some(SourceKind::Parquet),
        "fgb" => Some(SourceKind::FlatGeobuf),
        "geojson" | "json" => Some(SourceKind::GeoJson),
        _ => None,
    }
}

fn source_kind_from_signature(path: &str) -> Result<SourceKind, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = [0u8; 16];
    let len = file.read(&mut buf)?;
    let bytes = &buf[..len];
    if bytes.starts_with(b"PAR1") {
        return Ok(SourceKind::Parquet);
    }
    if bytes.starts_with(b"fgb\x03fgb\0") {
        return Ok(SourceKind::FlatGeobuf);
    }
    if bytes
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .is_some_and(|byte| byte == b'{' || byte == b'[')
    {
        return Ok(SourceKind::GeoJson);
    }
    Err("could not detect input format; pass --format parquet|flatgeobuf|geojson".into())
}

fn inspect_source_profile(
    kind: SourceKind,
    input: &str,
    selector: GeometrySelector,
) -> Result<GeometryProfile, Box<dyn std::error::Error>> {
    match kind {
        SourceKind::Parquet => {
            let mut dataset = open(File::open(input)?)?;
            Ok(dataset.inspect(InspectRequest {
                selector,
                exact: false,
            })?)
        }
        SourceKind::FlatGeobuf => {
            check_single_geometry_selector(&selector, "FlatGeobuf")?;
            #[cfg(feature = "flatgeobuf")]
            {
                Ok(open_flatgeobuf(File::open(input)?)?.profile())
            }
            #[cfg(not(feature = "flatgeobuf"))]
            {
                let _ = input;
                Err("this gp2psindex build was compiled without FlatGeobuf support".into())
            }
        }
        SourceKind::GeoJson => {
            check_single_geometry_selector(&selector, "GeoJSON")?;
            #[cfg(feature = "geojson")]
            {
                Ok(open_geojson(File::open(input)?)?.profile())
            }
            #[cfg(not(feature = "geojson"))]
            {
                let _ = input;
                Err("this gp2psindex build was compiled without GeoJSON support".into())
            }
        }
    }
}

fn convert_source(
    kind: SourceKind,
    input: &str,
    request: ConvertRequest,
    out: &mut Vec<u8>,
) -> Result<GeoArtifact, Box<dyn std::error::Error>> {
    match kind {
        SourceKind::Parquet => {
            let mut dataset = open(File::open(input)?)?;
            Ok(dataset.convert_into(request, out)?)
        }
        SourceKind::FlatGeobuf => {
            #[cfg(feature = "flatgeobuf")]
            {
                let mut dataset = open_flatgeobuf(File::open(input)?)?;
                Ok(dataset.convert_into(request, out)?)
            }
            #[cfg(not(feature = "flatgeobuf"))]
            {
                let _ = (input, request, out);
                Err("this gp2psindex build was compiled without FlatGeobuf support".into())
            }
        }
        SourceKind::GeoJson => {
            #[cfg(feature = "geojson")]
            {
                let mut dataset = open_geojson(File::open(input)?)?;
                Ok(dataset.convert_into(request, out)?)
            }
            #[cfg(not(feature = "geojson"))]
            {
                let _ = (input, request, out);
                Err("this gp2psindex build was compiled without GeoJSON support".into())
            }
        }
    }
}

fn scan_source(
    kind: SourceKind,
    input: &str,
    request: ScanRequest,
) -> Result<GeometryScan, Box<dyn std::error::Error>> {
    match kind {
        SourceKind::Parquet => {
            let mut dataset = open(File::open(input)?)?;
            Ok(dataset.scan(request)?)
        }
        SourceKind::FlatGeobuf => {
            #[cfg(feature = "flatgeobuf")]
            {
                let mut dataset = open_flatgeobuf(File::open(input)?)?;
                Ok(dataset.scan(request)?)
            }
            #[cfg(not(feature = "flatgeobuf"))]
            {
                let _ = (input, request);
                Err("this gp2psindex build was compiled without FlatGeobuf support".into())
            }
        }
        SourceKind::GeoJson => {
            #[cfg(feature = "geojson")]
            {
                let mut dataset = open_geojson(File::open(input)?)?;
                Ok(dataset.scan(request)?)
            }
            #[cfg(not(feature = "geojson"))]
            {
                let _ = (input, request);
                Err("this gp2psindex build was compiled without GeoJSON support".into())
            }
        }
    }
}

fn validate_source(
    kind: SourceKind,
    input: &str,
    request: ScanRequest,
    as_json: bool,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    match scan_source(kind, input, request) {
        Ok(scan) => {
            let profile = scan_profile(&scan);
            if as_json {
                serde_json::to_writer_pretty(
                    std::io::stdout(),
                    &serde_json::json!({ "ok": true, "profile": profile }),
                )?;
                println!();
            } else {
                println!("status: ok");
                print_profile(profile);
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(err) => {
            if as_json {
                serde_json::to_writer_pretty(
                    std::io::stdout(),
                    &serde_json::json!({ "ok": false, "error": err.to_string() }),
                )?;
                println!();
            } else {
                eprintln!("status: error");
                eprintln!("issue: {err}");
            }
            Ok(ExitCode::FAILURE)
        }
    }
}

fn scan_profile(scan: &GeometryScan) -> &GeometryProfile {
    match scan {
        GeometryScan::D2(scan) => &scan.profile,
        GeometryScan::D3(scan) => &scan.profile,
    }
}

fn check_single_geometry_selector(
    selector: &GeometrySelector,
    source: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match selector {
        GeometrySelector::Default | GeometrySelector::FirstUsable => Ok(()),
        GeometrySelector::Name(name) if name == "geometry" => Ok(()),
        GeometrySelector::Name(name) => Err(Box::new(GeoError::GeometryColumnNotFound(
            name.clone(),
        ))),
        GeometrySelector::GeoParquetPrimary | GeometrySelector::SingleNativeParquet => {
            Err(format!(
                "selector applies to Parquet sources; use Default or Name(\"geometry\") for {source}"
            )
            .into())
        }
    }
}

fn discover_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&["--format", "--json"])?;
    let input = parsed.required_pos(0, "input")?;
    parsed.no_extra_pos(1)?;
    let kind = source_kind(input, parsed.option("--format")?)?;
    match kind {
        SourceKind::Parquet => {
            let dataset = open(File::open(input)?)?;
            if parsed.flag("--json") {
                serde_json::to_writer_pretty(std::io::stdout(), dataset.discovery())?;
                println!();
            } else {
                print_discovery(dataset.discovery());
            }
        }
        SourceKind::FlatGeobuf | SourceKind::GeoJson => {
            let profile = inspect_source_profile(kind, input, GeometrySelector::Default)?;
            if parsed.flag("--json") {
                serde_json::to_writer_pretty(std::io::stdout(), &profile)?;
                println!();
            } else {
                print_profile(&profile);
            }
        }
    }
    Ok(())
}

fn inspect_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&["--format", "--geometry-column", "--exact", "--json"])?;
    let input = parsed.required_pos(0, "input")?;
    parsed.no_extra_pos(1)?;
    let kind = source_kind(input, parsed.option("--format")?)?;
    let selector = geometry_selector(parsed.option("--geometry-column")?);
    let profile = match kind {
        SourceKind::Parquet => {
            let mut dataset = open(File::open(input)?)?;
            dataset.inspect(InspectRequest {
                selector,
                exact: parsed.flag("--exact"),
            })?
        }
        SourceKind::FlatGeobuf | SourceKind::GeoJson => {
            inspect_source_profile(kind, input, selector)?
        }
    };
    if parsed.flag("--json") {
        serde_json::to_writer_pretty(std::io::stdout(), &profile)?;
        println!();
    } else {
        print_profile(&profile);
    }
    Ok(())
}

fn build_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&[
        "--format",
        "--geometry-column",
        "--dims",
        "--precision",
        "--nulls",
        "--payload",
        "--properties",
        "--antimeridian",
        "--no-interleave",
    ])?;
    let input = parsed.required_pos(0, "input")?;
    let output = parsed.required_pos(1, "output.psi")?;
    parsed.no_extra_pos(2)?;
    let kind = source_kind(input, parsed.option("--format")?)?;
    let payload = parse_payload(
        parsed.option("--payload")?.as_deref().unwrap_or("row-wkb"),
        parsed.option("--properties")?,
    )?;
    let request = ConvertRequest {
        selector: geometry_selector(parsed.option("--geometry-column")?),
        dims: parse_dims(parsed.option("--dims")?.as_deref().unwrap_or("auto"))?,
        nulls: parse_nulls(parsed.option("--nulls")?.as_deref().unwrap_or("skip"))?,
        envelope: parse_antimeridian(parsed.option("--antimeridian")?)?,
        precision: parse_precision(parsed.option("--precision")?.as_deref().unwrap_or("f64"))?,
        payload,
        interleaved: !parsed.flag("--no-interleave"),
        ..ConvertRequest::default()
    };
    let mut bytes = Vec::new();
    let artifact = convert_source(kind, input, request, &mut bytes)?;
    std::fs::write(output, &bytes)?;
    println!(
        "wrote {output}: {} bytes, {} features, {} index entries",
        bytes.len(),
        artifact.manifest.feature_count,
        artifact.manifest.index_entry_count
    );
    Ok(())
}

fn validate_cmd(args: &[String]) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&[
        "--format",
        "--geometry-column",
        "--exact",
        "--json",
        "--strict",
        "--dims",
        "--nulls",
        "--payload",
        "--properties",
        "--antimeridian",
    ])?;
    let input = parsed.required_pos(0, "input")?;
    parsed.no_extra_pos(1)?;
    let kind = source_kind(input, parsed.option("--format")?)?;
    let payload = parse_payload(
        parsed.option("--payload")?.as_deref().unwrap_or("row-wkb"),
        parsed.option("--properties")?,
    )?;
    let selector = geometry_selector(parsed.option("--geometry-column")?);
    let dims = parse_dims(parsed.option("--dims")?.as_deref().unwrap_or("auto"))?;
    let nulls = parse_nulls(parsed.option("--nulls")?.as_deref().unwrap_or("skip"))?;
    let envelope = parse_antimeridian(parsed.option("--antimeridian")?)?;
    match kind {
        SourceKind::Parquet => {
            let mut dataset = open(File::open(input)?)?;
            let report = dataset.validate(ValidateRequest {
                selector,
                exact: parsed.flag("--exact"),
                dims,
                nulls,
                envelope,
                payload,
            })?;
            if parsed.flag("--json") {
                serde_json::to_writer_pretty(std::io::stdout(), &report)?;
                println!();
            } else {
                print_validation(&report);
            }
            let has_warning = report
                .issues
                .iter()
                .any(|issue| issue.severity == ValidationSeverity::Warning);
            let failed = !report.ok || (parsed.flag("--strict") && has_warning);
            Ok(if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            })
        }
        SourceKind::FlatGeobuf | SourceKind::GeoJson => validate_source(
            kind,
            input,
            ScanRequest {
                selector,
                dims,
                nulls,
                envelope,
                payload,
            },
            parsed.flag("--json"),
        ),
    }
}

fn query_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&[
        "--format",
        "--bbox",
        "--radius",
        "--exact",
        "--predicate",
        "--treat-nonplanar-as-planar",
        "--geometry",
        "--properties",
        "--order",
        "--duplicates",
        "--json",
        "--ndjson",
        "--allow-source-mismatch",
    ])?;
    let source = parsed.required_pos(0, "source")?;
    let index_path = parsed.required_pos(1, "index.psi")?;
    parsed.no_extra_pos(2)?;
    let kind = source_kind(source, parsed.option("--format")?)?;

    if parsed.flag("--json") && parsed.flag("--ndjson") {
        return Err("--json and --ndjson are mutually exclusive".into());
    }

    let bytes = std::fs::read(index_path)?;
    let artifact = open_geo_index(SliceReader::new(bytes))?;
    let manifest = artifact.manifest().clone();

    let features = match artifact {
        GeoArtifactIndex::D2(index) => query_cmd_2d(&parsed, source, kind, &index)?,
        GeoArtifactIndex::D3(index) => query_cmd_3d(&parsed, &index)?,
    };

    query_cmd_finish(&parsed, source, kind, &manifest, features)
}

/// 2D query path: `--bbox` (4 numbers) or `--radius`, with optional `--exact`
/// planar/spherical filtering.
fn query_cmd_2d<R: RangeReader>(
    parsed: &Parsed<'_>,
    source: &str,
    kind: SourceKind,
    index: &GeoArtifactIndex2D<R>,
) -> Result<Vec<FeatureRef>, Box<dyn std::error::Error>> {
    let bbox = parsed.option("--bbox")?;
    let radius = parsed.option("--radius")?;
    let query = match (bbox, radius) {
        (Some(_), Some(_)) => return Err("--bbox and --radius are mutually exclusive".into()),
        (Some(value), None) => GeoQuery2D::box2d(parse_bbox(&value)?),
        (None, Some(value)) => {
            let (lon, lat, radius_metres) = parse_radius(&value)?;
            GeoQuery2D::spherical_radius(lon, lat, radius_metres)
        }
        (None, None) => return Err("--bbox or --radius is required".into()),
    };
    let radius_query = matches!(query, GeoQuery2D::SphericalRadius { .. });
    let exact = parsed.flag("--exact") || radius_query;
    let predicate = parsed.option("--predicate")?;
    let treat_nonplanar = parsed.flag("--treat-nonplanar-as-planar");
    if !exact && (predicate.is_some() || treat_nonplanar) {
        return Err("--predicate and --treat-nonplanar-as-planar require --exact".into());
    }
    if radius_query && treat_nonplanar {
        return Err("--treat-nonplanar-as-planar cannot be used with --radius".into());
    }

    let manifest = index.manifest().clone();
    let features = index.search_features(query.clone())?;
    if !exact {
        return Ok(features);
    }

    let predicate = parse_spatial_predicate(predicate.as_deref().unwrap_or("intersects"))?;
    let non_planar = if treat_nonplanar {
        NonPlanarExactPolicy::TreatAsPlanar
    } else {
        NonPlanarExactPolicy::Reject
    };
    if matches!(kind, SourceKind::Parquet) {
        let expected_source_fingerprint = (!parsed.flag("--allow-source-mismatch"))
            .then_some(manifest.source_fingerprint.clone());
        let mut dataset = open(File::open(source)?)?;
        return Ok(dataset.filter_features(FeatureFilterRequest {
            features,
            selector: GeometrySelector::Name(manifest.selected_column.clone()),
            query,
            predicate,
            non_planar,
            expected_source_fingerprint,
        })?);
    }

    let hits = index.search_hits(query.clone())?;
    Ok(index
        .filter_hits(
            hits,
            query,
            predicate,
            if treat_nonplanar {
                NonPlanarExactPolicy::TreatAsPlanar
            } else {
                NonPlanarExactPolicy::Reject
            },
        )?
        .into_iter()
        .map(|hit| hit.feature)
        .collect())
}

/// 3D query path: `--bbox` only (6 numbers). `--radius`, `--exact`,
/// `--predicate`, and `--treat-nonplanar-as-planar` are 2D-only concepts and
/// are rejected here with an explanatory error rather than dispatched.
fn query_cmd_3d<R: RangeReader>(
    parsed: &Parsed<'_>,
    index: &GeoArtifactIndex3D<R>,
) -> Result<Vec<FeatureRef>, Box<dyn std::error::Error>> {
    if parsed.option("--radius")?.is_some() {
        return Err("--radius is a 2D lon/lat query; this is a 3D index".into());
    }
    if parsed.flag("--exact") {
        return Err(
            "--exact is not supported for a 3D index: exact source-geometry filtering is \
             implemented only for 2D (the planar predicate stack is 2D-only). A 3D query returns \
             a bounding-box (envelope) candidate set, which for non-point geometry -- or any f32 \
             index -- is a superset, not the exact hit set"
                .into(),
        );
    }
    if parsed.option("--predicate")?.is_some() {
        return Err(
            "--predicate is a 2D-only option: it selects the predicate for the exact filter, \
             which is not implemented for 3D indexes"
                .into(),
        );
    }
    if parsed.flag("--treat-nonplanar-as-planar") {
        return Err(
            "--treat-nonplanar-as-planar is a 2D-only option: it tunes the exact filter, which \
             is not implemented for 3D indexes"
                .into(),
        );
    }

    let Some(value) = parsed.option("--bbox")? else {
        return Err("--bbox is required for a 3D index (--radius is 2D-only)".into());
    };
    let bbox3d = parse_bbox3d(&value)?;
    Ok(index.search_features(bbox3d)?)
}

/// Shared tail: read projected rows for `features` back from `source` and print them.
fn query_cmd_finish(
    parsed: &Parsed<'_>,
    source: &str,
    kind: SourceKind,
    manifest: &GeoArtifactManifest,
    features: Vec<FeatureRef>,
) -> Result<(), Box<dyn std::error::Error>> {
    let geometry = parse_geometry_read(parsed.option("--geometry")?.as_deref().unwrap_or("none"))?;
    let properties = parse_properties(parsed.option("--properties")?.as_deref().unwrap_or("all"))?;
    let order = parse_feature_order(parsed.option("--order")?.as_deref().unwrap_or("source"))?;
    let duplicates =
        parse_duplicates(parsed.option("--duplicates")?.as_deref().unwrap_or("dedup"))?;
    let expected_source_fingerprint =
        (!parsed.flag("--allow-source-mismatch")).then_some(manifest.source_fingerprint.clone());

    let request = FeatureReadRequest {
        features,
        selector: GeometrySelector::Name(manifest.selected_column.clone()),
        properties,
        geometry,
        order,
        duplicates,
        expected_source_fingerprint,
    };
    match kind {
        SourceKind::Parquet => {
            let mut dataset = open(File::open(source)?)?;
            let rows = dataset.read_features(request)?;
            print_query_rows(&rows, parsed.flag("--json"))?;
        }
        SourceKind::FlatGeobuf | SourceKind::GeoJson => {
            let records = read_feature_records(kind, source, request)?;
            print_feature_records(&records, parsed.flag("--json"))?;
        }
    }
    Ok(())
}

fn print_discovery(discovery: &GeoDiscovery) {
    println!("rows: {}", discovery.num_rows);
    if let Some(version) = &discovery.file_metadata.geoparquet_version {
        println!("geoparquet: {version}");
    }
    if let Some(primary) = &discovery.file_metadata.geoparquet_primary_column {
        println!("primary: {primary}");
    }
    println!(
        "selection: {}",
        selection_label(&discovery.default_selection)
    );
    println!(
        "{:<24} {:<18} {:<22} {:<8} {:<8} {:<5} {:<5}",
        "column", "source", "encoding", "dims", "bounds", "index", "wkb"
    );
    for column in &discovery.columns {
        println!(
            "{:<24} {:<18} {:<22} {:<8} {:<8} {:<5} {:<5}",
            column.name,
            format!("{:?}", column.source),
            column.encoding.to_string(),
            column.coordinate_dims.to_string(),
            yes_no(column.extent.is_some()),
            yes_no(column.capabilities.can_build_index),
            yes_no(column.capabilities.can_emit_row_wkb),
        );
    }
}

fn selection_label(selection: &packed_spatial_index_geo::SelectionStatus) -> String {
    match selection {
        packed_spatial_index_geo::SelectionStatus::Selected { column, reason } => {
            format!("selected `{column}` ({reason:?})")
        }
        packed_spatial_index_geo::SelectionStatus::Ambiguous { columns } => {
            format!("ambiguous {columns:?}")
        }
        packed_spatial_index_geo::SelectionStatus::Missing { column } => {
            format!("missing `{column}`")
        }
        packed_spatial_index_geo::SelectionStatus::None => "none".to_string(),
    }
}

fn print_profile(profile: &GeometryProfile) {
    println!("rows: {}", profile.num_rows);
    println!("column: {}", profile.column);
    println!("source: {:?}", profile.source);
    println!("encoding: {}", profile.encoding);
    println!("dims: {}", profile.coordinate_dims);
    println!("edges: {:?}", profile.edges);
    println!("crs: {:?}", profile.crs);
    if let Some(extent) = &profile.extent {
        println!("extent: {:?}", extent.values);
    }
}

fn print_validation(report: &ValidationReport) {
    println!("rows: {}", report.discovery.num_rows);
    println!("selection: {}", selection_label(&report.selected));
    println!("status: {}", if report.ok { "ok" } else { "error" });
    if let Some(profile) = &report.profile {
        println!("column: {}", profile.column);
        println!("source: {:?}", profile.source);
        println!("encoding: {}", profile.encoding);
        println!("dims: {}", profile.coordinate_dims);
        println!("edges: {:?}", profile.edges);
    }
    if !report.native_stats.is_empty() {
        println!("native geospatial stats:");
        println!(
            "{:<24} {:>8} {:>8} {:>8} {:<8} {:<5}",
            "column", "groups", "bbox", "types", "dims", "am"
        );
        for stats in &report.native_stats {
            println!(
                "{:<24} {:>8} {:>8} {:>8} {:<8} {:<5}",
                stats.column,
                stats.row_group_count,
                stats.groups_with_bbox,
                stats.groups_with_types,
                stats.inferred_dims.to_string(),
                yes_no(stats.has_antimeridian_wrap),
            );
        }
    }
    if report.issues.is_empty() {
        println!("issues: none");
    } else {
        println!("issues:");
        println!("{:<8} {:<28} {:<24} message", "severity", "code", "column");
        for issue in &report.issues {
            println!(
                "{:<8} {:<28} {:<24} {}",
                format!("{:?}", issue.severity).to_ascii_lowercase(),
                format!("{:?}", issue.code),
                issue.column.as_deref().unwrap_or("-"),
                issue.message
            );
        }
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn geometry_selector(name: Option<String>) -> GeometrySelector {
    name.map(GeometrySelector::Name)
        .unwrap_or(GeometrySelector::Default)
}

fn parse_dims(value: &str) -> Result<IndexDimsRequest, Box<dyn std::error::Error>> {
    match value {
        "auto" => Ok(IndexDimsRequest::Auto),
        "2d" | "2D" => Ok(IndexDimsRequest::D2),
        "3d" | "3D" => Ok(IndexDimsRequest::D3),
        _ => Err(format!("invalid --dims `{value}`").into()),
    }
}

fn parse_nulls(value: &str) -> Result<NullPolicy, Box<dyn std::error::Error>> {
    match value {
        "error" => Ok(NullPolicy::Error),
        "skip" => Ok(NullPolicy::Skip),
        _ => Err(format!("invalid --nulls `{value}`").into()),
    }
}

fn parse_precision(value: &str) -> Result<StoragePrecision, Box<dyn std::error::Error>> {
    match value {
        "f64" => Ok(StoragePrecision::F64),
        "f32" => Ok(StoragePrecision::F32),
        _ => Err(format!("invalid --precision `{value}`").into()),
    }
}

fn parse_geometry_read(value: &str) -> Result<GeometryReadMode, Box<dyn std::error::Error>> {
    match value {
        "none" => Ok(GeometryReadMode::Omit),
        "wkb" => Ok(GeometryReadMode::Wkb),
        _ => Err(format!("invalid --geometry `{value}`").into()),
    }
}

fn parse_feature_order(value: &str) -> Result<FeatureReadOrder, Box<dyn std::error::Error>> {
    match value {
        "source" => Ok(FeatureReadOrder::SourceOrder),
        "hit" => Ok(FeatureReadOrder::RequestOrder),
        _ => Err(format!("invalid --order `{value}`").into()),
    }
}

fn parse_duplicates(value: &str) -> Result<DuplicateFeatureRows, Box<dyn std::error::Error>> {
    match value {
        "dedup" => Ok(DuplicateFeatureRows::DedupRows),
        "parts" => Ok(DuplicateFeatureRows::KeepParts),
        _ => Err(format!("invalid --duplicates `{value}`").into()),
    }
}

fn parse_spatial_predicate(value: &str) -> Result<SpatialPredicate, Box<dyn std::error::Error>> {
    match value {
        "intersects" => Ok(SpatialPredicate::Intersects),
        _ => Err(format!("invalid --predicate `{value}`").into()),
    }
}

fn parse_radius(value: &str) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let parts = value
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        return Err("--radius expects three comma-separated numbers".into());
    }
    Ok((parts[0], parts[1], parts[2]))
}

fn parse_bbox(value: &str) -> Result<Box2D, Box<dyn std::error::Error>> {
    let parts = value
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 4 {
        return Err("--bbox expects four comma-separated numbers".into());
    }
    Ok(Box2D::new(parts[0], parts[1], parts[2], parts[3]))
}

fn parse_bbox3d(value: &str) -> Result<Box3D, Box<dyn std::error::Error>> {
    let parts = value
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 6 {
        return Err(
            "--bbox against a 3D index expects six comma-separated numbers \
             (xmin,ymin,zmin,xmax,ymax,zmax)"
                .into(),
        );
    }
    Ok(Box3D::new(
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
    ))
}

fn parse_antimeridian(value: Option<String>) -> Result<EnvelopePolicy, Box<dyn std::error::Error>> {
    let Some(value) = value else {
        return Ok(EnvelopePolicy::Planar);
    };
    let antimeridian = match value.as_str() {
        "reject" => AntimeridianPolicy::Reject,
        "split" => AntimeridianPolicy::Split,
        "world" => AntimeridianPolicy::ExpandToWorld,
        _ => return Err(format!("invalid --antimeridian `{value}`").into()),
    };
    Ok(EnvelopePolicy::Geographic { antimeridian })
}

fn parse_payload(
    value: &str,
    properties: Option<String>,
) -> Result<PayloadPlan, Box<dyn std::error::Error>> {
    match value {
        "none" => Ok(PayloadPlan::None),
        "row-ref" => Ok(PayloadPlan::RowRef),
        "row-wkb" => Ok(PayloadPlan::RowWkb),
        "feature-json" => Ok(PayloadPlan::FeatureJson {
            properties: parse_properties(properties.as_deref().unwrap_or("all"))?,
        }),
        _ => Err(format!("invalid --payload `{value}`").into()),
    }
    .and_then(|payload| match (&payload, properties) {
        (PayloadPlan::FeatureJson { .. }, _) | (_, None) => Ok(payload),
        _ => Err("--properties can only be used with --payload feature-json".into()),
    })
}

fn parse_properties(value: &str) -> Result<PropertyProjection, Box<dyn std::error::Error>> {
    match value {
        "none" => Ok(PropertyProjection::None),
        "all" => Ok(PropertyProjection::AllNonGeometry),
        value if value.starts_with("include:") => Ok(PropertyProjection::Include(split_names(
            value.trim_start_matches("include:"),
        ))),
        value if value.starts_with("exclude:") => Ok(PropertyProjection::Exclude(split_names(
            value.trim_start_matches("exclude:"),
        ))),
        _ => Err(format!("invalid --properties `{value}`").into()),
    }
}

fn split_names(value: &str) -> Vec<String> {
    value
        .split(',')
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn read_feature_records(
    kind: SourceKind,
    source: &str,
    request: FeatureReadRequest,
) -> Result<Vec<FeatureRecord>, Box<dyn std::error::Error>> {
    match kind {
        SourceKind::Parquet => Err("Parquet read-back uses FeatureRows, not FeatureRecord".into()),
        SourceKind::FlatGeobuf => {
            #[cfg(feature = "flatgeobuf")]
            {
                let mut dataset = open_flatgeobuf(File::open(source)?)?;
                Ok(dataset.read_features(request)?)
            }
            #[cfg(not(feature = "flatgeobuf"))]
            {
                let _ = (source, request);
                Err("this gp2psindex build was compiled without FlatGeobuf support".into())
            }
        }
        SourceKind::GeoJson => {
            #[cfg(feature = "geojson")]
            {
                let dataset = open_geojson(File::open(source)?)?;
                Ok(dataset.read_features(request)?)
            }
            #[cfg(not(feature = "geojson"))]
            {
                let _ = (source, request);
                Err("this gp2psindex build was compiled without GeoJSON support".into())
            }
        }
    }
}

fn print_feature_records(
    records: &[FeatureRecord],
    as_json_array: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let values = records
        .iter()
        .map(feature_record_value)
        .collect::<Result<Vec<_>, _>>()?;
    if as_json_array {
        serde_json::to_writer_pretty(std::io::stdout(), &values)?;
        println!();
    } else {
        for value in values {
            serde_json::to_writer(std::io::stdout(), &value)?;
            println!();
        }
    }
    Ok(())
}

fn feature_record_value(
    record: &FeatureRecord,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut feature = serde_json::Map::new();
    feature.insert(
        "type".to_string(),
        serde_json::Value::String("Feature".to_string()),
    );
    if let Some(id) = &record.feature.feature_id {
        feature.insert("id".to_string(), serde_json::Value::String(id.clone()));
    }
    feature.insert(
        "feature_ref".to_string(),
        serde_json::to_value(&record.feature)?,
    );
    feature.insert(
        "geometry".to_string(),
        record
            .geometry_json
            .clone()
            .unwrap_or(serde_json::Value::Null),
    );
    feature.insert("properties".to_string(), record.properties.clone());
    if let Some(wkb) = &record.geometry_wkb {
        feature.insert(
            "geometry_wkb".to_string(),
            serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(wkb)),
        );
    }
    Ok(serde_json::Value::Object(feature))
}

fn print_query_rows(
    rows: &FeatureRows,
    as_json_array: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let values = query_row_values(rows)?;
    if as_json_array {
        serde_json::to_writer_pretty(std::io::stdout(), &values)?;
        println!();
    } else {
        for value in values {
            serde_json::to_writer(std::io::stdout(), &value)?;
            println!();
        }
    }
    Ok(())
}

fn query_row_values(
    rows: &FeatureRows,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    (0..rows.features.len())
        .map(|row| {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "feature".to_string(),
                serde_json::to_value(&rows.features[row])?,
            );
            obj.insert("properties".to_string(), row_properties(&rows.batch, row)?);
            if let Some(wkb) = geometry_wkb_at(&rows.batch, row)? {
                obj.insert(
                    "geometry_wkb".to_string(),
                    serde_json::Value::String(
                        base64::engine::general_purpose::STANDARD.encode(wkb),
                    ),
                );
            }
            Ok(serde_json::Value::Object(obj))
        })
        .collect()
}

fn row_properties(
    batch: &RecordBatch,
    row: usize,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        if field.name() == "geometry_wkb" {
            continue;
        }
        fields.push(field.as_ref().clone());
        arrays.push(batch.column(idx).slice(row, 1));
    }
    if fields.is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    let projected = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
    let mut buf = Vec::new();
    let mut writer = LineDelimitedWriter::new(&mut buf);
    writer.write(&projected)?;
    writer.finish()?;
    Ok(serde_json::from_slice(trim_ascii(&buf))?)
}

fn geometry_wkb_at(
    batch: &RecordBatch,
    row: usize,
) -> Result<Option<&[u8]>, Box<dyn std::error::Error>> {
    let Some(array) = batch.column_by_name("geometry_wkb") else {
        return Ok(None);
    };
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(binary) = array.as_any().downcast_ref::<BinaryArray>() {
        Ok(Some(binary.value(row)))
    } else if let Some(binary) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        Ok(Some(binary.value(row)))
    } else if let Some(binary) = array.as_any().downcast_ref::<BinaryViewArray>() {
        Ok(Some(binary.value(row)))
    } else {
        Err("geometry_wkb column is not binary".into())
    }
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = bytes.len();
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &bytes[start..end]
}

struct Parsed<'a> {
    args: &'a [String],
}

impl<'a> Parsed<'a> {
    fn new(args: &'a [String]) -> Self {
        Self { args }
    }

    fn flag(&self, flag: &str) -> bool {
        self.args.iter().any(|arg| arg == flag)
    }

    fn no_unknown_flags(&self, known: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        for arg in self.args {
            if !arg.starts_with("--") {
                continue;
            }
            let flag = arg.split_once('=').map_or(arg.as_str(), |(flag, _)| flag);
            if !known.contains(&flag) {
                return Err(format!("unknown flag `{flag}`").into());
            }
        }
        Ok(())
    }

    fn option(&self, flag: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let equals = format!("{flag}=");
        for (idx, arg) in self.args.iter().enumerate() {
            if let Some(value) = arg.strip_prefix(&equals) {
                return Ok(Some(value.to_string()));
            }
            if arg == flag {
                let Some(value) = self.args.get(idx + 1) else {
                    return Err(format!("{flag} needs a value").into());
                };
                if value.starts_with("--") {
                    return Err(format!("{flag} needs a value").into());
                }
                return Ok(Some(value.clone()));
            }
        }
        Ok(None)
    }

    fn positionals(&self) -> Vec<&str> {
        let mut out = Vec::new();
        let mut skip = false;
        for arg in self.args {
            if skip {
                skip = false;
                continue;
            }
            if arg.starts_with("--") {
                if !arg.contains('=') && option_takes_value(arg) {
                    skip = true;
                }
                continue;
            }
            out.push(arg.as_str());
        }
        out
    }

    fn required_pos(&self, index: usize, name: &str) -> Result<&str, Box<dyn std::error::Error>> {
        self.positionals()
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing {name}").into())
    }

    fn no_extra_pos(&self, max: usize) -> Result<(), Box<dyn std::error::Error>> {
        let count = self.positionals().len();
        if count > max {
            return Err("too many positional arguments".into());
        }
        Ok(())
    }
}

fn option_takes_value(arg: &str) -> bool {
    matches!(
        arg,
        "--geometry-column"
            | "--dims"
            | "--format"
            | "--precision"
            | "--nulls"
            | "--payload"
            | "--properties"
            | "--antimeridian"
            | "--bbox"
            | "--geometry"
            | "--order"
            | "--duplicates"
            | "--predicate"
            | "--radius"
    )
}

#[allow(dead_code)]
fn map_geo_error(err: GeoError) -> Box<dyn std::error::Error> {
    Box::new(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_positional_convert_is_not_a_command() {
        let err = run(vec!["input.parquet".to_string()]).unwrap_err();
        assert!(err.to_string().contains("unknown command"));
    }

    #[test]
    fn parses_build_flags() {
        let args = vec![
            "in.parquet".to_string(),
            "out.psi".to_string(),
            "--geometry-column=geom".to_string(),
            "--dims".to_string(),
            "3d".to_string(),
            "--payload".to_string(),
            "feature-json".to_string(),
            "--properties".to_string(),
            "include:name,pop".to_string(),
            "--antimeridian".to_string(),
            "split".to_string(),
        ];
        let parsed = Parsed::new(&args);
        assert_eq!(parsed.required_pos(0, "input").unwrap(), "in.parquet");
        assert_eq!(parsed.required_pos(1, "output").unwrap(), "out.psi");
        assert_eq!(
            parsed.option("--geometry-column").unwrap().as_deref(),
            Some("geom")
        );
        assert!(matches!(
            parse_dims(parsed.option("--dims").unwrap().as_deref().unwrap()).unwrap(),
            IndexDimsRequest::D3
        ));
    }

    #[test]
    fn parse_bbox3d_accepts_six_numbers() {
        let bbox = parse_bbox3d("1,2,3,4,5,6").unwrap();
        assert_eq!(bbox, Box3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
    }

    #[test]
    fn parse_bbox3d_rejects_wrong_count() {
        let err = parse_bbox3d("1,2,3,4").unwrap_err();
        assert!(err.to_string().contains("six comma-separated numbers"));

        let err = parse_bbox3d("1,2,3,4,5,6,7").unwrap_err();
        assert!(err.to_string().contains("six comma-separated numbers"));
    }
}
