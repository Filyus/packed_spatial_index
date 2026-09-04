//! Command-line converter and inspector for geospatial Parquet inputs.

use std::fs::File;
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use arrow_json::LineDelimitedWriter;
use base64::Engine as _;
#[cfg(feature = "flatgeobuf")]
use packed_spatial_index_geo::geo_types::{Coord, LineString, MultiPolygon, Polygon};
use packed_spatial_index_geo::open_flatgeobuf;
use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, Box3D, ConvertRequest, CoordinateDims, DuplicateFeatureRows,
    EnvelopePolicy, FeatureFilterRequest, FeatureReadOrder, FeatureReadRequest, FeatureRecord,
    FeatureRef, FeatureRows, Frustum3D, GeoArtifact, GeoArtifactIndex, GeoArtifactIndex2D,
    GeoArtifactIndex3D, GeoArtifactManifest, GeoDiscovery, GeoError, GeoQuery2D, GeoQuery3D,
    GeometryProfile, GeometryReadMode, GeometryScan, GeometrySelector, Index2D, Index3D,
    IndexDimsRequest, InspectRequest, NonPlanarExactPolicy, NullPolicy, PayloadPlan,
    PrefixIndexPolicy, PropertyProjection, RangeReader, ScanRequest, SliceReader, SpatialPredicate,
    StoragePrecision, ValidateRequest, ValidationReport, ValidationSeverity, open_geo_index,
    open_geoparquet,
};
#[cfg(feature = "geojson")]
use packed_spatial_index_geo::{convert_geojson_stream, open_geojson};

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
      [--prefix-index auto|on|off]
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
  gp2psindex join <a.psi> <b.psi> --within N
      [--count]
        (print how many pairs match, then stop)
      (writes one NDJSON line {\"a\":i,\"b\":j} per pair to stdout, streamed.
       Pass the same path twice for a self-join: every unordered pair of
       distinct items once, an item never paired with itself.
       Distances are box-to-box Euclidean in the artifacts' coordinate
       units, zero when boxes overlap and inclusive at the bound, so
       --within 0 is the plain overlap join. Both artifacts must be the
       same dimensionality)
  gp2psindex anti-join <a.psi> <b.psi> --within N
      [--count]
        (print how many items have no partner, then stop)
      (the complement of join: one NDJSON line {\"a\":i} per item of a.psi
       with NO item of b.psi within the bound. The two paths must differ --
       against itself every item is at distance zero from itself; the
       question meant there is what `components` answers)
  gp2psindex components <a.psi> --within N
      [--count]
        (print the number of components, then stop)
      (connected components of the within-graph: one NDJSON line
       {\"item\":i,\"label\":l} per item, the label being the smallest item
       id in its component. An isolated item is its own label. Labels
       identify components, not clusters: proximity is not transitive)
  gp2psindex query <source> <index.psi>
      [--format parquet|flatgeobuf|geojson]
      (--bbox xmin,ymin,xmax,ymax | --radius lon,lat,metres
       | --polygon '[[[[x,y],...],...],...]')
        (--polygon takes GeoJSON MultiPolygon coordinates: ring 0 of each
         polygon is its exterior, the rest are holes. The polygon drives the
         index traversal itself, pruning subtrees outside it, so it needs no
         payload and --count works over it; --exact still refines the
         surviving entries against the source geometry)
      [--exact]
      [--predicate intersects]
      [--treat-nonplanar-as-planar]
        (vouches for the stored coordinates against the column's declared
         edge model: planar XY for --bbox, lon/lat degrees for --radius)
      [--geometry none|wkb]
      [--properties none|all|include:a,b|exclude:a,b]
      [--order source|match]
      [--duplicates dedup|parts]
      [--count]
        (print how many index entries match, then stop)
      [--limit n] [--offset n]
        (read back only one page of matches, in index entry order)
      [--json|--ndjson]
      [--allow-source-mismatch]
      (against a 3D index: --bbox takes xmin,ymin,zmin,xmax,ymax,zmax, or
       --frustum takes 24 numbers -- six inward-pointing planes as a,b,c,d;
       --radius/--polygon/--exact/--predicate/--treat-nonplanar-as-planar
       are 2D-only.
       --count and --limit/--offset describe the index's own match set, so
       they are refused together with --exact, which narrows it afterwards)";

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
        "join" => join_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        "anti-join" => anti_join_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        "components" => components_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
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
            let mut dataset = open_geoparquet(File::open(input)?)?;
            Ok(dataset.inspect(InspectRequest {
                selector,
                exact: false,
            })?)
        }
        SourceKind::FlatGeobuf => {
            check_single_geometry_selector(&selector, "FlatGeobuf")?;
            #[cfg(feature = "flatgeobuf")]
            {
                Ok(open_flatgeobuf(File::open(input)?)?.profile()?)
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
                Ok(open_geojson(File::open(input)?)?.profile()?)
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
            let mut dataset = open_geoparquet(File::open(input)?)?;
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
                let out_len = out.len();
                match convert_geojson_stream(File::open(input)?, request.clone(), out) {
                    Ok(artifact) => Ok(artifact),
                    Err(err) if is_geojson_stream_shape_error(&err) => {
                        out.truncate(out_len);
                        let mut dataset = open_geojson(File::open(input)?)?;
                        Ok(dataset.convert_into(request, out)?)
                    }
                    Err(err) => Err(Box::new(err)),
                }
            }
            #[cfg(not(feature = "geojson"))]
            {
                let _ = (input, request, out);
                Err("this gp2psindex build was compiled without GeoJSON support".into())
            }
        }
    }
}

#[cfg(feature = "geojson")]
fn is_geojson_stream_shape_error(err: &GeoError) -> bool {
    matches!(
        err,
        GeoError::GeoJson(message)
            if message.contains("document type is not `FeatureCollection`")
                || message.contains("FeatureCollection has no `features` array")
    )
}

fn scan_source(
    kind: SourceKind,
    input: &str,
    request: ScanRequest,
) -> Result<GeometryScan, Box<dyn std::error::Error>> {
    match kind {
        SourceKind::Parquet => {
            let mut dataset = open_geoparquet(File::open(input)?)?;
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
            let dataset = open_geoparquet(File::open(input)?)?;
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
            let mut dataset = open_geoparquet(File::open(input)?)?;
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
        "--prefix-index",
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
        prefix_index: parse_prefix_index(parsed.option("--prefix-index")?.as_deref())?,
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
            let mut dataset = open_geoparquet(File::open(input)?)?;
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

/// Owned core index over a whole artifact, dimension-dispatched — the shape the
/// distance join needs, and the same one the server's `JoinIndex` cache holds.
enum JoinIndex {
    D2(Index2D),
    D3(Index3D),
}

fn join_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&["--within", "--count"])?;
    let a_path = parsed.required_pos(0, "a.psi")?;
    let b_path = parsed.required_pos(1, "b.psi")?;
    parsed.no_extra_pos(2)?;
    let max_distance = parse_max_distance(parsed.option("--within")?.as_deref())?;

    let a_bytes = std::fs::read(a_path)?;
    // The same path twice is a self-join, and reading the file again would only
    // buy a second copy of identical bytes.
    let same = std::fs::canonicalize(a_path).ok() == std::fs::canonicalize(b_path).ok()
        || a_path == b_path;
    let b_bytes = if same {
        None
    } else {
        Some(std::fs::read(b_path)?)
    };

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let count = join_pairs(
        &a_bytes,
        b_bytes.as_deref(),
        max_distance,
        parsed.flag("--count"),
        &mut out,
    )?;
    if parsed.flag("--count") {
        writeln!(out, "{count}")?;
    }
    out.flush()?;
    Ok(())
}

/// Run the distance join and stream `{"a":i,"b":j}` lines into `out`, one per
/// pair, returning how many pairs matched.
///
/// The pairs are written from inside the join's visitor: the join is
/// output-bound (millions of pairs at a generous `max_distance`), so materializing
/// the pair vector first would cost more memory than the indexes do. `b` is
/// `None` for a self-join. Pair order is traversal order and is not an API.
fn join_pairs<W: std::io::Write>(
    a_bytes: &[u8],
    b_bytes: Option<&[u8]>,
    max_distance: f64,
    count_only: bool,
    out: &mut W,
) -> Result<usize, Box<dyn std::error::Error>> {
    let a = load_join_index(a_bytes, "a.psi")?;
    let mut total = 0usize;
    let mut emit = |i: usize, j: usize| -> ControlFlow<std::io::Error> {
        total += 1;
        if count_only {
            return ControlFlow::Continue(());
        }
        match writeln!(out, "{{\"a\":{i},\"b\":{j}}}") {
            Ok(()) => ControlFlow::Continue(()),
            Err(err) => ControlFlow::Break(err),
        }
    };

    let flow = match b_bytes {
        None => match &a {
            JoinIndex::D2(a) => a.self_join_within_with(max_distance, &mut emit),
            JoinIndex::D3(a) => a.self_join_within_with(max_distance, &mut emit),
        },
        Some(bytes) => {
            let b = load_join_index(bytes, "b.psi")?;
            match (&a, &b) {
                (JoinIndex::D2(a), JoinIndex::D2(b)) => {
                    a.join_within_with(b, max_distance, &mut emit)
                }
                (JoinIndex::D3(a), JoinIndex::D3(b)) => {
                    a.join_within_with(b, max_distance, &mut emit)
                }
                _ => {
                    return Err(
                        "a.psi and b.psi are different dimensions; a distance join needs both in 2D or both in 3D"
                            .into(),
                    );
                }
            }
        }
    };
    if let ControlFlow::Break(err) = flow {
        return Err(err.into());
    }
    Ok(total)
}

fn anti_join_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&["--within", "--count"])?;
    let a_path = parsed.required_pos(0, "a.psi")?;
    let b_path = parsed.required_pos(1, "b.psi")?;
    parsed.no_extra_pos(2)?;
    let max_distance = parse_max_distance(parsed.option("--within")?.as_deref())?;

    // Against itself every item is at distance zero from itself, so the literal
    // answer is always empty. The question people mean there -- which items have
    // no *other* item nearby -- is what `components` answers (an isolated item is
    // its own label). Refuse rather than quietly answer a different question,
    // exactly as the server's /anti-join does.
    let same = std::fs::canonicalize(a_path).ok() == std::fs::canonicalize(b_path).ok()
        || a_path == b_path;
    if same {
        return Err(
            "anti-join needs two different artifacts: every item is at distance zero from \
             itself, so the answer against the same file is always empty; for items with no \
             *other* item nearby, run `components` and look for labels that occur once"
                .into(),
        );
    }

    let a_bytes = std::fs::read(a_path)?;
    let b_bytes = std::fs::read(b_path)?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let count = anti_join_items(
        &a_bytes,
        &b_bytes,
        max_distance,
        parsed.flag("--count"),
        &mut out,
    )?;
    if parsed.flag("--count") {
        writeln!(out, "{count}")?;
    }
    out.flush()?;
    Ok(())
}

/// Run the anti-join and stream `{"a":i}` lines into `out`, one per item of
/// `a` with no item of `b` within `max_distance`, returning how many there were.
fn anti_join_items<W: std::io::Write>(
    a_bytes: &[u8],
    b_bytes: &[u8],
    max_distance: f64,
    count_only: bool,
    out: &mut W,
) -> Result<usize, Box<dyn std::error::Error>> {
    let a = load_join_index(a_bytes, "a.psi")?;
    let b = load_join_index(b_bytes, "b.psi")?;
    let mut total = 0usize;
    let mut emit = |i: usize| -> ControlFlow<std::io::Error> {
        total += 1;
        if count_only {
            return ControlFlow::Continue(());
        }
        match writeln!(out, "{{\"a\":{i}}}") {
            Ok(()) => ControlFlow::Continue(()),
            Err(err) => ControlFlow::Break(err),
        }
    };
    let flow = match (&a, &b) {
        (JoinIndex::D2(a), JoinIndex::D2(b)) => a.anti_join_within_with(b, max_distance, &mut emit),
        (JoinIndex::D3(a), JoinIndex::D3(b)) => a.anti_join_within_with(b, max_distance, &mut emit),
        _ => {
            return Err(
                "a.psi and b.psi are different dimensions; an anti-join needs both in 2D or both in 3D"
                    .into(),
            );
        }
    };
    if let ControlFlow::Break(err) = flow {
        return Err(err.into());
    }
    Ok(total)
}

fn components_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&["--within", "--count"])?;
    let path = parsed.required_pos(0, "a.psi")?;
    parsed.no_extra_pos(1)?;
    let max_distance = parse_max_distance(parsed.option("--within")?.as_deref())?;

    let bytes = std::fs::read(path)?;
    let labels = component_labels(&bytes, max_distance)?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    if parsed.flag("--count") {
        writeln!(out, "{}", count_components(&labels))?;
    } else {
        for (item, label) in labels.iter().enumerate() {
            writeln!(out, "{{\"item\":{item},\"label\":{label}}}")?;
        }
    }
    out.flush()?;
    Ok(())
}

/// One label per item: the smallest item id in that item's component of the
/// `max_distance`-proximity graph. Unlike the pair stream this is not
/// output-bound -- it is exactly one `usize` per item -- so it is collected.
fn component_labels(
    bytes: &[u8],
    max_distance: f64,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    Ok(match load_join_index(bytes, "a.psi")? {
        JoinIndex::D2(index) => index.self_join_within_components(max_distance),
        JoinIndex::D3(index) => index.self_join_within_components(max_distance),
    })
}

/// A label is the smallest id in its component, so an item is a component's
/// representative exactly when its label is its own id.
fn count_components(labels: &[usize]) -> usize {
    labels
        .iter()
        .enumerate()
        .filter(|(item, label)| item == *label)
        .count()
}

fn load_join_index(bytes: &[u8], what: &str) -> Result<JoinIndex, Box<dyn std::error::Error>> {
    let artifact = open_geo_index(SliceReader::new(bytes.to_vec()))?;
    match artifact.manifest().dims {
        CoordinateDims::Xy | CoordinateDims::Xym => Ok(JoinIndex::D2(Index2D::from_bytes(bytes)?)),
        CoordinateDims::Xyz | CoordinateDims::Xyzm => {
            Ok(JoinIndex::D3(Index3D::from_bytes(bytes)?))
        }
        CoordinateDims::Unknown => Err(format!(
            "{what} has unknown dimensions; a distance join needs a 2D or 3D artifact"
        )
        .into()),
    }
}

/// The distance bound, in the artifacts' coordinate units.
///
/// Required, finite and non-negative — the same contract as the server's
/// `within` query parameter, so the two surfaces reject the same inputs.
fn parse_max_distance(raw: Option<&str>) -> Result<f64, Box<dyn std::error::Error>> {
    let raw = raw
        .ok_or("--within is required: the distance bound in coordinate units, e.g. --within 500")?;
    let max_distance: f64 = raw
        .parse()
        .map_err(|_| format!("`{raw}` is not a number"))?;
    if !max_distance.is_finite() || max_distance < 0.0 {
        return Err(format!("--within must be a finite non-negative number, got `{raw}`").into());
    }
    Ok(max_distance)
}

fn query_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    parsed.no_unknown_flags(&[
        "--format",
        "--bbox",
        "--radius",
        "--polygon",
        "--frustum",
        "--exact",
        "--predicate",
        "--treat-nonplanar-as-planar",
        "--geometry",
        "--properties",
        "--order",
        "--duplicates",
        "--count",
        "--limit",
        "--offset",
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

    if parsed.flag("--count") {
        reject_post_index_narrowing(&parsed, "--count")?;
        if parsed.option("--limit")?.is_some() || parsed.option("--offset")?.is_some() {
            return Err("--count reports the whole match set; drop --limit/--offset".into());
        }
        let count = match artifact {
            GeoArtifactIndex::D2(index) => index.count_entries(query_2d(&parsed)?)?,
            GeoArtifactIndex::D3(index) => index.count_entries(query_3d(&parsed)?)?,
        };
        println!("{count}");
        return Ok(());
    }

    let page = query_page(&parsed)?;
    if page.is_some() {
        reject_post_index_narrowing(&parsed, "--limit/--offset")?;
    }

    let features = match artifact {
        GeoArtifactIndex::D2(index) => query_cmd_2d(&parsed, source, kind, &index, page)?,
        GeoArtifactIndex::D3(index) => query_cmd_3d(&parsed, &index, page)?,
    };

    query_cmd_finish(&parsed, source, kind, &manifest, features)
}

/// Parse `--limit`/`--offset` into a page request.
///
/// `--offset` alone is meaningless without a limit, so it requires one.
fn query_page(parsed: &Parsed<'_>) -> Result<Option<(usize, usize)>, Box<dyn std::error::Error>> {
    let limit = parsed
        .option("--limit")?
        .map(parse_page_number)
        .transpose()?;
    let offset = parsed
        .option("--offset")?
        .map(parse_page_number)
        .transpose()?;
    match (limit, offset) {
        (Some(limit), offset) => Ok(Some((offset.unwrap_or(0), limit))),
        (None, Some(_)) => Err("--offset requires --limit".into()),
        (None, None) => Ok(None),
    }
}

fn parse_page_number(value: String) -> Result<usize, Box<dyn std::error::Error>> {
    value
        .parse::<usize>()
        .map_err(|_| format!("`{value}` is not a non-negative integer").into())
}

/// Counting and paging both work on the index's own match set, which an exact
/// filter narrows afterwards. Reporting a count or a page from before that
/// filter would describe a different set than the one the same command without
/// the flag would print.
fn reject_post_index_narrowing(
    parsed: &Parsed<'_>,
    flag: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if parsed.flag("--exact") || parsed.option("--radius")?.is_some() {
        return Err(format!(
            "{flag} counts index matches, which --exact (and --radius, which is always exact) \
             narrows afterwards; the numbers would not agree with the unfiltered query"
        )
        .into());
    }
    Ok(())
}

/// 2D query path: `--bbox` (4 numbers) or `--radius`, with optional `--exact`
/// planar/spherical filtering.
fn query_cmd_2d<R: RangeReader>(
    parsed: &Parsed<'_>,
    source: &str,
    kind: SourceKind,
    index: &GeoArtifactIndex2D<R>,
    page: Option<(usize, usize)>,
) -> Result<Vec<FeatureRef>, Box<dyn std::error::Error>> {
    let query = query_2d(parsed)?;
    if let Some((offset, limit)) = page {
        let page = index.search_match_headers_page(query, offset, limit)?;
        return Ok(index
            .fetch_matches(&page.headers)?
            .into_iter()
            .map(|m| m.feature)
            .collect());
    }
    let radius_query = matches!(query, GeoQuery2D::SphericalRadius { .. });
    let exact = parsed.flag("--exact") || radius_query;
    let predicate = parsed.option("--predicate")?;
    let treat_nonplanar = parsed.flag("--treat-nonplanar-as-planar");
    if !exact && (predicate.is_some() || treat_nonplanar) {
        return Err("--predicate and --treat-nonplanar-as-planar require --exact".into());
    }

    let manifest = index.manifest().clone();
    let features = index.search_feature_refs(query.clone())?;
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
        let mut dataset = open_geoparquet(File::open(source)?)?;
        return Ok(dataset.filter_features(FeatureFilterRequest {
            features,
            selector: GeometrySelector::Name(manifest.selected_column.clone()),
            query,
            predicate,
            non_planar,
            expected_source_fingerprint,
        })?);
    }

    let matches = index.search_matches(query.clone())?;
    Ok(index
        .filter_matches(
            matches,
            query,
            predicate,
            if treat_nonplanar {
                NonPlanarExactPolicy::TreatAsPlanar
            } else {
                NonPlanarExactPolicy::Reject
            },
        )?
        .into_iter()
        .map(|m| m.feature)
        .collect())
}

/// Parse the 2D query geometry from `--bbox`, `--radius` or `--polygon`.
fn query_2d(parsed: &Parsed<'_>) -> Result<GeoQuery2D, Box<dyn std::error::Error>> {
    if parsed.option("--frustum")?.is_some() {
        return Err("--frustum is a 3D query; this is a 2D index -- use --bbox".into());
    }
    let shapes = [
        parsed.option("--bbox")?.map(|v| ("--bbox", v)),
        parsed.option("--radius")?.map(|v| ("--radius", v)),
        parsed.option("--polygon")?.map(|v| ("--polygon", v)),
    ];
    let mut given = shapes.into_iter().flatten();
    let Some((flag, value)) = given.next() else {
        return Err("--bbox, --radius or --polygon is required".into());
    };
    if given.next().is_some() {
        return Err("--bbox, --radius and --polygon are mutually exclusive".into());
    }
    match flag {
        "--bbox" => Ok(GeoQuery2D::box2d(parse_bbox(&value)?)),
        "--radius" => {
            let (lon, lat, radius_metres) = parse_radius(&value)?;
            Ok(GeoQuery2D::spherical_radius(lon, lat, radius_metres))
        }
        _ => Ok(GeoQuery2D::multi_polygon(parse_polygon(&value)?)),
    }
}

/// Parse the 3D query shape, `--bbox` or `--frustum`, rejecting the 2D-only
/// flags on the way.
fn query_3d(parsed: &Parsed<'_>) -> Result<GeoQuery3D, Box<dyn std::error::Error>> {
    reject_two_dimensional_flags(parsed)?;
    match (parsed.option("--bbox")?, parsed.option("--frustum")?) {
        (Some(_), Some(_)) => Err("--bbox and --frustum are mutually exclusive".into()),
        (Some(value), None) => Ok(GeoQuery3D::from(parse_bbox3d(&value)?)),
        (None, Some(value)) => Ok(GeoQuery3D::from(parse_frustum(&value)?)),
        (None, None) => Err(
            "--bbox or --frustum is required for a 3D index (--radius and --polygon are 2D-only)"
                .into(),
        ),
    }
}

fn reject_two_dimensional_flags(parsed: &Parsed<'_>) -> Result<(), Box<dyn std::error::Error>> {
    if parsed.option("--radius")?.is_some() {
        return Err("--radius is a 2D lon/lat query; this is a 3D index".into());
    }
    if parsed.option("--polygon")?.is_some() {
        return Err(
            "--polygon is a 2D query; this is a 3D index -- use --bbox or --frustum".into(),
        );
    }
    if parsed.flag("--exact") {
        return Err(
            "--exact is not supported for a 3D index: exact source-geometry filtering is \
             implemented only for 2D (the planar predicate stack is 2D-only). A 3D query returns \
             a bounding-box (envelope) candidate set, which for non-point geometry -- or any f32 \
             index -- is a superset, not the exact match set"
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
    Ok(())
}

/// 3D query path: `--bbox` only (6 numbers). `--radius`, `--exact`,
/// `--predicate`, and `--treat-nonplanar-as-planar` are 2D-only concepts and
/// are rejected here with an explanatory error rather than dispatched.
fn query_cmd_3d<R: RangeReader>(
    parsed: &Parsed<'_>,
    index: &GeoArtifactIndex3D<R>,
    page: Option<(usize, usize)>,
) -> Result<Vec<FeatureRef>, Box<dyn std::error::Error>> {
    let query = query_3d(parsed)?;
    if let Some((offset, limit)) = page {
        let page = index.search_match_headers_page(query, offset, limit)?;
        return Ok(index
            .fetch_matches(&page.headers)?
            .into_iter()
            .map(|m| m.feature)
            .collect());
    }
    Ok(index.search_feature_refs(query)?)
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
        geometry_json: true,
        order,
        duplicates,
        expected_source_fingerprint,
    };
    match kind {
        SourceKind::Parquet => {
            let mut dataset = open_geoparquet(File::open(source)?)?;
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

/// Whether the artifact should carry a contiguous copy of its feature refs.
/// `auto` measures the payload bodies; see `PrefixIndexPolicy`.
fn parse_prefix_index(
    value: Option<&str>,
) -> Result<PrefixIndexPolicy, Box<dyn std::error::Error>> {
    match value.unwrap_or("auto") {
        "auto" => Ok(PrefixIndexPolicy::Auto),
        "on" => Ok(PrefixIndexPolicy::Always),
        "off" => Ok(PrefixIndexPolicy::Never),
        other => Err(format!("invalid --prefix-index `{other}`").into()),
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
        "match" | "hit" => Ok(FeatureReadOrder::RequestOrder),
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

/// Parse `--polygon`: GeoJSON MultiPolygon coordinates, `[[[[x, y], ...], ...], ...]`.
///
/// The same array the server's `polygon=` takes and the crate's own
/// `GeoQuery2D` serializes, validated by the same rules: at least one polygon,
/// every exterior ring at least three points, every coordinate finite.
fn parse_polygon(raw: &str) -> Result<MultiPolygon<f64>, Box<dyn std::error::Error>> {
    let polygons: Vec<Vec<Vec<[f64; 2]>>> = serde_json::from_str(raw).map_err(|err| {
        format!("--polygon expects GeoJSON MultiPolygon coordinates [[[[x, y], ...]]]: {err}")
    })?;
    if polygons.is_empty() {
        return Err("--polygon must contain at least one polygon".into());
    }
    for (index, rings) in polygons.iter().enumerate() {
        let exterior = rings
            .first()
            .ok_or_else(|| format!("--polygon: polygon {index} has no exterior ring"))?;
        // Three distinct corners is the least that bounds any area; anything
        // smaller is a line or a point, which would match nothing while
        // looking like a region query that simply found nothing.
        if exterior.len() < 3 {
            return Err(format!(
                "--polygon: polygon {index} has an exterior ring of {} points; at least 3 are \
                 needed to bound an area",
                exterior.len()
            )
            .into());
        }
        if rings
            .iter()
            .flatten()
            .any(|[x, y]| !x.is_finite() || !y.is_finite())
        {
            return Err(format!("--polygon: polygon {index} has a non-finite coordinate").into());
        }
    }
    Ok(MultiPolygon::new(
        polygons
            .into_iter()
            .map(|rings| {
                let mut rings = rings.into_iter().map(|coords| {
                    LineString::new(coords.into_iter().map(|[x, y]| Coord { x, y }).collect())
                });
                let exterior = rings.next().unwrap_or_else(|| LineString::new(Vec::new()));
                Polygon::new(exterior, rings.collect())
            })
            .collect(),
    ))
}

/// Parse `--frustum`: 24 numbers, six inward-pointing planes as `a,b,c,d`.
///
/// Planes rather than a view-projection matrix, as on the server: a matrix
/// carries a clip-space convention and a storage order the flag cannot
/// recover, and either wrong moves the near plane without failing.
fn parse_frustum(raw: &str) -> Result<Frustum3D, Box<dyn std::error::Error>> {
    let mut values = Vec::with_capacity(24);
    for part in raw.split(',') {
        let part = part.trim();
        let value: f64 = part
            .parse()
            .map_err(|_| format!("--frustum value `{part}` is not a number"))?;
        if !value.is_finite() {
            return Err(format!("--frustum value `{part}` is not finite").into());
        }
        values.push(value);
    }
    if values.len() != 24 {
        return Err(format!(
            "--frustum expects 24 comma-separated numbers (six planes of a,b,c,d), got {}",
            values.len()
        )
        .into());
    }
    let mut planes = [[0.0f64; 4]; 6];
    for (plane, chunk) in planes.iter_mut().zip(values.as_chunks::<4>().0) {
        plane.copy_from_slice(chunk);
    }
    // A plane whose normal is zero tests nothing: `0*x + 0*y + 0*z + d` is a
    // constant, so the frustum silently becomes a half-open region instead
    // of failing. Cheaper to refuse than to explain.
    if let Some(index) = planes
        .iter()
        .position(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0)
    {
        return Err(
            format!("--frustum plane {index} has a zero normal, so it constrains nothing").into(),
        );
    }
    Ok(Frustum3D::from_planes(planes))
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
            | "--polygon"
            | "--frustum"
            | "--limit"
            | "--offset"
            | "--prefix-index"
            | "--within"
    )
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

    /// Two points 3.0 apart on the x axis, plus one far away.
    #[cfg(feature = "geojson")]
    fn artifact_2d(coords: &[(f64, f64)]) -> Vec<u8> {
        let features: Vec<String> = coords
            .iter()
            .map(|(x, y)| {
                format!(
                    r#"{{"type":"Feature","geometry":{{"type":"Point","coordinates":[{x},{y}]}},"properties":{{}}}}"#
                )
            })
            .collect();
        let doc = format!(
            r#"{{"type":"FeatureCollection","features":[{}]}}"#,
            features.join(",")
        );
        let mut bytes = Vec::new();
        packed_spatial_index_geo::open_geojson_slice(doc.as_bytes())
            .unwrap()
            .convert_into(ConvertRequest::default(), &mut bytes)
            .unwrap();
        bytes
    }

    #[cfg(feature = "geojson")]
    fn artifact_3d(coords: &[(f64, f64, f64)]) -> Vec<u8> {
        let features: Vec<String> = coords
            .iter()
            .map(|(x, y, z)| {
                format!(
                    r#"{{"type":"Feature","geometry":{{"type":"Point","coordinates":[{x},{y},{z}]}},"properties":{{}}}}"#
                )
            })
            .collect();
        let doc = format!(
            r#"{{"type":"FeatureCollection","features":[{}]}}"#,
            features.join(",")
        );
        let mut bytes = Vec::new();
        packed_spatial_index_geo::open_geojson_slice(doc.as_bytes())
            .unwrap()
            .convert_into(ConvertRequest::default(), &mut bytes)
            .unwrap();
        bytes
    }

    /// Pair order is traversal order, so every assertion sorts first.
    #[cfg(feature = "geojson")]
    fn sorted_pairs(out: &[u8]) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = String::from_utf8(out.to_vec())
            .unwrap()
            .lines()
            .map(|line| {
                let (a, b) = line
                    .trim_start_matches("{\"a\":")
                    .trim_end_matches('}')
                    .split_once(",\"b\":")
                    .unwrap_or_else(|| panic!("unexpected line `{line}`"));
                let pair: (usize, usize) = (a.parse().unwrap(), b.parse().unwrap());
                pair
            })
            .collect();
        pairs.sort_unstable();
        pairs
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn join_reports_pairs_within_max_distance() {
        let a = artifact_2d(&[(0.0, 0.0), (10.0, 0.0)]);
        let b = artifact_2d(&[(2.0, 0.0), (12.0, 0.0), (13.0, 0.0)]);

        let mut out = Vec::new();
        let count = join_pairs(&a, Some(&b), 2.0, false, &mut out).unwrap();
        // Both matching pairs sit exactly on the bound (2.0 and 2.0); the third
        // b point is 3.0 from a's item 1 and stays out. The bound is inclusive.
        assert_eq!(count, 2);
        assert_eq!(sorted_pairs(&out), vec![(0, 0), (1, 1)]);

        let mut out = Vec::new();
        let count = join_pairs(&a, Some(&b), 0.0, false, &mut out).unwrap();
        assert_eq!(count, 0);
        assert!(out.is_empty());
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn join_count_only_writes_nothing() {
        let a = artifact_2d(&[(0.0, 0.0), (10.0, 0.0)]);
        let b = artifact_2d(&[(2.0, 0.0), (12.0, 0.0), (13.0, 0.0)]);
        let mut out = Vec::new();
        let count = join_pairs(&a, Some(&b), 2.0, true, &mut out).unwrap();
        assert_eq!(count, 2);
        assert!(out.is_empty(), "count-only must not stream pairs");
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn self_join_pairs_each_distinct_pair_once() {
        let a = artifact_2d(&[(0.0, 0.0), (1.0, 0.0), (50.0, 0.0)]);
        let mut out = Vec::new();
        let count = join_pairs(&a, None, 1.0, false, &mut out).unwrap();
        assert_eq!(count, 1);
        // Which id lands on which side is traversal order, so normalize; the
        // pair is never (i, i) — a self-join reports distinct items only.
        let (i, j) = sorted_pairs(&out)[0];
        assert_eq!((i.min(j), i.max(j)), (0, 1));
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn join_works_in_3d() {
        let a = artifact_3d(&[(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]);
        let b = artifact_3d(&[(0.0, 0.0, 2.0)]);
        let mut out = Vec::new();
        let count = join_pairs(&a, Some(&b), 2.0, false, &mut out).unwrap();
        assert_eq!(count, 1);
        assert_eq!(sorted_pairs(&out), vec![(0, 0)]);
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn join_rejects_mixed_dimensions() {
        let a = artifact_2d(&[(0.0, 0.0)]);
        let b = artifact_3d(&[(0.0, 0.0, 0.0)]);
        let mut out = Vec::new();
        let err = join_pairs(&a, Some(&b), 1.0, false, &mut out).unwrap_err();
        assert!(err.to_string().contains("different dimensions"), "{err}");
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn join_cmd_reads_two_files() {
        let dir = std::env::temp_dir().join(format!("gp2psindex-join-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a_path = dir.join("a.psi");
        let b_path = dir.join("b.psi");
        std::fs::write(&a_path, artifact_2d(&[(0.0, 0.0), (10.0, 0.0)])).unwrap();
        std::fs::write(&b_path, artifact_2d(&[(2.0, 0.0)])).unwrap();

        join_cmd(&[
            a_path.to_string_lossy().into_owned(),
            b_path.to_string_lossy().into_owned(),
            "--within".to_string(),
            "2.0".to_string(),
            "--count".to_string(),
        ])
        .unwrap();

        // Passing the same path twice is a self-join, not a cross join.
        join_cmd(&[
            a_path.to_string_lossy().into_owned(),
            a_path.to_string_lossy().into_owned(),
            "--within=1.0".to_string(),
            "--count".to_string(),
        ])
        .unwrap();

        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn anti_join_reports_items_with_no_partner() {
        let a = artifact_2d(&[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]);
        let b = artifact_2d(&[(2.0, 0.0), (23.0, 0.0)]);
        let mut out = Vec::new();
        // Item 0 has b[0] at 2.0 (on the bound, inclusive); item 2 has b[1] at
        // 3.0, outside; item 1 has nothing within 8.0. Unpaired: 1 and 2.
        let count = anti_join_items(&a, &b, 2.0, false, &mut out).unwrap();
        assert_eq!(count, 2);
        let mut items: Vec<usize> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|line| {
                line.trim_start_matches("{\"a\":")
                    .trim_end_matches('}')
                    .parse()
                    .unwrap()
            })
            .collect();
        items.sort_unstable();
        assert_eq!(items, vec![1, 2]);

        let mut out = Vec::new();
        let count = anti_join_items(&a, &b, 2.0, true, &mut out).unwrap();
        assert_eq!(count, 2);
        assert!(out.is_empty(), "count-only must not stream items");

        // A huge bound pairs everything; nothing is unpaired.
        let mut out = Vec::new();
        assert_eq!(anti_join_items(&a, &b, 1000.0, true, &mut out).unwrap(), 0);
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn anti_join_rejects_mixed_dimensions_and_same_file() {
        let a = artifact_2d(&[(0.0, 0.0)]);
        let b = artifact_3d(&[(0.0, 0.0, 0.0)]);
        let mut out = Vec::new();
        let err = anti_join_items(&a, &b, 1.0, false, &mut out).unwrap_err();
        assert!(err.to_string().contains("different dimensions"), "{err}");

        let dir = std::env::temp_dir().join(format!("gp2psindex-antijoin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.psi");
        std::fs::write(&path, &a).unwrap();
        let path = path.to_string_lossy().into_owned();
        let err = anti_join_cmd(&[path.clone(), path, "--within=1".to_string()]).unwrap_err();
        assert!(err.to_string().contains("components"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "geojson")]
    #[test]
    fn components_label_by_smallest_member() {
        // 0-1 touch within 1.5, 2 is alone, 3-4 touch: three components.
        let a = artifact_2d(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (21.0, 0.0),
        ]);
        let labels = component_labels(&a, 1.5).unwrap();
        assert_eq!(labels, vec![0, 0, 2, 3, 3]);
        assert_eq!(count_components(&labels), 3);

        // A chain is one component however far its ends lie apart.
        let labels = component_labels(&a, 10.0).unwrap();
        assert_eq!(labels, vec![0, 0, 0, 0, 0]);
        assert_eq!(count_components(&labels), 1);

        // Zero bound: nothing overlaps, every point is its own component.
        let labels = component_labels(&a, 0.0).unwrap();
        assert_eq!(labels, vec![0, 1, 2, 3, 4]);
        assert_eq!(count_components(&labels), 5);
    }

    #[test]
    fn polygon_parses_geojson_multipolygon_coordinates() {
        let multi = parse_polygon("[[[[0,0],[4,0],[4,4],[0,4],[0,0]]]]").unwrap();
        assert_eq!(multi.0.len(), 1);
        assert_eq!(multi.0[0].exterior().0.len(), 5);

        for (bad, needle) in [
            ("[]", "at least one polygon"),
            ("[[]]", "no exterior ring"),
            ("[[[[0,0],[1,1]]]]", "at least 3"),
            ("[[[[0,0],[1,0],[0,1]]],[[]]]", "0 points"),
            ("not json", "MultiPolygon"),
        ] {
            let err = parse_polygon(bad).unwrap_err().to_string();
            assert!(err.contains(needle), "{bad}: {err}");
        }
    }

    #[test]
    fn frustum_parses_six_planes_and_rejects_zero_normals() {
        let raw = "1,0,0,0, -1,0,0,10, 0,1,0,0, 0,-1,0,10, 0,0,1,0, 0,0,-1,10";
        let frustum = parse_frustum(raw).unwrap();
        // Six axis-aligned planes bound the unit-ish box [0,10]^3.
        assert!(frustum.overlaps_box(Box3D::new(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)));
        assert!(!frustum.overlaps_box(Box3D::new(20.0, 20.0, 20.0, 21.0, 21.0, 21.0)));

        let err = parse_frustum("1,0,0,0").unwrap_err().to_string();
        assert!(err.contains("24"), "{err}");
        let zero = "0,0,0,1, -1,0,0,10, 0,1,0,0, 0,-1,0,10, 0,0,1,0, 0,0,-1,10";
        let err = parse_frustum(zero).unwrap_err().to_string();
        assert!(err.contains("zero normal"), "{err}");
        let err = parse_frustum("1,0,0,x").unwrap_err().to_string();
        assert!(err.contains("not a number"), "{err}");
    }

    #[test]
    fn max_distance_is_required_finite_and_non_negative() {
        assert!(
            parse_max_distance(None)
                .unwrap_err()
                .to_string()
                .contains("required")
        );
        for bad in ["-1", "nan", "inf", "abc"] {
            let err = parse_max_distance(Some(bad)).unwrap_err().to_string();
            assert!(
                err.contains("finite non-negative") || err.contains("not a number"),
                "{bad}: {err}"
            );
        }
        assert_eq!(parse_max_distance(Some("0")).unwrap(), 0.0);
        assert_eq!(parse_max_distance(Some("2.5")).unwrap(), 2.5);
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
