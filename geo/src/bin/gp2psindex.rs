//! Command-line converter and inspector for geospatial Parquet inputs.

use std::fs::File;
use std::process::ExitCode;
use std::sync::Arc;

use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use arrow_json::LineDelimitedWriter;
use base64::Engine as _;
use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, ConvertRequest, DuplicateFeatureRows, EnvelopePolicy,
    FeatureFilterRequest, FeatureReadOrder, FeatureReadRequest, FeatureRows, GeoArtifactIndex,
    GeoDiscovery, GeoError, GeometryProfile, GeometryReadMode, GeometrySelector, IndexDimsRequest,
    InspectRequest, NonPlanarExactPolicy, NullPolicy, PayloadPlan, PropertyProjection,
    QueryGeometry, SliceReader, SpatialPredicate, StoragePrecision, ValidateRequest,
    ValidationReport, ValidationSeverity, open, open_geo_index,
};

const USAGE: &str = "\
usage:
  gp2psindex discover <input.parquet> [--json]
  gp2psindex inspect <input.parquet> [--geometry-column name] [--exact] [--json]
  gp2psindex build <input.parquet> <output.psi>
      [--geometry-column name]
      [--dims auto|2d|3d]
      [--precision f64|f32]
      [--nulls error|skip]
      [--payload none|row-ref|row-wkb|feature-json]
      [--properties none|all|include:a,b|exclude:a,b]
      [--antimeridian reject|split|world]
      [--no-interleave]
  gp2psindex validate <input.parquet>
      [--geometry-column name]
      [--exact]
      [--json]
      [--strict]
      [--dims auto|2d|3d]
      [--nulls error|skip]
      [--payload none|row-ref|row-wkb|feature-json]
      [--properties none|all|include:a,b|exclude:a,b]
      [--antimeridian reject|split|world]
  gp2psindex query <source.parquet> <index.psi>
      (--bbox xmin,ymin,xmax,ymax | --radius lon,lat,metres)
      [--exact]
      [--predicate intersects]
      [--treat-nonplanar-as-planar]
      [--geometry none|wkb]
      [--properties none|all|include:a,b|exclude:a,b]
      [--order source|hit]
      [--duplicates dedup|parts]
      [--json|--ndjson]
      [--allow-source-mismatch]";

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

fn discover_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    let input = parsed.required_pos(0, "input.parquet")?;
    parsed.no_extra_pos(1)?;
    let dataset = open(File::open(input)?)?;
    if parsed.flag("--json") {
        serde_json::to_writer_pretty(std::io::stdout(), dataset.discovery())?;
        println!();
    } else {
        print_discovery(dataset.discovery());
    }
    Ok(())
}

fn inspect_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    let input = parsed.required_pos(0, "input.parquet")?;
    parsed.no_extra_pos(1)?;
    let selector = geometry_selector(parsed.option("--geometry-column")?);
    let mut dataset = open(File::open(input)?)?;
    let profile = dataset.inspect(InspectRequest {
        selector,
        exact: parsed.flag("--exact"),
    })?;
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
    let input = parsed.required_pos(0, "input.parquet")?;
    let output = parsed.required_pos(1, "output.psi")?;
    parsed.no_extra_pos(2)?;
    let mut dataset = open(File::open(input)?)?;
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
    let artifact = dataset.convert_into(request, &mut bytes)?;
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
    let input = parsed.required_pos(0, "input.parquet")?;
    parsed.no_extra_pos(1)?;
    let payload = parse_payload(
        parsed.option("--payload")?.as_deref().unwrap_or("row-wkb"),
        parsed.option("--properties")?,
    )?;
    let mut dataset = open(File::open(input)?)?;
    let report = dataset.validate(ValidateRequest {
        selector: geometry_selector(parsed.option("--geometry-column")?),
        exact: parsed.flag("--exact"),
        dims: parse_dims(parsed.option("--dims")?.as_deref().unwrap_or("auto"))?,
        nulls: parse_nulls(parsed.option("--nulls")?.as_deref().unwrap_or("skip"))?,
        envelope: parse_antimeridian(parsed.option("--antimeridian")?)?,
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

fn query_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    let source = parsed.required_pos(0, "source.parquet")?;
    let index_path = parsed.required_pos(1, "index.psi")?;
    parsed.no_extra_pos(2)?;
    let bbox = parsed.option("--bbox")?;
    let radius = parsed.option("--radius")?;
    let query = match (bbox, radius) {
        (Some(_), Some(_)) => return Err("--bbox and --radius are mutually exclusive".into()),
        (Some(value), None) => QueryGeometry::Box2D(parse_bbox(&value)?),
        (None, Some(value)) => {
            let (lon, lat, radius_metres) = parse_radius(&value)?;
            QueryGeometry::SphericalRadius {
                lon,
                lat,
                radius_metres,
            }
        }
        (None, None) => return Err("--bbox or --radius is required".into()),
    };
    let geometry = parse_geometry_read(parsed.option("--geometry")?.as_deref().unwrap_or("none"))?;
    let properties = parse_properties(parsed.option("--properties")?.as_deref().unwrap_or("all"))?;
    let order = parse_feature_order(parsed.option("--order")?.as_deref().unwrap_or("source"))?;
    let duplicates =
        parse_duplicates(parsed.option("--duplicates")?.as_deref().unwrap_or("dedup"))?;
    let radius_query = matches!(query, QueryGeometry::SphericalRadius { .. });
    let exact = parsed.flag("--exact") || radius_query;
    let predicate = parsed.option("--predicate")?;
    let treat_nonplanar = parsed.flag("--treat-nonplanar-as-planar");
    if !exact && (predicate.is_some() || treat_nonplanar) {
        return Err("--predicate and --treat-nonplanar-as-planar require --exact".into());
    }
    if radius_query && treat_nonplanar {
        return Err("--treat-nonplanar-as-planar cannot be used with --radius".into());
    }

    if parsed.flag("--json") && parsed.flag("--ndjson") {
        return Err("--json and --ndjson are mutually exclusive".into());
    }

    let bytes = std::fs::read(index_path)?;
    let artifact = open_geo_index(SliceReader::new(bytes))?;
    let manifest = artifact.manifest().clone();
    let candidate_boxes = query.candidate_boxes_2d()?;
    let features = match artifact {
        GeoArtifactIndex::D2(index) => {
            let mut features = Vec::new();
            for bbox in candidate_boxes {
                for feature in index.search_features(bbox)? {
                    if !features.contains(&feature) {
                        features.push(feature);
                    }
                }
            }
            features
        }
        GeoArtifactIndex::D3(_) => {
            return Err("query CLI currently accepts only 2D --bbox/--radius queries".into());
        }
    };
    let expected_source_fingerprint =
        (!parsed.flag("--allow-source-mismatch")).then_some(manifest.source_fingerprint.clone());
    let features = if exact {
        let mut dataset = open(File::open(source)?)?;
        dataset.filter_features(FeatureFilterRequest {
            features,
            selector: GeometrySelector::Name(manifest.selected_column.clone()),
            query,
            predicate: parse_spatial_predicate(predicate.as_deref().unwrap_or("intersects"))?,
            non_planar: if treat_nonplanar {
                NonPlanarExactPolicy::TreatAsPlanar
            } else {
                NonPlanarExactPolicy::Reject
            },
            expected_source_fingerprint: expected_source_fingerprint.clone(),
        })?
    } else {
        features
    };

    let mut dataset = open(File::open(source)?)?;
    let rows = dataset.read_features(FeatureReadRequest {
        features,
        selector: GeometrySelector::Name(manifest.selected_column.clone()),
        properties,
        geometry,
        order,
        duplicates,
        expected_source_fingerprint,
    })?;
    print_query_rows(&rows, parsed.flag("--json"))?;
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
}
