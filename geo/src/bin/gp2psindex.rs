use std::fs::File;
use std::process::ExitCode;

use packed_spatial_index_geo::{
    AntimeridianPolicy, ConvertRequest, EnvelopePolicy, GeoDiscovery, GeoError, GeometryProfile,
    GeometrySelector, IndexDimsRequest, InspectRequest, NullPolicy, PayloadPlan,
    PropertyProjection, StoragePrecision, open,
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
  gp2psindex validate <input.parquet> [--geometry-column name]";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err("missing command".into());
    };
    match command {
        "discover" => discover_cmd(&args[1..]),
        "inspect" => inspect_cmd(&args[1..]),
        "build" => build_cmd(&args[1..]),
        "validate" => validate_cmd(&args[1..]),
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

fn validate_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Parsed::new(args);
    let input = parsed.required_pos(0, "input.parquet")?;
    parsed.no_extra_pos(1)?;
    let selector = geometry_selector(parsed.option("--geometry-column")?);
    let mut dataset = open(File::open(input)?)?;
    let profile = dataset.inspect(InspectRequest {
        selector,
        exact: true,
    })?;
    println!(
        "ok: {} rows, column `{}`, encoding {}",
        profile.num_rows, profile.column, profile.encoding
    );
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
