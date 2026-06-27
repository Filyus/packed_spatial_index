//! `gp2psindex` — convert a geospatial Parquet file into a streamable `PSINDEX`.
//!
//! ```text
//! gp2psindex <input.parquet> [output.psi] [--f32] [--strict]
//!   [--geometry-column name] [--payload none|row-id|row-wkb]
//!   [--no-payload] [--no-interleave]
//! gp2psindex inspect <input.parquet> [--geometry-column name] [--json]
//! ```
//!
//! Defaults: 2D/3D auto-detected, payload is `original row id + WKB`, null/empty
//! rows dropped. Query the output with `packed_spatial_index`'s `StreamIndex2D` /
//! `StreamIndex3D` over a file or HTTP range source.

use std::fs::File;
use std::process::ExitCode;

use packed_spatial_index_geo::{
    ConvertOpts, ConvertPayload, GeometryColumnSelection, GeometryDiscovery,
    GeometryMetadataSource, GeometrySelectionReason, ReadOpts, convert_2d, convert_3d,
    discover_with_opts, inspect_with_opts,
};

const USAGE: &str = "usage: gp2psindex <input.parquet> [output.psi] [--f32] [--strict] [--geometry-column name] [--payload none|row-id|row-wkb] [--no-payload] [--no-interleave]\n       gp2psindex inspect <input.parquet> [--geometry-column name] [--json]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if is_inspect_command(&args) {
        return inspect_main(&args[1..]);
    }

    convert_main(&args)
}

fn is_inspect_command(args: &[String]) -> bool {
    args.first().is_some_and(|arg| arg == "inspect")
}

fn convert_main(args: &[String]) -> ExitCode {
    let flag = |name: &str| args.iter().any(|a| a == name);
    let positionals = positionals(args);

    if positionals.is_empty() || flag("--help") || flag("-h") {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }

    let input = positionals[0].clone();
    let output = positionals
        .get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{input}.psi"));

    let payload = match payload_mode(args) {
        Ok(payload) => payload,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let geometry_column = match geometry_column(args) {
        Ok(column) => column,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let opts = ConvertOpts {
        geometry_column,
        compact_f32: flag("--f32"),
        skip_null: !flag("--strict"),
        include_payload: !flag("--no-payload") && payload != ConvertPayload::None,
        payload,
        interleaved: !flag("--no-interleave"),
        ..Default::default()
    };

    match run(&input, &output, opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_main(args: &[String]) -> ExitCode {
    let flag = |name: &str| args.iter().any(|a| a == name);
    let positionals = positionals(args);
    if positionals.len() != 1 || flag("--help") || flag("-h") {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }

    let geometry_column = match geometry_column(args) {
        Ok(column) => column,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    let input = positionals[0];
    let opts = ReadOpts { geometry_column };
    match inspect_run(input, opts, flag("--json")) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn positionals(args: &[String]) -> Vec<&String> {
    let mut out = Vec::new();
    let mut skip_next_value = false;
    for arg in args {
        if skip_next_value {
            skip_next_value = false;
            continue;
        }
        if arg == "--payload" || arg == "--geometry-column" {
            skip_next_value = true;
            continue;
        }
        if arg.starts_with("--") {
            continue;
        }
        out.push(arg);
    }
    out
}

fn payload_mode(args: &[String]) -> Result<ConvertPayload, String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let value = if arg == "--payload" {
            iter.next().map(String::as_str)
        } else {
            arg.strip_prefix("--payload=")
        };
        match value {
            Some("none") => return Ok(ConvertPayload::None),
            Some("row-id") | Some("row-ids") => return Ok(ConvertPayload::RowIds),
            Some("row-wkb") => return Ok(ConvertPayload::RowWkb),
            Some(other) => return Err(format!("unknown payload mode `{other}`")),
            None if arg == "--payload" => return Err("--payload requires a value".to_string()),
            None => {}
        }
    }
    Ok(ConvertPayload::default())
}

fn geometry_column(args: &[String]) -> Result<Option<String>, String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let value = if arg == "--geometry-column" {
            iter.next().map(String::as_str)
        } else {
            arg.strip_prefix("--geometry-column=")
        };
        match value {
            Some("") => {
                return Err("--geometry-column requires a non-empty value".to_string());
            }
            Some(value) => return Ok(Some(value.to_string())),
            None if arg == "--geometry-column" => {
                return Err("--geometry-column requires a value".to_string());
            }
            None => {}
        }
    }
    Ok(None)
}

fn run(input: &str, output: &str, opts: ConvertOpts) -> Result<(), Box<dyn std::error::Error>> {
    let info = inspect_with_opts(
        File::open(input)?,
        ReadOpts {
            geometry_column: opts.geometry_column.clone(),
        },
    )?;
    eprintln!(
        "{:?} {} — {} rows, {}D, column {}, encoding {}, covering={}",
        info.metadata_source,
        info.version,
        info.num_rows,
        info.dims,
        info.geometry_column,
        info.encoding,
        info.has_covering
    );

    let bytes = if info.dims == 3 {
        convert_3d(File::open(input)?, opts)?
    } else {
        convert_2d(File::open(input)?, opts)?
    };

    std::fs::write(output, &bytes)?;
    eprintln!("wrote {output} ({} bytes)", bytes.len());
    Ok(())
}

fn inspect_run(input: &str, opts: ReadOpts, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let discovery = discover_with_opts(File::open(input)?, opts)?;
    if json {
        serde_json::to_writer_pretty(std::io::stdout(), &discovery)?;
        println!();
    } else {
        print_discovery(&discovery);
    }
    Ok(())
}

fn print_discovery(discovery: &GeometryDiscovery) {
    println!("rows: {}", discovery.num_rows);
    match (
        &discovery.geo_metadata_version,
        &discovery.geo_primary_column,
    ) {
        (Some(version), Some(primary)) => {
            println!("geoparquet: version {version}, primary column {primary}");
        }
        (Some(version), None) => println!("geoparquet: version {version}"),
        (None, _) => println!("geoparquet: none"),
    }
    println!("selection: {}", selection_label(&discovery.selection));
    println!("columns:");
    println!(
        "  {:<24} {:<18} {:<20} {:<7} {:<8} {:<8} {:<8} crs",
        "column", "source", "encoding", "dims", "cover", "index", "row_wkb"
    );
    for column in &discovery.columns {
        println!(
            "  {:<24} {:<18} {:<20} {:<7} {:<8} {:<8} {:<8} {}",
            column.name,
            source_label(column.metadata_source),
            column.encoding,
            dims_label(column.dims),
            yes_no(column.has_covering),
            yes_no(column.can_build_index),
            yes_no(column.can_emit_row_wkb),
            column.crs.as_deref().unwrap_or("-")
        );
    }
}

fn selection_label(selection: &GeometryColumnSelection) -> String {
    match selection {
        GeometryColumnSelection::Selected { column, reason } => {
            format!("selected `{column}` ({})", reason_label(*reason))
        }
        GeometryColumnSelection::Ambiguous { columns } => {
            format!("ambiguous: {}", columns.join(", "))
        }
        GeometryColumnSelection::Missing { column } => format!("missing `{column}`"),
        GeometryColumnSelection::None => "none".to_string(),
    }
}

fn reason_label(reason: GeometrySelectionReason) -> &'static str {
    match reason {
        GeometrySelectionReason::Explicit => "explicit",
        GeometrySelectionReason::GeoParquetPrimary => "GeoParquet primary",
        GeometrySelectionReason::SingleNativeParquet => "single native Parquet column",
    }
}

fn source_label(source: GeometryMetadataSource) -> &'static str {
    match source {
        GeometryMetadataSource::GeoParquet => "GeoParquet",
        GeometryMetadataSource::ParquetGeospatial => "ParquetGeospatial",
    }
}

fn dims_label(dims: Option<u8>) -> String {
    dims.map(|d| format!("{d}D"))
        .unwrap_or_else(|| "unknown".to_string())
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn payload_value_is_not_positional() {
        let args = args(&["in.parquet", "out.psi", "--payload", "row-id"]);
        let positional: Vec<&str> = positionals(&args).into_iter().map(String::as_str).collect();

        assert_eq!(positional, vec!["in.parquet", "out.psi"]);
        assert_eq!(payload_mode(&args), Ok(ConvertPayload::RowIds));
    }

    #[test]
    fn payload_equals_form_is_parsed() {
        let args = args(&["in.parquet", "--payload=row-wkb"]);

        assert_eq!(positionals(&args).len(), 1);
        assert_eq!(payload_mode(&args), Ok(ConvertPayload::RowWkb));
    }

    #[test]
    fn payload_requires_a_value() {
        let args = args(&["in.parquet", "--payload"]);
        let positional: Vec<&str> = positionals(&args).into_iter().map(String::as_str).collect();

        assert!(payload_mode(&args).is_err());
        assert_eq!(positional, vec!["in.parquet"]);
    }

    #[test]
    fn geometry_column_value_is_not_positional() {
        let args = args(&["in.parquet", "out.psi", "--geometry-column", "geom"]);
        let positional: Vec<&str> = positionals(&args).into_iter().map(String::as_str).collect();

        assert_eq!(positional, vec!["in.parquet", "out.psi"]);
        assert_eq!(geometry_column(&args), Ok(Some("geom".to_string())));
    }

    #[test]
    fn geometry_column_equals_form_is_parsed() {
        let args = args(&["in.parquet", "--geometry-column=geom"]);

        assert_eq!(positionals(&args).len(), 1);
        assert_eq!(geometry_column(&args), Ok(Some("geom".to_string())));
    }

    #[test]
    fn geometry_column_requires_a_value() {
        let args = args(&["in.parquet", "--geometry-column"]);
        let positional: Vec<&str> = positionals(&args).into_iter().map(String::as_str).collect();

        assert!(geometry_column(&args).is_err());
        assert_eq!(positional, vec!["in.parquet"]);
    }

    #[test]
    fn inspect_subcommand_keeps_input_as_only_positional() {
        let args = args(&[
            "inspect",
            "in.parquet",
            "--geometry-column",
            "geom",
            "--json",
        ]);
        let tail = &args[1..];
        let positional: Vec<&str> = positionals(tail).into_iter().map(String::as_str).collect();

        assert!(is_inspect_command(&args));
        assert_eq!(positional, vec!["in.parquet"]);
        assert_eq!(geometry_column(tail), Ok(Some("geom".to_string())));
    }
}
