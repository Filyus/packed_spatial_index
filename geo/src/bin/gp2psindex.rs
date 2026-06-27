//! `gp2psindex` — convert a GeoParquet file into a streamable `PSINDEX`.
//!
//! ```text
//! gp2psindex <input.parquet> [output.psi] [--f32] [--strict]
//!   [--payload none|row-id|row-wkb] [--no-payload] [--no-interleave]
//! ```
//!
//! Defaults: 2D/3D auto-detected, payload is `original row id + WKB`, null/empty
//! rows dropped. Query the output with `packed_spatial_index`'s `StreamIndex2D` /
//! `StreamIndex3D` over a file or HTTP range source.

use std::fs::File;
use std::process::ExitCode;

use packed_spatial_index_geo::{ConvertOpts, ConvertPayload, convert_2d, convert_3d, inspect};

const USAGE: &str = "usage: gp2psindex <input.parquet> [output.psi] [--f32] [--strict] [--payload none|row-id|row-wkb] [--no-payload] [--no-interleave]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let positionals = positionals(&args);

    if positionals.is_empty() || flag("--help") || flag("-h") {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }

    let input = positionals[0].clone();
    let output = positionals
        .get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{input}.psi"));

    let payload = match payload_mode(&args) {
        Ok(payload) => payload,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let opts = ConvertOpts {
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

fn positionals(args: &[String]) -> Vec<&String> {
    let mut out = Vec::new();
    let mut skip_payload_value = false;
    for arg in args {
        if skip_payload_value {
            skip_payload_value = false;
            continue;
        }
        if arg == "--payload" {
            skip_payload_value = true;
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

fn run(input: &str, output: &str, opts: ConvertOpts) -> Result<(), Box<dyn std::error::Error>> {
    let info = inspect(File::open(input)?)?;
    eprintln!(
        "GeoParquet {} — {} rows, {}D, encoding {}, covering={}",
        info.version, info.num_rows, info.dims, info.encoding, info.has_covering
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
}
