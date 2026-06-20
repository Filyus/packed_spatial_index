//! `gp2psindex` — convert a GeoParquet file into a streamable `PSINDEX`.
//!
//! ```text
//! gp2psindex <input.parquet> [output.psi] [--f32] [--strict] [--no-payload] [--no-interleave]
//! ```
//!
//! Defaults: 2D/3D auto-detected, geometry kept as a leaf-ordered interleaved WKB
//! payload, null/empty rows dropped. Query the output with `packed_spatial_index`'s
//! `StreamIndex2D` / `StreamIndex3D` over a file or HTTP range source.

use std::fs::File;
use std::process::ExitCode;

use packed_spatial_index_geo::{ConvertOpts, convert_2d, convert_3d, inspect};

const USAGE: &str = "usage: gp2psindex <input.parquet> [output.psi] [--f32] [--strict] [--no-payload] [--no-interleave]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let positionals: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();

    if positionals.is_empty() || flag("--help") || flag("-h") {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }

    let input = positionals[0].clone();
    let output = positionals
        .get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{input}.psi"));

    let opts = ConvertOpts {
        compact_f32: flag("--f32"),
        skip_null: !flag("--strict"),
        include_payload: !flag("--no-payload"),
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
