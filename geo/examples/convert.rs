//! Convert a GeoParquet file into a streamable `PSINDEX` sidecar.
//!
//! Run:
//! ```text
//! cargo run --example convert -- path/to/file.parquet
//! ```
//! It inspects the file, converts the right dimensionality, then writes
//! `<file>.psindex`. Query that output with `packed_spatial_index`'s
//! `StreamIndex2D` / `StreamIndex3D` (range / kNN / raycast straight from the
//! file or an HTTP range source) — re-exported here for convenience.

use std::fs::File;

use packed_spatial_index_geo::{ConvertOpts, convert_2d, convert_3d, inspect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: convert <file.parquet>")?;

    let info = inspect(File::open(&path)?)?;
    println!(
        "GeoParquet {} — {} rows, {}D, encoding {}, covering={}",
        info.version, info.num_rows, info.dims, info.encoding, info.has_covering
    );
    if let Some(bounds) = &info.bounds {
        println!("extent: {bounds:?}");
    }

    // Drop any null/empty geometries rather than failing the whole file.
    let opts = ConvertOpts {
        skip_null: true,
        ..Default::default()
    };
    let psindex = if info.dims == 3 {
        convert_3d(File::open(&path)?, opts)?
    } else {
        convert_2d(File::open(&path)?, opts)?
    };

    let out = format!("{path}.psindex");
    std::fs::write(&out, &psindex)?;
    println!("wrote {out} ({} bytes)", psindex.len());
    Ok(())
}
