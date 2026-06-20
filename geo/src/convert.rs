//! Converter (Model 2): build the index *and* attach each row's WKB geometry as
//! a leaf-ordered payload, plus the CRS, serialized to a self-describing
//! `PSINDEX` blob. The output is queryable by the streaming engine straight from
//! cloud storage — window / kNN / raycast returning the actual geometry in a few
//! range reads, with no Parquet re-read.

use packed_spatial_index::{Index2DBuilder, Index3DBuilder};
use parquet::file::reader::ChunkReader;

use crate::{ConvertOpts, GeoError, build, read};

/// WKB content type recorded in the index metadata so a reader knows the payload
/// bytes are Well-Known Binary geometry.
const WKB_CONTENT_TYPE: &str = "application/geo+wkb";

/// Convert a 2D GeoParquet source into `PSINDEX` bytes.
///
/// The bytes carry the index, each row's WKB geometry as a leaf-ordered payload,
/// and the CRS. Query them with [`StreamIndex2D`](crate::StreamIndex2D) over any
/// [`RangeReader`](crate::RangeReader) (a local file or an HTTP range source).
///
/// # Examples
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{convert_2d, ConvertOpts};
///
/// let psindex = convert_2d(File::open("cities.parquet")?, ConvertOpts::default())?;
/// std::fs::write("cities.psindex", &psindex)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn convert_2d<R: ChunkReader + 'static>(
    reader: R,
    opts: ConvertOpts,
) -> Result<Vec<u8>, GeoError> {
    let mut out = Vec::new();
    convert_2d_into(reader, opts, &mut out)?;
    Ok(out)
}

/// Convert a 3D GeoParquet source into `PSINDEX` bytes.
pub fn convert_3d<R: ChunkReader + 'static>(
    reader: R,
    opts: ConvertOpts,
) -> Result<Vec<u8>, GeoError> {
    let mut out = Vec::new();
    convert_3d_into(reader, opts, &mut out)?;
    Ok(out)
}

/// Convert a 2D GeoParquet source, appending the `PSINDEX` bytes to `out`. Lets
/// the caller reuse a buffer or write straight into one it already owns.
pub fn convert_2d_into<R: ChunkReader + 'static>(
    reader: R,
    opts: ConvertOpts,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let scan = read::scan_2d(reader, opts.include_payload, opts.skip_null)?;
    let builder = build::loaded_builder_2d(&scan.boxes, &opts.build);
    serialize_2d(
        builder,
        scan.wkb.as_deref(),
        scan.crs.as_deref(),
        &opts,
        out,
    )
}

/// Convert a 3D GeoParquet source, appending the `PSINDEX` bytes to `out`.
pub fn convert_3d_into<R: ChunkReader + 'static>(
    reader: R,
    opts: ConvertOpts,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let scan = read::scan_3d(reader, opts.include_payload, opts.skip_null)?;
    let builder = build::loaded_builder_3d(&scan.boxes, &opts.build);
    serialize_3d(
        builder,
        scan.wkb.as_deref(),
        scan.crs.as_deref(),
        &opts,
        out,
    )
}

fn serialize_2d(
    builder: Index2DBuilder,
    wkb: Option<&[Vec<u8>]>,
    crs: Option<&str>,
    opts: &ConvertOpts,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let payload = if opts.include_payload { wkb } else { None };
    if opts.compact_f32 {
        let index = builder.finish_f32()?;
        let mut s = index.serialize();
        if opts.interleaved && payload.is_some() {
            s = s.interleaved();
        }
        if let Some(c) = crs {
            s = s.crs(c);
        }
        if let Some(w) = payload {
            s = s.payloads(w).content_type(WKB_CONTENT_TYPE);
        }
        s.to_bytes_into(out)?;
    } else {
        let index = builder.finish()?;
        let mut s = index.serialize();
        if opts.interleaved && payload.is_some() {
            s = s.interleaved();
        }
        if let Some(c) = crs {
            s = s.crs(c);
        }
        if let Some(w) = payload {
            s = s.payloads(w).content_type(WKB_CONTENT_TYPE);
        }
        s.to_bytes_into(out)?;
    }
    Ok(())
}

fn serialize_3d(
    builder: Index3DBuilder,
    wkb: Option<&[Vec<u8>]>,
    crs: Option<&str>,
    opts: &ConvertOpts,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let payload = if opts.include_payload { wkb } else { None };
    if opts.compact_f32 {
        let index = builder.finish_f32()?;
        let mut s = index.serialize();
        if opts.interleaved && payload.is_some() {
            s = s.interleaved();
        }
        if let Some(c) = crs {
            s = s.crs(c);
        }
        if let Some(w) = payload {
            s = s.payloads(w).content_type(WKB_CONTENT_TYPE);
        }
        s.to_bytes_into(out)?;
    } else {
        let index = builder.finish()?;
        let mut s = index.serialize();
        if opts.interleaved && payload.is_some() {
            s = s.interleaved();
        }
        if let Some(c) = crs {
            s = s.crs(c);
        }
        if let Some(w) = payload {
            s = s.payloads(w).content_type(WKB_CONTENT_TYPE);
        }
        s.to_bytes_into(out)?;
    }
    Ok(())
}
