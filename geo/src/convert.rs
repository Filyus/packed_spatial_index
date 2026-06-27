//! Converter (Model 2): build the index and optionally attach a leaf-ordered
//! payload, plus the CRS, serialized to a self-describing `PSINDEX` blob. The
//! default payload stores the original GeoParquet row id followed by WKB bytes,
//! so a compacted converter output can still point back to source rows.

use packed_spatial_index::{Index2DBuilder, Index3DBuilder};
use parquet::file::reader::ChunkReader;

use crate::{ConvertOpts, GeoError, build, read};

/// Content type recorded for fixed-width little-endian `u64` row-id payloads.
pub const ROW_ID_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.geo.row-id";

/// Content type recorded for `u64le original_row_id` followed by WKB bytes.
pub const ROW_WKB_CONTENT_TYPE: &str = "application/vnd.packed-spatial-index.geo.row-wkb";

/// Payload written by the GeoParquet converter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConvertPayload {
    /// Write no payload. Search results are compact item ids only.
    None,
    /// Write one fixed-width little-endian `u64` original GeoParquet row id per
    /// indexed item. This is the compact sidecar-index mode.
    RowIds,
    /// Write `u64le original_row_id` followed by the item's WKB geometry.
    #[default]
    RowWkb,
}

impl ConvertPayload {
    fn needs_wkb(self) -> bool {
        matches!(self, ConvertPayload::RowWkb)
    }
}

/// Decode a [`ConvertPayload::RowIds`] payload blob.
pub fn decode_row_id_payload(payload: &[u8]) -> Option<u64> {
    let bytes: [u8; 8] = payload.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

/// Decode a [`ConvertPayload::RowWkb`] payload blob into the original
/// GeoParquet row id and WKB geometry bytes.
pub fn decode_row_wkb_payload(payload: &[u8]) -> Option<(u64, &[u8])> {
    if payload.len() < 8 {
        return None;
    }
    let mut row = [0u8; 8];
    row.copy_from_slice(&payload[..8]);
    Some((u64::from_le_bytes(row), &payload[8..]))
}

/// Convert a 2D GeoParquet source into `PSINDEX` bytes.
///
/// The bytes carry the index, the selected leaf-ordered payload, and the CRS.
/// By default each payload is `u64le original_row_id` followed by WKB geometry.
/// Query them with [`StreamIndex2D`](crate::StreamIndex2D) over any
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
    let want_wkb = opts.include_payload && opts.payload.needs_wkb();
    let scan = read::scan_2d(reader, want_wkb, opts.skip_null, &opts.read_opts())?;
    let builder = build::loaded_builder_2d(&scan.boxes, &opts.build);
    serialize_2d(
        builder,
        &scan.row_ids,
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
    let want_wkb = opts.include_payload && opts.payload.needs_wkb();
    let scan = read::scan_3d(reader, want_wkb, opts.skip_null, &opts.read_opts())?;
    let builder = build::loaded_builder_3d(&scan.boxes, &opts.build);
    serialize_3d(
        builder,
        &scan.row_ids,
        scan.wkb.as_deref(),
        scan.crs.as_deref(),
        &opts,
        out,
    )
}

fn serialize_2d(
    builder: Index2DBuilder,
    row_ids: &[u64],
    wkb: Option<&[Vec<u8>]>,
    crs: Option<&str>,
    opts: &ConvertOpts,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let payload = prepare_payload(row_ids, wkb, opts);
    if opts.compact_f32 {
        let index = builder.finish_f32()?;
        let mut s = index.serialize();
        if opts.interleaved && payload.is_some() {
            s = s.interleaved();
        }
        if let Some(c) = crs {
            s = s.crs(c);
        }
        match &payload {
            PreparedPayload::None => {}
            PreparedPayload::Variable {
                blobs,
                content_type,
            } => {
                s = s.payloads(blobs).content_type(content_type);
            }
            PreparedPayload::Fixed {
                flat,
                stride,
                content_type,
            } => {
                s = s.records(*stride, flat).content_type(content_type);
            }
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
        match &payload {
            PreparedPayload::None => {}
            PreparedPayload::Variable {
                blobs,
                content_type,
            } => {
                s = s.payloads(blobs).content_type(content_type);
            }
            PreparedPayload::Fixed {
                flat,
                stride,
                content_type,
            } => {
                s = s.records(*stride, flat).content_type(content_type);
            }
        }
        s.to_bytes_into(out)?;
    }
    Ok(())
}

fn serialize_3d(
    builder: Index3DBuilder,
    row_ids: &[u64],
    wkb: Option<&[Vec<u8>]>,
    crs: Option<&str>,
    opts: &ConvertOpts,
    out: &mut Vec<u8>,
) -> Result<(), GeoError> {
    let payload = prepare_payload(row_ids, wkb, opts);
    if opts.compact_f32 {
        let index = builder.finish_f32()?;
        let mut s = index.serialize();
        if opts.interleaved && payload.is_some() {
            s = s.interleaved();
        }
        if let Some(c) = crs {
            s = s.crs(c);
        }
        match &payload {
            PreparedPayload::None => {}
            PreparedPayload::Variable {
                blobs,
                content_type,
            } => {
                s = s.payloads(blobs).content_type(content_type);
            }
            PreparedPayload::Fixed {
                flat,
                stride,
                content_type,
            } => {
                s = s.records(*stride, flat).content_type(content_type);
            }
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
        match &payload {
            PreparedPayload::None => {}
            PreparedPayload::Variable {
                blobs,
                content_type,
            } => {
                s = s.payloads(blobs).content_type(content_type);
            }
            PreparedPayload::Fixed {
                flat,
                stride,
                content_type,
            } => {
                s = s.records(*stride, flat).content_type(content_type);
            }
        }
        s.to_bytes_into(out)?;
    }
    Ok(())
}

enum PreparedPayload {
    None,
    Variable {
        blobs: Vec<Vec<u8>>,
        content_type: &'static str,
    },
    Fixed {
        flat: Vec<u8>,
        stride: usize,
        content_type: &'static str,
    },
}

impl PreparedPayload {
    fn is_some(&self) -> bool {
        !matches!(self, PreparedPayload::None)
    }
}

fn prepare_payload(
    row_ids: &[u64],
    wkb: Option<&[Vec<u8>]>,
    opts: &ConvertOpts,
) -> PreparedPayload {
    if !opts.include_payload {
        return PreparedPayload::None;
    }
    match opts.payload {
        ConvertPayload::None => PreparedPayload::None,
        ConvertPayload::RowIds => {
            let mut flat = Vec::with_capacity(row_ids.len() * 8);
            for row in row_ids {
                flat.extend_from_slice(&row.to_le_bytes());
            }
            PreparedPayload::Fixed {
                flat,
                stride: 8,
                content_type: ROW_ID_CONTENT_TYPE,
            }
        }
        ConvertPayload::RowWkb => {
            let wkb = wkb.expect("WKB scan requested for RowWkb payload");
            let mut blobs = Vec::with_capacity(wkb.len());
            for (row, geom) in row_ids.iter().zip(wkb) {
                let mut blob = Vec::with_capacity(8 + geom.len());
                blob.extend_from_slice(&row.to_le_bytes());
                blob.extend_from_slice(geom);
                blobs.push(blob);
            }
            PreparedPayload::Variable {
                blobs,
                content_type: ROW_WKB_CONTENT_TYPE,
            }
        }
    }
}
