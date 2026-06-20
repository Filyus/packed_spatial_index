//! The one geo-specific primitive: read per-row bounding boxes (and, for the
//! converter, the raw WKB geometry) from a GeoParquet source, in file row order.
//!
//! Boxes come from the GeoParquet 1.1 *bbox covering* column when present (cheap,
//! no geometry decode); otherwise each geometry's envelope is computed from its
//! WKB. Only the `WKB` geometry encoding is supported in this version; native
//! geoarrow encodings return [`GeoError::UnsupportedEncoding`].

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, LargeBinaryArray,
    StructArray,
};
use arrow::record_batch::RecordBatch;
use geoarrow_schema::Dimension;
use geoparquet::metadata::{GeoParquetBboxCovering, GeoParquetColumnEncoding, GeoParquetMetadata};
use geozero::GeomProcessor;
use packed_spatial_index::{Box2D, Box3D};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::ChunkReader;

use crate::GeoError;

/// What the GeoParquet `geo` metadata tells us about the primary column.
struct GeoInfo {
    geometry_column: String,
    encoding: GeoParquetColumnEncoding,
    crs: Option<String>,
    covering: Option<GeoParquetBboxCovering>,
    is_3d: bool,
    version: String,
    bounds: Option<Vec<f64>>,
}

fn geo_info(meta: &GeoParquetMetadata) -> Result<GeoInfo, GeoError> {
    let name = &meta.primary_column;
    let col = meta.columns.get(name).ok_or(GeoError::NoGeometryColumn)?;
    let covering = col.covering.as_ref().map(|c| c.bbox.clone());
    let is_3d = covering.as_ref().is_some_and(|c| c.zmin.is_some())
        || col
            .geometry_types
            .iter()
            .any(|t| matches!(t.dimension(), Dimension::XYZ | Dimension::XYZM));
    Ok(GeoInfo {
        geometry_column: name.clone(),
        encoding: col.encoding,
        // PROJJSON object serialized back to a compact string for the index's
        // `META.crs` chunk.
        crs: col.crs.as_ref().map(|v| v.to_string()),
        covering,
        is_3d,
        version: meta.version.clone(),
        bounds: col.bbox.clone(),
    })
}

/// Read just the GeoParquet metadata and row count, without building a batch
/// reader. Returns the builder so a caller can go on to read batches.
fn read_meta<R: ChunkReader + 'static>(
    reader: R,
) -> Result<(GeoInfo, usize, ParquetRecordBatchReaderBuilder<R>), GeoError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(reader)?;
    let file_meta = builder.metadata().file_metadata();
    let gpq = GeoParquetMetadata::from_parquet_meta(file_meta)
        .ok_or_else(|| GeoError::Metadata("file has no `geo` metadata".to_string()))?
        .map_err(|e| GeoError::Metadata(e.to_string()))?;
    let info = geo_info(&gpq)?;
    let total = file_meta.num_rows().max(0) as usize;
    Ok((info, total, builder))
}

fn open<R: ChunkReader + 'static>(
    reader: R,
) -> Result<
    (
        GeoInfo,
        usize,
        parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    ),
    GeoError,
> {
    let (info, total, builder) = read_meta(reader)?;
    let batches = builder.build()?;
    Ok((info, total, batches))
}

/// A summary of a GeoParquet source's primary geometry column, from [`inspect`].
#[derive(Debug, Clone)]
pub struct GeoParquetInfo {
    /// GeoParquet spec version the file declares, e.g. `"1.1.0"`.
    pub version: String,
    /// Name of the primary geometry column.
    pub geometry_column: String,
    /// `2` or `3`.
    pub dims: u8,
    /// Geometry encoding, e.g. `"WKB"` or `"point"`.
    pub encoding: String,
    /// Column CRS as a PROJJSON string, if the file declares one.
    pub crs: Option<String>,
    /// Whether a per-row bbox covering column is present.
    pub has_covering: bool,
    /// The column's overall extent if the file records one: `[xmin, ymin, xmax,
    /// ymax]` (2D) or `[xmin, ymin, zmin, xmax, ymax, zmax]` (3D). Handy for an
    /// initial viewport.
    pub bounds: Option<Vec<f64>>,
    /// Number of rows in the file.
    pub num_rows: u64,
}

/// Inspect a GeoParquet source's geometry metadata without reading any rows.
///
/// # Examples
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::inspect;
///
/// let info = inspect(File::open("cities.parquet")?)?;
/// println!("{}D, {} rows, encoding {}", info.dims, info.num_rows, info.encoding);
/// if let Some(bounds) = &info.bounds {
///     println!("extent: {bounds:?}");
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn inspect<R: ChunkReader + 'static>(reader: R) -> Result<GeoParquetInfo, GeoError> {
    let (info, total, _builder) = read_meta(reader)?;
    Ok(GeoParquetInfo {
        version: info.version,
        geometry_column: info.geometry_column,
        dims: if info.is_3d { 3 } else { 2 },
        encoding: info.encoding.to_string(),
        crs: info.crs,
        has_covering: info.covering.is_some(),
        bounds: info.bounds,
        num_rows: total as u64,
    })
}

/// Report whether a GeoParquet source's primary geometry column is 2D or 3D.
pub fn detect_dims<R: ChunkReader + 'static>(reader: R) -> Result<u8, GeoError> {
    Ok(inspect(reader)?.dims)
}

/// Read every row's 2D bounding box, in file row order. Item `i` corresponds to
/// GeoParquet row `i`.
///
/// # Examples
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::read_bboxes_2d;
///
/// let boxes = read_bboxes_2d(File::open("cities.parquet")?)?;
/// println!("{} bounding boxes", boxes.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn read_bboxes_2d<R: ChunkReader + 'static>(reader: R) -> Result<Vec<Box2D>, GeoError> {
    Ok(scan_2d(reader, false, false)?.boxes)
}

/// Read every row's 3D bounding box, in file row order.
pub fn read_bboxes_3d<R: ChunkReader + 'static>(reader: R) -> Result<Vec<Box3D>, GeoError> {
    Ok(scan_3d(reader, false, false)?.boxes)
}

/// Result of a 2D scan: boxes (always) plus, when requested, the per-row WKB
/// geometry and the column CRS for the converter.
pub(crate) struct Scan2D {
    pub boxes: Vec<Box2D>,
    pub wkb: Option<Vec<Vec<u8>>>,
    pub crs: Option<String>,
}

pub(crate) struct Scan3D {
    pub boxes: Vec<Box3D>,
    pub wkb: Option<Vec<Vec<u8>>>,
    pub crs: Option<String>,
}

pub(crate) fn scan_2d<R: ChunkReader + 'static>(
    reader: R,
    want_wkb: bool,
    skip_null: bool,
) -> Result<Scan2D, GeoError> {
    let (info, total, batches) = open(reader)?;
    if info.is_3d {
        return Err(GeoError::DimMismatch {
            expected: 2,
            found: 3,
        });
    }
    // The geometry column is only decoded when we have no covering boxes (WKB
    // envelope fallback) or when the caller wants the WKB payload. With a covering
    // column and no payload requested, any encoding is fine.
    let need_wkb = want_wkb || info.covering.is_none();
    require_wkb_if(&info, need_wkb)?;

    let mut boxes = Vec::with_capacity(total);
    let mut wkb = want_wkb.then(|| Vec::with_capacity(total));
    let mut row_base = 0usize;

    for batch in batches {
        let batch = batch?;
        let n = batch.num_rows();
        let geom_bin = need_wkb
            .then(|| binary_column(&batch, &info.geometry_column))
            .transpose()?;

        // Covering boxes + the geometry column (for null detection), read once.
        let cov = info
            .covering
            .as_ref()
            .map(|cov| {
                Ok::<_, GeoError>((
                    batch
                        .column_by_name(&info.geometry_column)
                        .ok_or(GeoError::NoGeometryColumn)?,
                    f64_path(&batch, &cov.xmin)?,
                    f64_path(&batch, &cov.ymin)?,
                    f64_path(&batch, &cov.xmax)?,
                    f64_path(&batch, &cov.ymax)?,
                ))
            })
            .transpose()?;

        for i in 0..n {
            let bx = if let Some((geom, xmin, ymin, xmax, ymax)) = &cov {
                (!geom.is_null(i)).then(|| Box2D::new(xmin[i], ymin[i], xmax[i], ymax[i]))
            } else {
                let geom = geom_bin.as_ref().expect("need_wkb when no covering");
                if geom.is_null(i) {
                    None
                } else {
                    wkb_bounds_2d(geom.value(i)).map(|b| Box2D::new(b[0], b[1], b[2], b[3]))
                }
            };
            match bx {
                Some(b) => {
                    boxes.push(b);
                    if let Some(w) = wkb.as_mut() {
                        let geom = geom_bin.as_ref().expect("want_wkb implies binary column");
                        w.push(geom.value(i).to_vec());
                    }
                }
                None if skip_null => continue,
                None => return Err(GeoError::NullGeometry { row: row_base + i }),
            }
        }
        row_base += n;
    }

    Ok(Scan2D {
        boxes,
        wkb,
        crs: info.crs,
    })
}

pub(crate) fn scan_3d<R: ChunkReader + 'static>(
    reader: R,
    want_wkb: bool,
    skip_null: bool,
) -> Result<Scan3D, GeoError> {
    let (info, total, batches) = open(reader)?;
    if !info.is_3d {
        return Err(GeoError::DimMismatch {
            expected: 3,
            found: 2,
        });
    }
    // A 3D covering needs both zmin and zmax; otherwise fall back to WKB.
    let cov_3d = info
        .covering
        .as_ref()
        .filter(|c| c.zmin.is_some() && c.zmax.is_some());
    let need_wkb = want_wkb || cov_3d.is_none();
    require_wkb_if(&info, need_wkb)?;

    let mut boxes = Vec::with_capacity(total);
    let mut wkb = want_wkb.then(|| Vec::with_capacity(total));
    let mut row_base = 0usize;

    for batch in batches {
        let batch = batch?;
        let n = batch.num_rows();
        let geom_bin = need_wkb
            .then(|| binary_column(&batch, &info.geometry_column))
            .transpose()?;

        let cov = cov_3d
            .map(|cov| {
                Ok::<_, GeoError>((
                    batch
                        .column_by_name(&info.geometry_column)
                        .ok_or(GeoError::NoGeometryColumn)?,
                    f64_path(&batch, &cov.xmin)?,
                    f64_path(&batch, &cov.ymin)?,
                    f64_path(&batch, cov.zmin.as_ref().unwrap())?,
                    f64_path(&batch, &cov.xmax)?,
                    f64_path(&batch, &cov.ymax)?,
                    f64_path(&batch, cov.zmax.as_ref().unwrap())?,
                ))
            })
            .transpose()?;

        for i in 0..n {
            let bx = if let Some((geom, xmin, ymin, zmin, xmax, ymax, zmax)) = &cov {
                (!geom.is_null(i))
                    .then(|| Box3D::new(xmin[i], ymin[i], zmin[i], xmax[i], ymax[i], zmax[i]))
            } else {
                let geom = geom_bin.as_ref().expect("need_wkb when no 3D covering");
                if geom.is_null(i) {
                    None
                } else {
                    wkb_bounds_3d(geom.value(i))
                        .map(|b| Box3D::new(b[0], b[1], b[2], b[3], b[4], b[5]))
                }
            };
            match bx {
                Some(b) => {
                    boxes.push(b);
                    if let Some(w) = wkb.as_mut() {
                        let geom = geom_bin.as_ref().expect("want_wkb implies binary column");
                        w.push(geom.value(i).to_vec());
                    }
                }
                None if skip_null => continue,
                None => return Err(GeoError::NullGeometry { row: row_base + i }),
            }
        }
        row_base += n;
    }

    Ok(Scan3D {
        boxes,
        wkb,
        crs: info.crs,
    })
}

/// Require the `WKB` encoding only when the geometry column will actually be
/// decoded (no covering boxes, or the caller wants the WKB payload).
fn require_wkb_if(info: &GeoInfo, needed: bool) -> Result<(), GeoError> {
    if !needed || info.encoding == GeoParquetColumnEncoding::WKB {
        Ok(())
    } else {
        Err(GeoError::UnsupportedEncoding(info.encoding.to_string()))
    }
}

/// A binary geometry column: 32-bit offsets (`BinaryArray`), 64-bit offsets
/// (`LargeBinaryArray`), or the view layout (`BinaryViewArray`).
enum WkbCol<'a> {
    Bin(&'a BinaryArray),
    Large(&'a LargeBinaryArray),
    View(&'a BinaryViewArray),
}

impl WkbCol<'_> {
    fn is_null(&self, i: usize) -> bool {
        match self {
            WkbCol::Bin(a) => a.is_null(i),
            WkbCol::Large(a) => a.is_null(i),
            WkbCol::View(a) => a.is_null(i),
        }
    }
    fn value(&self, i: usize) -> &[u8] {
        match self {
            WkbCol::Bin(a) => a.value(i),
            WkbCol::Large(a) => a.value(i),
            WkbCol::View(a) => a.value(i),
        }
    }
}

fn binary_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<WkbCol<'a>, GeoError> {
    let arr = batch
        .column_by_name(name)
        .ok_or(GeoError::NoGeometryColumn)?;
    if let Some(a) = arr.as_any().downcast_ref::<BinaryArray>() {
        Ok(WkbCol::Bin(a))
    } else if let Some(a) = arr.as_any().downcast_ref::<LargeBinaryArray>() {
        Ok(WkbCol::Large(a))
    } else if let Some(a) = arr.as_any().downcast_ref::<BinaryViewArray>() {
        Ok(WkbCol::View(a))
    } else {
        Err(GeoError::UnsupportedEncoding(format!(
            "geometry column `{name}` is not binary WKB ({:?})",
            arr.data_type()
        )))
    }
}

/// Resolve a GeoParquet schema path (`["bbox", "xmin"]`) to a leaf array and read
/// it as `f64`, accepting either `Float64` or `Float32` storage.
fn f64_path(batch: &RecordBatch, path: &[String]) -> Result<Vec<f64>, GeoError> {
    let arr = descend(batch, path)?;
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        Ok((0..a.len()).map(|i| a.value(i)).collect())
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        Ok((0..a.len()).map(|i| a.value(i) as f64).collect())
    } else {
        Err(GeoError::Metadata(format!(
            "bbox covering path {path:?} is not a float column ({:?})",
            arr.data_type()
        )))
    }
}

fn descend<'a>(batch: &'a RecordBatch, path: &[String]) -> Result<&'a ArrayRef, GeoError> {
    let first = path
        .first()
        .ok_or_else(|| GeoError::Metadata("empty bbox covering path".to_string()))?;
    let mut cur = batch
        .column_by_name(first)
        .ok_or_else(|| GeoError::Metadata(format!("bbox covering column `{first}` not found")))?;
    for field in &path[1..] {
        let st = cur.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
            GeoError::Metadata(format!(
                "bbox covering path segment `{field}` is not a struct"
            ))
        })?;
        cur = st.column_by_name(field).ok_or_else(|| {
            GeoError::Metadata(format!("bbox covering field `{field}` not found"))
        })?;
    }
    Ok(cur)
}

// --- WKB envelope fallback (geozero) ----------------------------------------

struct Bounds {
    min: [f64; 3],
    max: [f64; 3],
    three_d: bool,
    any: bool,
}

impl Bounds {
    fn new(three_d: bool) -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
            three_d,
            any: false,
        }
    }
    fn add(&mut self, x: f64, y: f64, z: f64) {
        self.min[0] = self.min[0].min(x);
        self.min[1] = self.min[1].min(y);
        self.min[2] = self.min[2].min(z);
        self.max[0] = self.max[0].max(x);
        self.max[1] = self.max[1].max(y);
        self.max[2] = self.max[2].max(z);
        self.any = true;
    }
}

impl GeomProcessor for Bounds {
    fn multi_dim(&self) -> bool {
        self.three_d
    }
    fn xy(&mut self, x: f64, y: f64, _idx: usize) -> geozero::error::Result<()> {
        self.add(x, y, 0.0);
        Ok(())
    }
    fn coordinate(
        &mut self,
        x: f64,
        y: f64,
        z: Option<f64>,
        _m: Option<f64>,
        _t: Option<f64>,
        _tm: Option<u64>,
        _idx: usize,
    ) -> geozero::error::Result<()> {
        self.add(x, y, z.unwrap_or(0.0));
        Ok(())
    }
}

fn wkb_bounds(bytes: &[u8], three_d: bool) -> Option<Bounds> {
    let mut b = Bounds::new(three_d);
    let mut cur = std::io::Cursor::new(bytes);
    geozero::wkb::process_wkb_geom(&mut cur, &mut b).ok()?;
    b.any.then_some(b)
}

fn wkb_bounds_2d(bytes: &[u8]) -> Option<[f64; 4]> {
    let b = wkb_bounds(bytes, false)?;
    Some([b.min[0], b.min[1], b.max[0], b.max[1]])
}

fn wkb_bounds_3d(bytes: &[u8]) -> Option<[f64; 6]> {
    let b = wkb_bounds(bytes, true)?;
    Some([b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]])
}
