//! The one geo-specific primitive: read per-row bounding boxes (and, for the
//! converter, the raw WKB geometry) from a GeoParquet source, in file row order.
//!
//! Boxes come from the GeoParquet 1.1 *bbox covering* column when present (cheap,
//! no geometry decode); otherwise each geometry's envelope is computed from its
//! WKB. Apache Parquet `GEOMETRY` / `GEOGRAPHY` logical types are WKB by
//! definition; GeoParquet-native GeoArrow encodings still need a covering column
//! unless the caller only needs row ids.

use std::fmt;

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
use parquet::basic::{EdgeInterpolationAlgorithm, LogicalType, Type as ParquetPhysicalType};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::ChunkReader;

use crate::{GeoError, GeometryMetadataSource, ReadOpts};

/// What the GeoParquet `geo` metadata tells us about the primary column.
struct GeoInfo {
    geometry_column: String,
    encoding: GeometryEncoding,
    crs: Option<String>,
    covering: Option<GeoParquetBboxCovering>,
    dim: DimHint,
    version: String,
    bounds: Option<Vec<f64>>,
    metadata_source: GeometryMetadataSource,
}

#[derive(Debug, Clone)]
enum GeometryEncoding {
    GeoParquet(GeoParquetColumnEncoding),
    ParquetGeometry,
    ParquetGeography {
        algorithm: Option<EdgeInterpolationAlgorithm>,
    },
}

impl GeometryEncoding {
    fn is_wkb(&self) -> bool {
        matches!(
            self,
            GeometryEncoding::GeoParquet(GeoParquetColumnEncoding::WKB)
                | GeometryEncoding::ParquetGeometry
                | GeometryEncoding::ParquetGeography { .. }
        )
    }
}

impl fmt::Display for GeometryEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometryEncoding::GeoParquet(enc) => write!(f, "{enc}"),
            GeometryEncoding::ParquetGeometry => f.write_str("GEOMETRY"),
            GeometryEncoding::ParquetGeography { algorithm: None } => f.write_str("GEOGRAPHY"),
            GeometryEncoding::ParquetGeography {
                algorithm: Some(algorithm),
            } => write!(f, "GEOGRAPHY({algorithm})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DimHint {
    Two,
    Three,
    Unknown,
}

#[derive(Debug, Clone)]
struct NativeGeoColumn {
    name: String,
    column_index: usize,
    encoding: GeometryEncoding,
    crs: Option<String>,
}

fn geo_info(meta: &GeoParquetMetadata, opts: &ReadOpts) -> Result<GeoInfo, GeoError> {
    let name = opts
        .geometry_column
        .as_ref()
        .unwrap_or(&meta.primary_column);
    let col = meta.columns.get(name).ok_or_else(|| {
        opts.geometry_column
            .as_ref()
            .map(|name| GeoError::GeometryColumnNotFound(name.clone()))
            .unwrap_or(GeoError::NoGeometryColumn)
    })?;
    let covering = col.covering.as_ref().map(|c| c.bbox.clone());
    let is_3d = covering.as_ref().is_some_and(|c| c.zmin.is_some())
        || col
            .geometry_types
            .iter()
            .any(|t| matches!(t.dimension(), Dimension::XYZ | Dimension::XYZM));
    Ok(GeoInfo {
        geometry_column: name.clone(),
        encoding: GeometryEncoding::GeoParquet(col.encoding),
        // PROJJSON object serialized back to a compact string for the index's
        // `META.crs` chunk.
        crs: col.crs.as_ref().map(|v| v.to_string()),
        covering,
        dim: if is_3d { DimHint::Three } else { DimHint::Two },
        version: meta.version.clone(),
        bounds: col.bbox.clone(),
        metadata_source: GeometryMetadataSource::GeoParquet,
    })
}

fn native_info(
    meta: &ParquetMetaData,
    opts: &ReadOpts,
    candidates: &[NativeGeoColumn],
) -> Result<GeoInfo, GeoError> {
    let selected = if let Some(name) = &opts.geometry_column {
        candidates
            .iter()
            .find(|c| &c.name == name)
            .ok_or_else(|| GeoError::GeometryColumnNotFound(name.clone()))?
    } else {
        match candidates {
            [] => return Err(GeoError::NoGeometryColumn),
            [one] => one,
            many => {
                return Err(GeoError::AmbiguousGeometryColumn {
                    columns: many.iter().map(|c| c.name.clone()).collect(),
                });
            }
        }
    };

    Ok(GeoInfo {
        geometry_column: selected.name.clone(),
        encoding: selected.encoding.clone(),
        crs: selected.crs.clone(),
        covering: None,
        dim: native_dim_hint(meta, selected.column_index),
        version: "parquet-geospatial".to_string(),
        bounds: None,
        metadata_source: GeometryMetadataSource::ParquetGeospatial,
    })
}

fn native_geo_columns(meta: &ParquetMetaData) -> Vec<NativeGeoColumn> {
    meta.file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(column_index, col)| {
            let parts = col.path().parts();
            if parts.len() != 1
                || col.max_rep_level() != 0
                || col.physical_type() != ParquetPhysicalType::BYTE_ARRAY
            {
                return None;
            }

            match col.logical_type_ref()? {
                LogicalType::Geometry { crs } => Some(NativeGeoColumn {
                    name: parts[0].clone(),
                    column_index,
                    encoding: GeometryEncoding::ParquetGeometry,
                    crs: crs.clone(),
                }),
                LogicalType::Geography { crs, algorithm } => Some(NativeGeoColumn {
                    name: parts[0].clone(),
                    column_index,
                    encoding: GeometryEncoding::ParquetGeography {
                        algorithm: *algorithm,
                    },
                    crs: crs.clone(),
                }),
                _ => None,
            }
        })
        .collect()
}

fn native_dim_hint(meta: &ParquetMetaData, column_index: usize) -> DimHint {
    let mut saw_stats = false;
    for row_group in meta.row_groups() {
        let Some(types) = row_group
            .column(column_index)
            .geo_statistics()
            .and_then(|stats| stats.geospatial_types())
        else {
            return DimHint::Unknown;
        };

        saw_stats = true;
        if types.iter().any(|&ty| wkb_type_has_z(ty)) {
            return DimHint::Three;
        }
    }

    if saw_stats {
        DimHint::Two
    } else {
        DimHint::Unknown
    }
}

fn wkb_type_has_z(ty: i32) -> bool {
    (1000..2000).contains(&ty) || (3000..4000).contains(&ty)
}

/// Read just the GeoParquet metadata and row count, without building a batch
/// reader. Returns the builder so a caller can go on to read batches.
fn read_meta<R: ChunkReader + 'static>(
    reader: R,
    opts: &ReadOpts,
) -> Result<(GeoInfo, usize, ParquetRecordBatchReaderBuilder<R>), GeoError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(reader)?;
    let meta = builder.metadata();
    let file_meta = meta.file_metadata();
    let native = native_geo_columns(meta);
    let geo_meta = GeoParquetMetadata::from_parquet_meta(file_meta);
    let explicit_geo_column = opts.geometry_column.as_ref();
    let info = match geo_meta {
        Some(Ok(gpq))
            if explicit_geo_column.is_none()
                || gpq.columns.contains_key(explicit_geo_column.unwrap()) =>
        {
            geo_info(&gpq, opts)?
        }
        Some(Ok(gpq)) => native_info(meta, opts, &native).or_else(|_| geo_info(&gpq, opts))?,
        Some(Err(e)) => return Err(GeoError::Metadata(e.to_string())),
        None => native_info(meta, opts, &native)?,
    };
    let total = file_meta.num_rows().max(0) as usize;
    Ok((info, total, builder))
}

fn open<R: ChunkReader + 'static>(
    reader: R,
    opts: &ReadOpts,
) -> Result<
    (
        GeoInfo,
        usize,
        parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    ),
    GeoError,
> {
    let (info, total, builder) = read_meta(reader, opts)?;
    let batches = builder.build()?;
    Ok((info, total, batches))
}

/// A summary of a GeoParquet or native Parquet geospatial source's selected
/// geometry column, from [`inspect`].
#[derive(Debug, Clone)]
pub struct GeoParquetInfo {
    /// Metadata version/source marker. GeoParquet files report their declared
    /// spec version, e.g. `"1.1.0"`; native Parquet geospatial files report
    /// `"parquet-geospatial"`.
    pub version: String,
    /// Name of the selected geometry column.
    pub geometry_column: String,
    /// Where the selected geometry metadata came from.
    pub metadata_source: GeometryMetadataSource,
    /// `2` or `3`.
    pub dims: u8,
    /// Geometry encoding, e.g. `"WKB"`, `"point"`, `"GEOMETRY"`, or
    /// `"GEOGRAPHY(SPHERICAL)"`.
    pub encoding: String,
    /// Column CRS, if the file declares one. GeoParquet reports compact
    /// PROJJSON; native Parquet geospatial reports the logical type's `crs`
    /// string verbatim.
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

/// Inspect a source's selected geometry metadata.
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
    inspect_with_opts(reader, ReadOpts::default())
}

/// Inspect a source's selected geometry metadata with explicit reader options.
///
/// GeoParquet metadata provides dimensionality without reading rows. Native
/// Parquet geospatial files that lack geospatial type statistics may require a
/// WKB pass to infer whether any geometry carries Z coordinates.
pub fn inspect_with_opts<R: ChunkReader + 'static>(
    reader: R,
    opts: ReadOpts,
) -> Result<GeoParquetInfo, GeoError> {
    let (mut info, total, builder) = read_meta(reader, &opts)?;
    if info.dim == DimHint::Unknown {
        let batches = builder.build()?;
        info.dim = infer_dim_from_batches(batches, &info.geometry_column)?;
    }
    Ok(GeoParquetInfo {
        version: info.version,
        geometry_column: info.geometry_column,
        metadata_source: info.metadata_source,
        dims: if info.dim == DimHint::Three { 3 } else { 2 },
        encoding: info.encoding.to_string(),
        crs: info.crs,
        has_covering: info.covering.is_some(),
        bounds: info.bounds,
        num_rows: total as u64,
    })
}

/// Report whether a source's selected geometry column is 2D or 3D.
pub fn detect_dims<R: ChunkReader + 'static>(reader: R) -> Result<u8, GeoError> {
    Ok(inspect(reader)?.dims)
}

/// Report whether a source's selected geometry column is 2D or 3D with explicit
/// reader options.
pub fn detect_dims_with_opts<R: ChunkReader + 'static>(
    reader: R,
    opts: ReadOpts,
) -> Result<u8, GeoError> {
    Ok(inspect_with_opts(reader, opts)?.dims)
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
    read_bboxes_2d_with_opts(reader, ReadOpts::default())
}

/// Read every row's 2D bounding box with explicit reader options.
pub fn read_bboxes_2d_with_opts<R: ChunkReader + 'static>(
    reader: R,
    opts: ReadOpts,
) -> Result<Vec<Box2D>, GeoError> {
    Ok(scan_2d(reader, false, false, &opts)?.boxes)
}

/// Read every row's 3D bounding box, in file row order.
pub fn read_bboxes_3d<R: ChunkReader + 'static>(reader: R) -> Result<Vec<Box3D>, GeoError> {
    read_bboxes_3d_with_opts(reader, ReadOpts::default())
}

/// Read every row's 3D bounding box with explicit reader options.
pub fn read_bboxes_3d_with_opts<R: ChunkReader + 'static>(
    reader: R,
    opts: ReadOpts,
) -> Result<Vec<Box3D>, GeoError> {
    Ok(scan_3d(reader, false, false, &opts)?.boxes)
}

/// Result of a 2D scan: boxes (always) plus, when requested, the per-row WKB
/// geometry and the column CRS for the converter.
pub(crate) struct Scan2D {
    pub boxes: Vec<Box2D>,
    pub row_ids: Vec<u64>,
    pub wkb: Option<Vec<Vec<u8>>>,
    pub crs: Option<String>,
}

pub(crate) struct Scan3D {
    pub boxes: Vec<Box3D>,
    pub row_ids: Vec<u64>,
    pub wkb: Option<Vec<Vec<u8>>>,
    pub crs: Option<String>,
}

pub(crate) fn scan_2d<R: ChunkReader + 'static>(
    reader: R,
    want_wkb: bool,
    skip_null: bool,
    opts: &ReadOpts,
) -> Result<Scan2D, GeoError> {
    let (info, total, batches) = open(reader, opts)?;
    if info.dim == DimHint::Three {
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
    let mut row_ids = Vec::with_capacity(total);
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
                } else if let Some((b, has_z)) = wkb_bounds_2d(geom.value(i)) {
                    if info.dim == DimHint::Unknown && has_z {
                        return Err(GeoError::DimMismatch {
                            expected: 2,
                            found: 3,
                        });
                    }
                    Some(Box2D::new(b[0], b[1], b[2], b[3]))
                } else {
                    None
                }
            };
            match bx {
                Some(b) => {
                    boxes.push(b);
                    row_ids.push((row_base + i) as u64);
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
        row_ids,
        wkb,
        crs: info.crs,
    })
}

pub(crate) fn scan_3d<R: ChunkReader + 'static>(
    reader: R,
    want_wkb: bool,
    skip_null: bool,
    opts: &ReadOpts,
) -> Result<Scan3D, GeoError> {
    let (info, total, batches) = open(reader, opts)?;
    if info.dim == DimHint::Two {
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
    let mut row_ids = Vec::with_capacity(total);
    let mut wkb = want_wkb.then(|| Vec::with_capacity(total));
    let mut row_base = 0usize;
    let mut saw_z = info.dim == DimHint::Three;

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
                } else if let Some((b, has_z)) = wkb_bounds_3d(geom.value(i)) {
                    saw_z |= has_z;
                    Some(Box3D::new(b[0], b[1], b[2], b[3], b[4], b[5]))
                } else {
                    None
                }
            };
            match bx {
                Some(b) => {
                    boxes.push(b);
                    row_ids.push((row_base + i) as u64);
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

    if info.dim == DimHint::Unknown && !saw_z {
        return Err(GeoError::DimMismatch {
            expected: 3,
            found: 2,
        });
    }

    Ok(Scan3D {
        boxes,
        row_ids,
        wkb,
        crs: info.crs,
    })
}

/// Require the `WKB` encoding only when the geometry column will actually be
/// decoded (no covering boxes, or the caller wants the WKB payload).
fn require_wkb_if(info: &GeoInfo, needed: bool) -> Result<(), GeoError> {
    if !needed || info.encoding.is_wkb() {
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

fn infer_dim_from_batches(
    batches: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    geometry_column: &str,
) -> Result<DimHint, GeoError> {
    for batch in batches {
        let batch = batch?;
        let geom = binary_column(&batch, geometry_column)?;
        for i in 0..batch.num_rows() {
            if geom.is_null(i) {
                continue;
            }
            let Some(bounds) = wkb_bounds(geom.value(i), true) else {
                continue;
            };
            if bounds.has_z {
                return Ok(DimHint::Three);
            }
        }
    }

    Ok(DimHint::Two)
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
    has_z: bool,
    any: bool,
}

impl Bounds {
    fn new(three_d: bool) -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
            three_d,
            has_z: false,
            any: false,
        }
    }
    fn add(&mut self, x: f64, y: f64, z: Option<f64>) {
        self.min[0] = self.min[0].min(x);
        self.min[1] = self.min[1].min(y);
        self.max[0] = self.max[0].max(x);
        self.max[1] = self.max[1].max(y);
        if let Some(z) = z {
            self.min[2] = self.min[2].min(z);
            self.max[2] = self.max[2].max(z);
            self.has_z = true;
        } else if self.three_d {
            self.min[2] = self.min[2].min(0.0);
            self.max[2] = self.max[2].max(0.0);
        }
        self.any = true;
    }
}

impl GeomProcessor for Bounds {
    fn multi_dim(&self) -> bool {
        self.three_d
    }
    fn xy(&mut self, x: f64, y: f64, _idx: usize) -> geozero::error::Result<()> {
        self.add(x, y, None);
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
        self.add(x, y, z);
        Ok(())
    }
}

fn wkb_bounds(bytes: &[u8], three_d: bool) -> Option<Bounds> {
    let mut b = Bounds::new(three_d);
    let mut cur = std::io::Cursor::new(bytes);
    geozero::wkb::process_wkb_geom(&mut cur, &mut b).ok()?;
    b.any.then_some(b)
}

fn wkb_bounds_2d(bytes: &[u8]) -> Option<([f64; 4], bool)> {
    let b = wkb_bounds(bytes, true)?;
    Some(([b.min[0], b.min[1], b.max[0], b.max[1]], b.has_z))
}

fn wkb_bounds_3d(bytes: &[u8]) -> Option<([f64; 6], bool)> {
    let b = wkb_bounds(bytes, true)?;
    Some((
        [b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]],
        b.has_z,
    ))
}
