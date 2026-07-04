use parquet::basic::{LogicalType, Type as ParquetPhysicalType};
use parquet::file::metadata::ParquetMetaData;
use serde::{Deserialize, Serialize};

use crate::{CoordinateDims, GeoDiscovery, GeometryProfile, SelectionStatus};

/// Structured compatibility validation report for a dataset.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::{open_geoparquet, ValidateRequest, ValidationSeverity};
///
/// let mut dataset = open_geoparquet(File::open("cities.parquet")?)?;
/// let report = dataset.validate(ValidateRequest::default())?;
/// let warnings = report
///     .issues
///     .iter()
///     .filter(|issue| issue.severity == ValidationSeverity::Warning)
///     .count();
/// println!("validation ok: {}, warnings: {warnings}", report.ok);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Metadata-only geometry discovery.
    pub discovery: GeoDiscovery,
    /// Result of resolving the requested geometry selector.
    pub selected: SelectionStatus,
    /// Profile of the selected geometry column, if one could be resolved.
    pub profile: Option<GeometryProfile>,
    /// Native Parquet row-group geospatial statistics diagnostics.
    pub native_stats: Vec<NativeGeospatialStatsReport>,
    /// Validation issues discovered for the requested operation.
    pub issues: Vec<ValidationIssue>,
    /// True when no issue has `Error` severity.
    pub ok: bool,
}

/// One validation issue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue severity.
    pub severity: ValidationSeverity,
    /// Stable issue code.
    pub code: ValidationCode,
    /// Geometry column associated with the issue, if applicable.
    pub column: Option<String>,
    /// Human-readable explanation.
    pub message: String,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSeverity {
    /// Informational note.
    Info,
    /// Non-fatal compatibility or accuracy warning.
    Warning,
    /// Requested operation is not expected to work.
    Error,
}

/// Stable validation issue code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCode {
    /// No usable geometry columns were found.
    NoGeometryColumns,
    /// Several geometry columns exist and no safe default was selected.
    AmbiguousGeometryColumn,
    /// Requested geometry column was not found or is not usable.
    GeometryColumnNotFound,
    /// Geometry encoding is unsupported for validation or indexing.
    UnsupportedEncoding,
    /// The selected column cannot produce feature envelopes.
    CannotScanEnvelopes,
    /// The selected column cannot emit the requested payload.
    CannotEmitPayload,
    /// Dimensions are unknown from metadata/statistics.
    UnknownDimensions,
    /// Native Parquet geospatial statistics are missing for a native column.
    MissingNativeGeoStats,
    /// Native or scanned bounds cross the antimeridian.
    AntimeridianWrap,
    /// Geography/non-planar data is indexed as coordinate AABBs.
    GeographyCoordinateAabb,
    /// Exact row scan failed.
    ExactScanFailed,
    /// Requested `FeatureJson` property column is missing.
    ProjectedPropertyMissing,
}

/// Native Parquet geospatial statistics summary for one column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeGeospatialStatsReport {
    /// Column name.
    pub column: String,
    /// Number of row groups in the file.
    pub row_group_count: usize,
    /// Number of row groups with any native geospatial statistics.
    pub groups_with_stats: usize,
    /// Number of row groups with a geospatial bounding box.
    pub groups_with_bbox: usize,
    /// Number of row groups with geospatial type codes.
    pub groups_with_types: usize,
    /// Dimensions inferred from geospatial type codes.
    pub inferred_dims: CoordinateDims,
    /// Whether any row-group bbox has `xmin > xmax`.
    pub has_antimeridian_wrap: bool,
    /// Per-row-group details.
    pub row_groups: Vec<RowGroupGeospatialStats>,
}

/// Native Parquet geospatial statistics for one row group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowGroupGeospatialStats {
    /// Row group ordinal.
    pub row_group: u32,
    /// Optional row-group geospatial bounding box.
    pub bbox: Option<NativeBoundingBox>,
    /// Optional WKB/ISO geometry type codes from Parquet statistics.
    pub geospatial_types: Option<Vec<i32>>,
    /// Dimensions inferred from the type codes in this row group.
    pub inferred_dims: CoordinateDims,
}

/// Native Parquet geospatial bounding box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeBoundingBox {
    /// Minimum X / longitude / easting.
    pub xmin: f64,
    /// Maximum X / longitude / easting.
    pub xmax: f64,
    /// Minimum Y / latitude / northing.
    pub ymin: f64,
    /// Maximum Y / latitude / northing.
    pub ymax: f64,
    /// Minimum Z, if present.
    pub zmin: Option<f64>,
    /// Maximum Z, if present.
    pub zmax: Option<f64>,
    /// Minimum M, if present.
    pub mmin: Option<f64>,
    /// Maximum M, if present.
    pub mmax: Option<f64>,
    /// True when `xmin > xmax`, the Parquet antimeridian wrap convention.
    pub crosses_antimeridian: bool,
}

pub(crate) fn issue(
    severity: ValidationSeverity,
    code: ValidationCode,
    column: Option<String>,
    message: impl Into<String>,
) -> ValidationIssue {
    ValidationIssue {
        severity,
        code,
        column,
        message: message.into(),
    }
}

pub(crate) fn has_errors(issues: &[ValidationIssue]) -> bool {
    issues
        .iter()
        .any(|issue| issue.severity == ValidationSeverity::Error)
}

pub(crate) fn native_geospatial_stats(meta: &ParquetMetaData) -> Vec<NativeGeospatialStatsReport> {
    meta.file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(column_index, column)| {
            let parts = column.path().parts();
            if parts.len() != 1
                || column.max_rep_level() != 0
                || column.physical_type() != ParquetPhysicalType::BYTE_ARRAY
            {
                return None;
            }
            if !matches!(
                column.logical_type_ref(),
                Some(LogicalType::Geometry(_)) | Some(LogicalType::Geography(_))
            ) {
                return None;
            }
            Some(native_stats_for_column(
                meta,
                column_index,
                parts[0].clone(),
            ))
        })
        .collect()
}

pub(crate) fn coordinate_dims_from_wkb_type(ty: i32) -> CoordinateDims {
    if (3000..4000).contains(&ty) {
        CoordinateDims::Xyzm
    } else if (2000..3000).contains(&ty) {
        CoordinateDims::Xym
    } else if (1000..2000).contains(&ty) {
        CoordinateDims::Xyz
    } else {
        CoordinateDims::Xy
    }
}

fn native_stats_for_column(
    meta: &ParquetMetaData,
    column_index: usize,
    column: String,
) -> NativeGeospatialStatsReport {
    let mut row_groups = Vec::with_capacity(meta.num_row_groups());
    let mut groups_with_stats = 0usize;
    let mut groups_with_bbox = 0usize;
    let mut groups_with_types = 0usize;
    let mut inferred_dims = CoordinateDims::Unknown;
    let mut has_antimeridian_wrap = false;

    for (row_group_index, row_group) in meta.row_groups().iter().enumerate() {
        let stats = row_group.column(column_index).geo_statistics();
        if stats.is_some() {
            groups_with_stats += 1;
        }

        let bbox = stats
            .and_then(|stats| stats.bounding_box())
            .map(|bbox| NativeBoundingBox {
                xmin: bbox.get_xmin(),
                xmax: bbox.get_xmax(),
                ymin: bbox.get_ymin(),
                ymax: bbox.get_ymax(),
                zmin: bbox.get_zmin(),
                zmax: bbox.get_zmax(),
                mmin: bbox.get_mmin(),
                mmax: bbox.get_mmax(),
                crosses_antimeridian: bbox.get_xmin() > bbox.get_xmax(),
            });
        if bbox.is_some() {
            groups_with_bbox += 1;
        }
        if bbox.as_ref().is_some_and(|bbox| bbox.crosses_antimeridian) {
            has_antimeridian_wrap = true;
        }

        let geospatial_types = stats.and_then(|stats| stats.geospatial_types().cloned());
        let row_dims = geospatial_types
            .as_deref()
            .map(dims_from_types)
            .unwrap_or(CoordinateDims::Unknown);
        if geospatial_types.is_some() {
            groups_with_types += 1;
            inferred_dims = inferred_dims.merge(row_dims);
        }

        row_groups.push(RowGroupGeospatialStats {
            row_group: row_group_index as u32,
            bbox,
            geospatial_types,
            inferred_dims: row_dims,
        });
    }

    NativeGeospatialStatsReport {
        column,
        row_group_count: meta.num_row_groups(),
        groups_with_stats,
        groups_with_bbox,
        groups_with_types,
        inferred_dims,
        has_antimeridian_wrap,
        row_groups,
    }
}

fn dims_from_types(types: &[i32]) -> CoordinateDims {
    let mut dims = CoordinateDims::Unknown;
    for &ty in types {
        dims = dims.merge(coordinate_dims_from_wkb_type(ty));
    }
    dims
}
