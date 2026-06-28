use parquet::basic::{LogicalType, Type as ParquetPhysicalType};
use parquet::file::metadata::ParquetMetaData;

use crate::{
    CoordinateDims, NativeBoundingBox, NativeGeospatialStatsReport, RowGroupGeospatialStats,
    ValidationCode, ValidationIssue, ValidationSeverity,
};

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
