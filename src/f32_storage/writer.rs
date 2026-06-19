use super::columns::{F32ColumnRefs2D, F32ColumnRefs3D};
use crate::persistence::{MetaFields, PayloadError, write_index_container};

/// Write 2D f32 SoA columns into a `PSINDEX` container.
pub(crate) fn write_columns2d(
    out: &mut Vec<u8>,
    columns: F32ColumnRefs2D<'_>,
    interleaved: bool,
    payloads: Option<&[&[u8]]>,
    record_stride: Option<u32>,
    meta: &MetaFields<'_>,
) -> Result<(), PayloadError> {
    debug_assert_eq!(columns.min_xs.len(), columns.min_ys.len());
    debug_assert_eq!(columns.min_xs.len(), columns.max_xs.len());
    debug_assert_eq!(columns.min_xs.len(), columns.max_ys.len());
    debug_assert_eq!(columns.min_xs.len(), columns.indices.len());
    write_index_container(
        out,
        2,
        4,
        interleaved,
        columns.num_items,
        columns.min_xs.len(),
        columns.node_size,
        |bytes| {
            #[cfg(feature = "stream")]
            if interleaved {
                bytes.write_interleaved_f32_2d(
                    columns.min_xs,
                    columns.min_ys,
                    columns.max_xs,
                    columns.max_ys,
                    columns.indices,
                );
                return;
            }
            bytes.write_soa_boxes_f32_2d(
                columns.min_xs,
                columns.min_ys,
                columns.max_xs,
                columns.max_ys,
            );
            bytes.write_usize_slice_as_u64(columns.indices);
        },
        payloads,
        record_stride,
        &columns.indices[..columns.num_items],
        meta,
    )
}

/// Write 3D f32 SoA columns into a `PSINDEX` container.
pub(crate) fn write_columns3d(
    out: &mut Vec<u8>,
    columns: F32ColumnRefs3D<'_>,
    interleaved: bool,
    payloads: Option<&[&[u8]]>,
    record_stride: Option<u32>,
    meta: &MetaFields<'_>,
) -> Result<(), PayloadError> {
    debug_assert_eq!(columns.min_xs.len(), columns.min_ys.len());
    debug_assert_eq!(columns.min_xs.len(), columns.min_zs.len());
    debug_assert_eq!(columns.min_xs.len(), columns.max_xs.len());
    debug_assert_eq!(columns.min_xs.len(), columns.max_ys.len());
    debug_assert_eq!(columns.min_xs.len(), columns.max_zs.len());
    debug_assert_eq!(columns.min_xs.len(), columns.indices.len());
    write_index_container(
        out,
        3,
        4,
        interleaved,
        columns.num_items,
        columns.min_xs.len(),
        columns.node_size,
        |bytes| {
            #[cfg(feature = "stream")]
            if interleaved {
                bytes.write_interleaved_f32_3d(
                    columns.min_xs,
                    columns.min_ys,
                    columns.min_zs,
                    columns.max_xs,
                    columns.max_ys,
                    columns.max_zs,
                    columns.indices,
                );
                return;
            }
            bytes.write_soa_boxes_f32_3d(
                columns.min_xs,
                columns.min_ys,
                columns.min_zs,
                columns.max_xs,
                columns.max_ys,
                columns.max_zs,
            );
            bytes.write_usize_slice_as_u64(columns.indices);
        },
        payloads,
        record_stride,
        &columns.indices[..columns.num_items],
        meta,
    )
}
