//! Accelerator (Model 1): build an in-memory index over the GeoParquet row
//! bounding boxes. Item id equals the file row index, so query results are row
//! indices you can read back from the original Parquet.

use packed_spatial_index::{Box2D, Box3D, Index2D, Index2DBuilder, Index3D, Index3DBuilder};
use parquet::file::reader::ChunkReader;

use crate::{BuildOpts, GeoError, read};

/// Build an [`Index2D`] over the file's row bounding boxes.
pub fn build_index_2d<R: ChunkReader + 'static>(
    reader: R,
    opts: BuildOpts,
) -> Result<Index2D, GeoError> {
    let boxes = read::read_bboxes_2d(reader)?;
    Ok(loaded_builder_2d(&boxes, &opts).finish()?)
}

/// Build an [`Index3D`] over the file's row bounding boxes.
pub fn build_index_3d<R: ChunkReader + 'static>(
    reader: R,
    opts: BuildOpts,
) -> Result<Index3D, GeoError> {
    let boxes = read::read_bboxes_3d(reader)?;
    Ok(loaded_builder_3d(&boxes, &opts).finish()?)
}

/// A 2D builder configured from `opts` with every box already added (in row
/// order). Shared by the accelerator and the converter.
pub(crate) fn loaded_builder_2d(boxes: &[Box2D], opts: &BuildOpts) -> Index2DBuilder {
    let mut b = Index2DBuilder::new(boxes.len());
    if let Some(ns) = opts.node_size {
        b = b.node_size(ns);
    }
    b = b.parallel(opts.parallel);
    for &bx in boxes {
        b.add(bx);
    }
    b
}

pub(crate) fn loaded_builder_3d(boxes: &[Box3D], opts: &BuildOpts) -> Index3DBuilder {
    let mut b = Index3DBuilder::new(boxes.len());
    if let Some(ns) = opts.node_size {
        b = b.node_size(ns);
    }
    b = b.parallel(opts.parallel);
    for &bx in boxes {
        b.add(bx);
    }
    b
}
