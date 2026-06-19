use crate::persistence::{ParsedTree, read_f32_le_unchecked, read_u64_le_unchecked};

pub(crate) struct F32Columns2D {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) min_xs: Vec<f32>,
    pub(crate) min_ys: Vec<f32>,
    pub(crate) max_xs: Vec<f32>,
    pub(crate) max_ys: Vec<f32>,
    pub(crate) indices: Vec<usize>,
}

pub(crate) struct F32Columns3D {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) min_xs: Vec<f32>,
    pub(crate) min_ys: Vec<f32>,
    pub(crate) min_zs: Vec<f32>,
    pub(crate) max_xs: Vec<f32>,
    pub(crate) max_ys: Vec<f32>,
    pub(crate) max_zs: Vec<f32>,
    pub(crate) indices: Vec<usize>,
}

pub(crate) struct F32ColumnRefs2D<'a> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) min_xs: &'a [f32],
    pub(crate) min_ys: &'a [f32],
    pub(crate) max_xs: &'a [f32],
    pub(crate) max_ys: &'a [f32],
    pub(crate) indices: &'a [usize],
}

pub(crate) struct F32ColumnRefs3D<'a> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) min_xs: &'a [f32],
    pub(crate) min_ys: &'a [f32],
    pub(crate) min_zs: &'a [f32],
    pub(crate) max_xs: &'a [f32],
    pub(crate) max_ys: &'a [f32],
    pub(crate) max_zs: &'a [f32],
    pub(crate) indices: &'a [usize],
}

/// Materialize 2D SoA f32 columns from a parsed `f32` TREE chunk.
pub(crate) fn columns2d_from_parsed(parsed: &ParsedTree) -> F32Columns2D {
    let num_nodes = parsed.num_nodes;
    let mut min_xs = Vec::with_capacity(num_nodes);
    let mut min_ys = Vec::with_capacity(num_nodes);
    let mut max_xs = Vec::with_capacity(num_nodes);
    let mut max_ys = Vec::with_capacity(num_nodes);
    let mut indices = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let off = i * 16; // four f32 per 2D box record
        min_xs.push(read_f32_le_unchecked(parsed.entries, off));
        min_ys.push(read_f32_le_unchecked(parsed.entries, off + 4));
        max_xs.push(read_f32_le_unchecked(parsed.entries, off + 8));
        max_ys.push(read_f32_le_unchecked(parsed.entries, off + 12));
        indices.push(read_u64_le_unchecked(parsed.indices, i * 8) as usize);
    }
    F32Columns2D {
        node_size: parsed.node_size,
        num_items: parsed.num_items,
        level_bounds: parsed.level_bounds.clone(),
        min_xs,
        min_ys,
        max_xs,
        max_ys,
        indices,
    }
}

/// Materialize 3D SoA f32 columns from a parsed `f32` TREE chunk.
pub(crate) fn columns3d_from_parsed(parsed: &ParsedTree) -> F32Columns3D {
    let num_nodes = parsed.num_nodes;
    let mut min_xs = Vec::with_capacity(num_nodes);
    let mut min_ys = Vec::with_capacity(num_nodes);
    let mut min_zs = Vec::with_capacity(num_nodes);
    let mut max_xs = Vec::with_capacity(num_nodes);
    let mut max_ys = Vec::with_capacity(num_nodes);
    let mut max_zs = Vec::with_capacity(num_nodes);
    let mut indices = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let off = i * 24; // six f32 per 3D box record
        min_xs.push(read_f32_le_unchecked(parsed.entries, off));
        min_ys.push(read_f32_le_unchecked(parsed.entries, off + 4));
        min_zs.push(read_f32_le_unchecked(parsed.entries, off + 8));
        max_xs.push(read_f32_le_unchecked(parsed.entries, off + 12));
        max_ys.push(read_f32_le_unchecked(parsed.entries, off + 16));
        max_zs.push(read_f32_le_unchecked(parsed.entries, off + 20));
        indices.push(read_u64_le_unchecked(parsed.indices, i * 8) as usize);
    }
    F32Columns3D {
        node_size: parsed.node_size,
        num_items: parsed.num_items,
        level_bounds: parsed.level_bounds.clone(),
        min_xs,
        min_ys,
        min_zs,
        max_xs,
        max_ys,
        max_zs,
        indices,
    }
}
