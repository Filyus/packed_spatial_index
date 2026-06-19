use crate::geometry::{Box2D, Box3D};
use crate::persistence::{
    MetaFields, ParsedTree, PayloadError, read_f32_le_unchecked, read_u64_le_unchecked,
    write_index_container,
};

/// Round `x` down to the nearest `f32` that is `<= x`.
#[inline]
pub(crate) fn round_down(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) > x { r.next_down() } else { r }
}

/// Round `x` up to the nearest `f32` that is `>= x`.
#[inline]
pub(crate) fn round_up(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) < x { r.next_up() } else { r }
}

/// 2D box stored as four `f32` (`min_x, min_y, max_x, max_y`).
#[derive(Clone, Copy)]
pub(crate) struct Box2DF32 {
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) max_x: f32,
    pub(crate) max_y: f32,
}

impl Box2DF32 {
    /// Superset of `b` with bounds rounded outward (min down, max up).
    #[inline]
    pub(crate) fn from_box2d_outward(b: Box2D) -> Self {
        Self {
            min_x: round_down(b.min_x),
            min_y: round_down(b.min_y),
            max_x: round_up(b.max_x),
            max_y: round_up(b.max_y),
        }
    }

    /// `b` rounded inward (min up, max down) onto the f32 grid.
    #[inline]
    pub(crate) fn from_box2d_inward(b: Box2D) -> Self {
        Self {
            min_x: round_up(b.min_x),
            min_y: round_up(b.min_y),
            max_x: round_down(b.max_x),
            max_y: round_down(b.max_y),
        }
    }

    /// Read a 2D f32 box from SoA columns.
    #[inline]
    pub(crate) fn from_soa(
        min_xs: &[f32],
        min_ys: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        pos: usize,
    ) -> Self {
        Self {
            min_x: min_xs[pos],
            min_y: min_ys[pos],
            max_x: max_xs[pos],
            max_y: max_ys[pos],
        }
    }

    /// Read a 2D f32 box record from TREE bytes.
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn read_tree(entries: &[u8], pos: usize) -> Self {
        let off = pos * 16;
        Self {
            min_x: read_f32_le_unchecked(entries, off),
            min_y: read_f32_le_unchecked(entries, off + 4),
            max_x: read_f32_le_unchecked(entries, off + 8),
            max_y: read_f32_le_unchecked(entries, off + 12),
        }
    }

    /// Widen losslessly to an f64 box.
    #[inline]
    pub(crate) fn widen(self) -> Box2D {
        Box2D::new(
            self.min_x as f64,
            self.min_y as f64,
            self.max_x as f64,
            self.max_y as f64,
        )
    }

    #[inline]
    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    #[inline]
    pub(crate) fn definitely_overlaps_exact(self, query: Box2D) -> bool {
        (self.min_x.next_up() as f64 <= query.max_x)
            && (self.max_x.next_down() as f64 >= query.min_x)
            && (self.min_y.next_up() as f64 <= query.max_y)
            && (self.max_y.next_down() as f64 >= query.min_y)
    }

    /// True when `self` fully contains `other` (both already rounded).
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.min_x <= other.min_x
            && other.max_x <= self.max_x
            && self.min_y <= other.min_y
            && other.max_y <= self.max_y
    }
}

/// 3D box stored as six `f32`.
#[derive(Clone, Copy)]
pub(crate) struct Box3DF32 {
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) min_z: f32,
    pub(crate) max_x: f32,
    pub(crate) max_y: f32,
    pub(crate) max_z: f32,
}

impl Box3DF32 {
    /// Superset of `b` with bounds rounded outward (min down, max up).
    #[inline]
    pub(crate) fn from_box3d_outward(b: Box3D) -> Self {
        Self {
            min_x: round_down(b.min_x),
            min_y: round_down(b.min_y),
            min_z: round_down(b.min_z),
            max_x: round_up(b.max_x),
            max_y: round_up(b.max_y),
            max_z: round_up(b.max_z),
        }
    }

    /// `b` rounded inward (min up, max down) onto the f32 grid.
    #[inline]
    pub(crate) fn from_box3d_inward(b: Box3D) -> Self {
        Self {
            min_x: round_up(b.min_x),
            min_y: round_up(b.min_y),
            min_z: round_up(b.min_z),
            max_x: round_down(b.max_x),
            max_y: round_down(b.max_y),
            max_z: round_down(b.max_z),
        }
    }

    /// Read a 3D f32 box from SoA columns.
    #[inline]
    pub(crate) fn from_soa(
        min_xs: &[f32],
        min_ys: &[f32],
        min_zs: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        max_zs: &[f32],
        pos: usize,
    ) -> Self {
        Self {
            min_x: min_xs[pos],
            min_y: min_ys[pos],
            min_z: min_zs[pos],
            max_x: max_xs[pos],
            max_y: max_ys[pos],
            max_z: max_zs[pos],
        }
    }

    /// Read a 3D f32 box record from TREE bytes.
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn read_tree(entries: &[u8], pos: usize) -> Self {
        let off = pos * 24;
        Self {
            min_x: read_f32_le_unchecked(entries, off),
            min_y: read_f32_le_unchecked(entries, off + 4),
            min_z: read_f32_le_unchecked(entries, off + 8),
            max_x: read_f32_le_unchecked(entries, off + 12),
            max_y: read_f32_le_unchecked(entries, off + 16),
            max_z: read_f32_le_unchecked(entries, off + 20),
        }
    }

    /// Widen losslessly to an f64 box.
    #[inline]
    pub(crate) fn widen(self) -> Box3D {
        Box3D::new(
            self.min_x as f64,
            self.min_y as f64,
            self.min_z as f64,
            self.max_x as f64,
            self.max_y as f64,
            self.max_z as f64,
        )
    }

    #[inline]
    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }

    #[inline]
    pub(crate) fn definitely_overlaps_exact(self, query: Box3D) -> bool {
        (self.min_x.next_up() as f64 <= query.max_x)
            && (self.max_x.next_down() as f64 >= query.min_x)
            && (self.min_y.next_up() as f64 <= query.max_y)
            && (self.max_y.next_down() as f64 >= query.min_y)
            && (self.min_z.next_up() as f64 <= query.max_z)
            && (self.max_z.next_down() as f64 >= query.min_z)
    }

    /// True when `self` fully contains `other` (both already rounded).
    #[inline]
    #[cfg(feature = "simd")]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.min_x <= other.min_x
            && other.max_x <= self.max_x
            && self.min_y <= other.min_y
            && other.max_y <= self.max_y
            && self.min_z <= other.min_z
            && other.max_z <= self.max_z
    }
}

/// High bit of the stacked level word, set when the query fully contains a node so
/// its whole subtree can be collected without further overlap tests.
#[cfg(feature = "simd")]
pub(crate) const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);

#[cfg(feature = "simd")]
pub(crate) const LEVEL_MASK: usize = !CONTAINED_FLAG;

#[inline]
#[cfg(feature = "simd")]
pub(crate) fn encode_level(level: usize, contained: bool) -> usize {
    if contained {
        level | CONTAINED_FLAG
    } else {
        level
    }
}

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
