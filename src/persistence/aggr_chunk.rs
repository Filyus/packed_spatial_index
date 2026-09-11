use crate::aggregates::{AGGR_DESC_LEN, AGGR_FLAG_ITEMS, AggregateSource, Aggregates};

use super::ByteWriter;

pub(crate) const TAG_AGGR: [u8; 4] = *b"AGGR";

/// Padded length of the `AGGR` chunk for an index of the given shape.
pub(crate) fn aggr_chunk_len(agg: &Aggregates, num_nodes: usize, num_items: usize) -> usize {
    let node_stride = 24 * usize::from(agg.has_scalar()) + 8 * usize::from(agg.has_mask());
    let item_stride = 8 * (usize::from(agg.has_scalar()) + usize::from(agg.has_mask()));
    AGGR_DESC_LEN + node_stride * num_nodes + item_stride * num_items
}

/// Write the `AGGR` chunk body: descriptor, node summaries in tree order (the
/// same node order the `TREE` chunk stores), then per-item values in leaf
/// order. Only the columns present are written.
pub(crate) fn write_aggr_chunk(bytes: &mut ByteWriter<'_>, agg: &Aggregates) {
    bytes.write_u32(AGGR_DESC_LEN as u32); // desc_len
    bytes.write_u8(0); // ordering: tree level order
    bytes.write_u8(agg.columns_byte());
    bytes.write_u16(AGGR_FLAG_ITEMS); // flags: per-item values present
    bytes.write_u64(0); // reserved
    for pos in 0..agg.node_len() {
        if agg.has_scalar() {
            let (sum, min, max) = agg.node_scalar(pos);
            bytes.write_f64(sum);
            bytes.write_f64(min);
            bytes.write_f64(max);
        }
        if agg.has_mask() {
            bytes.write_u64(agg.node_mask(pos));
        }
    }
    let num_items = agg.item_len();
    for pos in 0..num_items {
        if agg.has_scalar() {
            bytes.write_f64(agg.item_scalar(pos));
        }
        if agg.has_mask() {
            bytes.write_u64(agg.item_mask(pos));
        }
    }
}
