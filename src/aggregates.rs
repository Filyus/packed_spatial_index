//! Node aggregates (the aR-tree chunk): per-node summaries of a per-item
//! scalar and/or category mask, stored alongside the tree and folded exactly
//! over a query region.
//!
//! The idea is the count story generalized. `estimate_count` exploits that the
//! number of items under a node is fixed by construction, so count brackets
//! come free from node boxes. For anything else — a sum, a min/max, a set of
//! categories — the values are not derivable, so they must be stored once:
//! one summary per node, written at build time and folded back at query time.
//! A node whose box lies entirely inside the window contributes its summary
//! whole, exactly as `count_overlaps` counts its leaf range whole; only nodes
//! cut by the window's edge descend, and only their leaves are read item by
//! item. The result is exact, and a window covering the whole extent costs one
//! summary read.
//!
//! The summaries live in an optional `AGGR` chunk (see `FORMAT.md`). Because
//! the chunk is optional and carries no data a spatial query needs, a reader
//! that does not know it skips it and loses nothing but the aggregates.

use crate::estimate::subtree_leaf_range;
use crate::index2d::{MASK_CHUNK, for_each_hit, for_each_hit_rev, frame};
use crate::persistence::{
    LoadError, read_f64_le_unchecked, read_u16_at, read_u32_at, read_u64_le_unchecked,
};
use crate::range::overlap_mask_at;
use crate::traversal::ScratchStack;
use crate::tree_access::TreeAccess;

/// Which summary columns an [`Aggregates`] carries.
pub(crate) const COLUMN_SCALAR: u8 = 1;
pub(crate) const COLUMN_MASK: u8 = 2;

/// Size of the `AGGR` descriptor (see `FORMAT.md`).
pub(crate) const AGGR_DESC_LEN: usize = 16;
/// Per-item values present (descriptor flags bit 0).
pub(crate) const AGGR_FLAG_ITEMS: u16 = 1;

/// The result of an aggregate query over a region: the exact fold of every
/// item the region hits.
///
/// `count` is always present. The other fields mirror the columns the index
/// was built with: an index built without [`aggregate_scalar`](crate::Index2DBuilder::aggregate_scalar)
/// reports `sum` / `min` / `max` as `None`, one built without
/// [`aggregate_mask`](crate::Index2DBuilder::aggregate_mask) reports `mask`
/// as `None`. When nothing matches, `count` is `0` and every column is `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aggregate {
    /// How many items the region hits (the same set
    /// [`count`](crate::Index2D::count) answers).
    pub count: u64,
    /// Sum of the per-item scalar over the hits, when the index carries the
    /// scalar column.
    pub sum: Option<f64>,
    /// Smallest per-item scalar among the hits, when the index carries the
    /// scalar column.
    pub min: Option<f64>,
    /// Largest per-item scalar among the hits, when the index carries the
    /// scalar column.
    pub max: Option<f64>,
    /// Bitwise OR of the per-item category masks over the hits, when the index
    /// carries the mask column.
    pub mask: Option<u64>,
}

/// Per-node summaries attached to an index: what
/// [`aggregate`](crate::Index2D::aggregate) folds over.
///
/// Two columns exist, chosen at build time and stored together:
///
/// - **scalar** — one `f64` per item; every node stores the sum, min and max
///   of its items' values;
/// - **mask** — one `u64` per item; every node stores the bitwise OR.
///
/// Node summaries sit in tree order (the positions queries walk), so a
/// fully-contained subtree folds in O(1). The per-item values are kept too —
/// and serialized — because the leaves a query window cuts must contribute
/// their items one by one for the answer to stay exact.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Aggregates {
    pub(crate) columns: u8,
    /// Per-node sum column, node positions in tree order.
    pub(crate) sums: Vec<f64>,
    pub(crate) mins: Vec<f64>,
    pub(crate) maxs: Vec<f64>,
    pub(crate) masks: Vec<u64>,
    /// Per-item values in leaf order (indexed by leaf position).
    pub(crate) item_scalars: Vec<f64>,
    pub(crate) item_masks: Vec<u64>,
}

impl Aggregates {
    /// Whether the scalar (sum / min / max) column is present.
    pub fn has_scalar(&self) -> bool {
        self.columns & COLUMN_SCALAR != 0
    }

    /// Whether the category-mask column is present.
    pub fn has_mask(&self) -> bool {
        self.columns & COLUMN_MASK != 0
    }

    pub(crate) fn columns_byte(&self) -> u8 {
        self.columns
    }

    /// Node positions stored (whichever column is present; both have one entry
    /// per node).
    pub(crate) fn node_len(&self) -> usize {
        self.sums.len().max(self.masks.len())
    }

    pub(crate) fn item_len(&self) -> usize {
        self.item_scalars.len().max(self.item_masks.len())
    }
}

/// A borrowed, zero-copy view of an `AGGR` chunk (what the `*View` readers
/// carry).
///
/// The summaries are read out of the file bytes on demand — one unaligned
/// little-endian read per field, the same way views read node boxes — so no
/// alignment is required of the buffer and nothing is copied.
#[derive(Debug, Clone, Copy)]
pub struct AggregatesView<'a> {
    columns: u8,
    /// The node-summary table, `node_stride` bytes per node in tree order.
    nodes: &'a [u8],
    /// The per-item values, `item_stride` bytes per item in leaf order.
    items: &'a [u8],
    node_stride: usize,
    item_stride: usize,
    /// Byte offset of the mask column inside a node record (after the three
    /// scalar fields, when they are present).
    mask_offset: usize,
}

impl<'a> AggregatesView<'a> {
    /// Whether the scalar (sum / min / max) column is present.
    pub fn has_scalar(&self) -> bool {
        self.columns & COLUMN_SCALAR != 0
    }

    /// Whether the category-mask column is present.
    pub fn has_mask(&self) -> bool {
        self.columns & COLUMN_MASK != 0
    }
}

/// Read one node's summary out of either representation.
///
/// The query kernel is shared by owned indexes and zero-copy views, so both
/// answer through this sealed trait; the implementations are two layouts of
/// the same columns.
pub(crate) trait AggregateSource {
    fn columns(&self) -> u8;
    /// `(sum, min, max)` at a node position; read only when the scalar column
    /// is present.
    fn node_scalar(&self, pos: usize) -> (f64, f64, f64);
    fn node_mask(&self, pos: usize) -> u64;
    fn item_scalar(&self, pos: usize) -> f64;
    fn item_mask(&self, pos: usize) -> u64;
}

impl AggregateSource for Aggregates {
    #[inline]
    fn columns(&self) -> u8 {
        self.columns
    }
    #[inline]
    fn node_scalar(&self, pos: usize) -> (f64, f64, f64) {
        (self.sums[pos], self.mins[pos], self.maxs[pos])
    }
    #[inline]
    fn node_mask(&self, pos: usize) -> u64 {
        self.masks[pos]
    }
    #[inline]
    fn item_scalar(&self, pos: usize) -> f64 {
        self.item_scalars[pos]
    }
    #[inline]
    fn item_mask(&self, pos: usize) -> u64 {
        self.item_masks[pos]
    }
}

impl AggregateSource for AggregatesView<'_> {
    #[inline]
    fn columns(&self) -> u8 {
        self.columns
    }
    #[inline]
    fn node_scalar(&self, pos: usize) -> (f64, f64, f64) {
        let base = pos * self.node_stride;
        (
            read_f64_le_unchecked(self.nodes, base),
            read_f64_le_unchecked(self.nodes, base + 8),
            read_f64_le_unchecked(self.nodes, base + 16),
        )
    }
    #[inline]
    fn node_mask(&self, pos: usize) -> u64 {
        read_u64_le_unchecked(self.nodes, pos * self.node_stride + self.mask_offset)
    }
    #[inline]
    fn item_scalar(&self, pos: usize) -> f64 {
        read_f64_le_unchecked(self.items, pos * self.item_stride)
    }
    #[inline]
    fn item_mask(&self, pos: usize) -> u64 {
        read_u64_le_unchecked(
            self.items,
            pos * self.item_stride + 8 * usize::from(self.has_scalar()),
        )
    }
}

/// Build the per-node summaries bottom-up from per-item values in leaf order.
///
/// `scalars` / `masks` are indexed by leaf position (the order the finished
/// index stores items in); passing at least one column is the caller's job.
/// The fold over internal nodes walks the same child ranges the builders
/// write: the children of node `pos` are the contiguous run
/// `[indices[pos], indices[pos] + node_size)` clipped to the child level.
pub(crate) fn build_aggregates(
    node_size: usize,
    level_bounds: &[usize],
    indices: &[usize],
    scalars: Option<&[f64]>,
    masks: Option<&[u64]>,
) -> Aggregates {
    let mut columns = 0;
    if scalars.is_some() {
        columns |= COLUMN_SCALAR;
    }
    if masks.is_some() {
        columns |= COLUMN_MASK;
    }
    let num_items = level_bounds.first().copied().unwrap_or(0);
    let num_nodes = *level_bounds.last().unwrap_or(&0);
    let mut out = Aggregates {
        columns,
        sums: Vec::new(),
        mins: Vec::new(),
        maxs: Vec::new(),
        masks: Vec::new(),
        item_scalars: scalars.map(<[f64]>::to_vec).unwrap_or_default(),
        item_masks: masks.map(<[u64]>::to_vec).unwrap_or_default(),
    };
    if scalars.is_some() {
        out.sums = vec![0.0; num_nodes];
        out.mins = vec![f64::INFINITY; num_nodes];
        out.maxs = vec![f64::NEG_INFINITY; num_nodes];
    }
    if masks.is_some() {
        out.masks = vec![0; num_nodes];
    }

    // Leaves first, straight from the item values.
    if let Some(scalars) = scalars {
        for (pos, &v) in scalars.iter().enumerate().take(num_items) {
            out.sums[pos] = v;
            out.mins[pos] = v;
            out.maxs[pos] = v;
        }
    }
    if let Some(masks) = masks {
        out.masks[..num_items].copy_from_slice(masks);
    }
    // Then every internal level over the previous one.
    for level in 1..level_bounds.len() {
        let level_start = level_bounds[level - 1];
        let level_end = level_bounds[level];
        for (offset, &first_child) in indices[level_start..level_end].iter().enumerate() {
            let pos = level_start + offset;
            let children = (first_child + node_size).min(level_start) - first_child;
            if out.has_scalar() {
                let (mut sum, mut min, mut max) = (0.0, f64::INFINITY, f64::NEG_INFINITY);
                for child in first_child..first_child + children {
                    sum += out.sums[child];
                    min = min.min(out.mins[child]);
                    max = max.max(out.maxs[child]);
                }
                out.sums[pos] = sum;
                out.mins[pos] = min;
                out.maxs[pos] = max;
            }
            if out.has_mask() {
                let mut mask = 0u64;
                for child in first_child..first_child + children {
                    mask |= out.masks[child];
                }
                out.masks[pos] = mask;
            }
        }
    }
    out
}

/// Permute the builder's insertion-order values into leaf order and build the
/// summaries. Every owned frontend calls this once at build (or SIMD build)
/// time; `indices` is the finished index's `leaf_order()`.
pub(crate) fn aggregates_for_index(
    node_size: usize,
    level_bounds: &[usize],
    indices: &[usize],
    scalars: Option<&[f64]>,
    masks: Option<&[u64]>,
) -> Aggregates {
    let num_items = level_bounds.first().copied().unwrap_or(0);
    let leaf_scalars = scalars.map(|v| {
        indices[..num_items]
            .iter()
            .map(|&id| v[id])
            .collect::<Vec<f64>>()
    });
    let leaf_masks = masks.map(|v| {
        indices[..num_items]
            .iter()
            .map(|&id| v[id])
            .collect::<Vec<u64>>()
    });
    build_aggregates(
        node_size,
        level_bounds,
        indices,
        leaf_scalars.as_deref(),
        leaf_masks.as_deref(),
    )
}

/// Fold the summaries over every item whose box overlaps the region.
///
/// The descent is the collect-path shape (`range::collect_region`): each
/// node's overlap tests run branch-free into a bitmask and the loop branches
/// once per hit. A hit child the region *contains* is folded on the spot from
/// its stored summary — it never enters the stack — and only the children a
/// region edge cuts are pushed, one packed frame each; at the leaves the hit
/// items are folded one by one. The answer is exact: it equals `search` + a
/// per-item fold, at a fraction of the leaves touched.
pub(crate) fn aggregate_region_core<T, S, O, C>(
    tree: &T,
    agg: &S,
    overlaps: O,
    contains: C,
) -> Aggregate
where
    T: TreeAccess,
    S: AggregateSource,
    O: Fn(T::Bounds) -> bool,
    C: Fn(T::Bounds) -> bool,
{
    let num_items = tree.tree_num_items();
    let scalar = agg.columns() & COLUMN_SCALAR != 0;
    let mask = agg.columns() & COLUMN_MASK != 0;
    let mut out = Aggregate {
        count: 0,
        sum: None,
        min: None,
        max: None,
        mask: None,
    };
    if num_items == 0 || agg.columns() == 0 {
        return out;
    }
    let mut acc = Fold {
        count: 0,
        sum: 0.0,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        or_mask: 0,
        scalar,
        mask,
    };

    let root = tree.tree_num_nodes() - 1;
    let root_bounds = tree.tree_bounds(root);
    if !overlaps(root_bounds) {
        return out;
    }
    let top = tree.tree_level_count() - 1;
    if top > 0 && contains(root_bounds) {
        acc.count = num_items as u64;
        acc.node(agg, root);
    } else {
        let mut stack = ScratchStack::take();
        let mut node_index = root;
        let mut level = top;
        loop {
            let end = (node_index + tree.tree_node_size()).min(tree.tree_level_bound(level));
            if level == 0 {
                let mut start = node_index;
                while start < end {
                    let stop = (start + MASK_CHUNK).min(end);
                    for_each_hit(overlap_mask_at(tree, start, stop, &overlaps), |i| {
                        acc.item(agg, start + i);
                    });
                    start = stop;
                }
            } else {
                let child_level = level - 1;
                // Chunks from the back, bits from the top: children pop in
                // forward order.
                let mut stop = end;
                while stop > node_index {
                    let start = stop.saturating_sub(MASK_CHUNK).max(node_index);
                    for_each_hit_rev(overlap_mask_at(tree, start, stop, &overlaps), |i| {
                        let pos = start + i;
                        if contains(tree.tree_bounds(pos)) {
                            acc.count += subtree_leaf_count(tree, pos, level);
                            acc.node(agg, pos);
                        } else {
                            stack.push(frame::pack(tree.tree_index(pos), child_level));
                        }
                    });
                    stop = start;
                }
            }
            match stack.pop() {
                Some(f) => {
                    node_index = frame::node(f);
                    level = frame::level(f);
                }
                None => break,
            }
        }
    }
    out.count = acc.count;
    if acc.count > 0 {
        if scalar {
            out.sum = Some(acc.sum);
            out.min = Some(acc.min);
            out.max = Some(acc.max);
        }
        if mask {
            out.mask = Some(acc.or_mask);
        }
    }
    out
}

/// The running fold of `aggregate_region_core`.
struct Fold {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    or_mask: u64,
    scalar: bool,
    mask: bool,
}

impl Fold {
    /// Fold one node's stored summary (a contained subtree).
    #[inline]
    fn node<S: AggregateSource>(&mut self, agg: &S, pos: usize) {
        if self.scalar {
            let (s, lo, hi) = agg.node_scalar(pos);
            self.sum += s;
            self.min = self.min.min(lo);
            self.max = self.max.max(hi);
        }
        if self.mask {
            self.or_mask |= agg.node_mask(pos);
        }
    }

    /// Fold one item at its leaf position.
    #[inline]
    fn item<S: AggregateSource>(&mut self, agg: &S, pos: usize) {
        self.count += 1;
        if self.scalar {
            let v = agg.item_scalar(pos);
            self.sum += v;
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        if self.mask {
            self.or_mask |= agg.item_mask(pos);
        }
    }
}

/// Leaf-array items under the node at `pos` on `level` (rank arithmetic, see
/// `estimate::subtree_leaf_range`).
#[inline]
fn subtree_leaf_count<T: TreeAccess>(tree: &T, pos: usize, level: usize) -> u64 {
    let level_start = if level == 0 {
        0
    } else {
        tree.tree_level_bound(level - 1)
    };
    let (start, end) = subtree_leaf_range(
        pos,
        level,
        level_start,
        tree.tree_node_size(),
        tree.tree_num_items(),
    );
    (end - start) as u64
}

/// Parse an `AGGR` chunk body into a zero-copy view.
///
/// `num_nodes` and `num_items` come from the parsed `TREE` chunk, so the
/// section lengths are derivable and any drift is rejected here.
pub(crate) fn parse_aggr_chunk<'a>(
    content: &'a [u8],
    num_nodes: usize,
    num_items: usize,
) -> Result<AggregatesView<'a>, LoadError> {
    if content.len() < AGGR_DESC_LEN {
        return Err(LoadError::InvalidAggregates);
    }
    let desc_len = read_u32_at(content, 0)? as usize;
    if desc_len < AGGR_DESC_LEN || content.len() < desc_len {
        return Err(LoadError::InvalidAggregates);
    }
    let ordering = content[4];
    let columns = content[5];
    let flags = read_u16_at(content, 6)?;
    // The per-item values are required: the query kernel reads them for every
    // leaf the window cuts, so a chunk without them (the flag is reserved for
    // a summary-only variant) cannot answer exactly and is rejected rather
    // than read past its end.
    if ordering != 0 || columns == 0 || columns & !0b11 != 0 || flags != AGGR_FLAG_ITEMS {
        return Err(LoadError::InvalidAggregates);
    }
    let scalar = columns & COLUMN_SCALAR != 0;
    let mask = columns & COLUMN_MASK != 0;
    let node_stride = 24 * usize::from(scalar) + 8 * usize::from(mask);
    let item_stride = 8 * (usize::from(scalar) + usize::from(mask));
    let nodes_len = node_stride
        .checked_mul(num_nodes)
        .ok_or(LoadError::IntegerOverflow)?;
    let items_len = item_stride
        .checked_mul(num_items)
        .ok_or(LoadError::IntegerOverflow)?;
    let expected = desc_len
        .checked_add(nodes_len)
        .and_then(|len| len.checked_add(items_len))
        .ok_or(LoadError::IntegerOverflow)?;
    if content.len() != expected {
        return Err(LoadError::InvalidAggregates);
    }
    Ok(AggregatesView {
        columns,
        nodes: &content[desc_len..desc_len + nodes_len],
        items: &content[desc_len + nodes_len..],
        node_stride,
        item_stride,
        mask_offset: 24 * usize::from(scalar),
    })
}

/// Materialize a parsed chunk into owned columns (the owned-index load path).
pub(crate) fn aggregates_from_view(view: &AggregatesView<'_>, num_items: usize) -> Aggregates {
    Aggregates {
        columns: view.columns,
        sums: if view.has_scalar() {
            (0..view.nodes.len() / view.node_stride)
                .map(|pos| view.node_scalar(pos).0)
                .collect()
        } else {
            Vec::new()
        },
        mins: if view.has_scalar() {
            (0..view.nodes.len() / view.node_stride)
                .map(|pos| view.node_scalar(pos).1)
                .collect()
        } else {
            Vec::new()
        },
        maxs: if view.has_scalar() {
            (0..view.nodes.len() / view.node_stride)
                .map(|pos| view.node_scalar(pos).2)
                .collect()
        } else {
            Vec::new()
        },
        masks: if view.has_mask() {
            (0..view.nodes.len() / view.node_stride)
                .map(|pos| view.node_mask(pos))
                .collect()
        } else {
            Vec::new()
        },
        item_scalars: if view.has_scalar() {
            (0..num_items).map(|pos| view.item_scalar(pos)).collect()
        } else {
            Vec::new()
        },
        item_masks: if view.has_mask() {
            (0..num_items).map(|pos| view.item_mask(pos)).collect()
        } else {
            Vec::new()
        },
    }
}
