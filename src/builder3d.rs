#[cfg(feature = "parallel")]
use crate::config::DEFAULT_PARALLEL_MIN_ITEMS;
use crate::{
    build::BuildError,
    config::DEFAULT_NODE_SIZE,
    geometry::Bounds3D,
    index3d::Index3D,
    sort3d::{
        ExperimentalSortKey3D, SortKey3D, SortKey3DContext, default_radix_bits_3d,
        encode_sort_by_key_3d, normalize_radix_bits_3d,
    },
    tree::{TreeLayout, compute_tree_layout, normalize_node_size},
};

/// Builder for [`Index3D`].
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Bounds3D, Index3DBuilder};
///
/// let mut builder = Index3DBuilder::new(2);
/// builder.add(Bounds3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
/// builder.add(Bounds3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
///
/// let index = builder.finish().unwrap();
/// assert_eq!(
///     index.search(Bounds3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)),
///     vec![0]
/// );
/// ```
#[must_use = "Index3DBuilder methods return an updated builder; assign the result or chain the call"]
pub struct Index3DBuilder {
    node_size: usize,
    num_items: usize,
    sort_key: ExperimentalSortKey3D,
    radix: bool,
    radix_bits: u32,
    #[cfg(feature = "parallel")]
    parallel: bool,
    #[cfg(feature = "parallel")]
    parallel_min_items: usize,
    boxes: Vec<Bounds3D>,
}

impl Index3DBuilder {
    /// Create a builder for exactly `count` items with [`DEFAULT_NODE_SIZE`].
    pub fn new(count: usize) -> Self {
        Self {
            node_size: DEFAULT_NODE_SIZE,
            num_items: count,
            sort_key: SortKey3D::Hilbert.into(),
            radix: true,
            radix_bits: default_radix_bits_3d(),
            #[cfg(feature = "parallel")]
            parallel: false,
            #[cfg(feature = "parallel")]
            parallel_min_items: DEFAULT_PARALLEL_MIN_ITEMS,
            boxes: Vec::with_capacity(count.saturating_add(1)),
        }
    }

    /// Set the maximum number of children per tree node (clamped to `[2, 65535]`).
    pub fn node_size(mut self, node_size: usize) -> Self {
        self.node_size = normalize_node_size(node_size);
        self
    }

    /// Choose the 3D sort key (default: [`SortKey3D::Hilbert`]).
    pub fn sort_key(mut self, key: SortKey3D) -> Self {
        self.sort_key = key.into();
        self
    }

    /// Choose an experimental 3D sort-key implementation.
    #[doc(hidden)]
    pub fn experimental_sort_key(mut self, key: ExperimentalSortKey3D) -> Self {
        self.sort_key = key;
        self
    }

    /// Use LSD radix sort on the u64 sort key instead of comparison-based sorting.
    #[doc(hidden)]
    pub fn radix(mut self, radix: bool) -> Self {
        self.radix = radix;
        self
    }

    /// Set the LSD radix-sort digit width for benchmarks and tuning.
    ///
    /// Values are clamped to `1..=16`; the default is 8.
    #[doc(hidden)]
    pub fn experimental_radix_bits(mut self, bits: u32) -> Self {
        self.radix_bits = normalize_radix_bits_3d(bits);
        self
    }

    /// Allow adaptive parallel builds through rayon.
    ///
    /// Default is `false`. When set to `true`, rayon is used only when the item
    /// count is at least the current `parallel_min_items` threshold.
    #[cfg(feature = "parallel")]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the minimum `count` at which [`parallel(true)`](Self::parallel)
    /// actually enables rayon.
    #[cfg(feature = "parallel")]
    pub fn parallel_min_items(mut self, min_items: usize) -> Self {
        self.parallel_min_items = min_items;
        self
    }

    /// Add item bounds.
    #[inline]
    pub fn add(&mut self, bounds: Bounds3D) {
        self.boxes.push(bounds);
    }

    /// Pack the tree and return the finished 3D index.
    pub fn finish(self) -> Result<Index3D, BuildError> {
        if self.boxes.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.boxes.len(),
                expected: self.num_items,
            });
        }
        Ok(self.build_unchecked())
    }

    fn build_unchecked(self) -> Index3D {
        let node_size = self.node_size;
        let num_items = self.num_items;
        let TreeLayout {
            level_bounds,
            num_nodes,
        } = compute_tree_layout(num_items, node_size);

        if num_items == 0 {
            return Index3D {
                node_size,
                num_items,
                level_bounds,
                boxes: Vec::new(),
                indices: Vec::new(),
            };
        }

        if num_items <= node_size {
            return build_single_node_index_3d(node_size, num_items, level_bounds, self.boxes);
        }

        let mut boxes = vec![Bounds3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); num_nodes];
        let mut indices = vec![0usize; num_nodes];
        let items = &self.boxes;

        #[cfg(feature = "parallel")]
        let use_parallel = self.parallel && num_items >= self.parallel_min_items;

        let extent = extent_3d(items);
        let context = SortKey3DContext::new(extent, self.radix, self.radix_bits);
        #[cfg(feature = "parallel")]
        let context = context.parallel(use_parallel);
        let order = encode_sort_by_key_3d(items, self.sort_key, context);

        #[cfg(feature = "parallel")]
        if use_parallel {
            reorder_parallel_3d(&mut boxes, &mut indices, &order, items, num_items);
        } else {
            reorder_serial_3d(&mut boxes, &mut indices, &order, items);
        }
        #[cfg(not(feature = "parallel"))]
        reorder_serial_3d(&mut boxes, &mut indices, &order, items);

        let mut read_pos = 0usize;
        let mut write_pos = num_items;
        for &level_end in &level_bounds[..level_bounds.len() - 1] {
            while read_pos < level_end {
                let node_index = read_pos;
                let mut node_bounds = empty_bounds_3d();
                let mut children = 0usize;
                while children < node_size && read_pos < level_end {
                    node_bounds.extend(boxes[read_pos]);
                    read_pos += 1;
                    children += 1;
                }
                boxes[write_pos] = node_bounds;
                indices[write_pos] = node_index;
                write_pos += 1;
            }
        }

        Index3D {
            node_size,
            num_items,
            level_bounds,
            boxes,
            indices,
        }
    }
}

fn reorder_serial_3d(
    boxes: &mut [Bounds3D],
    indices: &mut [usize],
    order: &[(u64, usize)],
    items: &[Bounds3D],
) {
    for (slot, &(_, original)) in order.iter().enumerate() {
        boxes[slot] = items[original];
        indices[slot] = original;
    }
}

fn build_single_node_index_3d(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    mut boxes: Vec<Bounds3D>,
) -> Index3D {
    let mut root = empty_bounds_3d();
    for &bounds in &boxes {
        root.extend(bounds);
    }
    boxes.push(root);

    let mut indices = Vec::with_capacity(num_items + 1);
    indices.extend(0..num_items);
    indices.push(0);

    Index3D {
        node_size,
        num_items,
        level_bounds,
        boxes,
        indices,
    }
}

fn extent_3d(items: &[Bounds3D]) -> Bounds3D {
    let mut extent = empty_bounds_3d();
    for &bounds in items {
        extent.extend(bounds);
    }
    extent
}

fn empty_bounds_3d() -> Bounds3D {
    Bounds3D::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    )
}

#[cfg(feature = "parallel")]
fn reorder_parallel_3d(
    boxes: &mut [Bounds3D],
    indices: &mut [usize],
    order: &[(u64, usize)],
    items: &[Bounds3D],
    num_items: usize,
) {
    use rayon::prelude::*;

    boxes[..num_items]
        .par_iter_mut()
        .zip(indices[..num_items].par_iter_mut())
        .zip(order.par_iter())
        .for_each(|((slot_box, slot_idx), &(_, original))| {
            *slot_box = items[original];
            *slot_idx = original;
        });
}
