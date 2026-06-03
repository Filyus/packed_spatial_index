#[cfg(feature = "parallel")]
use crate::config::DEFAULT_PARALLEL_MIN_ITEMS;
use crate::{
    build::BuildError,
    config::DEFAULT_NODE_SIZE,
    geometry::{Box3D, empty_box3d, extend_box3d},
    index3d::Index3D,
    sort3d::{
        ExperimentalSortKey3D, SortKey3D, SortKey3DContext, default_radix_bits_3d,
        encode_sort_by_key_3d, normalize_radix_bits_3d,
    },
    tree::{TreeLayout, normalize_node_size, try_compute_tree_layout},
};

/// Build parameters passed to the SoA/SIMD 3D builder.
#[cfg(feature = "simd")]
pub(crate) struct BuildConfig3D {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) sort_key: ExperimentalSortKey3D,
    pub(crate) radix: bool,
    pub(crate) radix_bits: u32,
    #[cfg(feature = "parallel")]
    pub(crate) parallel: bool,
    #[cfg(feature = "parallel")]
    pub(crate) parallel_min_items: usize,
}

/// Builder for [`Index3D`] and, with the `simd` feature, `SimdIndex3D`.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Box3D, Index3DBuilder};
///
/// let mut builder = Index3DBuilder::new(2);
/// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
/// builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
///
/// let index = builder.finish().unwrap();
/// assert_eq!(
///     index.search(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)),
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
    items: Vec<Box3D>,
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
            items: items_vec_with_root_capacity_3d(count),
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

    /// Add one indexed box.
    #[inline]
    pub fn add(&mut self, item: Box3D) {
        self.items.push(item);
    }

    /// Pack the tree and return the finished 3D index.
    pub fn finish(self) -> Result<Index3D, BuildError> {
        if self.items.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.items.len(),
                expected: self.num_items,
            });
        }
        self.build()
    }

    /// Pack the tree into the SIMD-accelerated SoA 3D index.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index3DBuilder, Box3D};
    ///
    /// let mut builder = Index3DBuilder::new(1);
    /// builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    ///
    /// let index = builder.finish_simd().unwrap();
    /// assert_eq!(index.search(Box3D::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)), vec![0]);
    /// ```
    #[cfg(feature = "simd")]
    pub fn finish_simd(self) -> Result<crate::SimdIndex3D, BuildError> {
        if self.items.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.items.len(),
                expected: self.num_items,
            });
        }
        let config = self.config();
        crate::index3d_soa::build_simd_index_3d(config, self.items)
    }

    #[cfg(feature = "simd")]
    fn config(&self) -> BuildConfig3D {
        BuildConfig3D {
            node_size: self.node_size,
            num_items: self.num_items,
            sort_key: self.sort_key,
            radix: self.radix,
            radix_bits: self.radix_bits,
            #[cfg(feature = "parallel")]
            parallel: self.parallel,
            #[cfg(feature = "parallel")]
            parallel_min_items: self.parallel_min_items,
        }
    }

    fn build(self) -> Result<Index3D, BuildError> {
        let node_size = self.node_size;
        let num_items = self.num_items;
        let TreeLayout {
            level_bounds,
            num_nodes,
        } = try_compute_tree_layout(num_items, node_size)?;

        if num_items == 0 {
            return Ok(Index3D {
                node_size,
                num_items,
                level_bounds,
                entries: Vec::new(),
                indices: Vec::new(),
            });
        }

        if num_items <= node_size {
            return Ok(build_single_node_index_3d(
                node_size,
                num_items,
                level_bounds,
                self.items,
            ));
        }

        let mut entries = vec![Box3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); num_nodes];
        let mut indices = vec![0usize; num_nodes];
        let items = &self.items;

        #[cfg(feature = "parallel")]
        let use_parallel = self.parallel && num_items >= self.parallel_min_items;

        let extent = extent_3d(items);
        let context = SortKey3DContext::new(extent, self.radix, self.radix_bits);
        #[cfg(feature = "parallel")]
        let context = context.parallel(use_parallel);
        let order = encode_sort_by_key_3d(items, self.sort_key, context);

        #[cfg(feature = "parallel")]
        if use_parallel {
            reorder_parallel_3d(&mut entries, &mut indices, &order, items, num_items);
        } else {
            reorder_serial_3d(&mut entries, &mut indices, &order, items);
        }
        #[cfg(not(feature = "parallel"))]
        reorder_serial_3d(&mut entries, &mut indices, &order, items);

        let mut read_pos = 0usize;
        let mut write_pos = num_items;
        for &level_end in &level_bounds[..level_bounds.len() - 1] {
            while read_pos < level_end {
                let node_index = read_pos;
                let mut node_bounds = empty_box3d();
                let mut children = 0usize;
                while children < node_size && read_pos < level_end {
                    let entry = entries[read_pos];
                    extend_box3d(&mut node_bounds, entry);
                    read_pos += 1;
                    children += 1;
                }
                entries[write_pos] = node_bounds;
                indices[write_pos] = node_index;
                write_pos += 1;
            }
        }

        Ok(Index3D {
            node_size,
            num_items,
            level_bounds,
            entries,
            indices,
        })
    }
}

fn items_vec_with_root_capacity_3d(count: usize) -> Vec<Box3D> {
    let capacity = count.saturating_add(1);
    if capacity <= (isize::MAX as usize) / std::mem::size_of::<Box3D>() {
        Vec::with_capacity(capacity)
    } else {
        Vec::new()
    }
}

fn reorder_serial_3d(
    entries: &mut [Box3D],
    indices: &mut [usize],
    order: &[(u64, usize)],
    items: &[Box3D],
) {
    for (slot, &(_, original)) in order.iter().enumerate() {
        entries[slot] = items[original];
        indices[slot] = original;
    }
}

fn build_single_node_index_3d(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    mut entries: Vec<Box3D>,
) -> Index3D {
    let mut root = empty_box3d();
    for &entry in &entries {
        extend_box3d(&mut root, entry);
    }
    entries.push(root);

    let mut indices = Vec::with_capacity(num_items + 1);
    indices.extend(0..num_items);
    indices.push(0);

    Index3D {
        node_size,
        num_items,
        level_bounds,
        entries,
        indices,
    }
}

fn extent_3d(items: &[Box3D]) -> Box3D {
    let mut extent = empty_box3d();
    for &item in items {
        extend_box3d(&mut extent, item);
    }
    extent
}

#[cfg(feature = "parallel")]
fn reorder_parallel_3d(
    entries: &mut [Box3D],
    indices: &mut [usize],
    order: &[(u64, usize)],
    items: &[Box3D],
    num_items: usize,
) {
    use rayon::prelude::*;

    entries[..num_items]
        .par_iter_mut()
        .zip(indices[..num_items].par_iter_mut())
        .zip(order.par_iter())
        .for_each(|((slot_box, slot_idx), &(_, original))| {
            *slot_box = items[original];
            *slot_idx = original;
        });
}
