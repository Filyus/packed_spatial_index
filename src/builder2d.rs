#[cfg(feature = "parallel")]
use crate::config::DEFAULT_PARALLEL_MIN_ITEMS;
use crate::{
    build::BuildError,
    config::DEFAULT_NODE_SIZE,
    geometry::{Box2D, Num, empty_box2d, extend_box2d},
    index2d::Index2D,
    sort2d::{
        DEFAULT_RADIX_BITS, SortKey2D, SortKey2DStrategy, SortKeyContext, encode_sort_by_key,
        normalize_radix_bits,
    },
    tree::{TreeLayout, normalize_node_size, try_compute_tree_layout},
};

/// Builder for [`Index2D`] and, with the `simd` feature, `SimdIndex2D`.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, Box2D};
///
/// let mut builder = Index2DBuilder::new(2);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
///
/// let index = builder.finish().unwrap();
/// assert_eq!(index.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
/// ```
#[must_use = "Index2DBuilder methods return an updated builder; assign the result or chain the call"]
pub struct Index2DBuilder {
    node_size: usize,
    num_items: usize,
    sort_key: SortKey2DStrategy,
    radix: bool,
    radix_bits: u32,
    #[cfg(feature = "parallel")]
    parallel: bool,
    #[cfg(feature = "parallel")]
    parallel_min_items: usize,
    items: Vec<Box2D>,
}

#[derive(Clone, Copy)]
#[cfg(feature = "simd")]
pub(crate) struct BuildConfig {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) sort_key: SortKey2DStrategy,
    pub(crate) radix: bool,
    pub(crate) radix_bits: u32,
    #[cfg(feature = "parallel")]
    pub(crate) parallel: bool,
    #[cfg(feature = "parallel")]
    pub(crate) parallel_min_items: usize,
}

impl Index2DBuilder {
    /// Create a builder for exactly `count` items with [`DEFAULT_NODE_SIZE`].
    pub fn new(count: usize) -> Self {
        Index2DBuilder {
            node_size: DEFAULT_NODE_SIZE,
            num_items: count,
            sort_key: SortKey2D::Hilbert.into(),
            radix: true,
            radix_bits: DEFAULT_RADIX_BITS,
            #[cfg(feature = "parallel")]
            parallel: false,
            #[cfg(feature = "parallel")]
            parallel_min_items: DEFAULT_PARALLEL_MIN_ITEMS,
            items: items_vec_with_root_capacity_2d(count),
        }
    }

    /// Set the maximum number of children per tree node (clamped to `[2, 65535]`).
    pub fn node_size(mut self, node_size: usize) -> Self {
        self.node_size = normalize_node_size(node_size);
        self
    }

    /// Choose the sort key (default: [`SortKey2D::Hilbert`]).
    pub fn sort_key(mut self, key: SortKey2D) -> Self {
        self.sort_key = key.into();
        self
    }

    /// Choose a sort-key implementation variant.
    #[doc(hidden)]
    pub fn sort_key_strategy(mut self, key: SortKey2DStrategy) -> Self {
        self.sort_key = key;
        self
    }

    /// Use LSD radix sort on the u32 Hilbert key instead of comparison-based sorting.
    #[doc(hidden)]
    pub fn radix(mut self, radix: bool) -> Self {
        self.radix = radix;
        self
    }

    /// Set the LSD radix-sort digit width for benchmarks and local performance tools.
    ///
    /// Values are clamped to `1..=16`; the default is 8.
    #[doc(hidden)]
    pub fn radix_sort_bits(mut self, bits: u32) -> Self {
        self.radix_bits = normalize_radix_bits(bits);
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
    pub fn add(&mut self, item: Box2D) {
        self.items.push(item);
    }

    /// Pack the tree and return the finished index.
    pub fn finish(self) -> Result<Index2D, BuildError> {
        if self.items.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.items.len(),
                expected: self.num_items,
            });
        }
        self.build()
    }

    /// Pack the tree into the SIMD-accelerated SoA index.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    ///
    /// let index = builder.finish_simd().unwrap();
    /// assert_eq!(index.search(Box2D::new(0.5, 0.5, 0.5, 0.5)), vec![0]);
    /// ```
    #[cfg(feature = "simd")]
    pub fn finish_simd(self) -> Result<crate::SimdIndex2D, BuildError> {
        if self.items.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.items.len(),
                expected: self.num_items,
            });
        }
        crate::index2d_soa::build_simd_index(self.config(), self.items)
    }

    /// Pack the tree into the f32-storage SIMD index.
    ///
    /// Coordinates are stored as `f32` rounded outward, halving box memory.
    /// [`search`](crate::SimdIndex2DF32::search) returns every exact hit, but
    /// may also include extra near-boundary hits. Use `search_exact` for exact
    /// range hits and `neighbors_exact` for exact KNN when the original f64
    /// boxes are available. Prefer f64 indexes for exact range queries with many
    /// hits and fastest exact KNN.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D};
    ///
    /// let mut builder = Index2DBuilder::new(1);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    ///
    /// let index = builder.finish_simd_f32().unwrap();
    /// assert!(index.search(Box2D::new(0.5, 0.5, 0.5, 0.5)).contains(&0));
    /// ```
    #[cfg(feature = "f32-storage")]
    pub fn finish_simd_f32(self) -> Result<crate::SimdIndex2DF32, BuildError> {
        if self.items.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.items.len(),
                expected: self.num_items,
            });
        }
        crate::index2d_f32::build_simd_index_f32(self.config(), self.items)
    }

    #[cfg(feature = "simd")]
    fn config(&self) -> BuildConfig {
        BuildConfig {
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

    fn build(self) -> Result<Index2D, BuildError> {
        let node_size = self.node_size;
        let num_items = self.num_items;
        let TreeLayout {
            level_bounds,
            num_nodes,
        } = try_compute_tree_layout(num_items, node_size)?;

        if num_items == 0 {
            return Ok(Index2D {
                node_size,
                num_items,
                level_bounds,
                entries: Vec::new(),
                indices: Vec::new(),
            });
        }

        if num_items <= node_size {
            return Ok(build_single_node_index(
                node_size,
                num_items,
                level_bounds,
                self.items,
            ));
        }

        let mut entries: Vec<Box2D> = vec![Box2D::new(0.0, 0.0, 0.0, 0.0); num_nodes];
        let mut indices: Vec<usize> = vec![0usize; num_nodes];
        let items = &self.items;

        #[cfg(feature = "parallel")]
        let use_parallel = self.parallel && num_items >= self.parallel_min_items;

        let mut min_x = Num::INFINITY;
        let mut min_y = Num::INFINITY;
        let mut max_x = Num::NEG_INFINITY;
        let mut max_y = Num::NEG_INFINITY;
        for b in items {
            min_x = min_x.min(b.min_x);
            min_y = min_y.min(b.min_y);
            max_x = max_x.max(b.max_x);
            max_y = max_y.max(b.max_y);
        }

        let scaled_width = u16::MAX as f64 / (max_x - min_x);
        let scaled_height = u16::MAX as f64 / (max_y - min_y);
        let context = SortKeyContext {
            scaled_width,
            scaled_height,
            min_x,
            min_y,
            radix: self.radix,
            radix_bits: self.radix_bits,
            #[cfg(feature = "parallel")]
            use_parallel,
        };

        let order = encode_sort_by_key(items, self.sort_key, context);

        #[cfg(feature = "parallel")]
        if use_parallel {
            reorder_parallel(&mut entries, &mut indices, &order, items, num_items);
        } else {
            reorder_serial(&mut entries, &mut indices, &order, items);
        }
        #[cfg(not(feature = "parallel"))]
        reorder_serial(&mut entries, &mut indices, &order, items);

        let mut read_pos = 0usize;
        let mut write_pos = num_items;
        for &level_end in &level_bounds[0..level_bounds.len() - 1] {
            while read_pos < level_end {
                let node_index = read_pos;
                let mut node_bounds = empty_box2d();
                let mut j = 0;
                while j < node_size && read_pos < level_end {
                    let b = entries[read_pos];
                    read_pos += 1;
                    extend_box2d(&mut node_bounds, b);
                    j += 1;
                }
                entries[write_pos] = node_bounds;
                indices[write_pos] = node_index;
                write_pos += 1;
            }
        }

        Ok(Index2D {
            node_size,
            num_items,
            level_bounds,
            entries,
            indices,
        })
    }
}

fn items_vec_with_root_capacity_2d(count: usize) -> Vec<Box2D> {
    let capacity = count.saturating_add(1);
    if capacity <= (isize::MAX as usize) / std::mem::size_of::<Box2D>() {
        Vec::with_capacity(capacity)
    } else {
        Vec::new()
    }
}

fn reorder_serial(
    entries: &mut [Box2D],
    indices: &mut [usize],
    order: &[(u32, u32)],
    items: &[Box2D],
) {
    for (slot, &(_, orig)) in order.iter().enumerate() {
        entries[slot] = items[orig as usize];
        indices[slot] = orig as usize;
    }
}

fn build_single_node_index(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    mut entries: Vec<Box2D>,
) -> Index2D {
    let mut root = empty_box2d();
    for &b in &entries {
        extend_box2d(&mut root, b);
    }
    entries.push(root);

    let mut indices = Vec::with_capacity(num_items + 1);
    indices.extend(0..num_items);
    indices.push(0);

    Index2D {
        node_size,
        num_items,
        level_bounds,
        entries,
        indices,
    }
}

#[cfg(feature = "parallel")]
fn reorder_parallel(
    entries: &mut [Box2D],
    indices: &mut [usize],
    order: &[(u32, u32)],
    items: &[Box2D],
    num_items: usize,
) {
    use rayon::prelude::*;

    entries[..num_items]
        .par_iter_mut()
        .zip(indices[..num_items].par_iter_mut())
        .zip(order.par_iter())
        .for_each(|((slot_box, slot_idx), &(_, orig))| {
            *slot_box = items[orig as usize];
            *slot_idx = orig as usize;
        });
}
