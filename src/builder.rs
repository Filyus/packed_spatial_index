use std::{error::Error, fmt};

use crate::{
    config::DEFAULT_NODE_SIZE,
    geometry::{Num, Rect},
    index::Index,
    sort::{
        DEFAULT_RADIX_BITS, ExperimentalSortKey, SortKey, encode_sort_serial, hilbert_coord,
        normalize_radix_bits,
    },
};
#[cfg(feature = "parallel")]
use crate::{config::DEFAULT_PARALLEL_MIN_ITEMS, sort::encode_sort_parallel};

/// Builder for [`Index`] and, with the `simd` feature, `SimdIndex`.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{IndexBuilder, Rect};
///
/// let mut builder = IndexBuilder::new(2);
/// builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
/// builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));
///
/// let index = builder.finish().unwrap();
/// assert_eq!(index.search(Rect::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
/// ```
#[must_use = "IndexBuilder methods return an updated builder; assign the result or chain the call"]
pub struct IndexBuilder {
    node_size: usize,
    num_items: usize,
    sort_key: ExperimentalSortKey,
    radix: bool,
    radix_bits: u32,
    #[cfg(feature = "parallel")]
    parallel: bool,
    #[cfg(feature = "parallel")]
    parallel_min_items: usize,
    boxes: Vec<Rect>,
}

#[derive(Clone, Copy)]
#[cfg(feature = "simd")]
pub(crate) struct BuildConfig {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) sort_key: ExperimentalSortKey,
    pub(crate) radix: bool,
    pub(crate) radix_bits: u32,
    #[cfg(feature = "parallel")]
    pub(crate) parallel: bool,
    #[cfg(feature = "parallel")]
    pub(crate) parallel_min_items: usize,
}

/// Build error for finishing an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The builder received the wrong number of items.
    ItemCount {
        /// Number actually added through `add`.
        added: usize,
        /// Expected by `IndexBuilder::new(count)`.
        expected: usize,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::ItemCount { added, expected } => write!(
                f,
                "added item count must match declared count (added {added}, expected {expected})"
            ),
        }
    }
}

impl Error for BuildError {}

impl IndexBuilder {
    /// Create a builder for exactly `count` items with [`DEFAULT_NODE_SIZE`].
    pub fn new(count: usize) -> Self {
        IndexBuilder {
            node_size: DEFAULT_NODE_SIZE,
            num_items: count,
            sort_key: SortKey::Hilbert.into(),
            radix: true,
            radix_bits: DEFAULT_RADIX_BITS,
            #[cfg(feature = "parallel")]
            parallel: false,
            #[cfg(feature = "parallel")]
            parallel_min_items: DEFAULT_PARALLEL_MIN_ITEMS,
            boxes: Vec::with_capacity(count.saturating_add(1)),
        }
    }

    /// Set the maximum number of children per tree node (clamped to `[2, 65535]`).
    pub fn node_size(mut self, node_size: usize) -> Self {
        self.node_size = node_size.clamp(2, 65535);
        self
    }

    /// Choose the sort key (default: [`SortKey::Hilbert`]).
    pub fn sort_key(mut self, key: SortKey) -> Self {
        self.sort_key = key.into();
        self
    }

    /// Choose an experimental sort-key implementation.
    #[doc(hidden)]
    pub fn experimental_sort_key(mut self, key: ExperimentalSortKey) -> Self {
        self.sort_key = key;
        self
    }

    /// Use LSD radix sort on the u32 Hilbert key instead of comparison-based sorting.
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

    /// Add a rectangle.
    #[inline]
    pub fn add(&mut self, rect: Rect) {
        self.boxes.push(rect);
    }

    /// Pack the tree and return the finished index.
    pub fn finish(self) -> Result<Index, BuildError> {
        if self.boxes.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.boxes.len(),
                expected: self.num_items,
            });
        }
        Ok(self.build_unchecked())
    }

    /// Pack the tree into the SIMD-accelerated SoA index.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{IndexBuilder, Rect};
    ///
    /// let mut builder = IndexBuilder::new(1);
    /// builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
    ///
    /// let index = builder.finish_simd().unwrap();
    /// assert_eq!(index.search(Rect::new(0.5, 0.5, 0.5, 0.5)), vec![0]);
    /// ```
    #[cfg(feature = "simd")]
    pub fn finish_simd(self) -> Result<crate::SimdIndex, BuildError> {
        if self.boxes.len() != self.num_items {
            return Err(BuildError::ItemCount {
                added: self.boxes.len(),
                expected: self.num_items,
            });
        }
        Ok(crate::index_soa::build_simd_index(
            self.config(),
            self.boxes,
        ))
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

    fn build_unchecked(self) -> Index {
        let node_size = self.node_size;
        let num_items = self.num_items;

        let mut level_bounds: Vec<usize> = Vec::new();
        let mut num_nodes = num_items;
        let mut n = num_items;
        level_bounds.push(n);
        if num_items > 0 {
            loop {
                n = (n as f64 / node_size as f64).ceil() as usize;
                num_nodes += n;
                level_bounds.push(num_nodes);
                if n == 1 {
                    break;
                }
            }
        }

        if num_items == 0 {
            return Index {
                node_size,
                num_items,
                level_bounds,
                boxes: Vec::new(),
                indices: Vec::new(),
            };
        }

        if num_items <= node_size {
            return build_single_node_index(node_size, num_items, level_bounds, self.boxes);
        }

        let mut boxes: Vec<Rect> = vec![Rect::new(0.0, 0.0, 0.0, 0.0); num_nodes];
        let mut indices: Vec<usize> = vec![0usize; num_nodes];
        let items = &self.boxes;

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
        let sort_key = self.sort_key;

        let encode = |i: usize, b: &Rect| -> (u32, u32) {
            let hx = hilbert_coord(scaled_width, b.min_x, b.max_x, min_x);
            let hy = hilbert_coord(scaled_height, b.min_y, b.max_y, min_y);
            (sort_key.encode(hx, hy), i as u32)
        };

        #[cfg(feature = "parallel")]
        let order: Vec<(u32, u32)> = if use_parallel {
            encode_sort_parallel(items, &encode)
        } else {
            encode_sort_serial(items, &encode, self.radix, self.radix_bits)
        };
        #[cfg(not(feature = "parallel"))]
        let order: Vec<(u32, u32)> =
            encode_sort_serial(items, &encode, self.radix, self.radix_bits);

        #[cfg(feature = "parallel")]
        if use_parallel {
            reorder_parallel(&mut boxes, &mut indices, &order, items, num_items);
        } else {
            reorder_serial(&mut boxes, &mut indices, &order, items);
        }
        #[cfg(not(feature = "parallel"))]
        reorder_serial(&mut boxes, &mut indices, &order, items);

        let mut read_pos = 0usize;
        let mut write_pos = num_items;
        for &level_end in &level_bounds[0..level_bounds.len() - 1] {
            while read_pos < level_end {
                let node_index = read_pos;
                let mut node_bounds = Rect::new(
                    Num::INFINITY,
                    Num::INFINITY,
                    Num::NEG_INFINITY,
                    Num::NEG_INFINITY,
                );
                let mut j = 0;
                while j < node_size && read_pos < level_end {
                    let b = boxes[read_pos];
                    read_pos += 1;
                    node_bounds.min_x = node_bounds.min_x.min(b.min_x);
                    node_bounds.min_y = node_bounds.min_y.min(b.min_y);
                    node_bounds.max_x = node_bounds.max_x.max(b.max_x);
                    node_bounds.max_y = node_bounds.max_y.max(b.max_y);
                    j += 1;
                }
                boxes[write_pos] = node_bounds;
                indices[write_pos] = node_index;
                write_pos += 1;
            }
        }

        Index {
            node_size,
            num_items,
            level_bounds,
            boxes,
            indices,
        }
    }
}

fn reorder_serial(boxes: &mut [Rect], indices: &mut [usize], order: &[(u32, u32)], items: &[Rect]) {
    for (slot, &(_, orig)) in order.iter().enumerate() {
        boxes[slot] = items[orig as usize];
        indices[slot] = orig as usize;
    }
}

fn build_single_node_index(
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    mut boxes: Vec<Rect>,
) -> Index {
    let mut root = Rect::new(
        Num::INFINITY,
        Num::INFINITY,
        Num::NEG_INFINITY,
        Num::NEG_INFINITY,
    );
    for b in &boxes {
        root.min_x = root.min_x.min(b.min_x);
        root.min_y = root.min_y.min(b.min_y);
        root.max_x = root.max_x.max(b.max_x);
        root.max_y = root.max_y.max(b.max_y);
    }
    boxes.push(root);

    let mut indices = Vec::with_capacity(num_items + 1);
    indices.extend(0..num_items);
    indices.push(0);

    Index {
        node_size,
        num_items,
        level_bounds,
        boxes,
        indices,
    }
}

#[cfg(feature = "parallel")]
fn reorder_parallel(
    boxes: &mut [Rect],
    indices: &mut [usize],
    order: &[(u32, u32)],
    items: &[Rect],
    num_items: usize,
) {
    use rayon::prelude::*;

    boxes[..num_items]
        .par_iter_mut()
        .zip(indices[..num_items].par_iter_mut())
        .zip(order.par_iter())
        .for_each(|((slot_box, slot_idx), &(_, orig))| {
            *slot_box = items[orig as usize];
            *slot_idx = orig as usize;
        });
}
