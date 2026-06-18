/// Default maximum number of children per tree node.
pub const DEFAULT_NODE_SIZE: usize = 16;
pub(crate) const DEFAULT_SEARCH_STACK_CAPACITY: usize = DEFAULT_NODE_SIZE;
pub(crate) const DEFAULT_NEIGHBOR_QUEUE_CAPACITY: usize = DEFAULT_NODE_SIZE;

/// Minimum index size at which `parallel(true)` enables rayon.
///
/// The measured serial/parallel build crossover is ~25–30k items (below it the
/// thread-pool spin-up costs more than it saves); 50k is a deliberately
/// conservative default so parallelism only kicks in once it clearly pays. Lower
/// it via [`Index2DBuilder::parallel_min_items`](crate::Index2DBuilder::parallel_min_items)
/// if you build many indexes in the 30–50k range.
#[cfg(feature = "parallel")]
pub const DEFAULT_PARALLEL_MIN_ITEMS: usize = 50_000;
