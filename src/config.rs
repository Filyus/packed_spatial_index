/// Default maximum number of children per tree node.
pub const DEFAULT_NODE_SIZE: usize = 16;
pub(crate) const DEFAULT_SEARCH_STACK_CAPACITY: usize = DEFAULT_NODE_SIZE;
pub(crate) const DEFAULT_NEIGHBOR_QUEUE_CAPACITY: usize = DEFAULT_NODE_SIZE;

/// Minimum index size at which `parallel(true)` enables rayon.
///
/// Set just above the measured serial/parallel build crossover (~30k items;
/// below it the thread-pool spin-up costs more than it saves). At 50k parallel
/// was ~1.13× faster, so 32k captures the 30–50k band while staying clear of the
/// noisy crossover. Override with
/// [`Index2DBuilder::parallel_min_items`](crate::Index2DBuilder::parallel_min_items)
/// — raise it if you build many small indexes back-to-back (avoid pool churn),
/// lower it toward 0 to always parallelize.
#[cfg(feature = "parallel")]
pub const DEFAULT_PARALLEL_MIN_ITEMS: usize = 32_000;
