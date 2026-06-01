/// Default maximum number of children per tree node.
pub const DEFAULT_NODE_SIZE: usize = 16;
pub(crate) const DEFAULT_SEARCH_STACK_CAPACITY: usize = DEFAULT_NODE_SIZE;
pub(crate) const DEFAULT_NEIGHBOR_QUEUE_CAPACITY: usize = DEFAULT_NODE_SIZE;

/// Minimum index size at which `parallel(true)` enables rayon.
#[cfg(feature = "parallel")]
pub const DEFAULT_PARALLEL_MIN_ITEMS: usize = 50_000;
