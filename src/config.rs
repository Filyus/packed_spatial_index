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

/// How far ahead the build's reorder gather prefetches its next source box.
///
/// The gather reads `items` in Hilbert order while writing its output
/// sequentially, so its loads are a permutation of a buffer far larger than L1
/// — but every address is already known, sitting in `order`, which makes it the
/// textbook case for a software prefetch.
///
/// Measured in isolation on a random permutation of 32-byte records (the shape
/// Hilbert order takes over randomly positioned input), median of 9 interleaved
/// rounds, pinned:
///
/// | distance | 100k  | 1M    |
/// | ---      | ---:  | ---:  |
/// | none     | 1.000 | 1.000 |
/// | 4        | 1.010 | 1.161 |
/// | 8        | 0.962 | 1.059 |
/// | 16       | 0.923 | 0.914 |
/// | 32       | 0.920 | 0.742 |
/// | 64       | 0.930 | 0.608 |
/// | 128      | 0.942 | 0.645 |
///
/// The optimum moves with the working set, so 64 is the compromise: a point
/// behind 32 at 100k and far ahead of it at 1M. Short distances are *worse*
/// than no prefetch at all — the hint lands too late to hide the miss and still
/// costs an instruction per item, which is why this is a measured constant
/// rather than a plausible one.
pub(crate) const GATHER_PREFETCH_DISTANCE: usize = 64;
