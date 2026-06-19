use super::StreamError;

/// Upper bound on how many nodes the open-time "directory" prefetch caches.
///
/// The tree is stored leaves-first with the root last, so the upper levels form
/// a contiguous suffix of the box section. We cache levels from the top down
/// while their combined node count stays within this budget; queries then reach
/// those levels with zero I/O and stream only the levels below. 8192 nodes is a
/// few hundred KiB of boxes — small to hold, yet enough to cover every level
/// above the leaves for indexes into the millions of items.
const DIRECTORY_NODE_BUDGET: usize = 8192;

/// Default for [`StreamLimits::coalesce_gap_bytes`]: records whose byte gap is no
/// larger than this are fetched in a single read. Coalescing trades a little
/// re-read for far fewer round trips, which dominates on high-latency (e.g. HTTP)
/// sources. A caller on a remote source can raise the limit to collapse more.
pub(super) const COALESCE_GAP_BYTES: u64 = 4096;

/// Per-query cost limits for a streaming index.
///
/// All fields are optional; `None` is unbounded (the default). The caller picks
/// values to fit its environment — for example a Cloudflare Worker bounds reads
/// by its subrequest limit and bytes/items by its memory budget. A query that
/// would exceed any limit aborts with [`StreamError::LimitExceeded`] instead of
/// running unbounded over a broad window.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamLimits {
    /// Maximum number of range reads a single query may issue.
    pub max_reads: Option<usize>,
    /// Maximum total bytes a single query may read.
    pub max_read_bytes: Option<u64>,
    /// Maximum number of items a single query may return.
    pub max_items: Option<usize>,
    /// Open-time only: how many bytes the cached upper-level directory may use.
    /// `None` keeps the small built-in default; raising it caches more (or all)
    /// internal levels, so descent costs fewer round-trips per query — trade
    /// memory for latency where memory is plentiful (e.g. a serverless isolate).
    /// Read once at `open`; ignored by `from_directory` and by per-query cost
    /// checks.
    pub directory_budget_bytes: Option<u64>,
    /// Max byte gap between two records (tree nodes or payload blobs) still
    /// fetched in one read. `None` keeps the small built-in default. Raising it
    /// over-reads the gaps to collapse round-trips, which dominates latency on a
    /// remote source: a wider window is a strong win there and pure waste on a
    /// local file. Bounded by `max_read_bytes`, so a broad query still aborts
    /// rather than over-reading unbounded.
    pub coalesce_gap_bytes: Option<u64>,
}

/// Running per-query cost counters checked against [`StreamLimits`].
pub(super) struct Budget {
    limits: StreamLimits,
    reads: usize,
    bytes: u64,
    items: usize,
}

impl Budget {
    pub(super) fn new(limits: StreamLimits) -> Self {
        Self {
            limits,
            reads: 0,
            bytes: 0,
            items: 0,
        }
    }

    /// Account for one read of `len` bytes; call before issuing it so an
    /// over-budget read is never performed.
    pub(super) fn charge_read(&mut self, len: usize) -> Result<(), StreamError> {
        self.reads += 1;
        self.bytes += len as u64;
        if self.limits.max_reads.is_some_and(|m| self.reads > m)
            || self.limits.max_read_bytes.is_some_and(|m| self.bytes > m)
        {
            return Err(StreamError::LimitExceeded);
        }
        Ok(())
    }

    /// Account for one returned item; call before emitting it.
    pub(super) fn charge_item(&mut self) -> Result<(), StreamError> {
        self.items += 1;
        if self.limits.max_items.is_some_and(|m| self.items > m) {
            return Err(StreamError::LimitExceeded);
        }
        Ok(())
    }
}

/// Node count the directory may cache: the caller's byte budget divided by the
/// per-node cache cost, or the small built-in default when unset. Per-node cost
/// is the box record plus 8 bytes for the separate SoA index (0 interleaved).
pub(super) fn directory_node_budget(
    limits: &StreamLimits,
    box_stride: usize,
    interleaved: bool,
) -> usize {
    match limits.directory_budget_bytes {
        Some(bytes) => {
            let per_node = (box_stride + if interleaved { 0 } else { 8 }).max(1) as u64;
            (bytes / per_node).min(usize::MAX as u64) as usize
        }
        None => DIRECTORY_NODE_BUDGET,
    }
}
