use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

use crate::config::DEFAULT_SEARCH_STACK_CAPACITY;

thread_local! {
    /// One traversal stack per thread, lent to the entry points that do not
    /// take a [`SearchWorkspace`]. A `Vec::with_capacity` per query costs
    /// ~20-35 ns, which measured as 15-21% of a narrow or point query on 100k
    /// boxes; keeping the buffer across queries removes it.
    static SCRATCH_STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// A traversal stack borrowed from the per-thread cache for the duration of
/// one query, handed back on drop.
///
/// `take` moves the cached `Vec` out, so a query started from inside another
/// query's visitor (or region predicate) finds the cache empty and simply
/// grows its own; whichever finishes last leaves its buffer for the next
/// query. Nothing is shared across threads and nothing outlives the query.
pub(crate) struct ScratchStack(Vec<usize>);

impl ScratchStack {
    /// Borrow the thread's cached stack (cleared, with at least the default
    /// capacity), or a fresh one if the cache is in use or gone.
    #[inline]
    pub(crate) fn take() -> Self {
        let mut stack = SCRATCH_STACK
            .try_with(|cell| std::mem::take(&mut *cell.borrow_mut()))
            .unwrap_or_default();
        stack.clear();
        if stack.capacity() < DEFAULT_SEARCH_STACK_CAPACITY {
            stack.reserve(DEFAULT_SEARCH_STACK_CAPACITY);
        }
        Self(stack)
    }
}

impl Drop for ScratchStack {
    #[inline]
    fn drop(&mut self) {
        let stack = std::mem::take(&mut self.0);
        // Ignore a cache already torn down (thread exit) or busy (a query's
        // visitor is mid-`take`, which cannot happen — `take` holds the borrow
        // for one move only); the buffer is then just freed.
        let _ = SCRATCH_STACK.try_with(|cell| {
            if let Ok(mut cached) = cell.try_borrow_mut()
                && cached.capacity() < stack.capacity()
            {
                *cached = stack;
            }
        });
    }
}

impl Deref for ScratchStack {
    type Target = Vec<usize>;
    #[inline]
    fn deref(&self) -> &Vec<usize> {
        &self.0
    }
}

impl DerefMut for ScratchStack {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<usize> {
        &mut self.0
    }
}

/// Reusable buffers for allocation-free repeated searches.
///
/// Use this when running many searches against the same index to reuse the
/// result vector and traversal stack.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, Box2D, SearchWorkspace};
///
/// let mut builder = Index2DBuilder::new(1);
/// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
/// let index = builder.finish().unwrap();
///
/// let mut workspace = SearchWorkspace::new();
/// let hits = index.search_with(Box2D::new(0.5, 0.5, 0.5, 0.5), &mut workspace);
/// assert_eq!(hits, &[0]);
/// ```
#[derive(Debug, Default)]
pub struct SearchWorkspace {
    pub(crate) results: Vec<usize>,
    pub(crate) stack: Vec<usize>,
}

impl SearchWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a workspace with preallocated result and traversal-stack capacity.
    pub fn with_capacity(results: usize, stack: usize) -> Self {
        Self {
            results: Vec::with_capacity(results),
            stack: Vec::with_capacity(stack),
        }
    }

    /// Results from the latest `search_with` call.
    pub fn results(&self) -> &[usize] {
        &self.results
    }
}

#[inline]
pub(crate) fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is a pure cache hint that never reads or dereferences
    // `ptr`, so any pointer value (including dangling or out of bounds) is sound; it is
    // `unsafe` only because it is a target-feature intrinsic, and SSE is baseline on
    // x86-64.
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }

    #[cfg(target_arch = "x86")]
    // SAFETY: `_mm_prefetch` is a pure cache hint that never reads or dereferences
    // `ptr`, so any pointer value is sound; it is `unsafe` only because it is a
    // target-feature intrinsic.
    unsafe {
        use std::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

pub(crate) fn upper_bound_level(level_bounds: &[usize], node_index: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = level_bounds.len() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if level_bounds[mid] > node_index {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
