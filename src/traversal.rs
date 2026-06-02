/// Reusable buffers for allocation-free repeated searches.
///
/// Use this when running many searches against the same index to reuse the
/// result vector and traversal stack.
///
/// # Example
///
/// ```
/// use packed_spatial_index::{Index2DBuilder, Bounds2D, SearchWorkspace};
///
/// let mut builder = Index2DBuilder::new(1);
/// builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
/// let index = builder.finish().unwrap();
///
/// let mut workspace = SearchWorkspace::new();
/// let hits = index.search_with(Bounds2D::new(0.5, 0.5, 0.5, 0.5), &mut workspace);
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
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }

    #[cfg(target_arch = "x86")]
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
