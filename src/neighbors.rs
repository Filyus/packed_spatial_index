use std::collections::BinaryHeap;

/// Reusable buffers for allocation-free repeated nearest-neighbor searches.
#[derive(Debug, Default)]
pub struct NeighborWorkspace {
    pub(crate) results: Vec<usize>,
    pub(crate) queue: BinaryHeap<NeighborState>,
    pub(crate) node_queue: BinaryHeap<NeighborNodeState>,
}

impl NeighborWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a workspace with preallocated result and priority-queue capacity.
    pub fn with_capacity(results: usize, queue: usize) -> Self {
        Self {
            results: Vec::with_capacity(results),
            queue: BinaryHeap::with_capacity(queue),
            node_queue: BinaryHeap::with_capacity(queue),
        }
    }

    /// Results from the latest `neighbors_with` call.
    pub fn results(&self) -> &[usize] {
        &self.results
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NeighborState {
    pub(crate) index: usize,
    pub(crate) is_leaf: bool,
    pub(crate) dist: f64,
}

impl NeighborState {
    #[inline]
    pub(crate) fn new(index: usize, is_leaf: bool, dist: f64) -> Self {
        Self {
            index,
            is_leaf,
            dist,
        }
    }
}

impl Eq for NeighborState {}

impl Ord for NeighborState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| self.is_leaf.cmp(&other.is_leaf))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for NeighborState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NeighborNodeState {
    pub(crate) index: usize,
    pub(crate) dist: f64,
}

impl NeighborNodeState {
    #[inline]
    pub(crate) fn new(index: usize, dist: f64) -> Self {
        Self { index, dist }
    }
}

impl Eq for NeighborNodeState {}

impl Ord for NeighborNodeState {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for NeighborNodeState {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn max_distance_squared(max_distance: f64) -> Option<f64> {
    if max_distance.is_nan() || max_distance.is_sign_negative() {
        None
    } else {
        Some(max_distance * max_distance)
    }
}
