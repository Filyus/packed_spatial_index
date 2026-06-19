use std::sync::Arc;

use crate::persistence::LoadError;

use super::payload::PayloadSection;
use super::{StreamCore, StreamError, StreamLimits};

/// The reader-independent half of a [`StreamCore`]: validated header counts,
/// section offsets, parsed level bounds, and the cached upper-level directory.
///
/// Splitting this out lets a caller open an index once (paying the directory
/// reads) and then run many queries, each with a *fresh* reader, without
/// re-reading the directory. In a serverless setting (one R2/HTTP reader per
/// request) this removes the directory round-trips — the dominant per-query
/// latency — from every request after the first. Cloning is a cheap in-memory
/// copy of the cached directory bytes; no I/O.
#[derive(Clone)]
pub(crate) struct StreamCoreParts {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) record: usize,
    pub(crate) box_stride: usize,
    pub(crate) interleaved: bool,
    pub(crate) box0: u64,
    pub(crate) idx0: u64,
    pub(crate) dir_node_start: usize,
    pub(crate) dir_boxes: Arc<[u8]>,
    pub(crate) dir_indices: Arc<[u8]>,
    pub(crate) payload: Option<PayloadSection>,
}

/// Choose the first node position to cache in the directory: walk levels from
/// the top down while their combined node count stays within `budget`. Always
/// includes the top level; never the leaves unless the whole tree fits.
pub(super) fn directory_start(level_bounds: &[usize], level_count: usize, budget: usize) -> usize {
    // Node count of level `l` = level_bounds[l] - level_bounds[l-1] (or - 0).
    let width = |level: usize| -> usize {
        let end = level_bounds[level];
        let start = if level == 0 {
            0
        } else {
            level_bounds[level - 1]
        };
        end - start
    };

    let mut first_level = level_count - 1;
    let mut cached_nodes = width(first_level);
    while first_level > 0 {
        let next = first_level - 1;
        let next_width = width(next);
        if cached_nodes + next_width > budget {
            break;
        }
        cached_nodes += next_width;
        first_level = next;
    }

    if first_level == 0 {
        0
    } else {
        level_bounds[first_level - 1]
    }
}

/// A reusable, reader-independent streaming directory.
///
/// Split one off any opened stream index with `into_directory`, then rebuild a
/// fresh index from it with `from_directory` and a new reader — no I/O, so the
/// directory round-trips are paid once, not per query. The typical use is a
/// serverless handler that opens the index once, caches the directory, and
/// serves each request with a per-request reader (R2 / HTTP range), eliminating
/// the directory reads that otherwise dominate per-query latency.
///
/// A directory carries its dimension and precision, so reattaching it to a
/// mismatched index type (e.g. a 2D directory to [`StreamIndex3D`](super::StreamIndex3D))
/// returns [`StreamError::Format`] rather than misreading.
#[derive(Clone)]
pub struct StreamDirectory {
    pub(crate) parts: StreamCoreParts,
}

impl StreamDirectory {
    /// Number of indexed items.
    pub fn num_items(&self) -> usize {
        self.parts.num_items
    }

    /// Whether the index has no items.
    pub fn is_empty(&self) -> bool {
        self.parts.num_items == 0
    }

    /// Packed node size of the index.
    pub fn node_size(&self) -> usize {
        self.parts.node_size
    }

    /// Whether the index carries a payload section.
    pub fn has_payload(&self) -> bool {
        self.parts.payload.is_some()
    }

    pub(crate) fn reattach<R>(
        &self,
        reader: R,
        limits: StreamLimits,
        expected_record: usize,
    ) -> Result<StreamCore<R>, StreamError> {
        if self.parts.record != expected_record {
            return Err(StreamError::Format(LoadError::UnsupportedVersion));
        }
        Ok(StreamCore::from_parts(self.parts.clone(), reader, limits))
    }
}
