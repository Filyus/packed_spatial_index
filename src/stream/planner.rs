use crate::persistence::LoadError;

use super::{StreamError, read_index};

/// A coalesced run to read from a section: the byte `offset`/`len` to fetch and
/// where each record lands (`(out index, byte offset within the run)`).
pub(super) struct GatherRun {
    pub(super) offset: u64,
    pub(super) len: usize,
    scatter: Vec<(usize, usize)>,
}

/// Plan the reads to gather `stride`-byte records for `positions` (sorted) from
/// the section at `section0`. Records covered by the directory `cache` are
/// copied into `out` immediately; the rest become coalesced [`GatherRun`]s for
/// the driver to read and scatter. `out` is cleared and sized to hold all
/// records in order.
pub(super) fn plan_gather(
    positions: &[usize],
    section0: u64,
    stride: usize,
    dir_node_start: usize,
    cache: &[u8],
    out: &mut Vec<u8>,
    max_gap: u64,
) -> Vec<GatherRun> {
    // The coalescing below (and the unchecked record reads its callers do into
    // `out`) assume positions are strictly ascending: `expand_frontier` sorts and
    // dedups every frontier, so this holds for a well-formed traversal. The assert
    // pins that contract so a future change cannot silently break the run gaps.
    debug_assert!(
        positions.windows(2).all(|w| w[0] < w[1]),
        "gather positions must be strictly ascending"
    );
    out.clear();
    out.resize(positions.len() * stride, 0);
    let mut streamed: Vec<(usize, usize)> = Vec::new();
    for (i, &pos) in positions.iter().enumerate() {
        if pos >= dir_node_start {
            let src = (pos - dir_node_start) * stride;
            out[i * stride..i * stride + stride].copy_from_slice(&cache[src..src + stride]);
        } else {
            streamed.push((i, pos));
        }
    }

    let mut runs = Vec::new();
    let mut j = 0;
    while j < streamed.len() {
        let lo = section0 + (streamed[j].1 * stride) as u64;
        let mut k = j;
        let mut end_pos = streamed[j].1 + 1;
        while k + 1 < streamed.len() {
            let next_pos = streamed[k + 1].1;
            let gap = (next_pos - end_pos) as u64 * stride as u64;
            if gap > max_gap {
                break;
            }
            k += 1;
            end_pos = next_pos + 1;
        }
        let hi = section0 + (end_pos * stride) as u64;
        let scatter = streamed[j..=k]
            .iter()
            .map(|&(out_i, pos)| (out_i, (section0 + (pos * stride) as u64 - lo) as usize))
            .collect();
        runs.push(GatherRun {
            offset: lo,
            len: (hi - lo) as usize,
            scatter,
        });
        j = k + 1;
    }
    runs
}

/// Scatter a run's fetched bytes into `out` at `stride`-byte records.
pub(super) fn apply_gather_run(out: &mut [u8], run: &GatherRun, buf: &[u8], stride: usize) {
    for &(out_i, within) in &run.scatter {
        out[out_i * stride..out_i * stride + stride].copy_from_slice(&buf[within..within + stride]);
    }
}

/// Expand surviving internal nodes into the next-level frontier, validating
/// child pointers against an untrusted source and sorting/deduping the result
/// (which keeps `plan_gather` fed ascending positions and caps the frontier at
/// the level width). `indices` holds the survivors' gathered child pointers.
pub(super) fn expand_frontier(
    level_bounds: &[usize],
    node_size: usize,
    level: usize,
    survivors_count: usize,
    indices: &[u8],
) -> Result<Vec<usize>, StreamError> {
    let child_level_end = level_bounds[level - 1];
    let child_level_start = if level >= 2 {
        level_bounds[level - 2]
    } else {
        0
    };
    let mut next = Vec::new();
    for i in 0..survivors_count {
        let child0 = read_index(indices, i)?;
        if child0 < child_level_start
            || child0 >= child_level_end
            || (child0 - child_level_start) % node_size != 0
        {
            return Err(StreamError::Format(LoadError::InvalidTree));
        }
        let end = (child0 + node_size).min(child_level_end);
        next.extend(child0..end);
    }
    next.sort_unstable();
    next.dedup();
    Ok(next)
}
