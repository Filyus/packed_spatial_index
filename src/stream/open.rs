//! Open-time planning shared by the sync and async streaming readers.
//!
//! Opening an index used to be a chain of dependent reads — superblock, chunk
//! directory, descriptors, then the cached upper levels — so a cold open over a
//! remote source cost one round trip per step. The planner here keeps the
//! validation in one place and says which byte ranges it still needs; the
//! driver (sync or async) fetches them and asks again. The first read is a
//! speculative head of [`StreamLimits::open_head_bytes`]. On a normal file it
//! holds the superblock, the chunk directory and the `TREE` descriptor; a small
//! file fits whole. What the head misses is wanted as one batch: the payload
//! descriptors and the directory together (plus, for the async reader, a
//! speculative read of the payload offset table's last entry). The async
//! reader issues a batch concurrently, so its cold open is two dependent round
//! trips instead of four to eight.

use crate::persistence::{
    CHUNK_ENTRY_LEN, CHUNK_FLAG_CRITICAL, FORMAT_VERSION, LoadError, PFIX_DESC_LEN, PYLD_DESC_LEN,
    PYLD_DESC_LEN_FIXED, SUPERBLOCK_LEN, TAG_PFIX, TAG_PYLD, TAG_TREE, TREE_DESC_LEN,
    derive_level_bounds, expected_tree_shape, parse_pfix_chunk, parse_pyld_chunk, parse_tree_chunk,
    read_u32_at, read_u64_at,
};

use super::StreamError;
use super::core::{align8_u64, checked_directory_span};
use super::directory::directory_start;
use super::limits::{StreamLimits, directory_node_budget};
use super::payload::{PayloadSection, PrefixSection};

/// Default for [`StreamLimits::open_head_bytes`]: the size of the speculative
/// first read at open. 16 KiB holds the superblock, the chunk directory and
/// the `TREE` descriptor of any ordinary file and the whole of a small one,
/// for about the cost of the 32-byte superblock read it replaces.
pub(super) const OPEN_HEAD_BYTES: u64 = 16 * 1024;

/// Length of the first read at open: the configured head clamped to the file
/// length when known and never shorter than the superblock.
pub(super) fn head_len(limits: &StreamLimits, file_len: Option<u64>) -> usize {
    let head = limits.open_head_bytes.unwrap_or(OPEN_HEAD_BYTES);
    let head = file_len.map_or(head, |len| head.min(len));
    (head.min(usize::MAX as u64) as usize).max(SUPERBLOCK_LEN)
}

/// Byte spans fetched so far while opening, keyed by absolute offset.
#[derive(Default)]
pub(super) struct OpenBytes {
    spans: Vec<(u64, Vec<u8>)>,
}

impl OpenBytes {
    pub(super) fn insert(&mut self, offset: u64, bytes: Vec<u8>) {
        self.spans.push((offset, bytes));
    }

    /// The span holding byte `pos`, if any.
    fn span_at(&self, pos: u64) -> Option<&(u64, Vec<u8>)> {
        self.spans
            .iter()
            .find(|(start, bytes)| *start <= pos && pos - *start < bytes.len() as u64)
    }

    /// Copy `[offset, offset + len)` out of the fetched spans, or `None` if any
    /// byte of it has not been fetched.
    fn get(&self, offset: u64, len: usize) -> Option<Vec<u8>> {
        let mut out = vec![0u8; len];
        let mut done = 0usize;
        while done < len {
            let pos = offset + done as u64;
            let (start, bytes) = self.span_at(pos)?;
            let within = (pos - start) as usize;
            let n = (bytes.len() - within).min(len - done);
            out[done..done + n].copy_from_slice(&bytes[within..within + n]);
            done += n;
        }
        Some(out)
    }

    /// Like [`get`](Self::get), but moves a span out when it is exactly the
    /// requested range, so a large directory is not copied.
    fn take(&mut self, offset: u64, len: usize) -> Option<Vec<u8>> {
        if let Some(i) = self
            .spans
            .iter()
            .position(|(start, bytes)| *start == offset && bytes.len() == len)
        {
            return Some(self.spans.swap_remove(i).1);
        }
        self.get(offset, len)
    }

    /// The parts of `[offset, offset + len)` not yet fetched.
    fn missing(&self, offset: u64, len: usize, out: &mut Vec<(u64, u64)>) {
        let end = offset + len as u64;
        let mut pos = offset;
        while pos < end {
            if let Some((start, bytes)) = self.span_at(pos) {
                pos = start + bytes.len() as u64;
                continue;
            }
            let next = self
                .spans
                .iter()
                .map(|(start, _)| *start)
                .filter(|&start| start > pos)
                .min()
                .unwrap_or(end)
                .min(end);
            out.push((pos, next));
            pos = next;
        }
    }
}

/// Collects the byte ranges the planner still needs and merges them into reads.
struct Wants<'a> {
    bytes: &'a OpenBytes,
    missing: Vec<(u64, u64)>,
}

impl<'a> Wants<'a> {
    fn new(bytes: &'a OpenBytes) -> Self {
        Self {
            bytes,
            missing: Vec::new(),
        }
    }

    /// Return the range if fetched, else record it as wanted.
    fn get(&mut self, offset: u64, len: usize) -> Option<Vec<u8>> {
        let got = self.bytes.get(offset, len);
        if got.is_none() {
            self.bytes.missing(offset, len, &mut self.missing);
        }
        got
    }

    /// Record the range as wanted without reading it.
    fn want(&mut self, offset: u64, len: usize) {
        self.bytes.missing(offset, len, &mut self.missing);
    }

    fn is_empty(&self) -> bool {
        self.missing.is_empty()
    }

    /// The wanted ranges as `(offset, len)` reads, sorted, with ranges no more
    /// than `gap` bytes apart merged into one.
    fn into_reads(mut self, gap: u64) -> Vec<(u64, usize)> {
        self.missing.sort_unstable();
        let mut reads: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in self.missing {
            match reads.last_mut() {
                Some((_, end)) if lo <= end.saturating_add(gap) => *end = (*end).max(hi),
                _ => reads.push((lo, hi)),
            }
        }
        reads
            .into_iter()
            .map(|(lo, hi)| (lo, (hi - lo) as usize))
            .collect()
    }
}

/// What opening still needs, or the validated layout once it has everything.
pub(super) enum OpenStep {
    /// Fetch these `(offset, len)` ranges, insert them and plan again. They
    /// do not depend on each other, so an async driver issues them at once.
    Read(Vec<(u64, usize)>),
    Ready(Box<OpenLayout>),
}

/// The validated shape of an opened index, with its directory bytes.
pub(super) struct OpenLayout {
    pub(super) node_size: usize,
    pub(super) num_items: usize,
    pub(super) num_nodes: usize,
    pub(super) level_count: usize,
    pub(super) level_bounds: Vec<usize>,
    pub(super) record: usize,
    pub(super) box_stride: usize,
    pub(super) interleaved: bool,
    pub(super) box0: u64,
    pub(super) idx0: u64,
    pub(super) dir_node_start: usize,
    pub(super) dir_boxes: Vec<u8>,
    pub(super) dir_indices: Vec<u8>,
    pub(super) payload: Option<PayloadSection>,
    pub(super) prefix: Option<PrefixSection>,
}

/// Validate as much of the index as the fetched `bytes` allow.
///
/// Returns [`OpenStep::Read`] with everything the next step needs that is
/// already locatable, so each call costs the driver one round trip. With
/// `speculate`, the variable-width payload's total blob length is fetched in
/// the same batch as the descriptors, at the offset the current writer puts it
/// (8-byte descriptor); a descriptor of another length costs one more batch.
/// The sync driver passes `false`: its reads are sequential anyway, so a guess
/// that misses only adds a read.
pub(super) fn plan_open(
    bytes: &mut OpenBytes,
    dimensions: usize,
    coord_bytes: usize,
    limits: &StreamLimits,
    file_len: Option<u64>,
    gap: u64,
    speculate: bool,
) -> Result<OpenStep, StreamError> {
    let mut wants = Wants::new(bytes);

    // Superblock: magic + version + chunk_count.
    let Some(head) = wants.get(0, SUPERBLOCK_LEN) else {
        return Ok(OpenStep::Read(wants.into_reads(gap)));
    };
    if &head[..8] != b"PSINDEX\0" {
        return Err(StreamError::Format(LoadError::BadMagic));
    }
    if u64::from_le_bytes(head[8..16].try_into().unwrap()) != FORMAT_VERSION {
        return Err(StreamError::Format(LoadError::UnsupportedVersion));
    }
    let chunk_count = read_u32_at(&head, 16)? as usize;
    let (dir_len, dir_end) = checked_directory_span(chunk_count, file_len)?;
    let Some(dir) = wants.get(SUPERBLOCK_LEN as u64, dir_len) else {
        return Ok(OpenStep::Read(wants.into_reads(gap)));
    };

    let mut max_end = dir_end;
    let mut tree: Option<(u64, u64)> = None;
    let mut pyld: Option<(u64, u64)> = None;
    let mut pfix: Option<(u64, u64)> = None;
    for i in 0..chunk_count {
        let base = i * CHUNK_ENTRY_LEN;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&dir[base..base + 4]);
        let flags = read_u32_at(&dir, base + 4)?;
        let offset = read_u64_at(&dir, base + 8)?;
        let len = read_u64_at(&dir, base + 16)?;
        let end = offset.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
        if file_len.is_some_and(|fl| end > fl) {
            return Err(StreamError::Format(LoadError::InvalidTree));
        }
        max_end = max_end.max(end);
        if tag == TAG_TREE {
            tree = Some((offset, len));
        } else if tag == TAG_PYLD {
            pyld = Some((offset, len));
        } else if tag == TAG_PFIX {
            pfix = Some((offset, len));
        } else if flags & CHUNK_FLAG_CRITICAL != 0 {
            return Err(StreamError::Format(LoadError::UnsupportedVersion));
        }
    }

    // Reject a file longer than the last chunk plus its alignment pad — a
    // stray trailing byte the directory does not account for.
    let aligned_end = align8_u64(max_end)?;
    if let Some(fl) = file_len
        && fl > aligned_end
    {
        return Err(StreamError::Format(LoadError::LengthMismatch {
            expected: max_end as usize,
            actual: fl as usize,
        }));
    }
    let (toff, tlen) = tree.ok_or(LoadError::InvalidTree)?;
    if tlen < TREE_DESC_LEN as u64 {
        return Err(StreamError::Format(LoadError::Truncated));
    }

    // The descriptors are all locatable now; want the ones the head missed in
    // one batch. The payload ones are read only if their chunk can hold them,
    // leaving the length errors to the checks below.
    let pyld_desc =
        pyld.map(|(poff, plen)| (poff, (PYLD_DESC_LEN_FIXED as u64).min(plen) as usize));
    let pfix_desc = match (pyld, pfix) {
        (Some(_), Some((poff, plen))) if plen >= PFIX_DESC_LEN as u64 => {
            Some((poff, PFIX_DESC_LEN))
        }
        _ => None,
    };
    let Some(desc) = wants.get(toff, TREE_DESC_LEN) else {
        for (offset, len) in pyld_desc.into_iter().chain(pfix_desc) {
            wants.want(offset, len);
        }
        return Ok(OpenStep::Read(wants.into_reads(gap)));
    };
    let (td, _) = parse_tree_chunk(&desc)?;
    if td.dimensions != dimensions || td.coord_bytes != coord_bytes {
        return Err(StreamError::Format(LoadError::UnsupportedVersion));
    }
    let (num_nodes, level_count) = expected_tree_shape(td.num_items, td.node_size)?;
    let record = dimensions
        .checked_mul(2 * coord_bytes)
        .ok_or(LoadError::IntegerOverflow)?;
    let box_stride = if td.interleaved { record + 8 } else { record };
    let box0 = toff + td.desc_len as u64;
    let node_len = num_nodes
        .checked_mul(box_stride + if td.interleaved { 0 } else { 8 })
        .ok_or(LoadError::IntegerOverflow)?;
    if tlen != td.desc_len as u64 + node_len as u64 {
        return Err(StreamError::Format(LoadError::InvalidTree));
    }
    let idx0 = if td.interleaved {
        box0
    } else {
        box0 + (num_nodes * record) as u64
    };
    let level_bounds = derive_level_bounds(td.num_items, td.node_size, level_count);

    // Directory: cache the upper levels (a contiguous suffix of the node
    // section) up to the byte budget. The interleaved layout carries indices
    // inside the node records, so the separate index cache is SoA only.
    let budget = directory_node_budget(limits, box_stride, td.interleaved);
    let dir_node_start = directory_start(&level_bounds, level_count, budget);
    let cached_nodes = num_nodes - dir_node_start;
    let dir_boxes_at = (
        box0 + (dir_node_start * box_stride) as u64,
        cached_nodes * box_stride,
    );
    let dir_indices_at = (
        idx0 + (dir_node_start * 8) as u64,
        if td.interleaved { 0 } else { cached_nodes * 8 },
    );
    wants.want(dir_boxes_at.0, dir_boxes_at.1);
    wants.want(dir_indices_at.0, dir_indices_at.1);

    // Optional payload chunk.
    let payload = match pyld {
        Some((poff, plen)) => {
            if plen < PYLD_DESC_LEN as u64 {
                return Err(StreamError::Format(LoadError::Truncated));
            }
            let dn = (PYLD_DESC_LEN_FIXED as u64).min(plen) as usize;
            let Some(pd) = wants.get(poff, dn) else {
                if let Some((offset, len)) = pfix_desc {
                    wants.want(offset, len);
                }
                // The offset table's last entry, where the current writer
                // puts it (8-byte variable-width descriptor).
                let last_end = (td.num_items as u64 + 2)
                    .checked_mul(8)
                    .and_then(|table| table.checked_add(PYLD_DESC_LEN as u64));
                if speculate && last_end.is_some_and(|end| end <= plen) {
                    let last_at = poff + PYLD_DESC_LEN as u64 + td.num_items as u64 * 8;
                    wants.want(last_at, 8);
                }
                return Ok(OpenStep::Read(wants.into_reads(gap)));
            };
            let (pdesc, _) = parse_pyld_chunk(&pd)?;
            let body0 = poff + pdesc.desc_len as u64;
            if pdesc.record_stride != 0 {
                let stride = pdesc.record_stride as u64;
                let blob_total = (td.num_items as u64)
                    .checked_mul(stride)
                    .ok_or(StreamError::Format(LoadError::IntegerOverflow))?;
                let need = pdesc.desc_len as u64 + blob_total;
                if plen != need {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                Some(PayloadSection {
                    offsets_start: 0,
                    blobs_start: body0,
                    blob_total,
                    stride,
                })
            } else {
                let offsets_start = body0;
                let last_at = offsets_start + (td.num_items as u64) * 8;
                let Some(last) = wants.get(last_at, 8) else {
                    if let Some((offset, len)) = pfix_desc {
                        wants.want(offset, len);
                    }
                    return Ok(OpenStep::Read(wants.into_reads(gap)));
                };
                let blob_total = u64::from_le_bytes(last.try_into().unwrap());
                let blobs_start = offsets_start + (td.num_items as u64 + 1) * 8;
                let need = pdesc.desc_len as u64 + (td.num_items as u64 + 1) * 8 + blob_total;
                if plen != need {
                    return Err(StreamError::Format(LoadError::InvalidTree));
                }
                Some(PayloadSection {
                    offsets_start,
                    blobs_start,
                    blob_total,
                    stride: 0,
                })
            }
        }
        None => None,
    };

    // The optional prefix section: a dense copy of the blob heads. Only
    // useful next to a payload, and only when its stride can satisfy the
    // requested prefix, both of which the scan re-checks per query.
    let prefix = match pfix {
        Some((poff, plen)) if payload.is_some() => {
            if plen < PFIX_DESC_LEN as u64 {
                return Err(StreamError::Format(LoadError::Truncated));
            }
            let Some(pd) = wants.get(poff, PFIX_DESC_LEN) else {
                return Ok(OpenStep::Read(wants.into_reads(gap)));
            };
            let desc = parse_pfix_chunk(&pd)?;
            let need = desc.desc_len as u64 + td.num_items as u64 * desc.record_stride as u64;
            if plen != need {
                return Err(StreamError::Format(LoadError::InvalidTree));
            }
            Some(PrefixSection {
                start: poff + desc.desc_len as u64,
                stride: desc.record_stride,
            })
        }
        _ => None,
    };

    if !wants.is_empty() {
        return Ok(OpenStep::Read(wants.into_reads(gap)));
    }
    let dir_boxes = bytes
        .take(dir_boxes_at.0, dir_boxes_at.1)
        .expect("directory boxes were fetched");
    let dir_indices = bytes
        .take(dir_indices_at.0, dir_indices_at.1)
        .expect("directory indices were fetched");
    Ok(OpenStep::Ready(Box::new(OpenLayout {
        node_size: td.node_size,
        num_items: td.num_items,
        num_nodes,
        level_count,
        level_bounds,
        record,
        box_stride,
        interleaved: td.interleaved,
        box0,
        idx0,
        dir_node_start,
        dir_boxes,
        dir_indices,
        payload,
        prefix,
    })))
}
