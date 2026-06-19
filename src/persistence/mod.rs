use crate::geometry::{Box2D, Box3D};

mod container;
mod errors;
mod metadata;
mod payload_chunk;
mod tree_chunk;

use self::container::{FORMAT_MAGIC, find_chunk, parse_container};
pub(crate) use container::{
    CHUNK_ENTRY_LEN, CHUNK_FLAG_CRITICAL, FORMAT_VERSION, SUPERBLOCK_LEN, plan_container,
};
pub use errors::{LoadError, PayloadError};
pub use metadata::{FileMetadata, read_metadata};
pub(crate) use metadata::{MetaFields, TAG_META};
pub(crate) use payload_chunk::{PYLD_DESC_LEN, PYLD_DESC_LEN_FIXED, TAG_PYLD, parse_pyld_chunk};
pub(crate) use tree_chunk::{TAG_TREE, TREE_DESC_LEN, parse_tree_chunk};

/// Validated slices of the optional trailing payload section. Borrowed by the
/// zero-copy views to serve `payload(id)`.
pub(crate) struct ParsedPayload<'a> {
    /// `(num_items + 1)` little-endian `u64` prefix offsets into `blobs`. Empty
    /// for a fixed-width payload (`stride != 0`), which needs no table.
    pub(crate) offsets: &'a [u8],
    /// Concatenated per-item payload bytes.
    pub(crate) blobs: &'a [u8],
    /// Fixed record stride in bytes, or `0` for a variable-width payload (use
    /// `offsets`). When non-zero every blob is exactly `stride` bytes, so the
    /// blob at leaf rank `r` is `blobs[r * stride ..][.. stride]`.
    pub(crate) stride: usize,
}

/// Slice the payload at leaf rank `r`: by arithmetic for a fixed-width payload,
/// or out of the leaf-ordered offset table for a variable-width one.
#[inline]
pub(crate) fn payload_slice<'a>(payload: &ParsedPayload<'a>, r: usize) -> &'a [u8] {
    if payload.stride != 0 {
        let start = r * payload.stride;
        &payload.blobs[start..start + payload.stride]
    } else {
        let start = read_u64_le_unchecked(payload.offsets, r * 8) as usize;
        let end = read_u64_le_unchecked(payload.offsets, (r + 1) * 8) as usize;
        &payload.blobs[start..end]
    }
}

/// Build the `insertion id -> leaf rank` map by inverting the leaf entries of
/// `indices` (which map leaf rank -> insertion id). The payload section is
/// leaf-ordered, so a view needs this to serve random `payload(id)` lookups.
pub(crate) fn build_id_to_leaf(indices: &[u8], num_items: usize) -> Vec<u32> {
    let mut id_to_leaf = vec![0u32; num_items];
    for r in 0..num_items {
        let id = read_u64_le_unchecked(indices, r * 8) as usize;
        id_to_leaf[id] = r as u32;
    }
    id_to_leaf
}

pub(crate) fn expected_tree_shape(
    num_items: usize,
    node_size: usize,
) -> Result<(usize, usize), LoadError> {
    let mut num_nodes = num_items;
    let mut levels = 1usize;
    let mut n = num_items;
    if num_items > 0 {
        loop {
            n = n.div_ceil(node_size);
            num_nodes = num_nodes.checked_add(n).ok_or(LoadError::IntegerOverflow)?;
            levels = levels.checked_add(1).ok_or(LoadError::IntegerOverflow)?;
            if n == 1 {
                break;
            }
        }
    }
    Ok((num_nodes, levels))
}

pub(crate) struct ByteWriter<'a> {
    bytes: &'a mut Vec<u8>,
    len: usize,
}

impl<'a> ByteWriter<'a> {
    /// Write into a caller-owned buffer. The buffer is cleared and reserved to at least
    /// `len` bytes; reusing the same buffer across calls amortizes the allocation and
    /// page-fault cost, which dominates serialization of large indexes. Every byte is
    /// written exactly once via `extend_from_slice`, so the buffer is never zero-filled.
    pub(crate) fn new(bytes: &'a mut Vec<u8>, len: usize) -> Self {
        bytes.clear();
        bytes.reserve_exact(len);
        Self { bytes, len }
    }

    pub(crate) fn write_magic(&mut self) {
        self.write_bytes(FORMAT_MAGIC);
    }

    #[inline]
    #[cfg(feature = "f32-storage")]
    pub(crate) fn write_f32(&mut self, value: f32) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Write 2D box records from f32 structure-of-arrays columns (one `[min_x,
    /// min_y, max_x, max_y]` record per node).
    #[cfg(feature = "f32-storage")]
    pub(crate) fn write_soa_boxes_f32_2d(
        &mut self,
        min_xs: &[f32],
        min_ys: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        for i in 0..min_xs.len() {
            self.write_f32(min_xs[i]);
            self.write_f32(min_ys[i]);
            self.write_f32(max_xs[i]);
            self.write_f32(max_ys[i]);
        }
    }

    /// Write 3D box records from f32 structure-of-arrays columns (one `[min_x,
    /// min_y, min_z, max_x, max_y, max_z]` record per node).
    #[cfg(feature = "f32-storage")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_soa_boxes_f32_3d(
        &mut self,
        min_xs: &[f32],
        min_ys: &[f32],
        min_zs: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        max_zs: &[f32],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), min_zs.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        debug_assert_eq!(min_xs.len(), max_zs.len());
        for i in 0..min_xs.len() {
            self.write_f32(min_xs[i]);
            self.write_f32(min_ys[i]);
            self.write_f32(min_zs[i]);
            self.write_f32(max_xs[i]);
            self.write_f32(max_ys[i]);
            self.write_f32(max_zs[i]);
        }
    }

    #[inline]
    pub(crate) fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    // Only the big-endian box-writing fallback uses this; on little-endian targets
    // the bulk memcpy path makes it dead, so suppress the lint there.
    #[inline]
    #[cfg_attr(target_endian = "little", allow(dead_code))]
    pub(crate) fn write_f64(&mut self, value: f64) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Write 2D boxes as little-endian box records.
    ///
    /// On little-endian targets the slice is copied in one bulk memcpy. Other targets
    /// fall back to per-field writes.
    #[inline]
    pub(crate) fn write_box2d_slice(&mut self, values: &[Box2D]) {
        #[cfg(target_endian = "little")]
        {
            debug_assert_eq!(
                core::mem::size_of::<Box2D>(),
                4 * core::mem::size_of::<f64>()
            );
            debug_assert_eq!(core::mem::align_of::<Box2D>(), core::mem::align_of::<f64>());
            // SAFETY: `Box2D` is `repr(C)` with exactly four contiguous `f64`
            // fields in the persisted order and no padding; on little-endian targets
            // those native bytes are the little-endian file encoding.
            unsafe { self.write_raw_slice_bytes(values) };
        }
        #[cfg(not(target_endian = "little"))]
        for item in values {
            self.write_f64(item.min_x);
            self.write_f64(item.min_y);
            self.write_f64(item.max_x);
            self.write_f64(item.max_y);
        }
    }

    /// Write 3D boxes as little-endian box records.
    ///
    /// On little-endian targets the slice is copied in one bulk memcpy. Other targets
    /// fall back to per-field writes.
    #[inline]
    pub(crate) fn write_box3d_slice(&mut self, values: &[Box3D]) {
        #[cfg(target_endian = "little")]
        {
            debug_assert_eq!(
                core::mem::size_of::<Box3D>(),
                6 * core::mem::size_of::<f64>()
            );
            debug_assert_eq!(core::mem::align_of::<Box3D>(), core::mem::align_of::<f64>());
            // SAFETY: `Box3D` is `repr(C)` with exactly six contiguous `f64`
            // fields in the persisted order and no padding; on little-endian targets
            // those native bytes are the little-endian file encoding.
            unsafe { self.write_raw_slice_bytes(values) };
        }
        #[cfg(not(target_endian = "little"))]
        for item in values {
            self.write_f64(item.min_x);
            self.write_f64(item.min_y);
            self.write_f64(item.min_z);
            self.write_f64(item.max_x);
            self.write_f64(item.max_y);
            self.write_f64(item.max_z);
        }
    }

    /// Write 2D nodes **interleaved**: each `[min_x, min_y, max_x, max_y]` box
    /// record immediately followed by its `u64` index entry. `entries` and
    /// `indices` have one element per node (same length). Produces the node data
    /// of an interleaved-layout `TREE` chunk, trading the bulk box memcpy for a
    /// layout a streaming reader fetches in one read per level.
    #[cfg(feature = "stream")]
    pub(crate) fn write_interleaved_2d(&mut self, entries: &[Box2D], indices: &[usize]) {
        debug_assert_eq!(entries.len(), indices.len());
        for (b, &idx) in entries.iter().zip(indices) {
            self.write_f64(b.min_x);
            self.write_f64(b.min_y);
            self.write_f64(b.max_x);
            self.write_f64(b.max_y);
            self.write_u64(idx as u64);
        }
    }

    /// Write 3D nodes **interleaved**: each `[min_x, min_y, min_z, max_x, max_y,
    /// max_z]` box record immediately followed by its `u64` index entry. See
    /// [`write_interleaved_2d`](Self::write_interleaved_2d).
    #[cfg(feature = "stream")]
    pub(crate) fn write_interleaved_3d(&mut self, entries: &[Box3D], indices: &[usize]) {
        debug_assert_eq!(entries.len(), indices.len());
        for (b, &idx) in entries.iter().zip(indices) {
            self.write_f64(b.min_x);
            self.write_f64(b.min_y);
            self.write_f64(b.min_z);
            self.write_f64(b.max_x);
            self.write_f64(b.max_y);
            self.write_f64(b.max_z);
            self.write_u64(idx as u64);
        }
    }

    /// Interleaved `f32` 2D nodes from SoA columns: each node's `[min_x, min_y,
    /// max_x, max_y]` f32 box immediately followed by its `u64` index.
    #[cfg(all(feature = "f32-storage", feature = "stream"))]
    pub(crate) fn write_interleaved_f32_2d(
        &mut self,
        min_xs: &[f32],
        min_ys: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        indices: &[usize],
    ) {
        for i in 0..indices.len() {
            self.write_f32(min_xs[i]);
            self.write_f32(min_ys[i]);
            self.write_f32(max_xs[i]);
            self.write_f32(max_ys[i]);
            self.write_u64(indices[i] as u64);
        }
    }

    /// Interleaved `f32` 3D nodes from SoA columns (24-byte box + `u64` index).
    #[cfg(all(feature = "f32-storage", feature = "stream"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_interleaved_f32_3d(
        &mut self,
        min_xs: &[f32],
        min_ys: &[f32],
        min_zs: &[f32],
        max_xs: &[f32],
        max_ys: &[f32],
        max_zs: &[f32],
        indices: &[usize],
    ) {
        for i in 0..indices.len() {
            self.write_f32(min_xs[i]);
            self.write_f32(min_ys[i]);
            self.write_f32(min_zs[i]);
            self.write_f32(max_xs[i]);
            self.write_f32(max_ys[i]);
            self.write_f32(max_zs[i]);
            self.write_u64(indices[i] as u64);
        }
    }

    /// Write 2D box records from structure-of-arrays columns (one `[min_x, min_y,
    /// max_x, max_y]` record per node). Produces the same bytes as
    /// [`write_box2d_slice`](Self::write_box2d_slice) on an equivalent AoS slice.
    #[cfg(feature = "simd")]
    pub(crate) fn write_soa_boxes_2d(
        &mut self,
        min_xs: &[f64],
        min_ys: &[f64],
        max_xs: &[f64],
        max_ys: &[f64],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        for i in 0..min_xs.len() {
            self.write_f64(min_xs[i]);
            self.write_f64(min_ys[i]);
            self.write_f64(max_xs[i]);
            self.write_f64(max_ys[i]);
        }
    }

    /// Write 3D box records from structure-of-arrays columns (one `[min_x, min_y,
    /// min_z, max_x, max_y, max_z]` record per node). Produces the same bytes as
    /// [`write_box3d_slice`](Self::write_box3d_slice) on an equivalent AoS slice.
    #[cfg(feature = "simd")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_soa_boxes_3d(
        &mut self,
        min_xs: &[f64],
        min_ys: &[f64],
        min_zs: &[f64],
        max_xs: &[f64],
        max_ys: &[f64],
        max_zs: &[f64],
    ) {
        debug_assert_eq!(min_xs.len(), min_ys.len());
        debug_assert_eq!(min_xs.len(), min_zs.len());
        debug_assert_eq!(min_xs.len(), max_xs.len());
        debug_assert_eq!(min_xs.len(), max_ys.len());
        debug_assert_eq!(min_xs.len(), max_zs.len());
        for i in 0..min_xs.len() {
            self.write_f64(min_xs[i]);
            self.write_f64(min_ys[i]);
            self.write_f64(min_zs[i]);
            self.write_f64(max_xs[i]);
            self.write_f64(max_ys[i]);
            self.write_f64(max_zs[i]);
        }
    }

    /// Write a slice of `usize` as little-endian `u64`. Bulk-copied on 64-bit
    /// little-endian targets (where `usize` and `u64` share a layout); other targets
    /// fall back to per-element widening writes.
    #[inline]
    pub(crate) fn write_usize_slice_as_u64(&mut self, values: &[usize]) {
        #[cfg(all(target_endian = "little", target_pointer_width = "64"))]
        {
            // SAFETY: on a 64-bit little-endian target `usize` has the same size and
            // byte order as `u64`, so the slice's bytes equal the LE encoding that the
            // per-element `write_u64` path would produce.
            unsafe { self.write_raw_slice_bytes(values) };
        }
        #[cfg(not(all(target_endian = "little", target_pointer_width = "64")))]
        for &value in values {
            self.write_u64(value as u64);
        }
    }

    /// Write the optional payload section in **leaf order**: a `(num_items + 1)`
    /// prefix-offset table plus the concatenated blobs, both ordered by leaf rank
    /// so a spatial query (which visits leaves in contiguous runs) fetches them
    /// in coalesced reads. `leaf_order[r]` is the insertion id of the item at
    /// leaf rank `r` (i.e. the leaf entry of `indices`); `payloads` is indexed by
    /// insertion id.
    pub(crate) fn write_payload_offsets_and_blobs<P: AsRef<[u8]>>(
        &mut self,
        payloads: &[P],
        leaf_order: &[usize],
    ) {
        let mut acc: u64 = 0;
        self.write_u64(0);
        for &id in leaf_order {
            acc += payloads[id].as_ref().len() as u64;
            self.write_u64(acc);
        }
        for &id in leaf_order {
            self.write_bytes(payloads[id].as_ref());
        }
    }

    /// Write a fixed-width payload section in **leaf order**: just the blobs,
    /// each exactly `stride` bytes, with no offset table (the reader addresses
    /// blob `r` at `r * stride`). `leaf_order[r]` is the insertion id at leaf
    /// rank `r`; callers must have validated every blob is `stride` bytes.
    pub(crate) fn write_payload_blobs_fixed(&mut self, payloads: &[&[u8]], leaf_order: &[usize]) {
        for &id in leaf_order {
            self.write_bytes(payloads[id]);
        }
    }

    pub(crate) fn finish(self) {
        debug_assert_eq!(self.bytes.len(), self.len);
    }

    #[inline]
    fn write_bytes(&mut self, source: &[u8]) {
        debug_assert!(self.bytes.len() + source.len() <= self.len);
        // Capacity is pre-reserved in `new`, so this appends without reallocating
        // and without re-zeroing the destination.
        self.bytes.extend_from_slice(source);
    }

    #[inline]
    unsafe fn write_raw_slice_bytes<T>(&mut self, values: &[T]) {
        // SAFETY: the caller guarantees that `values` can be persisted by copying its
        // native bytes directly. The slice itself is valid and initialized.
        let source = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        self.write_bytes(source);
    }
}

pub(crate) fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, LoadError> {
    let end = offset.checked_add(8).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

#[inline]
pub(crate) fn read_u64_le_unchecked(bytes: &[u8], offset: usize) -> u64 {
    debug_assert!(offset <= bytes.len());
    debug_assert!(bytes.len() - offset >= 8);

    let mut value = 0u64;
    // SAFETY: callers only use this for slices and offsets validated by
    // `parse_index`; unaligned byte buffers are copied into an aligned u64.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().add(offset),
            (&mut value as *mut u64).cast::<u8>(),
            8,
        );
    }
    u64::from_le(value)
}

#[inline]
pub(crate) fn read_f64_le_unchecked(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64_le_unchecked(bytes, offset))
}

#[inline]
#[cfg(any(feature = "f32-storage", feature = "stream"))]
pub(crate) fn read_f32_le_unchecked(bytes: &[u8], offset: usize) -> f32 {
    debug_assert!(offset + 4 <= bytes.len());
    let mut value = 0u32;
    // SAFETY: callers only use this for slices and offsets validated by
    // the f32 byte parsers; unaligned byte buffers are copied into an aligned u32.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().add(offset),
            (&mut value as *mut u32).cast::<u8>(),
            4,
        );
    }
    f32::from_bits(u32::from_le(value))
}

fn usize_from_u64(value: u64) -> Result<usize, LoadError> {
    usize::try_from(value).map_err(|_| LoadError::IntegerOverflow)
}

/// Frame a `TREE` chunk (+ optional `PYLD` / `META`) into a container in `out`.
/// The dimension-specific node bytes are written by `write_nodes`, called once
/// after the `TREE` descriptor; everything else (sizing, directory, alignment,
/// payload, metadata) is shared by the 2D and 3D serializers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_index_container(
    out: &mut Vec<u8>,
    dimensions: u8,
    coord_bytes: u8,
    interleaved: bool,
    num_items: usize,
    num_nodes: usize,
    node_size: usize,
    write_nodes: impl FnOnce(&mut ByteWriter),
    payloads: Option<&[&[u8]]>,
    record_stride: Option<u32>,
    leaf_order: &[usize],
    meta: &MetaFields<'_>,
) -> Result<(), PayloadError> {
    // Node bytes are `record + 8` per node regardless of layout (SoA splits them,
    // interleaved keeps them adjacent), so the TREE length is layout-independent.
    let record = dimensions as usize * 2 * coord_bytes as usize;
    let tree_len = TREE_DESC_LEN + num_nodes * (record + 8);

    let pyld_len = match payloads {
        Some(p) => {
            if p.len() != num_items {
                return Err(PayloadError::CountMismatch {
                    expected: num_items,
                    got: p.len(),
                });
            }
            match record_stride {
                // Fixed-width: every blob is `stride` bytes, no offset table.
                Some(stride) => {
                    let stride = stride as usize;
                    for b in p {
                        if b.len() != stride {
                            return Err(PayloadError::RecordSizeMismatch {
                                stride,
                                got: b.len(),
                            });
                        }
                    }
                    let blob_total = num_items
                        .checked_mul(stride)
                        .ok_or(PayloadError::TooLarge)?;
                    Some(PYLD_DESC_LEN_FIXED + blob_total)
                }
                // Variable-width: prefix-offset table plus the blobs.
                None => {
                    let mut blob_total: u64 = 0;
                    for b in p {
                        blob_total = blob_total
                            .checked_add(b.len() as u64)
                            .ok_or(PayloadError::TooLarge)?;
                    }
                    let blob_total =
                        usize::try_from(blob_total).map_err(|_| PayloadError::TooLarge)?;
                    Some(PYLD_DESC_LEN + (num_items + 1) * 8 + blob_total)
                }
            }
        }
        None => None,
    };
    let meta_len = (!meta.is_empty()).then(|| meta.content_len());

    // Chunks in write order: TREE, then optional PYLD, then optional META.
    let mut lens = vec![tree_len];
    if let Some(pl) = pyld_len {
        lens.push(pl);
    }
    if let Some(ml) = meta_len {
        lens.push(ml);
    }
    let (total, off) = plan_container(&lens).map_err(|_| PayloadError::TooLarge)?;
    let pyld_idx = pyld_len.map(|_| 1);
    let meta_idx = meta_len.map(|_| if pyld_len.is_some() { 2 } else { 1 });

    let mut bytes = ByteWriter::new(out, total);
    bytes.write_superblock(lens.len() as u32);
    bytes.write_chunk_entry(&TAG_TREE, true, off[0], tree_len);
    if let Some(i) = pyld_idx {
        bytes.write_chunk_entry(&TAG_PYLD, false, off[i], lens[i]);
    }
    if let Some(i) = meta_idx {
        bytes.write_chunk_entry(&TAG_META, false, off[i], lens[i]);
    }

    let mut pos = SUPERBLOCK_LEN + lens.len() * CHUNK_ENTRY_LEN;
    bytes.write_zeros(off[0] - pos);
    bytes.write_tree_desc(dimensions, coord_bytes, interleaved, num_items, node_size);
    write_nodes(&mut bytes);
    pos = off[0] + tree_len;
    if let (Some(i), Some(p)) = (pyld_idx, payloads) {
        bytes.write_zeros(off[i] - pos);
        bytes.write_pyld_desc(record_stride);
        match record_stride {
            Some(_) => bytes.write_payload_blobs_fixed(p, leaf_order),
            None => bytes.write_payload_offsets_and_blobs(p, leaf_order),
        }
        pos = off[i] + lens[i];
    }
    if let Some(i) = meta_idx {
        bytes.write_zeros(off[i] - pos);
        bytes.write_meta(meta);
        pos = off[i] + lens[i];
    }
    bytes.write_zeros(total - pos);
    bytes.finish();
    Ok(())
}

/// Primitive writers and chunk-body helpers. Container framing and descriptor
/// writers live next to their format parsers.
impl ByteWriter<'_> {
    pub(crate) fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_u8(&mut self, value: u8) {
        self.write_bytes(&[value]);
    }

    /// Append a raw byte slice. Used by the container round-trip tests to frame
    /// arbitrary chunk content; the real writers append typed fields directly.
    #[cfg(test)]
    pub(crate) fn write_raw(&mut self, bytes: &[u8]) {
        self.write_bytes(bytes);
    }

    /// Append `n` zero bytes (chunk descriptor reserved fields / alignment pads).
    pub(crate) fn write_zeros(&mut self, n: usize) {
        for _ in 0..n {
            self.write_bytes(&[0]);
        }
    }
}

pub(crate) fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, LoadError> {
    let end = offset.checked_add(4).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

pub(crate) fn read_u16_at(bytes: &[u8], offset: usize) -> Result<u16, LoadError> {
    let end = offset.checked_add(2).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

/// Derive the level-bound positions (cumulative node counts per level) from the
/// item count and node size. The format does not store them — they are a function of the
/// tree shape, so deriving avoids a second source of truth that could drift.
pub(crate) fn derive_level_bounds(
    num_items: usize,
    node_size: usize,
    level_count: usize,
) -> Vec<usize> {
    let mut bounds = Vec::with_capacity(level_count);
    let mut n = num_items;
    let mut total = n;
    bounds.push(total);
    while bounds.len() < level_count {
        n = n.div_ceil(node_size);
        total += n;
        bounds.push(total);
    }
    bounds
}

/// A parsed SoA tree: the node bytes (borrowed) plus the derived shape. The
/// `level_bounds` are owned because they are derived at load, not stored.
pub(crate) struct ParsedTree<'a> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) entries: &'a [u8],
    pub(crate) indices: &'a [u8],
}

/// Parse a container's `TREE` chunk for an in-memory (SoA) reader, plus the
/// optional payload. Rejects an interleaved tree (that layout is streaming-only).
pub(crate) fn parse_index(
    bytes: &[u8],
    dimensions: usize,
    coord_bytes: usize,
) -> Result<(ParsedTree<'_>, Option<ParsedPayload<'_>>), LoadError> {
    let chunks = parse_container(bytes, &[TAG_TREE])?;
    let tree_ref = find_chunk(&chunks, TAG_TREE).ok_or(LoadError::InvalidTree)?;
    let (desc, node_data) =
        parse_tree_chunk(&bytes[tree_ref.offset..tree_ref.offset + tree_ref.len])?;
    if desc.dimensions != dimensions || desc.coord_bytes != coord_bytes {
        return Err(LoadError::UnsupportedVersion);
    }
    if desc.interleaved {
        // The interleaved layout is read only by the streaming reader.
        return Err(LoadError::UnsupportedVersion);
    }

    let (num_nodes, level_count) = expected_tree_shape(desc.num_items, desc.node_size)?;
    let record = dimensions
        .checked_mul(2 * coord_bytes)
        .ok_or(LoadError::IntegerOverflow)?;
    let entries_len = num_nodes
        .checked_mul(record)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_len = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;
    let node_len = entries_len
        .checked_add(indices_len)
        .ok_or(LoadError::IntegerOverflow)?;
    if node_data.len() != node_len {
        return Err(LoadError::InvalidTree);
    }

    let parsed = ParsedTree {
        node_size: desc.node_size,
        num_items: desc.num_items,
        num_nodes,
        level_count,
        level_bounds: derive_level_bounds(desc.num_items, desc.node_size, level_count),
        entries: &node_data[..entries_len],
        indices: &node_data[entries_len..],
    };
    validate_tree_indices(&parsed)?;

    let payload = match find_chunk(&chunks, TAG_PYLD) {
        Some(p) => {
            let (pd, body) = parse_pyld_chunk(&bytes[p.offset..p.offset + p.len])?;
            Some(parse_payload_body(body, desc.num_items, pd.record_stride)?)
        }
        None => None,
    };
    Ok((parsed, payload))
}

/// Validate leaf and internal child pointers against the derived level bounds.
fn validate_tree_indices(p: &ParsedTree<'_>) -> Result<(), LoadError> {
    for pos in 0..p.num_items {
        let index = read_u64_at(p.indices, pos * 8).and_then(usize_from_u64)?;
        if index >= p.num_items {
            return Err(LoadError::InvalidTree);
        }
    }
    for level in 1..p.level_count {
        let level_start = p.level_bounds[level - 1];
        let level_end = p.level_bounds[level];
        let child_level_start = if level == 1 {
            0
        } else {
            p.level_bounds[level - 2]
        };
        let child_level_end = level_start;
        for pos in level_start..level_end {
            let index = read_u64_at(p.indices, pos * 8).and_then(usize_from_u64)?;
            if index < child_level_start || index >= child_level_end {
                return Err(LoadError::InvalidTree);
            }
            if (index - child_level_start) % p.node_size != 0 {
                return Err(LoadError::InvalidTree);
            }
        }
    }
    Ok(())
}

/// Validate and slice a `PYLD` chunk's post-descriptor bytes. A fixed-width
/// payload (`stride != 0`) is just `num_items * stride` blob bytes with no table;
/// a variable-width one is a `(num_items + 1)` prefix-offset table followed by the
/// blob region.
fn parse_payload_body(
    body: &[u8],
    num_items: usize,
    stride: usize,
) -> Result<ParsedPayload<'_>, LoadError> {
    if stride != 0 {
        let total = num_items
            .checked_mul(stride)
            .ok_or(LoadError::IntegerOverflow)?;
        if body.len() != total {
            return Err(LoadError::LengthMismatch {
                expected: total,
                actual: body.len(),
            });
        }
        return Ok(ParsedPayload {
            offsets: &[],
            blobs: body,
            stride,
        });
    }
    let offsets_len = num_items
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or(LoadError::IntegerOverflow)?;
    if body.len() < offsets_len {
        return Err(LoadError::Truncated);
    }
    let offsets = &body[..offsets_len];
    let mut prev = 0u64;
    for i in 0..=num_items {
        let off = read_u64_at(offsets, i * 8)?;
        if (i == 0 && off != 0) || off < prev {
            return Err(LoadError::InvalidTree);
        }
        prev = off;
    }
    let blob_total = usize_from_u64(prev)?;
    let blobs = &body[offsets_len..];
    if blobs.len() != blob_total {
        return Err(LoadError::LengthMismatch {
            expected: offsets_len + blob_total,
            actual: body.len(),
        });
    }
    Ok(ParsedPayload {
        offsets,
        blobs,
        stride: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Box2D, Index2D, Index2DBuilder};

    fn build(n: usize) -> Index2D {
        let mut builder = Index2DBuilder::new(n).node_size(16);
        for i in 0..n {
            let v = i as f64;
            builder.add(Box2D::new(v, v, v + 1.0, v + 1.0));
        }
        builder.finish().unwrap()
    }

    // ---- chunk container ----

    /// Frame chunks `(tag, critical, content)` into a container buffer.
    fn encode_container(chunks: &[([u8; 4], bool, Vec<u8>)]) -> Vec<u8> {
        let lens: Vec<usize> = chunks.iter().map(|(_, _, c)| c.len()).collect();
        let (total, offsets) = plan_container(&lens).unwrap();
        let mut buf = Vec::new();
        let mut w = ByteWriter::new(&mut buf, total);
        w.write_superblock(chunks.len() as u32);
        for (i, (tag, critical, c)) in chunks.iter().enumerate() {
            w.write_chunk_entry(tag, *critical, offsets[i], c.len());
        }
        let mut pos = SUPERBLOCK_LEN + chunks.len() * CHUNK_ENTRY_LEN;
        for (i, (_, _, c)) in chunks.iter().enumerate() {
            w.write_zeros(offsets[i] - pos);
            w.write_raw(c);
            pos = offsets[i] + c.len();
        }
        w.write_zeros(total - pos);
        w.finish();
        buf
    }

    fn tree_content(
        interleaved: bool,
        num_items: usize,
        node_size: usize,
        node_data: &[u8],
    ) -> Vec<u8> {
        let mut v = Vec::new();
        let mut w = ByteWriter::new(&mut v, TREE_DESC_LEN + node_data.len());
        w.write_tree_desc(2, 8, interleaved, num_items, node_size);
        w.write_raw(node_data);
        w.finish();
        v
    }

    #[test]
    fn v2_container_round_trips_tree_and_payload() {
        let node_data = vec![0xABu8; 40 * 5]; // 5 interleaved 2D nodes, arbitrary bytes
        let tree = tree_content(true, 4, 16, &node_data);

        let mut pyld = Vec::new();
        {
            let mut w = ByteWriter::new(&mut pyld, PYLD_DESC_LEN + 8 + b"blob".len());
            w.write_pyld_desc(None); // variable-width
            w.write_u64(0); // one-entry offset table fragment, just bytes here
            w.write_raw(b"blob");
            w.finish();
        }

        let buf = encode_container(&[(TAG_TREE, true, tree), (TAG_PYLD, false, pyld)]);

        let chunks = parse_container(&buf, &[TAG_TREE]).unwrap();
        assert_eq!(chunks.len(), 2);

        let tree_ref = find_chunk(&chunks, TAG_TREE).unwrap();
        let (desc, nd) =
            parse_tree_chunk(&buf[tree_ref.offset..tree_ref.offset + tree_ref.len]).unwrap();
        assert_eq!(desc.dimensions, 2);
        assert_eq!(desc.coord_bytes, 8);
        assert!(desc.interleaved);
        assert_eq!(desc.num_items, 4);
        assert_eq!(desc.node_size, 16);
        assert_eq!(nd, &node_data[..]);

        let pyld_ref = find_chunk(&chunks, TAG_PYLD).unwrap();
        let (_pd, body) =
            parse_pyld_chunk(&buf[pyld_ref.offset..pyld_ref.offset + pyld_ref.len]).unwrap();
        assert_eq!(&body[8..], b"blob"); // after the 8-byte table fragment
    }

    #[test]
    fn v2_unknown_critical_chunk_rejected() {
        let buf = encode_container(&[(*b"WHAT", true, vec![1, 2, 3, 4])]);
        // A reader that only knows TREE must reject an unknown critical chunk.
        assert_eq!(
            parse_container(&buf, &[TAG_TREE]).unwrap_err(),
            LoadError::UnsupportedVersion
        );
    }

    #[test]
    fn v2_unknown_optional_chunk_skipped() {
        let tree = tree_content(false, 4, 16, &[0u8; 32]);
        // An unknown OPTIONAL chunk is fine — the directory parses, reader ignores it.
        let buf = encode_container(&[(TAG_TREE, true, tree), (*b"XTRA", false, vec![9; 7])]);
        let chunks = parse_container(&buf, &[TAG_TREE]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(find_chunk(&chunks, TAG_TREE).is_some());
        assert!(find_chunk(&chunks, *b"XTRA").is_some());
    }

    #[test]
    fn v2_truncated_chunk_range_rejected() {
        let tree = tree_content(false, 4, 16, &[0u8; 16]);
        let mut buf = encode_container(&[(TAG_TREE, true, tree)]);
        buf.truncate(buf.len() - 4); // chunk now claims more bytes than exist
        assert!(matches!(
            parse_container(&buf, &[TAG_TREE]),
            Err(LoadError::InvalidTree | LoadError::Truncated)
        ));
    }

    #[test]
    fn payload_round_trip() {
        for &n in &[0usize, 1, 17, 100] {
            let index = build(n);
            let payloads: Vec<Vec<u8>> = (0..n).map(|i| format!("item-{i}").into_bytes()).collect();
            let bytes = index.to_bytes_with_payloads(&payloads).unwrap();

            // The TREE chunk is byte-identical to the index-only file's; only an
            // extra PYLD chunk (and the directory entry for it) is added.
            let index_only = index.to_bytes();
            let with = parse_container(&bytes, &[TAG_TREE]).unwrap();
            let plain = parse_container(&index_only, &[TAG_TREE]).unwrap();
            let wt = find_chunk(&with, TAG_TREE).unwrap();
            let pt = find_chunk(&plain, TAG_TREE).unwrap();
            assert_eq!(
                &bytes[wt.offset..wt.offset + wt.len],
                &index_only[pt.offset..pt.offset + pt.len]
            );
            assert!(find_chunk(&with, TAG_PYLD).is_some());

            let (_parsed, payload) = parse_index(&bytes, 2, 8).unwrap();
            let parsed = payload.expect("payload present");
            // The payload is leaf-ordered: slot `r` holds the blob of the item at
            // leaf rank `r`, whose insertion id is `index.indices[r]`.
            for r in 0..n {
                let insertion_id = index.indices[r];
                assert_eq!(payload_slice(&parsed, r), payloads[insertion_id].as_slice());
            }
        }
    }

    #[test]
    fn payload_count_mismatch_rejected() {
        let index = build(5);
        let payloads = vec![vec![1u8]; 3];
        assert_eq!(
            index.to_bytes_with_payloads(&payloads),
            Err(PayloadError::CountMismatch {
                expected: 5,
                got: 3
            })
        );
    }

    #[test]
    fn payload_file_loads_index_only_via_scalar_loader() {
        let index = build(10);
        let payloads: Vec<Vec<u8>> = (0..10).map(|_| vec![0u8; 4]).collect();
        let with = index.to_bytes_with_payloads(&payloads).unwrap();
        // The scalar loader reads the index from a payload file, ignoring the
        // payload (it validates the trailing section but does not retain it).
        let owned = Index2D::from_bytes(&with).unwrap();
        let query = Box2D::new(0.0, 0.0, 100.0, 100.0);
        assert_eq!(owned.search(query), index.search(query));
    }

    #[test]
    fn variable_length_payloads_round_trip() {
        let index = build(20);
        let payloads: Vec<Vec<u8>> = (0..20).map(|i| vec![i as u8; i]).collect(); // len 0..19
        let bytes = index.to_bytes_with_payloads(&payloads).unwrap();
        let (_parsed, payload) = parse_index(&bytes, 2, 8).unwrap();
        let parsed = payload.expect("payload present");
        for r in 0..20 {
            assert_eq!(
                payload_slice(&parsed, r),
                payloads[index.indices[r]].as_slice()
            );
        }
    }

    #[test]
    fn fixed_width_payload_round_trips_without_a_table() {
        const STRIDE: usize = 12;
        let n = 20;
        let index = build(n);
        // One 12-byte record per item, in item order.
        let mut flat = Vec::with_capacity(n * STRIDE);
        for i in 0..n {
            flat.extend_from_slice(&(i as u32).to_le_bytes());
            flat.extend_from_slice(&[i as u8; STRIDE - 4]);
        }
        let fixed = index.serialize().records(STRIDE, &flat).to_bytes().unwrap();
        let variable: Vec<Vec<u8>> = (0..n)
            .map(|i| flat[i * STRIDE..(i + 1) * STRIDE].to_vec())
            .collect();
        let var_bytes = index.to_bytes_with_payloads(&variable).unwrap();

        // The table-less layout is smaller by roughly the dropped offset table
        // (minus the 4-byte stride field the fixed descriptor adds, plus padding).
        let saving = var_bytes.len() - fixed.len();
        assert!((n * 8..=(n + 1) * 8).contains(&saving), "saving {saving}");

        let (_parsed, payload) = parse_index(&fixed, 2, 8).unwrap();
        let parsed = payload.expect("payload present");
        assert_eq!(parsed.stride, STRIDE);
        assert!(parsed.offsets.is_empty());
        for r in 0..n {
            let id = index.indices[r];
            assert_eq!(
                payload_slice(&parsed, r),
                &flat[id * STRIDE..(id + 1) * STRIDE]
            );
        }
    }

    #[test]
    fn fixed_width_wrong_record_size_rejected() {
        let index = build(4);
        // Three 8-byte records for four items, then declare stride 8: the count
        // (3 != 4) is caught first.
        let flat = vec![0u8; 3 * 8];
        assert!(matches!(
            index.serialize().records(8, &flat).to_bytes(),
            Err(PayloadError::CountMismatch { .. })
        ));
    }
}
