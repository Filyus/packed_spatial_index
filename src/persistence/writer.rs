use crate::geometry::{Box2D, Box3D};

use super::container::FORMAT_MAGIC;
use super::{
    CHUNK_ENTRY_LEN, MetaFields, PYLD_DESC_LEN, PYLD_DESC_LEN_FIXED, PayloadError, SUPERBLOCK_LEN,
    TAG_META, TAG_PYLD, TAG_TREE, TREE_DESC_LEN, plan_container,
};

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
    pub(super) fn write_bytes(&mut self, source: &[u8]) {
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
