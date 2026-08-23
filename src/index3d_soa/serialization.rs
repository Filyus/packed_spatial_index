use super::SimdIndex3D;
use crate::persistence::{
    ByteWriter, CHUNK_ENTRY_LEN, LoadError, SUPERBLOCK_LEN, TAG_TREE, TREE_DESC_LEN, parse_index,
    plan_container, read_f64_le_unchecked, read_u64_le_unchecked,
};

impl SimdIndex3D {
    /// Serialize into the stable little-endian `PSINDEX` 3D format.
    ///
    /// The output is byte-identical to [`Index3D::to_bytes`](crate::Index3D::to_bytes)
    /// for the same items, so a `SimdIndex3D` and an `Index3D` are interchangeable on
    /// disk: either can load bytes produced by the other.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.to_bytes_into(&mut out);
        out
    }

    /// Serialize into a caller-provided buffer, reusing its allocation.
    ///
    /// Equivalent to [`to_bytes`](Self::to_bytes) but writes into `out` (cleared first).
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        let num_nodes = self.min_xs.len();
        let tree_len = TREE_DESC_LEN + num_nodes * 48 + num_nodes * 8;
        let (total, off) = plan_container(&[tree_len]).expect("serialized index is too large");
        let mut bytes = ByteWriter::new(out, total);
        bytes.write_superblock(1);
        bytes.write_chunk_entry(&TAG_TREE, true, off[0], tree_len);
        bytes.write_zeros(off[0] - (SUPERBLOCK_LEN + CHUNK_ENTRY_LEN));
        bytes.write_tree_desc(3, 8, false, self.num_items, self.node_size);
        bytes.write_soa_boxes_3d(
            &self.min_xs,
            &self.min_ys,
            &self.min_zs,
            &self.max_xs,
            &self.max_ys,
            &self.max_zs,
        );
        bytes.write_usize_slice_as_u64(&self.indices);
        bytes.write_zeros(total - (off[0] + tree_len));
        bytes.finish();
    }

    /// Load a SIMD 3D index from bytes produced by [`to_bytes`](Self::to_bytes) or by
    /// [`Index3D::to_bytes`](crate::Index3D::to_bytes); the AoS box records are
    /// scattered into the structure-of-arrays columns.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 3, 8)?;
        if payload.is_some() {
            return Err(LoadError::PayloadNotSupported);
        }
        let num_nodes = parsed.num_nodes;
        let level_bounds = parsed.level_bounds;

        // One `collect` per column. Each is an exact-size allocation written
        // through a pointer with no per-element capacity check, and each loop
        // holds two live pointers instead of the seven a single zipped pass needs
        // (which spilled: see the measurement in the commit message).
        debug_assert_eq!(parsed.entries.len(), num_nodes * 48);
        debug_assert_eq!(parsed.indices.len(), num_nodes * 8);
        let records = parsed.entries.as_chunks::<48>().0;
        let min_xs: Vec<f64> = records
            .iter()
            .map(|r| read_f64_le_unchecked(r, 0))
            .collect();
        let min_ys: Vec<f64> = records
            .iter()
            .map(|r| read_f64_le_unchecked(r, 8))
            .collect();
        let min_zs: Vec<f64> = records
            .iter()
            .map(|r| read_f64_le_unchecked(r, 16))
            .collect();
        let max_xs: Vec<f64> = records
            .iter()
            .map(|r| read_f64_le_unchecked(r, 24))
            .collect();
        let max_ys: Vec<f64> = records
            .iter()
            .map(|r| read_f64_le_unchecked(r, 32))
            .collect();
        let max_zs: Vec<f64> = records
            .iter()
            .map(|r| read_f64_le_unchecked(r, 40))
            .collect();
        let indices: Vec<usize> = parsed
            .indices
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| read_u64_le_unchecked(b, 0) as usize)
            .collect();

        Ok(SimdIndex3D {
            node_size: parsed.node_size,
            num_items: parsed.num_items,
            level_bounds,
            min_xs,
            min_ys,
            min_zs,
            max_xs,
            max_ys,
            max_zs,
            indices,
        })
    }
}
