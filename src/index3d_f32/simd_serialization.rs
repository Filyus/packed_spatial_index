use crate::{
    f32_storage::{F32ColumnRefs3D, columns3d_from_parsed, write_columns3d},
    persistence::{LoadError, MetaFields, parse_index},
};

use super::{SimdIndex3DF32, index3d_from_columns};

impl SimdIndex3DF32 {
    /// Serialize into the little-endian `PSINDEX` format (f32 box records).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.to_bytes_into(&mut out);
        out
    }

    /// Serialize into a caller-provided buffer, reusing its allocation.
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        write_columns3d(
            out,
            F32ColumnRefs3D {
                node_size: self.node_size,
                num_items: self.num_items,
                min_xs: &self.min_xs,
                min_ys: &self.min_ys,
                min_zs: &self.min_zs,
                max_xs: &self.max_xs,
                max_ys: &self.max_ys,
                max_zs: &self.max_zs,
                indices: &self.indices,
            },
            false,
            None,
            None,
            &MetaFields::default(),
        )
        .expect("index-only serialization cannot fail");
    }

    /// Load from bytes produced by [`to_bytes`](Self::to_bytes). A payload
    /// section is rejected (this SIMD index carries boxes only).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let (parsed, payload) = parse_index(bytes, 3, 4)?;
        if payload.is_some() {
            return Err(LoadError::UnsupportedVersion);
        }
        Ok(Self::from_scalar(index3d_from_columns(
            columns3d_from_parsed(&parsed),
        )))
    }
}
