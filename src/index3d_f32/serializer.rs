use super::Index3DF32;
use crate::{
    f32_storage::{F32ColumnRefs3D, write_columns3d},
    persistence::{MetaFields, PayloadError},
    triangle::{Triangle3, records_as_bytes},
};

/// Serialization builder for [`Index3DF32`], created by
/// [`Index3DF32::serialize`]. Writes f32 box records plus an optional per-item
/// payload and descriptive metadata; the f32-box counterpart of
/// [`Serializer3D`](crate::Serializer3D).
pub struct Serializer3DF32<'a> {
    index: &'a Index3DF32,
    payloads: Option<Vec<&'a [u8]>>,
    record_stride: Option<u32>,
    interleaved: bool,
    meta: MetaFields<'a>,
}

impl<'a> Serializer3DF32<'a> {
    pub(super) fn new(index: &'a Index3DF32) -> Self {
        Self {
            index,
            payloads: None,
            record_stride: None,
            interleaved: false,
            meta: MetaFields::default(),
        }
    }

    /// Attach one opaque payload blob per item, in item order.
    pub fn payloads<P: AsRef<[u8]>>(mut self, payloads: &'a [P]) -> Self {
        self.payloads = Some(payloads.iter().map(|p| p.as_ref()).collect());
        self
    }

    /// Use the streaming-tuned interleaved node layout (box and index adjacent
    /// per node), so [`StreamIndex3DF32`](crate::StreamIndex3DF32) fetches each
    /// level in one coalesced read instead of two. Same file size.
    #[cfg(feature = "stream")]
    pub fn interleaved(mut self) -> Self {
        self.interleaved = true;
        self
    }

    /// Attach a fixed-width payload: `flat` is `num_items * stride` bytes (one
    /// `stride`-byte record per item). See
    /// [`Serializer3D::records`](crate::Serializer3D::records).
    pub fn records(mut self, stride: usize, flat: &'a [u8]) -> Self {
        self.record_stride = Some(stride as u32);
        self.payloads = Some(if stride == 0 {
            Vec::new()
        } else {
            flat.chunks_exact(stride).collect()
        });
        self
    }

    /// Attach a fixed-width triangle payload, one per item. A compact mesh BVH
    /// that streams through [`StreamIndex3DF32`](crate::StreamIndex3DF32).
    pub fn triangles<T: Triangle3>(self, triangles: &'a [T]) -> Self {
        let bytes = records_as_bytes(triangles);
        self.records(T::STRIDE, bytes)
    }

    /// Set the coordinate reference system identifier (opaque, e.g. `"EPSG:4979"`).
    pub fn crs(mut self, crs: &'a str) -> Self {
        self.meta.crs = Some(crs);
        self
    }

    /// Set the payload content type / media type.
    pub fn content_type(mut self, content_type: &'a str) -> Self {
        self.meta.content_type = Some(content_type);
        self
    }

    /// Set an attribution / license string.
    pub fn attribution(mut self, attribution: &'a str) -> Self {
        self.meta.attribution = Some(attribution);
        self
    }

    /// Serialize into a new buffer.
    pub fn to_bytes(self) -> Result<Vec<u8>, PayloadError> {
        let mut out = Vec::new();
        self.to_bytes_into(&mut out)?;
        Ok(out)
    }

    /// Serialize into a reused buffer (cleared first).
    pub fn to_bytes_into(self, out: &mut Vec<u8>) -> Result<(), PayloadError> {
        let idx = self.index;
        let record_stride = self.record_stride;
        let interleaved = self.interleaved;
        write_columns3d(
            out,
            F32ColumnRefs3D {
                node_size: idx.node_size,
                num_items: idx.num_items,
                min_xs: &idx.min_xs,
                min_ys: &idx.min_ys,
                min_zs: &idx.min_zs,
                max_xs: &idx.max_xs,
                max_ys: &idx.max_ys,
                max_zs: &idx.max_zs,
                indices: &idx.indices,
            },
            interleaved,
            self.payloads.as_deref(),
            record_stride,
            &self.meta,
        )
    }
}
