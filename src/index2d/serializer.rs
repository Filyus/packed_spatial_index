use super::Index2D;
use crate::{
    persistence::{MetaFields, PayloadError, write_index_container},
    triangle::{Triangle2, records_as_bytes},
};

/// Builder for [`Index2D`] serialization, created by [`Index2D::serialize`].
///
/// Set optional per-item payloads, the streaming-tuned interleaved layout, and
/// descriptive metadata (CRS / content type / attribution), then call
/// [`to_bytes`](Self::to_bytes) or [`to_bytes_into`](Self::to_bytes_into). The
/// metadata strings are stored opaquely and read back with
/// [`read_metadata`](crate::read_metadata).
pub struct Serializer2D<'a> {
    index: &'a Index2D,
    interleaved: bool,
    payloads: Option<Vec<&'a [u8]>>,
    /// `Some(stride)` selects the fixed-width (table-less) payload layout.
    record_stride: Option<u32>,
    meta: MetaFields<'a>,
}

impl<'a> Serializer2D<'a> {
    pub(super) fn new(index: &'a Index2D) -> Self {
        Self {
            index,
            interleaved: false,
            payloads: None,
            record_stride: None,
            meta: MetaFields::default(),
        }
    }

    /// Attach one opaque payload blob per item, in item order.
    pub fn payloads<P: AsRef<[u8]>>(mut self, payloads: &'a [P]) -> Self {
        self.payloads = Some(payloads.iter().map(|p| p.as_ref()).collect());
        self
    }

    /// Attach a **fixed-width** payload: `flat` is the concatenation of one
    /// `stride`-byte record per item, in item order (item `i` is
    /// `flat[i * stride ..][.. stride]`). Because every record is the same size,
    /// the offset table is dropped (the reader addresses record `r` by
    /// arithmetic), which shrinks the file and lets a view borrow the records as
    /// a zero-copy typed slice. `flat.len()` must be `num_items * stride`.
    pub fn records(mut self, stride: usize, flat: &'a [u8]) -> Self {
        self.record_stride = Some(stride as u32);
        self.payloads = Some(if stride == 0 {
            Vec::new()
        } else {
            flat.chunks_exact(stride).collect()
        });
        self
    }

    /// Attach a fixed-width triangle payload (`T` =
    /// [`Triangle2D`](crate::Triangle2D) for `f64` or
    /// [`Triangle2DF32`](crate::Triangle2DF32) for `f32`): one triangle per item,
    /// in item order. A convenience over [`records`](Self::records); pair it with
    /// [`Index2D::from_triangles`](crate::Index2D::from_triangles).
    pub fn triangles<T: Triangle2>(self, triangles: &'a [T]) -> Self {
        let bytes = records_as_bytes(triangles);
        self.records(T::STRIDE, bytes)
    }

    /// Use the streaming-tuned interleaved node layout (see
    /// [`Index2D::to_bytes_interleaved`]).
    #[cfg(feature = "stream")]
    pub fn interleaved(mut self) -> Self {
        self.interleaved = true;
        self
    }

    /// Set the coordinate reference system identifier (opaque, e.g. `"EPSG:4326"`).
    pub fn crs(mut self, crs: &'a str) -> Self {
        self.meta.crs = Some(crs);
        self
    }

    /// Set the payload content type / media type (e.g. `"application/geo+json"`).
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
        let interleaved = self.interleaved;
        let record_stride = self.record_stride;
        write_index_container(
            out,
            2,
            8,
            interleaved,
            idx.num_items,
            idx.entries.len(),
            idx.node_size,
            |bytes| {
                #[cfg(feature = "stream")]
                if interleaved {
                    bytes.write_interleaved_2d(&idx.entries, &idx.indices);
                    return;
                }
                bytes.write_box2d_slice(&idx.entries);
                bytes.write_usize_slice_as_u64(&idx.indices);
            },
            self.payloads.as_deref(),
            record_stride,
            &idx.indices[..idx.num_items],
            &self.meta,
        )
    }
}
