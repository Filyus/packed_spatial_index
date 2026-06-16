//! WASM entry for the CF Worker + R2 streaming demo.
//!
//! The Worker owns the R2 binding, so it hands us a JS callback
//! `read_range(offset, length) -> Promise<Uint8Array>`. We wrap it as an
//! [`AsyncRangeReader`] and answer a box query by streaming only the few range
//! reads the traversal needs. Reads/bytes are counted on the JS side (it wraps
//! the callback), so this module just returns the hits.

use std::io;

use js_sys::{Array, Function, Object, Promise, Reflect, Uint8Array};
use packed_spatial_index::{AsyncRangeReader, Box2D, StreamIndex2D, StreamLimits};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Bridges the Worker's R2 range `get` into the crate's async reader.
struct R2Reader {
    read_range: Function,
    len: Option<u64>,
}

impl AsyncRangeReader for R2Reader {
    async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        // read_range(offset, length) -> Promise<Uint8Array>
        let promise = self
            .read_range
            .call2(
                &JsValue::NULL,
                &JsValue::from_f64(offset as f64),
                &JsValue::from_f64(buf.len() as f64),
            )
            .map_err(js_io)?;
        let promise: Promise = promise
            .dyn_into()
            .map_err(|_| io_err("read_range must return a Promise"))?;
        let value = JsFuture::from(promise).await.map_err(js_io)?;
        let arr: Uint8Array = value
            .dyn_into()
            .map_err(|_| io_err("range result must be a Uint8Array"))?;
        // `read_exact_at` must fill the whole buffer; a short R2 range is an error.
        if arr.length() as usize != buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short range read",
            ));
        }
        arr.copy_to(buf);
        Ok(())
    }

    fn len(&self) -> Option<u64> {
        self.len
    }
}

fn io_err(msg: &str) -> io::Error {
    io::Error::other(msg)
}

fn js_io(v: JsValue) -> io::Error {
    io::Error::other(
        v.as_string()
            .unwrap_or_else(|| "js error in read_range".to_string()),
    )
}

/// Run one box query against the R2-backed index.
///
/// `read_range` is `(offset: number, length: number) => Promise<Uint8Array>`.
/// Returns `{ hits: number, payloadBytes: number, ids: number[] }` (ids capped).
#[wasm_bindgen]
pub async fn query(
    read_range: Function,
    file_len: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    max_reads: f64,
) -> Result<JsValue, JsValue> {
    let reader = R2Reader {
        read_range,
        len: if file_len > 0.0 {
            Some(file_len as u64)
        } else {
            None
        },
    };
    let limits = StreamLimits {
        max_reads: if max_reads > 0.0 {
            Some(max_reads as usize)
        } else {
            None
        },
        ..Default::default()
    };

    let stream = StreamIndex2D::open_with_limits_async(reader, limits)
        .await
        .map_err(stream_err)?;
    let hits = stream
        .search_payloads_async(Box2D::new(min_x, min_y, max_x, max_y))
        .await
        .map_err(stream_err)?;

    let payload_bytes: usize = hits.iter().map(|(_, b)| b.len()).sum();
    let ids = Array::new();
    for (id, _) in hits.iter().take(1000) {
        ids.push(&JsValue::from_f64(*id as f64));
    }

    let out = Object::new();
    set(&out, "hits", JsValue::from_f64(hits.len() as f64))?;
    set(&out, "payloadBytes", JsValue::from_f64(payload_bytes as f64))?;
    set(&out, "ids", ids.into())?;
    Ok(out.into())
}

fn set(obj: &Object, key: &str, val: JsValue) -> Result<(), JsValue> {
    Reflect::set(obj, &JsValue::from_str(key), &val).map(|_| ())
}

fn stream_err(e: packed_spatial_index::StreamError) -> JsValue {
    JsValue::from_str(&format!("{e:?}"))
}
