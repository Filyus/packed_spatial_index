//! WASM entry for the CF Worker + R2 streaming demo.
//!
//! The Worker owns the R2 binding, so it hands us a JS callback
//! `read_range(offset, length) -> Promise<Uint8Array>`. We wrap it as an
//! [`AsyncRangeReader`] and answer a box query by streaming only the few range
//! reads the traversal needs. Reads/bytes are counted on the JS side (it wraps
//! the callback), so this module just returns the hits.

use std::cell::RefCell;
use std::io;

use js_sys::{Array, Function, Object, Promise, Reflect, Uint8Array};
use packed_spatial_index::{AsyncRangeReader, Box2D, StreamDirectory, StreamIndex2D, StreamLimits};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

thread_local! {
    // The parsed directory, cached for the life of the warm isolate. After the
    // first request opens the index, every later request reattaches a fresh R2
    // reader to this directory with zero directory I/O — the round-trips that
    // otherwise dominate per-query latency are paid once, not per request.
    static DIRECTORY: RefCell<Option<StreamDirectory>> = const { RefCell::new(None) };
}

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
    // Fixed, conservative budget (no concurrency tracking — Cloudflare schedules
    // isolates; we just answer fast). Cache all internal levels for the fewest
    // round-trips, and cap result memory well under the isolate's 128 MB so a
    // broad query can't OOM and evict the warm directory. Peak ~32 MB.
    let limits = StreamLimits {
        max_reads: if max_reads > 0.0 {
            Some(max_reads as usize)
        } else {
            None
        },
        max_read_bytes: Some(16 * 1024 * 1024),
        max_items: Some(1_000_000),
        directory_budget_bytes: Some(16 * 1024 * 1024),
        // Over-read up to 256 KB to collapse round-trips: a strong win on R2
        // (high latency), bounded by max_read_bytes above.
        coalesce_gap_bytes: Some(256 * 1024),
    };

    // Reattach the cached directory if this warm isolate already has one (no
    // directory reads); otherwise open once and cache it for later requests.
    let cached = DIRECTORY.with(|d| d.borrow().clone());
    let stream = match cached {
        Some(dir) => {
            StreamIndex2D::from_directory_with_limits(&dir, reader, limits).map_err(stream_err)?
        }
        None => {
            let opened = StreamIndex2D::open_with_limits_async(reader, limits)
                .await
                .map_err(stream_err)?;
            let (dir, reader) = opened.into_directory();
            DIRECTORY.with(|d| *d.borrow_mut() = Some(dir.clone()));
            StreamIndex2D::from_directory_with_limits(&dir, reader, limits).map_err(stream_err)?
        }
    };
    let hits = stream
        .search_payloads_async(Box2D::new(min_x, min_y, max_x, max_y))
        .await
        .map_err(stream_err)?;

    let payload_bytes: usize = hits.iter().map(|(_, b)| b.len()).sum();
    // Return the first features' ids and their actual geometry (WKB, base64), so a
    // caller gets real geometry back over the network, not just a count. Capped to
    // bound the response.
    const FEATURE_CAP: usize = 1000;
    let ids = Array::new();
    let geometries = Array::new();
    for (id, wkb) in hits.iter().take(FEATURE_CAP) {
        ids.push(&JsValue::from_f64(*id as f64));
        geometries.push(&JsValue::from_str(&base64(wkb)));
    }

    let out = Object::new();
    set(&out, "hits", JsValue::from_f64(hits.len() as f64))?;
    set(&out, "payloadBytes", JsValue::from_f64(payload_bytes as f64))?;
    set(&out, "ids", ids.into())?;
    set(&out, "geometries", geometries.into())?;
    Ok(out.into())
}

/// Minimal standard base64 (no dependency) for the WKB payloads.
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        s.push(T[(n >> 18 & 63) as usize] as char);
        s.push(T[(n >> 12 & 63) as usize] as char);
        s.push(if chunk.len() > 1 { T[(n >> 6 & 63) as usize] as char } else { '=' });
        s.push(if chunk.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    s
}

fn set(obj: &Object, key: &str, val: JsValue) -> Result<(), JsValue> {
    Reflect::set(obj, &JsValue::from_str(key), &val).map(|_| ())
}

fn stream_err(e: packed_spatial_index::StreamError) -> JsValue {
    JsValue::from_str(&format!("{e:?}"))
}
