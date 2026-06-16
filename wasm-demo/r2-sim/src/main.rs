//! Reads/bytes simulator for the CF Worker + R2 streaming demo.
//!
//! Serializes a synthetic index three ways, opens each over a counting
//! `RangeReader`, and reports the range reads + bytes a `search_payloads` query
//! issues across a window-size matrix. The counts are produced by the crate's
//! real coalescing/traversal, so they equal what an R2-backed reader would do —
//! this is the demo's headline number without any cloud.

use std::cell::Cell;
use std::rc::Rc;

use packed_spatial_index::{Box2D, Index2D, Index2DBuilder};
use packed_spatial_index::{RangeReader, SliceReader, StreamIndex2D, StreamLimits};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const EXTENT: f64 = 1000.0;
const PAYLOAD_STRIDE: usize = 64; // a small per-feature blob

/// Shared read/byte tally. Held by the reader (moved into the stream) and by a
/// handle kept outside, so per-query counts are readable after `open`.
#[derive(Clone, Default)]
struct Counters(Rc<CounterState>);

#[derive(Default)]
struct CounterState {
    reads: Cell<u64>,
    bytes: Cell<u64>,
}

impl Counters {
    fn reset(&self) {
        self.0.reads.set(0);
        self.0.bytes.set(0);
    }
    fn get(&self) -> (u64, u64) {
        (self.0.reads.get(), self.0.bytes.get())
    }
}

/// Wraps a `RangeReader` and tallies into shared `Counters`. `&self` reads, so
/// the counters use `Cell` (the run is single-threaded, like a Worker isolate).
struct CountingReader<R> {
    inner: R,
    counters: Counters,
}

impl<R: RangeReader> RangeReader for CountingReader<R> {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.counters.0.reads.set(self.counters.0.reads.get() + 1);
        self.counters.0.bytes.set(self.counters.0.bytes.get() + buf.len() as u64);
        self.inner.read_exact_at(offset, buf)
    }
    fn len(&self) -> Option<u64> {
        self.inner.len()
    }
}

fn build_index(n: usize) -> Index2D {
    let mut rng = StdRng::seed_from_u64(0xA5E1);
    let mut builder = Index2DBuilder::new(n).node_size(16);
    for _ in 0..n {
        let x = rng.random_range(0.0..EXTENT);
        let y = rng.random_range(0.0..EXTENT);
        let w = rng.random_range(0.2..4.0f64);
        let h = rng.random_range(0.2..4.0f64);
        builder.add(Box2D::new(x, y, x + w, y + h));
    }
    builder.finish().unwrap()
}

/// Average per-query reads/bytes/hits for one serialized file over a window matrix.
fn measure(label: &str, bytes: &[u8]) {
    let counters = Counters::default();
    let reader = CountingReader { inner: SliceReader::new(bytes), counters: counters.clone() };
    // Cache all internal levels (serverless: trade a little memory for fewer
    // per-query round-trips). `None` would keep the small built-in default.
    let limits = StreamLimits {
        directory_budget_bytes: Some(64 * 1024 * 1024),
        ..Default::default()
    };
    let stream = StreamIndex2D::open_with_limits(reader, limits).expect("open");
    let (open_reads, open_bytes) = counters.get();

    println!(
        "\n=== {label}  (file {} KB, open {} reads / {} B, all-internal directory) ===",
        bytes.len() / 1024,
        open_reads,
        open_bytes,
    );
    println!("  window   hits   reads   bytes/query");
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    for frac in [0.005, 0.02, 0.08, 0.25, 1.0] {
        let side = EXTENT * frac;
        let (mut tot_reads, mut tot_bytes, mut tot_hits) = (0u64, 0u64, 0u64);
        let queries = 60u64;
        for _ in 0..queries {
            let x = rng.random_range(0.0..(EXTENT - side).max(0.0001));
            let y = rng.random_range(0.0..(EXTENT - side).max(0.0001));
            let q = Box2D::new(x, y, x + side, y + side);
            counters.reset();
            let hits = stream.search_payloads(q).expect("search");
            let (r, b) = counters.get();
            tot_reads += r;
            tot_bytes += b;
            tot_hits += hits.len() as u64;
        }
        println!(
            "  {:>5.1}%  {:>5}  {:>5.1}  {:>9.0}",
            frac * 100.0,
            tot_hits / queries,
            tot_reads as f64 / queries as f64,
            tot_bytes as f64 / queries as f64,
        );
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200_000);
    println!("n = {n} items, payload stride = {PAYLOAD_STRIDE} B");

    let index = build_index(n);
    let flat: Vec<u8> = (0..n)
        .flat_map(|i| std::iter::repeat_n((i & 0xff) as u8, PAYLOAD_STRIDE))
        .collect();
    let il_fixed = index
        .serialize()
        .interleaved()
        .records(PAYLOAD_STRIDE, &flat)
        .to_bytes()
        .unwrap();

    // Write-only fast path: emit the deployable file and skip the matrix measure
    // (used to seed R2 for large N, where the full measure would be slow).
    if let Some(path) = std::env::args().nth(2) {
        std::fs::write(&path, &il_fixed).expect("write index file");
        println!("wrote deployable index ({} KB) -> {path}", il_fixed.len() / 1024);
        return;
    }

    let blobs: Vec<Vec<u8>> = (0..n).map(|i| vec![(i & 0xff) as u8; PAYLOAD_STRIDE]).collect();
    let soa_var = index.serialize().payloads(&blobs).to_bytes().unwrap();
    let il_var = index.serialize().interleaved().payloads(&blobs).to_bytes().unwrap();
    measure("SoA + variable payload", &soa_var);
    measure("interleaved + variable payload", &il_var);
    measure("interleaved + fixed-width records", &il_fixed);
}
