//! R2 / out-of-core metric: range requests + bytes fetched for a polygon query
//! (region pruning) versus its bounding box, on the streaming PSINDEX artifact
//! path — the cost a Worker over R2 actually pays (per-request billing,
//! subrequest limits, egress / latency), not wall-clock.
//!
//! `search_hits(GeoQuery2D::polygon(..))` prunes subtrees outside the polygon
//! during the streamed descent, so it fetches less data (fewer payload bytes)
//! than fetching everything in the bounding box. The range-request COUNT is
//! shape-dependent — pruning fragments the coalesced runs, so a low-rejection
//! query can issue a few more requests while a high-rejection one issues fewer;
//! the bytes always shrink. A counting `RangeReader` measures both.
//!
//! Run: cargo run -p packed_spatial_index_geo --release --example r2_polygon_pruning

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use geo::BoundingRect;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};
use packed_spatial_index_geo::{
    Box2D, ConvertRequest, GeoArtifactIndex, GeoQuery2D, RangeReader, SliceReader, open,
    open_geo_index,
};

const N: usize = 100_000;

/// A `RangeReader` that counts read calls (range requests) and bytes fetched.
/// Counters live behind an `Rc` so a handle survives after the reader is moved
/// into the opened index.
#[derive(Default)]
struct Counters {
    reads: Cell<usize>,
    bytes: Cell<u64>,
}

struct CountingReader {
    inner: SliceReader<Vec<u8>>,
    counters: Rc<Counters>,
}

impl RangeReader for CountingReader {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.counters.reads.set(self.counters.reads.get() + 1);
        self.counters
            .bytes
            .set(self.counters.bytes.get() + buf.len() as u64);
        self.inner.read_exact_at(offset, buf)
    }

    fn len(&self) -> Option<u64> {
        self.inner.len()
    }
}

/// Open the artifact over a counting reader, run one `search_hits`, and return
/// `(range requests, bytes, hits)` for the query alone (excluding the open).
fn measure(artifact: &[u8], query: GeoQuery2D) -> (usize, u64, usize) {
    let counters = Rc::new(Counters::default());
    let reader = CountingReader {
        inner: SliceReader::new(artifact.to_vec()),
        counters: counters.clone(),
    };
    let GeoArtifactIndex::D2(index) = open_geo_index(reader).unwrap() else {
        panic!("expected a 2D artifact");
    };
    let open_reads = counters.reads.get();
    let open_bytes = counters.bytes.get();
    let hits = index.search_hits(query).unwrap();
    (
        counters.reads.get() - open_reads,
        counters.bytes.get() - open_bytes,
        hits.len(),
    )
}

fn main() {
    let parquet = build_dataset(N);
    let artifact = open(parquet)
        .unwrap()
        .convert(ConvertRequest::default())
        .unwrap();
    println!(
        "=== R2 metric: polygon (region-pruned) vs bounding box on the streaming artifact ===",
    );
    println!(
        "{} points, PSINDEX artifact {:.1} KiB. Lower reads / bytes is cheaper over R2.\n",
        N,
        artifact.len() as f64 / 1024.0
    );
    println!(
        "{:<8} {:>7} {:>16} {:>16} {:>9} {:>9}",
        "query", "reject", "bbox reads/KiB", "poly reads/KiB", "reads-", "bytes-"
    );
    println!("{}", "-".repeat(72));

    for (name, poly) in queries() {
        let rect = poly.bounding_rect().unwrap();
        let bbox = Box2D::new(rect.min().x, rect.min().y, rect.max().x, rect.max().y);

        let (b_reads, b_bytes, b_hits) = measure(&artifact, GeoQuery2D::box2d(bbox));
        let (p_reads, p_bytes, p_hits) = measure(&artifact, GeoQuery2D::polygon(poly));

        let reject = 100.0 * (1.0 - p_hits as f64 / b_hits as f64);
        let reads_cut = 100.0 * (1.0 - p_reads as f64 / b_reads as f64);
        let bytes_cut = 100.0 * (1.0 - p_bytes as f64 / b_bytes as f64);
        println!(
            "{:<8} {:>6.0}% {:>8}/{:>7.1} {:>8}/{:>7.1} {:>8.0}% {:>8.0}%",
            name,
            reject,
            b_reads,
            b_bytes as f64 / 1024.0,
            p_reads,
            p_bytes as f64 / 1024.0,
            reads_cut,
            bytes_cut,
        );
    }
    println!(
        "\nbbox = the polygon's bounding box (what a box query fetches). 'reads-' / 'bytes-' =\n\
         how much the polygon prunes vs that bbox. Payload is RowWkb (geometry); with a heavier\n\
         FeatureJson 'data' payload the byte savings scale up further.",
    );
}

// --- queries: non-convex shapes inside a sub-region of the [0,1000]^2 dataset,
// so both the bbox query and the region query are selective ------------------
fn queries() -> Vec<(&'static str, Polygon<f64>)> {
    vec![
        ("donut", donut()),
        ("star", star(300.0, 300.0, 200.0, 80.0, 5)),
        ("band", band(100.0, 100.0, 500.0, 500.0, 12.0)),
    ]
}

fn ring(pts: &[(f64, f64)]) -> LineString<f64> {
    let mut v: Vec<Coord<f64>> = pts.iter().map(|&(x, y)| Coord { x, y }).collect();
    if v.first() != v.last() {
        v.push(v[0]);
    }
    LineString::new(v)
}

fn rect_pts(a: f64, b: f64, c: f64, d: f64) -> Vec<(f64, f64)> {
    vec![(a, b), (c, b), (c, d), (a, d)]
}

fn donut() -> Polygon<f64> {
    Polygon::new(
        ring(&rect_pts(100.0, 100.0, 500.0, 500.0)),
        vec![ring(&rect_pts(250.0, 250.0, 350.0, 350.0))],
    )
}

fn star(cx: f64, cy: f64, r_out: f64, r_in: f64, points: usize) -> Polygon<f64> {
    let mut v = Vec::new();
    for i in 0..2 * points {
        let a = std::f64::consts::PI * i as f64 / points as f64 - std::f64::consts::FRAC_PI_2;
        let r = if i % 2 == 0 { r_out } else { r_in };
        v.push((cx + r * a.cos(), cy + r * a.sin()));
    }
    Polygon::new(ring(&v), vec![])
}

fn band(x0: f64, y0: f64, x1: f64, y1: f64, half_w: f64) -> Polygon<f64> {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len = (dx * dx + dy * dy).sqrt();
    let (nx, ny) = (-dy / len * half_w, dx / len * half_w);
    Polygon::new(
        ring(&[
            (x0 + nx, y0 + ny),
            (x1 + nx, y1 + ny),
            (x1 - nx, y1 - ny),
            (x0 - nx, y0 - ny),
        ]),
        vec![],
    )
}

// --- in-memory GeoParquet of N points ---------------------------------------
fn build_dataset(n: usize) -> Bytes {
    let mut seed = 0xdead_beef_cafe_1234u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut wkb: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
    let mut names: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let x = 1000.0 * rnd();
        let y = 1000.0 * rnd();
        let mut v = Vec::with_capacity(21);
        v.push(1u8);
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&x.to_le_bytes());
        v.extend_from_slice(&y.to_le_bytes());
        wkb.push(Some(v));
        names.push(format!("feature_{i:06}"));
    }

    let cols: Vec<(&str, ArrayRef)> = vec![
        (
            "geometry",
            Arc::new(BinaryArray::from(
                wkb.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        ("name", Arc::new(StringArray::from(names)) as ArrayRef),
    ];

    let batch = RecordBatch::try_from_iter(cols).unwrap();
    let geo_json = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"]}}}"#.to_string();
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new("geo".to_string(), geo_json)]))
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}
