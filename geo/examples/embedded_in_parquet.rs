//! Prototype for opengeospatial/geoparquet#279: can a row-level spatial index
//! live INSIDE a GeoParquet file — in the bytes the footer never references —
//! transparent to standard Parquet readers, located and streamed by ours?
//!
//! A Parquet file is `PAR1 [row groups] [footer] [footer_len] PAR1`; the footer
//! is found from the trailer and its column offsets are absolute, so bytes
//! inserted between the last row group and the footer are invisible to a
//! compliant reader. We splice a PSINDEX artifact there, tag it with a marker
//! placed right before the footer, then check:
//!   (a) the `parquet` crate still reads every row (index bytes ignored);
//!   (b) our reader finds the embedded index via the footer-relative marker and
//!       answers a polygon query reading only a fraction of the file — no
//!       sidecar, no row-group scan.
//!
//! A real spec extension would record the [offset, length] in the footer's
//! key/value metadata (or the `geo` metadata) instead of a trailing marker; the
//! marker just avoids re-encoding the Thrift footer for this experiment.
//!
//! Run: cargo run -p packed_spatial_index_geo --release --example embedded_in_parquet

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};
use packed_spatial_index_geo::{
    ConvertRequest, GeoArtifactIndex, GeoQuery2D, RangeReader, SliceReader, open_geo_index,
    open_geoparquet,
};

const N: usize = 100_000;
/// Marker written immediately before the footer so our reader can find the blob.
const PSIDX_MAGIC: &[u8; 4] = b"PSIX";

fn main() {
    let parquet = build_dataset(N);
    let index = open_geoparquet(parquet.clone())
        .unwrap()
        .convert(ConvertRequest::default())
        .unwrap();
    let embedded = splice_before_footer(&parquet, &index);

    println!("=== geoparquet#279 prototype: spatial index embedded in the Parquet file ===");
    println!("plain GeoParquet : {:>9} bytes", parquet.len());
    println!(
        "embedded variant : {:>9} bytes  (+{} index + 8 marker, spliced before the footer)\n",
        embedded.len(),
        index.len()
    );

    // Optional: dump the embedded file so other readers (pyarrow, GDAL, DuckDB)
    // can confirm it too — `cargo run --example embedded_in_parquet -- out.parquet`.
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, &embedded).unwrap();
        println!("wrote embedded file to {path}\n");
    }

    // (a) a standard Parquet reader must be unaffected by the inserted bytes.
    let rows = read_back_rowcount(&embedded);
    println!("[a] parquet crate read the spliced file: {rows} rows (expected {N})");
    assert_eq!(
        rows, N,
        "standard reader disagreed — splice is not transparent"
    );
    println!("    → standard reader is happy; the index bytes are invisible to it.\n");

    // (b) our reader discovers the embedded index (footer-relative marker) and
    //     answers a polygon query over just that byte range.
    let (base, len) = locate_embedded(&embedded);
    println!(
        "[b] found embedded index at byte {base} (len {len}) via the marker before the footer"
    );

    let counters = Rc::new(Counters::default());
    let reader = EmbeddedReader {
        file: SliceReader::new(embedded.clone()),
        base,
        len,
        counters: counters.clone(),
    };
    let GeoArtifactIndex::D2(index_reader) = open_geo_index(reader).unwrap() else {
        panic!("expected a 2D artifact");
    };
    let matches = index_reader
        .search_matches(GeoQuery2D::polygon(donut()))
        .unwrap();

    let read = counters.bytes.get();
    println!(
        "    polygon query → {} matches in {} range reads, {} bytes read = {:.2}% of the {}-byte file",
        matches.len(),
        counters.reads.get(),
        read,
        100.0 * read as f64 / embedded.len() as f64,
        embedded.len(),
    );
    println!("    → the query touched the embedded index only; the row-group data was never read.");
}

// --- splice + locate --------------------------------------------------------

/// Byte offset where the Parquet footer (FileMetaData) begins, via the trailer.
fn footer_start(buf: &[u8]) -> usize {
    let n = buf.len();
    assert_eq!(&buf[n - 4..], b"PAR1", "not a Parquet file");
    let footer_len = u32::from_le_bytes(buf[n - 8..n - 4].try_into().unwrap()) as usize;
    n - 8 - footer_len
}

/// Insert `index` between the row-group data and the footer, then a marker
/// (`PSIX` + u32 length) right before the footer. Row-group offsets are absolute
/// and unchanged; the footer simply relocates and is still found via the trailer.
fn splice_before_footer(parquet: &[u8], index: &[u8]) -> Vec<u8> {
    let fs = footer_start(parquet);
    let mut out = Vec::with_capacity(parquet.len() + index.len() + 8);
    out.extend_from_slice(&parquet[..fs]); // PAR1 + row groups
    out.extend_from_slice(index); // the PSINDEX blob
    out.extend_from_slice(PSIDX_MAGIC); // marker, found relative to the footer
    out.extend_from_slice(&(index.len() as u32).to_le_bytes());
    out.extend_from_slice(&parquet[fs..]); // footer + footer_len + PAR1
    out
}

/// Discover the embedded index: locate the footer normally, then read the marker
/// in the 8 bytes immediately before it. Returns `(base offset, length)`.
fn locate_embedded(buf: &[u8]) -> (u64, u64) {
    let fs = footer_start(buf);
    assert_eq!(
        &buf[fs - 8..fs - 4],
        PSIDX_MAGIC,
        "no embedded-index marker"
    );
    let blob_len = u32::from_le_bytes(buf[fs - 4..fs].try_into().unwrap()) as u64;
    ((fs - 8) as u64 - blob_len, blob_len)
}

/// Read the spliced file with the stock `parquet` reader and count rows.
fn read_back_rowcount(buf: &[u8]) -> usize {
    let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(buf)).unwrap();
    builder
        .build()
        .unwrap()
        .map(|batch| batch.unwrap().num_rows())
        .sum()
}

/// A `RangeReader` that exposes the embedded sub-range `[base, base+len)` as if
/// it were a standalone artifact, and counts the reads it serves.
#[derive(Default)]
struct Counters {
    reads: Cell<usize>,
    bytes: Cell<u64>,
}

struct EmbeddedReader {
    file: SliceReader<Vec<u8>>,
    base: u64,
    len: u64,
    counters: Rc<Counters>,
}

impl RangeReader for EmbeddedReader {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.counters.reads.set(self.counters.reads.get() + 1);
        self.counters
            .bytes
            .set(self.counters.bytes.get() + buf.len() as u64);
        self.file.read_exact_at(self.base + offset, buf)
    }

    fn len(&self) -> Option<u64> {
        Some(self.len)
    }
}

// --- a concave query + an in-memory GeoParquet of N points ------------------

fn donut() -> Polygon<f64> {
    Polygon::new(
        ring(&[
            (100.0, 100.0),
            (500.0, 100.0),
            (500.0, 500.0),
            (100.0, 500.0),
        ]),
        vec![ring(&[
            (250.0, 250.0),
            (350.0, 250.0),
            (350.0, 350.0),
            (250.0, 350.0),
        ])],
    )
}

fn ring(pts: &[(f64, f64)]) -> LineString<f64> {
    let mut v: Vec<Coord<f64>> = pts.iter().map(|&(x, y)| Coord { x, y }).collect();
    v.push(v[0]);
    LineString::new(v)
}

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
