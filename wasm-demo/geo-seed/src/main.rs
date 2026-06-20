//! Generate a realistic GeoParquet to seed the streaming demo.
//!
//! Writes `N` WKB Point features clustered around random "city" centres (so the
//! point cloud looks like populated areas rather than uniform noise), with a
//! `bbox` covering struct column and the GeoParquet 1.1 `geo` metadata. Convert
//! the output with `gp2psindex`, then serve the `.psi` from R2.
//!
//! ```text
//! cargo run --release -- [count] [out.parquet]
//! ```

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Float64Array, StructArray};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

/// Small deterministic PRNG (xorshift64*), so the seed is reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Approx. standard normal (sum of 4 uniforms, centred and scaled).
    fn normal(&mut self) -> f64 {
        let s: f64 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.0
    }
}

fn wkb_point(x: f64, y: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(21);
    v.push(1); // little-endian
    v.extend_from_slice(&1u32.to_le_bytes()); // Point
    v.extend_from_slice(&x.to_le_bytes());
    v.extend_from_slice(&y.to_le_bytes());
    v
}

fn main() {
    let mut args = std::env::args().skip(1);
    let count: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let out = args.next().unwrap_or_else(|| "cities.parquet".to_string());

    let mut rng = Rng(0x5EED_1234_ABCD_0001);

    // ~1 cluster per 300 points; centres over inhabited latitudes.
    let clusters = (count / 300).max(8);
    let centres: Vec<(f64, f64)> = (0..clusters)
        .map(|_| {
            let lon = rng.unit() * 360.0 - 180.0;
            let lat = rng.unit() * 130.0 - 60.0; // [-60, 70]
            (lon, lat)
        })
        .collect();

    let mut wkbs: Vec<Vec<u8>> = Vec::with_capacity(count);
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (
        Vec::with_capacity(count),
        Vec::with_capacity(count),
        Vec::with_capacity(count),
        Vec::with_capacity(count),
    );
    for _ in 0..count {
        let (cx, cy) = centres[(rng.next_u64() as usize) % centres.len()];
        let lon = (cx + rng.normal() * 1.5).clamp(-180.0, 180.0);
        let lat = (cy + rng.normal() * 1.5).clamp(-89.9, 89.9);
        wkbs.push(wkb_point(lon, lat));
        // Covering of a point is the point itself.
        xmin.push(lon);
        ymin.push(lat);
        xmax.push(lon);
        ymax.push(lat);
    }

    let geom: ArrayRef = Arc::new(BinaryArray::from(
        wkbs.iter().map(|w| Some(w.as_slice())).collect::<Vec<_>>(),
    ));
    let f = |name: &str, v: Vec<f64>| {
        (
            Arc::new(Field::new(name, DataType::Float64, false)),
            Arc::new(Float64Array::from(v)) as ArrayRef,
        )
    };
    let bbox: ArrayRef = Arc::new(StructArray::from(vec![
        f("xmin", xmin),
        f("ymin", ymin),
        f("xmax", xmax),
        f("ymax", ymax),
    ]));

    let batch = RecordBatch::try_from_iter(vec![("geometry", geom), ("bbox", bbox)]).unwrap();

    let geo = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"],"crs":{"id":{"authority":"OGC","code":"CRS84"}},"covering":{"bbox":{"xmin":["bbox","xmin"],"ymin":["bbox","ymin"],"xmax":["bbox","xmax"],"ymax":["bbox","ymax"]}}}}}"#;
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new("geo".to_string(), geo.to_string())]))
        .build();

    let file = std::fs::File::create(&out).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    eprintln!("wrote {out}: {count} WKB points in {clusters} clusters");
}
