//! When does exact polygon filtering pay off? Full end-to-end query time.
//!
//! For a non-rectangular (polygon) query the exact step is REQUIRED for a correct
//! answer — the index only narrows by bounding box. This bench instead measures
//! the *cost/benefit as a perf refinement*: it compares two full-query paths that
//! both start from the same bbox candidates, across query selectivity (rejection
//! rate) and row width (number of property columns):
//!
//!   A (no exact filter): materialize ALL bbox candidates.
//!   B (exact filter):    `filter_features(GeoQuery2D::polygon(..))`, then
//!                        materialize only the survivors.
//!
//! B wins when the rows it avoids materializing (rejection × row width) outweigh
//! the geometry read + predicate over all candidates that filtering costs.
//!
//! Run: cargo run -p packed_spatial_index_geo --release --example end_to_end_box_vs_polygon

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, BinaryArray, Float64Array, StringArray};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};
use packed_spatial_index_geo::{
    FeatureFilterRequest, FeatureReadRequest, GeoIndex, GeoQuery2D, GeometryReadMode,
    PropertyProjection, open,
};

const N: usize = 100_000;

fn main() {
    println!("=== exact polygon filtering: full query time, path A vs B ===");
    println!(
        "N={N} points uniform over [-10,10]^2. A = read all bbox candidates; \
         B = filter_features(polygon) then read survivors. B/A > 1 means exact wins.\n"
    );
    println!(
        "{:>4}  {:<6} {:>6} {:>8} {:>8} {:>13} {:>15} {:>7}",
        "f64", "query", "reject", "K", "M", "A read-all ms", "B filter+read ms", "B/A"
    );
    println!("{}", "-".repeat(76));

    for &cols in &[2usize, 40] {
        let data = build_dataset(N, cols);
        let mut indexed = open(data.clone()).unwrap();
        let GeoIndex::D2(index) = indexed.build(Default::default()).unwrap() else {
            panic!("2D");
        };

        for (name, poly) in queries() {
            let mut k = 0usize;
            let mut m = 0usize;

            // Path A: materialize ALL bbox candidates (skip the exact step).
            let a_ms = best_ms(5, || {
                let cands = index
                    .search_features(GeoQuery2D::polygon(poly.clone()))
                    .unwrap();
                k = cands.len();
                read_full(&data, cands)
            });

            // Path B: exact-filter the candidates, then materialize survivors.
            let b_ms = best_ms(5, || {
                let cands = index
                    .search_features(GeoQuery2D::polygon(poly.clone()))
                    .unwrap();
                let mut filter_source = open(data.clone()).unwrap();
                let survivors = filter_source
                    .filter_features(FeatureFilterRequest::intersects(
                        cands,
                        GeoQuery2D::polygon(poly.clone()),
                    ))
                    .unwrap();
                m = survivors.len();
                read_full(&data, survivors)
            });

            let reject = 100.0 * (1.0 - m as f64 / k as f64);
            let verdict = if a_ms / b_ms >= 1.0 { "win" } else { "loss" };
            println!(
                "{:>4}  {:<6} {:>5.0}% {:>8} {:>8} {:>13.1} {:>15.1} {:>5.2}x {}",
                cols,
                name,
                reject,
                k,
                m,
                a_ms,
                b_ms,
                a_ms / b_ms,
                verdict
            );
        }
        println!();
    }

    phase_breakdown();
}

/// Materialize the given feature refs with all property columns + WKB geometry.
fn read_full(data: &Bytes, features: Vec<packed_spatial_index_geo::FeatureRef>) -> usize {
    let mut source = open(data.clone()).unwrap();
    source
        .read_features(FeatureReadRequest {
            properties: PropertyProjection::AllNonGeometry,
            geometry: GeometryReadMode::Wkb,
            ..FeatureReadRequest::from_features(features)
        })
        .unwrap()
        .batch
        .num_rows()
}

/// Per-phase costs on one cell (wide rows, high rejection) to show the mechanism.
fn phase_breakdown() {
    let cols = 40;
    let data = build_dataset(N, cols);
    let mut indexed = open(data.clone()).unwrap();
    let GeoIndex::D2(index) = indexed.build(Default::default()).unwrap() else {
        panic!("2D");
    };
    let poly = band(-10.0, -10.0, 10.0, 10.0, 0.5);

    let t = Instant::now();
    let cands = index
        .search_features(GeoQuery2D::polygon(poly.clone()))
        .unwrap();
    let t_search = ms(t);
    let k = cands.len();

    let t = Instant::now();
    let mut filter_source = open(data.clone()).unwrap();
    let survivors = filter_source
        .filter_features(FeatureFilterRequest::intersects(
            cands.clone(),
            GeoQuery2D::polygon(poly),
        ))
        .unwrap();
    let t_filter = ms(t);
    let m = survivors.len();

    let t = Instant::now();
    read_full(&data, survivors);
    let t_read_m = ms(t);

    let t = Instant::now();
    read_full(&data, cands);
    let t_read_k = ms(t);

    println!("=== phase breakdown (f64 cols=40, query=band, K={k}, M={m}) ===");
    println!("  search bbox            {t_search:>8.2} ms   -> K candidates (in-memory index)");
    println!(
        "  filter_features        {t_filter:>8.2} ms   -> M survivors (reads geom of all K + predicate)"
    );
    println!("  read survivors (M)     {t_read_m:>8.2} ms   <- path B materialize");
    println!("  read candidates (K)    {t_read_k:>8.2} ms   <- path A materialize");
    println!(
        "\n  path A = search + read(K)     = {:.1} ms",
        t_search + t_read_k
    );
    println!(
        "  path B = search + filter + read(M) = {:.1} ms",
        t_search + t_filter + t_read_m
    );
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn best_ms<F: FnMut() -> usize>(runs: u32, mut f: F) -> f64 {
    let _ = f(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        let n = f();
        std::hint::black_box(n);
        best = best.min(ms(t));
    }
    best
}

// --- queries (non-convex shapes of increasing selectivity) ------------------
fn queries() -> Vec<(&'static str, Polygon<f64>)> {
    vec![
        ("donut", donut()),
        ("star", star(0.0, 0.0, 10.0, 4.0, 5)),
        ("band", band(-10.0, -10.0, 10.0, 10.0, 0.5)),
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
        ring(&rect_pts(-10.0, -10.0, 10.0, 10.0)),
        vec![ring(&rect_pts(-4.0, -4.0, 4.0, 4.0))],
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

// --- in-memory GeoParquet of N points with `prop_cols` f64 columns ----------
fn build_dataset(n: usize, prop_cols: usize) -> Bytes {
    let mut seed = 0xdead_beef_cafe_1234u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut wkb: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
    let mut names: Vec<String> = Vec::with_capacity(n);
    let mut props: Vec<Vec<f64>> = vec![Vec::with_capacity(n); prop_cols];
    for i in 0..n {
        let x = -10.0 + 20.0 * rnd();
        let y = -10.0 + 20.0 * rnd();
        let mut v = Vec::with_capacity(21);
        v.push(1u8);
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&x.to_le_bytes());
        v.extend_from_slice(&y.to_le_bytes());
        wkb.push(Some(v));
        names.push(format!("feature_{i:06}"));
        for col in props.iter_mut() {
            col.push(rnd() * 1000.0);
        }
    }

    let mut cols: Vec<(String, ArrayRef)> = vec![
        (
            "geometry".to_string(),
            Arc::new(BinaryArray::from(
                wkb.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "name".to_string(),
            Arc::new(StringArray::from(names)) as ArrayRef,
        ),
    ];
    for (i, col) in props.into_iter().enumerate() {
        cols.push((
            format!("v{i}"),
            Arc::new(Float64Array::from(col)) as ArrayRef,
        ));
    }

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
