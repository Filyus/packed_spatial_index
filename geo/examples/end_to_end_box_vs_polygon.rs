//! When does exact polygon filtering pay off? Full end-to-end query time.
//!
//! For a non-rectangular (polygon) query the exact step is REQUIRED for a correct
//! answer — the index only narrows by bounding box. This bench measures the
//! cost/benefit as a perf refinement, across query selectivity (rejection rate)
//! and row width (property columns), for three full-query paths off the same
//! bbox candidates:
//!
//!   A  read-all       — materialize ALL bbox candidates (no exact step).
//!   B  filter_features — exact-filter by RE-READING source geometry, then read
//!                        survivors.
//!   C  filter_hits     — exact-filter the geometry already in the artifact
//!                        payload (`search_hits` → `filter_hits`), then read
//!                        survivors. No source geometry re-read.
//!
//! B's filter re-reads every candidate's geometry (≈ the cost of reading the
//! rows), so it loses. C reuses the geometry the index already produced, so it
//! only pays to read the survivors.
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
    ConvertRequest, FeatureFilterRequest, FeatureReadRequest, FeatureRef, GeoArtifactIndex,
    GeoArtifactIndex2D, GeoQuery2D, GeometryReadMode, NonPlanarExactPolicy, PropertyProjection,
    SliceReader, SpatialPredicate, open_geo_index, open_geoparquet,
};

const N: usize = 100_000;

fn main() {
    println!("=== exact polygon filtering: full query time, A vs B vs C ===");
    println!(
        "N={N} points uniform over [-10,10]^2. A=read-all, B=filter_features (re-reads geometry), \
         C=filter_hits (geometry from artifact payload). Lower is better.\n"
    );
    println!(
        "{:>4}  {:<6} {:>6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>7}",
        "f64", "query", "reject", "K", "M", "A read", "B feat", "C hits", "C/A"
    );
    println!("{}", "-".repeat(82));

    for &cols in &[2usize, 40] {
        let data = build_dataset(N, cols);
        let artifact = open_geoparquet(data.clone())
            .unwrap()
            .convert(ConvertRequest::default())
            .unwrap();
        let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(artifact)).unwrap()
        else {
            panic!("2D");
        };

        for (name, poly) in queries() {
            let mut k = 0usize;
            let mut m = 0usize;

            // A: materialize ALL bbox candidates.
            let a_ms = best_ms(5, || {
                let cands = index
                    .search_features(GeoQuery2D::polygon(poly.clone()))
                    .unwrap();
                k = cands.len();
                read_full(&data, cands)
            });

            // B: exact-filter by re-reading source geometry, then read survivors.
            let b_ms = best_ms(5, || {
                let cands = index
                    .search_features(GeoQuery2D::polygon(poly.clone()))
                    .unwrap();
                let mut filter_source = open_geoparquet(data.clone()).unwrap();
                let survivors = filter_source
                    .filter_features(FeatureFilterRequest::intersects(
                        cands,
                        GeoQuery2D::polygon(poly.clone()),
                    ))
                    .unwrap();
                read_full(&data, survivors)
            });

            // C: exact-filter the geometry already in the payload, then read survivors.
            let c_ms = best_ms(5, || {
                let survivors = filter_hits_refs(&index, &poly);
                m = survivors.len();
                read_full(&data, survivors)
            });

            let reject = 100.0 * (1.0 - m as f64 / k as f64);
            println!(
                "{:>4}  {:<6} {:>5.0}% {:>8} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>5.2}x",
                cols,
                name,
                reject,
                k,
                m,
                a_ms,
                b_ms,
                c_ms,
                a_ms / c_ms
            );
        }
        println!();
    }

    phase_breakdown();
}

/// `search_hits` → `filter_hits`, returning surviving feature refs.
fn filter_hits_refs<R: packed_spatial_index_geo::RangeReader>(
    index: &GeoArtifactIndex2D<R>,
    poly: &Polygon<f64>,
) -> Vec<FeatureRef> {
    let hits = index
        .search_hits(GeoQuery2D::polygon(poly.clone()))
        .unwrap();
    index
        .filter_hits(
            hits,
            GeoQuery2D::polygon(poly.clone()),
            SpatialPredicate::Intersects,
            NonPlanarExactPolicy::Reject,
        )
        .unwrap()
        .into_iter()
        .map(|hit| hit.feature)
        .collect()
}

/// Materialize the given feature refs with all property columns + WKB geometry.
fn read_full(data: &Bytes, features: Vec<FeatureRef>) -> usize {
    let mut source = open_geoparquet(data.clone()).unwrap();
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
    let artifact = open_geoparquet(data.clone())
        .unwrap()
        .convert(ConvertRequest::default())
        .unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(artifact)).unwrap() else {
        panic!("2D");
    };
    let poly = band(-10.0, -10.0, 10.0, 10.0, 0.5);

    // Warm caches so the per-phase single-run timings below are comparable.
    std::hint::black_box(filter_hits_refs(&index, &poly));
    std::hint::black_box(read_full(
        &data,
        index
            .search_features(GeoQuery2D::polygon(poly.clone()))
            .unwrap(),
    ));

    let t = Instant::now();
    let hits = index
        .search_hits(GeoQuery2D::polygon(poly.clone()))
        .unwrap();
    let t_search_hits = ms(t);
    let k = hits.len();

    let t = Instant::now();
    let survivors: Vec<FeatureRef> = index
        .filter_hits(
            hits,
            GeoQuery2D::polygon(poly.clone()),
            SpatialPredicate::Intersects,
            NonPlanarExactPolicy::Reject,
        )
        .unwrap()
        .into_iter()
        .map(|hit| hit.feature)
        .collect();
    let t_filter_hits = ms(t);
    let m = survivors.len();

    let t = Instant::now();
    read_full(&data, survivors);
    let t_read_m = ms(t);

    let cands = index.search_features(GeoQuery2D::polygon(poly)).unwrap();
    let t = Instant::now();
    read_full(&data, cands);
    let t_read_k = ms(t);

    println!("=== phase breakdown (f64 cols=40, query=band, K={k}, M={m}) ===");
    println!("  search_hits (geom payload)  {t_search_hits:>8.2} ms");
    println!("  filter_hits (no re-read)    {t_filter_hits:>8.2} ms");
    println!("  read survivors (M)          {t_read_m:>8.2} ms");
    println!("  read candidates (K)         {t_read_k:>8.2} ms");
    println!(
        "\n  C trades a cheap geometry pass (search_hits + filter_hits) for reading M rows\n  \
         instead of K. It wins once read(K) - read(M) exceeds that pass — see the matrix."
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
