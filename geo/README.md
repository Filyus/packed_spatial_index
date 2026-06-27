# Packed Spatial Index — Geospatial Parquet

[![crates.io](https://img.shields.io/crates/v/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![docs.rs](https://docs.rs/packed_spatial_index_geo/badge.svg)](https://docs.rs/packed_spatial_index_geo)
[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![License](https://img.shields.io/crates/l/packed_spatial_index_geo.svg)](https://github.com/Filyus/packed_spatial_index/blob/main/LICENSE)

Build a [`packed_spatial_index`](https://crates.io/crates/packed_spatial_index)
spatial index from **GeoParquet** or Apache Parquet's native geospatial
`GEOMETRY` / `GEOGRAPHY` logical types. These formats store geometry plus, in
some cases, optional bbox/statistics metadata — but they do not provide a per-row
spatial index that pinpoints individual features. This crate fills the gap:

- **accelerator** — build an in-memory index over the rows; a query returns **row
  indices** into the original file
- **converter** — build the index and attach a leaf-ordered payload (by default,
  original source row id + WKB geometry), serialized to a self-describing,
  **streamable `PSINDEX`** that answers window / kNN / raycast queries straight
  from object storage in a handful of range reads, with no Parquet re-read
- **discovery / inspection** — list usable geometry candidates before selecting
  one, then read the selected column's metadata (dims, encoding, CRS, metadata
  source, covering, extent, row count)
- 2D and 3D, optional **`f32`** storage for half-size files, and **`skip_null`** to
  drop empty geometry

The heavy `arrow` / `parquet` / `geoparquet` dependencies live only here; the
`packed_spatial_index` core that *queries* the result stays lean (wasm / edge
friendly). Build runs server-side once; query runs anywhere.

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{build_index_2d, BuildOpts, Box2D};

let index = build_index_2d(File::open("cities.parquet")?, BuildOpts::default())?;
let rows = index.search(Box2D::new(-10.0, 35.0, 20.0, 60.0)); // -> Vec<usize>
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Installation

Requires Rust 1.89 or newer.

```toml
[dependencies]
packed_spatial_index_geo = "0.3"
```

## When to use it

Reach for the **accelerator** when the geospatial Parquet file stays put and you
just want fast windowed / kNN / raycast lookups against it: the index is tiny
(boxes + row ids), and a query hands you row indices to read back from the file.

Reach for the **converter** when you want a portable, cloud-served store: by
default it folds source row ids and geometry into one self-describing `PSINDEX`
blob that the core streaming engine queries directly over HTTP range requests.
Use `ConvertPayload::RowIds` when you want only a compact sidecar index that
points back to the original source rows.

```text
GeoParquet / Parquet geo  ──(this crate, native)──►  index / PSINDEX  ──(core, anywhere)──►  queries
```

The two phases need not share a runtime: convert on a server, query from the edge.

## Converter — a self-describing, streamable index

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{convert_2d, ConvertOpts};

let psindex: Vec<u8> = convert_2d(File::open("cities.parquet")?, ConvertOpts::default())?;
std::fs::write("cities.psindex", &psindex)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Serve `cities.psindex` over HTTP range requests (or read it locally) and query it
with the re-exported streaming types — no second dependency needed:

```rust,no_run
use packed_spatial_index_geo::{
    decode_row_wkb_payload, Box2D, SliceReader, StreamIndex2D,
};

let bytes = std::fs::read("cities.psindex")?;
let index = StreamIndex2D::open(SliceReader::new(bytes))?;
for (_item, payload) in index.search_payloads(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    let (row, wkb) = decode_row_wkb_payload(&payload).expect("default geo payload");
    println!("feature {row}: {} WKB bytes", wkb.len());
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`ConvertOpts { compact_f32: true, .. }` makes a roughly half-size file (queries
become a conservative superset; re-check exact hits against the payload geometry).
`ConvertOpts { skip_null: true, .. }` drops null or empty geometries instead of
erroring.

Payload modes:

- `ConvertPayload::RowWkb` (default): `u64le original_row_id` followed by WKB.
- `ConvertPayload::RowIds`: fixed-width `u64le original_row_id` only; smallest
  sidecar mode and compatible with GeoParquet-native GeoArrow encodings when a
  covering column is present.
- `ConvertPayload::None` or `include_payload: false`: no payload section.

The crate ships a CLI, `gp2psindex`, for the file-to-file path:

```text
cargo run --bin gp2psindex -- path/to/file.parquet      # writes path/to/file.parquet.psi
# flags: --f32  --strict (error on null)  --geometry-column name
#        --payload none|row-id|row-wkb  --no-payload  --no-interleave
cargo run --bin gp2psindex -- inspect path/to/file.parquet --json
```

## Scope

- Inputs may be GeoParquet files with `geo` metadata or native Apache Parquet
  `GEOMETRY` / `GEOGRAPHY` logical-type columns. When both are present,
  GeoParquet `primary_column` is the default; use `ReadOpts::geometry_column`,
  `BuildOpts::geometry_column`, `ConvertOpts::geometry_column`, or
  `gp2psindex --geometry-column` to select explicitly.
- Use `discover` / `discover_with_opts` when a file may contain several geometry
  candidates and you want metadata-only selection status before reading rows.
- Boxes come from the **bbox covering** column when present, otherwise from each
  geometry's **WKB** envelope.
- Native Parquet `GEOMETRY` / `GEOGRAPHY` columns are WKB by definition, so they
  work for envelope scans and `RowWkb` payloads even without GeoParquet `geo`
  metadata.
- `GEOGRAPHY` is indexed as a coordinate-axis-aligned bounding box over the
  stored WKB coordinates. This is a candidate index, not exact spherical or
  ellipsoidal predicate evaluation.
- The geometry is only decoded when there is no covering column, or when the
  converter needs WKB. So a native geoarrow encoding *with* a covering column
  works for the accelerator and `ConvertPayload::RowIds`; decoding a native
  encoding (no covering, or a WKB payload request) returns
  `GeoError::UnsupportedEncoding`.
- Geometry columns may be `Binary`, `LargeBinary`, or `BinaryView`.
- 2D and 3D (`XYZ` / `XYZM`).
- Null / empty geometry: the accelerator keeps `id == row index`, so it has no
  room to skip rows and returns `GeoError::NullGeometry`. The converter can drop
  such rows with `skip_null`; item ids are compacted, and the default / row-id
  payloads preserve the original source row id.

## License

Licensed under the [Apache License 2.0](https://github.com/Filyus/packed_spatial_index/blob/main/LICENSE).
