# packed_spatial_index_geo

Build a [`packed_spatial_index`](https://crates.io/crates/packed_spatial_index)
spatial index from a **GeoParquet** file.

GeoParquet stores geometry plus, since 1.1, an optional per-row *bbox covering*
column — but it has no per-row spatial index, only per-row-group statistics. That
prunes whole row groups; it cannot pinpoint individual features. This crate fills
the gap.

## Two-phase by design

The heavy `arrow` / `parquet` / `geoparquet` dependencies live only in this crate.
The `packed_spatial_index` core that *queries* the result stays lean (wasm / edge
friendly). Build runs server-side once; query runs anywhere.

```text
GeoParquet  ──(this crate, native)──►  index / PSINDEX  ──(core, anywhere)──►  queries
```

## Accelerator — query for row indices

`build_index_2d` / `build_index_3d` build an in-memory index whose item id is the
GeoParquet **row index**. Query results are row indices you read back from the
original file.

```rust,no_run
use std::fs::File;
use packed_spatial_index::Box2D;
use packed_spatial_index_geo::{build_index_2d, BuildOpts};

let file = File::open("cities.parquet")?;
let index = build_index_2d(file, BuildOpts::default())?;
let rows = index.search(Box2D::new(-10.0, 35.0, 20.0, 60.0)); // -> Vec<usize>
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Converter — a self-describing, streamable index

`convert_2d` / `convert_3d` build the index *and* attach each row's WKB geometry
as a leaf-ordered payload, plus the CRS, serialized to a `PSINDEX` blob. That blob
is queryable by the core streaming engine straight from cloud storage — a window /
kNN / raycast query returns the actual geometry in a handful of range reads, with
no Parquet re-read.

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{convert_2d, ConvertOpts};

let file = File::open("cities.parquet")?;
let psindex: Vec<u8> = convert_2d(file, ConvertOpts::default())?;
std::fs::write("cities.psindex", &psindex)?;
// Serve cities.psindex over HTTP range requests and query it with
// packed_spatial_index's StreamIndex2D — see the core crate's streaming docs.
# Ok::<(), Box<dyn std::error::Error>>(())
```

Set `ConvertOpts { compact_f32: true, .. }` for a roughly half-size file (queries
become a conservative superset; re-check exact hits against the payload geometry).

A runnable end-to-end version (inspect → convert → write a `.psindex`) lives in
[`examples/convert.rs`](examples/convert.rs):

```text
cargo run --example convert -- path/to/file.parquet
```

[`inspect`] reports a file's geometry metadata (column, dims, encoding, CRS,
covering, extent, row count) without reading any rows.

## Scope

* Boxes come from the **bbox covering** column when present, otherwise from each
  geometry's **WKB** envelope.
* The geometry is only decoded when there is no covering column, or when the
  converter needs the WKB payload. So a native geoarrow encoding *with* a covering
  column works for the accelerator; decoding a native encoding (no covering, or a
  payload request) returns `GeoError::UnsupportedEncoding`.
* Geometry columns may be `Binary`, `LargeBinary`, or `BinaryView`.
* 2D and 3D (`XYZ` / `XYZM`).
* Null / empty geometry: the accelerator keeps `id == row index`, so it has no
  room to skip rows and returns `GeoError::NullGeometry`. The converter can drop
  such rows with `ConvertOpts { skip_null: true, .. }` (its output is
  self-contained, so compacted ids are fine).

Licensed under Apache-2.0.
