# Geospatial Parquet Index

[![crates.io](https://img.shields.io/crates/v/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![docs.rs](https://docs.rs/packed_spatial_index_geo/badge.svg)](https://docs.rs/packed_spatial_index_geo)
[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![License](https://img.shields.io/crates/l/packed_spatial_index_geo.svg)](https://github.com/Filyus/packed_spatial_index/blob/main/LICENSE)

Build a [`packed_spatial_index`](https://crates.io/crates/packed_spatial_index)
spatial index for **GeoParquet** and native Apache Parquet `GEOMETRY` /
`GEOGRAPHY` columns. These formats store geometry plus, in some cases, optional
bbox/statistics metadata — but they do not provide a per-row spatial index that
pinpoints individual features. This crate fills the gap:

- **accelerator** — build an in-memory index over the features; a query returns
  `FeatureRef` values that preserve source row numbers even when rows are
  skipped or split
- **converter** — build the index and attach a leaf-ordered payload (by default,
  `FeatureRef` + WKB geometry), serialized to a self-describing,
  **streamable `PSINDEX`** that answers window / kNN / raycast queries straight
  from object storage in a handful of range reads, with no Parquet re-read
- **discovery / inspection** — open a `GeoDataset`, list usable geometry
  candidates before selecting one, then inspect the selected column's typed
  metadata (dims, encoding, CRS, edges, extent, row count)
- 2D and 3D, optional **`f32`** storage for half-size files, and **`skip_null`** to
  drop empty geometry

The heavy `arrow` / `parquet` / `geoparquet` dependencies live only here; the
`packed_spatial_index` core that *queries* the result stays lean (wasm / edge
friendly). Build runs server-side once; query runs anywhere.

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{open, BuildRequest, Box2D, GeoIndex};

let mut dataset = open(File::open("cities.parquet")?)?;
let index = dataset.build(BuildRequest::default())?;
let GeoIndex::D2(index) = index else { panic!("expected 2D geometry") };
let features = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0));
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Installation

Requires Rust 1.89 or newer.

```toml
[dependencies]
packed_spatial_index_geo = "0.5"
```

## API at a glance

Open the Parquet source once with [`open`][open], inspect the metadata-only
[`GeoDiscovery`][GeoDiscovery], then run the operation you need through the
[`GeoDataset`][GeoDataset] session. Geometry selection is explicit where it
matters: use [`GeometrySelector::Name`][GeometrySelector] for a named column, or
the default policy for GeoParquet primary / single native Parquet geospatial
files.

| Task | API |
| --- | --- |
| Open a source | [`open`][open] |
| List usable geometry columns | [`GeoDataset::discovery`][discovery], [`GeoDiscovery`][GeoDiscovery], [`GeometryColumnInfo`][GeometryColumnInfo] |
| Select a geometry column | [`GeoDataset::select`][select], [`GeometrySelector`][GeometrySelector], [`GeometryColumn`][GeometryColumn] |
| Profile the selected column | [`GeoDataset::inspect`][inspect], [`InspectRequest`][InspectRequest], [`GeometryProfile`][GeometryProfile] |
| Read feature boxes / payloads | [`GeoDataset::scan`][scan], [`ScanRequest`][ScanRequest], [`GeometryScan`][GeometryScan] |
| Build an in-memory feature index | [`GeoDataset::build`][build], [`BuildRequest`][BuildRequest], [`GeoIndex`][GeoIndex], [`GeoIndex2D::search_features`][search_features_2d], [`GeoIndex3D::search_features`][search_features_3d] |
| Convert to streamable `PSINDEX` | [`GeoDataset::convert`][convert], [`GeoDataset::convert_into`][convert_into], [`ConvertRequest`][ConvertRequest], [`GeoArtifact`][GeoArtifact] |
| Choose index dimensions / precision | [`IndexDimsRequest`][IndexDimsRequest], [`StoragePrecision`][StoragePrecision] |
| Control nulls and antimeridian behavior | [`NullPolicy`][NullPolicy], [`EnvelopePolicy`][EnvelopePolicy], [`AntimeridianPolicy`][AntimeridianPolicy] |
| Pick artifact payloads | [`PayloadPlan`][PayloadPlan], [`PropertyProjection`][PropertyProjection], [`FeatureRef`][FeatureRef] |
| Decode default payloads | [`decode_feature_ref_payload`][decode_feature_ref_payload], [`decode_feature_wkb_payload`][decode_feature_wkb_payload] |
| Read the geo artifact manifest | [`read_geo_manifest`][read_geo_manifest], [`GeoArtifactManifest`][GeoArtifactManifest] |
| Use the CLI | `gp2psindex discover`, `inspect`, `build`, `validate` |

The in-memory `GeoIndex` search methods return [`FeatureRef`][FeatureRef]
values. A feature ref always carries the source `row_number`; `part` is set when
one source row becomes multiple index entries, for example after antimeridian
splitting.

## When to use it

Reach for the **accelerator** when the geospatial Parquet file stays put and you
just want fast windowed / kNN / raycast lookups against it: the index is tiny
(boxes + feature refs), and a query hands you source row numbers to read back
from the file.

Reach for the **converter** when you want a portable, cloud-served store: by
default it folds feature refs and geometry into one self-describing `PSINDEX`
blob that the core streaming engine queries directly over HTTP range requests.
Use `PayloadPlan::RowRef` when you want only a compact sidecar index that points
back to the original source rows.

```text
GeoParquet / Parquet geo  ──(this crate, native)──►  index / PSINDEX  ──(core, anywhere)──►  queries
```

The two phases need not share a runtime: convert on a server, query from the edge.

## Converter — a self-describing, streamable index

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{open, ConvertRequest};

let mut dataset = open(File::open("cities.parquet")?)?;
let psindex: Vec<u8> = dataset.convert(ConvertRequest::default())?;
std::fs::write("cities.psindex", &psindex)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Serve `cities.psindex` over HTTP range requests (or read it locally) and query it
with the re-exported streaming types — no second dependency needed:

```rust,no_run
use packed_spatial_index_geo::{
    decode_feature_wkb_payload, Box2D, SliceReader, StreamIndex2D,
};

let bytes = std::fs::read("cities.psindex")?;
let index = StreamIndex2D::open(SliceReader::new(bytes))?;
for (_item, payload) in index.search_payloads(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    let (feature, wkb) = decode_feature_wkb_payload(&payload).expect("default geo payload");
    println!("row {}: {} WKB bytes", feature.row_number, wkb.len());
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`ConvertRequest { precision: StoragePrecision::F32, .. }` makes a roughly
half-size file (queries become a conservative superset; re-check exact hits
against the payload geometry). `ConvertRequest` skips null or empty geometries by
default; `BuildRequest` errors by default.

Payload modes:

- `PayloadPlan::RowWkb` (default): fixed `FeatureRef` record followed by WKB.
- `PayloadPlan::RowRef`: fixed-width `FeatureRef` only; smallest sidecar mode.
- `PayloadPlan::FeatureJson`: GeoJSON Feature bytes with projected properties.
- `PayloadPlan::None`: no payload section.

Converted `PSINDEX` files also carry an optional app-private `geoM` manifest
chunk. Core `packed_spatial_index` readers skip it; this crate reads it with
`read_geo_manifest`.

The crate ships a CLI, `gp2psindex`, for the file-to-file path:

```text
cargo run --bin gp2psindex -- discover path/to/file.parquet --json
cargo run --bin gp2psindex -- inspect path/to/file.parquet --exact
cargo run --bin gp2psindex -- build path/to/file.parquet path/to/file.psi \
  --payload row-wkb --dims auto --nulls skip
cargo run --bin gp2psindex -- validate path/to/file.parquet
```

## Scope

- Inputs may be GeoParquet files with `geo` metadata or native Apache Parquet
  `GEOMETRY` / `GEOGRAPHY` logical-type columns. When both are present,
  GeoParquet `primary_column` is the default; use `GeometrySelector::Name` or
  `gp2psindex --geometry-column` to select explicitly.
- Use `open(...).discovery()` when a file may contain several geometry
  candidates and you want metadata-only selection status before reading rows.
- Boxes come from the **bbox covering** column when present, otherwise from each
  geometry's **WKB** or **GeoArrow** envelope.
- Native Parquet `GEOMETRY` / `GEOGRAPHY` columns are WKB by definition, so they
  work for envelope scans and `RowWkb` payloads even without GeoParquet `geo`
  metadata.
- `GEOGRAPHY` is indexed as a coordinate-axis-aligned bounding box over the
  stored WKB coordinates. This is a candidate index, not exact spherical or
  ellipsoidal predicate evaluation.
- GeoParquet GeoArrow encodings `point`, `linestring`, `polygon`, `multipoint`,
  `multilinestring`, and `multipolygon` can be scanned without a covering column
  and can be emitted as ISO WKB payloads.
- Geometry columns may be `Binary`, `LargeBinary`, or `BinaryView`.
- 2D and 3D (`XYZ` / `XYZM`).
- Null / empty geometry: `BuildRequest` defaults to `NullPolicy::Error`;
  `ConvertRequest` defaults to `NullPolicy::Skip`. `FeatureRef::row_number`
  preserves the original source row number.

## License

Licensed under the [Apache License 2.0](https://github.com/Filyus/packed_spatial_index/blob/main/LICENSE).

<!-- docs.rs API links -->
[open]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open.html
[GeoDataset]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html
[discovery]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.discovery
[select]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.select
[inspect]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.inspect
[scan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.scan
[build]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.build
[convert]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.convert
[convert_into]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.convert_into
[GeoDiscovery]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDiscovery.html
[GeometryColumnInfo]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeometryColumnInfo.html
[GeometrySelector]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeometrySelector.html
[GeometryColumn]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeometryColumn.html
[InspectRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.InspectRequest.html
[GeometryProfile]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeometryProfile.html
[ScanRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.ScanRequest.html
[GeometryScan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeometryScan.html
[BuildRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.BuildRequest.html
[GeoIndex]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoIndex.html
[search_features_2d]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex2D.html#method.search_features
[search_features_3d]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex3D.html#method.search_features
[ConvertRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.ConvertRequest.html
[GeoArtifact]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifact.html
[IndexDimsRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.IndexDimsRequest.html
[StoragePrecision]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.StoragePrecision.html
[NullPolicy]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.NullPolicy.html
[EnvelopePolicy]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.EnvelopePolicy.html
[AntimeridianPolicy]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.AntimeridianPolicy.html
[PayloadPlan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.PayloadPlan.html
[PropertyProjection]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.PropertyProjection.html
[FeatureRef]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.FeatureRef.html
[decode_feature_ref_payload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.decode_feature_ref_payload.html
[decode_feature_wkb_payload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.decode_feature_wkb_payload.html
[read_geo_manifest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.read_geo_manifest.html
[GeoArtifactManifest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactManifest.html
