# Geospatial Source Index

[![crates.io](https://img.shields.io/crates/v/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![docs.rs](https://docs.rs/packed_spatial_index_geo/badge.svg)](https://docs.rs/packed_spatial_index_geo)
[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![License](https://img.shields.io/crates/l/packed_spatial_index_geo.svg)](https://github.com/Filyus/packed_spatial_index/blob/main/LICENSE)

Build a [`packed_spatial_index`](https://crates.io/crates/packed_spatial_index)
spatial index for **GeoParquet**, native Apache Parquet `GEOMETRY` /
`GEOGRAPHY` columns, **FlatGeobuf**, and **GeoJSON**. These formats store
geometry plus, in some cases, optional bbox/statistics metadata — but they do
not provide a portable per-row `PSINDEX` sidecar that pinpoints individual
features and can be streamed from object storage. This crate fills the gap:

- **accelerator** — build an in-memory index over the features; a query returns
  `FeatureRef` values that preserve source row numbers even when rows are
  skipped or split
- **converter** — build the index and attach a leaf-ordered payload (by default,
  `FeatureRef` + WKB geometry), serialized to a self-describing,
  **streamable `PSINDEX`** that answers window, polygon, and 3D frustum
  candidate queries straight from object storage in a handful of range reads,
  with no Parquet re-read. kNN and raycast queries use the in-memory
  accelerator path.
- **discovery / inspection** — open a `GeoDataset`, list usable geometry
  candidates before selecting one, then inspect the selected column's typed
  metadata (dims, encoding, CRS, edges, extent, row count)
- 2D and 3D, optional **`f32`** storage for half-size files, and **`skip_null`** to
  drop empty geometry

The source-side dependencies sit behind format features. Defaults enable
`parquet`, `flatgeobuf`, and `geojson`, so the CLI can build from all supported
inputs. Turn defaults off (`default-features = false`) and the crate is
query-only — open a pre-built `PSINDEX`, stream candidate queries, exact-filter
geometry from the payload — with no `arrow` / `parquet`, small enough for
**wasm / edge**. Build runs server-side once; query runs anywhere.

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{open, BuildRequest, Box2D, GeoIndex};

let mut dataset = open(File::open("cities.parquet")?)?;
let index = dataset.build(BuildRequest::default())?;
let GeoIndex::D2(index) = index else { panic!("expected 2D geometry") };
let features = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Installation

Requires Rust 1.89 or newer.

```toml
[dependencies]
packed_spatial_index_geo = "0.17"
```

### Features

- **`parquet`** *(default)* — the Parquet source side: `open`, `GeoDataset`
  (discovery, inspection, validation, feature read-back), `build` / `convert`,
  and the `gp2psindex` CLI. Pulls in `arrow` + `parquet`.
- **`flatgeobuf`** *(default)* — open `.fgb` sources with `open_flatgeobuf`,
  then scan, build, convert, and read features back without Arrow.
- **`geojson`** *(default)* — open GeoJSON with `open_geojson` /
  `open_geojson_slice`, then scan, build, convert, and read features back.
  GeoJSON is parsed eagerly in memory.
- **`async`** — open and query streamable `PSINDEX` artifacts over an
  [`AsyncRangeReader`][AsyncRangeReader], adding `open_geo_index_async` and `search_hits_async`.

To query a pre-built `PSINDEX` from a browser or edge worker, drop the default
feature so `arrow` / `parquet` never enter the build:

```toml
[dependencies]
packed_spatial_index_geo = { version = "0.17", default-features = false, features = ["async"] }
```

That leaves the crate query-only — [`open_geo_index`][open_geo_index] /
`open_geo_index_async`, `search_items` / `search_hits`,
[`GeoArtifactIndex2D::filter_hits`][filter_hits] (exact intersection over the
payload geometry), and payload decoding — compiling to `wasm32`. Only reading a
source file needs a format feature.

## API at a glance

Open the Parquet source once with [`open`][open], inspect the metadata-only
[`GeoDiscovery`][GeoDiscovery], then run the operation you need through the
[`GeoDataset`][GeoDataset] session. Geometry selection is explicit where it
matters: use [`GeometrySelector::Name`][GeometrySelector] for a named column, or
the default policy for GeoParquet primary / single native Parquet geospatial
files.

| Task | Start here |
| --- | --- |
| Open a source | [`open`][open] |
| Open GeoJSON | [`open_geojson`][open_geojson], [`open_geojson_slice`][open_geojson_slice] |
| Open FlatGeobuf | [`open_flatgeobuf`][open_flatgeobuf] |
| Discover columns | [`GeoDataset::discovery`][discovery], [`GeoDiscovery`][GeoDiscovery] |
| Select a column | [`GeoDataset::select`][select], [`GeometrySelector`][GeometrySelector] |
| Inspect metadata | [`GeoDataset::inspect`][inspect], [`InspectRequest`][InspectRequest] |
| Validate input | [`GeoDataset::validate`][validate] |
| Scan boxes / payloads | [`GeoDataset::scan`][scan], [`ScanRequest`][ScanRequest] |
| Build an index | [`GeoDataset::build`][build], [`GeoIndex`][GeoIndex] |
| Build a half-size (`f32`) index | [`IndexBuildOptions::precision`][IndexBuildOptions], [`StoragePrecision`][StoragePrecision] |
| Build an index and a `PSINDEX` in one scan | [`GeoIndex::from_scan`][from_scan], [`GeoArtifact::from_scan`][artifact_from_scan] |
| Query the index | [`GeoIndex2D::search_features`][search_features_2d], [`GeoIndex3D::search_features`][search_features_3d] |
| Nearest features (kNN) | [`GeoIndex2D::nearest_features`][nearest_features], `nearest_features_haversine` |
| Raycast features | [`GeoIndex3D::raycast_features`][raycast_features], `raycast_closest_feature` |
| Convert to `PSINDEX` | [`GeoDataset::convert`][convert] |
| Open a `PSINDEX` | [`open_geo_index`][open_geo_index] |
| Open a `PSINDEX` over async range I/O | [`open_geo_index_async`][open_geo_index_async] (`async` feature) |
| Query a `PSINDEX` | [`GeoArtifactIndex2D::search_hits`][search_hits], [`GeoHit`][GeoHit] |
| Choose a query shape | [`GeoQuery2D`][GeoQuery2D] (box / polygon / radius), [`GeoQuery3D`][GeoQuery3D] (box / frustum) |
| Exact-filter source hits | [`GeoDataset::filter_features`][filter_features], [`FeatureFilterRequest`][FeatureFilterRequest] |
| Exact-filter `PSINDEX` hits | [`GeoArtifactIndex2D::filter_hits`][filter_hits] |
| Read Parquet source rows | [`GeoDataset::read_features`][read_features], [`FeatureReadRequest`][FeatureReadRequest] |
| Read GeoJSON / FlatGeobuf source rows | `read_features`, [`FeatureRecord`][FeatureRecord] |
| Tune requests | [`IndexDimsRequest`][IndexDimsRequest], [`NullPolicy`][NullPolicy] |
| Pick payloads | [`PayloadPlan`][PayloadPlan], [`FeatureRef`][FeatureRef] |
| Use the CLI | See [CLI](#cli) |

The in-memory `GeoIndex` search methods return [`FeatureRef`][FeatureRef]
values. A feature ref always carries the source `row_number`; scans and converted
artifacts also fill `row_group` / `row_in_group` so source rows can be read back
efficiently. `part` is set when one source row becomes multiple index entries,
for example after antimeridian splitting.

Enable the `async` feature to open the same streamable artifacts through an
`AsyncRangeReader`. The async artifact methods mirror window, polygon, and 3D
frustum candidate queries, including payload-returning `search_hits_async`.

## Examples

The crate includes small runnable examples that use the bundled Apache Parquet
geospatial fixtures:

```text
cd geo
cargo run --example discover
cargo run --example build_index
cargo run --example convert_and_query
cargo run --example feature_json_payload
```

## Documentation

- **[Guide](docs/guide.md)** — validate before building, convert to a
  streamable `PSINDEX`, query source rows back with exact filtering, spherical
  radius queries.
- **[When to use it](docs/when-to-use.md)** — accelerator vs. converter, and
  how this crate differs from
  [`oxigdal-geoparquet`](https://crates.io/crates/oxigdal-geoparquet).
- **API reference** — [docs.rs/packed_spatial_index_geo](https://docs.rs/packed_spatial_index_geo).

## CLI

Install the command with Cargo:

```text
cargo install packed_spatial_index_geo --locked
```

Run it directly after install, or prefix the same arguments with
`cargo run --bin gp2psindex --` from a repository checkout.

Start with discovery when you do not know which geometry columns the file
contains. `gp2psindex` detects `.parquet`, `.fgb`, `.geojson`, and `.json`
inputs by extension and falls back to a small signature check; pass `--format`
to override detection.

```text
gp2psindex discover input.parquet --json
gp2psindex inspect places.geojson --json
gp2psindex inspect layer.fgb --format flatgeobuf
```

Inspect the selected column when you want the resolved encoding, CRS, dimensions,
edge model, and extent:

```text
gp2psindex inspect input.parquet --exact
```

Validate before building an artifact from files produced by another pipeline.
Default validation is metadata-only; `--exact` also scans rows and reports bad
WKB, null-policy failures, antimeridian rejects, and payload projection errors:

```text
gp2psindex validate input.parquet --json
gp2psindex validate input.parquet \
  --exact \
  --strict
```

Build a reusable `PSINDEX` sidecar for repeated queries:

```text
gp2psindex build input.parquet output.psi \
  --payload row-wkb \
  --dims auto \
  --nulls skip

gp2psindex build places.geojson places.psi \
  --payload feature-json \
  --properties all

gp2psindex build layer.fgb layer.psi \
  --payload row-wkb
```

Query a sidecar and read projected source properties back as NDJSON:

```text
gp2psindex query input.parquet output.psi \
  --bbox -10,35,20,60 \
  --properties include:name,pop

gp2psindex query places.geojson places.psi \
  --bbox -10,35,20,60 \
  --properties include:name \
  --json
```

Add `--exact` to filter bbox candidates with a planar geometry/rectangle
intersection predicate before reading the final rows:

```text
gp2psindex query input.parquet output.psi \
  --bbox -10,35,20,60 \
  --exact \
  --predicate intersects \
  --properties include:name,pop
```

Add WKB bytes to JSON output when a downstream step needs the geometry:

```text
gp2psindex query input.parquet output.psi \
  --bbox -10,35,20,60 \
  --properties include:name \
  --geometry wkb \
  --json
```

For `GEOGRAPHY(SPHERICAL)` / GeoParquet spherical edges, `query --radius`
performs a lon/lat radius lookup. See [Spherical radius queries in the
guide](docs/guide.md#spherical-radius-queries) for the API path and payload
options.

```text
gp2psindex query input.parquet output.psi \
  --radius -73.9857,40.7484,500 \
  --properties include:name \
  --json
```

## Scope

- Inputs may be GeoParquet files with `geo` metadata or native Apache Parquet
  `GEOMETRY` / `GEOGRAPHY` logical-type columns, FlatGeobuf files, or GeoJSON
  documents. When both GeoParquet metadata and native Parquet geometry columns
  are present, GeoParquet `primary_column` is the default; use
  `GeometrySelector::Name` or the CLI's `--geometry-column` option to select
  explicitly.
- FlatGeobuf and GeoJSON sources expose a single geometry named `geometry`.
  GeoJSON input accepts `FeatureCollection`, single `Feature`, and bare geometry
  documents; v1 parses the whole document in memory.
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
- Spherical radius exact filtering is available for spherical geography
  `Point` / `MultiPoint` sources. Ellipsoidal predicates and exact spherical
  line / polygon distance are out of scope for now.
- Box exact filtering is XY planar. `GEOGRAPHY` and non-planar edge models
  reject by default; opt in only when treating stored coordinates as planar is
  acceptable for the query.
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
[open_geojson]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geojson.html
[open_geojson_slice]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geojson_slice.html
[open_flatgeobuf]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_flatgeobuf.html
[GeoDataset]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html
[discovery]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.discovery
[select]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.select
[inspect]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.inspect
[validate]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.validate
[scan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.scan
[build]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.build
[convert]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.convert
[convert_into]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.convert_into
[filter_features]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.filter_features
[filter_hits]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.filter_hits
[search_hits]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.search_hits
[read_features]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.read_features
[GeoDiscovery]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDiscovery.html
[GeometryColumnInfo]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeometryColumnInfo.html
[GeometrySelector]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeometrySelector.html
[GeometryColumn]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeometryColumn.html
[InspectRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.InspectRequest.html
[GeometryProfile]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeometryProfile.html
[ValidateRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.ValidateRequest.html
[ValidationReport]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.ValidationReport.html
[ScanRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.ScanRequest.html
[GeometryScan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeometryScan.html
[BuildRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.BuildRequest.html
[GeoIndex]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoIndex.html
[search_features_2d]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex2D.html#method.search_features
[search_features_3d]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex3D.html#method.search_features
[ConvertRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.ConvertRequest.html
[GeoArtifact]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifact.html
[open_geo_index]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geo_index.html
[open_geo_index_async]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geo_index_async.html
[GeoArtifactIndex]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoArtifactIndex.html
[GeoHit]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoHit.html
[GeoQuery2D]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoQuery2D.html
[GeoQuery3D]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoQuery3D.html
[GeoPayload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoPayload.html
[IndexBuildOptions]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.IndexBuildOptions.html
[from_scan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoIndex.html#method.from_scan
[artifact_from_scan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifact.html#method.from_scan
[nearest_features]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex2D.html#method.nearest_features
[raycast_features]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex3D.html#method.raycast_features
[IndexDimsRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.IndexDimsRequest.html
[StoragePrecision]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.StoragePrecision.html
[NullPolicy]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.NullPolicy.html
[EnvelopePolicy]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.EnvelopePolicy.html
[AntimeridianPolicy]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.AntimeridianPolicy.html
[PayloadPlan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.PayloadPlan.html
[PropertyProjection]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.PropertyProjection.html
[FeatureRef]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.FeatureRef.html
[FeatureFilterRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.FeatureFilterRequest.html
[FeatureReadRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.FeatureReadRequest.html
[FeatureRecord]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.FeatureRecord.html
[FeatureRows]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.FeatureRows.html
[GeometryReadMode]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeometryReadMode.html
[FeatureReadOrder]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.FeatureReadOrder.html
[DuplicateFeatureRows]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.DuplicateFeatureRows.html
[decode_feature_ref_payload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.decode_feature_ref_payload.html
[decode_feature_wkb_payload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.decode_feature_wkb_payload.html
[read_geo_manifest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.read_geo_manifest.html
[GeoArtifactManifest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactManifest.html
[AsyncRangeReader]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/trait.AsyncRangeReader.html
