# Geospatial Source Index

[![crates.io](https://img.shields.io/crates/v/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![docs.rs](https://docs.rs/packed_spatial_index_geo/badge.svg)](https://docs.rs/packed_spatial_index_geo)
[![Rust CI](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml/badge.svg)](https://github.com/Filyus/packed_spatial_index/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/packed_spatial_index_geo.svg)](https://crates.io/crates/packed_spatial_index_geo)
[![License](https://img.shields.io/crates/l/packed_spatial_index_geo.svg)](https://github.com/Filyus/packed_spatial_index/blob/main/LICENSE)

Build a [`packed_spatial_index`](https://crates.io/crates/packed_spatial_index)
spatial index for **GeoParquet**, native Apache Parquet `GEOMETRY` /
`GEOGRAPHY` columns, **FlatGeobuf**, and **GeoJSON**. Those formats store
geometry (sometimes with bbox/statistics metadata), but no portable per-row
sidecar that pinpoints individual features and streams from object storage.
This crate fills the gap in three roles:

- **accelerator** — build an in-memory index over the features; queries return
  `FeatureRef` values that preserve source row numbers even when rows are
  skipped or split
- **converter** — attach a leaf-ordered payload (by default `FeatureRef` + WKB
  geometry) and serialize everything into a self-describing, **streamable
  `PSINDEX`** that answers window, polygon, and 3D frustum candidate queries
  straight from object storage in a handful of range reads — no Parquet
  re-read
- **discovery / inspection** — open a `GeoDataset`, list usable geometry
  candidates before selecting one, then inspect the selected column's typed
  metadata (dims, encoding, CRS, edges, extent, row count)

Both index roles support 2D and 3D, optional **`f32`** storage for half-size
files, and **`skip_null`** to drop empty geometry. kNN and raycast run on the
in-memory path; window, polygon, and frustum queries run on both.

The source-side dependencies sit behind format features. Defaults enable
`parquet`, `flatgeobuf`, and `geojson`, so the CLI can build from all supported
inputs. Turn defaults off (`default-features = false`) and the crate is
query-only — open a pre-built `PSINDEX`, stream candidate queries, exact-filter
geometry from the payload — with no `arrow` / `parquet`, small enough for
**wasm / edge**. Build runs server-side once; query runs anywhere.

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{open_geoparquet, BuildRequest, Box2D, GeoIndex};

let mut dataset = open_geoparquet(File::open("cities.parquet")?)?;
let index = dataset.build(BuildRequest::default())?;
let GeoIndex::D2(index) = index else { panic!("expected 2D geometry") };
let refs = index.search_feature_refs(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Installation

Requires Rust 1.89 or newer.

```toml
[dependencies]
packed_spatial_index_geo = "0.22"
```

### Features

- **`parquet`** *(default)* — the Parquet source side: `open_geoparquet`, `GeoDataset`
  (discovery, inspection, validation, feature read-back), `build` / `convert`,
  and the `gp2psindex` CLI. Pulls in `arrow` + `parquet`.
- **`flatgeobuf`** *(default)* — open `.fgb` sources with `open_flatgeobuf`,
  then scan, build, convert, and read features back without Arrow.
- **`geojson`** *(default)* — open GeoJSON with `open_geojson` /
  `open_geojson_slice`, then scan, build, convert, and read features back.
  `build_geojson_stream` / `convert_geojson_stream` provide one-shot
  `FeatureCollection` build and convert paths without retaining the full
  source document.
- **`async`** — open and query streamable `PSINDEX` artifacts over an
  [`AsyncRangeReader`][AsyncRangeReader], adding `open_geo_index_async` and `search_matches_async`.

To query a pre-built `PSINDEX` from a browser or edge worker, drop the default
feature so `arrow` / `parquet` never enter the build:

```toml
[dependencies]
packed_spatial_index_geo = { version = "0.22", default-features = false, features = ["async"] }
```

That leaves the crate query-only — [`open_geo_index`][open_geo_index] /
[`open_geo_index_async`][open_geo_index_async],
[`search_entry_ids`][artifact_search_entry_ids] /
[`search_matches`][search_matches],
[`GeoArtifactIndex2D::filter_matches`][filter_matches] (exact intersection over the
payload geometry), and payload decoding — compiling to `wasm32`. Only reading a
source file needs a format feature.

## API Map

The public API is split into three layers:

| Layer | Primary types | Scope |
| --- | --- | --- |
| Source session | [`GeoDataset`][GeoDataset], [`GeoDiscovery`][GeoDiscovery] | Format-aware open, discovery, scan, build, convert, filter, and row read-back |
| In-memory index | [`GeoIndex`][GeoIndex], [`GeoIndex2D`][GeoIndex2D], [`GeoIndex3D`][GeoIndex3D] | Repeated local range, nearest-neighbor, and raycast queries |
| Streamable artifact | [`GeoArtifactIndex`][GeoArtifactIndex], [`GeoArtifactDirectory`][GeoArtifactDirectory] | Query-only access to converted `PSINDEX` files over range reads |

### Source Sessions

A source session owns the format reader state. It is the API surface for
GeoParquet, native Parquet geospatial columns, FlatGeobuf, and GeoJSON before
or while they are converted into indexes or artifacts.

| Operation | API |
| --- | --- |
| Open source files | [`open_geoparquet`][open_geoparquet], [`open_geojson`][open_geojson], [`open_geojson_slice`][open_geojson_slice], [`open_flatgeobuf`][open_flatgeobuf] |
| Stream one-shot GeoJSON build / convert | [`build_geojson_stream`][build_geojson_stream], [`convert_geojson_stream`][convert_geojson_stream] |
| Discover and select geometry | [`GeoDataset::discovery`][discovery], [`GeoDiscovery`][GeoDiscovery], [`GeoDataset::select`][select], [`GeometrySelector`][GeometrySelector] |
| Inspect or validate metadata | [`GeoDataset::inspect`][inspect], [`InspectRequest`][InspectRequest], [`GeoDataset::validate`][validate] |
| Scan envelopes and payloads | [`GeoDataset::scan`][scan], [`ScanRequest`][ScanRequest] |
| Build an in-memory index | [`GeoDataset::build`][build], [`GeoIndex`][GeoIndex], [`IndexBuildOptions::precision`][IndexBuildOptions] |
| Convert to `PSINDEX` | [`GeoDataset::convert`][convert], [`GeoIndex::from_scan`][from_scan], [`GeoArtifact::from_scan`][artifact_from_scan] |
| Exact-filter source rows | [`GeoDataset::filter_features`][filter_features], [`FeatureFilterRequest`][FeatureFilterRequest] |
| Read source rows back | [`GeoDataset::read_features`][read_features], [`FeatureReadRequest`][FeatureReadRequest], [`FeatureRecord`][FeatureRecord] |

### In-Memory Indexes

An in-memory index keeps the built tree and feature references in process.
This layer covers low-latency repeated queries and algorithms that need local
tree traversal, such as kNN and raycast.

| Operation | API |
| --- | --- |
| Range candidates | [`GeoIndex2D::search_feature_refs`][search_feature_refs_2d], [`GeoIndex3D::search_feature_refs`][search_feature_refs_3d] |
| Planar nearest-neighbor | [`GeoIndex2D::nearest_feature_refs`][nearest_feature_refs] |
| Lon/lat nearest-neighbor | [`GeoIndex2D::nearest_feature_refs_haversine`][nearest_feature_refs_haversine] |
| Raycast candidates | [`GeoIndex3D::raycast_feature_refs`][raycast_feature_refs] |
| Closest raycast hit | [`GeoIndex3D::raycast_closest_feature_ref`][raycast_closest_feature_ref] |

### Streamable Artifacts

A `PSINDEX` artifact is opened independently of the original source file.
`GeoArtifactDirectory` caches parsed open metadata so request handlers can
reattach fresh range readers without rereading the container directory or
`geoM` manifest.

| Operation | API |
| --- | --- |
| Open artifact | [`open_geo_index`][open_geo_index], [`open_geo_index_async`][open_geo_index_async] (`async` feature) |
| Cache and reattach metadata | [`GeoArtifactDirectory`][GeoArtifactDirectory], [`into_directory`][into_directory] / [`from_directory`][from_directory] |
| Entry-id search | [`search_entry_ids`][artifact_search_entry_ids] |
| Count matches without materializing | [`count_entries`][artifact_count_entries] |
| Payload search | [`GeoArtifactIndex2D::search_matches`][search_matches], [`GeoMatch`][GeoMatch] |
| Feature-level payload search | [`GeoArtifactIndex2D::search_features`][artifact_search_features], [`GeoArtifactIndex2D::search_feature_matches`][artifact_search_feature_matches] |
| Paged payload reads | [`search_match_headers`][search_match_headers], [`fetch_matches`][fetch_matches] |
| Exact-filter payload matches | [`GeoArtifactIndex2D::filter_matches`][filter_matches] |
| Query shapes | [`GeoQuery2D`][GeoQuery2D] (box / polygon / radius), [`GeoQuery3D`][GeoQuery3D] (box / frustum) |
| Payload plans | [`PayloadPlan`][PayloadPlan], [`FeatureRef`][FeatureRef] |

Enable the `async` feature to open the same artifacts through an
`AsyncRangeReader`; the async methods mirror window, polygon, and 3D frustum
candidate queries, including payload-returning `search_matches_async`. When
each request needs a fresh range reader — a server or worker — split an opened
artifact with `into_directory`, cache the returned
[`GeoArtifactDirectory`][GeoArtifactDirectory], and let later requests
reattach through `from_directory` without repeating the container, `geoM`
manifest, or stream-directory reads.

### Result Vocabulary

One naming rule across the crate: the method name states what a query returns.

| Method shape | Returns | Granularity |
| --- | --- | --- |
| `*_entry_ids`, `count_entries` | index-entry ids, or just their count | entry-level |
| `*_feature_refs` | [`FeatureRef`][FeatureRef] values | entry-level |
| `*_matches` | [`GeoMatch`][GeoMatch] — ref plus decoded payload | entry-level |
| `*_match_headers` | [`GeoMatchHeader`][GeoMatchHeader] — ref plus payload size, no body | entry-level |
| `*_features`, `*_feature_matches` | deduplicated [`FeatureRef`][FeatureRef] / [`GeoMatch`][GeoMatch] | feature-level |

Entry-level results can repeat a source feature: splitting (for example at the
antimeridian) turns one feature into several index entries, told apart by
`FeatureRef::part`. Feature-level methods collapse those parts into one record
per source feature. A feature ref always carries the source `row_number`;
scans and converted artifacts also fill `row_group` / `row_in_group` so source
rows can be read back efficiently. The core crate calls entry ids "compact
item ids" — same values, lower-layer vocabulary.

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

For the streaming story, `r2_polygon_pruning` counts the range reads and bytes
a polygon query fetches from a simulated remote store, and
`embedded_in_parquet` queries a `PSINDEX` appended to a Parquet file without
reading the Parquet payload.

## Documentation

- **[Guide](docs/guide.md)** — validate before building, convert to a
  streamable `PSINDEX`, query source rows back with exact filtering, spherical
  radius queries.
- **[Memory model](docs/memory-model.md)** — what stays in memory while
  building from GeoParquet, FlatGeobuf, or GeoJSON, and why querying an existing
  `PSINDEX` is the low-memory path.
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

Start with discovery when you don't know which geometry columns the file
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

### Inputs

- GeoParquet files with `geo` metadata or native Apache Parquet `GEOMETRY` /
  `GEOGRAPHY` logical-type columns, FlatGeobuf files, or GeoJSON documents.
  When both GeoParquet metadata and native Parquet geometry columns are
  present, GeoParquet `primary_column` is the default; use
  `GeometrySelector::Name` or the CLI's `--geometry-column` option to select
  explicitly.
- FlatGeobuf and GeoJSON sources expose a single geometry named `geometry`.
  GeoJSON input accepts `FeatureCollection`, single `Feature`, and bare
  geometry documents through eager `open_geojson`. The streaming GeoJSON APIs
  accept `FeatureCollection` only.
- Use `open_geoparquet(...).discovery()` when a file may contain several
  geometry candidates and you want metadata-only selection status before
  reading rows.

### Envelopes and encodings

- Boxes come from the **bbox covering** column when present, otherwise from
  each geometry's **WKB** or **GeoArrow** envelope.
- Native Parquet `GEOMETRY` / `GEOGRAPHY` columns are WKB by definition, so
  they work for envelope scans and `RowWkb` payloads even without GeoParquet
  `geo` metadata.
- GeoParquet GeoArrow encodings `point`, `linestring`, `polygon`,
  `multipoint`, `multilinestring`, and `multipolygon` can be scanned without a
  covering column and can be emitted as ISO WKB payloads.
- Geometry columns may be `Binary`, `LargeBinary`, or `BinaryView`; dimensions
  may be 2D or 3D (`XYZ` / `XYZM`).
- Null / empty geometry: `BuildRequest` defaults to `NullPolicy::Error`;
  `ConvertRequest` defaults to `NullPolicy::Skip`. `FeatureRef::row_number`
  preserves the original source row number.

### Exactness limits

- `GEOGRAPHY` is indexed as a coordinate-axis-aligned bounding box over the
  stored WKB coordinates. This is a candidate index, not exact spherical or
  ellipsoidal predicate evaluation.
- Spherical radius exact filtering is available for spherical geography
  `Point` / `MultiPoint` sources. Ellipsoidal predicates and exact spherical
  line / polygon distance are out of scope for now.
- Box exact filtering is XY planar. `GEOGRAPHY` and non-planar edge models
  reject by default; opt in only when treating stored coordinates as planar is
  acceptable for the query.

## License

Licensed under the [Apache License 2.0](https://github.com/Filyus/packed_spatial_index/blob/main/LICENSE).

<!-- docs.rs API links -->
[open_geoparquet]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geoparquet.html
[open_geojson]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geojson.html
[open_geojson_slice]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geojson_slice.html
[build_geojson_stream]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.build_geojson_stream.html
[convert_geojson_stream]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.convert_geojson_stream.html
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
[filter_matches]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.filter_matches
[search_matches]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.search_matches
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
[GeoIndex2D]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex2D.html
[GeoIndex3D]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex3D.html
[search_feature_refs_2d]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex2D.html#method.search_feature_refs
[search_feature_refs_3d]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex3D.html#method.search_feature_refs
[ConvertRequest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.ConvertRequest.html
[GeoArtifact]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifact.html
[open_geo_index]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geo_index.html
[open_geo_index_async]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.open_geo_index_async.html
[GeoArtifactIndex]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoArtifactIndex.html
[GeoArtifactDirectory]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactDirectory.html
[into_directory]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoArtifactIndex.html#method.into_directory
[from_directory]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoArtifactIndex.html#method.from_directory
[artifact_search_entry_ids]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.search_entry_ids
[artifact_count_entries]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.count_entries
[GeoMatchHeader]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoMatchHeader.html
[GeoMatch]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoMatch.html
[artifact_search_features]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.search_features
[artifact_search_feature_matches]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.search_feature_matches
[search_match_headers]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.search_match_headers
[fetch_matches]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactIndex2D.html#method.fetch_matches
[GeoQuery2D]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoQuery2D.html
[GeoQuery3D]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoQuery3D.html
[GeoPayload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoPayload.html
[IndexBuildOptions]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.IndexBuildOptions.html
[from_scan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoIndex.html#method.from_scan
[artifact_from_scan]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifact.html#method.from_scan
[nearest_feature_refs]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex2D.html#method.nearest_feature_refs
[nearest_feature_refs_haversine]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex2D.html#method.nearest_feature_refs_haversine
[raycast_feature_refs]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex3D.html#method.raycast_feature_refs
[raycast_closest_feature_ref]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoIndex3D.html#method.raycast_closest_feature_ref
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
