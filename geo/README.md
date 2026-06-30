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
let features = index.search_features(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Installation

Requires Rust 1.89 or newer.

```toml
[dependencies]
packed_spatial_index_geo = "0.14"
```

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
| Discover columns | [`GeoDataset::discovery`][discovery], [`GeoDiscovery`][GeoDiscovery] |
| Select a column | [`GeoDataset::select`][select], [`GeometrySelector`][GeometrySelector] |
| Inspect metadata | [`GeoDataset::inspect`][inspect], [`InspectRequest`][InspectRequest] |
| Validate input | [`GeoDataset::validate`][validate] |
| Scan boxes / payloads | [`GeoDataset::scan`][scan], [`ScanRequest`][ScanRequest] |
| Build an index | [`GeoDataset::build`][build], [`GeoIndex`][GeoIndex] |
| Query the index | [`GeoIndex2D::search_features`][search_features_2d], [`GeoIndex3D::search_features`][search_features_3d] |
| Convert to `PSINDEX` | [`GeoDataset::convert`][convert] |
| Open a `PSINDEX` | [`open_geo_index`][open_geo_index] |
| Query a `PSINDEX` | [`GeoArtifactIndex2D::search_hits`][search_hits], [`GeoHit`][GeoHit] |
| Choose a query shape | [`GeoQuery2D`][GeoQuery2D] (box / polygon / radius) |
| Exact-filter source hits | [`GeoDataset::filter_features`][filter_features], [`FeatureFilterRequest`][FeatureFilterRequest] |
| Exact-filter `PSINDEX` hits | [`GeoArtifactIndex2D::filter_hits`][filter_hits] |
| Read source rows | [`GeoDataset::read_features`][read_features], [`FeatureReadRequest`][FeatureReadRequest] |
| Tune requests | [`IndexDimsRequest`][IndexDimsRequest], [`NullPolicy`][NullPolicy] |
| Pick payloads | [`PayloadPlan`][PayloadPlan], [`FeatureRef`][FeatureRef] |
| Use the CLI | See [CLI](#cli) |

The in-memory `GeoIndex` search methods return [`FeatureRef`][FeatureRef]
values. A feature ref always carries the source `row_number`; scans and converted
artifacts also fill `row_group` / `row_in_group` so source rows can be read back
efficiently. `part` is set when one source row becomes multiple index entries,
for example after antimeridian splitting.

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

## Validate inputs before building

Use [`GeoDataset::validate`][validate] when an input file comes from an
uncontrolled pipeline and you want a structured compatibility report before
building or converting:

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{open, ValidateRequest, ValidationSeverity};

let mut dataset = open(File::open("cities.parquet")?)?;
let report = dataset.validate(ValidateRequest::default())?;

for issue in &report.issues {
    if issue.severity == ValidationSeverity::Warning {
        eprintln!("warning: {}", issue.message);
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Validation is metadata-only by default. Set `ValidateRequest { exact: true, .. }`
to scan rows and report malformed WKB, null-policy failures, antimeridian
rejects, dimension mismatches, or payload projection failures as structured
issues. Native Parquet geospatial row-group statistics are reported as
diagnostics; they are not used as per-row index bounds.

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
GeoParquet / Parquet geo
  -> packed_spatial_index_geo
  -> index / PSINDEX
  -> core queries
```

The two phases need not share a runtime: convert on a server, query from the edge.

## How it differs from `oxigdal-geoparquet`

[`oxigdal-geoparquet`](https://crates.io/crates/oxigdal-geoparquet) is a
GeoParquet driver: it reads and writes GeoParquet, exposes metadata, and can
push bbox / attribute filters into a read. It is the right layer when your next
step is to open a Parquet file and read matching rows or row groups.

This crate is an index layer. It builds a per-feature spatial index from
GeoParquet or native Parquet geospatial columns, then keeps that index in memory
or writes it as a reusable `PSINDEX` sidecar / artifact. That matters when the
source Parquet is large, remote, or slow to scan repeatedly: object storage,
network filesystems, cold HDD archives, and lakehouse datasets all benefit from
answering the spatial lookup from a compact index first.

Use `oxigdal-geoparquet` when you need a GeoParquet reader or writer. Use this
crate when the file already exists and repeated bbox, kNN, or raycast lookups
should avoid scanning geometry rows again. The two can be used together: this
crate can identify candidate feature rows, and a Parquet reader can fetch the
attributes or full geometries from the source file.

## Converter: streamable index

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{open, ConvertRequest};

let mut dataset = open(File::open("cities.parquet")?)?;
let psindex: Vec<u8> = dataset.convert(ConvertRequest::default())?;
std::fs::write("cities.psindex", &psindex)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Serve `cities.psindex` over HTTP range requests (or read it locally) and query it
through the geo artifact reader:

```rust,no_run
use packed_spatial_index_geo::{
    open_geo_index, Box2D, GeoArtifactIndex, GeoPayload, SliceReader,
};

let bytes = std::fs::read("cities.psindex")?;
let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    panic!("expected a 2D artifact");
};

for hit in index.search_hits(Box2D::new(-10.0, 35.0, 20.0, 60.0))? {
    if let GeoPayload::RowWkb(wkb) = hit.payload {
        println!("row {}: {} WKB bytes", hit.feature.row_number, wkb.len());
    }
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

Converted `PSINDEX` files also carry an app-private `geoM` manifest chunk. Core
`packed_spatial_index` readers skip it; this crate reads it through
`open_geo_index` or, when only metadata is needed, `read_geo_manifest`.

## Query source rows

Use [`GeoDataset::read_features`][read_features] when a `PSINDEX` sidecar stores
only row refs, or when you want attributes from the original Parquet file after
an index query:

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{
    open, open_geo_index, Box2D, FeatureFilterRequest, FeatureReadRequest,
    GeoArtifactIndex, GeometryReadMode, PropertyProjection, SliceReader,
};

let bytes = std::fs::read("cities.psindex")?;
let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    panic!("expected a 2D artifact");
};
let manifest = index.manifest().clone();
let hits = index.search_hits(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;

let selector = packed_spatial_index_geo::GeometrySelector::Name(
    manifest.selected_column,
);
let expected_source_fingerprint = Some(manifest.source_fingerprint);
let bbox = Box2D::new(-10.0, 35.0, 20.0, 60.0);
let mut filter_source = open(File::open("cities.parquet")?)?;
let filtered = filter_source.filter_features(FeatureFilterRequest {
    selector: selector.clone(),
    expected_source_fingerprint: expected_source_fingerprint.clone(),
    ..FeatureFilterRequest::intersects_from_hits(hits, bbox)
})?;

let mut row_source = open(File::open("cities.parquet")?)?;
let rows = row_source.read_features(FeatureReadRequest {
    selector,
    expected_source_fingerprint,
    properties: PropertyProjection::Include(vec!["name".to_string()]),
    geometry: GeometryReadMode::Wkb,
    ..FeatureReadRequest::from_features(filtered)
})?;

println!("{} rows", rows.batch.num_rows());
# Ok::<(), Box<dyn std::error::Error>>(())
```

`filter_features` applies exact planar predicates to the source geometries, so
the final read-back step can work with true hits instead of bbox candidates. It
reads geometry WKB internally; open a fresh dataset session for `read_features`
after filtering.

The query is not limited to a rectangle. Pass `GeoQuery2D::polygon` or
`GeoQuery2D::multi_polygon` (the `geo_types` crate is re-exported) to query an
arbitrary planar polygon: index search still narrows candidates by the polygon's
bounding box; the exact step then drops the bbox false-positives that fall in
holes or concavities.

**When to filter exactly** — a non-rectangular query leaves bbox false-positives
(the index narrows only by bounding box); the exact step removes them:

- **Filter** when you need the exact shape; without it the result is the bbox
  superset (everything in the bounding box).
- **Use `filter_hits`, not `filter_features`, for speed.**
  `GeoArtifactIndex2D::filter_hits` tests the geometry that
  `search_hits` already fetched, so it adds no source re-read. Measured
  (~100k points, `examples/end_to_end_box_vs_polygon.rs`) it beats reading all
  candidates above ~60% rejection (93% × 40 columns ≈ 1.3×). `filter_features`
  re-reads every candidate's geometry from the source, so it loses to read-all in
  every case — use it only without a converted artifact.
- **Skip** when a bbox superset is acceptable (point data, where the bbox *is*
  the geometry) or rejection is low (below ~50%, where reading all candidates is
  faster anyway).

If candidate filtering is enough, skip the exact step and read the hit refs
directly:

```rust,no_run
# use std::fs::File;
# use packed_spatial_index_geo::{
#     open, Box2D, FeatureReadRequest, GeoArtifactIndex, GeometryReadMode,
#     PropertyProjection, SliceReader, open_geo_index,
# };
# let bytes = std::fs::read("cities.psindex")?;
# let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
#     panic!("expected a 2D artifact");
# };
# let manifest = index.manifest().clone();
# let hits = index.search_hits(Box2D::new(-10.0, 35.0, 20.0, 60.0))?;
# let mut source = open(File::open("cities.parquet")?)?;
# let rows = source.read_features(FeatureReadRequest {
    selector: packed_spatial_index_geo::GeometrySelector::Name(
        manifest.selected_column,
    ),
    expected_source_fingerprint: Some(manifest.source_fingerprint),
    properties: PropertyProjection::Include(vec!["name".to_string()]),
    geometry: GeometryReadMode::Wkb,
    ..FeatureReadRequest::from_hits(hits)
})?;

println!("{} rows", rows.batch.num_rows());
# Ok::<(), Box<dyn std::error::Error>>(())
```

This reads selected Parquet row groups and projected columns. It is not a
single-row byte seek into Parquet.

## CLI

Install the command with Cargo:

```text
cargo install packed_spatial_index_geo --locked
```

Run it directly after install, or prefix the same arguments with
`cargo run --bin gp2psindex --` from a repository checkout.

Start with discovery when you do not know which geometry columns the file
contains:

```text
gp2psindex discover input.parquet --json
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
```

Query a sidecar and read projected source properties back as NDJSON:

```text
gp2psindex query input.parquet output.psi \
  --bbox -10,35,20,60 \
  --properties include:name,pop
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

## Spherical radius queries

For `GEOGRAPHY(SPHERICAL)` / GeoParquet spherical edges, `query --radius`
performs a lon/lat radius lookup. The command first searches the 2D artifact
with one or two candidate boxes (splitting at the antimeridian when needed),
then applies exact spherical distance filtering before reading projected rows.
This release supports `Point` and `MultiPoint` geometries; lines and polygons
return a clear unsupported-geometry error.

```text
gp2psindex query input.parquet output.psi \
  --radius -73.9857,40.7484,500 \
  --properties include:name \
  --json
```

The API path uses the same request type as planar exact filtering:

```rust,no_run
use std::fs::File;
use packed_spatial_index_geo::{
    open, open_geo_index, FeatureFilterRequest, FeatureReadRequest,
    GeoArtifactIndex, PropertyProjection, SliceReader,
};

let bytes = std::fs::read("places.psi")?;
let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes))? else {
    panic!("expected a 2D artifact");
};

let query = packed_spatial_index_geo::GeoQuery2D::spherical_radius(
    -73.9857, 40.7484, 500.0,
);
let hits = index.search_hits(query.clone())?;

let mut filter_source = open(File::open("places.parquet")?)?;
let exact = filter_source.filter_features(
    FeatureFilterRequest::intersects_from_hits(hits, query),
)?;

let mut read_source = open(File::open("places.parquet")?)?;
let rows = read_source.read_features(FeatureReadRequest {
    properties: PropertyProjection::Include(vec!["name".to_string()]),
    ..FeatureReadRequest::from_features(exact)
})?;

println!("{} rows", rows.batch.num_rows());
# Ok::<(), Box<dyn std::error::Error>>(())
```

If the artifact should carry GeoJSON Feature payloads, name the properties you
want to keep:

```text
gp2psindex validate input.parquet \
  --exact \
  --strict \
  --payload feature-json \
  --properties include:name,pop
gp2psindex build input.parquet output.psi \
  --payload feature-json \
  --properties include:name,pop
```

## Scope

- Inputs may be GeoParquet files with `geo` metadata or native Apache Parquet
  `GEOMETRY` / `GEOGRAPHY` logical-type columns. When both are present,
  GeoParquet `primary_column` is the default; use `GeometrySelector::Name` or
  the CLI's `--geometry-column` option to select explicitly.
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
[GeoArtifactIndex]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoArtifactIndex.html
[GeoHit]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoHit.html
[GeoQuery2D]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoQuery2D.html
[GeoPayload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeoPayload.html
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
[FeatureRows]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.FeatureRows.html
[GeometryReadMode]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.GeometryReadMode.html
[FeatureReadOrder]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.FeatureReadOrder.html
[DuplicateFeatureRows]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/enum.DuplicateFeatureRows.html
[decode_feature_ref_payload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.decode_feature_ref_payload.html
[decode_feature_wkb_payload]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.decode_feature_wkb_payload.html
[read_geo_manifest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/fn.read_geo_manifest.html
[GeoArtifactManifest]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoArtifactManifest.html
