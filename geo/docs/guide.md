# Guide

Recipes for the workflows this crate supports. For the two-mode overview
(accelerator vs. converter) and how this crate relates to
[`oxigdal-geoparquet`](https://crates.io/crates/oxigdal-geoparquet), see
[When to use it](when-to-use.md).

## Validate inputs before building

Use [`GeoDataset::validate`][validate] when an input file comes from an
uncontrolled pipeline and you want a structured compatibility report before
building or converting:

```rust
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

## Convert to a streamable PSINDEX

```rust
use std::fs::File;
use packed_spatial_index_geo::{open, ConvertRequest};

let mut dataset = open(File::open("cities.parquet")?)?;
let psindex: Vec<u8> = dataset.convert(ConvertRequest::default())?;
std::fs::write("cities.psindex", &psindex)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Serve `cities.psindex` over HTTP range requests (or read it locally) and query
it through the geo artifact reader:

```rust
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
against the payload geometry). `ConvertRequest` skips null or empty geometries
by default; `BuildRequest` errors by default.

Payload modes:

- `PayloadPlan::RowWkb` (default): fixed `FeatureRef` record followed by WKB.
- `PayloadPlan::RowRef`: fixed-width `FeatureRef` only; smallest sidecar mode.
- `PayloadPlan::FeatureJson`: GeoJSON Feature bytes with projected properties.
- `PayloadPlan::None`: no payload section.

Converted `PSINDEX` files also carry an app-private `geoM` manifest chunk. Core
`packed_spatial_index` readers skip it; this crate reads it through
`open_geo_index` or, when only metadata is needed, `read_geo_manifest`.

## Build an index and a converted artifact together

`GeoDataset::build` and `GeoDataset::convert_into` each scan the source once
internally, so calling both on the same `GeoDataset` scans it twice. Scan
once and build both outputs from the result instead:

```rust
use std::fs::File;
use packed_spatial_index_geo::{
    open, ConvertRequest, GeoArtifact, GeoIndex, IndexBuildOptions, PayloadPlan, ScanRequest,
};

let mut dataset = open(File::open("cities.parquet")?)?;
let scan = dataset.scan(ScanRequest {
    payload: PayloadPlan::RowWkb,
    ..ScanRequest::default()
})?;

let index = GeoIndex::from_scan(&scan, &IndexBuildOptions::default())?;

let mut bytes = Vec::new();
let artifact = GeoArtifact::from_scan(
    &scan,
    &ConvertRequest::default(),
    dataset.source_fingerprint(),
    &mut bytes,
)?;
std::fs::write("cities.psindex", &bytes)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Both functions borrow the scan rather than consume it, and only the scan
itself reads the source. `GeoIndex::from_scan` never looks at the scan's
payload bytes, so pick the `PayloadPlan` the artifact needs; the index comes
out the same either way.

## Query source rows

Use [`GeoDataset::read_features`][read_features] when a `PSINDEX` sidecar
stores only row refs, or when you want attributes from the original Parquet
file after an index query:

```rust
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
the final read-back step can work with true hits instead of bbox candidates.
It reads geometry WKB internally; open a fresh dataset session for
`read_features` after filtering.

The query is not limited to a rectangle. Pass `GeoQuery2D::polygon` or
`GeoQuery2D::multi_polygon` (the `geo_types` crate is re-exported) to query an
arbitrary planar polygon: index search still narrows candidates by the
polygon's bounding box; the exact step then drops the bbox false-positives
that fall in holes or concavities.

**When to filter exactly** — a non-rectangular query leaves bbox
false-positives (the index narrows only by bounding box); the exact step
removes them:

- **Filter** when you need the exact shape; without it the result is the bbox
  superset (everything in the bounding box).
- **Use `filter_hits`, not `filter_features`, for speed.**
  `GeoArtifactIndex2D::filter_hits` tests the geometry that `search_hits`
  already fetched, so it adds no source re-read. Measured (~100k points,
  `examples/end_to_end_box_vs_polygon.rs`) it beats reading all candidates
  above ~60% rejection (93% × 40 columns ≈ 1.3×). `filter_features` re-reads
  every candidate's geometry from the source, so it loses to read-all in
  every case — use it only without a converted artifact.
- **Skip** when a bbox superset is acceptable (point data, where the bbox *is*
  the geometry) or rejection is low (below ~50%, where reading all candidates
  is faster anyway).

If candidate filtering is enough, skip the exact step and read the hit refs
directly:

```rust
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
let rows = source.read_features(FeatureReadRequest {
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

## Spherical radius queries

For `GEOGRAPHY(SPHERICAL)` / GeoParquet spherical edges, the CLI's
`query --radius` performs a lon/lat radius lookup: it first searches the 2D
artifact with one or two candidate boxes (splitting at the antimeridian when
needed), then applies exact spherical distance filtering before reading
projected rows. This release supports `Point` and `MultiPoint` geometries;
lines and polygons return a clear unsupported-geometry error.

```text
gp2psindex query input.parquet output.psi \
  --radius -73.9857,40.7484,500 \
  --properties include:name \
  --json
```

The API path uses the same request type as planar exact filtering:

```rust
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

[validate]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.validate
[read_features]: https://docs.rs/packed_spatial_index_geo/latest/packed_spatial_index_geo/struct.GeoDataset.html#method.read_features
