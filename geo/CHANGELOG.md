# Changelog

All notable changes to `packed_spatial_index_geo` are documented here.

## [Unreleased]

### API

- Added `GeoIndex::from_scan`, `GeoArtifact::from_scan`, and
  `GeoDataset::source_fingerprint`. `GeoDataset::build` and
  `GeoDataset::convert_into` each call `GeoDataset::scan` internally, so
  getting both an in-memory index and a converted artifact from one
  `GeoDataset` used to scan the source twice. Scan once and pass the result
  to both functions instead.

### Indexes

- Added `GeoIndex2DF32`/`GeoIndex3DF32`, f32-precision in-memory accelerator
  indexes, and `IndexBuildOptions::precision` (default `StoragePrecision::F64`,
  no behavior change for existing callers) to select them via
  `GeoDataset::build` or `GeoIndex::from_scan`. Half the box memory of
  `GeoIndex2D`/`GeoIndex3D`; supports `Box2D`/`Box3D` queries only —
  `GeoQuery2D::Polygon` and `GeoQuery2D::SphericalRadius` are rejected, since
  the underlying `Index2DF32`/`Index3DF32::search` take a plain box, not the
  generic query trait those variants need (a core-level ceiling, not planned
  to be lifted). `Index2DF32`/`Index3DF32` are now re-exported from this
  crate's root.

### Search

- `gp2psindex query` now accepts `--bbox` with six comma-separated numbers
  (`xmin,ymin,zmin,xmax,ymax,zmax`) against a 3D `.psi` index, calling the
  already-existing `GeoArtifactIndex3D::search_features`. `--radius`,
  `--exact`, and `--predicate` are 2D-only and are now rejected for a 3D
  artifact with an explanatory error (a `Box3D` query against a box index has
  no bounding-box false positives for `--exact` to filter), instead of the
  previous blanket "query CLI currently accepts only 2D" rejection.

## [0.14.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.14.0...psi-geo-v0.14.1) - 2026-07-01

### Documentation

- Split the recipe and decision-guide content out of `README.md` into
  `docs/guide.md` (validate before building, convert to a streamable
  `PSINDEX`, query source rows with exact filtering, spherical radius
  queries) and `docs/when-to-use.md` (accelerator vs. converter, how this
  crate differs from `oxigdal-geoparquet`), mirroring the core crate's
  `docs/guide.md` / `docs/when-to-use.md` split. `README.md` is now a landing
  page with a `## Documentation` section linking out, rather than one
  550-line file.
- Added missing rustdoc examples for `GeoQuery3D`, `GeoIndex3D::search_features`,
  and `GeoArtifactIndex2D`/`GeoArtifactIndex3D::search_hits`.
- Corrected an "the index is tiny" overclaim in `docs/when-to-use.md`: measured
  (100k simple points), even a payload-free index is ~95% the size of the
  source Parquet, since a per-row index scales with row count, not geometry
  size.

## [0.14.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.13.0...psi-geo-v0.14.0) - 2026-06-30

### Search

- A `GeoQuery2D::Polygon` query passed to `GeoArtifactIndex2D::search_hits` now
  prunes subtrees that fall outside the polygon during the streamed descent (via
  the core's new streaming region queries), so it fetches only the leaves the
  polygon overlaps — less data than its bounding box (e.g. ~50–80% fewer bytes at
  high rejection), the win for polygon queries over a remote artifact. For point
  data the result is already the exact in-polygon set; `filter_hits` remains the
  exact step for line / polygon geometries.

## [0.13.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.12.0...psi-geo-v0.13.0) - 2026-06-30

### Search

- Added arbitrary polygon / multipolygon queries through `GeoQuery2D::polygon`,
  `GeoQuery2D::multi_polygon`, and `From` conversions. Index search narrows
  candidates by the query's bounding box; exact `filter_features` then keeps only
  geometries that truly intersect the polygon, removing the bbox false-positives
  over holes and concavities. `geo_types` is re-exported for building queries.
- Breaking: `GeoQuery2D` is no longer `Copy` (it can carry a polygon); it stays
  `Clone`.
- Added `GeoArtifactIndex2D::filter_hits` to exact-filter `search_hits` results by
  the geometry already in their payloads (`RowWkb` or `FeatureJson`), with no
  source re-read. Unlike
  `filter_features` (which re-reads candidate geometry and so never beats reading
  all candidates), `filter_hits` reuses the geometry the index produced, so it
  wins above roughly 60% rejection.

### Performance

- `GeoIndex2D::search_features` and `GeoArtifactIndex2D::search_hits` now
  deduplicate candidates in O(K) rather than O(K²), so queries returning many
  candidates no longer spend quadratic time in the index (the artifact
  `search_features` wrapper inherits the fix). A box query returning 100k
  candidates drops from roughly 2 s to 3 ms.

## [0.12.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.11.0...psi-geo-v0.12.0) - 2026-06-29

### Search

- Added `GeoQuery2D` and `GeoQuery3D` query values for geo candidate and
  exact-filter APIs.
- Breaking: replaced `QueryGeometry` with `GeoQuery2D`.
- Breaking: replaced shape-specific exact-filter constructors such as
  `intersects_box2d`, `from_hits_intersects_box2d`, and
  `intersects_spherical_radius` with `FeatureFilterRequest::intersects` and
  `FeatureFilterRequest::intersects_from_hits`.
- Breaking: in-memory `GeoIndex2D::search_features` and
  `GeoIndex3D::search_features` now return `Result<Vec<FeatureRef>, GeoError>`,
  matching artifact search and allowing query validation errors to surface.

## [0.11.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.10.0...psi-geo-v0.11.0) - 2026-06-28

### Search

- Updated the public `packed_spatial_index` dependency to 0.19, keeping the geo
  crate aligned with the core overlap-query API and iterator type changes.

## [0.10.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.9.0...psi-geo-v0.10.0) - 2026-06-28

### Search

- Added spherical point-radius exact filtering for spherical geography
  `Point` / `MultiPoint` data through `QueryGeometry::SphericalRadius`,
  `FeatureFilterRequest::intersects_spherical_radius`, and
  `gp2psindex query --radius`.

## [0.9.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.8.0...psi-geo-v0.9.0) - 2026-06-28

### Search

- Added exact planar post-filtering with `GeoDataset::filter_features`,
  `FeatureFilterRequest`, and `gp2psindex query --exact`, so bbox candidates can
  be reduced against source geometries before reading final rows.

## [0.8.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.7.1...psi-geo-v0.8.0) - 2026-06-28

### Persistence

- Added source read-back from `FeatureRef` values through
  `GeoDataset::read_features`, including projected properties, optional WKB
  geometry, source fingerprint checks, and request-order / duplicate handling.
- Added `gp2psindex query` to query a `PSINDEX` sidecar and emit projected source
  rows as JSON / NDJSON.
- `FeatureRef` values produced by scan/build/convert now include row-group and
  row-in-group positions when available from Parquet metadata.

## [0.7.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.7.0...psi-geo-v0.7.1) - 2026-06-28

### Documentation

- Clarified the crate's role compared with `oxigdal-geoparquet` and tightened
  README command/table formatting for crates.io.

## [0.7.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.6.2...psi-geo-v0.7.0) - 2026-06-28

### Validation

- Added a structured validation API (`GeoDataset::validate`,
  `ValidateRequest`, `ValidationReport`) for compatibility diagnostics before
  building or converting geospatial Parquet inputs.
- Added native Parquet `GEOMETRY` / `GEOGRAPHY` row-group geospatial statistics
  diagnostics to validation reports.
- Added a richer `gp2psindex validate` command with JSON output, exact row-scan
  validation, strict warning handling, payload checks, and antimeridian options.

## [0.6.2](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.6.1...psi-geo-v0.6.2) - 2026-06-28

### Documentation

- Added compile-checked rustdoc examples directly on `GeoDataset` and its main
  workflow methods.
- Added compile-checked rustdoc examples for the main request, selector,
  payload, index, feature reference, and artifact manifest types.

## [0.6.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.6.0...psi-geo-v0.6.1) - 2026-06-28

### Documentation

- Added runnable examples for discovery, in-memory index building, artifact
  conversion/querying, and `FeatureJson` payloads.
- Added rustdoc coverage for the public session, artifact reader, request, and
  metadata types, with a missing-docs lint to keep future public API documented.

## [0.6.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.5.1...psi-geo-v0.6.0) - 2026-06-28

### Persistence

- Added a geo artifact reader API (`open_geo_index`, `GeoArtifactIndex`,
  `GeoHit`, `GeoPayload`) for querying converted `PSINDEX` files through the
  geospatial contract instead of manually decoding core payload bytes.
- Extended generated `geoM` manifests with index storage precision so readers
  can open 2D/3D and f64/f32 artifacts from the manifest alone.
- `FeatureJson` payloads now include a `feature_ref` member, allowing artifact
  queries to return the source `FeatureRef` alongside the GeoJSON Feature.

## [0.5.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.5.0...psi-geo-v0.5.1) - 2026-06-28

### Documentation

- Refined the crate description, README heading, and README opening copy so the
  crates.io landing page explains the GeoParquet/native Parquet indexing use
  case more cleanly.
- Added a README API-at-a-glance table for the `open(...) -> GeoDataset` session
  workflow and related request, payload, and artifact helpers.

## [0.5.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.4.1...psi-geo-v0.5.0) - 2026-06-28

### API

- Replaced the function-oriented public API with the `open(...) -> GeoDataset`
  session API. Discovery, inspection, scanning, building, and conversion now hang
  off the dataset, and geo-level search returns `FeatureRef` values rather than
  raw compact item ids.
- Made the CLI explicit-subcommand only: `discover`, `inspect`, `build`, and
  `validate`.
- Typed geometry discovery/profile metadata, GeoArrow envelope scanning without
  covering columns, GeoArrow-to-WKB payload emission, antimeridian split handling,
  `FeatureJson` payloads with projected properties, and the optional `geoM`
  manifest chunk in generated `PSINDEX` artifacts.

## [0.4.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.4.0...psi-geo-v0.4.1) - 2026-06-27

### Persistence

- Updated the Arrow / Parquet reader stack to `59` and parse the GeoParquet
  `geo` metadata directly, avoiding a stale `parquet` dependency in the
  companion reader.

## [0.4.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.3.1...psi-geo-v0.4.0) - 2026-06-27

### Discovery

- Metadata-only geometry discovery API (`discover`, `discover_with_opts`) that
  reports GeoParquet/native Parquet geospatial candidates, default selection
  status, and per-column index/payload capabilities.
- `gp2psindex inspect`, including `--geometry-column` and `--json` output for
  the discovery result.

## [0.3.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.3.0...psi-geo-v0.3.1) - 2026-06-27

### Documentation

- Refined README and rustdoc wording to describe GeoParquet and native Parquet
  geospatial inputs consistently.

## [0.3.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.2.0...psi-geo-v0.3.0) - 2026-06-27

### Geometry

- Native Apache Parquet `GEOMETRY` / `GEOGRAPHY` logical-type support, including
  files that have no GeoParquet `geo` metadata.
- Explicit geometry-column selection for readers, builders, converter options,
  and `gp2psindex --geometry-column`.
- `GeometryMetadataSource` on `GeoParquetInfo` to distinguish GeoParquet metadata
  from native Parquet geospatial logical types.
- `GEOGRAPHY` inputs are indexed as coordinate bounding boxes over their WKB
  coordinates; exact spherical or ellipsoidal predicates remain the caller's
  responsibility after candidate lookup.

## [0.2.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.1.0...psi-geo-v0.2.0) - 2026-06-27

### Persistence

- `ConvertPayload` payload modes for the converter: no payload, row-id-only
  sidecar payload, or original row id + WKB.
- Decode helpers and content-type constants for Geo converter payloads.
- `gp2psindex --payload none|row-id|row-wkb`.
- The default converter payload now stores `u64le original_row_id` followed by
  WKB, so outputs created with `skip_null` can still point back to source
  GeoParquet rows.
- Native GeoParquet with a covering column can be converted with
  `ConvertPayload::RowIds`, because that mode does not require geometry decoding.

## [0.1.0] - 2026-06-20

Initial release: build a [`packed_spatial_index`](https://crates.io/crates/packed_spatial_index)
spatial index from a GeoParquet file.

### Geometry

- **Primitive / introspection** — `read_bboxes_2d` / `read_bboxes_3d`,
  `inspect` + `GeoParquetInfo`, `detect_dims`.
- Boxes from the GeoParquet 1.1 bbox covering column when present, otherwise from
  the WKB envelope; `Binary` / `LargeBinary` / `BinaryView` geometry columns; 2D
  and 3D; optional `f32` storage; `skip_null`; interleaved payload.

### Indexes

- **Accelerator** — `build_index_2d` / `build_index_3d` build an in-memory index
  over the row bounding boxes; item id equals the GeoParquet row index.

### Persistence

- **Converter** — `convert_2d` / `convert_3d` (and the buffer-reusing `_into`
  variants) build the index, attach each row's WKB geometry as a leaf-ordered
  payload, and record the CRS, serialized to a streamable `PSINDEX` blob.
- **`gp2psindex` CLI** for the file-to-file path.
