# Changelog

All notable changes to `packed_spatial_index_geo` are documented here.

## [Unreleased]

### API

- Added `GeoMatchHeader::body_byte_len`, the length of a match header's payload
  body after its fixed feature-ref record. Exposes the `RowWkb` WKB byte length
  without callers re-deriving `payload_len - FEATURE_REF_RECORD_LEN`.

## [0.21.2](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.21.1...psi-geo-v0.21.2) - 2026-07-07

### API

- Re-exported `StreamLimits` from `packed_spatial_index_geo`, so callers using
  `open_geo_index_with_limits` or `from_directory_with_limits` can stay on the
  geo crate API surface.

### Documentation

- Added compile-checked rustdoc examples for artifact directory reattach,
  entry/feature/match searches, async artifact searches, paged match-header
  fetches, f32 in-memory queries, and one-shot streaming GeoJSON helpers.

## [0.21.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.21.0...psi-geo-v0.21.1) - 2026-07-06

### Documentation

- Added compile-checked rustdoc examples for counting artifact entries and
  paging match-header reads before fetching payload bodies.

## [0.21.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.20.0...psi-geo-v0.21.0) - 2026-07-06

### API

- Breaking: unified query-result naming so method names state what they
  return — `*_items` for index-entry ids, `*_feature_refs` for `FeatureRef`
  values, `*_matches` for payload-carrying records. Renamed `GeoHit` to
  `GeoMatch`; `search_hits` / `search_hits_async` to `search_matches` /
  `search_matches_async`; `filter_hits` to `filter_matches`;
  `FeatureReadRequest::from_hits` to `from_matches` and `from_features` to
  `from_feature_refs`; `FeatureFilterRequest::intersects_from_hits` to
  `intersects_from_matches`; `search_features` / `search_features_async` to
  `search_feature_refs` / `search_feature_refs_async`; `nearest_features`
  (and `_haversine`) to `nearest_feature_refs`; `raycast_features` to
  `raycast_feature_refs`; `raycast_closest_feature` to
  `raycast_closest_feature_ref`. The `feature refs` names make explicit that
  results are entry-level: a split source feature can contribute several
  entries, distinguished by `FeatureRef::part`. The entry side follows the
  same rule: `search_items` / `search_items_async` are now `search_entry_ids`
  / `search_entry_ids_async`, `GeoMatch::item` is `GeoMatch::entry_id`, and
  `GeoArtifactDirectory::num_items` is `num_entries` — "item" remains the
  core crate's word for the same ids. `gp2psindex --order` now accepts
  `match` alongside the older `hit` value.
- Added `count_entries` on the 2D/3D artifact indexes: counts matching index
  entries through the streaming visitor without materializing ids or
  payloads. Multi-candidate-box queries (for example antimeridian-crossing
  boxes) fall back to deduplicated id search; there is no async variant.
- Added paged match access for `RowRef` / `RowWkb` artifacts:
  `search_match_headers` returns per-entry identity and payload size without
  reading payload bodies, and `fetch_matches` materializes full `GeoMatch`
  values for a page of headers (`RowRef` pages rebuild with no I/O). The new
  `GeoMatchHeader` type supports the same sort/dedupe as `GeoMatch`.
  `FeatureJson` artifacts keep the full-decode path — their identity lives
  inside the JSON body — and `PayloadPlan::None` artifacts have no headers;
  both are rejected with `UnsupportedArtifact`.
- Added feature-level query results to the artifact indexes:
  `search_features` / `search_feature_matches` (+ `_async`) return one record
  per source feature, collapsing split index entries — the lowest-part entry
  survives as the representative and its `part` becomes `None`. Added
  `GeoMatch::sort_by_entry` / `GeoMatch::dedupe_by_feature` and
  `FeatureRef::same_feature` / `cmp_feature` / `cmp_entry` so callers can
  compose the same sort/dedupe with their own filtering (for example an exact
  geometry filter between search and dedupe).

### Persistence

- Fixed split part numbers missing from artifact payloads: scan encoded each
  payload before envelope splitting duplicated entries, so decoded
  `FeatureRef::part` was always `None` for split (for example antimeridian)
  entries. Duplicated payloads are now re-stamped with their part number for
  `RowRef`, `RowWkb`, and `FeatureJson` plans.

## [0.20.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.19.2...psi-geo-v0.20.0) - 2026-07-05

### API

- Added `GeoArtifactDirectory` plus `into_directory` / `from_directory` helpers
  on geo artifact indexes so servers and workers can cache parsed artifact
  metadata and reattach fresh readers without repeating open-time range reads.

## [0.19.2](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.19.1...psi-geo-v0.19.2) - 2026-07-05

### Documentation

- Moved the memory model into a dedicated guide page covering index building,
  artifact querying, GeoJSON streaming, and read-back geometry materialization.

## [0.19.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.19.0...psi-geo-v0.19.1) - 2026-07-05

### Documentation

- Updated the GeoJSON memory model and install snippets to describe the
  `0.19` eager-vs-streaming source paths and opt-in geometry JSON read-back.

## [0.19.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.18.2...psi-geo-v0.19.0) - 2026-07-05

### API

- Added `FeatureReadRequest::geometry_json` and made GeoJSON geometry read-back
  opt-in so `read_features` can return WKB or projected properties without
  materializing unwanted JSON geometry.
- Added one-shot GeoJSON `FeatureCollection` streaming entrypoints for building
  and converting large GeoJSON inputs without retaining the full parsed document.

### Performance

- Reduced GeoJSON source scan and conversion memory peaks by walking raw feature
  geometry directly for bounds and WKB emission instead of reparsing serialized
  geometry JSON.
- Reduced FlatGeobuf read-back memory peaks by materializing only requested
  geometry output and by preserving source-order reads without extra record
  cloning where possible.
- Improved FlatGeobuf `FeatureJson` payload conversion by assembling payloads
  from raw geometry JSON instead of parsing the geometry back into a
  `serde_json::Value`.
- Reduced temporary allocation in GeoParquet bbox-covering scans and added fast
  WKB paths for common GeoJSON conversion and point exact-filter cases.

## [0.18.2](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.18.1...psi-geo-v0.18.2) - 2026-07-04

### Safety

- Hardened WKB envelope scanning with a crate-local bounded parser that caps
  nesting depth, rejects impossible count hints before iterating, and avoids the
  unbounded recursive `geozero` WKB reader on the source scan path.

## [0.18.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.18.0...psi-geo-v0.18.1) - 2026-07-04

### Performance

- Reduced temporary allocation and payload copying during source scans by
  reusing known row / feature count hints and moving payload bytes for unsplit
  index entries.

### Documentation

- Clarified the source-side memory model for GeoJSON, FlatGeobuf, GeoParquet,
  and the range-friendly converted `PSINDEX` query path.

## [0.18.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.17.0...psi-geo-v0.18.0) - 2026-07-04

### API

- Added FlatGeobuf and GeoJSON source support behind default `flatgeobuf` and
  `geojson` features. New entrypoints `open_flatgeobuf`, `open_geojson`, and
  `open_geojson_slice` can scan, build, convert, and read features back through
  the shared source-side builder core.
- **Breaking:** renamed `open` to `open_geoparquet` so all source entrypoints
  are symmetric (`open_geoparquet` / `open_geojson` / `open_flatgeobuf`); no
  format is privileged by an unsuffixed name.
- Added a `GeoSource` trait (`profile` / `source_fingerprint` / `scan` /
  `build` / `convert` / `convert_into`) implemented by every source type, so
  build/convert pipelines can be written generically over `impl GeoSource`.
  Each type keeps these as inherent methods too. Read-back stays off the trait
  (Parquet returns Arrow `FeatureRows`; other sources return `FeatureRecord`).
- Added `GeoDataset::profile` and made every source's `profile` return
  `Result<GeometryProfile, GeoError>` for a uniform metadata-profile call.
- Added `FeatureRecord` read-back for non-Arrow sources and moved
  `FeatureReadRequest` / `GeometryReadMode` / read ordering and duplicate
  controls to the format-neutral source API.
- Added `gp2psindex --format parquet|flatgeobuf|geojson` plus extension /
  signature detection for `discover`, `inspect`, `build`, `validate`, and
  `query`.
- Marked source metadata enums and `GeoError` as `#[non_exhaustive]`, and added
  FlatGeobuf / GeoJSON source and encoding variants for the new input formats.

### Persistence

- Converted FlatGeobuf and GeoJSON sources now record `source_format:
  "flatgeobuf"` / `"geojson"` and a stable source fingerprint in `geoM`.

## [0.17.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.16.1...psi-geo-v0.17.0) - 2026-07-03

### API

- Added a default `parquet` feature that gates the `arrow` and `parquet`
  dependencies. With `default-features = false` the crate is query-only: it
  opens pre-built `PSINDEX` artifacts and queries them — `open_geo_index` /
  `open_geo_index_async`, `search_items` / `search_hits`,
  `GeoArtifactIndex2D::filter_hits` (exact intersection over the payload
  geometry), payload decoding — with no `arrow` or `parquet`, so the query side
  builds for `wasm32`. The Parquet source side (`open`, `GeoDataset`
  discovery/inspection/validation/read-back, `build` / `convert`, the
  `gp2psindex` CLI) keeps requiring the default `parquet` feature, so existing
  dependants are unaffected.

## [0.16.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.16.0...psi-geo-v0.16.1) - 2026-07-02

### API

- `gp2psindex` now rejects unknown command-line flags instead of silently
  accepting mistyped options.

### Safety

- Hardened geo artifact opening against oversized chunk directories, oversized
  `geoM` manifests, and overflowing aligned ranges before large reads or
  allocations.

### Geometry

- WKB ISO dimension codes (`1000`/`2000`/`3000` plus base type) now drive
  detected geometry dimensions correctly, and non-finite WKB coordinates are
  rejected instead of indexed as valid bounds.
- GeoParquet bbox covering intervals now treat `xmin <= xmax` as a normal
  covering interval and `xmin > xmax` as an explicit antimeridian wrap; planar
  scans reject wrapped covering intervals unless geographic antimeridian
  handling is requested.

### Persistence

- Geo artifact payload content types now come from the selected `PayloadPlan`
  instead of payload byte sniffing.

### Performance

- GeoParquet scans now project only geometry, covering, and requested
  FeatureJson property roots; RowRef scans can use bbox coverings without
  parsing WKB, and FeatureJson property payloads are written at batch level.

### Validation

- Geographic envelope policies now reject known projected CRS columns, while
  missing or unknown CRS metadata remains allowed for validation/reporting.

## [0.16.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.15.0...psi-geo-v0.16.0) - 2026-07-02

### API

- Added async artifact opening and query APIs behind the `async` feature:
  `open_geo_index_async`, `open_geo_index_with_limits_async`, and
  `search_items_async` / `search_features_async` / `search_hits_async` on 2D
  and 3D artifact indexes.

### Search

- `GeoArtifactIndex2D::search_items` now uses polygon region pruning for
  `GeoQuery2D::Polygon`, including payload-free artifacts where `search_hits`
  is unavailable.

### Documentation

- Clarified that streamable geo artifacts answer window, polygon, and 3D
  frustum candidate queries from object storage; kNN and raycast use the
  in-memory accelerator path.

## [0.15.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.14.1...psi-geo-v0.15.0) - 2026-07-01

### API

- Added `GeoIndex::from_scan`, `GeoArtifact::from_scan`, and
  `GeoDataset::source_fingerprint`, so callers can scan a source once and
  reuse that scan for both in-memory indexes and converted artifacts.
- `GeoArtifact::from_scan` now preserves the scan's recorded payload and
  geometry-policy metadata in the `geoM` manifest. It returns the new
  `GeoError::ScanPayloadMismatch` when a conversion request asks for a
  different payload plan than the scan produced.
- **Breaking:** `GeometryScan2D`/`GeometryScan3D` now expose payload and scan
  provenance through read-only accessors: `payload()`, `payloads()`, `nulls()`,
  and `envelope()`. `boxes`, `features`, and `profile` remain public fields.

### Indexes

- Added `GeoIndex2DF32`/`GeoIndex3DF32`, f32-precision in-memory accelerator
  indexes, selectable with `IndexBuildOptions::precision` via
  `GeoDataset::build` or `GeoIndex::from_scan`. They use half the box storage
  of the default f64 indexes and support `Box2D`/`Box3D` queries.
  `Index2DF32`/`Index3DF32` are now re-exported from this crate's root.
- **Breaking:** `GeoIndex` gained `D2F32`/`D3F32` variants; a `match` on
  `GeoIndex` without a wildcard arm must handle them.

### Search

- `gp2psindex query` now accepts `--bbox` with six comma-separated numbers
  (`xmin,ymin,zmin,xmax,ymax,zmax`) against a 3D `.psi` index. 2D-only flags
  such as `--radius`, `--exact`, and `--predicate` now produce clearer errors
  when used with 3D artifacts.
- Added `GeoQuery3D::Frustum3D`/`GeoQuery3D::frustum3d`, a candidate-pruning
  view-frustum query for `GeoIndex3D::search_features` and
  `GeoArtifactIndex3D::search_items`/`search_features`/`search_hits`
  (both f64 and f32 artifacts). Frustum search is a bounding-box candidate
  filter, not an exact geometry intersection test. `Frustum3D` and
  `ClipSpaceZ` are now re-exported from this crate's root.
- **Breaking:** `GeoQuery3D` gained a `Frustum3D` variant (an exhaustive
  `match` needs the new arm), and `GeoQuery3D::candidate_box_3d` now returns
  `Result<Box3D, GeoError>` (`Err` for a degenerate frustum) instead of an
  infallible `Box3D`.
- Updated the public `packed_spatial_index` dependency to 0.21.1, picking up
  scale-invariant frustum-plane handling for 3D frustum queries.
- Added `GeoIndex2D::raycast_features`/`raycast_closest_feature` and
  `GeoIndex3D::raycast_features`/`raycast_closest_feature` (plus
  `f32`-accelerator `raycast_features`) for in-memory accelerator indexes.
  Raycast returns bounding-box candidates; callers that need exact geometry
  hits should run their own narrow-phase test. `Ray2D`/`Ray3D` are now
  re-exported from this crate's root.

### Nearest Neighbors

- Added `GeoIndex2D::nearest_features`/`nearest_features_haversine` and
  `GeoIndex3D::nearest_features` (plus `f32`-accelerator equivalents on
  `GeoIndex2DF32`/`GeoIndex3DF32`) for in-memory accelerator indexes. Results
  are nearest-first with each hit's distance; 2D lon/lat data can use the
  haversine variant for great-circle distance.
- `Point2D`, `Point3D`, `haversine_distance_2d`, and `EARTH_RADIUS_M` are now
  re-exported from this crate's root.

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
