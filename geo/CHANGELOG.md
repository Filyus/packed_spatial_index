# Changelog

All notable changes to `packed_spatial_index_geo` are documented here.

## [Unreleased]

## [0.27.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.26.0...psi-geo-v0.27.0) - 2026-09-06

### API

- Requires `packed_spatial_index` 0.29. That release adds the ordered pick, the
  radius query, the distance-join family (`join_within`, anti-join, components,
  closest pair), selectivity estimation and the interleaved-layout owned
  loader, which the CLI's new verbs read artifacts through. Callers that also
  depend on `packed_spatial_index` directly must move to 0.29 together with
  this crate, because the core types in this API — `Box3D`, `Frustum3D`,
  `Index3D` — come from it.
- `gp2psindex query --pick ox,oy,oz,dx,dy,dz --half-angle deg [--limit k]` on
  a 3D artifact: the click's ordered broad phase, the CLI face of the core
  crate's `search_pick`. Prints one NDJSON line per candidate —
  `{"entry":i,"distanceSquared":d,"entryT":t}` — in pick order: boxes the ray
  pierces first (near-to-far), then boxes it only grazes by increasing
  perpendicular distance. Refused with the aggregate flags and every query
  shape.
- `GeoArtifactIndex2D::estimate_entries` / `GeoArtifactIndex3D::estimate_entries`
  and `directory_floor`, the geo faces of the core crate's selectivity
  estimation: an exact `[lower, upper]` bracket on how many index entries a
  bbox matches plus a point estimate, read from node boxes; at or above the
  directory floor it costs no reads. Bbox only.
  `gp2psindex query --bbox … --estimate` prints the bracket as one JSON line.
- `gp2psindex join <a.psi> <b.psi> --within N` reports every pair of items
  whose boxes lie within `max_distance` of each other, the CLI face of the
  core crate's distance join: one `{"a":i,"b":j}` NDJSON line per pair,
  streamed from inside the join's visitor (the join is output-bound, so
  materializing the pairs would cost more memory than the indexes do);
  `--count` prints the count and streams nothing. The same path twice is a
  self-join — every unordered pair of distinct items once — and both
  artifacts must be the same dimensionality. Semantics mirror the server's
  `/collections/{id}/join/{other}`: box-to-box Euclidean distance in
  coordinate units, zero when the boxes overlap, inclusive at the bound, so
  `--within 0` is the plain overlap join.
- `gp2psindex anti-join <a.psi> <b.psi> --within N` and
  `gp2psindex components <a.psi> --within N`, the CLI faces of the server's
  `/anti-join` and `/components`. The anti-join streams one `{"a":i}` line per
  item of `a` with no item of `b` within the bound (and refuses the same path
  twice: against itself every item is at distance zero from itself, so the
  question meant is `components`). Components print one `{"item":i,"label":l}`
  line per item, the label being the smallest item id in the component;
  `--count` prints the count and streams nothing.
- `gp2psindex closest-pair <a.psi> <b.psi>` prints the single nearest pair and
  its box-to-box distance as one JSON line, or `null` when there is none; the
  same path twice reports the nearest pair of distinct items within one
  artifact. The CLI face of the server's `/closest-pair/{other}`.
- `gp2psindex query` takes `--polygon` (GeoJSON MultiPolygon coordinates, 2D
  only; the polygon drives the index traversal itself, so it needs no payload
  and `--count` works over it) and, against a 3D index, `--frustum` (24
  numbers, six inward-pointing planes as `a,b,c,d`). Both are the shapes the
  server's `/search` already accepted, validated by the same rules.

## [0.26.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.25.0...psi-geo-v0.26.0) - 2026-08-25

### API

- Requires `packed_spatial_index` 0.28. That release adds `count(query)` to
  every frontend that answers a range query, including the streaming readers
  this crate reads artifacts through, plus `count_region` for their shape
  queries. Callers that also depend on `packed_spatial_index` directly must move
  to 0.28 together with this crate, because the core types in this API —
  `Box2D`, `Index2D`, `BuildError` — come from it.

### Performance

- `count_entries` and `count_entries_async` now count inside the core's
  traversal instead of running their own visitor over it, for every query shape
  they accept: a bbox, a polygon, and in 3D a frustum. The answers and the reads
  are unchanged — this is the same walk with the counting moved one layer down —
  but it is no longer four lines of visitor per call site, and it inherits
  whatever the core does to that traversal. The 0.28 core also answers a fully
  contained subtree from its leaf range rather than testing every item in it,
  which is what a wide `count_entries` over a dense artifact spends its time on.

## [0.25.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.24.0...psi-geo-v0.25.0) - 2026-08-12

### API

- Requires `packed_spatial_index` 0.27. That release rejects boxes with
  `min > max` or a `NaN` bound at build time, so `GeoError::Build` can now carry
  its new `BuildError::InvalidItemBounds`. No source in this crate can produce
  one: a geometry with no coordinates — an empty `LineString`, a null geometry
  member — is already `GeoError::NullGeometry` at scan time, or dropped under
  `NullPolicy::Skip`, so the crossed envelope a `±INFINITY` accumulator would
  otherwise yield never reaches a builder; and the antimeridian case splits into
  two well-formed boxes precisely to avoid one. Callers that also depend on
  `packed_spatial_index` directly must move to 0.27 together with this crate,
  because the core types in this API — `Box2D`, `Index2D`, `BuildError` — come
  from it.

## [0.24.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.23.0...psi-geo-v0.24.0) - 2026-07-31

### API

- Added the query-shape vocabulary an artifact frontend answers a search with:
  `PayloadMode`, `IdentityMode`, `ResultLevel`, and the rules that resolve
  them against an artifact — `resolve_identity`, `resolve_level` (with
  `LevelError`), `needs_payload_bodies`, `stores_feature_ids`, and
  `public_feature_json`. Added `GeoErrorClass` and `classify_geo_error`
  alongside them, mapping a `GeoError` to the status and code a frontend
  should report for it. The native server and the Worker demo each carried
  their own copy of all of this, and the copies had already drifted: the
  Worker's classifier never learned `UnsupportedGeodeticGeometry`, so that
  error reported 500 where the server reports 422.
- Re-exported `StreamError` from the core crate. `GeoError::Stream` wraps it,
  so a caller could not tell an exhausted read budget from a corrupt artifact
  without naming the type.
- **Breaking:** `ConvertRequest` gained `prefix_index: PrefixIndexPolicy`,
  deciding whether the artifact carries a contiguous copy of its feature refs.
  `gp2psindex build` exposes it as `--prefix-index auto|on|off`.
- `gp2psindex query` gained `--count`, `--limit`, and `--offset`, so the CLI
  can answer "how many" without reading source rows and can read back one page
  instead of materializing every match. Both describe the index's own match
  set, so they are refused together with `--exact` (and `--radius`, which is
  always exact), which narrows that set afterwards.
- Added the header and body-fetch family to async 3D artifacts:
  `search_match_headers_async`, `search_match_headers_page_async`,
  `fetch_matches_async`, `search_payload_headers_async`,
  `search_payload_headers_page_async`, and
  `fetch_payload_header_matches_async`. Async 3D previously exposed five
  methods where 2D exposed twelve, so a 3D caller over async I/O had to read
  full matches where a 2D caller could page headers. A 3D query never expands
  to several candidate boxes, so these page without the cross-box
  deduplication the 2D versions need.
- Added `count_entries_async` to `GeoArtifactIndex2D` and `GeoArtifactIndex3D`.
  The docs claimed no async variant was possible because the async layer
  exposed no visitor, which has not been true since the region visitors landed;
  a box is its own region, so the count streams there too instead of
  materializing entry ids.
- Added `search_match_headers_page` to `GeoArtifactIndex2D` and
  `GeoArtifactIndex3D`. Bounded, deterministic entry pagination was previously
  async-only, so synchronous callers had to materialize every match header and
  page in memory. It reports the exact pre-pagination count while retaining at
  most `offset + limit` headers, and `GeoMatchHeaderPage` is no longer gated
  behind the `async` feature.
- `IndexDimsRequest::D2` now projects away z instead of rejecting a source that
  has one, matching what "force 2D envelopes" always claimed. `D3` still
  requires the scan to find a z extent: promoting 2D input would have to invent
  one, placing every entry at zero. The `DimMismatch` message now says that
  scanned envelopes decide dimensionality, since a bbox covering without z
  bounds is the usual reason a `Point Z` source scans as 2D.
- Updated the public `packed_spatial_index` dependency to 0.26.0. That is where
  `Serializer*::payload_prefix_len` and `StreamLimits::prefix_coalesce_gap_bytes`
  live, so the `prefix_index` option above cannot be built against an older
  core — this release requires it rather than merely preferring it.

### Validation

- Added `ValidationCode::CoveringMissingZ` for a column that declares a z
  coordinate its bbox covering cannot describe. That condition previously
  reported as `UnknownDimensions`, which already means two unrelated things, so
  a client could not tell "the covering cannot carry z" from "metadata never
  said". **Breaking:** `ValidationCode` and `DiscoveryWarning` are now
  `#[non_exhaustive]` — the only geo vocabularies that were not — so exhaustive
  matches on them need a fallback arm. Future codes are additive after this.
- `validate` reports that warning whenever a column's bbox covering cannot
  describe the z its geometry types declare, since envelopes taken from such a
  covering index 2D boxes.

### Documentation

- Documented that `GeoArtifactIndex3D` has no `filter_matches` and `GeoQuery3D`
  no polygon variant by design: there is no exact 3D predicate for one to
  narrow candidates with, so the asymmetry with 2D is a scope boundary rather
  than a feature still missing.
- Documented that the payload plan selects the envelope source (bbox covering
  versus decoded WKB) and therefore the index dimensions, including that
  `BuildRequest` always scans with `PayloadPlan::None` while
  `ConvertRequest::default()` uses `RowWkb`. The README and guide now say so in
  prose, not only rustdoc.
- Documented that a geographic envelope decides "crosses the antimeridian" by
  the shortest way round, the RFC 7946 reading, while every source this crate
  reads declares planar edges. Where the two disagree, splitting indexes the
  shortest-path interpretation.
- Documented that `GeoMatchHeader` carries partial identity: its `FeatureRef`
  comes from the fixed payload prefix, which has no room for a source
  `feature_id`, so a header reports `None` where a `GeoMatch` for the same
  entry may report `Some(..)`. The previous wording claimed headers carry
  everything sorting, deduplication, and pagination need, which is true only
  because `row_number` already identifies a source feature — `cmp_feature`
  includes `feature_id` in its key, and that unstated dependency is now
  recorded on both types.

### Geometry

- `GeoArtifactIndex2D::filter_matches` now checks the artifact's payload plan
  before iterating candidates, so a `RowRef` artifact reports the documented
  `PayloadDecode` error even when the candidate list is empty. It previously
  returned `Ok(vec![])` in that case.
- A spherical-radius filter now wraps longitudes outside `[-180, 180]` instead
  of dropping the point. A dataset stored in `[0, 360)` silently matched
  nothing east of the antimeridian, even though the query side already wrapped
  its own longitude. Wrapping is also arithmetic rather than stepping by 360,
  so an out-of-range source coordinate can no longer hang the wrap.
- `NonPlanarExactPolicy::TreatAsPlanar` now also applies to spherical-radius
  exact filtering, which previously required `edges: spherical` with no way to
  opt out. GeoJSON and GeoParquet without an `edges` member declare planar edges
  while storing lon/lat degrees, so such a search returned bbox candidates that
  `filter_matches` / `filter_features` then always refused to narrow. A
  spherical-radius filter against a known-projected CRS is still rejected: no
  policy makes a metre radius meaningful there, matching the guard the scan side
  already applies to geographic envelopes. `gp2psindex query` accepts
  `--treat-nonplanar-as-planar` alongside `--radius` to reach it; the two used
  to be mutually exclusive, which left the CLI unable to filter exactly against
  any planar-declaring lon/lat source.
- A GeoParquet bbox covering without `zmin`/`zmax` now builds a 2D index even
  when the column declares `Point Z` or similar. Previously the declared
  geometry types decided the index dimensions while the covering supplied the
  envelopes, so every entry was placed at z == 0 and any z-restricted query
  silently matched nothing. Scanning with a payload plan that reads geometry
  (`RowWkb` / `FeatureJson`) still indexes real z extents.

### Persistence

- **Breaking:** `GeoArtifactManifest` gained `stores_feature_ids:
  Option<bool>`, recording whether any payload body actually carries a source
  feature id. The payload plan was the only clue before, and it answers a
  different question — where an id *would* live, not whether one exists — so a
  reader keying on it spends a page of body reads on every `FeatureJson`
  artifact built from Parquet or FlatGeobuf, neither of which ever assigns one.
  `None` on artifacts written before the field existed; that means unknown, not
  false.
- **Breaking:** a `FeatureJson` body now writes its `feature_ref` member only
  when the source feature has an id. The member re-encodes the fixed-width
  record the payload already carries in front of it, and the id is the only
  field that record has no room for — so without one it was pure duplication,
  measured at 23.5% of a 100k-feature artifact built from GeoParquet (42.5 MB
  to 32.6 MB). Readers already preferred the record and fell back to the
  member, including the published 0.22 and 0.23, so an artifact written this
  way decodes to the same `FeatureRef` everywhere. Legacy bodies that carry no
  fixed record are still read from the member.
- **Breaking:** a `FeatureJson` body now omits the GeoJSON `id` member when the
  source feature has none, instead of writing `""`. Only GeoJSON sources supply
  a feature id at all — Parquet and FlatGeobuf never do — so every body they
  produced claimed an identity the feature did not have, and no reader could
  tell that apart from an id that really is the empty string. RFC 7946 makes
  `id` optional. Existing artifacts keep their `""`; the change applies to
  newly written bodies.
- A converted artifact now carries a contiguous copy of its feature refs when a
  header search would otherwise cost one range request per match — the
  difference between a usable and an unusable paged query over object storage.
  `PrefixIndexPolicy::Auto` decides from the median stored payload rather than
  from the payload plan, because `RowWkb` sits on both sides of the line: a 2D
  point is 45 bytes with its ref and its prefixes still coalesce, while a
  two-point line at 65 bytes takes 1001 reads over 1000 entries where the
  section takes 2. That cliff is the whole rule; there is no slope between the
  two, and above it the request count stays at one per match however large the
  payload grows. The price is file size — the scan reads the same bytes either
  way — worst at the cliff at about a quarter and a tenth by 190 bytes.
  `RowRef` never gets a section: its whole payload is the ref.
- The `geoM` manifest now records the dimensions the index was actually built
  in rather than the ones the source profile declared. A scan whose geometries
  were all skipped, or an empty source, left the profile at `Unknown` and wrote
  an artifact that `open_geo_index` then rejected as having unknown coordinate
  dimensions. Such artifacts now open as empty 2D indexes.

## [0.23.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.22.0...psi-geo-v0.23.0) - 2026-07-15

### API

- Added async header-only searches and selected-body fetches to
  `GeoArtifactIndex2D`. `search_match_headers_async` / `fetch_matches_async`
  work with decoded feature refs; `search_payload_headers_async` /
  `fetch_payload_header_matches_async` use the lighter new `GeoPayloadHeader`
  when source rows are not duplicated. `GeoPayloadHeader` exposes entry id,
  payload length, deterministic entry sorting, and body length.
- Added bounded, deterministic entry pagination:
  `search_match_headers_page_async` returns `GeoMatchHeaderPage` and
  `search_payload_headers_page_async` returns `GeoPayloadHeaderPage`. Both
  report the exact pre-pagination match count while retaining at most
  `offset + limit` headers for single-box and polygon queries.
- `FeatureJson` artifacts now support synchronous 2D/3D header searches and
  selected-body fetches, plus the new async 2D path. Added `feature_json_body`
  for callers that need the embedded GeoJSON bytes from a FeatureRef-prefixed
  payload while retaining compatibility with legacy raw-JSON payloads.

### Safety

- Hardened untrusted artifact opening and header reads. The reader now rejects
  manifest count or payload-presence mismatches, malformed or mislabeled row
  payloads, and payload identities that do not agree with their index entries.
  `decode_feature_ref_payload` now rejects non-zero reserved bytes, and
  `FeatureJson` payload prefixes are detected structurally rather than by a
  leading byte alone.

### Persistence

- Newly built `FeatureJson` payloads now store a fixed 24-byte `FeatureRef`
  prefix before the GeoJSON body, enabling header-only searches without reading
  every body. Legacy raw-JSON artifacts remain readable through
  `feature_json_body`. `FEATURE_JSON_CONTENT_TYPE` now identifies the complete
  binary record as
  `application/vnd.packed-spatial-index.feature-json` instead of advertising it
  as standalone `application/geo+json`.

### Performance

- Reduced object-storage work for paginated payload queries. Async header pages
  keep only `offset + limit` headers for single-box and polygon searches, while
  selected-body fetches read payload bodies by leaf rank instead of scanning
  every matching body.

## [0.22.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.21.2...psi-geo-v0.22.0) - 2026-07-07

### API

- Added `GeoMatchHeader::body_byte_len`, the length of a match header's payload
  body after its fixed feature-ref record. Exposes the `RowWkb` WKB byte length
  without callers re-deriving `payload_len - FEATURE_REF_RECORD_LEN`.

### Performance

- Deduplicated multi-box `search_entry_ids` results with constant-time entry-id
  lookup, avoiding quadratic scans for queries such as antimeridian-split boxes.

### Documentation

- Documented `read_geo_manifest` as a trusted-input helper and pointed
  untrusted artifact readers at `open_geo_index_with_limits`.

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
