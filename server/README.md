# packed_spatial_index_server

Local native HTTP server for querying existing geospatial `.psindex` artifacts.

A reference server: it exists to show what the `packed_spatial_index_geo` query
API looks like behind HTTP. It is not published to crates.io, and its changes
are recorded in git history rather than a changelog.

The MVP is artifact-first: it does not build or convert sources, does not read
back original source files, and does not use remote/object storage. It opens
each configured artifact at startup, caches the parsed geo manifest and stream
directory, then attaches a fresh local file reader per request.

## Catalog

```toml
[server]
addr = "127.0.0.1:3000"

[server.limits]
max_reads = 0
max_read_bytes = 536870912
max_items = 1000000

[[collections]]
id = "places"
title = "Places"
description = "Local places index"
artifact = "data/places.psindex"
```

Artifact paths are resolved relative to the catalog file.

`[server.limits]` bounds the cost of a single query; the values above are the
defaults and `0` lifts a limit. A query that exceeds one is answered with 422
`query_too_large` rather than running unbounded.

Limits matter because `numberMatched` is exact, so a query still visits every
match even when `limit` is small. What it *retains* depends on the collection:
a paged header search keeps only the requested page, but that needs entry-level
paging, so it does not apply when the artifact can split one source row across
several entries (antimeridian handling) and the request asks for feature-level
results — there the matched set is grouped in memory first. Payload-less
collections have no paged path at all, so limits are their only bound.

## Run

```powershell
cargo run --manifest-path server/Cargo.toml -- --catalog psindex-server.toml
```

Logging follows `RUST_LOG`; `RUST_LOG=info` prints one line per request with
method, path, status, and latency. Ctrl-C (or SIGTERM on Unix, Ctrl-Break on
Windows) shuts down after in-flight requests finish.

## Endpoints

- `GET /health`
- `GET /collections`
- `GET /collections/{id}`
- `GET /collections/{id}/items?bbox=minx,miny,maxx,maxy&limit=&offset=&predicate=`
- `GET /collections/{id}/search?bbox=minx,miny,maxx,maxy&limit=&offset=&predicate=&level=&payload=&identity=`

`/search` is the artifact-native endpoint; it works for every payload kind
(`none`, `row_ref`, `row_wkb`, `feature_json`) and returns a JSON envelope with
a `matches` array. `/items` is the GeoJSON view: it returns a
`FeatureCollection` and requires a `feature_json` payload; other artifacts get
a 422 pointing at `/search`. `/items` also rejects `/search`-only options
(`level`, `payload`, `identity`) with `unsupported_query`.

Unknown query parameters and unknown catalog keys are rejected (`invalid_query`
and a startup error) rather than ignored, so a misspelled name fails loudly
instead of silently resolving to a default. This applies to the two endpoints
that take parameters; `/collections` and `/collections/{id}` take none, so
there is nothing a typo could quietly become and they ignore the query string
rather than refusing an incidental `?f=json`.

Query parameters:

- `bbox` — required; 4 numbers for 2D artifacts, 6 for 3D.
- `predicate=bbox|intersects` — `bbox` (default) intersects stored envelopes
  only; `intersects` refines candidates with exact geometry intersection from
  artifact payloads. Unsupported combinations (3D, payload without geometry)
  return `unsupported_predicate`; edge-model/query mismatches that are only
  discovered while filtering, such as non-planar exact predicates, return
  `unsupported_query`.
- `level=feature|entry` — `/search` only. `feature` (default when the artifact
  stores feature refs) returns one match per source feature, deduplicating
  split index entries such as antimeridian parts; `entry` returns raw index
  entries. Payload-less artifacts only support `entry` and it becomes the
  default for them.
- `payload=none|summary|full` — `/search` only; default `summary`. Summary
  returns payload kind and cheap metadata; `full` materializes stored values
  such as base64 WKB or embedded GeoJSON features.
- `identity=ref|full` — `/search` only; default `ref`. A match always carries
  the fixed-width feature reference (row number, row group, part), but a source
  `featureId` lives inside the payload body, so `full` reads bodies for the
  returned page to include it. It only changes `feature_json` collections;
  other payload kinds store no source id to recover, so `full` is accepted and
  echoed there but reads nothing extra, and `capabilities.identityModes` lists
  `full` only where it can actually add something.
- `limit`, `offset` — pagination over the matched set.

Responses echo the effective query (after defaults) under `query`, so a client
can always see which `level` and `predicate` actually applied. `numberMatched`
counts matches before pagination and `numberReturned` after. Each match
carries `entryId` (index entry ordinal in the artifact; stable per artifact
build, not across rebuilds) and, when the payload stores one, a `featureRef`
back to the source feature.

Collection metadata reports the artifact `payloadKind` plus a `capabilities`
object listing the accepted `predicates`, `levels`, `payloadModes`, and
`identityModes`, and whether `/items` is available.
