# packed_spatial_index_server

Local native HTTP server for querying existing geospatial `.psindex` artifacts.

The MVP is artifact-first: it does not build or convert sources, does not read
back original source files, and does not use remote/object storage. It opens
each configured artifact at startup, caches the parsed geo manifest and stream
directory, then attaches a fresh local file reader per request.

## Catalog

```toml
[server]
addr = "127.0.0.1:3000"

[[collections]]
id = "places"
title = "Places"
description = "Local places index"
artifact = "data/places.psindex"
```

Artifact paths are resolved relative to the catalog file.

## Run

```powershell
cargo run --manifest-path server/Cargo.toml -- --catalog psindex-server.toml
```

## Endpoints

- `GET /health`
- `GET /collections`
- `GET /collections/{id}`
- `GET /collections/{id}/items?bbox=minx,miny,maxx,maxy&limit=&offset=&predicate=`
- `GET /collections/{id}/search?bbox=minx,miny,maxx,maxy&limit=&offset=&predicate=&level=&payload=`

`/search` is the artifact-native endpoint; it works for every payload kind
(`none`, `row_ref`, `row_wkb`, `feature_json`) and returns a JSON envelope with
a `matches` array. `/items` is the GeoJSON view: it returns a
`FeatureCollection` and requires a `feature_json` payload; other artifacts get
a 422 pointing at `/search`. `/items` also rejects `/search`-only options
(`level`, `payload`) with `unsupported_query`.

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
- `limit`, `offset` — pagination over the matched set.

Responses echo the effective query (after defaults) under `query`, so a client
can always see which `level` and `predicate` actually applied. `numberMatched`
counts matches before pagination and `numberReturned` after. Each match
carries `entryId` (index entry ordinal in the artifact; stable per artifact
build, not across rebuilds) and, when the payload stores one, a `featureRef`
back to the source feature.

Collection metadata reports the artifact `payloadKind` plus a `capabilities`
object listing the accepted `predicates`, `levels`, and `payloadModes`, and
whether `/items` is available.
