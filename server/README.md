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
- `GET /collections/{id}/items?bbox=minx,miny,maxx,maxy&limit=&offset=&exact=`
- `GET /collections/{id}/hits?bbox=minx,miny,maxx,maxy&limit=&offset=&exact=&payload=`

`/items` returns a GeoJSON `FeatureCollection` only when the artifact carries a
`FeatureJson` payload. Use `/hits` for all payload modes, including `RowRef`,
`RowWkb`, and payload-less artifacts.

`/hits` accepts `payload=none|summary|full`; the default is `summary`. Summary
mode returns payload kind and cheap metadata, while `full` materializes stored
payload values such as base64 WKB or embedded GeoJSON features.

`/hits` reports matching index entries. `/items` reports source features and
deduplicates split entries such as antimeridian parts.
