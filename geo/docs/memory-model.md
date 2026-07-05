# Memory model

Memory use depends mostly on where you are in the workflow. Building or
converting from a source file creates a new index and needs workspace memory for
that result. Querying an existing `.psindex` is the low-memory path: it reads
only the artifact ranges touched by the query.

## Quick Guide

| Workflow | Result | What grows in memory | Use when |
| --- | --- | --- | --- |
| `GeoDataset::build` | `GeoIndex` in the current process | Index size, mostly feature count and precision | You want immediate repeated queries and don't need a `.psindex` file |
| `GeoDataset::convert` / `convert_into` | `.psindex` bytes | Index size plus selected payload and output bytes | You want a reusable artifact for later queries |
| `open_geo_index` / `open_geo_index_async` | Reader over an existing `.psindex` | Only the ranges touched by each query; `GeoArtifactDirectory` can cache open metadata across readers | You already have a `.psindex` and want the low-memory query path |
| `read_features` | Source rows | Requested rows and requested geometry output | You already have hits and need properties or geometry from the source |

The short version: **`build` keeps the index in RAM**, **`convert` writes the
index as `.psindex` bytes**, and **`open_geo_index` queries an existing
`.psindex` by range**.

For servers and workers that create a new range reader per request,
`GeoArtifactIndex::into_directory` splits off a reusable `GeoArtifactDirectory`.
Reattaching with `GeoArtifactIndex::from_directory` avoids repeating the
container, `geoM` manifest, and stream-directory reads on warm requests.

## Source Formats

| Source API | What happens to the input | Best fit |
| --- | --- | --- |
| `open_geoparquet` | Reads projected Parquet columns in Arrow batches. Bbox covering columns can avoid decoding geometry just to compute envelopes. | Large GeoParquet or native Parquet geospatial files |
| `open_flatgeobuf` | Reads the header, then streams features during `scan`, `build`, and `convert`. Parsed features are not kept as a full in-memory collection. | FlatGeobuf sources and one-shot conversion |
| `open_geojson` / `open_geojson_slice` | Reads the whole GeoJSON document and keeps feature records in memory. | Repeated operations or read-back from the same opened GeoJSON dataset |
| `build_geojson_stream` / `convert_geojson_stream` | Reads a `FeatureCollection` one feature at a time. The full GeoJSON document is not retained. | Large one-shot GeoJSON build or convert jobs |

All source formats eventually feed the same builder. After scanning, memory is
driven mostly by how many features are indexed and which payload is selected.

## Payload Choices

Payload choice often dominates artifact size and build memory.

| Payload plan | What it stores | Memory impact | Use when |
| --- | --- | --- | --- |
| `PayloadPlan::None` | No per-hit payload | Smallest | You only need spatial matches |
| `PayloadPlan::RowRef` | Source row identity | Small | You can read rows from the source later |
| `PayloadPlan::RowWkb` | Source row identity plus WKB geometry | Tracks geometry size | You want exact filtering from the artifact |
| `PayloadPlan::FeatureJson` | GeoJSON Feature bytes with selected properties | Usually largest | You want query results that are already feature JSON |

For sidecar indexes next to a source file, `RowRef` is the smallest useful
payload. For standalone artifacts that need exact filtering without source
geometry reads, `RowWkb` is the default tradeoff.

## Read-Back

`read_features` is separate from building and querying. It opens the original
source and returns requested rows.

| Request option | Geometry work | Memory shape |
| --- | --- | --- |
| Projected properties only | No geometry JSON is created | Usually smallest |
| `GeometryReadMode::Wkb` | Materializes WKB for requested rows | Rows include geometry bytes |
| `FeatureReadRequest::geometry_json = true` | Materializes JSON geometry for requested rows | Rows include JSON geometry values |

Keep `geometry_json` off unless a downstream step needs JSON geometry. This
avoids creating large `serde_json::Value` geometry trees during read-back.

## Choosing a Path

| Situation | Prefer |
| --- | --- |
| Large GeoParquet or native Parquet source | `open_geoparquet(...).convert(...)` or `build(...)` |
| FlatGeobuf source with one-shot conversion | `open_flatgeobuf(...).convert(...)` |
| Large GeoJSON `FeatureCollection` | `convert_geojson_stream` or `build_geojson_stream` |
| Single GeoJSON `Feature` or bare geometry | `open_geojson` |
| Browser / edge / object-storage queries | Pre-build `.psindex`, then use `open_geo_index_async` |
| Smallest sidecar next to an existing source | `PayloadPlan::RowRef` |
| Exact filtering from the artifact without source geometry reads | `PayloadPlan::RowWkb` |
