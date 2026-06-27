# Changelog

All notable changes to `packed_spatial_index_geo` are documented here.

## [Unreleased]

### Added
- Metadata-only geometry discovery API (`discover`, `discover_with_opts`) that
  reports GeoParquet/native Parquet geospatial candidates, default selection
  status, and per-column index/payload capabilities.
- `gp2psindex inspect`, including `--geometry-column` and `--json` output for
  the discovery result.

## [0.3.1](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.3.0...psi-geo-v0.3.1) - 2026-06-27

### Changed
- Refined README and rustdoc wording to describe GeoParquet and native Parquet
  geospatial inputs consistently.

## [0.3.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.2.0...psi-geo-v0.3.0) - 2026-06-27

### Added
- Native Apache Parquet `GEOMETRY` / `GEOGRAPHY` logical-type support, including
  files that have no GeoParquet `geo` metadata.
- Explicit geometry-column selection for readers, builders, converter options,
  and `gp2psindex --geometry-column`.
- `GeometryMetadataSource` on `GeoParquetInfo` to distinguish GeoParquet metadata
  from native Parquet geospatial logical types.

### Changed
- `GEOGRAPHY` inputs are indexed as coordinate bounding boxes over their WKB
  coordinates; exact spherical or ellipsoidal predicates remain the caller's
  responsibility after candidate lookup.

## [0.2.0](https://github.com/Filyus/packed_spatial_index/compare/psi-geo-v0.1.0...psi-geo-v0.2.0) - 2026-06-27

### Added
- `ConvertPayload` payload modes for the converter: no payload, row-id-only
  sidecar payload, or original row id + WKB.
- Decode helpers and content-type constants for Geo converter payloads.
- `gp2psindex --payload none|row-id|row-wkb`.

### Changed
- The default converter payload now stores `u64le original_row_id` followed by
  WKB, so outputs created with `skip_null` can still point back to source
  GeoParquet rows.
- Native GeoParquet with a covering column can be converted with
  `ConvertPayload::RowIds`, because that mode does not require geometry decoding.

## [0.1.0] - 2026-06-20

Initial release: build a [`packed_spatial_index`](https://crates.io/crates/packed_spatial_index)
spatial index from a GeoParquet file.

### Added
- **Accelerator** — `build_index_2d` / `build_index_3d` build an in-memory index
  over the row bounding boxes; item id equals the GeoParquet row index.
- **Converter** — `convert_2d` / `convert_3d` (and the buffer-reusing `_into`
  variants) build the index, attach each row's WKB geometry as a leaf-ordered
  payload, and record the CRS, serialized to a streamable `PSINDEX` blob.
- **Primitive / introspection** — `read_bboxes_2d` / `read_bboxes_3d`,
  `inspect` + `GeoParquetInfo`, `detect_dims`.
- **`gp2psindex` CLI** for the file-to-file path.
- Boxes from the GeoParquet 1.1 bbox covering column when present, otherwise from
  the WKB envelope; `Binary` / `LargeBinary` / `BinaryView` geometry columns; 2D
  and 3D; optional `f32` storage; `skip_null`; interleaved payload.
