# Changelog

All notable changes to `packed_spatial_index_geo` are documented here.

## [Unreleased]

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
