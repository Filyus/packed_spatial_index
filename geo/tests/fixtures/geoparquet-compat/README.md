# GeoParquet compatibility fixtures

Small regression fixtures copied from the ignored external compatibility
corpus.

Sources:

- `data-point-encoding_wkb.parquet`: `opengeospatial/geoparquet`
  `test_data/data-point-encoding_wkb.parquet`, OGC GeoParquet test data.
- `example-crs_vermont-custom.parquet`: `geoarrow/geoarrow-data`
  `example-crs/files/example-crs_vermont-custom.parquet`.
- `example_point_native.parquet`: `geoarrow/geoarrow-data`
  `example/files/example_point_native.parquet`.
- `example_geometry-mixed-dimensions_geo.parquet`: `geoarrow/geoarrow-data`
  `example/files/example_geometry-mixed-dimensions_geo.parquet`.

The upstream repositories publish these samples for interoperability testing.
Keep this directory intentionally small; the larger compatibility corpus stays
under ignored `dev/`.
