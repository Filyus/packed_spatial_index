# When to use it

GeoParquet and native Apache Parquet `GEOMETRY` / `GEOGRAPHY` columns store
geometry plus, in some cases, optional bbox/statistics metadata — but neither
gives you a per-row spatial index that pinpoints individual features. This
crate builds that index from either format, in one of two modes.

## Reach for the accelerator when

The geospatial Parquet file stays put and you just want fast windowed / kNN /
raycast lookups against it. The index holds one box and one feature ref per
row — no geometry copy — so its size tracks row count, not geometry size:
measured on 100k simple points, that's already ~95% of the source Parquet's
size; it gets smaller relative to the source as geometries grow larger or more
complex. A query hands you source row numbers to read back from the file. No
conversion step, no second artifact to manage — open the source, build, query.

## Reach for the converter when

You want a portable, cloud-served store. By default it folds feature refs and
geometry into one self-describing `PSINDEX` blob that the core streaming
engine queries directly over HTTP range requests, with no Parquet re-read.
Use `PayloadPlan::RowRef` instead when you want the smallest sidecar option: a
box and a feature ref per row with no geometry copy, measured at about
three-quarters the size of the default `RowWkb` payload (which duplicates each
row's WKB geometry alongside the ref).

```text
GeoParquet / Parquet geo
  -> packed_spatial_index_geo
  -> index / PSINDEX
  -> core queries
```

The two phases need not share a runtime: convert on a server, query from the
edge. Build runs server-side once; query runs anywhere the core streaming
engine does — wasm, an edge Worker, a browser.

## How it differs from `oxigdal-geoparquet`

[`oxigdal-geoparquet`](https://crates.io/crates/oxigdal-geoparquet) is a
GeoParquet driver: it reads and writes GeoParquet, exposes metadata, and can
push bbox / attribute filters into a read. It is the right layer when your
next step is to open a Parquet file and read matching rows or row groups.

This crate is an index layer. It builds a per-feature spatial index from
GeoParquet or native Parquet geospatial columns, then keeps that index in
memory or writes it as a reusable `PSINDEX` sidecar / artifact. That matters
when the source Parquet is large, remote, or slow to scan repeatedly: object
storage, network filesystems, cold HDD archives, and lakehouse datasets all
benefit from answering the spatial lookup from a compact index first.

Reach for `oxigdal-geoparquet` when you need a GeoParquet reader or writer.
Reach for this crate when the file already exists and repeated bbox, kNN, or
raycast lookups should avoid scanning geometry rows again.

They also compose: this crate can identify candidate feature rows, and a
Parquet reader can fetch the attributes or full geometries from the source
file for the rows that survive the query.
