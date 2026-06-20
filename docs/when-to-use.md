# When to use it

`packed_spatial_index` fits when your geometry is **static or rebuilt in batches**
and you want to query it cheaply — in-process at high throughput, or served
straight from object storage to the edge, a browser, or a game, with no backend.
Pack the boxes once into an immutable, byte-deterministic index; query it
in-process, mmap the serialized bytes, or stream them over the network.

A spatial database (PostGIS, SpatiaLite, any R-tree / GiST engine) is the opposite
shape — a running server with mutable state. The two win in different situations;
this page is the honest split.

## Reach for this crate when

- **You want spatial queries without a server.** The serialized index lives on
  S3 / R2 / a CDN, and a Cloudflare Worker, a wasm bundle, or a browser fetches
  only the bytes a query touches. A 100 MB index answers a windowed query in a
  handful of range reads with no backend at all.
- **Cost and scale matter.** Object storage is cheap and scales to zero; there is
  no idle database to keep warm. Readers are just file readers, so any number of
  them scale out for free.
- **You're embedding.** A dependency-light crate that runs in wasm, on the edge,
  in a browser, or inside a render loop — where a database cannot go.
- **Throughput on static geometry is the goal.** A packed, struct-of-arrays,
  cache-friendly layout with runtime-dispatched SIMD and reusable query buffers,
  for millions of read-only queries without per-row or transactional overhead.
- **You want the index to be reproducible.** A byte-identical build you can hash,
  cache, ship through CI, mmap, or serve from a CDN.
- **Your queries go beyond a database's comfort zone.** Range / intersection, kNN
  (including great-circle distance), all-hits and closest-hit ray casts, triangle
  / convex-polygon / view-frustum culling and spatial joins — as a library with
  tight loops. In a graphics or game context a database is the wrong tool
  entirely.

## Reach for a spatial database when

- **The data changes frequently.** Databases handle continuous inserts, updates
  and deletes in place. This crate is static, so a change means a rebuild. That is
  fine when data changes rarely or in batches: rebuild and publish a new file (a
  versioned key gives atomic swaps, rollback and CDN-cache control, like deploying
  a static asset). Reach for a database when writes are frequent, or
  must land without a full rebuild.
- **You need rich or exact queries.** Attribute joins, arbitrary predicates,
  aggregation, the full GIS function set (`ST_*`), reprojection, topology and
  exact geometry operations (intersection, buffer, distance to the real shape).
  This crate indexes bounding boxes and returns candidate ids plus an opaque
  payload; any exact-geometry step is on you.
- **You filter by attribute and space together.** "Restaurants within this polygon
  open after 9pm." This crate indexes only AABBs plus an opaque per-item payload;
  combined attribute filtering is not built in (an attribute-column profile is
  possible future work, not a feature today).
- **You need transactions, concurrency, durability — ACID.** A database's domain.
- **Maturity and ecosystem are decisive.** PostGIS is battle-tested with decades
  of tooling behind it.

## They also compose

This is not an either/or. A database or a batch pipeline is a fine *source*, and
`packed_spatial_index` is the cheap, serverless *read path* in front of it: build
the index from whatever holds the data, then serve queries from a static file.

## Example: GeoParquet

The contrast is sharpest in cloud-native geo. The database path is "load the
GeoParquet into PostGIS, stand up and operate a server, point clients at it." The
[`packed_spatial_index_geo`](../geo) path is "convert the GeoParquet to a
`PSINDEX` once, drop it on R2, let the edge range-query it." Same windowed and kNN
queries, returning the matching features' geometry — with no server in the loop.
When the data is a static dump and the queries are read-only, removing the server
is the whole point.
