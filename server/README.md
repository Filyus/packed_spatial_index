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

cors = []

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

`cors` lists the browser origins allowed to read this server, exactly — there
is no wildcard. It is empty by default, so a page from another origin cannot
read the responses. Since the server has no authentication, opening it to an
origin is a per-deployment decision rather than a default.

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
- `GET /collections/{id}/items?bbox=minx,miny,maxx,maxy|radius=|polygon=&limit=&offset=&predicate=`
- `GET /collections/{id}/search?bbox=minx,miny,maxx,maxy|frustum=|radius=|polygon=&limit=&offset=&predicate=&nonplanar=&level=&payload=&identity=&count=`
- `POST /collections/{id}/search` — the same search as a JSON body
- `GET /collections/{id}/join/{other}?epsilon=&limit=&count=`
- `GET /collections/{id}/anti-join/{other}?epsilon=&limit=&count=`
- `GET /collections/{id}/components?epsilon=&count=`

`/search` is the artifact-native endpoint; it works for every payload kind
(`none`, `row_ref`, `row_wkb`, `feature_json`) and returns a JSON envelope with
a `matches` array. `/items` is the GeoJSON view: it returns a
`FeatureCollection` and requires a `feature_json` payload; other artifacts get
a 422 pointing at `/search`. `/items` also rejects `/search`-only options
(`level`, `payload`, `identity`, `count`, `frustum`) with `unsupported_query`.

Failures are reported as `{"error":{"code","message"}}`; the two Cloudflare
Worker demos under `wasm-demo/` use the same envelope and code vocabulary, so a
client written against one reads the others without special cases.

Unknown query parameters and unknown catalog keys are rejected (`invalid_query`
and a startup error) rather than ignored, so a misspelled name fails loudly
instead of silently resolving to a default. This applies to the two endpoints
that take parameters; `/collections` and `/collections/{id}` take none, so
there is nothing a typo could quietly become and they ignore the query string
rather than refusing an incidental `?f=json`.

Query parameters:

- `bbox` — 4 numbers for 2D artifacts, 6 for 3D. Required unless another
  query shape (`frustum`, `radius`, `polygon`) is given.
- `frustum=a,b,c,d,...` — `/search` only, **3D artifacts only**, mutually
  exclusive with `bbox`. Six inward-pointing planes as 24 numbers: a point is
  inside a plane when `a*x + b*y + c*z + d >= 0`, and inside the frustum when it
  is inside all six. The planes need not be normalized — only the sign is used.

  This is what a tilted or perspective view should send. The axis-aligned box
  around a frustum is far larger than the frustum, so a client that can only
  send a `bbox` pays reads for geometry it will never draw; measured on 200k
  boxes, the frustum returns 1.7–4x fewer candidates and answers 3.2–13.6x
  faster than the corner-box workaround.

  Planes rather than a view-projection matrix, deliberately. A matrix carries
  two conventions the wire cannot recover — the clip-space depth range (which
  this project refuses to default silently, because it is not derivable from the
  matrix) and row- versus column-major storage — and getting either wrong moves
  the near plane without failing anywhere. A client resolves both locally, where
  it knows the answer, with `Frustum3D::from_view_projection`, and sends the
  planes.

  The query is a **conservative superset**: it returns every box overlapping the
  frustum and may include a few just past an edge or corner (the standard
  p-vertex test), and it never drops a visible one. There is no exact narrow
  phase for a frustum in this crate, so `predicate=intersects` is refused with
  it; do the precise test on the returned candidates. `count=only` works.

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
  returned page to include it. Like `level`, it is resolved against the
  collection and the rest of the request rather than taken literally, and the
  echoed value is the mode that applied:
  - a collection that stores no source id resolves to `ref` whatever was asked,
    so `full` never buys a page of reads for an identical answer. `full` is
    still accepted — a client querying a mixed catalog should not have to vary
    its request per collection — and `capabilities.identityModes` lists it only
    where it can add something. Only `feature_json` bodies have room for an id,
    and only GeoJSON sources supply one, so a `feature_json` artifact built
    from Parquet or FlatGeobuf lands here too.
  - `payload=full` resolves to `full`, because the bodies are read regardless
    and the returned GeoJSON feature carries its own `id` anyway. Withholding
    it from `featureRef` there would hide nothing.
- `count=records|only` — `/search` only; default `records`. `only` answers
  `numberMatched` and nothing else: `matches` is empty, `numberReturned` is 0,
  and the index counts the matches without materializing one of them, which is
  what a "how many are in this bbox" caller actually wants. Two cases are
  refused with `unsupported_query` rather than answered with a number that
  means something else, and `capabilities.countModes` says so in advance:
  - `predicate=intersects`, because exact filtering narrows the match set
    *after* the index answers, so the index count is an upper bound.
  - `level=feature` on a collection whose entries can duplicate a source row
    (a split antimeridian geometry, a multi-part feature). Collapsing entries
    to features means reading the matches, which is the work this mode exists
    to skip; `level=entry` counts fine there. Where entries never duplicate a
    row, feature and entry counts are the same number and both are allowed.
- `radius=lon,lat,metres` — `/search` and `/items`, **2D artifacts only**, one
  query shape per request. A spherical cap: degrees on a sphere and metres on
  its surface. The artifact's coordinates are whatever its source used, so this
  is meaningful for lon/lat data and nonsense for a projected artifact — the
  same contract `gp2psindex query --radius` has.

  A radius is **always exact**. The index can only answer with the boxes
  covering the cap, so the distance test runs against source geometry
  afterwards, exactly as `predicate=intersects` does — which is why it needs an
  artifact with a geometry payload (`capabilities.queryShapes` lists `radius`
  only where that holds), and why `count=only` refuses it: the index count
  would not be the number the caller sees.
- `polygon=[[[[x, y], ...], ...], ...]` — `/search` and `/items`, **2D
  artifacts only**, one query shape per request. GeoJSON MultiPolygon
  coordinates: ring 0 of each polygon is its exterior, the rest are holes. It
  is the array this crate's own `GeoQuery2D` serializes, so a client sends its
  geometry's `coordinates` member verbatim.

  Unlike a radius, a polygon is **not** an exact query: the polygon drives the
  index traversal itself, pruning subtrees that fall outside it rather than
  fetching everything in its bounding box. So it needs no payload, every 2D
  artifact takes one, and `count=only` works over it. `predicate=intersects`
  still refines the surviving candidates against real source geometry.
- `nonplanar=reject|treat_as_planar` — default `reject`. What exact filtering
  does when the artifact's declared edge model cannot answer the query it was
  given. The case that matters: a spherical radius against a column declaring
  **planar** edges, which is what GeoJSON and GeoParquet without an `edges`
  member declare — that is most real data. Rejecting is right by default, since
  the answer would be to a different question, and useless as the only option,
  so `treat_as_planar` reads the stored coordinates as planar XY for the
  predicate. `gp2psindex query` exposes the same choice as
  `--treat-nonplanar-as-planar`. Only an exact query consults it, and responses
  echo it only when one ran.
- `limit`, `offset` — pagination over the matched set. They do not change
  `numberMatched`, so they have no effect under `count=only`.

### POST /search

A polygon large enough to outgrow a URL goes in a body instead:

```
POST /collections/places/search
Content-Type: application/json

{"polygon": [[[[-1,-1],[41,-1],[-1,41],[-1,-1]]]], "limit": 100, "level": "entry"}
```

The field names are the query parameters, typed rather than stringly (`bbox` is
an array of numbers, `limit` a number). The body converts into the same
parameters and takes the identical path from there, so what a search *means*
has one implementation and cannot drift between the two transports.

GET stays the canonical form because it is the cacheable one — a query in a URL
is served by a CDN, bookmarked, logged whole — which is the point of putting a
static artifact behind one. POST is the escape hatch, not the upgrade.

On POST the body is the whole request and **the query string must be empty**
(`invalid_query` otherwise). Merging the two would mean deciding which side
wins per parameter, and every parameter added later would inherit that
question. Unknown fields are refused like unknown parameters.

Responses echo the effective query (after defaults and collection-dependent
resolution) under `query`, so a client can always see which `level`,
`identity`, `count`, and `predicate` actually applied, and `query` carries
exactly one of `bbox`, `frustum`, `radius` or `polygon` — the shape the search
used. `numberMatched`
counts matches before pagination and `numberReturned` after. Each match
carries `entryId` (index entry ordinal in the artifact; stable per artifact
build, not across rebuilds) and, when the payload stores one, a `featureRef`
back to the source feature.

Collection metadata reports the artifact `payloadKind` plus a `capabilities`
object listing the accepted `predicates`, `levels`, `payloadModes`, and
`identityModes`, and whether `/items` is available.

## Distance joins

Three endpoints over one idea: the *ε-proximity graph*, whose nodes are index
entries and whose edges join two entries whose boxes lie within `epsilon` of
each other. The distance is box-to-box Euclidean in the artifact's coordinate
units — zero when the boxes overlap, inclusive at the bound, so `epsilon=0`
asks the plain overlap question. `epsilon` is required on all three and must be
a finite non-negative number (`invalid_epsilon` otherwise). Both collections
must be the same dimensionality; 2D against 3D is `unsupported_query` (422).

Like every query here these are a broad phase: the box distance is a lower
bound on the true distance between the underlying geometries, so results are
candidates and the exact predicate stays with the caller.

### GET /collections/{id}/join/{other}

The edges. Every pair of entries within `epsilon`, as `pairs: [{a, b}]` where
`a` is an entry ordinal in `id` and `b` one in `other`. Joining a collection
with itself reports each unordered pair once.

```
GET /collections/towers/join/cables?epsilon=500&limit=100
```

`limit` truncates the returned array; `numberMatched` always reports the true
total, because the traversal runs to completion either way. `count=only`
returns the total with no pairs. There is no `offset`: the pair stream has no
resumable cursor, so a later page would cost a full rerun — shrink `epsilon`
instead.

### GET /collections/{id}/anti-join/{other}

The complement: entries of `id` with *no* entry of `other` within `epsilon`,
as `items: [ordinals]`. "Which towers no cable comes near." `limit` and
`count` behave exactly as on `/join`.

`other` may not equal `id` (422). Against itself every entry is at distance
zero from itself, so the literal answer is always empty; the question people
mean there — entries with no *other* entry nearby — is what `/components`
answers, where an isolated entry is its own label. The endpoint says so rather
than quietly answering a different question for one case.

### GET /collections/{id}/components

The connected components of one collection's own graph — there is no `{other}`
segment, because a component is a property of a single graph. Every entry gets
a label: the smallest entry ordinal in its component. An entry with no
neighbour is its own label.

```
GET /collections/towers/components?epsilon=50
{"collectionId":"towers","epsilon":50.0,"count":"records",
 "itemCount":4,"componentCount":2,"labels":[0,0,0,3]}
```

**The labels identify components; they are not clusters.** Distance proximity
is not transitive — a chain of entries each within `epsilon` of the next is one
component no matter how far its ends lie apart — so this reports exactly what
the graph defines, and whether a chained component should stay merged is the
caller's decision.

There is no `limit`: `labels` has one entry per index entry by definition, so a
truncated labelling would not be a labelling of anything. A large collection
therefore returns a large body; `count=only` returns `componentCount` alone
when that is all you need.
