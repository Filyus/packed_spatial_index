# GeoPSINDEX Cloudflare Worker + R2 feature API

End-to-end demo for the main static serving story:

```text
GeoParquet -> gp2psindex -> immutable .psindex in R2 -> HTTP feature/search API
```

The Worker never reads back the source GeoParquet and never talks to a database.
It streams byte ranges from one `cities.psindex` R2 object, caches the parsed
GeoPSINDEX directory in the warm isolate, and serves GeoJSON features directly
from the artifact's embedded `feature-json` payload.

This example is geo-first and intentionally separate from
[`../worker`](../worker), which remains the low-level core `PSINDEX` range-read
demo.

## Endpoints

- `GET /health`
- `GET /collections`
- `GET /collections/cities`
- `GET /collections/cities/search?bbox=minx,miny,maxx,maxy&limit=&offset=&payload=none|summary|full&level=entry|feature`
- `GET /collections/cities/items?bbox=minx,miny,maxx,maxy&limit=&offset=`

`/search` returns an artifact-native envelope with `numberMatched`,
`numberReturned`, `query`, `payloadKind`, and `matches`. `/items` returns a
GeoJSON `FeatureCollection`. Search is bbox-only in this milestone; exact
predicate/source read-back is deliberately left to the native server.
For artifacts whose manifest says entries cannot duplicate rows, summary search
uses an entry/rank header path and omits `featureRef`; request `payload=full` or
use `/items` when the client needs the returned GeoJSON features.

Every R2-backed response includes `X-PSI-Reads` and `X-PSI-Bytes`. Search and
items responses also include `reads`, `bytes`, and `ms` in the JSON body.

## Local build

```sh
npm install
npm run build:wasm
npm run seed:geo
npm run typecheck
```

`seed:geo` writes:

- `cities.parquet`: deterministic clustered GeoParquet from `../geo-seed`
- `cities.psindex`: `gp2psindex build --payload feature-json --properties all`

The wasm module depends on `packed_spatial_index_geo` with
`default-features = false, features = ["async"]`, so it keeps Arrow/Parquet out
of the Worker. The conversion CLI still uses the full geo crate locally.

## Deploy

```sh
# one-time auth, or set CLOUDFLARE_API_TOKEN
wrangler login

npm run bucket:create      # ok if the bucket already exists
npm run upload             # uploads cities.psindex to psi-geo-demo/cities.psindex
npm run deploy
```

Defaults:

- Worker: `psi-geo-r2-demo`
- R2 bucket: `psi-geo-demo`
- Object key: `cities.psindex`
- Collection id: `cities`

## Live smoke

```sh
npm run smoke:live -- https://psi-geo-r2-demo.<your-subdomain>.workers.dev
```

The smoke script checks `/health`, `/collections`, `/search`, and `/items` with
a deterministic bbox around one seed-data cluster:

```text
bbox=64,23,71,29
```

Copied response from a deployed Worker:

```json
{
  "collectionId": "cities",
  "query": {
    "bbox": [64, 23, 71, 29],
    "predicate": "bbox",
    "level": "feature",
    "payload": "summary",
    "limit": 3,
    "offset": 0
  },
  "payloadKind": "feature_json",
  "numberMatched": 553,
  "numberReturned": 3,
  "matches": [
    {
      "entryId": 169,
      "payload": {
        "kind": "feature_json"
      }
    }
  ],
  "reads": 1,
  "bytes": 7360,
  "ms": 122
}
```

The same deployed Worker also handles a world-sized bbox without reading all
GeoJSON bodies:

| query | matched | returned | reads | bytes | ms |
|---|---:|---:|---:|---:|---:|
| `/search?bbox=-180,-90,180,90&limit=3&payload=summary` | 100000 | 3 | 1 | 800008 | 36 |
| `/items?bbox=-180,-90,180,90&limit=3` | 100000 | 3 | 6 | 972629 | 112 |

The exact `reads`, `bytes`, and `ms` vary with cold/warm isolate state, but the
important proof is stable: a public HTTP API can serve feature results from a
single immutable R2 object with bounded range reads and no database.
