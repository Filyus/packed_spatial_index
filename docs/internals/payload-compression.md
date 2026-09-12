# Payload compression: what is worth a format revision

The `PYLD` and `PFIX` descriptors both reserve a `compression` byte at `0`
(none). Two candidates were ranked as worth building — an Elias-Fano offset
table, and byte-stream-split for float payload columns. Neither had a number
attached, and the byte they would claim is a one-way door, so both were measured
before any format code was written.

Short answer: **the offset table is worth compressing exactly when payloads are
small, and byte-stream-split is not worth building at all.**

## The offset table

Variable-width `PYLD` stores `(num_items + 1) x u64` prefix offsets ahead of the
blobs. The table costs 8 bytes per item and the blobs cost whatever they cost, so
the table's share of the payload section is

```
table / PYLD  =  8 / (8 + mean_blob_bytes)
```

and nothing else — not the item count, not the geometry type, not the extent.
That single identity is most of the analysis, and it is why no large corpus was
needed to settle it.

Elias-Fano stores a monotone sequence of `n` values over a universe `U` in about
`n * (2 + floor(log2(U / n)))` bits. Here `U / n` is the mean blob size, so each
offset costs about `2 + log2(mean_blob)` bits instead of 64 — an 83–91% cut of
the table across every realistic blob size. What changes with the data is not how
well the table compresses but how much of the file the table was:

| mean blob | table / PYLD | Elias-Fano bits/offset | saved, of PYLD |
| ---: | ---: | ---: | ---: |
| 21 B | 27.6% | 6 | 25.0% |
| 40 B | 16.7% | 7 | 14.8% |
| 100 B | 7.4% | 8 | 6.5% |
| 250 B | 3.1% | 9 | 2.7% |
| 1 000 B | 0.8% | 11 | 0.7% |
| 4 000 B | 0.2% | 13 | 0.2% |

Measured against the WKB payloads of every corpus in `dev/`, which lands where
the identity says it should:

| corpus | items | mean blob | table / PYLD | saved |
| --- | ---: | ---: | ---: | ---: |
| `natural-earth_cities` (points) | 243 | 21 B | 27.7% | 25.1% |
| `natural-earth_countries-bounds` | 177 | 94 B | 7.9% | 6.9% |
| `geospatial.parquet` | 164 | 100 B | 7.5% | 6.5% |
| `natural-earth_countries-geography` | 177 | 978 B | 0.8% | 0.7% |

So the verdict splits on geometry, and sharply. A point corpus — a WKB point is
21 bytes, and points are the single most common payload anyone indexes — spends
**over a quarter** of its payload section on the offset table, and Elias-Fano
gets essentially all of it back. A polygon corpus spends under a percent, and
compressing it would be a rounding error dressed up as a format revision.

This also says something the ranking did not: the feature that competes with
Elias-Fano here is the one already shipped. Fixed-width `PYLD`
(`record_stride > 0`) drops the offset table entirely, and a point payload is
fixed-width by construction. Elias-Fano earns its byte on *variable*-width small
blobs — mixed points and short lines, or points carrying a short attribute tail —
rather than on points alone.

## Byte-stream-split

BSS transposes the `K` bytes of each float into `K` byte planes so a downstream
compressor sees each exponent and mantissa byte in its own run. It shrinks
nothing by itself, so it is only meaningful paired with zstd, and it applies only
to a fixed-stride float payload — which in this crate means triangle meshes
(72 B per `f64` triangle in 3D, 36 B in `f32`).

Measured on 200 000 triangles in leaf (Hilbert) order, zstd level 9, as the
ratio of `zstd(BSS(x))` to `zstd(x)` — above 1.0 means BSS made the file bigger:

| mesh | zstd alone (f64) | BSS / zstd (f64) | BSS / zstd (f32) |
| --- | ---: | ---: | ---: |
| structured grid | 0.110 | **4.15** | 2.39 |
| irregular, full precision | 0.946 | 0.85 | 0.83 |
| irregular, rounded to 1 mm | 0.388 | **1.85** | 0.88 |

The one shape BSS helps is the one where compression was already pointless:
irregular full-precision coordinates are near-incompressible (zstd 0.946), and
BSS takes 15% off a stream that was not shrinking anyway. Everywhere zstd
actually works — structured meshes, or any mesh whose authoring tool rounded
coordinates to a millimetre, which is most of them — BSS destroys the whole-record
literal matches zstd was living on and costs 1.9–4.1x.

And the row that ends the argument is the third one: rounding coordinates to
1 mm takes plain zstd from 0.946 to 0.388. Quantizing the data is worth about
2.4x, against BSS's best case of 1.17x on data nobody should be storing at full
precision. If mesh payload size ever becomes a complaint, the answer is to
quantize, not to transpose.

**Byte-stream-split is rejected.** Not deferred: it was measured across three
data shapes and two precisions, and it loses on every shape where the result
would matter.

## Reproducing

The two scripts are arithmetic and a compressor, not benchmarks — no pinning, no
interleaving, nothing timed. The offset-table table comes from the closed form
above plus `pyarrow` over `dev/**/*.parquet`; the BSS table generates its meshes,
sorts them by the Hilbert index of the triangle centroid (the order the format
stores), and compresses both layouts with `zstandard` at level 9.

Both scripts are parked on `probe/payload-compression-numbers` rather than
carried on `main`: neither needs to run again unless the claim is challenged (the
offset-table result is a closed form and the BSS result is a rejection), but the
next person to doubt either should not have to rebuild them.
