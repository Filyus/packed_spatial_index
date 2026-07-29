# Payload prefix scans: why a header search costs one read per match

Some queries need only the **first few bytes** of every matching payload, not the
whole body. The geospatial layer is the motivating case: it prefixes each payload
blob with a fixed 24-byte feature reference, and a "header search" decodes just
that prefix for every match so a caller can page, deduplicate, or count without
materializing megabytes of GeoJSON.

That scan is served by `visit_payload_prefixes` (and its `_async` twin), and on a
local file it behaves exactly as intended. Over object storage it does not: it
issues **one range request per match**. This note explains why, why the two
obvious knobs do not fix it, and what does.

The numbers below are all measured, on a 100 000-point GeoPSINDEX artifact with
embedded GeoJSON bodies (~370 B each, 40 MB total), through a counting
`AsyncRangeReader`. "small bbox" matches 553 items, "world bbox" matches all
100 000. Reproduce them by pointing a counting reader at any artifact built with
`gp2psindex build --payload feature-json` and calling
`search_match_headers_page_async`.

## The shape of the problem

The payload section stores blobs **concatenated in leaf rank order**, preceded by
a `(num_items + 1)` offset table (see [FORMAT.md](../../FORMAT.md)). A spatial
query visits leaves in contiguous runs, so both of these are cheap to fetch:

- the **offset table** — a dense `u64` array; a run of matching ranks is one
  contiguous slice;
- **whole bodies** — a run of matching ranks is one contiguous byte span.

A prefix scan is neither. It wants 24 bytes at every blob start, and blob starts
are ~370 B apart. The bytes it needs are *strided*, not contiguous, and the
strided reads never coalesce:

```rust
// src/stream/core.rs and src/stream/async_io.rs
let gap = self.coalesce_gap().min(prefix_len as u64);
```

Two prefix spans merge into one read only when the bytes between them fit in
`gap`. Clamping `gap` to `prefix_len` means "never skip more than one prefix
worth of bytes" — with 370-byte bodies the distance between consecutive prefixes
is always larger, so nothing ever merges.

| path | small bbox | world bbox |
| --- | --- | --- |
| `search_payload_headers_page` (offsets only, `prefix_len = 0`) | 1 read, 7 360 B | 1 read, 800 008 B |
| body fetch for one page of 3 | 3 reads, 156 539 B | 6 reads, 972 629 B |
| **`search_match_headers_page` (`prefix_len = 24`)** | **554 reads**, 20 632 B | **100 001 reads**, 3 200 008 B |

One read per match. On a local file that is 554 `pread` calls and nobody notices.
On R2 or S3 it is 554 HTTP round trips and 554 billed operations for 20 KB of
data.

## Why the clamp is there, and why raising it does not help

The clamp is a deliberate choice, not an oversight: it guarantees a prefix scan
never reads more than `2 × prefix_len` bytes per match, so asking for 24 bytes
cannot accidentally pull a megabyte body along with it. For a local file that is
the right trade — reads are nearly free, bytes are not.

Raising it trades the axis the other way, and the trade is worse than it looks
because there is no useful middle. Sweeping `gap = min(coalesce_gap, prefix_len × K)`:

| `K` (gap = 24·K) | small bbox reads | small bbox bytes |
| --- | --- | --- |
| 1 (today) | 554 | 20 632 |
| 4 | 554 | 20 632 |
| 16 | 7 | 201 701 |
| 64 | 6 | 202 738 |
| 1024 | 3 | 215 093 |
| unclamped (256 KiB) | 2 | 333 300 |

The knee is at the body stride. Below it nothing merges; above it *everything*
merges, because merging two prefixes means reading the body between them, and
once you read one body you have already paid for the next gap too. There is no
setting that reads 553 prefixes in a few reads without also reading the 553
bodies they are embedded in.

And on a wide query the merged form is worse than either alternative: the world
bbox unclamped reads **36.7 MB** — almost the entire artifact — to collect 2.4 MB
of prefixes. 100 001 reads is unusable; 36.7 MB is unusable. A prefix scan over
many matches is simply expensive today, in one currency or the other.

The cheap path in the table above is cheap for a reason worth stating plainly:
`prefix_len = 0` touches no blob bytes at all, only the offset table. It is fast
*because it returns no feature references*. Callers that need identity cannot use
it.

## The fix: a contiguous prefix section

The prefixes are fixed-size and rank-indexed. Every other fixed-size rank-indexed
array in this format — the offset table, the tree's SoA `indices`, a fixed-width
`PYLD` — is stored contiguously, which is exactly why those are cheap to scan. The
prefixes are the one such array that is *interleaved into variable-length data*.
Storing a copy of them contiguously removes the striding, and with it the whole
trade-off.

### Layout

A new optional chunk, tag from the format-reserved uppercase space (`PFIX`), with
a descriptor deliberately shaped like `PYLD`'s so the parse path is the same:

```text
0       4     desc_len       u32 = 12
4       1     ordering       u8 = 0 (leaf rank)
5       1     compression    u8 = 0 (none)
6       2     reserved       (zero)
8       4     record_stride  u32, bytes per item (24 for a geo feature ref)
--- then num_items * record_stride bytes, leaf rank order
```

The content is `min(record_stride, payload_len)` bytes copied from the head of
each blob, zero-padded when a blob is shorter. It duplicates bytes that remain in
the bodies; it does **not** relocate them. That matters for compatibility (below)
and costs `record_stride` bytes per item — 2.4 MB on the 40 MB artifact above,
about 6%.

Rank alignment is free: the writer already receives `leaf_order` and writes
`PYLD` from it, so a second section built in the same pass is aligned with `PYLD`
by construction.

### Reader

`visit_payload_prefixes` gains one branch: when a `PFIX` section is present and
`prefix_len <= pfix.record_stride`, read the prefixes from it instead of from the
blobs. Everything downstream is unchanged — the visitor still emits
`PayloadPrefix { id, leaf_rank, prefix, payload_len }`, and `payload_len` still
comes from the offset table, which the scan already reads.

`StreamCoreParts` gains an `Option<PrefixSection>` alongside `payload`; the open
path reads one more descriptor. The directory node budget is unaffected — it
covers tree nodes only.

### Compatibility

Strictly additive, in both directions, with no `format_version` bump. FORMAT.md
already commits to this: *"A reader rejects a file that contains a critical chunk
whose tag it does not understand, and silently skips an optional chunk it does
not understand. New chunk types can therefore be added without breaking older
readers, as long as they are marked optional."*

- **Old reader, new artifact** — skips `PFIX`, reads prefixes from the blobs
  exactly as today. Correct, just not faster.
- **New reader, old artifact** — no `PFIX` section, falls back to the blob scan.
  Correct, just not faster.

Because the prefix bytes stay in the bodies, nothing that decodes a body changes:
`has_feature_ref_prefix` and `feature_json_body` in the geo layer keep working on
old and new artifacts alike. Relocating the prefixes instead of copying them
would save the 6% and break exactly this — an old reader would see a body with no
prefix and silently treat it as a reference-less legacy payload.

Existing artifacts do not gain the section until they are rebuilt. No migration
tool is needed: this is an optimization, not a semantic change.

### Measured, not projected

A `--payload row-ref` artifact has uniformly 24-byte bodies, so its `PYLD` blobs
*are* a contiguous rank-indexed 24-byte array — the same thing `PFIX` would be. A
prefix scan over one therefore measures the proposal directly:

| | reads | bytes |
| --- | --- | --- |
| today, `feature-json` bodies | 554 | 20 632 |
| contiguous section, small bbox | **2** | 29 416 |
| today, world bbox | 100 001 | 3 200 008 |
| contiguous section, world bbox | **2** | 3 200 008 |

The byte counts barely move — the same 24 bytes per match are needed either way.
Only the read count collapses, by two to five orders of magnitude. Unlike raising
the coalescing gap, this is not a trade: it is strictly better on both axes than
the clamped scan on wide queries, and costs 43% more bytes than the clamped scan
on narrow ones while cutting 554 requests to 2.

## What it unblocks

- A remote reader can return feature references from a summary search at the same
  cost as returning none. Today the geo Worker demo avoids the header path
  precisely to avoid the read storm, which leaves its `/search` returning records
  without `featureRef` — a contract divergence from the native server that exists
  for no reason other than this.
- The native server's paged search stops being local-file-only in practice.
- `count`-style queries that need identity (dedupe by feature) become viable over
  object storage.

## Adjacent, cheaper win

The geo layer always writes variable-width payloads, including for
`PayloadPlan::RowRef`, whose bodies are uniformly 24 bytes. Those artifacts could
use fixed-width `PYLD` (`record_stride = 24`) instead: it drops the
`(num_items + 1) × 8` offset table entirely and makes the prefix scan contiguous
for free, with no new chunk. It only helps `RowRef`, which is the payload kind
that needs prefix scans least — but it is a few lines and shrinks those artifacts
by 800 KB per 100 000 items.

## Non-goals

- Compressing the section. `compression = 0` is the only accepted value in
  `PYLD`'s descriptor today; keeping `PFIX` symmetric leaves the door open
  without opening it.
- Making `PFIX` critical. Nothing in it cannot be recovered from the bodies.
- Changing the clamp. With `PFIX` present the clamped blob scan is a fallback
  path, and its current value is right for the local files it still serves.
