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
the bodies; it does **not** relocate them (see [Copy, not move](#copy-not-move)).

Rank alignment is free: the writer already receives `leaf_order` and writes
`PYLD` from it, so a second section built in the same pass is aligned with `PYLD`
by construction.

## When it is worth it: the body-size cliff

The pathology is not "prefix scans are strided" — it is "the stride is wider than
the gap", and that has an exact boundary. Two consecutive matching ranks merge
when the bytes between them fit in `gap = prefix_len`; the distance from one
prefix's end to the next prefix's start is `body - prefix_len`. So they merge iff

```text
body <= 2 * prefix_len
```

which is **48 bytes** for a 24-byte feature reference. It is a cliff, not a slope.
Sweeping body size over 100 000 items with a query matching 1 001 consecutive
ranks, prefix scan only:

| body | reads | bytes | a `PFIX` section would cost |
| --- | --- | --- | --- |
| 40 B | 11 | 355 188 | 26.5% of the file |
| 48 B | 11 | 363 188 | 24.3% |
| **49 B** | **1 011** | 339 188 | 24.1% |
| 64 B | 1 011 | 339 188 | 20.9% |
| 128 B | 1 011 | 339 188 | 13.4% |
| 370 B | 1 011 | 339 188 | 5.7% |
| 4096 B | 1 011 | 339 188 | 0.6% |

One byte of body turns 11 reads into 1 011.

### Which real payloads land where

WKB sizes are fixed, so this is arithmetic, not estimation. With the 24-byte
feature-ref prefix:

| payload | body | side of the cliff |
| --- | --- | --- |
| `row-ref` | 24 B | below |
| `row-wkb`, 2D point | 45 B | below, **by 3 bytes** |
| `row-wkb`, 3D point | 53 B | above |
| `row-wkb`, 2-vertex 2D line | 65 B | above |
| `row-wkb`, 5-vertex 2D polygon | 117 B | above |
| `feature-json` | ≥ ~100 B, typically 300+ | far above |

So the story is not "`feature-json` is the odd one out". `feature-json` is merely
*always* above the cliff; `row-wkb` is above it for everything except a 2D point.
The reassuring `row-wkb` measurement in the geo demo is an accident of its seed
being 2D points — a demo of 3D points, or of any line or polygon, would show the
same read storm. Below the cliff there are only bare references.

`FeatureJson` is not a test-only payload kind, either: it is what `/items` and
every GeoJSON response in the geo layer are built on, and it is the plan the
worker demo ships. It is simply the plan whose bodies are unambiguously fat.

### Two tools, not one

Above the cliff there are two ways out, and they suit different body sizes.

**Lift the clamp** (a reader-side knob, no format change). Merging two prefixes
means reading the body between them, so this costs about `body` bytes per match
instead of `prefix_len`. Same sweep, clamp lifted:

| body | reads | bytes | vs clamped |
| --- | --- | --- | --- |
| 64 B | 11 | 379 188 | +12% bytes, 92× fewer reads |
| 128 B | 11 | 443 188 | +31% bytes, 92× fewer reads |
| 370 B | 11 | 685 188 | +102% bytes |
| 4096 B | 11 | 4 411 188 | +1200% bytes |

For a small body this is an excellent trade and costs nothing on disk. For a fat
one it degenerates into reading the whole payload region — the 36.7 MB world
query from the opening section.

**Write the section.** Constant `prefix_len` bytes per match at any body size, at
the price of `prefix_len / body` file growth: 24% at the cliff, 5.7% at 370 B,
0.6% at 4 KB.

The two costs are reciprocal, which makes the boundary easy to state. Let
`r = body / prefix_len`:

- lifting the clamp wastes about `r ×` the query bytes and 0% of the file;
- the section wastes 0 query bytes and about `1/r` of the file.

They cross at `r = 1`, but neither cost matters much while it is small, so the
useful rule is a band rather than a point. Taking 10% as "small":

| regime | `r` | body (24-byte prefix) | tool |
| --- | --- | --- | --- |
| already fine | ≤ 2 | ≤ 48 B | nothing |
| small bodies | 2 – 10 | 49 – 240 B | lift the clamp |
| fat bodies | > 10 | > 240 B | write the section |

In the middle band, over-reading wastes at most 10× on query bytes and those
bytes are small in absolute terms; above it, the section costs under 10% of the
file and nothing per query.

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

Existing artifacts do not gain the section until they are rebuilt. No migration
tool is needed: this is an optimization, not a semantic change.

### Copy, not move

The obvious objection to copying is that it duplicates 24 bytes per item that are
already in the bodies. Why not relocate the prefixes into `PFIX` and leave the
bodies as bare WKB or bare GeoJSON?

The tempting answer — "because an old reader would see a body with no prefix and
silently treat it as a reference-less legacy payload" — is true but not decisive,
because it is fixable. Mark `PFIX` **critical** and an old reader rejects the file
outright instead of misreading it; that is precisely what the critical bit is for.
Silence is a property of moving prefixes under an *optional* chunk, not of moving
them.

The decisive reasons are the other three:

- **The win is small and one-dimensional.** Moving saves `record_stride` bytes per
  item, which is 5.9% of a `feature-json` artifact and nothing at all for the
  payload kinds that never get a section. It saves no reads: the body fetch spans
  the same runs either way.
- **Bodies stop being self-describing.** Today a blob alone recovers its feature
  reference — `has_feature_ref_prefix`, `decode_feature_ref_payload`,
  `feature_json_body` and `stamp_payload_part` all work on the blob and nothing
  else. After a move, every decode needs a rank-join against `PFIX`. And the
  blob-prefix path cannot be deleted anyway, because artifacts built before the
  change still have to load — so the cost is two decode paths, permanently, in
  exchange for 5.9%.
- **It changes the kind of change this is.** Copying is additive: nothing that
  reads today stops reading. Moving makes every artifact the new writer produces
  unreadable by every older reader. That is a reasonable thing to do for a
  reasonable price, and 5.9% of one payload kind is not one.

The conditional worth recording: **if a format break is happening anyway** — a
`format_version` bump, or a new critical chunk introduced for some other reason —
then moving is the right design and copying becomes waste. Revisit this then, not
before.

### Measured, not projected

A `--payload row-ref` artifact has uniformly 24-byte bodies, so its `PYLD` blobs
*are* a contiguous rank-indexed 24-byte array — the same thing `PFIX` would be. A
prefix scan over one therefore measures the proposal directly. The clamp does not
apply to a dedicated section (there are no bodies to over-read into, so the gap is
the reader's ordinary `coalesce_gap`), so the figures below are that scan with the
clamp lifted — 7 reads become 2:

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

## Who decides: an option below, automatic above

Both tools need a policy, and the two layers know different things.

**The core cannot decide, and should not guess.** `PYLD` blobs are opaque bytes;
nothing in this crate knows that the first 24 of them mean anything. So both knobs
are explicit here:

- a serializer option — `payload_prefix_len(n)` — that writes the section. Absent,
  no section, exactly as today.
- a `StreamLimits` field for the prefix coalescing gap, defaulting to today's
  `prefix_len`. It belongs with the other limits because it is a property of the
  *reader's* storage, not of the data: 1 011 range requests is a catastrophe over
  R2 and a non-event on a local file. The same artifact wants different answers in
  the server and in the Worker.

**The geo layer can decide, and should.** It knows `prefix_len` is always 24 and
it holds every body at write time, so it can measure instead of guess. The rule
should read the body distribution, **not the payload plan** — the table above is
the reason: `RowWkb` straddles the cliff, so `--payload row-wkb` predicts nothing.
A median body size, compared against the two thresholds, predicts exactly.

Default it to automatic, with an explicit override (`--prefix-index auto|on|off`),
because the two error directions are not symmetric. Emitting a section that was
not needed costs under 10% of file size, once. Not emitting one that was needed
costs one range request per match, on every query, forever — and the failure is
invisible at build time and only shows up as a bill.

The one thing automatic must not do is surprise someone building a local-only
artifact with a 24% size increase, which is exactly what the `r > 10` threshold
is there to prevent: below it the answer is a reader-side knob that costs no
bytes on disk at all.

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
by 800 KB per 100 000 items. It is also the case where the *whole payload* is the
prefix, so it is the one payload plan for which a `PFIX` section would be pure
duplication and must never be emitted.

## Non-goals

- Compressing the section. `compression = 0` is the only accepted value in
  `PYLD`'s descriptor today; keeping `PFIX` symmetric leaves the door open
  without opening it.
- Making `PFIX` critical. Nothing in it cannot be recovered from the bodies.
- Changing the clamp's *default*. It is right for the local files it was written
  for, and for every artifact below the cliff. Making it configurable is the
  middle band's whole answer; moving the default is not.
