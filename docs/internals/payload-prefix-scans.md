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

This is what shipped, as the optional `PFIX` chunk (format revision 13, no
`format_version` bump). The prefixes are fixed-size and rank-indexed. Every other fixed-size rank-indexed
array in this format — the offset table, the tree's SoA `indices`, a fixed-width
`PYLD` — is stored contiguously, which is exactly why those are cheap to scan. The
prefixes are the one such array that is *interleaved into variable-length data*.
Storing a copy of them contiguously removes the striding, and with it the whole
trade-off.

### Layout

An optional chunk, tag from the format-reserved uppercase space (`PFIX`), with a
descriptor deliberately shaped like `PYLD`'s so the parse path is the same:

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

`visit_payload_prefixes` has one extra branch: when a `PFIX` section is present
and `prefix_len <= pfix.record_stride`, it reads the prefixes from there instead
of from the blobs. Everything downstream is unchanged — the visitor still emits
`PayloadPrefix { id, leaf_rank, prefix, payload_len }`, `payload_len` still comes
from the offset table the scan already reads, and the four public
`visit_payload_prefixes*` entry points did not change at all. They just got
faster.

The section lives on `StreamCoreParts`, not only on `StreamCore`, so a directory
split off with `into_directory` and reattached to a fresh reader keeps it — the
warm-isolate path is exactly the one the section exists for. The open path costs
one more descriptor read; the directory node budget is unaffected, since it
covers tree nodes only.

Coalescing inside the section uses the ordinary record gap rather than the prefix
gap: what lies between two prefixes there is other prefixes, not bodies, so
merging over them is cheap by construction.

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

### What it actually did

The same 100 000-point GeoParquet source, converted twice — once with the section
and once with `--prefix-index off` — and scanned through a counting reader:

| bbox | | reads | bytes |
| --- | --- | --- | --- |
| small (553 matches) | without | 554 | 20 632 |
| small | **with** | **2** | 29 416 |
| world (100 000 matches) | without | 100 001 | 3 200 008 |
| world | **with** | **2** | 3 200 008 |

The read count collapses by two to five orders of magnitude. The byte counts
barely move — the same 24 bytes per match are needed either way — so unlike
raising the coalescing gap this is not a trade: it is strictly better on both
axes on a wide query, and costs 43% more bytes on a narrow one while cutting 554
requests to 2. The artifact grew from 40 927 824 to 43 327 864 bytes, **+5.9%**,
exactly the 24 bytes per entry the section is.

Compatibility was checked in both directions with a binary built from the commit
before this work: it reads a section-carrying artifact and returns byte-identical
query output, and the new reader falls back to the blob scan on an artifact
without one.

## Who decides: explicit below, automatic above

Both tools need a policy, and the two layers know different things.

| layer | knob | default |
| --- | --- | --- |
| core, writer | `Serializer*::payload_prefix_len(n)` | absent — no section |
| core, reader | `StreamLimits::prefix_coalesce_gap_bytes` | `prefix_len`, i.e. the clamp |
| geo, writer | `ConvertRequest::prefix_index`, `gp2psindex build --prefix-index` | `auto` |

**The core cannot decide, and does not guess.** `PYLD` blobs are opaque bytes;
nothing in that crate knows the first 24 of them mean anything. So both of its
knobs are explicit, and both default to the old behaviour. The reader-side one
belongs with the limits rather than with the data because it is a property of the
*reader's* storage: 1 011 range requests is a catastrophe over R2 and a non-event
on a local file, and the same artifact wants opposite answers in the native
server and in the Worker.

**The geo layer can decide, and does.** It knows the prefix is always a 24-byte
feature ref and it holds every body at write time, so `auto` measures instead of
guessing. It reads the body distribution, **not the payload plan** — the cliff
table is the reason: `RowWkb` straddles it, so `--payload row-wkb` predicts
nothing. It compares the *median* body against the threshold, so one huge
geometry cannot buy a section the rest of the collection never uses.

Automatic by default because the two error directions are not symmetric. Emitting
a section that was not needed costs under 10% of file size, once. Not emitting one
that was needed costs a range request per match, on every query, forever — and
that failure is invisible at build time and shows up as a bill.

What automatic must not do is surprise someone building a local-only artifact
with a 24% size increase, which is what the `r > 10` threshold prevents: below it
the answer is the reader-side knob, which costs no bytes on disk at all.

## What it unblocks

- A remote reader can return feature references from a summary search at the same
  cost as returning none. The geo Worker demo avoided the header path precisely
  to dodge the read storm, which left its `/search` returning records without
  `featureRef` — a contract divergence from the native server that existed for no
  other reason. It is now free to converge.
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
