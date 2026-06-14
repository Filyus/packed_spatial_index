# Safety

This crate parses binary indexes — including, with the `stream` feature, data
fetched from a remote object store — so it treats serialized input as untrusted.
This document collects the memory-safety story and the hardening done against
malformed or adversarial buffers.

## Memory safety (`unsafe`)

The public API is safe Rust. `unsafe` is confined to narrow, audited paths:

- validated unaligned little-endian reads in the byte-backed views;
- bulk `repr(C)` byte copies during serialization on little-endian targets;
- runtime-feature-gated x86-64 SIMD (AVX-512) loads / prefetch.

Each `unsafe` block carries a `// SAFETY:` justification; CI runs Clippy with
`undocumented_unsafe_blocks` denied, so an unjustified block fails the build. The
reads behind these blocks operate only on offsets and lengths that the validation
below has already bounds-checked, so a malformed buffer returns a `LoadError` /
`StreamError` rather than triggering out-of-bounds access.

## Untrusted input — in-memory loaders

`from_bytes` (owned indexes and zero-copy views) validates the whole buffer
before any query can run. It rejects, with a `LoadError` that names the category
(never a byte offset):

- wrong magic, or an unsupported `format_version`;
- a chunk whose byte range falls outside the buffer; an unknown **critical**
  chunk; or trailing bytes beyond the last chunk's alignment pad;
- a missing `TREE` chunk, or a `TREE` whose length is inconsistent with the tree
  shape derived from `num_items` and `node_size`;
- `node_size` outside `2..=65535`;
- leaf indices outside `0..num_items`;
- internal pointers outside the previous level, or not at a child-group start;
- a `PYLD` offset table that is not `0`-based and non-decreasing.

Because the entire structure is checked up front, the traversal itself can then
trust it. See [FORMAT.md](FORMAT.md#validation) for the field-level rules.

## Untrusted input — streaming reader (remote)

The streaming reader (`StreamIndex2D` / `StreamIndex3D` over a `RangeReader` /
`AsyncRangeReader`) is the harder case: it must serve queries **without reading
the whole file**, so it cannot pre-validate structures it never fetches (a 1M-item
offset table alone is 8 MB). The guiding rule is therefore:

> When a structure can't be validated as a whole, every per-access computation
> that subtracts from or indexes by an untrusted value needs a **complete** bounds
> check at the point of use. A partial check is a hole.

Concretely, the reader hardens three places.

### 1. Pointers validated as they are followed

`open` validates the superblock and chunk directory (every chunk's byte range
against the file length), reads the small `TREE` descriptor, and derives the tree
shape. During descent, each child pointer is checked against the bounds of the
level it points into — range, and child-group alignment — exactly as the in-memory loader
does, but lazily, one frontier at a time.

### 2. Frontier ordering and dedup (DoS hardening)

The coalescing read planner assumes the positions it gathers are **ascending and
unique** (it computes byte gaps by subtraction). A corrupt index whose child
pointers are in range but **reordered** would produce a non-monotonic frontier
and underflow that subtraction; **aliased** pointers (two parents → one child)
would duplicate child groups and let the frontier blow up level over level
(time / memory DoS). Each expanded frontier is therefore `sort_unstable` +
`dedup`ed — a no-op on a well-formed tree, a hard bound (the level width) on a
malicious one. Per-pointer range checks alone are *not* enough; ordering and
uniqueness are part of the contract.

### 3. Payload offset bounds (underflow hardening)

The payload offset table is read in coalesced runs, never wholesale, so a run's
entries may be out of order relative to each other. Slicing a blob requires
`blob_lo <= o0 <= o1 <= blob_hi`; the *lower* bound (`o0 >= blob_lo`) matters as
much as the upper one, because `o0 - blob_lo` is an unsigned subtraction that
would underflow on an out-of-range offset. The check is complete on every blob
emitted, and an offset that fails it is rejected as `InvalidTree`.

The reader fetches each byte range **once per query** (it never re-reads a range
to re-decide), so there is no within-query time-of-check/time-of-use window. A
file-derived value is validated immediately before the access that consumes it,
on the same bytes that produced it.

### 4. Per-query cost limits

Even a perfectly well-formed index can be expensive to serve: a broad query can
read a lot and return a lot. `StreamLimits { max_reads, max_read_bytes,
max_items }` (set via `open_with_limits`) bounds each query; exceeding any limit
aborts it with `StreamError::LimitExceeded` rather than running unbounded. The
limits are caller-supplied — the library provides the knob, the caller (which
knows its environment, e.g. a Worker's subrequest and memory budgets) picks the
values. Because read coalescing keeps the *read count* low even for wide queries,
`max_read_bytes` and `max_items` are the effective guards against large result
sets; `max_reads` mainly caps scattered queries.

### Guarantee and boundary

Together these give a concrete guarantee: a malformed, adversarial, or even
*inconsistent* backing store (one that returns different bytes on different reads)
can cause a query to return the wrong results or a `StreamError`, but **never**
memory unsafety, a panic, an unbounded loop, or unbounded work. The reader does
**not** authenticate the data — if you need to detect tampering (as opposed to
merely surviving it), verify the bytes out of band (e.g. a content hash). Within
that boundary, untrusted and remote input is handled safely.

## Fuzzing and tests

The validation paths are pinned by tests, not just asserted:

- byte-flip fuzz across the whole file (header, index, and — for payload files —
  the offset table and blobs): `open` plus `search` / `search_payloads` must
  return `Ok` or `Err`, never panic. Covers the SoA, interleaved, and payload
  layouts.
- targeted adversarial cases: reordered / aliased child pointers and out-of-order
  payload offsets are rejected (`InvalidTree`), not misread.
- a hostile-reader test serves a valid header but returns adversarial,
  per-read-varying garbage for the entire node and payload region; the full
  descent runs against it and must not panic.
- the streamed result set is checked for equality against the in-memory loader as
  an oracle, across many random queries and sizes.

These run on every CI build alongside Clippy (`-D warnings`, undocumented-unsafe
denied), the full feature matrix, and the MSRV check.
