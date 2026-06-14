# Packed Spatial Index Binary Format

| Revision | Last revised |
| -------- | ------------ |
| 12       | 2026-06-15   |

This document describes the binary format used by packed spatial indexes
(`format_version` 2).

## Related Methods

| Method | Description |
| ------ | ----------- |
| `fn to_bytes(&self) -> Vec<u8>` | Serialize into a new byte buffer. |
| `fn to_bytes_into(&self, out: &mut Vec<u8>)` | Serialize into an existing byte buffer. |
| `fn to_bytes_with_payloads(&self, payloads: &[P]) -> Result<Vec<u8>, PayloadError>` | Serialize the index plus one blob per item (adds a `PYLD` chunk). |
| `fn to_bytes_interleaved(&self) -> Vec<u8>` | Serialize with the interleaved `TREE` layout (streaming-tuned). |
| `fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError>` | Load and validate a byte buffer. |

## Byte Order

All integer and coordinate fields are little-endian.

## Overview

A file is a **chunk container**: a fixed superblock, a flat directory of typed
chunks, then the chunks themselves.

```text
+-------------------+
| superblock (32 B) |  magic, version, chunk_count
+-------------------+
| chunk directory   |  chunk_count entries x 24 B
+-------------------+
| chunk 0           |  (8-byte aligned)
| chunk 1           |
| ...               |
+-------------------+
```

Each directory entry carries a **critical** bit. A reader rejects a file that
contains a *critical* chunk whose tag it does not understand, and silently skips
an *optional* chunk it does not understand. New chunk types can therefore be
added without breaking older readers, as long as they are marked optional.

Only **non-derivable** data is stored. The tree's `num_nodes`, `level_count`, and
the whole `level_bounds` table are functions of `num_items` and `node_size`, so
they are recomputed at load rather than stored — there is no second copy that
could drift.

Two chunk types are defined:

| Tag    | Critical | Contents |
| ------ | -------- | -------- |
| `TREE` | yes      | The packed tree: a descriptor plus the node data. Exactly one. |
| `PYLD` | no       | Optional payload: one opaque blob per item. At most one. |
| `META` | no       | Optional descriptive fields (CRS / content type / attribution). At most one. |

## Superblock

```text
offset  size  field
0       8     magic          b"PSINDEX\0"
8       8     format_version  u64 = 2
16      4     chunk_count     u32
20      12    reserved        (zero)
```

A reader rejects any `magic` other than `PSINDEX\0` and any `format_version`
other than `2`.

## Chunk directory

The directory begins at offset 32 and holds `chunk_count` entries of 24 bytes:

```text
offset  size  field
0       4     tag        4 ASCII bytes (e.g. "TREE")
4       4     flags      u32; bit 0 = critical
8       8     offset     u64, absolute byte offset of the chunk content
16      8     length     u64, byte length of the chunk content
```

Chunk content is 8-byte aligned. Each chunk's `[offset, offset + length)` must
lie within the file. A file may end with up to 7 alignment-pad bytes after the
last chunk; any further trailing bytes are rejected.

## `TREE` chunk

The `TREE` chunk is a descriptor followed by the raw node data.

```text
descriptor (24 B):
offset  size  field
0       4     desc_len    u32 = 24
4       1     dimensions  u8: 2 or 3
5       1     coord_bytes u8: 4 (f32) or 8 (f64)
6       1     layout      u8: 0 = SoA, 1 = interleaved
7       1     reserved
8       8     num_items   u64
16      2     node_size   u16, in 2..=65535
18      6     reserved
```

Node data follows the descriptor (at `offset + desc_len`). Let
`record = dimensions * 2 * coord_bytes` (the box size: 32 / 48 / 16 / 24 bytes
for 2D-f64 / 3D-f64 / 2D-f32 / 3D-f32) and `num_nodes` be the derived node count.

- **SoA layout** (`layout = 0`): a `boxes` section (`record` x `num_nodes`)
  followed by an `indices` section (8 x `num_nodes`).
- **Interleaved layout** (`layout = 1`): a single `nodes` section in which each
  node's `record` box bytes are immediately followed by its 8-byte index entry
  (stride `record + 8`).

The total node bytes are identical either way (`record + 8` per node), so the
layout does not change the file size. The interleaved layout lets a streaming
reader fetch a node's box and child pointer in one coalesced read per level
instead of two; it is produced by `to_bytes_interleaved` /
`to_bytes_interleaved_with_payloads` and read by the streaming reader. The
in-memory loaders and SIMD views read the SoA layout only and reject an
interleaved `TREE` (`UnsupportedVersion`).

`desc_len` is stored so the descriptor can grow (new fields appended) without
breaking readers: an older reader reads the prefix it understands and skips to
`offset + desc_len` for the node data.

### Box records

Each node has one box record:

```text
2D: min_x, min_y, max_x, max_y
3D: min_x, min_y, min_z, max_x, max_y, max_z
```

Fields are `f64` when `coord_bytes = 8`, `f32` when `coord_bytes = 4`. `f32`
boxes are rounded outward.

### Tree storage

Nodes are stored in level order:

1. leaves, in sorted packed order;
2. parent levels;
3. root as the final node, for non-empty trees.

The per-level boundaries are derived (not stored). For a non-empty tree:

```text
level_width[0]     = num_items
level_width[i + 1] = ceil(level_width[i] / node_size)
level_bounds[i]    = sum(level_width[0..=i])     (exclusive end of level i)
```

The first bound equals `num_items`; the last equals `num_nodes`. An empty tree
has `num_items = 0`, `num_nodes = 0`, `level_count = 1`.

### Indices

Each index entry pairs with the box at the same node position.

| Node kind | Stored value |
| --------- | ------------ |
| leaf      | Original insertion index. |
| internal  | Position of the first child in the previous level. |

An internal node spans up to `node_size` children, clipped by the previous
level's end.

## `PYLD` chunk (optional)

The `PYLD` chunk carries one opaque blob per item, making a file self-contained
(the spatial index plus the data it indexes). It is a descriptor followed by the
blob region, optionally preceded by a prefix-offset table.

```text
descriptor (8 or 12 B):
offset  size  field
0       4     desc_len       u32 = 8 (variable-width) or 12 (fixed-width)
4       1     ordering       u8 = 0 (leaf rank)
5       1     compression    u8 = 0 (none)
6       2     reserved
8       4     record_stride  u32, present only when desc_len = 12; 0 = variable

then:
payload_offsets   (num_items + 1) x u64   (variable-width only)
payload_blobs     concatenated blobs
```

There are two layouts, chosen by `record_stride`:

- **Variable-width** (`record_stride` absent or `0`): a `(num_items + 1)`
  prefix-offset table precedes the blobs. The blob at leaf rank `r` is
  `payload_blobs[payload_offsets[r] .. payload_offsets[r + 1]]`.
- **Fixed-width** (`record_stride > 0`): every blob is exactly `record_stride`
  bytes and there is **no** offset table. The blob at leaf rank `r` is
  `payload_blobs[r * record_stride ..][.. record_stride]`. This drops the table
  (smaller file, one fewer streamed read) and lets a reader borrow the blobs as a
  typed slice. Triangle meshes use it (a 2D/3D triangle is 48/72 fixed bytes in
  `f64`, or 24/36 in `f32`).

Both layouts order the blobs by **leaf rank** (the position of an item among the
leaves in Hilbert order), so a spatial query, which visits leaves in contiguous
runs, fetches them in coalesced reads.

The item at leaf rank `r` has its original insertion index (what queries return)
in the leaf entry of `indices` at rank `r`; to look a blob up by insertion index,
invert that mapping. For the variable-width table, `payload_offsets[0]` is `0`,
the table is non-decreasing, and `payload_offsets[num_items]` equals the blob
region length.

Blobs are opaque bytes with no required interpretation. A reader that does not
handle payloads simply skips the optional `PYLD` chunk and reads the index from
`TREE`.

## `META` chunk (optional)

The `META` chunk carries small descriptive fields. It is a flat list of fields,
read until the chunk ends:

```text
repeat until end of chunk:
  field_id  u16
  length    u32
  value     length bytes (UTF-8)
```

| field_id | field          | example value             |
| -------- | -------------- | ------------------------- |
| 0        | `crs`          | `"EPSG:4326"`             |
| 1        | `content_type` | `"application/geo+json"`  |
| 2        | `attribution`  | `"© Example"`             |

Values are **opaque** strings the writer supplied; this format does not parse or
interpret them (the CRS is whatever identifier the producer chose). Only the
fields actually set are written. An unknown `field_id` is **skipped**, so new
fields are non-breaking. A reader gets the metadata with `read_metadata` without
loading the index; index loaders skip the chunk entirely.

`META` holds only generic descriptive fields. Derivable facts (extent, counts)
are recomputed rather than stored, and application-specific data belongs in an
application-private chunk (a lowercase-first tag), not here.

## Extensibility

The container is designed so future additions do not break readers:

- a new **optional** chunk type is ignored by older readers (skipped via the
  directory), so it is non-breaking;
- a new **critical** chunk type is rejected by older readers, which is the safe
  outcome when they cannot interpret required data;
- a `TREE` / `PYLD` descriptor may gain trailing fields (its `desc_len` grows);
  older readers read the prefix they know.

`format_version` is bumped only on a change that breaks these rules.

### Tag namespace

Chunk tags are split into two spaces by the case of the first byte:

- **uppercase ASCII first byte** (e.g. `TREE`, `PYLD`) — reserved for this format.
  New format-defined chunks come from this space.
- **lowercase ASCII first byte** — free for application-private chunks. This
  format will never define such a tag, so applications can attach their own
  chunks without risking a collision with a future format version.

Application chunks should be marked **optional** (so format-only readers skip
them); a critical application chunk is allowed but limits the file to
application-aware readers.

## Validation

Loaders reject:

- wrong magic, or a `format_version` other than `2`;
- a chunk whose byte range falls outside the buffer;
- an unknown **critical** chunk;
- trailing bytes beyond the last chunk's alignment pad;
- a missing `TREE` chunk, or a `TREE` whose length does not match the derived
  tree shape;
- `node_size` outside `2..=65535`;
- leaf indices outside `0..num_items`;
- internal pointers outside the previous level, or not at a child-group start;
- a `PYLD` offset table that is not `0`-based and non-decreasing, or whose final
  offset does not match the blob region length.

`LoadError` reports the failure category, not a byte offset.

The streaming reader cannot validate the whole buffer up front (it fetches only
what a query needs), so it validates chunk ranges, pointers, and payload offsets
lazily as it follows them. See [SAFETY.md](SAFETY.md) for that hardening.
