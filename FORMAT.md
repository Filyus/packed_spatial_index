# Packed Spatial Index Binary Format

| Revision | Last revised |
| -------- | ------------ |
| 10       | 2026-06-14   |

This document describes the binary format used by packed spatial indexes.

## Related Methods

| Method | Description |
| ------ | ----------- |
| `fn to_bytes(&self) -> Vec<u8>` | Serialize into a new byte buffer. |
| `fn to_bytes_into(&self, out: &mut Vec<u8>)` | Serialize into an existing byte buffer. |
| `fn to_bytes_with_payloads(&self, payloads: &[P]) -> Result<Vec<u8>, PayloadError>` | Serialize the index plus one blob per item (sets the payload flag). |
| `fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError>` | Load and validate a byte buffer. |

## Byte Order

All integer and coordinate fields are little-endian.

## Layout

A serialized index buffer has one header and three contiguous sections, plus an
optional payload section when the payload flag is set:

| Part              | Bytes                  | Present                |
| ----------------- | ---------------------- | ---------------------- |
| header            | 64                     | always                 |
| `level_bounds`    | 8 x `level_count`      | always                 |
| `boxes`           | `record` x `num_nodes` | always                 |
| `indices`         | 8 x `num_nodes`        | always                 |
| `payload_offsets` | 8 x (`num_items` + 1)  | payload flag set       |
| `payload_blobs`   | `payload_offsets[num_items]` | payload flag set |

There is no padding. Each section starts on an 8-byte offset.
`record` is selected by `flags`. The optional payload section is described under
[Payload](#payload-optional); the index sections are byte-identical with or
without it.

## Header

```text
offset  size  field
0       8     magic
8       8     format_version
16      8     header_len
24      8     flags
32      8     node_size
40      8     num_items
48      8     num_nodes
56      8     level_count
```

All header fields after `magic` are `u64`.

| Field            | Value / meaning |
| ---------------- | --------------- |
| `magic`          | `b"PSINDEX\0"` |
| `format_version` | `1` |
| `header_len`     | `64` |
| `flags`          | Variant selector. See [Variants](#variants). |
| `node_size`      | Maximum children per internal node. Valid range: `2..=65535`. |
| `num_items`      | Number of leaf items. Leaf indices must be lower than this value. |
| `num_nodes`      | Total stored nodes: leaves plus internal nodes. |
| `level_count`    | Number of tree levels. Empty indexes use `1`. |

## Variants

The low 8 bits of `flags` select dimension and coordinate width:

| Flag | Coordinates | Record | Owned types              |
| ---- | ----------- | ------ | ------------------------ |
| `0`  | 2D `f64`    | 32 B   | `Index2D`, `SimdIndex2D` |
| `1`  | 3D `f64`    | 48 B   | `Index3D`, `SimdIndex3D` |
| `2`  | 2D `f32`    | 16 B   | `SimdIndex2DF32`         |
| `3`  | 3D `f32`    | 24 B   | `SimdIndex3DF32`         |

- each type's `...View` reads the same flag;
- flags `0` and `1` are shared by scalar and `f64` SIMD indexes;
- flags `2` and `3` are used by the `f32-storage` feature;
- `f32` boxes are rounded outward;
- other low-byte flag values are reserved.

Bit `8` (`0x100`) is the **payload flag**: when set, a payload section follows
the index (see [Payload](#payload-optional)). It is orthogonal to the dimension
bits. A reader must never interpret the trailing payload bytes as index data: it
either rejects a file with this bit set, or reads only the index (validating but
ignoring the payload). The scalar `Index2D` / `Index3D` loaders and views do the
latter — the views additionally expose the blobs; the SIMD loaders reject.
Other high bits are reserved.

## Payload (optional)

When the payload flag is set, two sections follow `indices`, carrying one opaque
blob per item so the file is self-contained (the spatial index plus the data it
indexes). Both are ordered by **leaf rank** — the position of an item among the
leaves (`indices[0..num_items]`), i.e. the Hilbert order — so that a spatial
query, which visits leaves in contiguous runs, fetches their blobs and offsets in
coalesced reads.

- `payload_offsets`: `num_items + 1` little-endian `u64` prefix offsets into
  `payload_blobs`, indexed by leaf rank. `payload_offsets[0]` is `0` and the
  table is non-decreasing; `payload_offsets[num_items]` equals the total blob
  byte length.
- `payload_blobs`: the concatenated blobs in leaf order.

The blob of the item at leaf rank `r` is
`payload_blobs[payload_offsets[r] .. payload_offsets[r + 1]]`, and that item's
original insertion index (what queries return) is `indices[r]`. To look a blob up
by insertion index, invert `indices` to get the leaf rank. Blobs are opaque
bytes with no required alignment or interpretation.

Loaders reject a payload section whose offset table is not `0`-based and
non-decreasing, whose final offset does not match the blob region length, or
whose declared total length does not match the buffer.

## Box Records

Each node has one box record.

```text
2D: min_x, min_y, max_x, max_y
3D: min_x, min_y, min_z, max_x, max_y, max_z
```

Fields are `f64` for flags `0` and `1`.
Fields are `f32` for flags `2` and `3`.

## Tree Storage

Nodes are stored in level order:

1. leaves, in sorted packed order;
2. parent levels;
3. root as the final node, for non-empty trees.

`level_bounds[i]` is the exclusive end offset of level `i`.
The first entry equals `num_items`.
The last entry equals `num_nodes`.

For a non-empty tree:

```text
level_width[0] = num_items
level_width[i + 1] = ceil(level_width[i] / node_size)
```

For an empty tree:

```text
num_items = 0
num_nodes = 0
level_count = 1
level_bounds = [0]
```

## Indices

Each `indices` entry pairs with the box at the same position.

| Node kind | Stored value |
| --------- | ------------ |
| leaf      | Original insertion index. |
| internal  | Offset of the first child in the previous level. |

An internal node spans up to `node_size` children.
The span is clipped by the previous level's end offset.

## Validation

Loaders reject:

- wrong magic;
- unsupported `format_version`, `header_len`, or `flags`;
- truncated buffers;
- buffers with trailing bytes;
- `node_size` outside `2..=65535`;
- tree shape that does not match `num_items` and `node_size`;
- invalid `level_bounds`;
- leaf indices outside `0..num_items`;
- internal pointers outside the previous level;
- internal pointers not at a child-group start.

`LoadError` reports the failure category, not a byte offset.
