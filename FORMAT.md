# Packed Spatial Index Binary Format

| Revision | Last revised |
| -------- | ------------ |
| 8        | 2026-06-05   |

This document describes the binary format used by packed spatial indexes.

## Related Methods

| Method | Description |
| ------ | ----------- |
| `fn to_bytes(&self) -> Vec<u8>` | Serialize into a new byte buffer. |
| `fn to_bytes_into(&self, out: &mut Vec<u8>)` | Serialize into an existing byte buffer. |
| `fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError>` | Load and validate a byte buffer. |

## Byte Order

All integer and coordinate fields are little-endian.

## Layout

A serialized index buffer has one header and three contiguous sections:

| Part           | Bytes                  |
| -------------- | ---------------------- |
| header         | 64                     |
| `level_bounds` | 8 x `level_count`      |
| `boxes`        | `record` x `num_nodes` |
| `indices`      | 8 x `num_nodes`        |

There is no padding. Each section starts on an 8-byte offset.
`record` is selected by `flags`.

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

`flags` selects dimension and coordinate width:

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
- other flag values are reserved.

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
