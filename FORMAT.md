# Binary Format

`Index2D::to_bytes` and `Index3D::to_bytes` write canonical packed layouts to a
stable little-endian byte format. `SimdIndex2D::to_bytes` and
`SimdIndex3D::to_bytes` write byte-identical canonical layouts from their SoA
columns, so scalar and SIMD indexes are interchangeable on disk.

`Index2D::from_bytes` and `Index3D::from_bytes` load the same format into owned
vectors, while `Index2DView::from_bytes` and `Index3DView::from_bytes` borrow
the buffer without allocating during load. SIMD indexes load the canonical
bytes into owned SoA columns; there is no separate persisted SoA format and no
zero-copy SIMD view format.

## Magic And Version

The current format magic is eight bytes:

```text
b"PSINDEX\0"
```

It expands to:

- `PS` = Packed Spatial;
- `INDEX` = index;
- `\0` = one trailing NUL byte to keep the signature exactly eight bytes.

The binary format version is stored separately as a little-endian `u64` header
field. The current version is `1`.

`header_len` is the fixed byte length of this header. `flags` currently selects
the coordinate dimension:

- `0`: 2D boxes;
- `1`: 3D boxes.

Other flag values are reserved.

## Layout

All integers are unsigned little-endian 64-bit values. All coordinates are
little-endian IEEE-754 `f64` values.

```text
offset  size  field
0       8     magic: b"PSINDEX\0"
8       8     format_version: u64 = 1
16      8     header_len: u64 = 64
24      8     flags: u64 = 0 for 2D, 1 for 3D
32      8     node_size
40      8     num_items
48      8     num_nodes
56      8     level_count
64      ...   level_bounds: [u64; level_count]
...     ...   boxes: [box; num_nodes]
...     ...   indices: [u64; num_nodes]
```

There is no padding between sections.

The fixed header is 64 bytes, so every section starts on an 8-byte logical
offset.

Box records are:

```text
2D: f64 min_x, f64 min_y, f64 max_x, f64 max_y
3D: f64 min_x, f64 min_y, f64 min_z, f64 max_x, f64 max_y, f64 max_z
```

## Tree Storage

Nodes are stored in packed level order:

- leaf item boxes first, in the sorted packed order;
- then parent levels, each packed contiguously;
- the root is the final node for non-empty indexes.

`level_bounds` stores cumulative end offsets for each level. For example,
`level_bounds[0]` is the end of the leaf level, and the final bound equals
`num_nodes`.

For an empty index:

- `num_items = 0`
- `num_nodes = 0`
- `level_count = 1`
- `level_bounds = [0]`

## Indices

The `indices` section has one entry for every stored box:

- leaf entries are original insertion indices into the caller's payload array;
- internal entries are offsets of the first child node in the previous level.

The child range for an internal node starts at that stored offset and spans up
to `node_size` entries, clamped to the previous level's end.

## Validation

Loaders reject malformed buffers before exposing safe search APIs. Validation
checks include:

- exact magic match, supported `format_version`, supported `header_len`, and
  supported `flags` for the requested loader;
- complete header and sections;
- exact byte length;
- `node_size` in `2..=65535`;
- tree shape matching `num_items`, `node_size`, `num_nodes`, and `level_count`;
- monotonic level bounds with the expected cumulative values;
- leaf item indices within `0..num_items`;
- internal child pointers pointing to node starts in the previous level.

`LoadError` reports the broad failure category. It does not promise a byte
offset for malformed input.

## Compatibility

The byte format is intended for data produced by `packed_spatial_index`
`Index2D::to_bytes`, `Index3D::to_bytes`, `SimdIndex2D::to_bytes`, and
`SimdIndex3D::to_bytes`. The crate preserves the meaning of
`format_version = 1`; incompatible changes should use a new version value.
