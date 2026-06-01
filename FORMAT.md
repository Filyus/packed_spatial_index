# Binary Format

`Index::to_bytes` writes the canonical `Index` layout to a stable
little-endian byte format. `Index::from_bytes` loads the same format into owned
vectors, and `IndexView::from_bytes` borrows the buffer without allocating
during load.

`SimdIndex` does not have a separate persisted SoA format.

## Magic And Version

The current format magic is eight bytes:

```text
b"PSINDEX\0"
```

It expands to:

- `PS` = Packed Spatial;
- `INDEX` = Index;
- `\0` = one trailing NUL byte to keep the signature exactly eight bytes.

The binary format version is stored separately as a little-endian `u64` header
field. The current version is `1`.

`header_len` is the fixed byte length of this header. `flags` is reserved for
future format options and must be zero in version 1.

## Layout

All integers are unsigned little-endian 64-bit values. All coordinates are
little-endian IEEE-754 `f64` values.

```text
offset  size  field
0       8     magic: b"PSINDEX\0"
8       8     format_version: u64 = 1
16      8     header_len: u64 = 64
24      8     flags: u64 = 0
32      8     node_size
40      8     num_items
48      8     num_nodes
56      8     level_count
64      ...   level_bounds: [u64; level_count]
...     ...   boxes: [f64 min_x, f64 min_y, f64 max_x, f64 max_y; num_nodes]
...     ...   indices: [u64; num_nodes]
```

There is no padding between sections.

The fixed header is 64 bytes, so every section starts on an 8-byte logical
offset.

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
  zero `flags`;
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
`Index::to_bytes`. The crate preserves the meaning of `format_version = 1`;
incompatible changes should use a new version value.
