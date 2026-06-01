# Binary Format

`Index::to_bytes` writes the canonical `Index` layout to a stable
little-endian byte format. `Index::from_bytes` loads the same format into owned
vectors, and `IndexView::from_bytes` borrows the buffer without allocating
during load.

`SimdIndex` does not have a separate persisted SoA format.

## Version

The current format magic is:

```text
PSIDX001
```

The marker is both the file signature and the format version. A future breaking
format should use a different marker.

## Layout

All integers are unsigned little-endian 64-bit values. All coordinates are
little-endian IEEE-754 `f64` values.

```text
offset  size  field
0       8     magic/version: ASCII "PSIDX001"
8       8     node_size
16      8     num_items
24      8     num_nodes
32      8     level_count
40      ...   level_bounds: [u64; level_count]
...     ...   boxes: [f64 min_x, f64 min_y, f64 max_x, f64 max_y; num_nodes]
...     ...   indices: [u64; num_nodes]
```

There is no padding between sections.

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

- exact magic/version match;
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
`Index::to_bytes`. The crate preserves the meaning of `PSIDX001`; incompatible
changes should use a new magic/version marker.
