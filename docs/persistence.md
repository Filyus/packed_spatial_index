# Persistence

`Index2D` and `Index3D` serialize to a stable little-endian byte format and load
back either as owned indexes or as zero-copy views. The binary layout is
documented in [`FORMAT.md`](https://github.com/Filyus/packed_spatial_index/blob/main/FORMAT.md).

```rust
use packed_spatial_index::{Index2D, Index2DBuilder, Index2DView, Box2D};

let mut builder = Index2DBuilder::new(1);
builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
let index = builder.finish()?;

let bytes = index.to_bytes();
let mut reusable = Vec::new();
index.to_bytes_into(&mut reusable); // reuse one buffer across saves
assert_eq!(reusable, bytes);

let owned = Index2D::from_bytes(&bytes)?;          // owns its tree
let view = Index2DView::from_bytes(&bytes)?;       // borrows the bytes, no alloc

assert_eq!(owned.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
assert_eq!(view.search(Box2D::new(0.0, 0.0, 2.0, 2.0)), vec![0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

3D persistence uses the same header and sections, with a dimension flag and six
`f64` coordinates per stored box. With the `simd` feature, `SimdIndex2D` and
`SimdIndex3D` read and write the same canonical bytes as the scalar indexes:
loading a SIMD index scatters the canonical box records into SoA columns, while
`SimdIndex2DView` / `SimdIndex3DView` borrow the same bytes for zero-copy
SIMD-over-AoS queries.

The `f32` indexes persist to their own `f32` box layout (distinct format flags),
with matching `from_bytes` loaders and zero-copy views.

## Large / on-disk indexes (mmap)

Loaded buffers are validated before they can be searched, and the `*View` types
borrow the bytes without allocating an owned tree. That makes a memory-mapped
file the simple way to query an index that is large or lives on disk: map the
`PSINDEX` file, hand the byte slice to a view, and the OS pages in only the tree
nodes a query actually touches. This works for files larger than RAM — the
kernel evicts unused pages — and needs no extra crate API.

```rust,ignore
// Cargo.toml: memmap2 = "0.9"
use std::fs::File;
use memmap2::Mmap;
use packed_spatial_index::{Index2DView, Box2D};

let file = File::open("city.psindex")?;
// SAFETY: the file must not be mutated while mapped.
let mmap = unsafe { Mmap::map(&file)? };

let view = Index2DView::from_bytes(&mmap)?; // borrows the mapping, no copy
let hits = view.search(Box2D::new(0.0, 0.0, 100.0, 100.0));
# Ok::<(), Box<dyn std::error::Error>>(())
```

The crate stays dependency-free here: it only consumes `&[u8]`, so you pick the
mmap crate (`memmap2`, etc.). Storing the bytes as a database BLOB works the same
way once the blob is in memory — load it and call `from_bytes`.

## Streaming (out-of-core / remote)

For sources that cannot be memory-mapped — remote object storage, a database
blob you do not want to download whole — the `stream` feature queries a
serialized index by reading only the byte ranges a traversal needs. It works
over the **same `PSINDEX` bytes**; nothing special is needed at write time.

`StreamIndex2D` / `StreamIndex3D` open over any `RangeReader` (a one-method
trait: `read_exact_at`). The crate ships `FileReader` (local file, positioned
reads) and `SliceReader` (in-memory / mmap); a remote source is your own impl.
At open the reader validates the header and level bounds and prefetches the
upper tree levels; each query then streams only the lower levels, coalescing
adjacent node ranges into few reads. Queries are fallible (a read can fail) and
return item indices like the in-memory `search`. Child pointers are validated as
they are followed, so the reader is safe on untrusted data.

```rust,ignore
// Cargo.toml: packed_spatial_index = { version = "0.6", features = ["stream"] }
use packed_spatial_index::{Box2D, FileReader, StreamIndex2D};

let reader = FileReader::open("planet.psindex")?;
let index = StreamIndex2D::open(reader)?;
let hits = index.search(Box2D::new(0.0, 0.0, 100.0, 100.0))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a remote `RangeReader` backed by HTTP range requests, see the
[`RangeReader` docs](https://docs.rs/packed_spatial_index/latest/packed_spatial_index/trait.RangeReader.html).
Compared with a memory map, streaming adds per-query I/O but never needs the
whole file local or mapped, which is what makes a planet-scale index servable
from object storage. kNN and raycast streaming are not implemented yet — see the
project backlog.
