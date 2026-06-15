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

3D persistence uses the same container format, with the dimension recorded in the
`TREE` descriptor and six `f64` coordinates per stored box. With the `simd`
feature, `SimdIndex2D` and
`SimdIndex3D` read and write the same canonical bytes as the scalar indexes:
loading a SIMD index scatters the canonical box records into SoA columns, while
`SimdIndex2DView` / `SimdIndex3DView` borrow the same bytes for zero-copy
SIMD-over-AoS queries.

The compact `f32` indexes persist to their own `f32` box layout (`coord_bytes =
4`), halving the box bytes. With `f32-storage` the scalar `Index2DF32` /
`Index3DF32` build, query, `serialize()` (boxes plus an optional payload and
metadata), and `from_bytes` without `simd`; the SIMD `SimdIndex*F32` frontends
add the vectorized queries. The stored f32 boxes are rounded outward, so range
and ray results are a conservative superset — use `search_exact` (refining
against your `f64` boxes) when you need exact hits.

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
At open the reader validates the superblock and chunk directory and prefetches
the upper tree levels; each query then streams only the lower levels, coalescing
adjacent node ranges into few reads. Queries are fallible (a read can fail) and
return item indices like the in-memory `search`. Child pointers are validated as
they are followed, so the reader is safe on untrusted data.

```rust,ignore
// Cargo.toml: packed_spatial_index = { version = "0.7", features = ["stream"] }
use packed_spatial_index::{Box2D, FileReader, StreamIndex2D};

let reader = FileReader::open("planet.psindex")?;
let index = StreamIndex2D::open(reader)?;
let hits = index.search(Box2D::new(0.0, 0.0, 100.0, 100.0))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a remote source, implement `RangeReader` over HTTP range requests. It is one
method — return the bytes for `[offset, offset + len)`:

```rust,ignore
use packed_spatial_index::RangeReader;
use std::io::{self, Read};

struct HttpRange {
    url: String,
}

impl RangeReader for HttpRange {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let end = offset + buf.len() as u64 - 1;
        // Any HTTP client works; the request is a single ranged GET.
        let resp = ureq::get(&self.url)
            .set("Range", &format!("bytes={offset}-{end}"))
            .call()
            .map_err(io::Error::other)?;
        resp.into_reader().read_exact(buf)
    }

    // Optional: returning the object size (e.g. from a HEAD request) lets `open`
    // cross-check the declared length. `None` skips that check.
    fn len(&self) -> Option<u64> {
        None
    }
}
```

The server only needs to honor the HTTP `Range` header — S3, R2, GCS, and most
CDNs do. For async I/O (a browser `fetch`, or a Cloudflare Worker over R2) enable
the `async` feature and implement `AsyncRangeReader` the same way; `open_async` /
`search_async` mirror the sync API and issue a level's reads concurrently.

**What streams today:** 2D and 3D range search (`search` / `search_into` /
`visit`), optionally returning a stored blob per hit (`search_payloads`, when the
file was written with `to_bytes_with_payloads`), sync or async, with optional
per-query cost limits (`open_with_limits` + `StreamLimits` — bound reads, bytes,
and items so a broad query cannot run unbounded). For a remote-tuned layout,
`to_bytes_interleaved` stores each node's box and child pointer together so the
descent fetches them in one read per level instead of two.

**Compact `f32` streaming:** `StreamIndex2DF32` / `StreamIndex3DF32` stream a
compact `f32`-box file for half the box bytes over the wire, with the same
`search` / `search_payloads` / async API. The stored boxes are rounded outward,
so results are a conservative superset (filter against your own `f64` boxes for
exact). Write the file with the scalar `Index3DF32` or the SIMD `SimdIndex3DF32`;
`.interleaved()` on its `serialize()` builder enables the one-read-per-level
layout, and a triangle payload makes it a compact streamable mesh BVH — all
without the `simd` feature.

**What does not stream:** nearest-neighbor and raycast queries are in-memory only.
Their best-first traversal jumps around the tree, so adjacent reads do not
coalesce and streaming would be a read per node — load those with `from_bytes` or
a memory map instead.

Compared with a memory map, streaming trades per-query I/O for never needing the
whole file local or mapped — which is what makes a planet-scale index servable
straight from object storage. Untrusted and remote bytes are handled safely; see
[SAFETY.md](../SAFETY.md).
