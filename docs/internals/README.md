# Internals: how the engine goes fast

Deep-dives into the techniques behind the query kernels — for contributors and the
curious. These explain *how* a result is produced quickly. For *what* each query
does and when to use it, see the [guide](../guide.md). For the numbers, see
[performance.md](../performance.md).

- **[simd.md](simd.md)** — runtime-dispatched SIMD range search, visit and raycast
  (AVX-512 `VPCOMPRESSQ`, the AVX2 left-pack that emulates the missing compress,
  the SSE2 fallback). The "path to the black belt" of the box-test kernels.
- **[knn.md](knn.md)** — nearest-neighbor traversal as two-queue *distance
  browsing* (Hjaltason & Samet): why kNN keeps a node queue and an item queue
  separate, the k=1 fast path, and how custom metrics and f32 refinement reuse the
  one collect kernel.
- **[prefetch.md](prefetch.md)** — hiding the cold cache miss on every tree node by
  prefetching the next node's box while the current one is hit-tested. A free
  latency hint on range search and all-hits raycast.
