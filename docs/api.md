# API map

Every query family, every type, one page. This is the map; the per-method
reference is on [docs.rs](https://docs.rs/packed_spatial_index), and if you know
what you want but not what it is called, start from the guide's
[I need … → use …](guide.md#choosing-a-query-method) table instead.

Method names follow one rule: the operation leads, the way you want the answer
trails it as `_into`, `_with`, `_iter`, `_each`, `_any` or `_first`. The guide
explains it under [How the names are built](guide.md#how-the-names-are-built).

Every in-memory **f64** query exists on `Index2D` / `Index3D`, on the
`simd`-feature `SimdIndex2D` / `SimdIndex3D`, and on the zero-copy views. The
compact `f32` indexes and the streaming readers cover a subset — the
[coverage matrix](guide.md#coverage-matrix) says which cells are empty by
design. Results are item indices in insertion order, and their order is
unspecified unless you asked for one with `search_ordered`.

## Queries

| Query | Methods |
| --- | --- |
| Range / overlap | [`search`][search], [`search_iter`][search_iter], [`search_into`][search_into], [`search_with`][search_with], [`any`][any], [`first`][first], [`count`][count], [`visit`][visit] |
| Nearest neighbors (point) | [`neighbors`][neighbors], [`neighbors_within`][neighbors_within], [`neighbors_into`][neighbors_into], [`neighbors_with`][neighbors_with], [`neighbors_each`][neighbors_each] |
| Nearest neighbors (box) | [`neighbors_of_box`][neighbors_of_box], [`neighbors_of_box_within`][neighbors_of_box_within], [`neighbors_of_box_into`][neighbors_of_box_into], [`neighbors_of_box_with`][neighbors_of_box_with], [`neighbors_of_box_each`][neighbors_of_box_each] |
| Geographic / custom-metric kNN | [`neighbors_metric`][neighbors_metric], [`neighbors_metric_into`][neighbors_metric_into], [`neighbors_metric_each`][neighbors_metric_each] — pass a `\|box\| -> f64` distance (e.g. [`haversine_distance_2d`][haversine_distance_2d] for lon/lat) |
| Ordered region | [`search_ordered`][search_ordered], [`search_ordered_into`][search_ordered_into], [`search_ordered_each`][search_ordered_each] — the same region shapes, emitted in nondecreasing order of a `\|box\| -> f64` key (e.g. [`view_depth_3d`][view_depth_3d] for front-to-back), so a budget can stop the traversal |
| Ray segment | [`raycast`][raycast], [`raycast_into`][raycast_into], [`raycast_with`][raycast_with], [`raycast_closest`][raycast_closest], [`raycast_closest_with`][raycast_closest_with], [`raycast_each`][raycast_each], [`raycast_any`][raycast_any] |
| Spatial join | [`join`][join], [`join_each`][join_each] between two indexes; [`pairs`][pairs], [`pairs_each`][pairs_each] for the overlapping pairs within one |
| Aggregate over a window | [`aggregate`][aggregate] — the exact count / sum / min / max / mask-OR of the hits, folded from per-node summaries (`AGGR` chunk); needs `aggregate_scalar` / `aggregate_mask` at build time |
| Estimate before you query | [`estimate_count`][estimate_count] — an exact `[lower, upper]` bracket on the hit count from node boxes alone, plus a point estimate; the streaming readers answer it from the cached directory without a read |
| Radius (within ε) | [`search_within`][search_within], [`search_within_into`][search_within_into], [`search_within_each`][search_within_each], [`search_within_any`][search_within_any], [`count_within`][count_within] — every item whose box lies within `max_distance` of a query box, `max_distance = 0.0` reproducing `search` |
| Distance join (ε-join) | [`join_within`][join_within], [`join_within_each`][join_within_each], [`pairs_within`][pairs_within], [`pairs_within_each`][pairs_within_each], [`anti_join_within`][anti_join_within], [`pairs_within_components`][pairs_within_components] |
| Closest pair | [`closest_pair`][closest_pair] within one index, [`closest_pair_to`][closest_pair_to] between two — the single nearest pair, with no `max_distance` to guess |
| Extent / exact | [`extent`][extent], and [`search_exact`][search_exact] / [`neighbors_exact`][neighbors_exact] on the `f32` indexes |

The range / overlap methods accept `Box2D` / `Box3D` queries and borrowed
region geometry such as `Triangle2D`, `ConvexPolygon2D`, and `Frustum3D`. On the
SIMD and `f32` frontends the shapes live on a parallel `*_region` family
(`search_region` / `search_region_each` / `count_region` / `search_region_any` /
`search_region_first`), so their `Box` entry points keep the SIMD kernel to themselves.

Both spellings, on the same boxes:

```rust
use packed_spatial_index::{Box2D, Index2DBuilder, Triangle2D};

let boxes = [
    Box2D::new(0.2, 0.2, 0.3, 0.3), // inside the triangle
    Box2D::new(9.0, 9.0, 9.5, 9.5), // outside it, but inside its bounding box
];
let build = || {
    let mut b = Index2DBuilder::new(boxes.len());
    for &bx in &boxes {
        b.add(bx);
    }
    b
};
let tri = Triangle2D::new([0.0, 0.0], [10.0, 0.0], [0.0, 10.0]);

// Owned indexes and views take the shape through `search` itself.
assert_eq!(build().finish().unwrap().search(&tri), vec![0]);

// SIMD and f32 frontends keep `search` for boxes, so shapes go on `_region`.
#[cfg(feature = "simd")]
assert_eq!(build().finish_simd().unwrap().search_region(&tri), vec![0]);
```

## Types

- **Geometry**: [`Box2D`][Box2D], [`Box3D`][Box3D] (inclusive `overlaps` /
  `contains` / `contains_point` / `from_point`), [`Point2D`][Point2D],
  [`Point3D`][Point3D], [`Ray2D`][Ray2D], [`Ray3D`][Ray3D],
  [`Triangle2D`][Triangle2D] / [`ConvexPolygon2D`][ConvexPolygon2D] (2D region
  queries), [`Frustum3D`][Frustum3D] (3D culling; [`ClipSpaceZ`][ClipSpaceZ]
  picks the NDC depth convention for `from_view_projection`),
  [`HalfSpace2D`][HalfSpace2D] / [`HalfSpace3D`][HalfSpace3D] (unbounded
  cross-sections — no bounding box to pre-filter with),
  [`Capsule2D`][Capsule2D] / [`Capsule3D`][Capsule3D] (a thick ray, picking
  tolerance in world units), [`Cone3D`][Cone3D] (sensor FOV / spotlight;
  [`Cone3DError`][Cone3DError] from `try_new`).
- **Builders**: [`Index2DBuilder`][Index2DBuilder],
  [`Index3DBuilder`][Index3DBuilder] — [`finish`][finish] (scalar f64),
  [`finish_simd`][finish_simd] (SoA + SIMD), [`finish_f32`][finish_f32] (compact
  scalar f32), [`finish_simd_f32`][finish_simd_f32] (compact f32 + SIMD).
- **Indexes**: [`Index2D`][Index2D] / [`Index3D`][Index3D] (scalar f64),
  [`SimdIndex2D`][SimdIndex2D] / [`SimdIndex3D`][SimdIndex3D] (SIMD f64),
  [`Index2DF32`][Index2DF32] / [`Index3DF32`][Index3DF32] (half-memory scalar
  f32), [`SimdIndex2DF32`][SimdIndex2DF32] / [`SimdIndex3DF32`][SimdIndex3DF32]
  (half-memory f32 + SIMD).
- **Views**: zero-copy [`Index2DView`][Index2DView] /
  [`Index3DView`][Index3DView] (and SIMD / f32 view variants) over serialized
  bytes.
- **Streaming**: [`StreamIndex2D`][StreamIndex2D] / [`StreamIndex3D`][StreamIndex3D]
  (and compact `StreamIndex2DF32` / `StreamIndex3DF32`) query a serialized index
  over a `RangeReader` without loading it whole (`stream` feature). A windowed
  query over a 100 MB index served from object storage costs only a handful of
  range reads. See the [Cloudflare Worker + R2 example](../wasm-demo/worker).
- **Distance metrics**: [`haversine_distance_2d`][haversine_distance_2d] and the
  [`EARTH_RADIUS_M`][EARTH_RADIUS_M] constant feed great-circle distances into the
  custom-metric kNN closures.
- **Ordering keys**: [`view_depth_2d`][view_depth_2d] /
  [`view_depth_3d`][view_depth_3d] give depth along a view axis, the ready-made
  key for a front-to-back `search_ordered`.
- **Workspaces**: [`SearchWorkspace`][SearchWorkspace] /
  [`NeighborWorkspace`][NeighborWorkspace] reuse buffers in loops.
- **Sorting / errors**: [`SortKey2D`][SortKey2D] / [`SortKey3D`][SortKey3D]
  (default `Hilbert`), [`BoundsError`][BoundsError], [`BuildError`][BuildError],
  [`LoadError`][LoadError].

[search]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search
[search_iter]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_iter
[search_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_into
[search_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_with
[any]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.any
[first]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.first
[count]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.count
[visit]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.visit
[neighbors]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors
[neighbors_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_within
[neighbors_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_into
[neighbors_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_with
[neighbors_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_each
[neighbors_metric]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_metric
[neighbors_metric_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_metric_into
[neighbors_metric_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_metric_each
[haversine_distance_2d]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/fn.haversine_distance_2d.html
[search_ordered]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_ordered
[search_ordered_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_ordered_into
[search_ordered_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_ordered_each
[view_depth_2d]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/fn.view_depth_2d.html
[view_depth_3d]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/fn.view_depth_3d.html
[EARTH_RADIUS_M]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/constant.EARTH_RADIUS_M.html
[neighbors_of_box]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box
[neighbors_of_box_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box_within
[neighbors_of_box_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box_into
[neighbors_of_box_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box_with
[neighbors_of_box_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.neighbors_of_box_each
[raycast]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast
[raycast_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_into
[raycast_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_with
[raycast_closest]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_closest
[raycast_closest_with]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_closest_with
[raycast_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_each
[raycast_any]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.raycast_any
[join]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.join
[join_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.join_each
[pairs]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.pairs
[pairs_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.pairs_each
[estimate_count]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.estimate_count
[search_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_within
[search_within_into]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_within_into
[search_within_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_within_each
[search_within_any]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.search_within_any
[count_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.count_within
[closest_pair_to]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.closest_pair_to
[closest_pair]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.closest_pair
[join_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.join_within
[join_within_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.join_within_each
[pairs_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.pairs_within
[pairs_within_each]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.pairs_within_each
[anti_join_within]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.anti_join_within
[pairs_within_components]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.pairs_within_components
[extent]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.extent
[search_exact]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2DF32.html#method.search_exact
[neighbors_exact]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2DF32.html#method.neighbors_exact
[Box2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Box2D.html
[Box3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Box3D.html
[Point2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Point2D.html
[Point3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Point3D.html
[Ray2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Ray2D.html
[Ray3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Ray3D.html
[Triangle2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Triangle2D.html
[ConvexPolygon2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.ConvexPolygon2D.html
[Frustum3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Frustum3D.html
[ClipSpaceZ]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.ClipSpaceZ.html
[HalfSpace2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.HalfSpace2D.html
[HalfSpace3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.HalfSpace3D.html
[Capsule2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Capsule2D.html
[Capsule3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Capsule3D.html
[Cone3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Cone3D.html
[Cone3DError]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.Cone3DError.html
[Index2DBuilder]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html
[Index3DBuilder]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index3DBuilder.html
[Index2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html
[Index3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index3D.html
[SimdIndex2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2D.html
[SimdIndex3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex3D.html
[SimdIndex2DF32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex2DF32.html
[SimdIndex3DF32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SimdIndex3DF32.html
[Index2DF32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DF32.html
[Index3DF32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index3DF32.html
[StreamIndex2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.StreamIndex2D.html
[StreamIndex3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.StreamIndex3D.html
[Index2DView]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DView.html
[Index3DView]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index3DView.html
[SearchWorkspace]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.SearchWorkspace.html
[NeighborWorkspace]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.NeighborWorkspace.html
[finish]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html#method.finish
[finish_simd]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html#method.finish_simd
[finish_simd_f32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html#method.finish_simd_f32
[finish_f32]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2DBuilder.html#method.finish_f32
[SortKey2D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.SortKey2D.html
[SortKey3D]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.SortKey3D.html
[BoundsError]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.BoundsError.html
[BuildError]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.BuildError.html
[LoadError]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/enum.LoadError.html

[aggregate]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/struct.Index2D.html#method.aggregate
