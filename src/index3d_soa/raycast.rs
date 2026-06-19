use std::{collections::BinaryHeap, ops::ControlFlow};

use wide::f64x4;

#[cfg(target_arch = "x86_64")]
use crate::leftpack::leftpack4;
use crate::{
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    geometry::Box3D,
    neighbors::{NeighborNodeState, NeighborState, NeighborWorkspace},
    ray::Ray3D,
    traversal::{SearchWorkspace, upper_bound_level},
};

use super::{SimdIndex3D, load4};

impl SimdIndex3D {
    #[inline]
    fn box_at_soa(&self, pos: usize) -> Box3D {
        Box3D::new(
            self.min_xs[pos],
            self.min_ys[pos],
            self.min_zs[pos],
            self.max_xs[pos],
            self.max_ys[pos],
            self.max_zs[pos],
        )
    }

    /// SoA/SIMD ordered closest-hit raycast: same result as
    /// [`Index3D::raycast_closest_with`](crate::Index3D::raycast_closest_with), but the
    /// ray/AABB slab test is evaluated four children at a time with `wide::f64x4`. The
    /// dominant cost of sparse-scene ray traversal is the per-child slab arithmetic, so
    /// the SoA columns let it run 4-wide.
    ///
    /// Axis-parallel rays are handled by a masked slab test to avoid `0 * inf = NaN`
    /// at box faces: AVX-512 when available, otherwise `wide::f64x4`.
    pub fn raycast_closest_with(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // The plain AVX-512 slab is multiply-only (fastest, but not NaN-safe for
                // axis-parallel rays); the masked variant handles a zero direction.
                // SAFETY: only reached after confirming avx512f is available.
                return unsafe {
                    if ray.has_zero_direction() {
                        self.raycast_closest_avx512_masked(ray, workspace)
                    } else {
                        self.raycast_closest_avx512(ray, workspace)
                    }
                };
            }
        }
        self.raycast_closest_wide(ray, workspace)
    }

    fn raycast_closest_wide(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        let queue = &mut workspace.node_queue;
        queue.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return None;
        }
        let root = self.min_xs.len() - 1;
        let root_t = ray.enter_t(self.box_at_soa(root))?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        let ox = f64x4::splat(ray.origin.x);
        let oy = f64x4::splat(ray.origin.y);
        let oz = f64x4::splat(ray.origin.z);
        let ix = f64x4::splat(ray.inv_dir_x);
        let iy = f64x4::splat(ray.inv_dir_y);
        let iz = f64x4::splat(ray.inv_dir_z);
        let zero = f64x4::splat(0.0);
        let maxd = f64x4::splat(ray.max_distance);
        let pos_inf = f64x4::splat(f64::INFINITY);
        let neg_inf = f64x4::splat(f64::NEG_INFINITY);
        // Direction is constant for the whole ray, so degeneracy is a per-axis flag,
        // not per-lane. For a zero-direction axis the slab imposes no `t` bound when the
        // origin is inside (inclusive, so a ray exactly on a face still hits) and an
        // empty interval otherwise — computed with `blend` to avoid the `0 * inf = NaN`
        // that the multiply path would hit.
        let (zx, zy, zz) = (ray.dir_x == 0.0, ray.dir_y == 0.0, ray.dir_z == 0.0);
        let axis = |mn: f64x4, mx: f64x4, o: f64x4, inv: f64x4, degenerate: bool| {
            if degenerate {
                let inside = mn.simd_le(o) & o.simd_le(mx);
                (
                    inside.blend(neg_inf, pos_inf),
                    inside.blend(pos_inf, neg_inf),
                )
            } else {
                let t1 = (mn - o) * inv;
                let t2 = (mx - o) * inv;
                (t1.fast_min(t2), t1.fast_max(t2))
            }
        };

        while let Some(node) = queue.pop() {
            if node.dist >= best_t {
                break;
            }
            let upper = upper_bound_level(&self.level_bounds, node.index);
            let end = (node.index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node.index < self.num_items;

            let mut pos = node.index;
            while pos + 4 <= end {
                let (nx, fx) = axis(
                    load4(&self.min_xs, pos),
                    load4(&self.max_xs, pos),
                    ox,
                    ix,
                    zx,
                );
                let (ny, fy) = axis(
                    load4(&self.min_ys, pos),
                    load4(&self.max_ys, pos),
                    oy,
                    iy,
                    zy,
                );
                let (nz, fz) = axis(
                    load4(&self.min_zs, pos),
                    load4(&self.max_zs, pos),
                    oz,
                    iz,
                    zz,
                );
                let near = nx.fast_max(ny).fast_max(nz).fast_max(zero);
                let far = fx.fast_min(fy).fast_min(fz).fast_min(maxd);
                let bits = near.simd_le(far).to_bitmask();
                if bits != 0 {
                    let tn = near.to_array();
                    // `k` indexes `tn` and selects the mask bit, so a range loop is clearest.
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..4 {
                        if bits & (1 << k) != 0 && tn[k] < best_t {
                            if is_leaf {
                                best_t = tn[k];
                                best_index = Some(self.indices[pos + k]);
                            } else {
                                queue.push(NeighborNodeState::new(self.indices[pos + k], tn[k]));
                            }
                        }
                    }
                }
                pos += 4;
            }
            while pos < end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos))
                    && t < best_t
                {
                    if is_leaf {
                        best_t = t;
                        best_index = Some(self.indices[pos]);
                    } else {
                        queue.push(NeighborNodeState::new(self.indices[pos], t));
                    }
                }
                pos += 1;
            }
        }

        best_index.map(|index| (index, best_t))
    }

    /// AVX-512 closest-hit: the slab test runs eight children at a time. With
    /// `node_size == 8` that is exactly one vector op per node.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn raycast_closest_avx512(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        use std::arch::x86_64::*;

        let queue = &mut workspace.node_queue;
        queue.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return None;
        }
        let root = self.min_xs.len() - 1;
        let root_t = ray.enter_t(self.box_at_soa(root))?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        let ox = _mm512_set1_pd(ray.origin.x);
        let oy = _mm512_set1_pd(ray.origin.y);
        let oz = _mm512_set1_pd(ray.origin.z);
        let ix = _mm512_set1_pd(ray.inv_dir_x);
        let iy = _mm512_set1_pd(ray.inv_dir_y);
        let iz = _mm512_set1_pd(ray.inv_dir_z);
        let zero = _mm512_setzero_pd();
        let maxd = _mm512_set1_pd(ray.max_distance);

        while let Some(node) = queue.pop() {
            if node.dist >= best_t {
                break;
            }
            let upper = upper_bound_level(&self.level_bounds, node.index);
            let end = (node.index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node.index < self.num_items;
            let mut pos = node.index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end <= len`, so all eight lanes are in bounds.
                let (mnx, mxx, mny, mxy, mnz, mxz) = unsafe {
                    (
                        _mm512_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_zs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_zs.as_ptr().add(pos)),
                    )
                };
                let t1x = _mm512_mul_pd(_mm512_sub_pd(mnx, ox), ix);
                let t2x = _mm512_mul_pd(_mm512_sub_pd(mxx, ox), ix);
                let t1y = _mm512_mul_pd(_mm512_sub_pd(mny, oy), iy);
                let t2y = _mm512_mul_pd(_mm512_sub_pd(mxy, oy), iy);
                let t1z = _mm512_mul_pd(_mm512_sub_pd(mnz, oz), iz);
                let t2z = _mm512_mul_pd(_mm512_sub_pd(mxz, oz), iz);
                let near = _mm512_max_pd(
                    _mm512_max_pd(
                        _mm512_max_pd(_mm512_min_pd(t1x, t2x), _mm512_min_pd(t1y, t2y)),
                        _mm512_min_pd(t1z, t2z),
                    ),
                    zero,
                );
                let far = _mm512_min_pd(
                    _mm512_min_pd(
                        _mm512_min_pd(_mm512_max_pd(t1x, t2x), _mm512_max_pd(t1y, t2y)),
                        _mm512_max_pd(t1z, t2z),
                    ),
                    maxd,
                );
                let mut bits: u8 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(near, far);
                if bits != 0 {
                    let mut tn = [0.0f64; 8];
                    // SAFETY: `tn` holds eight `f64`, matching the 512-bit store.
                    unsafe { _mm512_storeu_pd(tn.as_mut_ptr(), near) };
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if tn[k] < best_t {
                            if is_leaf {
                                best_t = tn[k];
                                best_index = Some(self.indices[pos + k]);
                            } else {
                                queue.push(NeighborNodeState::new(self.indices[pos + k], tn[k]));
                            }
                        }
                    }
                }
                pos += 8;
            }
            while pos < end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos))
                    && t < best_t
                {
                    if is_leaf {
                        best_t = t;
                        best_index = Some(self.indices[pos]);
                    } else {
                        queue.push(NeighborNodeState::new(self.indices[pos], t));
                    }
                }
                pos += 1;
            }
        }

        best_index.map(|index| (index, best_t))
    }

    /// AVX-512 closest-hit for axis-parallel rays: a zero-direction axis is handled with
    /// `_mm512_mask_blend_pd` over an inclusive inside-test, so it is NaN-safe even when
    /// the ray grazes a box face. Only invoked for rays with a zero direction component.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn raycast_closest_avx512_masked(
        &self,
        ray: Ray3D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        use std::arch::x86_64::*;

        let queue = &mut workspace.node_queue;
        queue.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return None;
        }
        let root = self.min_xs.len() - 1;
        let root_t = ray.enter_t(self.box_at_soa(root))?;
        let mut best_t = ray.max_distance;
        let mut best_index = None;
        queue.push(NeighborNodeState::new(root, root_t));

        let ox = _mm512_set1_pd(ray.origin.x);
        let oy = _mm512_set1_pd(ray.origin.y);
        let oz = _mm512_set1_pd(ray.origin.z);
        let ix = _mm512_set1_pd(ray.inv_dir_x);
        let iy = _mm512_set1_pd(ray.inv_dir_y);
        let iz = _mm512_set1_pd(ray.inv_dir_z);
        let zero = _mm512_setzero_pd();
        let maxd = _mm512_set1_pd(ray.max_distance);
        let pos_inf = _mm512_set1_pd(f64::INFINITY);
        let neg_inf = _mm512_set1_pd(f64::NEG_INFINITY);
        let (zx, zy, zz) = (ray.dir_x == 0.0, ray.dir_y == 0.0, ray.dir_z == 0.0);

        // `(near, far)` interval for one axis across eight children.
        let axis = |mn, mx, o, inv, degenerate: bool| {
            if degenerate {
                let inside = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(mn, o)
                    & _mm512_cmp_pd_mask::<_CMP_LE_OQ>(o, mx);
                (
                    _mm512_mask_blend_pd(inside, pos_inf, neg_inf),
                    _mm512_mask_blend_pd(inside, neg_inf, pos_inf),
                )
            } else {
                let t1 = _mm512_mul_pd(_mm512_sub_pd(mn, o), inv);
                let t2 = _mm512_mul_pd(_mm512_sub_pd(mx, o), inv);
                (_mm512_min_pd(t1, t2), _mm512_max_pd(t1, t2))
            }
        };

        while let Some(node) = queue.pop() {
            if node.dist >= best_t {
                break;
            }
            let upper = upper_bound_level(&self.level_bounds, node.index);
            let end = (node.index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node.index < self.num_items;

            let mut pos = node.index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end <= len`, so all eight lanes are in bounds.
                let (nx, fx, ny, fy, nz, fz) = unsafe {
                    let (mnx, mxx, mny, mxy, mnz, mxz) = (
                        _mm512_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_zs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_zs.as_ptr().add(pos)),
                    );
                    let (nx, fx) = axis(mnx, mxx, ox, ix, zx);
                    let (ny, fy) = axis(mny, mxy, oy, iy, zy);
                    let (nz, fz) = axis(mnz, mxz, oz, iz, zz);
                    (nx, fx, ny, fy, nz, fz)
                };
                let near = _mm512_max_pd(_mm512_max_pd(_mm512_max_pd(nx, ny), nz), zero);
                let far = _mm512_min_pd(_mm512_min_pd(_mm512_min_pd(fx, fy), fz), maxd);
                let mut bits: u8 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(near, far);
                if bits != 0 {
                    let mut tn = [0.0f64; 8];
                    // SAFETY: `tn` holds eight `f64`, matching the 512-bit store.
                    unsafe { _mm512_storeu_pd(tn.as_mut_ptr(), near) };
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if tn[k] < best_t {
                            if is_leaf {
                                best_t = tn[k];
                                best_index = Some(self.indices[pos + k]);
                            } else {
                                queue.push(NeighborNodeState::new(self.indices[pos + k], tn[k]));
                            }
                        }
                    }
                }
                pos += 8;
            }
            while pos < end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos))
                    && t < best_t
                {
                    if is_leaf {
                        best_t = t;
                        best_index = Some(self.indices[pos]);
                    } else {
                        queue.push(NeighborNodeState::new(self.indices[pos], t));
                    }
                }
                pos += 1;
            }
        }

        best_index.map(|index| (index, best_t))
    }

    /// Return the nearest item whose box the ray segment enters, as
    /// `(item index, entry t)`, or `None` when the segment hits nothing.
    ///
    /// Nodes are visited front-to-back by entry distance and pruned once a
    /// closer hit is known, so the cost is roughly independent of
    /// `max_distance` after the first hit. `t` is `0.0` when the ray origin
    /// starts inside the item's box.
    pub fn raycast_closest(&self, ray: Ray3D) -> Option<(usize, f64)> {
        let mut workspace = NeighborWorkspace::new();
        self.raycast_closest_with(ray, &mut workspace)
    }

    /// Return the indices of all items whose boxes the ray segment touches.
    pub fn raycast(&self, ray: Ray3D) -> Vec<usize> {
        let mut results = Vec::new();
        self.raycast_into(ray, &mut results);
        results
    }

    /// Raycast with a reusable result buffer.
    pub fn raycast_into(&self, ray: Ray3D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.raycast_into_stack(ray, results, &mut stack);
    }

    /// Raycast with reusable result and traversal buffers.
    pub fn raycast_with<'a>(&self, ray: Ray3D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.raycast_into_stack(ray, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Buffer-explicit raycast (mirrors `search_into_stack`). The per-node slab
    /// test is vectorized: AVX-512 (eight children at a time) for non-degenerate
    /// rays where available, otherwise `wide::f64x4`. Axis-parallel rays always
    /// take the `wide` path, whose `blend` kernel is NaN-safe at box faces.
    #[doc(hidden)]
    pub fn raycast_into_stack(&self, ray: Ray3D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        #[cfg(target_arch = "x86_64")]
        {
            if !ray.has_zero_direction() {
                if std::is_x86_feature_detected!("avx512f") {
                    // SAFETY: reached only after confirming avx512f is available.
                    unsafe { self.raycast_collect_avx512(ray, results, stack) };
                    return;
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: reached only after confirming avx2 is available.
                    unsafe { self.raycast_collect_avx2(ray, results, stack) };
                    return;
                }
            }
        }
        self.raycast_collect_wide(ray, results, stack);
    }

    /// Force the `wide` all-hits raycast path (doc-hidden; benchmarks/tests).
    #[doc(hidden)]
    pub fn raycast_wide_into(&self, ray: Ray3D, results: &mut Vec<usize>) {
        let mut stack = Vec::new();
        self.raycast_collect_wide(ray, results, &mut stack);
    }

    /// Force the AVX2 all-hits raycast path (doc-hidden; benchmarks/tests).
    #[doc(hidden)]
    pub fn raycast_avx2_into(&self, ray: Ray3D, results: &mut Vec<usize>) {
        let mut stack = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if !ray.has_zero_direction() && std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the avx2 feature check.
                unsafe { self.raycast_collect_avx2(ray, results, &mut stack) };
                return;
            }
        }
        self.raycast_collect_wide(ray, results, &mut stack);
    }

    fn raycast_collect_wide(&self, ray: Ray3D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        results.clear();
        stack.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return;
        }

        let ox = f64x4::splat(ray.origin.x);
        let oy = f64x4::splat(ray.origin.y);
        let oz = f64x4::splat(ray.origin.z);
        let ix = f64x4::splat(ray.inv_dir_x);
        let iy = f64x4::splat(ray.inv_dir_y);
        let iz = f64x4::splat(ray.inv_dir_z);
        let zero = f64x4::splat(0.0);
        let maxd = f64x4::splat(ray.max_distance);
        let pos_inf = f64x4::splat(f64::INFINITY);
        let neg_inf = f64x4::splat(f64::NEG_INFINITY);
        // A zero-direction axis imposes no `t` bound when the origin is inside
        // (inclusive, so a ray on a face still hits) and an empty interval
        // otherwise, computed with `blend` to dodge the `0 * inf = NaN` of the
        // multiply path.
        let (zx, zy, zz) = (ray.dir_x == 0.0, ray.dir_y == 0.0, ray.dir_z == 0.0);
        let axis = |mn: f64x4, mx: f64x4, o: f64x4, inv: f64x4, degenerate: bool| {
            if degenerate {
                let inside = mn.simd_le(o) & o.simd_le(mx);
                (
                    inside.blend(neg_inf, pos_inf),
                    inside.blend(pos_inf, neg_inf),
                )
            } else {
                let t1 = (mn - o) * inv;
                let t2 = (mx - o) * inv;
                (t1.fast_min(t2), t1.fast_max(t2))
            }
        };

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let child_level = level.wrapping_sub(1);

            let mut pos = node_index;
            while pos + 4 <= end {
                let (nx, fx) = axis(
                    load4(&self.min_xs, pos),
                    load4(&self.max_xs, pos),
                    ox,
                    ix,
                    zx,
                );
                let (ny, fy) = axis(
                    load4(&self.min_ys, pos),
                    load4(&self.max_ys, pos),
                    oy,
                    iy,
                    zy,
                );
                let (nz, fz) = axis(
                    load4(&self.min_zs, pos),
                    load4(&self.max_zs, pos),
                    oz,
                    iz,
                    zz,
                );
                let near = nx.fast_max(ny).fast_max(nz).fast_max(zero);
                let far = fx.fast_min(fy).fast_min(fz).fast_min(maxd);
                let mut bits = near.simd_le(far).to_bitmask();
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let index = self.indices[pos + k];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
                    }
                }
                pos += 4;
            }
            while pos < end {
                if ray.intersects_box(self.box_at_soa(pos)) {
                    let index = self.indices[pos];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }

    /// AVX-512 all-hits slab test, eight children at a time. Only called for
    /// non-degenerate rays (no zero direction component), so the multiply-only
    /// slab is NaN-safe.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn raycast_collect_avx512(
        &self,
        ray: Ray3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        use std::arch::x86_64::*;

        results.clear();
        stack.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return;
        }

        let ox = _mm512_set1_pd(ray.origin.x);
        let oy = _mm512_set1_pd(ray.origin.y);
        let oz = _mm512_set1_pd(ray.origin.z);
        let ix = _mm512_set1_pd(ray.inv_dir_x);
        let iy = _mm512_set1_pd(ray.inv_dir_y);
        let iz = _mm512_set1_pd(ray.inv_dir_z);
        let zero = _mm512_setzero_pd();
        let maxd = _mm512_set1_pd(ray.max_distance);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let child_level = level.wrapping_sub(1);
            if is_leaf {
                results.reserve(end - node_index);
            }

            let mut pos = node_index;
            while pos + 8 <= end {
                // SAFETY: `pos + 8 <= end <= len`, so all eight lanes are in bounds.
                let (mnx, mxx, mny, mxy, mnz, mxz) = unsafe {
                    (
                        _mm512_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.min_zs.as_ptr().add(pos)),
                        _mm512_loadu_pd(self.max_zs.as_ptr().add(pos)),
                    )
                };
                let t1x = _mm512_mul_pd(_mm512_sub_pd(mnx, ox), ix);
                let t2x = _mm512_mul_pd(_mm512_sub_pd(mxx, ox), ix);
                let t1y = _mm512_mul_pd(_mm512_sub_pd(mny, oy), iy);
                let t2y = _mm512_mul_pd(_mm512_sub_pd(mxy, oy), iy);
                let t1z = _mm512_mul_pd(_mm512_sub_pd(mnz, oz), iz);
                let t2z = _mm512_mul_pd(_mm512_sub_pd(mxz, oz), iz);
                let near = _mm512_max_pd(
                    _mm512_max_pd(
                        _mm512_max_pd(_mm512_min_pd(t1x, t2x), _mm512_min_pd(t1y, t2y)),
                        _mm512_min_pd(t1z, t2z),
                    ),
                    zero,
                );
                let far = _mm512_min_pd(
                    _mm512_min_pd(
                        _mm512_min_pd(_mm512_max_pd(t1x, t2x), _mm512_max_pd(t1y, t2y)),
                        _mm512_max_pd(t1z, t2z),
                    ),
                    maxd,
                );
                let mut bits: u8 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(near, far);
                if is_leaf {
                    // VPCOMPRESSQ pack the hit indices (capacity reserved above).
                    // SAFETY: `pos + 8 <= end <= indices.len()`; `results` has
                    // `end - node_index` slack.
                    unsafe {
                        let dst = results.as_mut_ptr().add(results.len()) as *mut i64;
                        let vidx = _mm512_loadu_epi64(self.indices.as_ptr().add(pos) as *const i64);
                        _mm512_mask_compressstoreu_epi64(dst, bits, vidx);
                        results.set_len(results.len() + bits.count_ones() as usize);
                    }
                } else {
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        stack.push(self.indices[pos + k]);
                        stack.push(child_level);
                    }
                }
                pos += 8;
            }
            while pos < end {
                if ray.intersects_box(self.box_at_soa(pos)) {
                    let index = self.indices[pos];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }

    /// AVX2 all-hits raycast (4-wide slab test, AVX2 left-pack leaf collection).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn raycast_collect_avx2(
        &self,
        ray: Ray3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        use std::arch::x86_64::*;

        results.clear();
        stack.clear();
        if self.num_items == 0 || ray.max_distance < 0.0 || ray.max_distance.is_nan() {
            return;
        }

        let ox = _mm256_set1_pd(ray.origin.x);
        let oy = _mm256_set1_pd(ray.origin.y);
        let oz = _mm256_set1_pd(ray.origin.z);
        let ix = _mm256_set1_pd(ray.inv_dir_x);
        let iy = _mm256_set1_pd(ray.inv_dir_y);
        let iz = _mm256_set1_pd(ray.inv_dir_z);
        let zero = _mm256_setzero_pd();
        let maxd = _mm256_set1_pd(ray.max_distance);

        let mut node_index = self.min_xs.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            let child_level = level.wrapping_sub(1);
            if is_leaf {
                results.reserve(end - node_index + 4);
            }

            let mut pos = node_index;
            while pos + 4 <= end {
                // SAFETY: `pos + 4 <= end <= len`, so all four lanes are in bounds.
                let (mnx, mxx, mny, mxy, mnz, mxz) = unsafe {
                    (
                        _mm256_loadu_pd(self.min_xs.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.max_xs.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.min_ys.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.max_ys.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.min_zs.as_ptr().add(pos)),
                        _mm256_loadu_pd(self.max_zs.as_ptr().add(pos)),
                    )
                };
                let t1x = _mm256_mul_pd(_mm256_sub_pd(mnx, ox), ix);
                let t2x = _mm256_mul_pd(_mm256_sub_pd(mxx, ox), ix);
                let t1y = _mm256_mul_pd(_mm256_sub_pd(mny, oy), iy);
                let t2y = _mm256_mul_pd(_mm256_sub_pd(mxy, oy), iy);
                let t1z = _mm256_mul_pd(_mm256_sub_pd(mnz, oz), iz);
                let t2z = _mm256_mul_pd(_mm256_sub_pd(mxz, oz), iz);
                let near = _mm256_max_pd(
                    _mm256_max_pd(
                        _mm256_max_pd(_mm256_min_pd(t1x, t2x), _mm256_min_pd(t1y, t2y)),
                        _mm256_min_pd(t1z, t2z),
                    ),
                    zero,
                );
                let far = _mm256_min_pd(
                    _mm256_min_pd(
                        _mm256_min_pd(_mm256_max_pd(t1x, t2x), _mm256_max_pd(t1y, t2y)),
                        _mm256_max_pd(t1z, t2z),
                    ),
                    maxd,
                );
                let mut bits = _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LE_OQ>(near, far)) as usize;
                if is_leaf {
                    if bits != 0 {
                        // SAFETY: `pos + 4 <= end <= indices.len()`; `results` has
                        // `end - node_index + 4` slack reserved.
                        unsafe {
                            let added = leftpack4(
                                self.indices.as_ptr().add(pos),
                                bits as u32,
                                results.as_mut_ptr().add(results.len()),
                            );
                            results.set_len(results.len() + added);
                        }
                    }
                } else {
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        stack.push(self.indices[pos + k]);
                        stack.push(child_level);
                    }
                }
                pos += 4;
            }
            while pos < end {
                if ray.intersects_box(self.box_at_soa(pos)) {
                    let index = self.indices[pos];
                    if is_leaf {
                        results.push(index);
                    } else {
                        stack.push(index);
                        stack.push(child_level);
                    }
                }
                pos += 1;
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return;
            }
        }
    }
}

impl SimdIndex3D {
    /// Visit items in nondecreasing entry-`t` order along the ray segment.
    ///
    /// The visitor receives `(item index, entry t)`. Return
    /// [`ControlFlow::Break`] to stop early - for example after the first N
    /// occluders. `t` is `0.0` when the ray origin starts inside a box.
    pub fn visit_raycast<B, F>(&self, ray: Ray3D, mut visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        if self.num_items == 0 {
            return ControlFlow::Continue(());
        }

        let mut node_index = self.min_xs.len() - 1;
        loop {
            let upper = upper_bound_level(&self.level_bounds, node_index);
            let end = (node_index + self.node_size).min(self.level_bounds[upper]);
            let is_leaf = node_index < self.num_items;

            for pos in node_index..end {
                if let Some(t) = ray.enter_t(self.box_at_soa(pos)) {
                    queue.push(NeighborState::new(self.indices[pos], is_leaf, t));
                }
            }

            let mut continue_search = false;
            while let Some(state) = queue.pop() {
                if state.is_leaf {
                    visitor(state.index, state.dist)?;
                } else {
                    node_index = state.index;
                    continue_search = true;
                    break;
                }
            }
            if !continue_search {
                return ControlFlow::Continue(());
            }
        }
    }
}
