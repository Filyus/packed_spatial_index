use std::{collections::BinaryHeap, ops::ControlFlow};

use crate::{
    config::{DEFAULT_NEIGHBOR_QUEUE_CAPACITY, DEFAULT_SEARCH_STACK_CAPACITY},
    neighbors::NeighborWorkspace,
    ray::Ray2D,
    raycast as scalar_raycast,
    traversal::{SearchWorkspace, upper_bound_level},
};

use super::{Index2D, Index2DView};

impl Index2D {
    /// Return the indices of all items whose boxes the ray segment touches.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Box2D, Index2DBuilder, Point2D, Ray2D};
    ///
    /// let mut builder = Index2DBuilder::new(2);
    /// builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    /// builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    /// let index = builder.finish().unwrap();
    ///
    /// let ray = Ray2D::new(Point2D::new(-1.0, 0.5), 1.0, 0.0, 10.0);
    /// assert_eq!(index.raycast(ray), vec![0]);
    /// assert_eq!(index.raycast_closest(ray), Some((0, 1.0)));
    /// ```
    pub fn raycast(&self, ray: Ray2D) -> Vec<usize> {
        let mut results = Vec::new();
        self.raycast_into(ray, &mut results);
        results
    }

    /// Raycast with a reusable result buffer.
    pub fn raycast_into(&self, ray: Ray2D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.raycast_into_stack(ray, results, &mut stack);
    }

    /// Raycast with reusable result and traversal buffers.
    pub fn raycast_with<'a>(&self, ray: Ray2D, workspace: &'a mut SearchWorkspace) -> &'a [usize] {
        self.raycast_into_stack(ray, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Buffer-explicit raycast (mirrors `search_into_stack`).
    #[doc(hidden)]
    pub fn raycast_into_stack(&self, ray: Ray2D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        scalar_raycast::collect_hits(
            self.entries.len(),
            self.num_items,
            self.node_size,
            self.level_bounds.len(),
            |level| self.level_bounds[level],
            |pos| self.indices[pos],
            |pos| ray.intersects_box(self.entries[pos]),
            true,
            results,
            stack,
        );
    }

    /// Return the nearest item whose box the ray segment enters, as
    /// `(item index, entry t)`, or `None` when the segment hits nothing.
    ///
    /// Nodes are visited front-to-back by entry distance and pruned once a
    /// closer hit is known, so the cost is roughly independent of
    /// `max_distance` after the first hit. `t` is `0.0` when the ray origin
    /// starts inside the item's box, and is measured in units of the ray
    /// direction length (see [`Ray2D::new`]).
    pub fn raycast_closest(&self, ray: Ray2D) -> Option<(usize, f64)> {
        let mut workspace = NeighborWorkspace::new();
        self.raycast_closest_with(ray, &mut workspace)
    }

    /// Closest-hit raycast with a reusable priority-queue workspace.
    pub fn raycast_closest_with(
        &self,
        ray: Ray2D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        scalar_raycast::closest_hit(
            self.entries.len(),
            self.num_items,
            self.node_size,
            ray.max_distance,
            |node| self.level_bounds[upper_bound_level(&self.level_bounds, node)],
            |pos| self.indices[pos],
            |pos| ray.enter_t(self.entries[pos]),
            &mut workspace.node_queue,
        )
    }

    /// Visit items in nondecreasing entry-`t` order along the ray segment.
    ///
    /// The visitor receives `(item index, entry t)`. Return
    /// [`ControlFlow::Break`] to stop early - for example after the first N
    /// occluders. `t` is `0.0` when the ray origin starts inside a box.
    pub fn visit_raycast<B, F>(&self, ray: Ray2D, mut visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        scalar_raycast::visit_hits(
            self.entries.len(),
            self.num_items,
            self.node_size,
            |node| self.level_bounds[upper_bound_level(&self.level_bounds, node)],
            |pos| self.indices[pos],
            |pos| ray.enter_t(self.entries[pos]),
            &mut queue,
            &mut visitor,
        )
    }
}

impl Index2DView<'_> {
    /// Return the indices of all items whose boxes the ray segment touches.
    pub fn raycast(&self, ray: Ray2D) -> Vec<usize> {
        let mut results = Vec::new();
        self.raycast_into(ray, &mut results);
        results
    }

    /// Raycast with a reusable result buffer.
    pub fn raycast_into(&self, ray: Ray2D, results: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.raycast_into_stack(ray, results, &mut stack);
    }

    /// Raycast with reusable result and traversal buffers.
    pub fn raycast_with<'na>(
        &self,
        ray: Ray2D,
        workspace: &'na mut SearchWorkspace,
    ) -> &'na [usize] {
        self.raycast_into_stack(ray, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    /// Buffer-explicit raycast (mirrors `search_into_stack`).
    #[doc(hidden)]
    pub fn raycast_into_stack(&self, ray: Ray2D, results: &mut Vec<usize>, stack: &mut Vec<usize>) {
        scalar_raycast::collect_hits(
            self.num_nodes,
            self.num_items,
            self.node_size,
            self.level_count,
            |level| self.level_bound_unchecked(level),
            |pos| self.index_at_unchecked(pos),
            |pos| ray.intersects_box(self.entry_at_unchecked(pos)),
            false,
            results,
            stack,
        );
    }

    /// Return the nearest item whose box the ray segment enters, as
    /// `(item index, entry t)`, or `None` when the segment hits nothing.
    /// See [`Index2D::raycast_closest`](crate::Index2D::raycast_closest).
    pub fn raycast_closest(&self, ray: Ray2D) -> Option<(usize, f64)> {
        let mut workspace = NeighborWorkspace::new();
        self.raycast_closest_with(ray, &mut workspace)
    }

    /// Closest-hit raycast with a reusable priority-queue workspace.
    pub fn raycast_closest_with(
        &self,
        ray: Ray2D,
        workspace: &mut NeighborWorkspace,
    ) -> Option<(usize, f64)> {
        scalar_raycast::closest_hit(
            self.num_nodes,
            self.num_items,
            self.node_size,
            ray.max_distance,
            |node| self.level_bound_unchecked(self.upper_bound_level(node)),
            |pos| self.index_at_unchecked(pos),
            |pos| ray.enter_t(self.entry_at_unchecked(pos)),
            &mut workspace.node_queue,
        )
    }

    /// Visit items in nondecreasing entry-`t` order along the ray segment.
    ///
    /// The visitor receives `(item index, entry t)`. Return
    /// [`ControlFlow::Break`] to stop early - for example after the first N
    /// occluders. `t` is `0.0` when the ray origin starts inside a box.
    pub fn visit_raycast<B, F>(&self, ray: Ray2D, mut visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize, f64) -> ControlFlow<B>,
    {
        let mut queue = BinaryHeap::with_capacity(DEFAULT_NEIGHBOR_QUEUE_CAPACITY);
        scalar_raycast::visit_hits(
            self.num_nodes,
            self.num_items,
            self.node_size,
            |node| self.level_bound_unchecked(self.upper_bound_level(node)),
            |pos| self.index_at_unchecked(pos),
            |pos| ray.enter_t(self.entry_at_unchecked(pos)),
            &mut queue,
            &mut visitor,
        )
    }
}
