use std::ops::ControlFlow;

use crate::{
    config::DEFAULT_SEARCH_STACK_CAPACITY, geometry::Box2D, polygon::ConvexPolygon2D,
    range::visit_region, tree_access::leaf_group_range, triangle::Triangle2D,
};

use super::{Index2D, Index2DView};

impl Index2D {
    /// Item indices whose box overlaps the 2D triangle `tri`.
    ///
    /// A tight region query: like `search(tri.aabb())` but with the bounding-box
    /// corners that the triangle misses rejected during the traversal, so the
    /// result is exactly the items the triangle's filled area overlaps. Subtrees
    /// fully inside the triangle are accepted without per-item tests, so the cost
    /// stays close to the bounding-box query while the result set is tighter.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D, Triangle2D};
    ///
    /// let mut b = Index2DBuilder::new(2);
    /// b.add(Box2D::new(0.2, 0.2, 0.3, 0.3)); // inside the triangle
    /// b.add(Box2D::new(9.0, 9.0, 9.5, 9.5)); // in the bbox corner, outside the triangle
    /// let index = b.finish()?;
    ///
    /// let tri = Triangle2D::new([0.0, 0.0], [10.0, 0.0], [0.0, 10.0]);
    /// assert_eq!(index.search_triangle(tri), vec![0]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn search_triangle(&self, tri: Triangle2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_triangle_into(tri, &mut out);
        out
    }

    /// [`search_triangle`](Self::search_triangle) into a reused buffer (cleared first).
    pub fn search_triangle_into(&self, tri: Triangle2D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_triangle_with_stack(tri, &mut stack, |i| {
            out.push(i);
            ControlFlow::<()>::Continue(())
        });
    }

    /// Whether any item's box overlaps `tri`, short-circuiting on the first real
    /// hit. The triangle-tight analogue of `any(tri.aabb())`, which over-reports
    /// items that only touch the bounding box.
    pub fn any_triangle(&self, tri: Triangle2D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_triangle_with_stack(tri, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Visit each item whose box overlaps `tri`; return [`ControlFlow::Break`]
    /// from `visitor` to stop early.
    pub fn visit_triangle<B, F>(&self, tri: Triangle2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_triangle_with_stack(tri, &mut stack, visitor)
    }

    fn visit_triangle_with_stack<B, F>(
        &self,
        tri: Triangle2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        visit_region(
            self,
            stack,
            |b| tri.overlaps_box(b),
            |b| tri.contains_box(b),
            visitor,
        )
    }

    /// Diagnostics for the triangle query: `(results, nodes_visited, sat_tests,
    /// contained_subtrees)`. `sat_tests` counts `overlaps_box` calls (the cost
    /// the bounding-box query avoids), `contained_subtrees` the whole subtrees
    /// accepted without per-item tests.
    #[doc(hidden)]
    pub fn search_triangle_visited(&self, tri: Triangle2D) -> (usize, usize, usize, usize) {
        let (mut results, mut visited, mut sat, mut contained_subtrees) = (0, 0, 0, 0);
        if self.num_items == 0 {
            return (0, 0, 0, 0);
        }
        const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
        const LEVEL_MASK: usize = !CONTAINED_FLAG;
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            if contained {
                let (start, leaf_end) = leaf_group_range(self, node_index, end, level);
                results += leaf_end - start;
            } else {
                for pos in node_index..end {
                    visited += 1;
                    sat += 1;
                    let b = self.entries[pos];
                    if !tri.overlaps_box(b) {
                        continue;
                    }
                    if is_leaf {
                        results += 1;
                    } else {
                        stack.push(self.indices[pos]);
                        if tri.contains_box(b) {
                            contained_subtrees += 1;
                            stack.push((level - 1) | CONTAINED_FLAG);
                        } else {
                            stack.push(level - 1);
                        }
                    }
                }
            }
            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return (results, visited, sat, contained_subtrees);
            }
        }
    }

    /// Item indices whose box overlaps the convex polygon `poly`.
    ///
    /// The N-gon generalization of [`search_triangle`](Self::search_triangle):
    /// the exact box-vs-convex-polygon test rejects, during the traversal, the
    /// bounding-box area the polygon misses. A four-vertex polygon is a 2D view
    /// frustum / FOV trapezoid; any convex shape works. Subtrees fully inside the
    /// polygon are accepted without per-item tests, so it stays faster than
    /// collecting `search(poly_bbox)` and filtering by hand. For a triangle,
    /// [`search_triangle`](Self::search_triangle) is a touch faster (fixed three
    /// vertices, no per-edge loop).
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index2DBuilder, Box2D, ConvexPolygon2D};
    ///
    /// let mut b = Index2DBuilder::new(2);
    /// b.add(Box2D::new(1.0, 1.0, 2.0, 2.0));   // inside the trapezoid
    /// b.add(Box2D::new(0.0, 5.0, 0.5, 5.5));   // in the bbox, past the narrow end
    /// let index = b.finish()?;
    ///
    /// // A trapezoid (a 2D frustum): narrow near edge, wide far edge.
    /// let trapezoid = ConvexPolygon2D::new(vec![
    ///     [0.0, 0.0], [10.0, -4.0], [10.0, 8.0], [0.0, 3.0],
    /// ]);
    /// assert_eq!(index.search_polygon(&trapezoid), vec![0]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn search_polygon(&self, poly: &ConvexPolygon2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_polygon_into(poly, &mut out);
        out
    }

    /// [`search_polygon`](Self::search_polygon) into a reused buffer (cleared first).
    pub fn search_polygon_into(&self, poly: &ConvexPolygon2D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_polygon_with_stack(poly, &mut stack, |i| {
            out.push(i);
            ControlFlow::<()>::Continue(())
        });
    }

    /// Whether any item's box overlaps `poly`, short-circuiting on the first hit.
    pub fn any_polygon(&self, poly: &ConvexPolygon2D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_polygon_with_stack(poly, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Visit each item whose box overlaps `poly`; return [`ControlFlow::Break`]
    /// from `visitor` to stop early.
    pub fn visit_polygon<B, F>(&self, poly: &ConvexPolygon2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_polygon_with_stack(poly, &mut stack, visitor)
    }

    fn visit_polygon_with_stack<B, F>(
        &self,
        poly: &ConvexPolygon2D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        visit_region(
            self,
            stack,
            |b| poly.overlaps_box(b),
            |b| poly.contains_box(b),
            visitor,
        )
    }

    /// Diagnostics for the polygon query: `(results, nodes_visited, sat_tests,
    /// contained_subtrees)`.
    #[doc(hidden)]
    pub fn search_polygon_visited(&self, poly: &ConvexPolygon2D) -> (usize, usize, usize, usize) {
        let (mut results, mut visited, mut sat, mut contained_subtrees) = (0, 0, 0, 0);
        if self.num_items == 0 {
            return (0, 0, 0, 0);
        }
        const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);
        const LEVEL_MASK: usize = !CONTAINED_FLAG;
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let mut node_index = self.entries.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        let mut contained = false;
        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            if contained {
                let (start, leaf_end) = leaf_group_range(self, node_index, end, level);
                results += leaf_end - start;
            } else {
                for pos in node_index..end {
                    visited += 1;
                    sat += 1;
                    let b = self.entries[pos];
                    if !poly.overlaps_box(b) {
                        continue;
                    }
                    if is_leaf {
                        results += 1;
                    } else {
                        stack.push(self.indices[pos]);
                        if poly.contains_box(b) {
                            contained_subtrees += 1;
                            stack.push((level - 1) | CONTAINED_FLAG);
                        } else {
                            stack.push(level - 1);
                        }
                    }
                }
            }
            if stack.len() > 1 {
                let encoded = stack.pop().unwrap();
                level = encoded & LEVEL_MASK;
                contained = (encoded & CONTAINED_FLAG) != 0;
                node_index = stack.pop().unwrap();
            } else {
                return (results, visited, sat, contained_subtrees);
            }
        }
    }
}

impl Index2DView<'_> {
    /// Item indices whose box overlaps the 2D triangle `tri`. The zero-copy view
    /// counterpart of [`Index2D::search_triangle`].
    pub fn search_triangle(&self, tri: Triangle2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_triangle_into(tri, &mut out);
        out
    }

    /// [`search_triangle`](Self::search_triangle) into a reused buffer (cleared first).
    pub fn search_triangle_into(&self, tri: Triangle2D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_region_with_stack(
            &mut stack,
            |b| tri.overlaps_box(b),
            |b| tri.contains_box(b),
            |i| {
                out.push(i);
                ControlFlow::<()>::Continue(())
            },
        );
    }

    /// Whether any item's box overlaps `tri`, short-circuiting on the first hit.
    pub fn any_triangle(&self, tri: Triangle2D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| tri.overlaps_box(b),
            |b| tri.contains_box(b),
            |_| ControlFlow::Break(()),
        )
        .is_break()
    }

    /// Visit each item whose box overlaps `tri`; return [`ControlFlow::Break`] to stop early.
    pub fn visit_triangle<B, F>(&self, tri: Triangle2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| tri.overlaps_box(b),
            |b| tri.contains_box(b),
            visitor,
        )
    }

    /// Item indices whose box overlaps the convex polygon `poly`. The zero-copy
    /// view counterpart of [`Index2D::search_polygon`].
    pub fn search_polygon(&self, poly: &ConvexPolygon2D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_polygon_into(poly, &mut out);
        out
    }

    /// [`search_polygon`](Self::search_polygon) into a reused buffer (cleared first).
    pub fn search_polygon_into(&self, poly: &ConvexPolygon2D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_region_with_stack(
            &mut stack,
            |b| poly.overlaps_box(b),
            |b| poly.contains_box(b),
            |i| {
                out.push(i);
                ControlFlow::<()>::Continue(())
            },
        );
    }

    /// Whether any item's box overlaps `poly`, short-circuiting on the first hit.
    pub fn any_polygon(&self, poly: &ConvexPolygon2D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| poly.overlaps_box(b),
            |b| poly.contains_box(b),
            |_| ControlFlow::Break(()),
        )
        .is_break()
    }

    /// Visit each item whose box overlaps `poly`; return [`ControlFlow::Break`] to stop early.
    pub fn visit_polygon<B, F>(&self, poly: &ConvexPolygon2D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| poly.overlaps_box(b),
            |b| poly.contains_box(b),
            visitor,
        )
    }

    /// Shared region traversal over the byte-backed tree, with the contained
    /// fast path: `overlaps` prunes/leaf-tests, `contains` accepts whole subtrees.
    fn visit_region_with_stack<B>(
        &self,
        stack: &mut Vec<usize>,
        overlaps: impl Fn(Box2D) -> bool,
        contains: impl Fn(Box2D) -> bool,
        visitor: impl FnMut(usize) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        visit_region(self, stack, overlaps, contains, visitor)
    }
}
