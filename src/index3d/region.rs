use std::ops::ControlFlow;

use crate::{
    config::DEFAULT_SEARCH_STACK_CAPACITY, frustum::Frustum3D, geometry::Box3D,
    range::visit_region, tree_access::leaf_group_range,
};

use super::{Index3D, Index3DView};

impl Index3D {
    /// Item indices whose box overlaps the view frustum `frustum`.
    ///
    /// A **conservative** culling query: it returns every item whose box is inside
    /// or crosses the frustum, and may include a few boxes that lie just outside a
    /// frustum edge or corner (the standard p-vertex test). It never drops a
    /// visible box. Far tighter than `search` over the frustum's bounding box,
    /// which pulls in the whole corner volume the frustum's slanted sides miss.
    ///
    /// # Example
    ///
    /// ```
    /// use packed_spatial_index::{Index3DBuilder, Box3D, Frustum3D};
    ///
    /// let mut b = Index3DBuilder::new(2);
    /// b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)); // inside the unit cube
    /// b.add(Box3D::new(9.0, 9.0, 9.0, 9.5, 9.5, 9.5)); // far outside
    /// let index = b.finish()?;
    ///
    /// // Six axis-aligned planes bounding the unit cube [0,2]^3.
    /// let frustum = Frustum3D::from_planes([
    ///     [1.0, 0.0, 0.0, 0.0],   // x >= 0
    ///     [-1.0, 0.0, 0.0, 2.0],  // x <= 2
    ///     [0.0, 1.0, 0.0, 0.0],   // y >= 0
    ///     [0.0, -1.0, 0.0, 2.0],  // y <= 2
    ///     [0.0, 0.0, 1.0, 0.0],   // z >= 0
    ///     [0.0, 0.0, -1.0, 2.0],  // z <= 2
    /// ]);
    /// assert_eq!(index.search_frustum(frustum), vec![0]);
    /// # Ok::<(), packed_spatial_index::BuildError>(())
    /// ```
    pub fn search_frustum(&self, frustum: Frustum3D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_frustum_into(frustum, &mut out);
        out
    }

    /// [`search_frustum`](Self::search_frustum) into a reused buffer (cleared first).
    pub fn search_frustum_into(&self, frustum: Frustum3D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_frustum_with_stack(frustum, &mut stack, |i| {
            out.push(i);
            ControlFlow::<()>::Continue(())
        });
    }

    /// Whether any item's box overlaps `frustum`, short-circuiting on the first
    /// hit (conservative, like [`search_frustum`](Self::search_frustum)).
    pub fn any_frustum(&self, frustum: Frustum3D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_frustum_with_stack(frustum, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    /// Visit each item whose box overlaps `frustum`; return [`ControlFlow::Break`]
    /// from `visitor` to stop early (conservative).
    pub fn visit_frustum<B, F>(&self, frustum: Frustum3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_frustum_with_stack(frustum, &mut stack, visitor)
    }

    fn visit_frustum_with_stack<B, F>(
        &self,
        frustum: Frustum3D,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        visit_region(
            self,
            stack,
            |b| frustum.overlaps_box(b),
            |b| frustum.contains_box(b),
            visitor,
        )
    }

    /// Diagnostics for the frustum query: `(results, nodes_visited, plane_tests,
    /// contained_subtrees)`. `plane_tests` counts `overlaps_box` calls (the cost
    /// the bounding-box query avoids), `contained_subtrees` the whole subtrees
    /// accepted without per-item tests.
    #[doc(hidden)]
    pub fn search_frustum_visited(&self, frustum: Frustum3D) -> (usize, usize, usize, usize) {
        let (mut results, mut visited, mut planes, mut contained_subtrees) = (0, 0, 0, 0);
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
                    planes += 1;
                    let b = self.entries[pos];
                    if !frustum.overlaps_box(b) {
                        continue;
                    }
                    if is_leaf {
                        results += 1;
                    } else {
                        stack.push(self.indices[pos]);
                        if frustum.contains_box(b) {
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
                return (results, visited, planes, contained_subtrees);
            }
        }
    }
}

impl Index3DView<'_> {
    /// Item indices whose box overlaps the view frustum `frustum`. The zero-copy
    /// view counterpart of [`Index3D::search_frustum`] (conservative culling).
    pub fn search_frustum(&self, frustum: Frustum3D) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_frustum_into(frustum, &mut out);
        out
    }

    /// [`search_frustum`](Self::search_frustum) into a reused buffer (cleared first).
    pub fn search_frustum_into(&self, frustum: Frustum3D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = self.visit_region_with_stack(
            &mut stack,
            |b| frustum.overlaps_box(b),
            |b| frustum.contains_box(b),
            |i| {
                out.push(i);
                ControlFlow::<()>::Continue(())
            },
        );
    }

    /// Whether any item's box overlaps `frustum`, short-circuiting (conservative).
    pub fn any_frustum(&self, frustum: Frustum3D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| frustum.overlaps_box(b),
            |b| frustum.contains_box(b),
            |_| ControlFlow::Break(()),
        )
        .is_break()
    }

    /// Visit each item whose box overlaps `frustum`; return [`ControlFlow::Break`]
    /// to stop early (conservative).
    pub fn visit_frustum<B, F>(&self, frustum: Frustum3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        self.visit_region_with_stack(
            &mut stack,
            |b| frustum.overlaps_box(b),
            |b| frustum.contains_box(b),
            visitor,
        )
    }

    /// Shared region traversal over the byte-backed tree, with the contained
    /// fast path: `overlaps` prunes/leaf-tests, `contains` accepts whole subtrees.
    fn visit_region_with_stack<B>(
        &self,
        stack: &mut Vec<usize>,
        overlaps: impl Fn(Box3D) -> bool,
        contains: impl Fn(Box3D) -> bool,
        visitor: impl FnMut(usize) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        visit_region(self, stack, overlaps, contains, visitor)
    }
}
