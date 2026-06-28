use std::ops::ControlFlow;

use crate::{
    config::DEFAULT_SEARCH_STACK_CAPACITY,
    geometry::{Box3D, Overlaps3D},
    range::visit_region,
    traversal::SearchWorkspace,
};

use super::{Index3D, Index3DView, RegionSearch3DIter, Search3DIter};

#[doc(hidden)]
pub trait SearchQuery3D: Sized {
    type Iter<'a>: Iterator<Item = usize> + std::iter::FusedIterator
    where
        Self: 'a;

    fn search_into_index(self, index: &Index3D, out: &mut Vec<usize>);
    fn search_with_index<'a>(
        self,
        index: &Index3D,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize];
    fn any_index(self, index: &Index3D) -> bool;
    fn first_index(self, index: &Index3D) -> Option<usize>;
    fn visit_index<B, F>(self, index: &Index3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>;
    fn search_iter_index<'a>(self, index: &'a Index3D) -> Self::Iter<'a>
    where
        Self: 'a;

    fn search_into_view(self, view: &Index3DView<'_>, out: &mut Vec<usize>);
    fn search_with_view<'a>(
        self,
        view: &Index3DView<'_>,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize];
    fn any_view(self, view: &Index3DView<'_>) -> bool;
    fn first_view(self, view: &Index3DView<'_>) -> Option<usize>;
    fn visit_view<B, F>(self, view: &Index3DView<'_>, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>;
}

impl SearchQuery3D for Box3D {
    type Iter<'a> = Search3DIter<'a>;

    #[inline]
    fn search_into_index(self, index: &Index3D, out: &mut Vec<usize>) {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        index.search_into_stack(self, out, &mut stack);
    }

    #[inline]
    fn search_with_index<'a>(
        self,
        index: &Index3D,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize] {
        index.search_into_stack(self, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    #[inline]
    fn any_index(self, index: &Index3D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        index
            .visit_with_stack(self, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    #[inline]
    fn first_index(self, index: &Index3D) -> Option<usize> {
        match self.visit_index(index, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    #[inline]
    fn visit_index<B, F>(self, index: &Index3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        index.visit_with_stack(self, &mut stack, visitor)
    }

    #[inline]
    fn search_iter_index<'a>(self, index: &'a Index3D) -> Self::Iter<'a>
    where
        Self: 'a,
    {
        Search3DIter::new(index, self)
    }

    #[inline]
    fn search_into_view(self, view: &Index3DView<'_>, out: &mut Vec<usize>) {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        view.search_into_stack(self, out, &mut stack);
    }

    #[inline]
    fn search_with_view<'a>(
        self,
        view: &Index3DView<'_>,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize] {
        view.search_into_stack(self, &mut workspace.results, &mut workspace.stack);
        &workspace.results
    }

    #[inline]
    fn any_view(self, view: &Index3DView<'_>) -> bool {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        view.visit_with_stack(self, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    #[inline]
    fn first_view(self, view: &Index3DView<'_>) -> Option<usize> {
        match self.visit_view(view, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    #[inline]
    fn visit_view<B, F>(self, view: &Index3DView<'_>, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        view.visit_with_stack(self, &mut stack, visitor)
    }
}

impl<Q: Overlaps3D> SearchQuery3D for &Q {
    type Iter<'a>
        = RegionSearch3DIter<'a, Self>
    where
        Self: 'a;

    #[inline]
    fn search_into_index(self, index: &Index3D, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = index.visit_region_with_stack(self, &mut stack, |i| {
            out.push(i);
            ControlFlow::<()>::Continue(())
        });
    }

    #[inline]
    fn search_with_index<'a>(
        self,
        index: &Index3D,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize] {
        workspace.results.clear();
        let _ = index.visit_region_with_stack(self, &mut workspace.stack, |i| {
            workspace.results.push(i);
            ControlFlow::<()>::Continue(())
        });
        &workspace.results
    }

    #[inline]
    fn any_index(self, index: &Index3D) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        index
            .visit_region_with_stack(self, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    #[inline]
    fn first_index(self, index: &Index3D) -> Option<usize> {
        match self.visit_index(index, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    #[inline]
    fn visit_index<B, F>(self, index: &Index3D, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        index.visit_region_with_stack(self, &mut stack, visitor)
    }

    #[inline]
    fn search_iter_index<'a>(self, index: &'a Index3D) -> Self::Iter<'a>
    where
        Self: 'a,
    {
        RegionSearch3DIter::new(index, self)
    }

    #[inline]
    fn search_into_view(self, view: &Index3DView<'_>, out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        let _ = view.visit_region_with_stack(self, &mut stack, |i| {
            out.push(i);
            ControlFlow::<()>::Continue(())
        });
    }

    #[inline]
    fn search_with_view<'a>(
        self,
        view: &Index3DView<'_>,
        workspace: &'a mut SearchWorkspace,
    ) -> &'a [usize] {
        workspace.results.clear();
        let _ = view.visit_region_with_stack(self, &mut workspace.stack, |i| {
            workspace.results.push(i);
            ControlFlow::<()>::Continue(())
        });
        &workspace.results
    }

    #[inline]
    fn any_view(self, view: &Index3DView<'_>) -> bool {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        view.visit_region_with_stack(self, &mut stack, |_| ControlFlow::Break(()))
            .is_break()
    }

    #[inline]
    fn first_view(self, view: &Index3DView<'_>) -> Option<usize> {
        match self.visit_view(view, ControlFlow::Break) {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(()) => None,
        }
    }

    #[inline]
    fn visit_view<B, F>(self, view: &Index3DView<'_>, visitor: F) -> ControlFlow<B>
    where
        F: FnMut(usize) -> ControlFlow<B>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(DEFAULT_SEARCH_STACK_CAPACITY);
        view.visit_region_with_stack(self, &mut stack, visitor)
    }
}

impl Index3D {
    fn visit_region_with_stack<B, Q, F>(
        &self,
        query: &Q,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        Q: Overlaps3D,
        F: FnMut(usize) -> ControlFlow<B>,
    {
        visit_region(
            self,
            stack,
            |b| query.overlaps_box(b),
            |b| query.contains_box(b),
            visitor,
        )
    }
}

impl Index3DView<'_> {
    fn visit_region_with_stack<B, Q, F>(
        &self,
        query: &Q,
        stack: &mut Vec<usize>,
        visitor: F,
    ) -> ControlFlow<B>
    where
        Q: Overlaps3D,
        F: FnMut(usize) -> ControlFlow<B>,
    {
        visit_region(
            self,
            stack,
            |b| query.overlaps_box(b),
            |b| query.contains_box(b),
            visitor,
        )
    }
}
