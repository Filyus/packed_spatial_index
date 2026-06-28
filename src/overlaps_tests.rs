#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use crate::geometry::{Overlaps2D, Overlaps3D};
    use crate::polygon::ConvexPolygon2D;
    use crate::triangle::Triangle2D;
    use crate::{Box2D, Box3D, Frustum3D, Index2DBuilder, Index3DBuilder, SearchWorkspace};

    fn unit_cube_frustum() -> Frustum3D {
        Frustum3D::from_planes([
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 2.0],
        ])
    }

    #[test]
    fn search_with_box() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
        b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let query = Box2D::new(0.0, 0.0, 2.0, 2.0);
        assert_eq!(index.search(query), vec![0]);
    }

    #[test]
    fn search_with_triangle() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(0.2, 0.2, 0.3, 0.3)); // inside triangle
        b.add(Box2D::new(9.0, 9.0, 9.5, 9.5)); // far away
        let index = b.finish().unwrap();

        let tri = Triangle2D::new([0.0, 0.0], [10.0, 0.0], [0.0, 10.0]);
        assert_eq!(index.search(&tri), vec![0]);
    }

    #[test]
    fn search_with_convex_polygon() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(1.0, 1.0, 2.0, 2.0)); // inside trapezoid
        b.add(Box2D::new(0.0, 5.0, 0.5, 5.5)); // in bbox but outside trapezoid
        let index = b.finish().unwrap();

        // A trapezoid (2D frustum)
        let trapezoid =
            ConvexPolygon2D::new(vec![[0.0, 0.0], [10.0, -4.0], [10.0, 8.0], [0.0, 3.0]]);
        assert_eq!(index.search(&trapezoid), vec![0]);
    }

    #[test]
    fn search_into_reuses_buffer() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
        b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let mut results = Vec::new();
        let tri = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);

        index.search_into(&tri, &mut results);
        assert_eq!(results, vec![0]);

        // Buffer is cleared and reused
        let tri2 = Triangle2D::new([5.0, 5.0], [7.0, 5.0], [5.0, 7.0]);
        index.search_into(&tri2, &mut results);
        assert_eq!(results, vec![1]);
    }

    #[test]
    fn overlaps2d_trait_for_box() {
        let box1 = Box2D::new(0.0, 0.0, 1.0, 1.0);
        let box2 = Box2D::new(0.5, 0.5, 1.5, 1.5);

        assert!(box1.overlaps_box(box2));
        assert!(!box1.contains_box(box2));

        let box3 = Box2D::new(0.2, 0.2, 0.8, 0.8);
        assert!(box1.contains_box(box3));
    }

    #[test]
    fn overlaps2d_trait_for_triangle() {
        let tri = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);
        let bx = Box2D::new(0.5, 0.5, 1.5, 1.5);

        assert!(tri.overlaps_box(bx));
    }

    #[test]
    fn overlaps2d_trait_for_convex_polygon() {
        let poly = ConvexPolygon2D::new(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
        let bx = Box2D::new(0.5, 0.5, 1.5, 1.5);
        assert!(poly.overlaps_box(bx));

        let inner_box = Box2D::new(0.5, 0.5, 1.5, 1.5);
        assert!(poly.contains_box(inner_box));
    }

    #[test]
    fn any_and_visit_short_circuit() {
        let mut b = Index2DBuilder::new(3);
        b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
        b.add(Box2D::new(0.5, 0.5, 1.5, 1.5));
        b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let query = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);
        assert!(index.any(&query));
        assert_eq!(index.first(&query), Some(0));

        let mut visited = Vec::new();
        let stopped = index.visit(&query, |i| {
            visited.push(i);
            ControlFlow::Break(i)
        });
        assert_eq!(stopped, ControlFlow::Break(0));
        assert_eq!(visited, vec![0]);
    }

    #[test]
    fn search_with_reuses_workspace() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
        b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let mut workspace = SearchWorkspace::new();
        let query = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);
        let hits = index.search_with(&query, &mut workspace);
        assert_eq!(hits, &[0]);
        assert_eq!(workspace.results(), &[0]);
    }

    #[test]
    fn search_iter_accepts_region_query() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
        b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let query = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);
        let hits: Vec<_> = index.search_iter(&query).collect();
        assert_eq!(hits, vec![0]);
    }

    #[test]
    fn view_search_works() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
        b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
        let bytes = b.finish().unwrap().to_bytes();

        let view = crate::Index2DView::from_bytes(&bytes).unwrap();
        let tri = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);

        assert_eq!(view.search(&tri), vec![0]);
    }

    #[test]
    fn view_search_into_works() {
        let mut b = Index2DBuilder::new(2);
        b.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
        b.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
        let bytes = b.finish().unwrap().to_bytes();

        let view = crate::Index2DView::from_bytes(&bytes).unwrap();
        let tri = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);

        let mut results = Vec::new();
        view.search_into(&tri, &mut results);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn search_3d_with_box() {
        let mut b = Index3DBuilder::new(2);
        b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        b.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let query = Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
        assert_eq!(index.search(query), vec![0]);
    }

    #[test]
    fn search_3d_with_frustum() {
        let mut b = Index3DBuilder::new(2);
        b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        b.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let frustum = unit_cube_frustum();
        assert_eq!(index.search(&frustum), vec![0]);
    }

    #[test]
    fn search_into_3d_reuses_buffer() {
        let mut b = Index3DBuilder::new(2);
        b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        b.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let mut results = Vec::new();
        index.search_into(&unit_cube_frustum(), &mut results);
        assert_eq!(results, vec![0]);

        let query = Box3D::new(4.0, 4.0, 4.0, 7.0, 7.0, 7.0);
        index.search_into(query, &mut results);
        assert_eq!(results, vec![1]);
    }

    #[test]
    fn overlaps3d_trait_for_box_and_frustum() {
        let box1 = Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
        let box2 = Box3D::new(1.0, 1.0, 1.0, 3.0, 3.0, 3.0);
        let box3 = Box3D::new(0.5, 0.5, 0.5, 1.5, 1.5, 1.5);

        assert!(box1.overlaps_box(box2));
        assert!(!box1.contains_box(box2));
        assert!(box1.contains_box(box3));

        let frustum = unit_cube_frustum();
        assert!(frustum.overlaps_box(box3));
        assert!(frustum.contains_box(box3));
    }

    #[test]
    fn any_and_visit_3d_short_circuit() {
        let mut b = Index3DBuilder::new(3);
        b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        b.add(Box3D::new(0.5, 0.5, 0.5, 1.5, 1.5, 1.5));
        b.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let query = unit_cube_frustum();
        assert!(index.any(&query));
        assert_eq!(index.first(&query), Some(0));

        let mut visited = Vec::new();
        let stopped = index.visit(&query, |i| {
            visited.push(i);
            ControlFlow::Break(i)
        });
        assert_eq!(stopped, ControlFlow::Break(0));
        assert_eq!(visited, vec![0]);
    }

    #[test]
    fn search_iter_3d_accepts_region_query() {
        let mut b = Index3DBuilder::new(2);
        b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        b.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
        let index = b.finish().unwrap();

        let query = unit_cube_frustum();
        let hits: Vec<_> = index.search_iter(&query).collect();
        assert_eq!(hits, vec![0]);
    }

    #[test]
    fn view_search_3d_works() {
        let mut b = Index3DBuilder::new(2);
        b.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        b.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
        let bytes = b.finish().unwrap().to_bytes();

        let view = crate::Index3DView::from_bytes(&bytes).unwrap();
        let query = unit_cube_frustum();
        assert_eq!(view.search(&query), vec![0]);

        let mut results = Vec::new();
        view.search_into(&query, &mut results);
        assert_eq!(results, vec![0]);
        assert!(view.any(&query));
        assert_eq!(view.first(&query), Some(0));

        let mut workspace = SearchWorkspace::new();
        assert_eq!(view.search_with(&query, &mut workspace), &[0]);
    }
}
