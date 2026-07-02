use packed_spatial_index::{Box3D, Index3DBuilder, NeighborWorkspace, Point3D};
use std::ops::ControlFlow;

fn build_index() -> packed_spatial_index::Index3D {
    let mut builder = Index3DBuilder::new(2).node_size(2);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0));
    builder.add(Box3D::new(5.0, 0.0, 0.0, 6.0, 1.0, 1.0));
    builder.finish().unwrap()
}

#[test]
fn neighbors_3d_accept_negative_zero_cutoff() {
    let index = build_index();
    let point = Point3D::new(1.0, 1.0, 1.0);
    assert_eq!(index.neighbors_within(point, 4, -0.0), vec![0]);
}

#[test]
fn nan_query_point_returns_empty_3d_neighbors() {
    let index = build_index();
    let point = Point3D::new(f64::NAN, 1.0, 1.0);

    assert!(index.neighbors(point, 4).is_empty());
    assert!(index.neighbors_within(point, 4, 10.0).is_empty());

    let mut out = vec![usize::MAX];
    index.neighbors_into(point, 4, 10.0, &mut out);
    assert!(out.is_empty());

    let mut workspace = NeighborWorkspace::with_capacity(8, 8);
    assert!(
        index
            .neighbors_with(point, 4, 10.0, &mut workspace)
            .is_empty()
    );

    let mut visited = false;
    let flow: ControlFlow<()> = index.visit_neighbors(point, 10.0, |_, _| {
        visited = true;
        ControlFlow::Continue(())
    });
    assert!(flow.is_continue());
    assert!(!visited);
}
