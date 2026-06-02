use packed_spatial_index::{Bounds3D, Index3DBuilder, NeighborWorkspace, Point3D, SearchWorkspace};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Index3DBuilder::new(3);
    builder.add(Bounds3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    builder.add(Bounds3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    builder.add(Bounds3D::new(0.5, 0.5, 0.5, 2.0, 2.0, 2.0));
    let index = builder.finish()?;

    let mut search_workspace = SearchWorkspace::with_capacity(8, 8);
    let hits = index.search_with(
        Bounds3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
        &mut search_workspace,
    );
    println!("hits: {hits:?}");

    let mut neighbor_workspace = NeighborWorkspace::with_capacity(4, 8);
    let nearest = index.neighbors_with(
        Point3D::new(1.5, 1.5, 1.5),
        2,
        f64::INFINITY,
        &mut neighbor_workspace,
    );
    println!("nearest: {nearest:?}");

    Ok(())
}
