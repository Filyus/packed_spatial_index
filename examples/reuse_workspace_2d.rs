use packed_spatial_index::{Box2D, Index2DBuilder, NeighborWorkspace, Point2D, SearchWorkspace};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Index2DBuilder::new(3);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    builder.add(Box2D::new(0.5, 0.5, 2.0, 2.0));
    let index = builder.finish()?;

    let mut search_workspace = SearchWorkspace::with_capacity(8, 8);
    let hits = index.search_with(Box2D::new(0.0, 0.0, 2.0, 2.0), &mut search_workspace);
    println!("hits: {hits:?}");

    let mut neighbor_workspace = NeighborWorkspace::with_capacity(4, 8);
    let nearest = index.neighbors_with(
        Point2D::new(1.5, 1.5),
        2,
        f64::INFINITY,
        &mut neighbor_workspace,
    );
    println!("nearest: {nearest:?}");

    Ok(())
}
