use packed_spatial_index::{IndexBuilder, NeighborWorkspace, Point, Rect, SearchWorkspace};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = IndexBuilder::new(3);
    builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));
    builder.add(Rect::new(0.5, 0.5, 2.0, 2.0));
    let index = builder.finish()?;

    let mut search_workspace = SearchWorkspace::with_capacity(8, 8);
    let hits = index.search_with(Rect::new(0.0, 0.0, 2.0, 2.0), &mut search_workspace);
    println!("hits: {hits:?}");

    let mut neighbor_workspace = NeighborWorkspace::with_capacity(4, 8);
    let nearest = index.neighbors_with(
        Point::new(1.5, 1.5),
        2,
        f64::INFINITY,
        &mut neighbor_workspace,
    );
    println!("nearest: {nearest:?}");

    Ok(())
}
