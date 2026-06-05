use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boxes = [
        Box2D::new(1.0 + 1e-8, 0.0, 1.0 + 1e-8, 0.0),
        Box2D::new(1.0, 0.0, 1.0, 0.0),
    ];

    let mut builder = Index2DBuilder::new(boxes.len());
    for &b in &boxes {
        builder.add(b);
    }
    let index = builder.finish_simd_f32()?;

    let query = Box2D::new(1.0, 0.0, 1.0, 0.0);
    let mut rounded_hits = index.search(query);
    rounded_hits.sort_unstable();
    println!("rounded range hits: {rounded_hits:?}");

    let exact = index.search_exact(query, |i| boxes[i]);
    println!("exact range hits: {exact:?}");

    let nearest = index.neighbors_exact(Point2D::new(1.0, 0.0), 1, |i| boxes[i]);
    println!("exact nearest hit: {nearest:?}");

    Ok(())
}
