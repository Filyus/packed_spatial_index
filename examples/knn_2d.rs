use packed_spatial_index::{Box2D, Index2DBuilder, Point2D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Index2DBuilder::new(4);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    builder.add(Box2D::new(2.0, 2.0, 3.0, 3.0));
    builder.add(Box2D::new(-4.0, -4.0, -3.0, -3.0));

    let index = builder.finish()?;
    let nearest = index.neighbors(Point2D::new(1.5, 1.5), 2);

    println!("{nearest:?}");
    Ok(())
}
