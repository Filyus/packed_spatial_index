use packed_spatial_index::{IndexBuilder, Point, Rect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = IndexBuilder::new(4);
    builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));
    builder.add(Rect::new(2.0, 2.0, 3.0, 3.0));
    builder.add(Rect::new(-4.0, -4.0, -3.0, -3.0));

    let index = builder.finish()?;
    let nearest = index.neighbors(Point::new(1.5, 1.5), 2);

    println!("{nearest:?}");
    Ok(())
}
