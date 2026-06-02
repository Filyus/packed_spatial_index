use packed_spatial_index::{Box2D, Index2DBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boxes = [
        Box2D::new(0.0, 0.0, 1.0, 1.0),
        Box2D::new(5.0, 5.0, 6.0, 6.0),
        Box2D::new(0.5, 0.5, 2.0, 2.0),
    ];

    let mut builder = Index2DBuilder::new(boxes.len());
    for bounds in boxes {
        builder.add(bounds);
    }

    let index = builder.finish()?;
    let hits = index.search(Box2D::new(0.0, 0.0, 1.5, 1.5));

    println!("{hits:?}");
    Ok(())
}
