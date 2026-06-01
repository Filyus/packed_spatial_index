use packed_spatial_index::{IndexBuilder, Rect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boxes = [
        Rect::new(0.0, 0.0, 1.0, 1.0),
        Rect::new(5.0, 5.0, 6.0, 6.0),
        Rect::new(0.5, 0.5, 2.0, 2.0),
    ];

    let mut builder = IndexBuilder::new(boxes.len());
    for rect in boxes {
        builder.add(rect);
    }

    let index = builder.finish()?;
    let hits = index.search(Rect::new(0.0, 0.0, 1.5, 1.5));

    println!("{hits:?}");
    Ok(())
}
