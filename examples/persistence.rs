use packed_spatial_index::{Index, IndexBuilder, IndexView, Rect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = IndexBuilder::new(2);
    builder.add(Rect::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Rect::new(5.0, 5.0, 6.0, 6.0));
    let index = builder.finish()?;

    let bytes = index.to_bytes();
    let owned = Index::from_bytes(&bytes)?;
    let view = IndexView::from_bytes(&bytes)?;

    let query = Rect::new(0.0, 0.0, 2.0, 2.0);
    assert_eq!(index.search(query), owned.search(query));
    assert_eq!(index.search(query), view.search(query));

    println!("serialized {} bytes", bytes.len());
    Ok(())
}
