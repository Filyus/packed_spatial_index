use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Index3DBuilder::new(2);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    builder.add(Box3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    let index = builder.finish()?;

    let bytes = index.to_bytes();
    let owned = Index3D::from_bytes(&bytes)?;
    let view = Index3DView::from_bytes(&bytes)?;

    let query = Box3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    assert_eq!(index.search(query), owned.search(query));
    assert_eq!(index.search(query), view.search(query));

    println!("serialized 3D index to {} bytes", bytes.len());
    Ok(())
}
