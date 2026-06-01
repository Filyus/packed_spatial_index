use packed_spatial_index::{
    Bounds2D, Bounds3D, Index2D, Index2DBuilder, Index2DView, Index3D, Index3DBuilder, Index3DView,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Index2DBuilder::new(2);
    builder.add(Bounds2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Bounds2D::new(5.0, 5.0, 6.0, 6.0));
    let index = builder.finish()?;

    let bytes = index.to_bytes();
    let owned = Index2D::from_bytes(&bytes)?;
    let view = Index2DView::from_bytes(&bytes)?;

    let query = Bounds2D::new(0.0, 0.0, 2.0, 2.0);
    assert_eq!(index.search(query), owned.search(query));
    assert_eq!(index.search(query), view.search(query));

    println!("serialized 2D index to {} bytes", bytes.len());

    let mut builder = Index3DBuilder::new(2);
    builder.add(Bounds3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    builder.add(Bounds3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    let index = builder.finish()?;

    let bytes = index.to_bytes();
    let owned = Index3D::from_bytes(&bytes)?;
    let view = Index3DView::from_bytes(&bytes)?;

    let query = Bounds3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    assert_eq!(index.search(query), owned.search(query));
    assert_eq!(index.search(query), view.search(query));

    println!("serialized 3D index to {} bytes", bytes.len());
    Ok(())
}
