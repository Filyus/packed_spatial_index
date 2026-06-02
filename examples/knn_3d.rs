use packed_spatial_index::{Bounds3D, Index3DBuilder, Point3D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Index3DBuilder::new(4);
    builder.add(Bounds3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    builder.add(Bounds3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0));
    builder.add(Bounds3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0));
    builder.add(Bounds3D::new(-4.0, -4.0, -1.0, -3.0, -3.0, 0.0));

    let index = builder.finish()?;
    let nearest = index.neighbors(Point3D::new(1.5, 1.5, 1.5), 2);
    let nearby = index.neighbors_within(Point3D::new(1.5, 1.5, 1.5), 8, 3.0);

    println!("nearest: {nearest:?}");
    println!("within radius: {nearby:?}");
    Ok(())
}
