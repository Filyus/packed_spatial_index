use packed_spatial_index::{Bounds3D, Index3DBuilder, Point3D};

fn main() {
    let boxes = [
        Bounds3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Bounds3D::new(5.0, 5.0, 5.0, 6.0, 6.0, 6.0),
        Bounds3D::new(0.5, 0.5, 0.5, 2.0, 2.0, 2.0),
    ];

    let mut builder = Index3DBuilder::new(boxes.len());
    for bounds in boxes {
        builder.add(bounds);
    }
    let index = builder.finish().unwrap();

    let mut hits = index.search(Bounds3D::new(0.0, 0.0, 0.0, 1.5, 1.5, 1.5));
    hits.sort_unstable();
    assert_eq!(hits, vec![0, 2]);

    assert_eq!(index.neighbors(Point3D::new(5.25, 5.25, 5.25), 1), vec![1]);
    println!("hits={hits:?}");
}
