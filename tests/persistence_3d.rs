use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index2DView, Index3D, Index3DBuilder, Index3DView, LoadError,
    Point3D,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn persistence_3d_round_trip_and_view_agree() {
    let mut rng = StdRng::seed_from_u64(0x3D5150);
    let boxes = random_boxes_3d(&mut rng, 500);
    let index = build_index_3d(&boxes, 8);

    let bytes = index.to_bytes();
    // superblock: magic, version 2, one chunk (TREE).
    assert_eq!(&bytes[..8], b"PSINDEX\0");
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 2);
    assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 1);

    let loaded = Index3D::from_bytes(&bytes).unwrap();
    let view = Index3DView::from_bytes(&bytes).unwrap();

    assert_eq!(loaded.num_items(), index.num_items());
    assert_eq!(view.num_items(), index.num_items());
    assert_eq!(loaded.node_size(), index.node_size());
    assert_eq!(view.node_size(), index.node_size());
    assert_eq!(loaded.extent(), index.extent());
    assert_eq!(view.extent(), index.extent());

    for _ in 0..100 {
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let qz: f64 = rng.random_range(0.0..1000.0);
        let query = Box3D::new(qx, qy, qz, qx + 40.0, qy + 40.0, qz + 40.0);

        let mut expected = index.search(query);
        let mut owned = loaded.search(query);
        let mut borrowed = view.search(query);
        expected.sort_unstable();
        owned.sort_unstable();
        borrowed.sort_unstable();
        assert_eq!(expected, owned);
        assert_eq!(expected, borrowed);

        let point = Point3D::new(qx, qy, qz);
        assert_eq!(
            index.neighbors_within(point, 12, 100.0),
            loaded.neighbors_within(point, 12, 100.0)
        );
        assert_eq!(
            index.neighbors_within(point, 12, 100.0),
            view.neighbors_within(point, 12, 100.0)
        );
    }
}

#[test]
fn to_bytes_into_3d_matches_owned_serialization_and_reuses_capacity() {
    let mut rng = StdRng::seed_from_u64(0x3DB17E);
    let boxes = random_boxes_3d(&mut rng, 128);
    let index = build_index_3d(&boxes, 8);
    let expected = index.to_bytes();

    let mut bytes = Vec::with_capacity(expected.len() + 128);
    bytes.extend_from_slice(b"stale bytes that must be cleared");
    let capacity = bytes.capacity();

    index.to_bytes_into(&mut bytes);
    assert_eq!(bytes, expected);
    assert_eq!(bytes.capacity(), capacity);

    index.to_bytes_into(&mut bytes);
    assert_eq!(bytes, expected);
    assert_eq!(bytes.capacity(), capacity);
}

#[test]
fn persistence_3d_handles_edge_shapes() {
    let cases: Vec<Vec<Box3D>> = vec![
        Vec::new(),
        vec![Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)],
        vec![
            Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            Box3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0),
            Box3D::new(4.0, 4.0, 4.0, 5.0, 5.0, 5.0),
        ],
        vec![
            Box3D::new(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            Box3D::new(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            Box3D::new(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        ],
    ];

    for boxes in cases {
        let index = build_index_3d(&boxes, 16);
        let bytes = index.to_bytes();
        let loaded = Index3D::from_bytes(&bytes).unwrap();
        let view = Index3DView::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.extent(), index.extent());
        assert_eq!(view.extent(), index.extent());
        let query = Box3D::new(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0);
        assert_eq!(index.search(query), loaded.search(query));
        assert_eq!(index.search(query), view.search(query));
        assert_eq!(
            index.neighbors(Point3D::new(0.0, 0.0, 0.0), 3),
            loaded.neighbors(Point3D::new(0.0, 0.0, 0.0), 3)
        );
        assert_eq!(
            index.neighbors(Point3D::new(0.0, 0.0, 0.0), 3),
            view.neighbors(Point3D::new(0.0, 0.0, 0.0), 3)
        );
    }
}

#[test]
fn persistence_rejects_cross_dimension_buffers() {
    let mut builder = Index2DBuilder::new(1);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    let bytes_2d = builder.finish().unwrap().to_bytes();
    assert!(matches!(
        Index3DView::from_bytes(&bytes_2d),
        Err(LoadError::UnsupportedVersion)
    ));

    let mut builder = Index3DBuilder::new(1);
    builder.add(Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    let bytes_3d = builder.finish().unwrap().to_bytes();
    assert!(matches!(
        Index2DView::from_bytes(&bytes_3d),
        Err(LoadError::UnsupportedVersion)
    ));
}

#[test]
fn persistence_3d_rejects_malformed_buffers() {
    let boxes: Vec<Box3D> = (0..40)
        .map(|i| {
            let x = i as f64;
            Box3D::new(x, x, x, x + 0.5, x + 0.5, x + 0.5)
        })
        .collect();
    let bytes = build_index_3d(&boxes, 4).to_bytes();

    // TREE chunk at 56; descriptor 24B (num_items @ +8 u64, node_size @ +16
    // u16), then 48B box records and 8B index entries.
    let tree = 56usize;
    let num_items = u64::from_le_bytes(bytes[tree + 8..tree + 16].try_into().unwrap()) as usize;
    let node_size = u16::from_le_bytes(bytes[tree + 16..tree + 18].try_into().unwrap()) as usize;
    let mut num_nodes = num_items;
    let mut n = num_items;
    if num_items > 0 {
        loop {
            n = n.div_ceil(node_size);
            num_nodes += n;
            if n == 1 {
                break;
            }
        }
    }
    let indices_offset = tree + 24 + num_nodes * 48;

    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        Index3DView::from_bytes(&bad_magic),
        Err(LoadError::BadMagic)
    ));

    let mut bad_version = bytes.clone();
    bad_version[8..16].copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        Index3DView::from_bytes(&bad_version),
        Err(LoadError::UnsupportedVersion)
    ));

    assert!(matches!(
        Index3DView::from_bytes(&bytes[..bytes.len() - 1]),
        Err(LoadError::InvalidTree | LoadError::Truncated)
    ));

    let mut extra = bytes.clone();
    extra.push(0);
    assert!(matches!(
        Index3DView::from_bytes(&extra),
        Err(LoadError::LengthMismatch { .. })
    ));

    let mut bad_tag = bytes.clone();
    bad_tag[32..36].copy_from_slice(b"JUNK");
    assert!(matches!(
        Index3DView::from_bytes(&bad_tag),
        Err(LoadError::UnsupportedVersion)
    ));

    let mut invalid_node_size = bytes.clone();
    invalid_node_size[tree + 16..tree + 18].copy_from_slice(&1u16.to_le_bytes());
    assert!(matches!(
        Index3DView::from_bytes(&invalid_node_size),
        Err(LoadError::InvalidNodeSize { node_size: 1 })
    ));

    let mut bad_num_items = bytes.clone();
    bad_num_items[tree + 8..tree + 16].copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        Index3DView::from_bytes(&bad_num_items),
        Err(LoadError::InvalidTree)
    ));

    let mut invalid_leaf_index = bytes.clone();
    invalid_leaf_index[indices_offset..indices_offset + 8].copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        Index3DView::from_bytes(&invalid_leaf_index),
        Err(LoadError::InvalidTree)
    ));

    let mut invalid_child_pointer = bytes.clone();
    let last_index_offset = indices_offset + (num_nodes - 1) * 8;
    invalid_child_pointer[last_index_offset..last_index_offset + 8]
        .copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        Index3DView::from_bytes(&invalid_child_pointer),
        Err(LoadError::InvalidTree)
    ));
}

fn build_index_3d(boxes: &[Box3D], node_size: usize) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len()).node_size(node_size);
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish().unwrap()
}

fn random_boxes_3d(rng: &mut StdRng, n: usize) -> Vec<Box3D> {
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..1000.0);
            let y: f64 = rng.random_range(0.0..1000.0);
            let z: f64 = rng.random_range(0.0..1000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}
