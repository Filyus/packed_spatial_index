mod common;

use common::{build_index, random_boxes};
use packed_spatial_index::{Index, IndexView, LoadError, Point, Rect};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn persistence_round_trip_and_view_agree() {
    let mut rng = StdRng::seed_from_u64(0x5150);
    let boxes = random_boxes(&mut rng, 500);
    let index = build_index(&boxes, 8);

    let bytes = index.to_bytes();
    assert_eq!(&bytes[..8], b"PSINDEX\0");
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 64);
    assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 0);

    let loaded = Index::from_bytes(&bytes).unwrap();
    let view = IndexView::from_bytes(&bytes).unwrap();

    assert_eq!(loaded.num_items(), index.num_items());
    assert_eq!(view.num_items(), index.num_items());
    assert_eq!(loaded.node_size(), index.node_size());
    assert_eq!(view.node_size(), index.node_size());
    assert_eq!(loaded.bounds(), index.bounds());
    assert_eq!(view.bounds(), index.bounds());

    for _ in 0..100 {
        let qx: f64 = rng.random_range(0.0..1000.0);
        let qy: f64 = rng.random_range(0.0..1000.0);
        let query = Rect::new(qx, qy, qx + 40.0, qy + 40.0);

        let mut expected = index.search(query);
        let mut owned = loaded.search(query);
        let mut borrowed = view.search(query);
        expected.sort_unstable();
        owned.sort_unstable();
        borrowed.sort_unstable();
        assert_eq!(expected, owned);
        assert_eq!(expected, borrowed);

        let point = Point::new(qx, qy);
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
fn persistence_handles_edge_shapes() {
    let cases: Vec<Vec<[f64; 4]>> = vec![
        Vec::new(),
        vec![[0.0, 0.0, 1.0, 1.0]],
        vec![
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
        ],
        vec![
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
    ];

    for boxes in cases {
        let index = build_index(&boxes, 16);
        let bytes = index.to_bytes();
        let loaded = Index::from_bytes(&bytes).unwrap();
        let view = IndexView::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.bounds(), index.bounds());
        assert_eq!(view.bounds(), index.bounds());
        let query = Rect::new(-100.0, -100.0, 100.0, 100.0);
        assert_eq!(index.search(query), loaded.search(query));
        assert_eq!(index.search(query), view.search(query));
        assert_eq!(
            index.neighbors(Point::new(0.0, 0.0), 3),
            loaded.neighbors(Point::new(0.0, 0.0), 3)
        );
        assert_eq!(
            index.neighbors(Point::new(0.0, 0.0), 3),
            view.neighbors(Point::new(0.0, 0.0), 3)
        );
    }
}

#[test]
fn persistence_rejects_malformed_buffers() {
    let boxes: Vec<[f64; 4]> = (0..40)
        .map(|i| {
            let x = i as f64;
            [x, x, x + 0.5, x + 0.5]
        })
        .collect();
    let bytes = build_index(&boxes, 4).to_bytes();

    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        IndexView::from_bytes(&bad_magic),
        Err(LoadError::BadMagic)
    ));

    let mut bad_version = bytes.clone();
    bad_version[8..16].copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        IndexView::from_bytes(&bad_version),
        Err(LoadError::UnsupportedVersion)
    ));

    let mut bad_header_len = bytes.clone();
    bad_header_len[16..24].copy_from_slice(&48u64.to_le_bytes());
    assert!(matches!(
        IndexView::from_bytes(&bad_header_len),
        Err(LoadError::UnsupportedVersion)
    ));

    let mut bad_flags = bytes.clone();
    bad_flags[24..32].copy_from_slice(&1u64.to_le_bytes());
    assert!(matches!(
        IndexView::from_bytes(&bad_flags),
        Err(LoadError::UnsupportedVersion)
    ));

    assert!(matches!(
        IndexView::from_bytes(&bytes[..bytes.len() - 1]),
        Err(LoadError::Truncated)
    ));

    let mut extra = bytes.clone();
    extra.push(0);
    assert!(matches!(
        IndexView::from_bytes(&extra),
        Err(LoadError::LengthMismatch { .. })
    ));

    let mut invalid_node_size = bytes.clone();
    invalid_node_size[32..40].copy_from_slice(&1u64.to_le_bytes());
    assert!(matches!(
        IndexView::from_bytes(&invalid_node_size),
        Err(LoadError::InvalidNodeSize { node_size: 1 })
    ));

    let mut invalid_level_bounds = bytes.clone();
    invalid_level_bounds[64..72].copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        IndexView::from_bytes(&invalid_level_bounds),
        Err(LoadError::InvalidTree)
    ));

    let num_nodes = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
    let level_count = u64::from_le_bytes(bytes[56..64].try_into().unwrap()) as usize;
    let indices_offset = 64 + level_count * 8 + num_nodes * 32;

    let mut invalid_leaf_index = bytes.clone();
    invalid_leaf_index[indices_offset..indices_offset + 8].copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        IndexView::from_bytes(&invalid_leaf_index),
        Err(LoadError::InvalidTree)
    ));

    let mut invalid_child_pointer = bytes.clone();
    let last_index_offset = indices_offset + (num_nodes - 1) * 8;
    invalid_child_pointer[last_index_offset..last_index_offset + 8]
        .copy_from_slice(&999u64.to_le_bytes());
    assert!(matches!(
        IndexView::from_bytes(&invalid_child_pointer),
        Err(LoadError::InvalidTree)
    ));
}
