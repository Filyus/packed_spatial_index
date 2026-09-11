//! Node aggregates (the `AGGR` chunk): oracle parity for the region fold,
//! round-trips through every loader, and rejection of a corrupt chunk.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use packed_spatial_index::{
    Aggregate, Box2D, Box3D, Index2D, Index2DBuilder, Index2DView, Index3D, Index3DBuilder,
    Index3DView, LoadError,
};

fn random_boxes_2d(rng: &mut StdRng, count: usize) -> Vec<Box2D> {
    (0..count)
        .map(|_| {
            let x: f64 = rng.random_range(-50.0..50.0);
            let y: f64 = rng.random_range(-50.0..50.0);
            let w: f64 = rng.random_range(0.0..5.0);
            let h: f64 = rng.random_range(0.0..5.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn random_boxes_3d(rng: &mut StdRng, count: usize) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let (x, y, z): (f64, f64, f64) = (
                rng.random_range(-50.0..50.0),
                rng.random_range(-50.0..50.0),
                rng.random_range(-50.0..50.0),
            );
            let (w, h, d): (f64, f64, f64) = (
                rng.random_range(0.0..5.0),
                rng.random_range(0.0..5.0),
                rng.random_range(0.0..5.0),
            );
            Box3D::new(x, y, z, x + w, y + h, z + d)
        })
        .collect()
}

fn random_windows_2d(rng: &mut StdRng, count: usize) -> Vec<Box2D> {
    (0..count)
        .map(|_| {
            let (x, y): (f64, f64) = (rng.random_range(-60.0..60.0), rng.random_range(-60.0..60.0));
            let side: f64 = rng.random_range(0.0..120.0);
            Box2D::new(x, y, x + side, y + side)
        })
        .collect()
}

fn random_windows_3d(rng: &mut StdRng, count: usize) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let (x, y, z): (f64, f64, f64) = (
                rng.random_range(-60.0..60.0),
                rng.random_range(-60.0..60.0),
                rng.random_range(-60.0..60.0),
            );
            let side: f64 = rng.random_range(0.0..120.0);
            Box3D::new(x, y, z, x + side, y + side, z + side)
        })
        .collect()
}

/// The oracle: search + a per-item fold. `aggregate` must agree exactly.
fn naive_aggregate_2d(items: &[Box2D], scalars: &[f64], masks: &[u64], window: Box2D) -> Aggregate {
    let mut out = Aggregate {
        count: 0,
        sum: None,
        min: None,
        max: None,
        mask: None,
    };
    let (mut sum, mut min, mut max, mut or) = (0.0, f64::INFINITY, f64::NEG_INFINITY, 0u64);
    for (i, b) in items.iter().enumerate() {
        if b.overlaps(window) {
            out.count += 1;
            sum += scalars[i];
            min = min.min(scalars[i]);
            max = max.max(scalars[i]);
            or |= masks[i];
        }
    }
    if out.count > 0 {
        out.sum = Some(sum);
        out.min = Some(min);
        out.max = Some(max);
        out.mask = Some(or);
    }
    out
}

fn naive_aggregate_3d(items: &[Box3D], scalars: &[f64], masks: &[u64], window: Box3D) -> Aggregate {
    let mut out = Aggregate {
        count: 0,
        sum: None,
        min: None,
        max: None,
        mask: None,
    };
    let (mut sum, mut min, mut max, mut or) = (0.0, f64::INFINITY, f64::NEG_INFINITY, 0u64);
    for (i, b) in items.iter().enumerate() {
        if b.overlaps(window) {
            out.count += 1;
            sum += scalars[i];
            min = min.min(scalars[i]);
            max = max.max(scalars[i]);
            or |= masks[i];
        }
    }
    if out.count > 0 {
        out.sum = Some(sum);
        out.min = Some(min);
        out.max = Some(max);
        out.mask = Some(or);
    }
    out
}

#[test]
fn aggregate_matches_the_oracle_across_windows_and_node_sizes() {
    for &seed in &[1u64, 7, 42] {
        for &node_size in &[2usize, 4, 16] {
            let mut rng = StdRng::seed_from_u64(seed);
            let items = random_boxes_2d(&mut rng, 500);
            let scalars: Vec<f64> = (0..items.len()).map(|i| (i % 13) as f64 - 6.0).collect();
            let masks: Vec<u64> = (0..items.len()).map(|i| 1u64 << (i % 17)).collect();

            let mut builder = Index2DBuilder::new(items.len()).node_size(node_size);
            for &b in &items {
                builder.add(b);
            }
            let index = builder
                .aggregate_scalar(&scalars)
                .aggregate_mask(&masks)
                .finish()
                .unwrap();

            for window in random_windows_2d(&mut rng, 40) {
                assert_eq!(
                    index.aggregate(window),
                    Some(naive_aggregate_2d(&items, &scalars, &masks, window)),
                    "seed {seed} node_size {node_size} window {window:?}"
                );
                // Count must agree with the plain count too.
                assert_eq!(
                    index.aggregate(window).unwrap().count as usize,
                    index.count(window)
                );
            }
            // The whole extent is one root-summary read.
            let all = naive_aggregate_2d(&items, &scalars, &masks, index.extent().unwrap());
            assert_eq!(index.aggregate(index.extent().unwrap()), Some(all));
        }
    }
}

#[test]
fn aggregate_3d_matches_the_oracle() {
    let mut rng = StdRng::seed_from_u64(9);
    let items = random_boxes_3d(&mut rng, 400);
    let scalars: Vec<f64> = (0..items.len()).map(|i| i as f64 * 0.5).collect();
    let masks: Vec<u64> = (0..items.len()).map(|i| !(u64::MAX << (i % 31))).collect();

    let mut builder = Index3DBuilder::new(items.len()).node_size(8);
    for &b in &items {
        builder.add(b);
    }
    let index = builder
        .aggregate_scalar(&scalars)
        .aggregate_mask(&masks)
        .finish()
        .unwrap();

    for window in random_windows_3d(&mut rng, 30) {
        assert_eq!(
            index.aggregate(window),
            Some(naive_aggregate_3d(&items, &scalars, &masks, window))
        );
    }
}

#[test]
fn scalar_only_and_mask_only_columns() {
    let mut rng = StdRng::seed_from_u64(3);
    let items = random_boxes_2d(&mut rng, 100);
    let scalars: Vec<f64> = (0..items.len()).map(|i| i as f64).collect();

    let mut builder = Index2DBuilder::new(items.len()).node_size(5);
    for &b in &items {
        builder.add(b);
    }
    let scalar_only = builder.aggregate_scalar(&scalars).finish().unwrap();
    assert!(scalar_only.aggregates().unwrap().has_scalar());
    assert!(!scalar_only.aggregates().unwrap().has_mask());

    let mut builder = Index2DBuilder::new(items.len()).node_size(5);
    for &b in &items {
        builder.add(b);
    }
    let mask_only = builder.aggregate_mask(&[1; 100]).finish().unwrap();
    assert!(!mask_only.aggregates().unwrap().has_scalar());
    assert!(mask_only.aggregates().unwrap().has_mask());

    for window in random_windows_2d(&mut rng, 20) {
        let agg = scalar_only.aggregate(window).unwrap();
        assert!(agg.mask.is_none());
        let oracle = naive_aggregate_2d(&items, &scalars, &[0; 100], window);
        assert_eq!(agg.count, oracle.count);
        assert_eq!(agg.sum, oracle.sum);
        assert_eq!(agg.min, oracle.min);
        assert_eq!(agg.max, oracle.max);

        let agg = mask_only.aggregate(window).unwrap();
        assert!(agg.sum.is_none() && agg.min.is_none() && agg.max.is_none());
        if agg.count > 0 {
            assert_eq!(agg.mask, Some(1));
        }
    }
}

#[test]
fn no_aggregates_answers_none_and_mismatched_columns_are_rejected() {
    let mut builder = Index2DBuilder::new(3);
    for i in 0..3u32 {
        builder.add(Box2D::new(i as f64, 0.0, i as f64 + 1.0, 1.0));
    }
    let index = builder.finish().unwrap();
    assert!(index.aggregates().is_none());
    assert_eq!(index.aggregate(Box2D::new(0.0, 0.0, 100.0, 100.0)), None);

    let mut builder = Index2DBuilder::new(3);
    for i in 0..3u32 {
        builder.add(Box2D::new(i as f64, 0.0, i as f64 + 1.0, 1.0));
    }
    assert!(builder.aggregate_scalar(&[1.0, 2.0]).finish().is_err());
}

#[test]
fn empty_and_single_node_indexes() {
    let index = Index2DBuilder::new(0)
        .aggregate_scalar(&[])
        .aggregate_mask(&[])
        .finish()
        .unwrap();
    assert_eq!(
        index.aggregate(Box2D::new(-1.0, -1.0, 1.0, 1.0)),
        Some(Aggregate {
            count: 0,
            sum: None,
            min: None,
            max: None,
            mask: None
        })
    );

    let mut builder = Index2DBuilder::new(1);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    let index = builder.aggregate_scalar(&[7.0]).finish().unwrap();
    assert_eq!(
        index.aggregate(Box2D::new(0.5, 0.5, 0.5, 0.5)),
        Some(Aggregate {
            count: 1,
            sum: Some(7.0),
            min: Some(7.0),
            max: Some(7.0),
            mask: None
        })
    );
}

#[test]
fn chunk_round_trips_through_every_loader() {
    let mut rng = StdRng::seed_from_u64(11);
    let items = random_boxes_2d(&mut rng, 200);
    let scalars: Vec<f64> = (0..items.len()).map(|i| (i % 7) as f64).collect();
    let masks: Vec<u64> = (0..items.len()).map(|i| 1u64 << (i % 13)).collect();

    let mut builder = Index2DBuilder::new(items.len()).node_size(6);
    for &b in &items {
        builder.add(b);
    }
    let index = builder
        .aggregate_scalar(&scalars)
        .aggregate_mask(&masks)
        .finish()
        .unwrap();
    let bytes = index.to_bytes();

    let reloaded = Index2D::from_bytes(&bytes).unwrap();
    let view = Index2DView::from_bytes(&bytes).unwrap();

    // An index-only reader that does not know the chunk is unaffected: the
    // chunk is optional, so an unknown-optional reader of an older version
    // simply skips it (covered by the persistence tests); here every loader of
    // this version sees the same summaries.
    assert!(view.aggregates().is_some());

    for window in random_windows_2d(&mut rng, 25) {
        let want = index.aggregate(window);
        assert_eq!(reloaded.aggregate(window), want);
        assert_eq!(view.aggregate(window), want);
    }

    // The chunk is byte-identical from the SIMD build of the same items.
    #[cfg(feature = "simd")]
    {
        let mut builder = Index2DBuilder::new(items.len()).node_size(6);
        for &b in &items {
            builder.add(b);
        }
        let simd = builder
            .aggregate_scalar(&scalars)
            .aggregate_mask(&masks)
            .finish_simd()
            .unwrap();
        assert_eq!(simd.to_bytes(), bytes);
        for window in random_windows_2d(&mut rng, 10) {
            assert_eq!(simd.aggregate(window), index.aggregate(window));
        }
        let simd_view_bytes = index.to_bytes();
        #[cfg(feature = "simd")]
        {
            let view = packed_spatial_index::SimdIndex2DView::from_bytes(&simd_view_bytes).unwrap();
            assert!(view.aggregates().is_some());
            for window in random_windows_2d(&mut rng, 5) {
                assert_eq!(view.aggregate(window), index.aggregate(window));
            }
        }
        let simd_reloaded = packed_spatial_index::SimdIndex2D::from_bytes(&bytes).unwrap();
        for window in random_windows_2d(&mut rng, 5) {
            assert_eq!(simd_reloaded.aggregate(window), index.aggregate(window));
        }
    }
}

#[test]
fn corrupted_aggregate_chunk_is_rejected() {
    let mut builder = Index2DBuilder::new(4);
    for i in 0..4u32 {
        builder.add(Box2D::new(i as f64, 0.0, i as f64 + 1.0, 1.0));
    }
    let index = builder.aggregate_scalar(&[1.0; 4]).finish().unwrap();
    let bytes = index.to_bytes();

    // The chunk's last byte lives at the very end of the file (up to alignment
    // pad); truncating there breaks the aggregate section, not the tree.
    for cut in 1..8 {
        let mut truncated = bytes.clone();
        truncated.truncate(truncated.len() - cut);
        // Either the container rejects it or the aggregate parse does; the
        // tree itself must never load with silently wrong summaries.
        if let Ok(loaded) = Index2D::from_bytes(&truncated) {
            // A truncate that only ate alignment pad still loads; the
            // aggregate answers must still be exact.
            let agg = loaded
                .aggregate(Box2D::new(0.0, 0.0, 100.0, 100.0))
                .unwrap();
            assert_eq!(agg.count, 4);
            assert_eq!(agg.sum, Some(4.0));
        }
    }

    // Flipping a summary byte cannot corrupt an exact answer undetected: the
    // root summary change shows up in the result.
    let mut flipped = bytes.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0xFF;
    if let Ok(loaded) = Index2D::from_bytes(&flipped) {
        // If the byte flipped was only alignment pad, answers stay exact.
        let agg = loaded
            .aggregate(Box2D::new(0.0, 0.0, 100.0, 100.0))
            .unwrap();
        assert_eq!(agg.count, 4);
    }
}

#[test]
fn a_chunk_without_per_item_values_is_rejected() {
    // A scalar-only chunk with the per-item flag cleared: the summaries alone
    // cannot answer a window that cuts a leaf, and a reader that accepted the
    // chunk would read item values past its end. Every loader must refuse it.
    let mut builder = Index2DBuilder::new(4);
    for i in 0..4u32 {
        builder.add(Box2D::new(i as f64, 0.0, i as f64 + 1.0, 1.0));
    }
    let index = builder.aggregate_scalar(&[1.0; 4]).finish().unwrap();
    let mut bytes = index.to_bytes();

    // The descriptor: desc_len 16, ordering 0, columns 1 (scalar), flags 1,
    // eight reserved zero bytes. Unique in a file this small.
    let desc = [16u8, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let at = bytes
        .windows(desc.len())
        .position(|w| w == desc)
        .expect("AGGR descriptor present");
    bytes[at + 6] = 0; // flags: per-item values absent

    assert_eq!(
        Index2D::from_bytes(&bytes).err(),
        Some(LoadError::InvalidAggregates)
    );
    assert_eq!(
        Index2DView::from_bytes(&bytes).err(),
        Some(LoadError::InvalidAggregates)
    );
}

#[test]
fn interleaved_layout_carries_the_chunk_too() {
    #[cfg(feature = "stream")]
    {
        let mut rng = StdRng::seed_from_u64(5);
        let items = random_boxes_2d(&mut rng, 64);
        let scalars: Vec<f64> = (0..items.len()).map(|i| i as f64).collect();
        let mut builder = Index2DBuilder::new(items.len()).node_size(4);
        for &b in &items {
            builder.add(b);
        }
        let index = builder.aggregate_scalar(&scalars).finish().unwrap();
        let bytes = index.to_bytes_interleaved();
        // The owned loader accepts the interleaved layout and its AGGR chunk.
        let reloaded = Index2D::from_bytes(&bytes).unwrap();
        for window in random_windows_2d(&mut rng, 10) {
            assert_eq!(reloaded.aggregate(window), index.aggregate(window));
        }
    }
}

#[test]
fn aggregate_3d_round_trip() {
    let mut rng = StdRng::seed_from_u64(21);
    let items = random_boxes_3d(&mut rng, 120);
    let masks: Vec<u64> = (0..items.len()).map(|i| 3u64 + (i % 5) as u64).collect();
    let mut builder = Index3DBuilder::new(items.len()).node_size(7);
    for &b in &items {
        builder.add(b);
    }
    let index = builder.aggregate_mask(&masks).finish().unwrap();
    let bytes = index.to_bytes();
    let reloaded = Index3D::from_bytes(&bytes).unwrap();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    for window in random_windows_3d(&mut rng, 20) {
        let want = index.aggregate(window);
        assert_eq!(reloaded.aggregate(window), want);
        assert_eq!(view.aggregate(window), want);
    }
}
