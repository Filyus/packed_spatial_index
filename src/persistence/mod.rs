mod container;
mod errors;
mod metadata;
mod payload;
mod payload_chunk;
mod tree_chunk;
mod writer;

use self::container::{find_chunk, parse_container};
pub(crate) use container::{
    CHUNK_ENTRY_LEN, CHUNK_FLAG_CRITICAL, FORMAT_VERSION, SUPERBLOCK_LEN, plan_container,
};
pub use errors::{LoadError, PayloadError};
pub use metadata::{FileMetadata, read_metadata};
pub(crate) use metadata::{MetaFields, TAG_META};
pub(crate) use payload::{ParsedPayload, build_id_to_leaf, parse_payload_body, payload_slice};
pub(crate) use payload_chunk::{PYLD_DESC_LEN, PYLD_DESC_LEN_FIXED, TAG_PYLD, parse_pyld_chunk};
pub(crate) use tree_chunk::{TAG_TREE, TREE_DESC_LEN, parse_tree_chunk};
pub(crate) use writer::{ByteWriter, write_index_container};

pub(crate) fn expected_tree_shape(
    num_items: usize,
    node_size: usize,
) -> Result<(usize, usize), LoadError> {
    let mut num_nodes = num_items;
    let mut levels = 1usize;
    let mut n = num_items;
    if num_items > 0 {
        loop {
            n = n.div_ceil(node_size);
            num_nodes = num_nodes.checked_add(n).ok_or(LoadError::IntegerOverflow)?;
            levels = levels.checked_add(1).ok_or(LoadError::IntegerOverflow)?;
            if n == 1 {
                break;
            }
        }
    }
    Ok((num_nodes, levels))
}

pub(crate) fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, LoadError> {
    let end = offset.checked_add(8).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

#[inline]
pub(crate) fn read_u64_le_unchecked(bytes: &[u8], offset: usize) -> u64 {
    debug_assert!(offset <= bytes.len());
    debug_assert!(bytes.len() - offset >= 8);

    let mut value = 0u64;
    // SAFETY: callers only use this for slices and offsets validated by
    // `parse_index`; unaligned byte buffers are copied into an aligned u64.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().add(offset),
            (&mut value as *mut u64).cast::<u8>(),
            8,
        );
    }
    u64::from_le(value)
}

#[inline]
pub(crate) fn read_f64_le_unchecked(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64_le_unchecked(bytes, offset))
}

#[inline]
#[cfg(any(feature = "f32-storage", feature = "stream"))]
pub(crate) fn read_f32_le_unchecked(bytes: &[u8], offset: usize) -> f32 {
    debug_assert!(offset + 4 <= bytes.len());
    let mut value = 0u32;
    // SAFETY: callers only use this for slices and offsets validated by
    // the f32 byte parsers; unaligned byte buffers are copied into an aligned u32.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr().add(offset),
            (&mut value as *mut u32).cast::<u8>(),
            4,
        );
    }
    f32::from_bits(u32::from_le(value))
}

fn usize_from_u64(value: u64) -> Result<usize, LoadError> {
    usize::try_from(value).map_err(|_| LoadError::IntegerOverflow)
}

pub(crate) fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, LoadError> {
    let end = offset.checked_add(4).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

pub(crate) fn read_u16_at(bytes: &[u8], offset: usize) -> Result<u16, LoadError> {
    let end = offset.checked_add(2).ok_or(LoadError::IntegerOverflow)?;
    let slice = bytes.get(offset..end).ok_or(LoadError::Truncated)?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

/// Derive the level-bound positions (cumulative node counts per level) from the
/// item count and node size. The format does not store them — they are a function of the
/// tree shape, so deriving avoids a second source of truth that could drift.
pub(crate) fn derive_level_bounds(
    num_items: usize,
    node_size: usize,
    level_count: usize,
) -> Vec<usize> {
    let mut bounds = Vec::with_capacity(level_count);
    let mut n = num_items;
    let mut total = n;
    bounds.push(total);
    while bounds.len() < level_count {
        n = n.div_ceil(node_size);
        total += n;
        bounds.push(total);
    }
    bounds
}

/// A parsed SoA tree: the node bytes (borrowed) plus the derived shape. The
/// `level_bounds` are owned because they are derived at load, not stored.
pub(crate) struct ParsedTree<'a> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_count: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) entries: &'a [u8],
    pub(crate) indices: &'a [u8],
}

/// Parse a container's `TREE` chunk for an in-memory (SoA) reader, plus the
/// optional payload. Rejects an interleaved tree (that layout is streaming-only).
pub(crate) fn parse_index(
    bytes: &[u8],
    dimensions: usize,
    coord_bytes: usize,
) -> Result<(ParsedTree<'_>, Option<ParsedPayload<'_>>), LoadError> {
    let chunks = parse_container(bytes, &[TAG_TREE])?;
    let tree_ref = find_chunk(&chunks, TAG_TREE).ok_or(LoadError::InvalidTree)?;
    let (desc, node_data) =
        parse_tree_chunk(&bytes[tree_ref.offset..tree_ref.offset + tree_ref.len])?;
    if desc.dimensions != dimensions || desc.coord_bytes != coord_bytes {
        return Err(LoadError::UnsupportedVersion);
    }
    if desc.interleaved {
        // The interleaved layout is read only by the streaming reader.
        return Err(LoadError::UnsupportedVersion);
    }

    let (num_nodes, level_count) = expected_tree_shape(desc.num_items, desc.node_size)?;
    let record = dimensions
        .checked_mul(2 * coord_bytes)
        .ok_or(LoadError::IntegerOverflow)?;
    let entries_len = num_nodes
        .checked_mul(record)
        .ok_or(LoadError::IntegerOverflow)?;
    let indices_len = num_nodes.checked_mul(8).ok_or(LoadError::IntegerOverflow)?;
    let node_len = entries_len
        .checked_add(indices_len)
        .ok_or(LoadError::IntegerOverflow)?;
    if node_data.len() != node_len {
        return Err(LoadError::InvalidTree);
    }

    let parsed = ParsedTree {
        node_size: desc.node_size,
        num_items: desc.num_items,
        num_nodes,
        level_count,
        level_bounds: derive_level_bounds(desc.num_items, desc.node_size, level_count),
        entries: &node_data[..entries_len],
        indices: &node_data[entries_len..],
    };
    validate_tree_indices(&parsed)?;

    let payload = match find_chunk(&chunks, TAG_PYLD) {
        Some(p) => {
            let (pd, body) = parse_pyld_chunk(&bytes[p.offset..p.offset + p.len])?;
            Some(parse_payload_body(body, desc.num_items, pd.record_stride)?)
        }
        None => None,
    };
    Ok((parsed, payload))
}

/// Validate leaf and internal child pointers against the derived level bounds.
fn validate_tree_indices(p: &ParsedTree<'_>) -> Result<(), LoadError> {
    for pos in 0..p.num_items {
        let index = read_u64_at(p.indices, pos * 8).and_then(usize_from_u64)?;
        if index >= p.num_items {
            return Err(LoadError::InvalidTree);
        }
    }
    for level in 1..p.level_count {
        let level_start = p.level_bounds[level - 1];
        let level_end = p.level_bounds[level];
        let child_level_start = if level == 1 {
            0
        } else {
            p.level_bounds[level - 2]
        };
        let child_level_end = level_start;
        for pos in level_start..level_end {
            let index = read_u64_at(p.indices, pos * 8).and_then(usize_from_u64)?;
            if index < child_level_start || index >= child_level_end {
                return Err(LoadError::InvalidTree);
            }
            if (index - child_level_start) % p.node_size != 0 {
                return Err(LoadError::InvalidTree);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Box2D, Index2D, Index2DBuilder};

    fn build(n: usize) -> Index2D {
        let mut builder = Index2DBuilder::new(n).node_size(16);
        for i in 0..n {
            let v = i as f64;
            builder.add(Box2D::new(v, v, v + 1.0, v + 1.0));
        }
        builder.finish().unwrap()
    }

    // ---- chunk container ----

    /// Frame chunks `(tag, critical, content)` into a container buffer.
    fn encode_container(chunks: &[([u8; 4], bool, Vec<u8>)]) -> Vec<u8> {
        let lens: Vec<usize> = chunks.iter().map(|(_, _, c)| c.len()).collect();
        let (total, offsets) = plan_container(&lens).unwrap();
        let mut buf = Vec::new();
        let mut w = ByteWriter::new(&mut buf, total);
        w.write_superblock(chunks.len() as u32);
        for (i, (tag, critical, c)) in chunks.iter().enumerate() {
            w.write_chunk_entry(tag, *critical, offsets[i], c.len());
        }
        let mut pos = SUPERBLOCK_LEN + chunks.len() * CHUNK_ENTRY_LEN;
        for (i, (_, _, c)) in chunks.iter().enumerate() {
            w.write_zeros(offsets[i] - pos);
            w.write_raw(c);
            pos = offsets[i] + c.len();
        }
        w.write_zeros(total - pos);
        w.finish();
        buf
    }

    fn tree_content(
        interleaved: bool,
        num_items: usize,
        node_size: usize,
        node_data: &[u8],
    ) -> Vec<u8> {
        let mut v = Vec::new();
        let mut w = ByteWriter::new(&mut v, TREE_DESC_LEN + node_data.len());
        w.write_tree_desc(2, 8, interleaved, num_items, node_size);
        w.write_raw(node_data);
        w.finish();
        v
    }

    #[test]
    fn v2_container_round_trips_tree_and_payload() {
        let node_data = vec![0xABu8; 40 * 5]; // 5 interleaved 2D nodes, arbitrary bytes
        let tree = tree_content(true, 4, 16, &node_data);

        let mut pyld = Vec::new();
        {
            let mut w = ByteWriter::new(&mut pyld, PYLD_DESC_LEN + 8 + b"blob".len());
            w.write_pyld_desc(None); // variable-width
            w.write_u64(0); // one-entry offset table fragment, just bytes here
            w.write_raw(b"blob");
            w.finish();
        }

        let buf = encode_container(&[(TAG_TREE, true, tree), (TAG_PYLD, false, pyld)]);

        let chunks = parse_container(&buf, &[TAG_TREE]).unwrap();
        assert_eq!(chunks.len(), 2);

        let tree_ref = find_chunk(&chunks, TAG_TREE).unwrap();
        let (desc, nd) =
            parse_tree_chunk(&buf[tree_ref.offset..tree_ref.offset + tree_ref.len]).unwrap();
        assert_eq!(desc.dimensions, 2);
        assert_eq!(desc.coord_bytes, 8);
        assert!(desc.interleaved);
        assert_eq!(desc.num_items, 4);
        assert_eq!(desc.node_size, 16);
        assert_eq!(nd, &node_data[..]);

        let pyld_ref = find_chunk(&chunks, TAG_PYLD).unwrap();
        let (_pd, body) =
            parse_pyld_chunk(&buf[pyld_ref.offset..pyld_ref.offset + pyld_ref.len]).unwrap();
        assert_eq!(&body[8..], b"blob"); // after the 8-byte table fragment
    }

    #[test]
    fn v2_unknown_critical_chunk_rejected() {
        let buf = encode_container(&[(*b"WHAT", true, vec![1, 2, 3, 4])]);
        // A reader that only knows TREE must reject an unknown critical chunk.
        assert_eq!(
            parse_container(&buf, &[TAG_TREE]).unwrap_err(),
            LoadError::UnsupportedVersion
        );
    }

    #[test]
    fn v2_unknown_optional_chunk_skipped() {
        let tree = tree_content(false, 4, 16, &[0u8; 32]);
        // An unknown OPTIONAL chunk is fine — the directory parses, reader ignores it.
        let buf = encode_container(&[(TAG_TREE, true, tree), (*b"XTRA", false, vec![9; 7])]);
        let chunks = parse_container(&buf, &[TAG_TREE]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(find_chunk(&chunks, TAG_TREE).is_some());
        assert!(find_chunk(&chunks, *b"XTRA").is_some());
    }

    #[test]
    fn v2_truncated_chunk_range_rejected() {
        let tree = tree_content(false, 4, 16, &[0u8; 16]);
        let mut buf = encode_container(&[(TAG_TREE, true, tree)]);
        buf.truncate(buf.len() - 4); // chunk now claims more bytes than exist
        assert!(matches!(
            parse_container(&buf, &[TAG_TREE]),
            Err(LoadError::InvalidTree | LoadError::Truncated)
        ));
    }

    #[test]
    fn payload_round_trip() {
        for &n in &[0usize, 1, 17, 100] {
            let index = build(n);
            let payloads: Vec<Vec<u8>> = (0..n).map(|i| format!("item-{i}").into_bytes()).collect();
            let bytes = index.to_bytes_with_payloads(&payloads).unwrap();

            // The TREE chunk is byte-identical to the index-only file's; only an
            // extra PYLD chunk (and the directory entry for it) is added.
            let index_only = index.to_bytes();
            let with = parse_container(&bytes, &[TAG_TREE]).unwrap();
            let plain = parse_container(&index_only, &[TAG_TREE]).unwrap();
            let wt = find_chunk(&with, TAG_TREE).unwrap();
            let pt = find_chunk(&plain, TAG_TREE).unwrap();
            assert_eq!(
                &bytes[wt.offset..wt.offset + wt.len],
                &index_only[pt.offset..pt.offset + pt.len]
            );
            assert!(find_chunk(&with, TAG_PYLD).is_some());

            let (_parsed, payload) = parse_index(&bytes, 2, 8).unwrap();
            let parsed = payload.expect("payload present");
            // The payload is leaf-ordered: slot `r` holds the blob of the item at
            // leaf rank `r`, whose insertion id is `index.indices[r]`.
            for r in 0..n {
                let insertion_id = index.indices[r];
                assert_eq!(payload_slice(&parsed, r), payloads[insertion_id].as_slice());
            }
        }
    }

    #[test]
    fn payload_count_mismatch_rejected() {
        let index = build(5);
        let payloads = vec![vec![1u8]; 3];
        assert_eq!(
            index.to_bytes_with_payloads(&payloads),
            Err(PayloadError::CountMismatch {
                expected: 5,
                got: 3
            })
        );
    }

    #[test]
    fn payload_file_loads_index_only_via_scalar_loader() {
        let index = build(10);
        let payloads: Vec<Vec<u8>> = (0..10).map(|_| vec![0u8; 4]).collect();
        let with = index.to_bytes_with_payloads(&payloads).unwrap();
        // The scalar loader reads the index from a payload file, ignoring the
        // payload (it validates the trailing section but does not retain it).
        let owned = Index2D::from_bytes(&with).unwrap();
        let query = Box2D::new(0.0, 0.0, 100.0, 100.0);
        assert_eq!(owned.search(query), index.search(query));
    }

    #[test]
    fn variable_length_payloads_round_trip() {
        let index = build(20);
        let payloads: Vec<Vec<u8>> = (0..20).map(|i| vec![i as u8; i]).collect(); // len 0..19
        let bytes = index.to_bytes_with_payloads(&payloads).unwrap();
        let (_parsed, payload) = parse_index(&bytes, 2, 8).unwrap();
        let parsed = payload.expect("payload present");
        for r in 0..20 {
            assert_eq!(
                payload_slice(&parsed, r),
                payloads[index.indices[r]].as_slice()
            );
        }
    }

    #[test]
    fn fixed_width_payload_round_trips_without_a_table() {
        const STRIDE: usize = 12;
        let n = 20;
        let index = build(n);
        // One 12-byte record per item, in item order.
        let mut flat = Vec::with_capacity(n * STRIDE);
        for i in 0..n {
            flat.extend_from_slice(&(i as u32).to_le_bytes());
            flat.extend_from_slice(&[i as u8; STRIDE - 4]);
        }
        let fixed = index.serialize().records(STRIDE, &flat).to_bytes().unwrap();
        let variable: Vec<Vec<u8>> = (0..n)
            .map(|i| flat[i * STRIDE..(i + 1) * STRIDE].to_vec())
            .collect();
        let var_bytes = index.to_bytes_with_payloads(&variable).unwrap();

        // The table-less layout is smaller by roughly the dropped offset table
        // (minus the 4-byte stride field the fixed descriptor adds, plus padding).
        let saving = var_bytes.len() - fixed.len();
        assert!((n * 8..=(n + 1) * 8).contains(&saving), "saving {saving}");

        let (_parsed, payload) = parse_index(&fixed, 2, 8).unwrap();
        let parsed = payload.expect("payload present");
        assert_eq!(parsed.stride, STRIDE);
        assert!(parsed.offsets.is_empty());
        for r in 0..n {
            let id = index.indices[r];
            assert_eq!(
                payload_slice(&parsed, r),
                &flat[id * STRIDE..(id + 1) * STRIDE]
            );
        }
    }

    #[test]
    fn fixed_width_wrong_record_size_rejected() {
        let index = build(4);
        // Three 8-byte records for four items, then declare stride 8: the count
        // (3 != 4) is caught first.
        let flat = vec![0u8; 3 * 8];
        assert!(matches!(
            index.serialize().records(8, &flat).to_bytes(),
            Err(PayloadError::CountMismatch { .. })
        ));
    }
}
