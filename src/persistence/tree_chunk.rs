use super::{ByteWriter, LoadError, read_u16_at, read_u32_at, read_u64_at, usize_from_u64};

/// The packed tree (descriptor + raw node data). Critical.
pub(crate) const TAG_TREE: [u8; 4] = *b"TREE";

/// Minimum `TREE` descriptor length this version writes.
pub(crate) const TREE_DESC_LEN: usize = 24;

/// Decoded `TREE` descriptor. `num_nodes` / `level_count` are *not* stored; the
/// caller derives them from `num_items` + `node_size`.
pub(crate) struct TreeDesc {
    pub(crate) dimensions: usize,
    pub(crate) coord_bytes: usize,
    pub(crate) interleaved: bool,
    pub(crate) num_items: usize,
    pub(crate) node_size: usize,
    /// Byte length of the descriptor (so node data starts at `desc_len`). Only
    /// the streaming reader consults it; the in-memory parser uses the node-data
    /// slice this comes paired with.
    #[cfg_attr(not(feature = "stream"), allow(dead_code))]
    pub(crate) desc_len: usize,
}

/// Parse a `TREE` chunk's descriptor; returns it plus the node-data slice that
/// follows. Validates the fixed fields but not the node data (the dimension
/// parsers do that against the derived tree shape).
pub(crate) fn parse_tree_chunk(chunk: &[u8]) -> Result<(TreeDesc, &[u8]), LoadError> {
    if chunk.len() < TREE_DESC_LEN {
        return Err(LoadError::Truncated);
    }
    let desc_len = read_u32_at(chunk, 0)? as usize;
    if desc_len < TREE_DESC_LEN || desc_len > chunk.len() {
        return Err(LoadError::InvalidTree);
    }
    let dimensions = chunk[4] as usize;
    let coord_bytes = chunk[5] as usize;
    let layout = chunk[6];
    if (dimensions != 2 && dimensions != 3) || (coord_bytes != 4 && coord_bytes != 8) || layout > 1
    {
        return Err(LoadError::UnsupportedVersion);
    }
    let num_items = read_u64_at(chunk, 8).and_then(usize_from_u64)?;
    let node_size = read_u16_at(chunk, 16)? as usize;
    if !(2..=65535).contains(&node_size) {
        return Err(LoadError::InvalidNodeSize { node_size });
    }
    Ok((
        TreeDesc {
            dimensions,
            coord_bytes,
            interleaved: layout == 1,
            num_items,
            node_size,
            desc_len,
        },
        &chunk[desc_len..],
    ))
}

impl ByteWriter<'_> {
    pub(crate) fn write_tree_desc(
        &mut self,
        dimensions: u8,
        coord_bytes: u8,
        interleaved: bool,
        num_items: usize,
        node_size: usize,
    ) {
        self.write_u32(TREE_DESC_LEN as u32);
        self.write_u8(dimensions);
        self.write_u8(coord_bytes);
        self.write_u8(if interleaved { 1 } else { 0 });
        self.write_u8(0);
        self.write_u64(num_items as u64);
        self.write_u16(node_size as u16);
        self.write_zeros(6);
    }
}
