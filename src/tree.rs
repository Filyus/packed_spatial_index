pub(crate) const MIN_NODE_SIZE: usize = 2;
pub(crate) const MAX_NODE_SIZE: usize = 65_535;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeLayout {
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) num_nodes: usize,
}

#[inline]
pub(crate) fn normalize_node_size(node_size: usize) -> usize {
    node_size.clamp(MIN_NODE_SIZE, MAX_NODE_SIZE)
}

pub(crate) fn compute_tree_layout(num_items: usize, node_size: usize) -> TreeLayout {
    debug_assert!(node_size >= MIN_NODE_SIZE);
    debug_assert!(node_size <= MAX_NODE_SIZE);

    let mut level_bounds = Vec::new();
    let mut num_nodes = num_items;
    let mut level_width = num_items;
    level_bounds.push(level_width);

    if num_items > 0 {
        loop {
            level_width = level_width.div_ceil(node_size);
            num_nodes = num_nodes
                .checked_add(level_width)
                .expect("packed tree node count overflow");
            level_bounds.push(num_nodes);
            if level_width == 1 {
                break;
            }
        }
    }

    TreeLayout {
        level_bounds,
        num_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_tree_layout, normalize_node_size};

    #[test]
    fn empty_layout_has_leaf_level_only() {
        let layout = compute_tree_layout(0, 16);
        assert_eq!(layout.level_bounds, vec![0]);
        assert_eq!(layout.num_nodes, 0);
    }

    #[test]
    fn single_node_layout_adds_root_node() {
        let layout = compute_tree_layout(16, 16);
        assert_eq!(layout.level_bounds, vec![16, 17]);
        assert_eq!(layout.num_nodes, 17);
    }

    #[test]
    fn multi_level_layout_uses_integer_div_ceil() {
        let layout = compute_tree_layout(257, 16);
        assert_eq!(layout.level_bounds, vec![257, 274, 276, 277]);
        assert_eq!(layout.num_nodes, 277);
    }

    #[test]
    fn multi_level_layout_respects_node_size() {
        let layout = compute_tree_layout(65, 8);
        assert_eq!(layout.level_bounds, vec![65, 74, 76, 77]);
        assert_eq!(layout.num_nodes, 77);
    }

    #[test]
    fn node_size_is_clamped_to_supported_range() {
        assert_eq!(normalize_node_size(0), 2);
        assert_eq!(normalize_node_size(1), 2);
        assert_eq!(normalize_node_size(16), 16);
        assert_eq!(normalize_node_size(usize::MAX), 65_535);
    }
}
