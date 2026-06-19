/// High bit of the stacked level word, set when the query fully contains a node so
/// its whole subtree can be collected without further overlap tests.
pub(crate) const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);

pub(crate) const LEVEL_MASK: usize = !CONTAINED_FLAG;

#[inline]
pub(crate) fn encode_level(level: usize, contained: bool) -> usize {
    if contained {
        level | CONTAINED_FLAG
    } else {
        level
    }
}
