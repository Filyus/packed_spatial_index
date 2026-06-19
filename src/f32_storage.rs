/// Round `x` down to the nearest `f32` that is `<= x`.
#[inline]
pub(crate) fn round_down(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) > x { r.next_down() } else { r }
}

/// Round `x` up to the nearest `f32` that is `>= x`.
#[inline]
pub(crate) fn round_up(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64) < x { r.next_up() } else { r }
}

/// High bit of the stacked level word, set when the query fully contains a node so
/// its whole subtree can be collected without further overlap tests.
#[cfg(feature = "simd")]
pub(crate) const CONTAINED_FLAG: usize = 1usize << (usize::BITS - 1);

#[cfg(feature = "simd")]
pub(crate) const LEVEL_MASK: usize = !CONTAINED_FLAG;

#[inline]
#[cfg(feature = "simd")]
pub(crate) fn encode_level(level: usize, contained: bool) -> usize {
    if contained {
        level | CONTAINED_FLAG
    } else {
        level
    }
}
