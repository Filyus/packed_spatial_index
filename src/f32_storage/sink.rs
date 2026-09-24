/// Where the scalar f32 collect puts its hits: one item at a time from a
/// tested leaf, or a whole leaf range at once from a subtree the query covers.
/// Two methods rather than two closures, because both would need `&mut` to
/// the same output.
pub(crate) trait HitSink {
    /// One tested leaf item.
    fn one(&mut self, index: usize);
    /// Every item of a covered subtree, untested.
    fn all(&mut self, indices: &[usize]);
}

impl HitSink for Vec<usize> {
    #[inline]
    fn one(&mut self, index: usize) {
        self.push(index);
    }
    #[inline]
    fn all(&mut self, indices: &[usize]) {
        self.extend_from_slice(indices);
    }
}

/// Counts the hits without keeping them: a covered subtree adds its length.
pub(crate) struct CountSink(pub(crate) usize);

impl HitSink for CountSink {
    #[inline]
    fn one(&mut self, _: usize) {
        self.0 += 1;
    }
    #[inline]
    fn all(&mut self, indices: &[usize]) {
        self.0 += indices.len();
    }
}
