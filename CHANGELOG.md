# Changelog

All notable changes to this crate are documented here.

## [Unreleased]

### Performance
- Extend the covered-range optimization across the whole SIMD family: the owned
  indexes (`SimdIndex2D`, `SimdIndex3D`), the zero-copy views (`SimdIndex2DView`,
  `SimdIndex3DView`), and the `f32-storage` variants and their views. When a query
  fully contains a node, its whole subtree is collected by copying the contiguous
  leaf-index range (or visiting it directly) instead of running per-item overlap
  tests. Large-window searches are ~2.4x faster and full-extent searches up to
  ~12x faster, so the SIMD paths now match or beat the AoS index across every
  window size instead of regressing on large windows. On the conservative
  (non-refined) `f32` path the shortcut uses the rounded query and the stored
  rounded boxes, so it returns exactly the same set as the per-item traversal; the
  exact (`search_exact`) path is unchanged and still re-checks each candidate.


## [0.4.2](https://github.com/Filyus/packed_spatial_index/compare/v0.4.1...v0.4.2) - 2026-06-08

### SIMD
- Update SIMD comparisons for `wide` 1.5.

### Documentation
- Add README notes for AI usage and prior art.
- Clarify the live WASM demo link.

### WASM Demo
- Publish the interactive demo through GitHub Pages.

## [0.4.1](https://github.com/Filyus/packed_spatial_index/compare/v0.4.0...v0.4.1) - 2026-06-05

### API
- Add opt-in `f32-storage` SIMD indexes for compact coordinate storage.
- Add exact range and KNN callbacks for `f32-storage` indexes using
  caller-owned `f64` boxes.

### Binary Format
- Document the packed spatial index binary format.
- Add distinct f32 box layout flags for `f32-storage` indexes.

### WASM
- Add the interactive WASM demo for 2D and 3D searches.
- Add 3D depth slicing, depth coloring, and an interactive depth legend.
- Tighten demo controls, query overlays, status bar, and wrapper helpers.

### Benchmarks
- Add f32-vs-f64 storage benchmarks for range queries and KNN.

### Documentation
- Document f32 storage trade-offs, exact query APIs, and benchmark guidance.

### Examples
- Add an f32 exact-query example.

### Tests
- Add f32 storage coverage for range search, exact range search, KNN,
  persistence, and views.
- Add proptest search and persistence robustness checks
- Rustfmt proptest files

## [0.4.0](https://github.com/Filyus/packed_spatial_index/compare/v0.3.3...v0.4.0) - 2026-06-03

### API
- Return `BuildError::TreeTooLarge` instead of panicking when a requested tree
  layout cannot fit in memory.

### Benchmarks
- Move internal performance tools out of the published examples and into a
  local benchmark tools package.


## [0.3.3](https://github.com/Filyus/packed_spatial_index/compare/v0.3.2...v0.3.3) - 2026-06-03

### Geometry
- Add point box constructors
- Share box accumulator helpers

### Documentation
- Add docs.rs feature badges and verify the docs.rs build
- Clarify query API guidance

### Lint
- Require SAFETY comments on all unsafe blocks

## [0.3.2](https://github.com/Filyus/packed_spatial_index/compare/v0.3.1...v0.3.2) - 2026-06-02

### SIMD

- Add zero-copy SIMD views

### Documentation

- Clarify release-plz release flow
- Document environment approval setup
- Fold tag fallback into first release
- Reorder release guide sections

### Build, CI, and Packaging

- Add safe release-plz draft workflow
- Make release-plz dry run preview only
- Run semver checks in release-plz workflow
- Simplify release workflows
- Clarify workflow names
- Use action-oriented workflow names
- Rename prepare workflow file
- Use lowercase manual run names
