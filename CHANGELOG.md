# Changelog

All notable changes to this crate are documented here.

## [Unreleased]


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
