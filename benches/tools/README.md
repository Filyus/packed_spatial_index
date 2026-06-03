# Local Performance Tools

This package contains developer-only programs for comparing internal build and
layout choices. Each `src/*.rs` file is listed as a Cargo binary target in
`Cargo.toml`. They are intentionally kept out of the crate `examples/` directory
so published examples stay small and user-facing.

Run a tool from the repository root:

```bash
cargo run --release --manifest-path benches/tools/Cargo.toml --bin sortkey_quality_2d
cargo run --release --manifest-path benches/tools/Cargo.toml --bin node_size_3d
```

These tools use the hidden `bench-internals` feature and are excluded from the
published crate package.
