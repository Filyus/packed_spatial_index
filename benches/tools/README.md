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

## Output

Each tool prints **JSONL** on stdout — one JSON object per line, every row tagged
with a `tool` field (a `*_meta` row first, then data rows) — so a run can be
parsed, diffed, or aggregated instead of scraping a printed table. Read it by eye
with `jq`:

```bash
cargo run --release --manifest-path benches/tools/Cargo.toml --bin node_size_2d | jq
```

Set `BENCH_PIN_CORE=<n>` to pin the timing thread to one performance core for
low-noise numbers (a status line goes to stderr, so stdout stays pure JSONL):

```bash
BENCH_PIN_CORE=8 cargo run --release --manifest-path benches/tools/Cargo.toml --bin node_size_2d > run.jsonl
```
