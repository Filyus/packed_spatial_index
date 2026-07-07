# Agent guide

Operational notes for an AI agent (Codex/Claude) working in this repo.

## Commit conventions

- Subjects use **domain prefixes**, not conventional `feat:`/`fix:`. Pick the
  prefix from the [changelog taxonomy](RELEASING-AGENT.md#changelog-taxonomy)
  (e.g. `search:`, `knn:`, `raycast:`, `server:`, `perf:`, `builder2d:`,
  `docs:`, `bench:`, `lint:`). The prefix decides the changelog group.
- Put benchmark tables and perf numbers in the commit **body**, never the
  subject — only the subject reaches release notes.
- Measure before committing a perf change: A/B with `git stash` on a quiet
  machine, keep it only on a stable win, revert dead ends.

## Workflow

- Work directly on `main`. This is a single-maintainer repo; do not open a
  feature branch unless a change is speculative or you were asked to.
- Commit each task as soon as it is done — one task, one commit, with the
  matching domain prefix. Do not batch unrelated changes across tasks and then
  split them afterwards (that is double work, and the prefix grouping gets
  muddy). Run `cargo fmt --check` + `cargo clippy` + `cargo test` for the
  touched crate(s) before committing; the `geo/` companion crate is formatted
  separately (`cargo fmt --all --manifest-path geo/Cargo.toml`).
- Push only when the maintainer asks, or when preparing a release.

## Architecture discipline

The crate shares internal kernels (`TreeAccess` + `range` / `raycast` /
`neighbors::best_first`) so a traversal change lands in one place instead of
drifting across the owned / view / SIMD / f32 × 2D / 3D frontends. That is the
main maintenance win — reach for the shared kernel for cross-cutting algorithm
changes. But it cuts both ways, so:

- **Keep genuinely-different cases local.** Owned indexes iterate `&entries[..]`
  slices (LLVM autovectorizes the per-node test); byte views read per element.
  Do **not** route an owned hot path through the generic per-element kernel — it
  cost owned `visit` ~50–75% once before it was reverted to the local slice loop.
  A shared kernel that needs many special cases for ordinary behavior is a sign
  to stop sharing.
- **Measure owned paths specifically** when touching a shared kernel: the generic
  `TreeAccess` loop is free for views but not for slice-backed owned loops. Tests
  passing is **not** proof of no regression — A/B the hot path vs the prior commit.
- **File Boundary Rule:** extract for a separate reason to change or to remove
  real duplication; do not extract if a normal fix then has to touch four files
  instead of one. A 1000-line file can be fine.

## Releases

Releases are prepared **by hand** (no generator tool). The agent prepares the
version bump and changelog, shows the diff, and pushes the release commit only
after the maintainer OKs the wording; the maintainer alone approves the `release`
GitHub environment that publishes.

Full step list and the changelog taxonomy: **[`RELEASING-AGENT.md`](RELEASING-AGENT.md)**.
The human-facing contract: [`RELEASING.md`](RELEASING.md).
