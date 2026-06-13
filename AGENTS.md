# Agent guide

Operational notes for an AI agent (Codex/Claude) working in this repo.

## Commit conventions

- Subjects use **domain prefixes**, not conventional `feat:`/`fix:`. Pick the
  prefix from the [changelog taxonomy](RELEASING-AGENT.md#changelog-taxonomy)
  (e.g. `search:`, `knn:`, `raycast:`, `perf:`, `builder2d:`, `docs:`, `bench:`,
  `lint:`). The prefix decides the changelog group.
- Put benchmark tables and perf numbers in the commit **body**, never the
  subject — only the subject reaches release notes.
- Measure before committing a perf change: A/B with `git stash` on a quiet
  machine, keep it only on a stable win, revert dead ends.

## Releases

Releases are prepared **by hand** (no generator tool). The agent prepares the
version bump and changelog, shows the diff, and pushes the release commit only
after the maintainer OKs the wording; the maintainer alone approves the `release`
GitHub environment that publishes.

Full step list and the changelog taxonomy: **[`RELEASING-AGENT.md`](RELEASING-AGENT.md)**.
The human-facing contract: [`RELEASING.md`](RELEASING.md).
