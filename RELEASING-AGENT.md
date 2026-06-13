# Release preparation (agent)

How an AI agent prepares a release **by hand**. Companion to the human-facing
[`RELEASING.md`](RELEASING.md); referenced from [`AGENTS.md`](AGENTS.md).

The version bump and changelog are written by hand — there is no generator tool
in the loop. Our commits use domain prefixes (not conventional `feat:`/`fix:`),
so the bump is decided from the API surface and the changelog is grouped using
the [taxonomy](#changelog-taxonomy) below. The publish pipeline
(`.github/workflows/publish.yml`) runs its own preflight and tagging.

## Roles (keep them separate)

- **Agent**: prepares the bump + changelog, shows the diff, and — only after the
  maintainer OKs the wording — creates and pushes the release commit. Stops
  there.
- **Maintainer**: reviews the changelog wording before the push, and approves
  the `release` GitHub environment after CI + preflight. Only that approval
  publishes. The agent cannot and must not approve it.

## Steps

1. **Preconditions.** On `main`, clean tree, in sync with `origin/main`:
   ```sh
   git fetch origin
   git status --short                                    # must be empty
   git rev-list --left-right --count origin/main...HEAD  # must be "0  0"
   ```
   If the tree is dirty, classify per `RELEASING.md` (commit the missing change
   first and wait for CI, or ask) — never fold stray work into the release
   commit.

2. **Decide the version** from the public API surface (pre-1.0):
   - new public API only -> **minor** (`0.Y+1.0`)
   - bug/behavior fix only -> **patch**
   - removed/changed public API -> **major**

   `cargo semver-checks` (run by the publish preflight, not locally) is the
   backstop. Bump `version` in `Cargo.toml`.

3. **Build the changelog section** from `git log vPREV..HEAD`:
   - Heading inserted directly under `## [Unreleased]`:
     `## [X.Y.Z](https://github.com/Filyus/packed_spatial_index/compare/vPREV...vX.Y.Z) - YYYY-MM-DD`
   - Group commits by the [taxonomy](#changelog-taxonomy) below, in priority
     order.
   - Rewrite terse subjects into clear, **user-facing** notes: name the new
     public methods/types and a one-line "why it matters"; keep headline perf
     numbers brief.
   - **Drop** the internal-only groups and any commit touching only `wasm-demo/`
     (the demo is `exclude`d from the published crate; verify with
     `git show --stat <sha>`). A wasm-demo commit may carry a crate-domain prefix
     (`perf:`, `fix:`) by mistake — drop it regardless of prefix.

4. **Version-facing docs.** If the minor changed, update the README install pin
   (`packed_spatial_index = "0.X"`). Touch no other docs in the release commit.

5. **Show the diff and pause:**
   ```sh
   git diff -- Cargo.toml CHANGELOG.md README.md
   ```
   Wait for the maintainer to OK the changelog wording. Do **not** commit first.

6. **After the maintainer OKs**, commit exactly these files with the exact
   subject, then push:
   ```sh
   git add Cargo.toml CHANGELOG.md README.md
   git commit -m "release: prepare packed_spatial_index vX.Y.Z"
   git push origin main
   ```
   The subject must match `Cargo.toml`'s version exactly, or the publish workflow
   refuses to publish.

7. **After CI passes, start the publish workflow.** It does not run
   automatically (crates.io Trusted Publishing rejects the `workflow_run` event),
   so start it by hand against `main`:
   ```sh
   gh workflow run publish.yml --ref main
   ```
   It runs on the current `main` `HEAD`, which must still be the release commit —
   so do not push anything else before publishing. This only starts the pipeline;
   it gates at the `release` environment for the maintainer's approval.

8. **Stop.** Do not run semver/docs.rs/`--dry-run` locally (the preflight does).
   Do not publish, tag, or approve the `release` environment — the maintainer
   approves, and only then does the workflow publish, tag, and create the
   release.

## Changelog taxonomy

Commit domain prefixes map to changelog groups, rendered in **priority order**
(low number first). This table is the single source of truth for the taxonomy.
"Changelog" says whether the group reaches the user-facing release notes or is
dropped as internal noise.

| Prio | Group | Example prefixes | Changelog |
|---:|---|---|---|
| 00 | API | `api`, `builder`, `config`, `defaults`, `errors` | keep |
| 01 | Safety | `safety`, `unsafe` | keep |
| 02 | 2D | `2d`, `index2d`, `builder2d`, `sort2d`, `box2d`, `bounds2d`, `point2d` | keep |
| 03 | 3D | `3d`, `index3d`, `builder3d`, `sort3d`, `box3d`, `bounds3d`, `point3d` | keep |
| 04 | Geometry | `geometry`, `box(es)`, `bounds`, `point(s)` | keep |
| 05 | Indexes | `index` | keep |
| 06 | Search | `search`, `visit(or)`, `traversal`, `workspace`, `raycast`, `ray(s)` | keep |
| 07 | Nearest Neighbors | `knn`, `neighbor(s)`, `nearest` | keep |
| 08 | Persistence | `persistence`, `serialize`, `load`, `view(s)`, `format`, `bytes`, `zero-copy`, `stream` | keep |
| 09 | SIMD | `simd`, `soa`, `avx`, `avx512`, `sse` | keep |
| 10 | WASM | `wasm`, `wasm-demo` | depends — drop if it touches only `wasm-demo/` |
| 11 | Performance | `perf`, `parallel`, `radix`, `node(-size)`, `prefetch` | keep |
| 12 | Sorting and Encoding | `sort`, `sortkey`, `hilbert`, `morton`, `encoder(s)` | keep |
| 20 | Benchmarks | `bench(es)`, `flatgeobuf`, `static-aabb`, `compare` | drop (dev-only) |
| 90 | Documentation | `docs`, `readme`, `rustdoc`, `example(s)` | case-by-case |
| 91 | Tests | `test(s)`, `correctness`, `fuzz` | drop |
| 92 | Refactoring | `refactor`, `layout`, `tree`, `internal`, `module(s)` | drop |
| 93 | Lint | `lint(s)` | drop |
| 99 | Build, CI, and Packaging | `build`, `ci`, `deps`, `workflow`, `github`, `publish`, `tag`, `msrv`, `license` | drop |
| — | (skipped) | `release`, `repo`, `changelog` | drop (never in notes) |

Rules of thumb:
- "keep" groups are crate-user-facing; write a clear bullet per change.
- "drop" groups are internal; omit them from the user-facing changelog.
- "depends"/"case-by-case": include only if it affects crate users (a WASM crate
  API change, a doc fix worth surfacing), not demo-only or trivial churn.
- The `wasm-demo/` rule wins over the prefix: a demo-only commit is dropped even
  if it is prefixed `perf:`/`fix:`/etc.
