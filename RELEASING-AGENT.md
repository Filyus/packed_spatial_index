# Release preparation (agent)

How an AI agent prepares a release **by hand**. Companion to the human-facing
[`RELEASING.md`](RELEASING.md); referenced from [`AGENTS.md`](AGENTS.md).

The two crates — `packed_spatial_index` and `packed_spatial_index_geo` — are
versioned and released **independently**, one crate per release. Each has its own
changelog and its own release tag prefix:

| Crate | Tag prefix | Example |
|---|---|---|
| `packed_spatial_index` | `psi` | `psi-v0.18.2` |
| `packed_spatial_index_geo` | `psi-geo` | `psi-geo-v0.1.0` |

Older core releases originally used bare `vX.Y.Z` tags, but the release history
has been migrated to the `psi-vX.Y.Z` namespace. Do not create new bare tags.

Our commits use **domain prefixes** (not conventional `feat:`/`fix:`), and the
changelog is grouped using the [taxonomy](#changelog-taxonomy) below. The publish
pipeline (`.github/workflows/publish.yml`, run per crate) does its own preflight,
tagging, and GitHub release.

## Roles (keep them separate)

- **Agent**: prepares the bump + changelog for one crate, shows the diff, and —
  only after the maintainer OKs the wording — creates and pushes the release
  commit. Stops there unless explicitly asked to start the workflow.
- **Maintainer**: reviews the changelog wording before the push, and approves the
  `release` GitHub environment after CI + preflight. Only that approval
  publishes. The agent cannot and must not approve it.

## Dependency order

`packed_spatial_index` <- `packed_spatial_index_geo`. A dependent can only be
released after the dependency version it pins is published on crates.io. If a
release bumps `packed_spatial_index`, releasing geo to pick it up is a
**separate** release: bump the geo pin, write its changelog, then publish/tag geo.

## Steps (for one crate `<crate>`)

`<crate>` must be one of:

- `packed_spatial_index`;
- `packed_spatial_index_geo`.

### 1. Preconditions

On `main`, clean tree, in sync with `origin/main`:

```sh
git fetch origin
git status --short                                    # must be empty
git rev-list --left-right --count origin/main...HEAD  # must be "0  0"
```

If the tree is dirty, classify per `RELEASING.md`; never fold stray work into
the release commit.

For geo changes, run the formatter exactly as the geo CI lane does:

```sh
cargo fmt --all --check --manifest-path geo/Cargo.toml
```

The root `cargo fmt --all` does not cover `geo/`, because the companion crate is
kept outside the root workspace.

### 2. Decide the version

Use the selected crate's public API surface (pre-1.0):

- new public API only -> minor (`0.Y+1.0`);
- bug/behavior fix only -> patch;
- removed/changed public API -> major.

Manifest paths:

- `packed_spatial_index`: `Cargo.toml`;
- `packed_spatial_index_geo`: `geo/Cargo.toml`.

If geo should pick up a newly released core, update the
`packed_spatial_index` dependency pin in `geo/Cargo.toml` as part of the geo
release, after the core version is published.

### 3. Build or promote the changelog section

Changelog paths:

- `packed_spatial_index`: `CHANGELOG.md`;
- `packed_spatial_index_geo`: `geo/CHANGELOG.md`.

Feature commits may already have added user-facing notes under `## [Unreleased]`.
That is allowed and often preferred: start from those notes, audit them against
`git log <previous-tag>..HEAD` (or all relevant history for the first release),
add anything missing, and remove internal noise. Do not rewrite or delete good
`Unreleased` notes merely because they were committed before release prep.

Heading under `## [Unreleased]`:

```text
## [X.Y.Z](https://github.com/Filyus/packed_spatial_index/compare/<previous-tag>...<release-tag>) - YYYY-MM-DD
```

For a first release:

```text
## [X.Y.Z] - YYYY-MM-DD
```

For core releases, use `psi-vX.Y.Z`. For geo releases, use `psi-geo-vX.Y.Z`.

Include only commits that affect the selected crate. Verify with
`git show --stat <sha>` when a prefix is ambiguous. Group by the
[taxonomy](#changelog-taxonomy), in priority order.

Rewrite terse subjects into clear, **user-facing** notes: name the affected
public methods/types/features and a one-line "why it matters". Drop internal
noise: tests only, lint, CI/workflow only, benchmark-only, demo-only, and
`release:` commits.

When the wording is ready, promote the selected crate's `Unreleased` content to
the dated version section. Leave `## [Unreleased]` present above it and empty.

### 4. Version-facing docs

If the minor changed, update install snippets in the relevant README:

- `packed_spatial_index`: `README.md`;
- `packed_spatial_index_geo`: `geo/README.md`.

Touch no other docs in the release commit unless they must mention the new
version.

### 5. Show the diff and pause

```sh
git diff -- Cargo.toml CHANGELOG.md README.md geo/Cargo.toml geo/CHANGELOG.md geo/README.md
```

Wait for the maintainer to OK the changelog wording. Do **not** commit first.
If the wording was already committed under `## [Unreleased]`, still show the
final release diff so the maintainer can review the promoted `## [X.Y.Z]`
section before the release commit is created.

### 6. Commit and push after approval

Commit exactly the release files that changed for release prep with the exact
subject:

```sh
git commit -m "release: prepare <crate> vX.Y.Z"
git push origin main
```

The subject must match the selected crate's manifest version exactly, or the
publish workflow refuses to publish. The changelog prose may have landed in an
earlier feature commit, but this release commit's `HEAD` must contain a
non-empty `## [X.Y.Z]` section for the selected crate.

### 7. Start the publish workflow after CI passes

```sh
gh workflow run publish.yml --ref main -f crate=<crate>
```

It runs against `main` `HEAD`, which must still be the release commit. This only
starts the pipeline; it gates at the `release` environment for the maintainer.

### 8. Stop

Do not publish, tag, create releases, or approve the `release` environment unless
the maintainer explicitly asks for that specific action.

## First release note

For a brand-new crate, Trusted Publishing cannot create the crate on crates.io.
The maintainer publishes the first version locally with a short-lived token, then
runs:

```sh
gh workflow run tag-first-release.yml --ref main \
  -f crate=<crate> \
  -f version=X.Y.Z \
  -f confirm="tag <crate>"
```

The first-release tag is `psi-vX.Y.Z` for core or `psi-geo-vX.Y.Z` for geo. The
workflow also creates the GitHub Release from the selected crate's changelog.

## Changelog taxonomy

Commit domain prefixes map to changelog groups, rendered in priority order (low
number first). The "Crate" column is a routing hint; the actual crate is decided
by which files the commit touched.

| Prio | Group | Crate | Example prefixes | Changelog |
|---:|---|---|---|---|
| 00 | API | the touched crate | `api`, `builder`, `config`, `defaults`, `errors` | keep |
| 01 | Safety | the touched crate | `safety`, `unsafe`, `security`, `hardening` | keep |
| 02 | 2D | `packed_spatial_index` | `2d`, `index2d`, `builder2d`, `sort2d`, `box2d`, `bounds2d`, `point2d` | keep |
| 03 | 3D | `packed_spatial_index` | `3d`, `index3d`, `builder3d`, `sort3d`, `box3d`, `bounds3d`, `point3d` | keep |
| 04 | Geometry | the touched crate | `geometry`, `geo`, `geoparquet`, `boxes`, `bounds`, `points` | keep |
| 05 | Indexes | the touched crate | `index`, `builder`, `accelerator` | keep |
| 06 | Search | `packed_spatial_index` | `search`, `visit`, `traversal`, `workspace`, `raycast`, `rays` | keep |
| 07 | Nearest Neighbors | `packed_spatial_index` | `knn`, `neighbors`, `nearest` | keep |
| 08 | Persistence | the touched crate | `persistence`, `serialize`, `load`, `views`, `format`, `bytes`, `zero-copy`, `stream`, `converter` | keep |
| 09 | SIMD | `packed_spatial_index` | `simd`, `soa`, `avx`, `avx512`, `sse` | keep |
| 10 | WASM | web / the touched crate | `wasm`, `wasm-demo`, `demo` | depends |
| 11 | Performance | the touched crate | `perf`, `parallel`, `radix`, `node-size`, `prefetch` | keep if measured and user-facing |
| 12 | Sorting and Encoding | `packed_spatial_index` | `sort`, `sortkey`, `hilbert`, `morton`, `encoders` | keep |
| 13 | Discovery | `packed_spatial_index_geo` | `discovery`, `discover`, `inspect` | keep |
| 14 | Validation | `packed_spatial_index_geo` | `validation`, `validate`, `diagnostics` | keep |
| 15 | Server | `packed_spatial_index_server` | `server`, `http`, `catalog`, `features-api` | keep |
| 20 | Benchmarks | — | `bench`, `benches`, `flatgeobuf`, `static-aabb`, `compare` | drop |
| 90 | Documentation | the touched crate | `docs`, `readme`, `rustdoc`, `examples` | case-by-case |
| 91 | Tests | — | `test`, `tests`, `correctness`, `fuzz` | drop |
| 92 | Refactoring | — | `refactor`, `layout`, `tree`, `internal`, `modules` | drop |
| 93 | Lint | — | `lint`, `fmt`, `clippy`, `style` | drop |
| 99 | Build, CI, and Packaging | — | `build`, `ci`, `deps`, `workflow`, `github`, `publish`, `tag`, `msrv`, `license` | drop unless release behavior changed |
| — | (skipped) | — | `release`, `repo`, `changelog` | drop |

Rules of thumb:

- "keep" groups are crate-user-facing; write a clear bullet per change.
- Both `CHANGELOG.md` and `geo/CHANGELOG.md` use these domain `###` headers — not
  Keep-a-Changelog `Added`/`Changed`/`Fixed`. (geo's pre-0.13 history was
  back-filled to the taxonomy on 2026-06-29; `Discovery` and `Validation` are
  geo-companion domains.)
- Include `geo:` changes in `geo/CHANGELOG.md` when they touch the companion
  crate or its package/release behavior.
- Omit browser-demo-only polish even if the prefix looks user-facing.
- Never include `release:` commits in release notes.
