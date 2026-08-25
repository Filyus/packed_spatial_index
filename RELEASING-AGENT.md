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

There is deliberately no workflow triggered by the release tag. A tag pushed with
`GITHUB_TOKEN` raises no workflow events, so such a workflow would never run —
one existed for two months and never did, while `publish.yml` created every
release itself. Whatever a release has to produce goes in the workflow that
creates the tag, or is dispatched explicitly from it.

## Roles (keep them separate)

- **Agent**: prepares the bump + changelog for one crate and shows the diff.
  **That is the one stop.** After the maintainer OKs the wording it runs the rest
  without asking again: commit, push, wait for CI, start the publish workflow,
  and then check the crate is on crates.io and the GitHub Release exists. Asking
  again before the gate buys nothing — the gate is a button, and nobody but the
  maintainer can press it, early or otherwise.
- **Maintainer**: reviews the changelog wording before the push — that wording is
  what ships, as the GitHub Release body — and presses approve on the `release`
  GitHub environment when the pipeline reaches it. Only that approval publishes.
  The agent cannot and must not approve it.
  - That gate exists because the `release` environment carries a
    required-reviewer rule in the repository settings, not because the workflow
    names an environment. It also allows self-review, so it is a confirmation
    rather than a second pair of eyes.

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

Establish what is **actually published**, from the registry and the tags rather
than from the manifest:

```sh
git ls-remote --tags origin | grep -E "psi(-geo)?-v"
curl -s "https://crates.io/api/v1/crates/<crate>" | grep -o '"max_version":"[^"]*"'
```

A version bumped in the manifest, dated in the changelog and never published is
a real state: preparation that stopped before the commit or before the workflow.
Treat the last tag as the previous release, not the manifest. When the prepared
version exists nowhere outside the repository, fold the new work into its
section and re-date it rather than opening the next number — nobody can depend
on a version that was never there, and skipping it leaves a hole in the history
plus a compare link pointing at a tag that does not exist.

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

While a crate is `0.x`, "major" means `0.Y+1.0` as well: Cargo already treats a
change of `Y` as breaking for `0.x` dependents, so `0.Y+1.0` is the breaking
slot and there is no cheaper one. A release may therefore carry new API and a
breaking change together. This does **not** mean breaking changes are free —
say so plainly in the release notes, and prefer a non-breaking shape when one
exists. It means the cost is the same whether you break once or three times in
the same release, so batch them rather than deferring a needed break to avoid a
version number. Reaching `1.0.0` is a deliberate decision about stability, not
the automatic consequence of removing something.

Manifest paths:

- `packed_spatial_index`: `Cargo.toml`;
- `packed_spatial_index_geo`: `geo/Cargo.toml`.

Raising the pin in a dependent so it *requires* a newly released core is part
of that dependent's own release — it is an API-visible change and wants a
changelog line. That is separate from step 4, which keeps the pins resolvable
and happens in every release commit regardless.

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

### 4. Bump every pin that names this crate

The crates in this repository depend on each other by `path` **and** an exact
`version`. The path makes the build local; the version is what crates.io will
see. So the moment a manifest version changes, every dependent's requirement
stops matching the local crate and `cargo` refuses to resolve it — the whole
lane fails with `failed to select a version for the requirement ...`, and it
fails for the *dependent*, which is a job nobody is looking at during a release
of something else.

Those pins belong in the release commit, not in a follow-up. They are not
"other docs" under step 5: without them `CI: Rust checks` is red on the release
commit, and preflight requires a green one.

| Releasing | Also bump |
|---|---|
| `packed_spatial_index` | `geo/Cargo.toml`; `server/Cargo.toml`; `server/Cargo.lock` |
| `packed_spatial_index_geo` | `server/Cargo.toml` (**two** entries — default and `geojson`); `server/Cargo.lock` |

Find them rather than trusting the table, since a new dependent will not be in
it. Only a pin that carries a `version` can break — a `path`-only dependency
(the fuzz target, the wasm demos, `benches/tools`) resolves whatever the
manifest says and needs nothing:

```sh
grep -rn "<crate> = {.*version" --include=Cargo.toml . | grep -v "/target/"
```

Refresh each lockfile with `cargo update -p <crate> --manifest-path
<dependent>/Cargo.toml`; editing the version by hand leaves the lockfile's
checksum entry stale.

A dependent's pin may name a version that is not on crates.io yet — that is
normal and correct between the two releases. The `path` satisfies it locally,
so CI is green, while publishing the dependent stays blocked until the
dependency is actually published. Preflight checks exactly that, which is why
the dependent's release is a separate one that comes second.

### 5. Version-facing docs

If the minor changed, update install snippets in the relevant README:

- `packed_spatial_index`: `README.md`;
- `packed_spatial_index_geo`: `geo/README.md`.

Touch no other docs in the release commit unless they must mention the new
version.

### 6. Run the checks that fail on code

Run these before there is a release commit to protect. Preflight runs them too,
but by then the commit is pushed and has to stay at the head of `main`, so a
failure costs a rewrite of a shared branch:

```sh
rustup update nightly            # which rustdoc lints fire changes with the toolchain
cargo install cargo-semver-checks --locked   # its rustdoc JSON support lags nightly
RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc \
  --manifest-path <manifest> --no-deps --all-features
cargo semver-checks --manifest-path <manifest>
```

Update both, and in that order. `cargo-semver-checks` reads rustdoc's JSON,
whose format version moves with nightly, so a fresh nightly against a
months-old `cargo-semver-checks` fails with `unsupported rustdoc format vNN`
before it checks anything. CI installs the tool fresh on every run and never
sees this; only a local check does.

Neither is covered by `CI: Rust checks`. Its `lint & docs` lane builds docs on
**stable** and without `--cfg docsrs`, so a lint that only exists on nightly —
`rustdoc::redundant_explicit_links` among them — passes CI and fails preflight.
Semver compatibility is checked nowhere but preflight. Update nightly first: a
stale local toolchain passes what the runner rejects.

### 7. Show the diff and pause

```sh
git diff -- Cargo.toml CHANGELOG.md README.md geo/Cargo.toml geo/CHANGELOG.md geo/README.md
```

Wait for the maintainer to OK the changelog wording. Do **not** commit first.
If the wording was already committed under `## [Unreleased]`, still show the
final release diff so the maintainer can review the promoted `## [X.Y.Z]`
section before the release commit is created.

This pause is owed for each crate separately. A maintainer who delegated the
commands still reviews each changelog, and silence is not an OK.

It is also the **only** pause in the release. Everything after it is mechanical
and reversible until the `release` environment gate, which no amount of asking
can pass on the maintainer's behalf.

### 8. Commit and push after approval

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

### 9. Start the publish workflow after CI passes

Without asking — step 7 already covered it.

```sh
gh workflow run publish.yml --ref main -f crate=<crate>
```

It runs against `main` `HEAD`, which must still be the release commit. Start it
in the same turn that observes CI go green: nothing else guards that commit's
place at the head, and any push landing meanwhile invalidates it. `Pages: deploy
WASM demo` also runs on the push and is not a gate — it deploys the demo, and
preflight neither waits for it nor should. This only starts the pipeline; it
gates at the `release` environment for the maintainer.

### 10. If preflight fails

The subject check reads the head of `main`, so the repair is a normal commit and
a *new* release commit:

- fix the cause in its own commit, with its own domain prefix — code never
  belongs in the release commit;
- put the release commit back on top. Prefer rebuilding the pair over an empty
  marker commit: reset to the commit before the original release commit,
  cherry-pick the fix, cherry-pick the release commit, and
  `push --force-with-lease`. The alternative leaves two commits with the same
  release subject, one of them empty, in the history forever;
- a force-push to `main` is safe only while nothing points at those commits.
  Once the tag and the crates.io version exist, that door is closed;
- the rewrite discards the working tree. Commit or copy aside anything
  uncommitted first — `git reset --hard` takes unrelated drafts with it.

### 11. Wait at the gate, then confirm the release happened

Do not publish, tag, create releases, or approve the `release` environment. The
pipeline does the first three itself once the maintainer presses approve, and the
approval is theirs alone.

A green publish workflow is not the same as a release. Check the thing it was
supposed to produce: the version on crates.io, and the GitHub Release for the tag
with the changelog section as its body. Read that from the registry and the
release list, never from the manifest — a version bumped and dated but published
nowhere is a real state, and it looks identical in the repository.

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
