# Releasing

This repository publishes two independently versioned crates:

- `packed_spatial_index` from the repository root;
- `packed_spatial_index_geo` from `geo/`.

Each crate has its own `Cargo.toml`, its own `CHANGELOG.md`, and its own release
tag prefix:

| Crate | Tag prefix | Example |
|---|---|---|
| `packed_spatial_index` | `psi` | `psi-v0.18.2` |
| `packed_spatial_index_geo` | `psi-geo` | `psi-geo-v0.1.0` |

Older `packed_spatial_index` releases originally used bare `vX.Y.Z` tags, but
the release history has been migrated to the `psi-vX.Y.Z` namespace. Do not
create new bare release tags.

In the steps below, **`<crate>` is the crate being released**. Substitute either
`packed_spatial_index` or `packed_spatial_index_geo`.

## Dependency order

`packed_spatial_index_geo` depends on `packed_spatial_index`. If a geo release
needs a newer core version, release `packed_spatial_index` first, wait until that
version is visible on crates.io, then update/publish `packed_spatial_index_geo`.
Each crate release gets its own commit, changelog section, and tag.

## Normal Release

Use this path after the crate already exists on crates.io and Trusted Publishing
is configured for that crate.

### 1. Prepare the release locally

Start from `main` and inspect the working tree:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
```

If `git status --short` prints anything, classify the changes before preparing
the release. Commit any missing feature/fix/doc work first and wait for CI, or
ask. Do not fold in-progress work into the release commit.

Prepare the bump and changelog by hand, following
[`RELEASING-AGENT.md`](RELEASING-AGENT.md). The release files are:

| Crate | Manifest | Changelog | Release-facing docs |
|---|---|---|---|
| `packed_spatial_index` | `Cargo.toml` | `CHANGELOG.md` | `README.md`, only if the install pin changes |
| `packed_spatial_index_geo` | `geo/Cargo.toml` | `geo/CHANGELOG.md` | `geo/README.md`, only if the install pin changes |

The release commit should normally change only that crate's manifest,
changelog, and necessary release-facing docs. Show the maintainer the diff before
committing:

```powershell
git diff -- Cargo.toml CHANGELOG.md README.md geo/Cargo.toml geo/CHANGELOG.md geo/README.md
```

### 2. Commit and push

After approval, create one release commit with this exact subject:

```text
release: prepare <crate> vX.Y.Z
```

Example:

```powershell
git add <release-files>
git commit -m "release: prepare packed_spatial_index_geo v0.1.1"
git push origin main
```

The exact subject matters. The publish workflow ignores ordinary pushes and only
continues when the subject matches the selected crate and the version in that
crate's `Cargo.toml`.

### 3. Start the publish workflow

The push to `main` starts `CI: Rust checks`. Wait for it to pass, then start the
publish workflow for the crate:

```powershell
gh workflow run publish.yml --ref main -f crate=<crate>
```

The workflow runs against the current `main` `HEAD`, which must still be the
release commit.

Preflight checks:

- a successful `CI: Rust checks` run exists for this commit;
- the commit subject is exactly `release: prepare <crate> vX.Y.Z`;
- `X.Y.Z` matches the selected crate's manifest version;
- any internal dependency pin, such as geo's `packed_spatial_index` dependency,
  is already published on crates.io;
- the selected crate's changelog has a `## [X.Y.Z]` section;
- `cargo semver-checks` succeeds if the crate already exists on crates.io;
- the docs.rs-style nightly docs build succeeds;
- `<crate> X.Y.Z` is not already published;
- tag `psi-vX.Y.Z` or `psi-geo-vX.Y.Z` does not already exist;
- `cargo publish --dry-run` succeeds for the selected manifest.

If preflight succeeds, the workflow enters the `release` environment and waits
for maintainer approval.

### 4. Final approval

The maintainer approves the waiting `release` environment deployment. Before
approving, check:

- the workflow is `Release: publish crate`;
- the selected crate and version are intended;
- the release SHA is the release commit that passed CI;
- the changelog section is the one reviewed before the release commit.

After approval, the workflow authenticates to crates.io through Trusted
Publishing, publishes the selected crate, creates the annotated release tag,
extracts the selected changelog section, and creates the GitHub Release.

## First Release

Trusted Publishing cannot publish a crate that does not exist yet. For a brand
new crate, do the first publish locally with a short-lived crates.io token, then
create the tag with the one-off workflow.

1. Push the `release: prepare <crate> vX.Y.Z` commit to `main` if there is a
   release commit, and wait for CI. For an initial crate already prepared on
   `main`, ensure the current `main` contains the intended version/changelog.
2. Create a crates.io token:
   - expiration: short, for example one day;
   - scope: `publish-new`;
   - crate restriction: unrestricted, because the crate does not exist yet.
3. Publish locally:

   ```bash
   cargo login <token>
   cargo publish --manifest-path geo/Cargo.toml
   cargo logout
   ```

   For another new crate, substitute its manifest path. If publishing several
   new crates, publish in dependency order and wait for each dependency to become
   visible on crates.io before publishing its dependent.

4. Revoke the token.
5. Create the first release tag with the one-off workflow:

   ```powershell
   gh workflow run tag-first-release.yml --ref main `
     -f crate=packed_spatial_index_geo `
     -f version=0.1.0 `
     -f confirm="tag packed_spatial_index_geo"
   ```

   The workflow verifies that the selected version is already published and that
   `psi-geo-v0.1.0` is absent, then pushes the tag and creates the GitHub
   Release from `geo/CHANGELOG.md`.

6. Configure Trusted Publishing for the crate before its next release.

## One-Time Setup

This repository is already configured. Use this section as the checklist when
recreating repository settings.

### GitHub Actions permissions

In the GitHub repository settings, open:

`Settings` -> `Actions` -> `General` -> `Workflow permissions`

Enable:

- `Read and write permissions`.

This is required so the publish/tag/release workflows can push annotated tags and
create GitHub Releases with `GITHUB_TOKEN`.

### Release environment approval

The publish workflow uses `environment: release` for the real publish job.
Configure that environment with required reviewers so a real publish cannot
proceed without approval after preflight.

Expected settings:

- Environment name: `release`;
- Required reviewers: `Filyus`;
- Prevent self-review: disabled, so the repository owner can approve their own
  release deployment;
- Wait timer: `0`;
- Deployment branch policy: none. The workflow checks that it was started from
  `main`, that `main` `HEAD` is the matching release commit, and that CI passed.

### Trusted Publishing

Trusted Publishing is the expected publishing path for normal updates after a
crate exists on crates.io. Configure one Trusted Publisher entry per crate:

- `packed_spatial_index`;
- `packed_spatial_index_geo`.

Each entry should point at:

- Publisher: `GitHub`;
- Repository owner: `Filyus`;
- Repository name: `packed_spatial_index`;
- Workflow filename: `publish.yml`;
- Environment name: `release`.
