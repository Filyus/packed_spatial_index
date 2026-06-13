# Releasing

Normal releases are optimized for a solo maintainer working with Codex:

1. Codex verifies that the working tree is clean and current with `origin/main`.
2. Codex prepares the version bump and release notes locally.
3. The maintainer reviews the `CHANGELOG.md` diff.
4. Codex pushes one release commit to `main`.
5. CI validates that exact commit.
6. Once CI is green, the `Release: publish crate` workflow is started manually
   (`gh workflow run publish.yml --ref main`, or the "Run workflow" button). It
   runs preflight checks, then waits for the `release` environment approval.
7. The maintainer approves the environment deployment. Only then does the
   workflow publish to crates.io, create the annotated tag, and create the
   GitHub Release.

The publish is started by hand rather than automatically after CI: crates.io
Trusted Publishing rejects GitHub's `workflow_run` event, so the publish runs on
the `workflow_dispatch` trigger instead.

Release preparation (version bump + changelog) is done by hand following
[`RELEASING-AGENT.md`](RELEASING-AGENT.md). Nothing in this flow publishes
crates, pushes tags, or creates GitHub Releases outside the gated publish
workflow.

## Normal Release

Use this path after the crate already exists on crates.io and Trusted
Publishing is configured.

### 1. Prepare the Release Locally

Codex must start from `main` and inspect the working tree before generating
anything:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
```

If `git status --short` prints anything, classify the changes before preparing
the release:

- If the current conversation makes it clear what normal feature/fix/docs commit
  is missing, create that commit first, push it to `main`, and wait for CI.
- If the changes are clearly unrelated local work, ask whether to commit, stash,
  discard, or postpone them.
- If the changes are ambiguous, stop and ask. Do not guess and do not fold them
  into the release commit.

The release commit should be easy to review and should not accidentally absorb
in-progress work.

Codex prepares the version bump and changelog **by hand**, following the step
list in [`RELEASING-AGENT.md`](RELEASING-AGENT.md). In short:

- bump `version` in `Cargo.toml` (minor for new API, patch for fixes);
- write the `CHANGELOG.md` section grouped by the
  [changelog taxonomy](RELEASING-AGENT.md#changelog-taxonomy);
- rewrite terse commit subjects into clear, user-facing notes;
- remove internal noise (wasm-demo-only commits, lint, CI/workflow, benches);
- keep feature/code changes out of the release commit.

Do not run release-only checks manually. The publish workflow's preflight runs
semver compatibility, docs.rs, duplicate-version, duplicate-tag, and
`cargo publish --dry-run` checks before the final approval gate.

The release commit should normally change only:

- `Cargo.toml`;
- `CHANGELOG.md`;
- small release-facing docs, only when they must mention the new version.

Codex must show the maintainer the diff before committing:

```powershell
git diff -- Cargo.toml CHANGELOG.md RELEASING.md README.md
```

The maintainer reviews the changelog wording and says whether it is ready.

### 2. Commit and Push

After approval, Codex reads the version from `Cargo.toml` and creates one
release commit with this exact subject:

```text
release: prepare packed_spatial_index vX.Y.Z
```

Example:

```powershell
$version = python -c "import tomllib; print(tomllib.load(open('Cargo.toml', 'rb'))['package']['version'])"

git add Cargo.toml CHANGELOG.md README.md RELEASING.md
git commit -m "release: prepare packed_spatial_index v$version"
git push origin main
```

The exact commit subject matters. The publish workflow ignores ordinary pushes
and only continues when the subject matches the version in `Cargo.toml`.

### 3. Start the Publish Workflow and Preflight

The push to `main` starts `CI: Rust checks`. Wait for it to pass.

Once CI is green, start `Release: publish crate` manually against `main`:

```powershell
gh workflow run publish.yml --ref main
```

(or use the "Run workflow" button on the workflow's Actions page). It runs on the
current `main` `HEAD`, which must still be the release commit. crates.io Trusted
Publishing does not accept GitHub's `workflow_run` event, so the publish is
started by hand on the `workflow_dispatch` trigger rather than automatically
after CI.

If `main` `HEAD` is not a release commit, the workflow exits early:

```text
Not a release commit; nothing to publish.
```

For a release commit, preflight checks:

- a successful `CI: Rust checks` run exists for this commit;

- the commit subject is exactly `release: prepare packed_spatial_index vX.Y.Z`;
- `X.Y.Z` matches the `Cargo.toml` package version;
- `CHANGELOG.md` has a `## [X.Y.Z]` section;
- `cargo semver-checks` succeeds;
- the docs.rs-style nightly documentation build succeeds;
- `packed_spatial_index X.Y.Z` is not already published on crates.io;
- tag `vX.Y.Z` does not already exist;
- `cargo publish --dry-run` succeeds.

If preflight succeeds, the workflow enters the `release` environment and waits
for manual approval.

### 4. Final Approval

The maintainer approves the waiting `release` environment deployment in GitHub
Actions.

Before approving, check:

- the workflow is `Release: publish crate`;
- the release SHA is the release commit that passed CI;
- the version is the intended `Cargo.toml` version;
- the changelog section is the one reviewed before the release commit.

After approval, the workflow:

1. authenticates to crates.io through Trusted Publishing;
2. runs `cargo publish`;
3. creates annotated tag `vX.Y.Z`;
4. extracts the `CHANGELOG.md` section for `X.Y.Z`;
5. creates the GitHub Release from those notes.

No normal release should require entering a version, setting `publish: true`, or
typing a confirmation phrase. Those were part of the old manual publish
workflow.

### Rerun

`Release: publish crate` is started with `workflow_dispatch` and has no inputs,
so a rerun is just starting it again (`gh workflow run publish.yml --ref main`).
It runs against the current `main` commit and always requires:

- a valid release commit subject (`main` `HEAD`);
- successful `CI: Rust checks` for that exact commit;
- all preflight checks;
- the final `release` environment approval.

A failed publish (e.g. a transient crates.io error) leaves nothing published as
long as it stopped before `cargo publish`; rerun once `main` `HEAD` is still the
release commit.

## First Release

Trusted Publishing cannot publish a crate that does not exist yet. For a brand
new crate, do the first publish locally with a short-lived token, then configure
Trusted Publishing for normal updates.

1. Push the release commit to `main` and wait for CI to pass.
2. Create a crates.io token:
   - expiration: short, for example one day;
   - scope: `publish-new`;
   - crate restriction: unrestricted, because the crate does not exist yet.
3. Publish locally:

   ```bash
   cargo login <token>
   cargo publish
   cargo logout
   ```

4. Revoke the token.
5. Run the `First release only: create tag` workflow from `main`:
   - `version`: the exact `Cargo.toml` package version;
   - `confirm`: `tag packed_spatial_index`.
6. Configure Trusted Publishing before the next release.

If tag creation is interrupted, rerun the same workflow only after checking that
the version is already published on crates.io and the `v<version>` tag is
missing.

## One-Time Setup

This repository is already configured. Use this section as the checklist for
new projects or when recreating repository settings.

### GitHub Actions Permissions

In the GitHub repository settings, open:

`Settings` -> `Actions` -> `General` -> `Workflow permissions`

Enable:

- `Read and write permissions`.

This is required so `Release: publish crate` can push the annotated tag and
create the GitHub Release with `GITHUB_TOKEN`.

### Release Environment Approval

The `Release: publish crate` workflow uses `environment: release` for the real
publish job. Configure that environment with required reviewers so a real
publish cannot proceed without a GitHub approval after preflight succeeds.

For this repository the expected environment settings are:

- Environment name: `release`;
- Required reviewers: `Filyus`;
- Prevent self-review: disabled, so the repository owner can approve their own
  release deployment;
- Wait timer: `0`;
- Deployment branch policy: none. The workflow itself checks that it was started
  from `main`, that `main` `HEAD` is a release commit, and that CI passed for it.

To configure another personal repository through `gh`:

```powershell
$owner = "Filyus"
$repo = "packed_spatial_index"
$reviewer = "Filyus"
$reviewerId = gh api "users/$reviewer" --jq ".id"

@"
{
  "wait_timer": 0,
  "prevent_self_review": false,
  "reviewers": [
    { "type": "User", "id": $reviewerId }
  ],
  "deployment_branch_policy": null,
  "can_admins_bypass": true
}
"@ | gh api -X PUT "repos/$owner/$repo/environments/release" --input -
```

Then verify:

```powershell
gh api "repos/$owner/$repo/environments/release" `
  --jq ".protection_rules"
```

For organization repositories, use a team reviewer instead of a user reviewer
when that better matches ownership:

```json
{ "type": "Team", "id": 123456 }
```

### Trusted Publishing

Trusted Publishing is the expected publishing path for normal updates after the
crate exists on crates.io. The crates.io Trusted Publisher entry should be:

- Publisher: `GitHub`;
- Repository owner: `Filyus`;
- Repository name: `packed_spatial_index`;
- Workflow filename: `publish.yml`;
- Environment name: `release`.

The crates.io form should show that the workflow file exists at
`.github/workflows/publish.yml`.
