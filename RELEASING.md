# Releasing

Normal releases are optimized for a solo maintainer working with Codex:

1. Codex verifies that the working tree is clean and current with `origin/main`.
2. Codex prepares the version bump and release notes locally.
3. The maintainer reviews the `CHANGELOG.md` diff.
4. Codex pushes one release commit to `main`.
5. CI validates that exact commit.
6. `Release: publish crate` starts automatically after successful CI, runs
   preflight checks, then waits for the `release` environment approval.
7. The maintainer approves the environment deployment. Only then does the
   workflow publish to crates.io, create the annotated tag, and create the
   GitHub Release.

`release-plz` is used only for local release preparation. It does not publish
crates, push tags, or create GitHub Releases.

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

If `git status --short` prints anything, classify the changes before running
`release-plz update`:

- If the current conversation makes it clear what normal feature/fix/docs commit
  is missing, create that commit first, push it to `main`, and wait for CI.
- If the changes are clearly unrelated local work, ask whether to commit, stash,
  discard, or postpone them.
- If the changes are ambiguous, stop and ask. Do not guess and do not fold them
  into the release commit.

The release commit should be easy to review and should not accidentally absorb
in-progress work.

Generate the version bump and changelog draft:

```powershell
release-plz update --config release-plz.toml
```

Then Codex must polish `CHANGELOG.md` before asking for approval:

- keep the generated version and compare link;
- rewrite terse commit messages into clear release notes;
- group related changes by user-facing topic;
- remove internal noise that is not useful to crate users;
- keep feature/code changes out of the release commit.

Do not run release-only checks manually. The automatic publish preflight runs
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

### 3. CI and Automatic Preflight

The push to `main` starts `CI: Rust checks`.

After successful CI, `Release: publish crate` starts automatically through a
`workflow_run` trigger. It checks out the exact CI commit SHA, not the latest
moving `main`.

For an ordinary non-release push, the workflow exits early:

```text
Not a release commit; nothing to publish.
```

For a release commit, preflight checks:

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

### Manual Rerun

`Release: publish crate` still has `workflow_dispatch` as a recovery path. It
has no inputs.

Use it only when the automatic run did not complete for infrastructure reasons.
It runs against the current `main` commit and still requires:

- a valid release commit subject;
- successful `CI: Rust checks` for that exact commit;
- all preflight checks;
- the final `release` environment approval.

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
- Deployment branch policy: none. The workflow itself checks the release commit
  through the CI `workflow_run` event and the commit subject.

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
