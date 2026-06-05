# Releasing

Releases are intentionally two-step:

1. `Release: prepare version` prepares a draft release PR with the version bump
   and `CHANGELOG.md` update.
2. `Release: publish crate` publishes the reviewed version and creates the
   annotated `v<version>` tag.

`release-plz` is configured in PR-only mode. It does not publish crates, push
tags, or create GitHub Releases.

## Normal Release

Use this path after the crate already exists on crates.io and Trusted Publishing
is configured.

1. Push normal changes to `main` and wait for CI to pass.
2. Run the `Release: prepare version` workflow from `main` with
   `dry_run: false`.
3. Treat the draft release PR as the maintainer-controlled release branch.
   Check it out locally if the generated notes need cleanup:

   ```powershell
   gh pr checkout <number>
   # edit release notes or release-facing docs
   git commit -m "docs: clarify release notes"
   git push
   git switch main
   ```

4. Review the draft release PR:
   - check the version bump;
   - edit `CHANGELOG.md` if the generated notes need a clearer summary;
   - update small release-facing docs if they must mention the new version;
   - keep code and feature changes out of the release PR.
5. Wait for CI on the release PR to pass.
6. Mark the PR as ready, then merge it.
   Use `Rebase and merge` in GitHub for linear history, or merge locally:

   ```powershell
   git fetch origin
   git switch main
   git merge --ff-only origin/<release-branch>
   git push origin main
   ```

7. Wait for CI on `main` to pass.
8. Run the `Release: publish crate` workflow from `main`.
9. Set `version` to the exact `Cargo.toml` package version, for example
   `0.3.1`.
10. Set:
    - `publish`: `true`;
    - `confirm`: `publish packed_spatial_index`.
11. Approve the `release` environment after preflight succeeds.

If the requested version does not match `Cargo.toml`, the publish workflow fails
before publishing. The confirmation phrase deliberately does not include the
version; the workflow checks the version separately.

Optional previews:

- To preview the generated version/changelog PR without opening it, run
  `Release: prepare version` with `dry_run: true`.
- To run publish preflight without entering the approval step, run
  `Release: publish crate` with `publish: false`.

The publish preflight does not repeat the full Rust test matrix. It verifies
that `CI: Rust checks` already passed for the exact `main` commit, then checks
the requested version, crates.io availability, release tag availability, and
`cargo publish --dry-run`.

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

- `Read and write permissions`;
- `Allow GitHub Actions to create and approve pull requests`.

This is required so `Release: prepare version` can create and update draft
release PRs with `GITHUB_TOKEN`.

### Release Environment Approval

The `Release: publish crate` workflow uses `environment: release` for the real
publish job. Configure that environment with required reviewers so a real
publish cannot proceed without a GitHub approval after preflight succeeds.

For this repository the expected environment settings are:

- Environment name: `release`;
- Required reviewers: `Filyus`;
- Prevent self-review: disabled, so the repository owner can approve their own
  manually triggered publish;
- Wait timer: `0`;
- Deployment branch policy: none. The workflow itself checks that it runs from
  `main`.

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
