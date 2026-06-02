# Releasing

Releases are intentionally two-step:

1. `Release: prepare PR` prepares a draft release PR with the version bump and
   `CHANGELOG.md` update.
2. `Release: publish crate` publishes the reviewed version and creates the
   annotated `v<version>` tag.

`release-plz` is configured in PR-only mode. It does not publish crates, push
tags, or create GitHub Releases.

## Normal Release

Use this path after the crate already exists on crates.io and Trusted Publishing
is configured.

1. Push normal changes to `main` and wait for CI to pass.
2. Run the `Release: prepare PR` workflow from `main` with `dry_run: true`.
3. Read the workflow log. It should show the version/changelog changes it would
   prepare as a local `git diff`, without opening a PR.
4. If the dry run looks right, run `Release: prepare PR` again with
   `dry_run: false`.
5. Review the draft release PR:
   - check the version bump;
   - edit `CHANGELOG.md` if the generated notes need a clearer user-facing
     summary;
   - keep the PR as the only version/changelog change for that release.
6. Merge the release PR.
7. Wait for CI on `main` to pass.
8. Run the `Release: publish crate` workflow from `main`.
9. Set `version` to the exact `Cargo.toml` package version, for example
   `0.3.1`.
10. For a dry run, keep `publish` as `false`; `confirm` can stay empty.
11. For the real publish, set:
    - `publish`: `true`;
    - `confirm`: `publish packed_spatial_index`.
12. Approve the `release` environment when GitHub asks for confirmation.

If the requested version does not match `Cargo.toml`, the publish workflow fails
before publishing. The confirmation phrase deliberately does not include the
version; the workflow checks the version separately.

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
5. Create and push the annotated release tag:

   ```bash
   version="$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["packages"][0]["version"])')"
   git tag -a "v$version" -m "packed_spatial_index $version"
   git push origin "v$version"
   ```

6. Configure Trusted Publishing before the next release.

If the local publish succeeds but tag creation is interrupted, create the tag
manually only after checking that the version is already published on crates.io
and the `v<version>` tag is missing.

## One-Time Setup

This repository is already configured. Use this section as the checklist for
new projects or when recreating repository settings.

### GitHub Actions Permissions

In the GitHub repository settings, open:

`Settings` -> `Actions` -> `General` -> `Workflow permissions`

Enable:

- `Read and write permissions`;
- `Allow GitHub Actions to create and approve pull requests`.

This is required so `Release: prepare PR` can create and update draft release PRs
with `GITHUB_TOKEN`.

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
