# Releasing

Releases are intentionally two-step:

1. `Release-plz PR` prepares a draft release PR with the version bump and
   `CHANGELOG.md` update.
2. `Publish to crates.io` publishes the reviewed version and creates the
   annotated `v<version>` tag.

`release-plz` is configured in PR-only mode. It does not publish crates, push
tags, or create GitHub Releases.

## One-Time Setup

### GitHub Actions Permissions

In the GitHub repository settings, open:

`Settings` -> `Actions` -> `General` -> `Workflow permissions`

Enable:

- `Read and write permissions`;
- `Allow GitHub Actions to create and approve pull requests`.

This is required so `Release-plz PR` can create and update draft release PRs
with `GITHUB_TOKEN`.

### Trusted Publishing

Trusted Publishing is already the expected publishing path for normal updates.
The crates.io Trusted Publisher entry should be:

- Publisher: `GitHub`;
- Repository owner: `Filyus`;
- Repository name: `packed_spatial_index`;
- Workflow filename: `publish.yml`;
- Environment name: `release`.

The crates.io form should show that the workflow file exists at
`.github/workflows/publish.yml`. The `release` GitHub environment should exist
in repository settings; require reviewers there when the plan supports it.

## Normal Release

1. Push normal changes to `main` and wait for CI to pass.
2. Run the `Release-plz PR` workflow from `main` with `dry_run: true`.
3. Read the workflow log. It should show the version/changelog changes it would
   prepare as a local `git diff`, without opening a PR.
4. If the dry run looks right, run `Release-plz PR` again with
   `dry_run: false`.
5. Review the draft release PR:
   - check the version bump;
   - edit `CHANGELOG.md` if the generated notes need a clearer user-facing
     summary;
   - keep the PR as the only version/changelog change for that release.
6. Merge the release PR.
7. Wait for CI on `main` to pass.
8. Run the `Publish to crates.io` workflow from `main`.
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

## First Release Fallback

Trusted Publishing cannot publish a crate that does not exist yet. For a new
crate, do the first publish locally with a short-lived token:

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
5. Run the `Create release tag` workflow from `main`:
   - `version`: the exact `Cargo.toml` package version;
   - `confirm`: `tag packed_spatial_index`.
6. Configure Trusted Publishing before the next release.

## Tag-Only Fallback

Use `Create release tag` only when the version is already published on
crates.io and the `v<version>` tag is missing.

The workflow checks:

- it is running from `main`;
- `Cargo.toml` has the requested version;
- that version exists on crates.io;
- the remote tag does not exist yet;
- `confirm` is exactly `tag packed_spatial_index`.
