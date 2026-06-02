# Releasing

Releases are manual and controlled through GitHub Actions. Normal CI already
runs on every `main` commit. `release-plz` is configured only to prepare draft
release PRs; publishing, tag creation, and GitHub Releases remain handled by
the explicit manual workflows below.

## First Release

Trusted Publishing is configured after the crate exists on crates.io, so the
first publish is the only local-token exception.

1. Push the release commit to `main` and wait for CI to pass.
2. Create a short-lived crates.io token:
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
6. Configure Trusted Publishing for future releases.

## Trusted Publishing Setup

After the first version exists on crates.io, configure Trusted Publishing on
the crate page:

- Publisher: `GitHub`;
- Repository owner: `Filyus`;
- Repository name: `packed_spatial_index`;
- Workflow filename: `publish.yml`;
- Environment name: `release`.

The crates.io form should show that the workflow file was found at
`.github/workflows/publish.yml`. The environment field is optional on crates.io,
but this repository's publish workflow uses `environment: release`, so configure
that exact GitHub Actions environment in the repository settings. If the
repository plan supports required reviewers for environments, require approval
there too.

## Updates

Before using `Release-plz PR`, GitHub repository settings must allow GitHub
Actions to create pull requests with `GITHUB_TOKEN`. The workflow is manual and
does not publish, push tags, or create GitHub Releases.

1. Run the `Release-plz PR` workflow from `main` with `dry_run: true` and read
   the log.
2. If the dry run looks right, run `Release-plz PR` again with `dry_run: false`.
   It may open or update a draft release PR that changes `Cargo.toml` and
   `CHANGELOG.md`.
3. Review the generated changelog and version bump. Edit the draft PR if the
   release notes need a more user-facing summary.
4. Merge the release PR and wait for CI on `main` to pass.
5. Run the `Publish to crates.io` workflow from `main`.
6. Set `version` to the exact `Cargo.toml` package version, for example
   `0.3.1`.
7. For a dry run, keep `publish` as `false`; `confirm` can stay empty.
8. For a real publish, set:
   - `publish`: `true`;
   - `confirm`: `publish packed_spatial_index`.
9. Approve the `release` environment when GitHub asks for confirmation.

If `version` is mistyped, the workflow fails before publishing. The confirmation
phrase deliberately does not include the version; the version is checked only
against `Cargo.toml`.

## Tag-Only Workflow

Use `Create release tag` only after the version is already published on
crates.io and the `v<version>` tag is missing. It checks `main`, `Cargo.toml`,
crates.io, the remote tag list, and the `tag packed_spatial_index` confirmation
phrase before pushing the tag.
