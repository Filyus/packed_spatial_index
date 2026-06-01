# Releasing

Releases are intentionally manual and use crates.io Trusted Publishing for
updates after the first version.

## First Release

Trusted Publishing can only be configured after the crate exists on crates.io.
For the first release:

1. Create a short-lived crates.io token with only `publish-new`.
2. Run the GitHub Actions `Publish to crates.io` workflow with:
   - `version`: the current `Cargo.toml` version;
   - `publish`: `false`.
3. If the workflow passes, publish locally:

   ```bash
   cargo login
   cargo publish
   cargo logout
   ```

4. Revoke the crates.io token.

## Trusted Publishing Setup

After the first release, configure the crate on crates.io:

- Provider: `GitHub Actions`
- Repository: `Filyus/packed_spatial_index`
- Workflow file: `publish.yml`
- Environment: `release`

In GitHub, create the `release` environment and require a reviewer before
deployment. This keeps the publish job manual even after the workflow has
already been started. If the repository plan does not support required
reviewers for environments, keep the environment anyway for the crates.io
Trusted Publishing claim; the workflow still requires an explicit `publish:
true` input and confirmation phrase.

## Updating

1. Update `Cargo.toml` version and release notes.
2. Merge the release commit to `main`.
3. Run the `Publish to crates.io` workflow from `main`.
4. Set `version` to the exact `Cargo.toml` version.
5. Keep `publish` as `false` for a dry run, or set it to `true` to publish.
6. For a real publish, set `confirm` to `publish packed_spatial_index <version>`.
7. Approve the `release` environment when GitHub asks for confirmation.

The workflow validates that it is running from `main`, checks the requested
version against `Cargo.toml`, runs formatting/tests/clippy/docs, performs
`cargo publish --dry-run`, rejects already-published versions, checks the
confirmation phrase, and only then requests a short-lived crates.io token.
