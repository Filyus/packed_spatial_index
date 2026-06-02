# Releasing

Releases are manual and controlled through GitHub Actions. Normal CI already
runs on every `main` commit. The publish workflow creates the annotated
`v<version>` tag after a successful publish; the tag workflow exists for the
first local-token release and other tag-only recovery cases.

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

- provider: `GitHub Actions`;
- repository: `Filyus/packed_spatial_index`;
- workflow file: `publish.yml`;
- environment: `release`.

In GitHub, create the `release` environment. If the repository plan supports
required reviewers for environments, require approval there too.

## Updates

1. Update `Cargo.toml` version. If the minor version changes, update the README
   install snippet too.
2. Push the release commit to `main` and wait for CI to pass.
3. Run the `Publish to crates.io` workflow from `main`.
4. Set `version` to the exact `Cargo.toml` package version, for example
   `0.3.1`.
5. For a dry run, keep `publish` as `false`; `confirm` can stay empty.
6. For a real publish, set:
   - `publish`: `true`;
   - `confirm`: `publish packed_spatial_index`.
7. Approve the `release` environment when GitHub asks for confirmation.

If `version` is mistyped, the workflow fails before publishing. The confirmation
phrase deliberately does not include the version; the version is checked only
against `Cargo.toml`.

## Tag-Only Workflow

Use `Create release tag` only after the version is already published on
crates.io and the `v<version>` tag is missing. It checks `main`, `Cargo.toml`,
crates.io, the remote tag list, and the `tag packed_spatial_index` confirmation
phrase before pushing the tag.
