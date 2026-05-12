<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing NeMo Flow

This document is the maintainer playbook for cutting NeMo Flow releases. It
describes the release contract, the version files that must be updated, the tag
format that CI accepts, the package surfaces that are published, and the checks
to run before and after a tag push.

## Source Of Truth

This section defines where release history and release-facing details are maintained.

- There is no `CHANGELOG.md` in this repository.
- The documentation site has a release-notes landing page for current
  documentation-visible release status.
- The source of truth for complete release history and tag-specific release
  notes is always GitHub Releases for this repository.

Do not copy full GitHub Release notes into `CHANGELOG.md` or the docs site.
The docs release-notes page can summarize support status and point users to
GitHub Releases.

## Published Surfaces

The release pipeline publishes these package surfaces from a tag push:

| Ecosystem | Published Surface |
|---|---|
| crates.io | `nemo-flow`, `nemo-flow-adaptive`, `nemo-flow-ffi`, `nemo-flow-cli` |
| PyPI | `nemo-flow` |
| npm | `nemo-flow-node`, `nemo-flow-wasm` |
| GitHub Pages | The documentation site, including the versioned docs build |

Go remains source-first. There is no separate Go package-manager publication
step in the repository release workflow.

## Version Model

NeMo Flow versions are anchored on the workspace SemVer in the repository root
`Cargo.toml`.

- The root `Cargo.toml` `workspace.package.version` is the canonical release
  version for the Rust workspace.
- The root `Cargo.toml` `workspace.dependencies` entries for
  `nemo-flow`, `nemo-flow-adaptive`, `nemo-flow-ffi`, and `nemo-flow-cli` must
  stay aligned with that same version.
- `crates/node/package.json` carries the base npm version for the Node.js
  package. The repository-root `package-lock.json` carries the npm workspace
  lock entries and must be updated with it.
- The Python package version is derived at packaging time. `pyproject.toml`
  stays `dynamic = ["version"]` in the repository, and the packaging recipe
  writes a concrete version into `pyproject.toml` and `crates/python/Cargo.toml`
  in the ephemeral packaging workspace.
- The published WebAssembly npm package version is derived from the Rust workspace
  version during `wasm-pack` packaging.

For non-tag CI builds, packaging recipes append a commit-derived suffix:

- Node.js and WebAssembly use `-<short_sha>`.
- Python uses `+<short_sha>` and converts prerelease labels into PEP 440 form.
  For example, `0.2.0-rc.1` becomes `0.2.0rc1` when packaged for PyPI.

## Release Tags

Release tags must use raw Rust-compatible SemVer without a leading `v`.

- Use `0.1.0` for stable releases.
- Use `0.1.0-rc.1` for prereleases.
- Do not use tags such as `v0.1.0` or `v0.1.0-rc.1`.

CI rejects tags that do not match the required format.

The tag text must match the version that the packaging jobs publish.

## Before You Cut A Release

Before you create a release tag, confirm the following:

1. The intended release commit is already on `main` or on the release branch
   you intend to tag.
2. The release commit contains the final version bump, docs updates, and any
   public API changes that belong in the release.
3. The working tree you use for local validation is clean or disposable.
4. Registry credentials and repository settings are in place:
   - GitHub Actions `id-token: write` access for the top-level crates.io publish job
   - crates.io trusted publishers for `nemo-flow`, `nemo-flow-adaptive`,
     `nemo-flow-ffi`, and `nemo-flow-cli` are configured for the top-level
     [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) workflow
   - GitHub Actions `id-token: write` access is available for the top-level npm publish job
   - GitHub Actions `id-token: write` access for the top-level PyPI publish job
5. The GitHub Release entry is ready to become the only canonical release-notes
   surface.

## Prepare The Release Commit

Update the versioned source files in the release PR or release-prep commit:

1. Update the root [`Cargo.toml`](Cargo.toml) workspace version.
2. Update the root [`Cargo.toml`](Cargo.toml) `workspace.dependencies` versions
   for `nemo-flow`, `nemo-flow-adaptive`, `nemo-flow-ffi`, and
   `nemo-flow-cli`.
3. Update [`crates/node/package.json`](crates/node/package.json) and the
   `crates/node` entry in the root [`package-lock.json`](package-lock.json) to
   the same release version.
4. Review docs and snippets that mention explicit versions, including:
   - [`README.md`](README.md)
   - [`CONTRIBUTING.md`](CONTRIBUTING.md)
   - [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
   - Any binding README or example that pins a release number

Do not commit a static Python package version into `pyproject.toml` just to cut
the release. The packaging workflow stamps that file during the build.

## Local Validation

Run the checks that match the surfaces affected by the release. For a normal
repository release, the safest baseline is:

```bash
uv run pre-commit run --all-files
just test-rust
just test-python
just test-go
just test-node
just test-wasm
./scripts/build-docs.sh linkcheck
./scripts/build-docs.sh pages
```

If you want to validate the packaging recipes before pushing a tag, run:

```bash
just --set output_dir "$PWD/target/release-artifacts" --set ref_name 0.1.0 package-node
just --set output_dir "$PWD/target/release-artifacts" --set ref_name 0.1.0 package-python
just --set output_dir "$PWD/target/release-artifacts" --set ref_name 0.1.0 package-wasm
```

Be aware that the local packaging recipes intentionally rewrite version fields
in place while they build artifacts. In a disposable CI workspace that is fine.
In a local checkout, restore those temporary manifest edits before continuing if
you are not committing them.

## Cut The Tag

After the release commit is merged and validated, create and push the raw
SemVer tag:

```bash
git fetch origin
git checkout main
git pull --ff-only
git tag 0.1.0
git push origin 0.1.0
```

Use the prerelease form when needed:

```bash
git tag 0.1.0-rc.1
git push origin 0.1.0-rc.1
```

## What CI Does On A Tag Push

Pushing a valid tag triggers [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml).
That workflow then calls [`.github/workflows/ci_pipe.yml`](.github/workflows/ci_pipe.yml)
for the shared release documentation and packaging stages.

The release pipeline then:

1. Validates the tag format in the `prepare` job.
2. Skips repo checks and the Rust, Python, Go, Node.js, and WebAssembly test jobs.
   Run those checks before creating and pushing the release tag.
3. Builds and uploads the versioned GitHub Pages documentation artifact.
4. Builds publishable package artifacts with the exact tag version:
   - `package-node` packs the npm Node.js package.
   - `package-python` builds platform wheels.
   - `package-wasm` packs the npm WebAssembly package.
5. Publishes packages from the top-level workflow after the reusable packaging
   jobs complete:
   - `publish-rust` stamps Cargo workspace versions from the release tag, then
     runs `cargo publish --package` for `nemo-flow`, `nemo-flow-adaptive`,
     `nemo-flow-ffi`, and `nemo-flow-cli` through trusted publishing from
     the top-level workflow
   - `publish-python` uploads the wheel artifacts to PyPI with trusted
     publishing from the top-level workflow
   - `publish-npm` publishes the Node.js and WebAssembly npm packages through npm
     trusted publishing from the top-level workflow
     - Stable tags publish to the npm `latest` dist-tag
     - Prerelease tags such as `0.1.0-rc.1` publish to the npm `next`
       dist-tag so they do not become the default upgrade target
6. Deploys the GitHub Pages docs site.

The workflow boundary is split intentionally:

- [`.github/workflows/ci_pipe.yml`](.github/workflows/ci_pipe.yml) produces the
  publishable package artifacts, runs the docs build, and uploads the GitHub
  Pages artifact.
- [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) owns all crates.io,
  PyPI, and npm publication decisions and credentials.
- [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) performs only the
  `actions/deploy-pages` step for documentation publication.
- This layout also satisfies the official `pypa/gh-action-pypi-publish`
  guidance that trusted publishing should not run inside reusable workflows.

npm trusted publishing has its own registry-side constraints:

- Each npm package can only have one trusted publisher configured at a time.
- Because this repository publishes both `nemo-flow-node` and
  `nemo-flow-wasm`, configure trusted publishers for both packages before
  pushing a release tag.
- npm trusted publishing currently supports GitHub-hosted runners, not
  self-hosted runners.

Stable docs versioning is narrower than package publication:

- Stable released docs are selected from tags that match `X.Y.Z`.
- Prerelease tags such as `0.1.0-rc.1` still run the docs workflow, but they
  are not treated as stable released versions by the Sphinx multiversion
  configuration.
- Those prerelease docs can still appear in the published version switcher as
  prerelease snapshots when they are among the selected recent release tags.

## Publish The GitHub Release Entry

After the tag pipeline succeeds, publish or finalize the GitHub Release entry
for that tag.

- Keep complete release notes in GitHub Releases.
- Do not copy those notes into `CHANGELOG.md` or duplicate the full release
  history in the docs site.
- If you use GitHub-generated notes, review them before publishing. The
  category mapping lives in [`.github/release.yml`](.github/release.yml).

## Post-Release Checks

After the release is live, verify:

1. The expected crates are visible on crates.io.
2. The `nemo-flow` wheel is visible on PyPI.
3. The `nemo-flow-node` and `nemo-flow-wasm` packages are visible on npm.
4. The GitHub Pages deployment completed successfully.
5. The GitHub Release page is complete and accurate.

## If Something Fails

Use the failure point to decide how to recover:

- If the tag pipeline fails before any registry publish step succeeds, fix the
  issue and rerun or replace the tag as appropriate.
- If any package has already been published to a public registry, do not reuse
  the same version number. Prepare a follow-up patch or prerelease instead.
- If only the GitHub Release text is wrong, edit the GitHub Release entry
  directly. Do not create a duplicate notes surface in the repository.
