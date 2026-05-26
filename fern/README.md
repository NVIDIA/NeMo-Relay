<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Fern Docs

NeMo Relay uses Fern to publish the documentation site at
`nvidia-nemo-relay.docs.buildwithfern.com/nemo-relay`.

## Branch Model

Documentation is authored on `main`:

- `docs/` contains the Markdown and MDX page source.
- `docs/index.yml` contains the source navigation tree.
- `fern/` contains local Fern configuration, shared assets, and custom
  components.

The `docs-website` branch is mostly CI-managed. Do not hand-edit generated Fern
content on that branch. The root `.gitignore`, `README.md`, and
`.github/workflows/publish-fern-docs.yml` are branch-local files that can be
updated directly on `docs-website` when the branch maintenance guidance,
ignored-file rules, or manual fallback workflow changes.

The `.github/workflows/fern-docs.yml` workflow syncs the authored docs into this
published layout:

- `.github/workflows/publish-fern-docs.yml` contains the branch-local publish
  workflow for manual dispatch or direct branch pushes.
- `.gitignore` keeps source-branch and local tooling files from appearing as
  untracked noise when `docs-website` is checked out.
- `README.md` contains branch-local maintenance guidance.
- `fern/pages-dev/` contains the generated dev documentation pages.
- `fern/versions/dev.yml` contains the dev navigation rewritten for the
  published branch layout.
- `fern/pages-vX.Y.Z/` and `fern/versions/vX.Y.Z.yml` contain snapshots created
  from release tags.
- `fern/docs.yml` preserves the version list accumulated on `docs-website`.

The branch intentionally contains only the Fern publish surface plus the
branch-local `.gitignore`, README, and workflow. Generator support directories such as
`_generated/` and `_source/` are excluded from the published branch. The sync is
implemented by `scripts/docs/sync_fern_docs_branch.py` so the generated branch
layout can be validated outside GitHub Actions.

## Publishing

The Fern publish workflow uses the GitHub secret `DOCS_FERN_TOKEN` and passes it
to the Fern CLI as `FERN_TOKEN`.

- Pushes to `pull-request/**` that affect docs generate a Fern preview and add
  the preview URL to the pull request.
- Pushes to `main` that affect docs regenerate API reference pages, sync
  `docs-website`, and publish the dev docs.
- Raw SemVer tags such as `0.1.0`, `0.1.0-beta.1`, and `0.1.0-rc.1` create or
  replace a public docs version displayed and slugged with a leading `v`.
  Prerelease indicators are stripped from the public version path, so
  `0.1.0-beta.1`, `0.1.0-rc.1`, and `0.1.0` all target `v0.1.0`.

Stable tags use `availability: stable` and update the default `Latest` version.
Beta and release-candidate tags use `availability: beta`, replace the same base
version snapshot, and do not update the default `Latest` version. Alpha tags and
tags with a leading `v` are not published.

## Local Validation

Run the normal docs check from the repository root:

```bash
just docs
```

To inspect the generated docs-website layout locally, run the sync helper
against a temporary target checkout:

```bash
python scripts/docs/sync_fern_docs_branch.py sync-dev \
  --source-root . \
  --target-root /path/to/docs-website-checkout
```
