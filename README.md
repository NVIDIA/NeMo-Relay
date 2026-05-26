<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Docs Website Branch

This branch is the Fern publishing branch for NeMo Relay. It contains
CI-managed generated docs content plus a small set of branch-local maintenance
files.

## Edit Policy

Do not make manual documentation content changes on this branch. Documentation
authoring happens on `main` or a pull-request branch based on `main`.

The source of truth is:

- `docs/` on the source branch for Markdown and MDX documentation content.
- `docs/index.yml` on the source branch for the navigation tree.
- `fern/` on the source branch for Fern configuration, assets, custom
  components, styling, and docs maintainer guidance.
- `scripts/docs/sync_fern_docs_branch.py` on the source branch for generating
  the generated Fern content in this branch layout.

The branch-local files in this branch can be updated directly here:

- `.gitignore`
- `README.md`
- `.github/workflows/publish-fern-docs.yml`

Generated Fern content can be overwritten by the next docs sync.

## Branch Contents

This branch intentionally contains only the Fern publish surface and the
branch-local maintenance files:

- `.github/workflows/publish-fern-docs.yml`: branch-local manual or direct-push
  publish workflow.
- `.gitignore`: keeps source-branch and local tooling files from appearing as
  untracked noise when this branch is checked out.
- `README.md`: branch-local maintenance guidance.
- `fern/docs.yml`: site-level Fern configuration and accumulated version list.
- `fern/fern.config.json`: Fern organization and CLI version pin.
- `fern/pages-dev/`: generated dev documentation content.
- `fern/versions/dev.yml`: generated dev navigation rewritten for this branch.
- `fern/pages-vX.Y.Z/`: generated version snapshots.
- `fern/versions/vX.Y.Z.yml`: generated navigation for version snapshots.
- `fern/assets/`, `fern/components/`, and `fern/main.css`: site assets,
  components, and styling used by Fern.

Generator support directories such as `_generated/` and `_source/` are excluded
from this branch. Generated API reference pages that Fern serves are included
under the published page tree.

## How Sync Works

The source-branch workflow `.github/workflows/fern-docs.yml` checks out both the
source branch and this branch, then runs:

```bash
python scripts/docs/sync_fern_docs_branch.py sync-dev \
  --source-root /path/to/source-checkout \
  --target-root /path/to/docs-website-checkout
```

That command:

1. Copies source `docs/` content into `fern/pages-dev/`.
2. Rewrites `docs/index.yml` into `fern/versions/dev.yml`.
3. Copies the Fern assets, components, config, CSS, and docs guide from source
   `fern/`.
4. Preserves the existing `products[0]` version list from this branch's
   `fern/docs.yml`.

It does not regenerate this root `README.md` or the branch-local publish
workflow. Update those files directly on `docs-website` when needed.

## Publishing

Publishing uses the GitHub secret `DOCS_FERN_TOKEN`. Workflows expose it to the
Fern CLI as `FERN_TOKEN`.

Normal publishing is handled by the source-branch `.github/workflows/fern-docs.yml`
workflow after it commits generated changes to this branch.

The branch-local `.github/workflows/publish-fern-docs.yml` workflow is present
for manual dispatch or direct pushes to this branch. It installs the Fern CLI
version pinned in `fern/fern.config.json` and runs:

```bash
fern generate --docs
```

## Versioning

The source-branch workflow creates docs snapshots from accepted raw SemVer tags:

- Stable tags such as `0.2.0` create or replace `v0.2.0`, set
  `availability: stable`, and update the default `Latest` version.
- Beta and release-candidate tags such as `0.2.0-beta.1` and `0.2.0-rc.2`
  create or replace the same base version, `v0.2.0`, set
  `availability: beta`, and do not update `Latest`.
- Alpha tags such as `0.2.0-alpha.1` are not published.
- Tags with a leading `v` are not accepted by the NeMo Relay release policy.

Prerelease indicators are stripped from public docs paths. For example,
`0.2.0-beta.1`, `0.2.0-rc.2`, and `0.2.0` all target:

```text
fern/pages-v0.2.0/
fern/versions/v0.2.0.yml
```

This keeps the version selector from accumulating beta and release-candidate
entries for the same base release.

## Recovery

If this branch becomes stale or malformed, regenerate it from a validated source
checkout rather than editing files by hand:

```bash
just docs
python scripts/docs/sync_fern_docs_branch.py sync-dev \
  --source-root . \
  --target-root /path/to/docs-website-checkout
```

Then review the generated diff on `docs-website`, preserve any branch-local
README or workflow edits, commit the result, and publish through the workflow.
