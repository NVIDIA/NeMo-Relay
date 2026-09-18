---
name: check-release-deployments
description: Verify whether a tagged NeMo Relay release reached GitHub Actions, crates.io, PyPI, and npm. Use for publication status of a specific tag; not for publishing packages or creating tags.
license: Apache-2.0
---

# Check Release Deployments

Require the release tag as input. Do not infer it from the current checkout,
Git history, or package metadata.

The checker requires Bash, `curl`, `gh`, `jq`, `just`, and `uv`; it does not run
on Windows. From the repository root, run:

```bash
bash .agents/skills/check-release-deployments/scripts/check_release_deployments.sh <tag>
```

It validates Relay's raw-SemVer tag format, converts the version to PEP 440 for
PyPI, and checks the published package sets documented in `RELEASING.md`. It
also reports GitHub Actions runs triggered by the tag. A failed publication
workflow causes only that registry's checks to be skipped.

- `☑` means the exact package version is deployed.
- `○` means that exact package version is not deployed.
- Any other value is the HTTP status returned by the registry.

This is a read-only workflow. Do not retry an absent or failed deployment by
publishing packages unless the user separately asks to publish.
