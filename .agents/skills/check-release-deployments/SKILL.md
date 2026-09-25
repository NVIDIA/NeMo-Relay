---
name: check-release-deployments
description: Verify whether a tagged NeMo Relay release reached GitHub Actions, tagged Go module source, crates.io, PyPI, and npm. Use for publication status of a specific tag; not for publishing packages or creating tags.
license: Apache-2.0
---

# Check Release Deployments

Require the release tag as input. Do not infer it from the current checkout,
Git history, or package metadata.

The checker requires Bash, `curl`, `gh`, `jq`, `just`, and `uv`. Authenticate
`gh` for `NVIDIA/NeMo-Relay` and verify it with `gh auth status`. The checker
does not run on Windows. From the repository root, run:

```bash
.agents/skills/check-release-deployments/scripts/check_release_deployments.sh <tag>
```

It validates Relay's raw-SemVer tag format, converts the version to PEP 440 for
PyPI, and checks the published package sets documented in `RELEASING.md`. It
also verifies that the source-first Go module is present at the tag and reports
the release-related GitHub Actions jobs. Registry checks run only after their
publication jobs are verified successful. If the tag or release jobs cannot be
verified, package checks are skipped.

- `☑` means the exact package version is deployed.
- `○` means that exact package version is not deployed.
- `000` means a lookup or response could not be verified.
- Other numeric values are HTTP statuses returned by the registry.

The Go row checks the tagged `go.mod` source because Relay does not publish a
separate Go package-manager artifact.

This is a read-only workflow. Do not retry an absent or failed deployment by
publishing packages unless the user separately asks to publish.
