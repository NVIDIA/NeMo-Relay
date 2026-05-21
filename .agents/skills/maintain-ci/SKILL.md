---
name: maintain-ci
description: Maintain and review NeMo Relay GitHub Actions workflows with explicit per-job permissions, pinned action SHAs, deterministic caching, reusable workflow permission boundaries, and local validation
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain GitHub Actions CI

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when a change touches `.github/workflows/*.yml` or
`.github/workflows/*.yaml`, or when reviewing CI behavior for security,
reliability, or reproducibility.

## Standards

- Put `permissions:` on each job that needs token access.
- Avoid workflow-level permissions unless the repository intentionally centralizes
  them and the inheritance tradeoff is documented.
- Keep third-party actions pinned to full commit SHAs and preserve the readable
  version comment after the SHA.
- Prefer action-native or ecosystem-native caching over generic
  `actions/cache`.
- Use lockfiles or dependency manifests to drive cache invalidation.
- Keep deploy and publish permissions isolated to the jobs that need them.
- Read both caller and callee when a workflow uses `workflow_call`.
- Put release-tag validation in the earliest practical caller job when the
  pipeline has tag-based publish behavior.
- Keep release-tag policy aligned with `RELEASING.md`: raw SemVer tags only,
  no leading `v`.

## Permission Model

- `contents: read` is the default minimum for checkout-based build, test, docs,
  and packaging jobs.
- `pull-requests: read` is required for PR metadata lookup jobs.
- `pages: write` and `id-token: write` should be limited to Pages deployment
  jobs and any caller that invokes them through a reusable workflow.
- For reusable workflows, the caller must grant every permission the called
  jobs require. The callee cannot elevate beyond what the caller provides.

## Caching

- Prefer `astral-sh/setup-uv` cache support with `cache-dependency-glob`
  anchored to `uv.lock`.
- Prefer `Swatinem/rust-cache` with explicit `shared-key` and `workspaces`
  instead of ad hoc target-directory caching.
- Avoid caching generated outputs that can hide stale behavior unless the repo
  already relies on them deliberately.

## Review Checklist

- [ ] Each job has the minimum permissions it needs
- [ ] Reusable workflow callers grant only the scopes their callees require
- [ ] Every external action is pinned to a full SHA
- [ ] Cache settings are tied to lockfiles, manifests, or explicit tool versions
- [ ] Secrets are only passed to the jobs that consume them
- [ ] Concurrency, branch filters, and publish guards still reflect release intent
- [ ] Artifact upload, download, and Pages deploy steps have matching permissions
- [ ] Tag-triggered release workflows fail early when a tag violates repo policy

## Validation

Start with the narrowest useful checks:

```bash
ruby -e 'require "yaml"; Dir[".github/workflows/*.{yml,yaml}"].each { |f| YAML.load_file(f) }; puts "yaml-ok"'
uv run pre-commit run --files .github/workflows/ci.yaml .github/workflows/ci_pipe.yml
```

Use ripgrep to inspect the workflow graph before editing:

```bash
rg -n "uses:|permissions:|workflow_call|secrets:|upload-artifact|download-artifact|upload-pages-artifact|deploy-pages|codecov|cache" .github/workflows
```

If local lint passes but the question is whether GitHub will authorize the run,
inspect GitHub's permission model and the upstream action or reusable workflow
source instead of assuming local success proves remote success.

## Canonical References

- `.github/workflows/ci.yaml`
- `.github/workflows/ci_pipe.yml`
- `RELEASING.md`
- `.pre-commit-config.yaml`
- `maintain-packaging`
- `validate-change`
