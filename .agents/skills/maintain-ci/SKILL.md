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
- Keep Codecov component paths aligned with new crates, packages, and generated
  outputs. Dynamic plugin SDK/protocol paths belong in the plugin component.
- Keep pure-Python plugin SDK packaging as a single wheel artifact instead of
  duplicating it across every platform matrix entry.

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

## Compile-First Language CI

- In the centralized language workflow, compile enabled native test artifacts
  before the independent test steps. Pass `--set skip_build true` only after the
  matching build recipe and `just build-test-plugin-fixtures` have succeeded.
- Use `just build-rust-tests <scope...>` before execution and
  `just run-rust-tests <run-scope> [graph-scope...]` afterward. The supported
  scopes are `rust`, `python`, `node`, `go`, and `all`. Multiple graph scopes are
  resolved into one Cargo selection; pass the same list to each run and select
  only the owning tests with `run-scope`. CI exports the enabled list through
  `NEMO_RELAY_RUST_TEST_SCOPES` so language recipes reuse the compiled graph.
- Use `just build-language-test-artifacts <scope...>` to compile normal-profile
  Node, Go FFI, and Python plugin CLI artifacts in one Cargo graph. The
  supported scopes are `node`, `openclaw`, `go`, and `python-plugin`; the recipe
  then lets NAPI relink only its binding crate to generate the Node loader and
  declarations, and preserves the isolated Windows Go FFI target required
  during coverage runs.
- `build-python-plugin-e2e` prepares the CLI and worker integration target;
  `build-openclaw` prepares the shared Node native module and OpenClaw workspace.
- Use `just build-language-package-artifacts node rust` on compatible native
  package runners so the Node module and CLI share one release Cargo graph.
  Keep Linux GNU and musl packaging separate where their container, linker, or
  target ABI contracts differ.

## Review Checklist

- [ ] Each job has the minimum permissions it needs
- [ ] Reusable workflow callers grant only the scopes their callees require
- [ ] Every external action is pinned to a full SHA
- [ ] Cache settings are tied to lockfiles, manifests, or explicit tool versions
- [ ] Secrets are only passed to the jobs that consume them
- [ ] Codecov upload counts match `codecov.yml` after adding or removing upload
      jobs
- [ ] Package artifacts include any first-class SDK packages introduced by the
      change
- [ ] Concurrency, branch filters, and publish guards still reflect release intent
- [ ] Artifact upload, download, and Pages deploy steps have matching permissions
- [ ] Tag-triggered release workflows fail early when a tag violates repo policy

## Validation

Start with the narrowest useful checks:

```bash
ruby -e 'require "yaml"; Dir[".github/workflows/*.{yml,yaml}"].each { |f| YAML.load_file(f) }; puts "yaml-ok"'
uv run pre-commit run --files .github/workflows/ci.yaml .github/workflows/ci_languages.yml
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
- `.github/workflows/ci_languages.yml`
- `RELEASING.md`
- `.pre-commit-config.yaml`
- `maintain-packaging`
- `validate-change`
- `maintain-dynamic-plugins`
