---
name: prepare-code-freeze
description: Prepare a NeMo Relay code freeze by creating the release branch, updating nightly alpha branch config, bumping main to the next version, and opening the required PR
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Prepare Code Freeze

Use this skill when the user asks to start, prepare, or automate a NeMo Relay
code freeze.

## Companion Guidance

Use `update-project-version` for version bump semantics and `prepare-pr` before
opening the PR.

## Workflow

This workflow assumes `upstream` is the NVIDIA repository remote
(`NVIDIA/NeMo-Relay`). The `origin` remote can be a maintainer's personal fork.

1. Confirm or infer the target release version from `upstream/main:Cargo.toml`.
   Derive the release branch as `release/<major>.<minor>`.
2. Prompt for `<next-version>` if the user did not provide it. This is the
   version that `main` moves to after the release branch is cut.
3. Fetch the latest `main` and create the release branch from `upstream/main`:

   ```bash
   git fetch upstream main
   git branch release/<major>.<minor> upstream/main
   git push upstream release/<major>.<minor>
   ```

   If the remote release branch already exists, verify it points where expected
   before continuing.
4. Create a PR branch from latest `upstream/main`, for example
   `docs/code-freeze-<major>.<minor>`.
5. Update `.github/nightly-alpha-branches.yaml` to include the new release
   branch.
6. Run `just set-version <next-version>` to bump all release-versioned package
   surfaces on `main`.
7. Validate with targeted checks:

   ```bash
   ruby -e 'require "yaml"; YAML.load_file(".github/nightly-alpha-branches.yaml"); YAML.load_file(".github/workflows/nightly-alpha-tag.yaml")'
   just set-version <next-version>
   git diff --check
   ```

8. Open a PR targeting `main` using `.github/pull_request_template.md`. The PR
   must mention:
   - the new release branch
   - the nightly alpha branch config update
   - the `just set-version <next-version>` bump
   - that release-bound PRs now target the new `release/*` branch

## Guardrails

- Do not create release tags. Code freeze only creates the branch and the main
  PR.
- Do not target the code-freeze PR at the release branch. It targets `main`.
- Do not leave uncommitted user changes mixed into the code-freeze PR branch.
