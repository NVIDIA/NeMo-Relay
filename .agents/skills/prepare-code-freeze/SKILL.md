---
name: prepare-code-freeze
description: Execute a requested NeMo Relay code freeze by cutting a release branch, updating nightly alpha configuration, bumping main, and preparing the required PR. Do not use for an ordinary version bump or release-note draft.
license: Apache-2.0
---

# Prepare Code Freeze

Use this skill when the user asks to start, prepare, or automate a NeMo Relay
code freeze.

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
   `chore/version-bump-<major>.<minor>`.
5. Update `.github/nightly-alpha-branches.yaml` to include the new release
   branch.
6. Run `just set-version <next-version>` to update all project-owned
   unified-release version surfaces on `main`, including Cargo, Python, Node,
   plugins, lockfiles, and coding-agent manifests.
   Regenerate the dynamic worker-plugin fixture lockfile so its path
   dependencies use the new workspace version:

   ```bash
   cargo generate-lockfile --manifest-path crates/core/tests/fixtures/worker_plugin/Cargo.toml
   ```
7. Search documentation source for references to the old version and update
   current-version install commands, package examples, and configuration
   examples to `<next-version>` where appropriate:

   ```bash
   rg -n '<old-version>' README.md docs fern --glob '!docs/_build/**' || true
   ```

   Review matches before changing them. Leave intentional historical references
   alone, such as release notes, changelogs, generated build output, and
   third-party dependency attribution entries.
8. Validate with targeted checks:

   ```bash
   ruby -e 'require "yaml"; YAML.load_file(".github/nightly-alpha-branches.yaml"); YAML.load_file(".github/workflows/nightly-alpha-tag.yaml")'
   just set-version <next-version>
   cargo generate-lockfile --manifest-path crates/core/tests/fixtures/worker_plugin/Cargo.toml
   cargo build --locked --manifest-path crates/core/tests/fixtures/worker_plugin/Cargo.toml
   rg -n '<old-version>' README.md docs fern --glob '!docs/_build/**' || true
   git diff --check
   ```

   Any remaining documentation matches for `<old-version>` should be intentional
   and called out in the PR description.
9. After explicit approval to publish, open a PR targeting `main` using
   `.github/pull_request_template.md`. The PR
   must mention:
   - the new release branch
   - the nightly alpha branch config update
   - the `just set-version <next-version>` bump
   - documentation old-version reference updates or intentional leftovers
   - that subsequent release-bound PRs target the new `release/*` branch

## Guardrails

- Do not create release tags. Code freeze only creates the branch and the main
  PR.
- Do not target the code-freeze PR at the release branch. It targets `main`.
- Do not leave uncommitted user changes mixed into the code-freeze PR branch.
