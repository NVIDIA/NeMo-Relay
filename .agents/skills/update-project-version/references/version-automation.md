<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Version Automation Topology

Read this reference only when adding, removing, relocating, or debugging a
project-owned version surface. The `justfile` is authoritative if this summary
drifts.

## Current Coverage

`set_project_version` coordinates these helpers:

- `set_cargo_workspace_version` updates the workspace version and every
  versioned `nemo-relay*` workspace dependency, then verifies all NeMo Relay
  workspace packages resolve to the target version through Cargo metadata.
- `set_node_package_versions` updates the Node binding, OpenClaw plugin, PI
  extension, their workspace lockfile entries, and project-owned npm dependency
  pins.
- `set_example_package_versions` updates exact Rust and Python SDK pins and
  refreshes their checked lockfiles without upgrading third-party packages.
  Before the Node package is published, its checked example links the local
  workspace package so its lockfile remains installable.
- `set_python_package_version` preserves the root package's dynamic Cargo
  version, keeps the PyO3 crate on the workspace version, updates the CLI binary
  package, and aligns the root CLI extra pin.
- `set_python_plugin_package_version` updates the Python worker plugin SDK.
- `set_coding_agent_plugin_versions` updates the Claude Code and Codex plugin
  manifests under `integrations/coding-agents/`.

Generated attribution files derive from Cargo and npm lockfiles; regenerate
only the attribution surface whose input changed.

## Adding A Versioned Surface

1. Decide whether the package or plugin participates in the unified NeMo Relay
   release. Independently versioned components need their own explicit source
   of truth and must not be silently swept into this automation.
2. For a unified-release surface, decide whether it uses the Cargo/npm SemVer
   string or its PEP 440 translation.
3. Add the manifest field and any project-owned dependency or lockfile entry.
4. Extend the narrow helper that owns that ecosystem. Add an explicit presence
   check so a moved or removed field fails instead of drifting silently. Keep
   this assertion with the helper; do not add a separate test that repeats its
   field list.
5. Call a new helper from `set_project_version` if no existing helper owns the
   surface.
6. Run `just set-version <target>` and inspect the diff.
7. Audit all project-owned manifest versions and internal dependency pins
   against the target, then search manifest and lockfile types for the exact
   previous version as a fast omission signal. Any project-owned mismatch must
   be updated by the automation or documented as an intentional exception.
8. Run the same command again to confirm idempotence, then perform focused
   checks for the helper and generated surfaces that changed.

Avoid maintaining a second exhaustive field list in the skill entrypoint. The
helper implementation and its fail-fast assertions are the executable source
of truth.
