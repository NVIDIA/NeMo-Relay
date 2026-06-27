---
name: nemo-relay-migrate-from-flow
description: Migrate applications, examples, integrations, documentation, package manifests, and repository code from NeMo Flow naming and packages to NeMo Relay across Python, Rust, Node.js, Go, WebAssembly, C FFI, CLI, config, and observability surfaces; use when a user asks to rename nemo_flow/nemo-flow/NeMo Flow APIs, automate a migration, update imports or dependencies, or validate a Flow-to-Relay conversion
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

<!-- Non-semantic CI refresh marker: 2026-06-06. -->

# Migrate From NeMo Flow To NeMo Relay

Use this skill when a user has existing NeMo Flow code or documentation and
wants it converted to NeMo Relay. Treat the migration as a mechanical rename
plus language-specific validation, not a behavior rewrite.

## Default Workflow

1. Inspect the working tree and identify touched surfaces: Rust, Python,
   Node.js, Go, WebAssembly, C FFI, CLI/config, docs, or integration patches.
2. Run the bundled helper in dry-run mode before editing:
   `python skills/nemo-relay-migrate-from-flow/scripts/migrate_from_nemo_flow.py <path> --rename-paths`
3. Review the reported text edits and path renames. If the scope is correct,
   rerun with `--write --rename-paths`.
4. Apply language-specific cleanup for package manager lockfiles, generated
   artifacts, and public API examples.
5. Search for remaining Flow names and verify the affected language surfaces.

## Mechanical Rename Map

- Brand and repository: `NeMo Flow` -> `NeMo Relay`,
  `NeMo-Flow` -> `NeMo-Relay`
- Python: `nemo-flow` -> `nemo-relay`, `nemo_flow` -> `nemo_relay`,
  `python/nemo_flow` -> `python/nemo_relay`
- Rust: `nemo-flow` -> `nemo-relay`, `nemo-flow-adaptive` ->
  `nemo-relay-adaptive`, `nemo_flow::` -> `nemo_relay::`
- Node.js and WebAssembly: `nemo-flow-node` -> `nemo-relay-node`,
  `nemo-flow-wasm` -> `nemo-relay-wasm`, and related entry points such as
  `/typed`, `/plugin`, `/adaptive`, and `/observability`
- Go: `github.com/NVIDIA/NeMo-Flow/go/nemo_flow` ->
  `github.com/NVIDIA/NeMo-Relay/go/nemo_relay`, package aliases
  `nemo_flow` -> `nemo_relay`, and source directories `go/nemo_flow` ->
  `go/nemo_relay`
- C FFI: `nemo_flow.h` -> `nemo_relay.h`, `nemo_flow_*` ->
  `nemo_relay_*`, `NemoFlow*` -> `NemoRelay*`, and `NEMO_FLOW_*` ->
  `NEMO_RELAY_*`
- CLI/config: `nemo-flow` -> `nemo-relay`, `.nemo-flow` -> `.nemo-relay`,
  `~/.config/nemo-flow` -> `~/.config/nemo-relay`, `NEMO_FLOW_*` ->
  `NEMO_RELAY_*`, and `x-nemo-flow-*` -> `x-nemo-relay-*`

Do not replace bare `flow`, `Flow`, or `FlowError`. Those can be domain words
or intentional compatibility names.

## Language Cleanup

- **Python**: update `pyproject.toml`, imports, type stubs, integration
  package paths, extras, and native module names. Regenerate or refresh lockfiles
  with the user's package workflow after source edits.
- **Rust**: update `Cargo.toml` crate names, workspace dependencies, package
  references, and `use nemo_relay::...` imports. Let Cargo regenerate
  `Cargo.lock` when dependencies changed.
- **Node.js**: update `package.json`, workspace names, package-lock entries,
  native addon artifact names, and imports from `nemo-relay-node` or
  `nemo-relay-wasm`. Run the package manager to refresh locks.
- **Go**: update `go.mod`, import paths, package declarations, aliases, and any
  local directory layout under `go/nemo_relay`.
- **C FFI**: update header includes, exported symbol names, status and callback
  type names, macro constants, loader paths, and downstream bindings.
- **Docs and examples**: update badges, package install commands, repository
  links, hosted docs URLs, CLI commands, config paths, and integration patch
  names.

## Automation Helper

Use `scripts/migrate_from_nemo_flow.py` for first-pass edits. It:

- runs as a dry run unless `--write` is passed
- skips common vendor, build, cache, and generated directories
- skips lockfiles unless `--include-lockfiles` is passed
- can report or perform path renames with `--rename-paths`
- rewrites only explicit NeMo Flow identifiers, package names, repository names,
  config paths, headers, environment variables, and FFI type prefixes

Run it from either the source repository or the user's target project:

```bash
python skills/nemo-relay-migrate-from-flow/scripts/migrate_from_nemo_flow.py . --rename-paths
python skills/nemo-relay-migrate-from-flow/scripts/migrate_from_nemo_flow.py . --write --rename-paths
```

Use `--include-lockfiles` only when the user wants lockfiles edited directly;
otherwise regenerate them with Cargo, uv/pip, npm, or Go tooling.

## Verification

- Search for remaining explicit Flow identifiers:
  `rg -n "NeMo Flow|NeMo-Flow|nemo_flow|nemo-flow|NEMO_FLOW|NemoFlow|nemo_flow\\.h|nemo_flow_"`
- Run targeted tests for every affected language surface.
- For Rust changes, run `cargo test` or the repository's Rust test recipe.
- For Python changes, run the relevant import check and tests in the target
  environment.
- For Node.js or WebAssembly changes, run package install, type checks, and
  package tests.
- For Go changes, run `go test ./...` from the updated module.
- For docs-only migrations, build or link-check docs if the site navigation,
  install commands, or API references changed.

## Related Skills

- `nemo-relay-start`
- `nemo-relay-instrument-calls`
- `nemo-relay-debug-runtime-integration`
