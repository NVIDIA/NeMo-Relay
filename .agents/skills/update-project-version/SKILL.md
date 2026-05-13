---
name: update-project-version
description: Update the NeMo Flow project version across Cargo, Node, generated WebAssembly package metadata, and lockfiles without leaving release surfaces out of sync
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Update Project Version

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when changing the released NeMo Flow version, including
pre-release or build-metadata variants used during packaging.

## Source Of Truth

- `Cargo.toml` `[workspace.package].version` is the source of truth for the Rust
  workspace and Python build versioning.
- Keep `Cargo.toml` `[workspace.dependencies]` self-references aligned when the
  workspace version changes.
- `crates/node/package.json` carries its own npm package version and must stay
  aligned with the workspace-root `package-lock.json`.
- `integrations/openclaw/package.json` carries the OpenClaw npm plugin version
  and must stay aligned with the workspace-root `package-lock.json`.
- `package-lock.json` records Node package versions under
  `packages["crates/node"].version` and
  `packages["integrations/openclaw"].version`. The workspace-root lockfile may
  not have a top-level `version` field.
- `crates/wasm/package.json` is a local dev manifest. Do not treat it as the
  publishable package manifest unless it gains an explicit `version` field.
- The publishable WebAssembly npm package version is derived from
  `crates/wasm/Cargo.toml` through `wasm-pack` output plus
  `crates/wasm/scripts/prepare_pkg.mjs`.

## Workflow

1. Read the current version from `Cargo.toml` and decide the exact target
   version string.
2. Run `just set-version <version>` to update release-version source files:
   - `[workspace.package].version`
   - `workspace.dependencies.nemo-flow.version`
   - `workspace.dependencies.nemo-flow-adaptive.version`
   - `workspace.dependencies.nemo-flow-ffi.version`
   - `crates/node/package.json` `version`
   - `integrations/openclaw/package.json` `version`
   - `package-lock.json` `packages["crates/node"].version`
   - `package-lock.json` `packages["integrations/openclaw"].version`
3. If editing helper code, keep `set_project_version`,
   `set_cargo_workspace_version`, and `set_node_package_versions` aligned with
   those same fields. `set_node_package_version` remains a compatibility alias.
   `set_npm_package_version` remains the reusable npm JSON helper for Node and
   WebAssembly packaging recipes.
4. Refresh generated surfaces:
   - Run `cargo check --workspace` to refresh `Cargo.lock` if workspace package
     entries changed.
   - If Cargo metadata changed and committed attribution files must stay fresh,
     regenerate `ATTRIBUTIONS-Rust.md` with
     `./scripts/generate_attributions.sh rust`.
   - If `package-lock.json` changed, regenerate
     `ATTRIBUTIONS-Node.md` with
     `./scripts/generate_attributions.sh node`.
   - If the change needs WebAssembly publish validation, rebuild the generated package
     with `just build-wasm` or
     `NEMO_FLOW_WASM_RELEASE=1 npm run build:pkg --workspace=nemo-flow-wasm`. Inspect
     `crates/wasm/pkg/package.json`, not `crates/wasm/package.json`.
5. Audit remaining references to the old version with targeted search. Separate
   true version pins from examples, generated attribution files, and unrelated
   third-party versions.

## Validation

- `rg -n '^version =|nemo-flow = \\{ version =|nemo-flow-adaptive = \\{ version =' Cargo.toml`
- `rg -n '\"version\"' crates/node/package.json integrations/openclaw/package.json package-lock.json`
- `cargo check --workspace`
- If Rust attribution files are expected to stay current:
  `./scripts/generate_attributions.sh rust`
- If Node packaging changed materially: run `npm install --ignore-scripts` from
  the repository root or stronger Node validation through `just test-node`
- If validating the WebAssembly publish surface: inspect the regenerated
  `crates/wasm/pkg/package.json`

## Release Notes

- `just package-node`, `just package-python`, and `just package-wasm` may set
  temporary non-release versions for packaging. Do not commit those temporary
  suffixes as the canonical project version unless the release process requires
  that exact string.

## Avoid

- Updating only `Cargo.toml` or only Node package metadata
- Assuming `crates/wasm/package.json` is the published npm manifest
- Forgetting `Cargo.lock`, `ATTRIBUTIONS-Rust.md`, or `ATTRIBUTIONS-Node.md`
  after changing versioned inputs that feed them
- Doing blind repository-wide search/replace across docs, patches, and
  generated attribution files

## References

- `Cargo.toml`
- `Cargo.lock`
- `package.json`
- `package-lock.json`
- `crates/node/package.json`
- `integrations/openclaw/package.json`
- `crates/wasm/Cargo.toml`
- `crates/wasm/package.json`
- `crates/wasm/scripts/prepare_pkg.mjs`
- `justfile`
- `scripts/licensing/attributions_lockfile_md.py`
