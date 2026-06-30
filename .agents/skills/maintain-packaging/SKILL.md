---
name: maintain-packaging
description: Maintain NeMo Relay package metadata, module paths, generated artifacts, and release-facing build surfaces
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain Release And Packaging Surfaces

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when a change affects how NeMo Relay is built, packaged, named, or
consumed outside the source tree.

## Audit Areas

- Rust `Cargo.toml` package names and workspace metadata
- Python packaging in `pyproject.toml`
- Python worker plugin SDK packaging in `python/plugin/pyproject.toml`
- Go module path in `go/nemo_relay/go.mod`
- Node workspace metadata in root `package.json` and `package-lock.json`
- Node package metadata in `crates/node/package.json`
- WebAssembly package naming and generated package expectations
- FFI header and library naming
- CI workflows, install commands, and example commands
- `justfile` build, test, clean, version, and package recipes for plugin crates
  and packages
- Release tags, release-note surfaces, and registry-facing version translation

## Checklist

- [ ] Package names, import paths, and module names are internally consistent
- [ ] Generated artifacts still land where downstream consumers expect
- [ ] Docs and examples use the current install/import/build commands
- [ ] CI references the same package names as local workflows
- [ ] Public packaging changes are reflected in release-facing docs
- [ ] `nemo-relay-plugin` Rust and Python packages track the project SemVer
      policy and Python wheels use valid PEP 440 translation
- [ ] Release tags still use raw SemVer without a leading `v`
- [ ] Release history and release notes still point to GitHub Releases, not `CHANGELOG.md` or docs pages

## References

- `pyproject.toml`
- `python/plugin/pyproject.toml`
- `go/nemo_relay/go.mod`
- `package.json`
- `package-lock.json`
- `crates/node/package.json`
- `RELEASING.md`
- `.github/workflows/ci_python.yml`
- `.github/workflows/ci.yaml`
- `.gitlab-ci.yml`
- `maintain-dynamic-plugins`
