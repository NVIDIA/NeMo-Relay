<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Maintain NeMo Flow package metadata, module paths, generated artifacts, and release-facing build surfaces
---

# Maintain Release And Packaging Surfaces

Use this skill when a change affects how NeMo Flow is built, packaged, named, or
consumed outside the source tree.

## Audit Areas

- Rust `Cargo.toml` package names and workspace metadata
- Python packaging in `pyproject.toml`
- Go module path in `go/nemo_flow/go.mod`
- Node package metadata in `crates/node/package.json`
- WASM package naming and generated package expectations
- FFI header and library naming
- CI workflows, install commands, and example commands

## Checklist

- [ ] Package names, import paths, and module names are internally consistent
- [ ] Generated artifacts still land where downstream consumers expect
- [ ] Docs and examples use the current install/import/build commands
- [ ] CI references the same package names as local workflows
- [ ] Public packaging changes are reflected in release-facing docs

## References

- `pyproject.toml`
- `go/nemo_flow/go.mod`
- `crates/node/package.json`
- `.github/workflows/ci_pipe.yml`
- `.gitlab-ci.yml`
