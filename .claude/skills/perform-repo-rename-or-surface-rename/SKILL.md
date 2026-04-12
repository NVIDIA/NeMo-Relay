<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Perform a coordinated repository, package, crate, module, or symbol rename across NeMo Flow
---

# Perform A Repo Rename Or Surface Rename

Use this skill for coordinated naming changes such as repository renames, crate
prefix changes, package/module renames, import-path changes, FFI symbol renames,
or branding text updates that must preserve functional identifiers.

## Rename Buckets To Audit

- repository references
- Rust crate names and module prefixes
- Python package name and top-level module
- Go module path and package paths
- Node package names
- WASM crate or generated package names
- C header names and symbol prefixes
- docs, examples, CI, and patch artifacts

## Rules

- Separate **branding text** from **functional identifiers**.
- Preserve repository and import paths exactly where code depends on them.
- Update generated or generated-from-build surfaces such as `nemo_flow.h` through
  the proper build step.
- Search for old names after the rename and validate every public language
  surface.

## Checklist

- [ ] Manifests updated
- [ ] Source imports and symbol names updated
- [ ] Docs and examples updated
- [ ] Patch files and scripts updated
- [ ] No stale old names remain in tracked files where they would break behavior
- [ ] Full multi-language validation passes

## References

- `README.md`
- `docs/language-bindings.md`
- `docs/api-reference.md`
- `validate-change`
