---
name: rename-surfaces
description: Perform a coordinated repository, package, crate, module, or symbol rename across NeMo Relay
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Perform A Repo Rename Or Surface Rename

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill for coordinated naming changes such as repository renames, crate
prefix changes, package/module renames, import-path changes, FFI symbol renames,
or branding text updates that must preserve functional identifiers.

## Rename Buckets To Audit

- Repository references
- Rust crate names and module prefixes
- Python package name and top-level module
- Go module path and package paths
- Node package names
- C header names and symbol prefixes
- Docs, examples, CI, and integration packages

## Rules

- Separate **branding text** from **functional identifiers**.
- Preserve repository and import paths exactly where code depends on them.
- Update generated or generated-from-build surfaces such as
  `crates/ffi/nemo_relay.h` through the proper build step.
- Search for old names after the rename and validate every public language
  surface.

## Checklist

- [ ] Manifests updated
- [ ] Source imports and symbol names updated
- [ ] Docs and examples updated
- [ ] Integration packages and scripts updated
- [ ] No stale old names remain in tracked files where they would break behavior
- [ ] Full multi-language validation passes

## References

- `README.md`
- `docs/getting-started/quick-start.md`
- `docs/reference/api/index.md`
- `validate-change`
