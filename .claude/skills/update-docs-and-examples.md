<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Update NeMo Flow docs, examples, and README surfaces when public behavior or packaging changes
---

# Update Docs And Examples

Use this skill when a code change affects public behavior, binding usage,
examples, packaging names, or repo entry points.

## Required Review Path

Check these surfaces when public behavior changed:

- `README.md`
- `docs/README.md`
- relevant reference docs under `docs/`
- crate or package READMEs for the affected surface
- `examples/README.md`
- examples in the affected language

## Common Update Buckets

- **API shape changed**
  Update `docs/api-reference.md`, `docs/language-bindings.md`, and any typed or
  optimizer references.
- **Binding-specific usage changed**
  Update the matching `docs/getting-started-*.md` file and the binding README.
- **Observability changed**
  Update ATIF, OpenTelemetry, or OpenInference docs as applicable.
- **Optimizer changed**
  Update `docs/optimizer-layer.md`, `docs/optimizer-api-reference.md`, and any
  examples that rely on the old config shape.
- **Packaging, repo names, or import paths changed**
  Update manifests, install commands, examples, and textual repo references
  together.

## Checklist

- [ ] Entry-point docs still match the current workspace layout
- [ ] Commands, import paths, and package names are current
- [ ] Examples reflect the preferred public API, not stale internal patterns
- [ ] Language-specific docs stay aligned where parity matters
- [ ] Docs mention optional dependencies and patch workflow where relevant

## References

- Contributor checklist: `.github/CONTRIBUTING.md`
- Docs index: `docs/README.md`
- Examples index: `examples/README.md`
