<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Add a new third-party framework integration (e.g., LangChain, CrewAI, AutoGen)
---

# Adding a Framework Integration

Nexus integrations with upstream projects are maintained as git submodules
under `third_party/` with corresponding patch files in `patches/`.

## Overview

See `docs/integration-best-practices.md` for the full guide (649 lines covering
10 sections with code examples).

## Quick Reference

### Structure

```
third_party/<name>/     # git submodule pointing to upstream repo
patches/<name>/         # Nexus integration patches
  0001-add-nat-nexus-integration.patch
```

### Key Patterns (from the integration guide)

1. **Lazy import** — `nat_nexus` is optional; use try/except import
2. **Transparent fallback** — If nat_nexus unavailable, framework works normally
3. **Wrap LLM calls** — Use `nat_nexus.llm.execute()` around provider calls
4. **Wrap tool calls** — Use `nat_nexus.tools.execute()` around tool invocations
5. **Create scopes** — Push agent/function scopes at appropriate boundaries
6. **Thread/async safety** — Propagate scope stacks across thread boundaries

### Applying patches

```bash
cd third_party/<name>
git checkout .                          # Reset to upstream HEAD
git apply ../../patches/<name>/*.patch  # Apply Nexus patches
```

### Regenerating patches

```bash
cd third_party/<name>
git diff HEAD -- . > ../../patches/<name>/0001-add-nat-nexus-integration.patch
```

### Updating upstream

```bash
cd third_party/<name>
git fetch origin
git checkout <new-tag-or-commit>
cd ../..
git add third_party/<name>
# Re-apply and resolve any conflicts in the patch, then regenerate
```

## Full Guide

See `docs/integration-best-practices.md` for complete patterns, code examples,
and the 10-item integration checklist.
