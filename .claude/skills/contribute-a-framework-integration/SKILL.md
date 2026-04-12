<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Contribute a new or updated third-party framework integration for NeMo Flow
---

# Contribute A Framework Integration

Use this skill when contributing an integration with an upstream framework such
as LangChain, LangGraph, or another patched third-party project.

## Default Guidance

- keep NeMo Flow optional
- preserve the framework's original behavior when NeMo Flow is absent
- wrap tool and LLM paths at the correct framework boundary
- keep the tracked patch artifact minimal and reproducible

## Checklist

- [ ] integration pattern follows `docs/integration-best-practices.md`
- [ ] patch applies cleanly via `./scripts/apply-patches.sh --check`
- [ ] patch artifact regenerated if the local checkout changed
- [ ] relevant integration tests or smoke path pass
- [ ] docs updated if activation or usage changed

## References

- `add-integration`
- `maintain-integration-patches`
- `validate-change`
