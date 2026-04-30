---
name: contribute-integration
description: Contribute a new or updated third-party framework integration for NeMo Flow
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Contribute A Framework Integration

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when contributing an integration with an upstream framework such
as LangChain, LangGraph, or another patched third-party project.

## Default Guidance

- Keep NeMo Flow optional
- Preserve the framework's original behavior when NeMo Flow is absent
- Wrap tool and LLM paths at the correct framework boundary
- Keep the tracked patch artifact minimal and reproducible

## Checklist

- [ ] Integration pattern follows `docs/integrate-frameworks/adding-scopes.md`
- [ ] Patch applies cleanly via `./scripts/apply-patches.sh --check`
- [ ] Patch artifact regenerated if the local checkout changed
- [ ] Relevant integration tests or smoke path pass
- [ ] Docs updated if activation or usage changed

Use the root `./scripts/*.sh` commands in docs and contributor guidance. Their
implementations now live under `scripts/third-party/`.

## References

- `add-integration`
- `maintain-integration-patches`
- `docs/integrate-frameworks/about.md`
- `validate-change`
