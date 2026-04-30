---
name: contribute-api
description: Contribute a new NeMo Flow public API surface safely, with binding parity and docs in mind
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Contribute A New API Surface

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when contributing a public API addition or behavior change to the
runtime or bindings.

## Default Guidance

- Start from the shared core behavior first
- Decide which bindings must expose the new surface
- Follow the parity checklist in `add-binding-feature`
- Update docs and examples in the same branch

## Minimum Acceptance

- Public behavior is clearly described
- Every affected binding is covered
- The validation matrix matches the changed surfaces
- PR notes explain the user-facing change

## References

- `add-binding-feature`
- `validate-change`
- `CONTRIBUTING.md`
- `review-doc-style`
