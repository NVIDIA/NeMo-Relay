---
name: small-fix
description: Make a small, reviewable NeMo Flow bug fix without widening scope unnecessarily
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Contribute A Small Fix

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill for narrowly scoped bug fixes or behavior corrections.

## Rules

- Reproduce or identify the failing behavior first
- Keep the change as small as possible
- Avoid opportunistic refactors unless they are required to fix the bug safely
- Add or update the smallest meaningful test that proves the fix

## Checklist

- [ ] Scope of the fix is explicit
- [ ] Affected language surfaces are understood
- [ ] A regression test or focused validation path exists
- [ ] Docs updated if public behavior changed
- [ ] PR notes can explain what failed before and why the fix is safe

## References

- `CONTRIBUTING.md`
- `validate-change`
