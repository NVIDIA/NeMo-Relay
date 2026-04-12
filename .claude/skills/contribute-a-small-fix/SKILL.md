<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Make a small, reviewable NeMo Flow bug fix without widening scope unnecessarily
---

# Contribute A Small Fix

Use this skill for narrowly scoped bug fixes or behavior corrections.

## Rules

- reproduce or identify the failing behavior first
- keep the change as small as possible
- avoid opportunistic refactors unless they are required to fix the bug safely
- add or update the smallest meaningful test that proves the fix

## Checklist

- [ ] scope of the fix is explicit
- [ ] affected language surfaces are understood
- [ ] a regression test or focused validation path exists
- [ ] docs updated if public behavior changed
- [ ] PR notes can explain what failed before and why the fix is safe

## References

- `.github/CONTRIBUTING.md`
- `validate-change`
