<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Prepare a NeMo Flow branch for review with the right tests, docs, and contributor hygiene
---

# Prepare A PR For NeMo Flow

Use this skill at the end of a contributor or maintainer change before opening a
pull request.

## Checklist

- [ ] branch scope is coherent and reviewable
- [ ] relevant tests passed
- [ ] `uv run pre-commit run --all-files` passed or issues are understood
- [ ] docs and examples updated for any public behavior changes
- [ ] commit messages and PR summary explain what changed, why, and how it was tested
- [ ] breaking changes or renamed surfaces are called out explicitly

## PR Description Should Cover

- what changed
- why the change exists
- key implementation notes
- tests run
- any breaking behavior or migration notes

## References

- `.github/CONTRIBUTING.md`
- `validate-change`
