<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Contribute a new NeMo Flow public API surface safely, with binding parity and docs in mind
---

# Contribute A New API Surface

Use this skill when contributing a public API addition or behavior change to the
runtime or bindings.

## Default Guidance

- start from the shared core behavior first
- decide which bindings must expose the new surface
- follow the parity checklist in `add-binding-feature`
- update docs and examples in the same branch

## Minimum Acceptance

- public behavior is clearly described
- every affected binding is covered
- the validation matrix matches the changed surfaces
- PR notes explain the user-facing change

## References

- `add-binding-feature`
- `update-docs-and-examples`
- `validate-change`
