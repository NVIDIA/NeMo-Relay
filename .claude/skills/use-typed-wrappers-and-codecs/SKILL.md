<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Use NeMo Flow typed wrappers and codecs without losing middleware behavior
---

# Use Typed Wrappers And Codecs

Use this skill when an application wants stronger domain types than raw JSON for
tool or LLM integration.

## Default Guidance

- Prefer plain JSON first for initial adoption.
- Reach for typed wrappers when the application already has stable domain models.
- Keep in mind that middleware still operates on JSON, not typed objects.

## Key Rules

- typed wrappers are currently a first-class path for Python and Node.js
- request/response conversion belongs in codecs
- intercepts and guardrails see JSON values after encoding
- changes made by middleware survive into the decode step

## Choose A Codec

- `JsonPassthrough` for JSON-native values
- `DataclassCodec` or `PydanticCodec` in Python when the models already exist
- custom codecs for domain-specific wire shapes
- `BestEffortAnyCodec` only when broad flexibility is worth the looser contract

## References

- `docs/typed-wrappers.md`
- `docs/typed-api-reference.md`
- `docs/llm-codecs.md`
