<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Configure and use NeMo Flow OpenInference export for OTLP backends that understand OpenInference semantics
---

# Export OpenInference Traces

Use this skill when the destination expects OpenInference semantic conventions,
for example Arize Phoenix or another OpenInference-aware OTLP backend.

## Default Path

- Build the binding-specific `OpenInferenceConfig`
- Set endpoint, transport, service metadata, and headers
- Construct and register the subscriber
- Run instrumented scoped work
- Deregister, flush, and shut down when done

## Important Semantics

- spans include OpenInference semantic attributes
- LLM spans derive `input.value` from request content, not request headers
- scope types map to OpenInference span kinds
- orphan mark events still export as zero-duration spans

## Troubleshooting Focus

- no spans in the OpenInference-aware backend
- expected semantic attributes missing
- wrong scope types or no active scope
- wrong OTLP transport for the chosen binding

## References

- `docs/observability-with-openinference.md`
- `crates/openinference/README.md`
