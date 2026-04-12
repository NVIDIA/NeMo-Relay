<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Export NeMo Flow activity as ATIF trajectories for replay, analysis, or interchange
---

# Export ATIF Trajectories

Use this skill when the user wants execution traces as ATIF documents rather than
live OTLP spans.

## Default Path

- Create an `AtifExporter`
- Register it before the instrumented work
- Run scoped tool and LLM activity
- Call `export()` or `export_json()`
- Clear or deregister when done

## Important Semantics

- ATIF exports the full event buffer collected so far
- LLM end events become agent steps
- tool observations are derived from tool end events
- consecutive tool observations can be merged
- trajectories reflect post-guardrail input/output data

## Checklist

- [ ] Session and agent metadata chosen
- [ ] Exporter registered before the relevant run
- [ ] Scope boundaries are correct so ancestry is meaningful
- [ ] Export timing is clear: whole buffer vs clear-between-runs

## References

- `docs/atif-export.md`
- `docs/language-bindings.md`
