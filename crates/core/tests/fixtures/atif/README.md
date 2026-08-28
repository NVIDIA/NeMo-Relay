<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ATIF interleaving fixtures

These minimized ATOF `0.1` fixtures model two LLM calls whose end events occur
544 milliseconds apart. They contain deterministic synthetic identifiers,
timestamps, model names, and payloads. They do not contain original repository
content, filesystem paths, account metadata, encrypted content, or telemetry
identifiers.

## `filtered_flat_interleaved_llm_ends.atof.jsonl`

This regression fixture retains only the two interleaved LLM start/end pairs.
Without usable agent-scope topology, ATIF conversion uses one flat
`StepConversionState`:

1. The child and parent LLM calls start.
2. The child response ends with a message whose phase is `final_answer`.
3. The parent response ends with a `wait_agent` tool call.

Before the fix, the second end event replaced the pending enrichment for the
first agent step, leaving the first step without `extra`. The test verifies the
child message, response-item ID, phase, invocation ID, and ancestry parent ID.

This case proves preservation in the flat fallback path. It does not establish
collaboration sender or recipient identity.

## `synthetic_agent_scoped_interleaving.atof.jsonl`

This control fixture uses the same LLM events and ordering, but adds an included
root agent and two child agent scopes. ATIF conversion can therefore partition
the LLM events into separate subagent trajectories.

The test verifies that both responses are present and retain the ancestry of
their respective owner scopes. This case proves agent-tree isolation; it is a
synthetic control rather than a claim that missing topology can be recovered.
