<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LangGraph Patch Setup

This directory contains the NeMo Relay integration patch for
`third_party/langgraph`.

The patch adds LangGraph lifecycle, checkpoint, interrupt, retry, superstep,
and edge event emission through `langgraph._nemo_relay`. Tests for this patch
live in the first-party `third_party/langgraph_tests` directory.

## Setup

From the NeMo Relay repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/langgraph apply ../../patches/langgraph/0001-add-nemo-relay-integration.patch
```

For local runtime validation, expose the patched LangGraph package on
`PYTHONPATH` while running the first-party tests:

```bash
PYTHONPATH=third_party/langgraph/libs/langgraph uv run pytest third_party/langgraph_tests -q
```

## Usage Example

Run a LangGraph graph inside an active NeMo Relay scope. The patch emits graph
lifecycle, superstep, edge, retry, interrupt, checkpoint save, and checkpoint
restore events through `langgraph._nemo_relay`.

```python
from typing import TypedDict

import nemo_relay
from langgraph.graph import END, StateGraph


class State(TypedDict):
    value: int


def increment(state: State) -> State:
    return {"value": state["value"] + 1}


builder = StateGraph(State)
builder.add_node("increment", increment)
builder.set_entry_point("increment")
builder.add_edge("increment", END)
graph = builder.compile()

with nemo_relay.scope.scope("langgraph-run", nemo_relay.ScopeType.Agent):
    result = graph.invoke({"value": 0})
    print(result)
```

Register a NeMo Relay subscriber or ATIF exporter before invoking the graph if
you want to inspect the emitted events.

## Validation

Run a syntax check for the patched LangGraph files:

```bash
uv run python -m py_compile \
  third_party/langgraph/libs/langgraph/langgraph/_nemo_relay.py \
  third_party/langgraph/libs/langgraph/langgraph/pregel/_loop.py \
  third_party/langgraph/libs/langgraph/langgraph/pregel/_retry.py \
  third_party/langgraph/libs/langgraph/langgraph/pregel/_write.py \
  third_party/langgraph/libs/langgraph/langgraph/pregel/main.py
```

Also rerun the patch applicability check:

```bash
./scripts/apply-patches.sh --check
```
