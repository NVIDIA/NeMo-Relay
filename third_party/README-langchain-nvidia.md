<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LangChain NVIDIA Patch Setup

This directory contains the NeMo Relay integration patch for
`third_party/langchain-nvidia`, specifically the `libs/ai-endpoints`
`langchain_nvidia_ai_endpoints` package.

The patch adds optional NeMo Relay LLM execution wrappers for ChatNVIDIA. The
integration stays inactive unless `nemo_relay` is importable and a NeMo Relay
scope stack is already active.

## Setup

From the NeMo Relay repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/langchain-nvidia apply ../../patches/langchain-nvidia/0001-add-nemo-relay-integration.patch
```

For local runtime validation, install the NeMo Relay Python package and the
patched LangChain NVIDIA package into the same environment:

```bash
uv venv .venv
. .venv/bin/activate
uv pip install -e .
uv pip install -e third_party/langchain-nvidia/libs/ai-endpoints
```

## Usage Example

Use ChatNVIDIA inside an active NeMo Relay scope. The patched package detects
the active scope stack and routes the request through `nemo_relay.llm.execute`
or `nemo_relay.llm.stream_execute`; otherwise it falls back to the vanilla
ChatNVIDIA path.

```python
import nemo_relay
from langchain_nvidia_ai_endpoints import ChatNVIDIA

with nemo_relay.scope.scope("langchain-nvidia-request", nemo_relay.ScopeType.Agent):
    model = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
    response = model.invoke("Summarize NeMo Relay in one sentence.")
    print(response.content)
```

For streaming calls, use ChatNVIDIA's normal streaming API. The patch wraps the
stream with the OpenAI Chat codec because NVIDIA AI Endpoints use an
OpenAI-compatible request shape.

## Validation

Run a structural syntax check for the patched files:

```bash
uv run python -m py_compile \
  third_party/langchain-nvidia/libs/ai-endpoints/langchain_nvidia_ai_endpoints/_nemo_relay.py \
  third_party/langchain-nvidia/libs/ai-endpoints/langchain_nvidia_ai_endpoints/chat_models.py
```

Also rerun the repository patch applicability check before review:

```bash
./scripts/apply-patches.sh --check
```
