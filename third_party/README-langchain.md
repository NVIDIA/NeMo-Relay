<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LangChain Patch Setup

This directory contains the NeMo Flow integration patch for
`third_party/langchain`.

The patch touches LangChain Core callbacks/tools plus the OpenAI and Anthropic
partner packages. It adds optional NeMo Flow request, streaming, and callback
bridges that no-op when `nemo_flow` is unavailable or no scope stack is active.

For an alternate approach refer to [the public API-based integration in `python/nemo_flow/integrations/langchain`](../python/nemo_flow/integrations/langchain/README.md).

## Setup

From the NeMo Flow repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/langchain apply ../../patches/langchain/0001-add-nemo-flow-integration.patch
```

For local runtime validation, install NeMo Flow and the relevant editable
LangChain packages into the same Python environment:

```bash
uv venv .venv
. .venv/bin/activate
uv pip install -e .
uv pip install -e third_party/langchain/libs/core
uv pip install -e third_party/langchain/libs/partners/openai
uv pip install -e third_party/langchain/libs/partners/anthropic
```

## Usage Example

Use the callback handler for LangChain run scopes and run provider calls inside
an active NeMo Flow scope. The OpenAI and Anthropic partner patches wrap LLM
execution with provider-specific codecs when a NeMo Flow scope stack is active.

```python
import nemo_flow
from langchain_core.callbacks import NemoFlowCallbackHandler
from langchain_openai import ChatOpenAI

handler = NemoFlowCallbackHandler()

with nemo_flow.scope.scope("langchain-request", nemo_flow.ScopeType.Agent):
    model = ChatOpenAI(model="gpt-5.4")
    response = model.invoke(
        "Summarize NeMo Flow in one sentence.",
        config={"callbacks": [handler]},
    )
    print(response.content)
```

For Anthropic, use the same pattern with `langchain_anthropic.ChatAnthropic`.
The patch chooses `AnthropicMessagesCodec` for Anthropic requests and
`OpenAIChatCodec` for OpenAI requests.

## Validation

Run the NeMo Flow callback test from the LangChain Core package:

```bash
cd third_party/langchain/libs/core
uv run --group test pytest tests/unit_tests/callbacks/test_nemo_flow_handler.py -q
```

Run a syntax check for the patched Python files from the NeMo Flow repository
root:

```bash
uv run python -m py_compile \
  third_party/langchain/libs/core/langchain_core/callbacks/nemo_flow_handler.py \
  third_party/langchain/libs/core/langchain_core/tools/base.py \
  third_party/langchain/libs/core/langchain_core/utils/_nemo_flow.py \
  third_party/langchain/libs/partners/anthropic/langchain_anthropic/_nemo_flow.py \
  third_party/langchain/libs/partners/anthropic/langchain_anthropic/chat_models.py \
  third_party/langchain/libs/partners/openai/langchain_openai/chat_models/_nemo_flow.py \
  third_party/langchain/libs/partners/openai/langchain_openai/chat_models/base.py
```

Also rerun the root integration codec coverage:

```bash
uv run pytest python/tests/test_integration_codecs.py -q
./scripts/apply-patches.sh --check
```
