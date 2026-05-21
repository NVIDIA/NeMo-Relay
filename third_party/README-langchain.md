<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LangChain Patch Setup

This directory contains the NeMo Relay integration patch for
`third_party/langchain`.

The patch touches LangChain Core callbacks/tools plus the OpenAI and Anthropic
partner packages. It adds optional NeMo Relay request, streaming, and callback
bridges that no-op when `nemo_relay` is unavailable or no scope stack is active.

For an alternate approach refer to [the public API-based integration in `python/nemo_relay/integrations/langchain`](../python/nemo_relay/integrations/langchain/README.md).

## Setup

From the NeMo Relay repository root:

```bash
./scripts/bootstrap-third-party.sh
./scripts/apply-patches.sh --check
git -C third_party/langchain apply ../../patches/langchain/0001-add-nemo-relay-integration.patch
```

For local runtime validation, install NeMo Relay and the relevant editable
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
an active NeMo Relay scope. The OpenAI and Anthropic partner patches wrap LLM
execution with provider-specific codecs when a NeMo Relay scope stack is active.

```python
import nemo_relay
from langchain_core.callbacks import NemoRelayCallbackHandler
from langchain_openai import ChatOpenAI

handler = NemoRelayCallbackHandler()

with nemo_relay.scope.scope("langchain-request", nemo_relay.ScopeType.Agent):
    model = ChatOpenAI(model="gpt-5.4")
    response = model.invoke(
        "Summarize NeMo Relay in one sentence.",
        config={"callbacks": [handler]},
    )
    print(response.content)
```

For Anthropic, use the same pattern with `langchain_anthropic.ChatAnthropic`.
The patch chooses `AnthropicMessagesCodec` for Anthropic requests and
`OpenAIChatCodec` for OpenAI requests.

## Validation

Run the NeMo Relay callback test from the LangChain Core package:

```bash
cd third_party/langchain/libs/core
uv run --group test pytest tests/unit_tests/callbacks/test_nemo_relay_handler.py -q
```

Run a syntax check for the patched Python files from the NeMo Relay repository
root:

```bash
uv run python -m py_compile \
  third_party/langchain/libs/core/langchain_core/callbacks/nemo_relay_handler.py \
  third_party/langchain/libs/core/langchain_core/tools/base.py \
  third_party/langchain/libs/core/langchain_core/utils/_nemo_relay.py \
  third_party/langchain/libs/partners/anthropic/langchain_anthropic/_nemo_relay.py \
  third_party/langchain/libs/partners/anthropic/langchain_anthropic/chat_models.py \
  third_party/langchain/libs/partners/openai/langchain_openai/chat_models/_nemo_relay.py \
  third_party/langchain/libs/partners/openai/langchain_openai/chat_models/base.py
```

Also rerun the root integration codec coverage:

```bash
uv run pytest python/tests/test_integration_codecs.py -q
./scripts/apply-patches.sh --check
```
