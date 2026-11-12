# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo_relay
from nemo_relay import LLMRequest


async def run_example() -> tuple[nemo_relay.Json, LLMRequest]:
    """Apply selected middleware without invoking a callback."""
    tool_arguments = await nemo_relay.tools.request_intercepts("search", {"query": "weather"})
    await nemo_relay.tools.conditional_execution("search", tool_arguments)

    llm_request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})
    outcome = await nemo_relay.llm.request_intercepts("demo-provider", llm_request)
    await nemo_relay.llm.conditional_execution(outcome.request)
    return (tool_arguments, outcome.request)
