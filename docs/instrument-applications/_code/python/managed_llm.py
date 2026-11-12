# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo_relay
from nemo_relay import LLMRequest


async def run_example() -> nemo_relay.Json:
    """Execute the local provider callback through Relay."""
    request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})

    async def invoke(provider_request: LLMRequest) -> dict[str, object]:
        return {"text": "hi", "request": provider_request.content}

    response = await nemo_relay.llm.execute(
        "demo-provider",
        request,
        invoke,
        model_name="demo-model",
    )
    return response
