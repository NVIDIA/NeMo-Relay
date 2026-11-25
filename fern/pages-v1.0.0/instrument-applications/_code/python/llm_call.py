# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import nemo_relay


def log_event(event: nemo_relay.Event) -> None:
    print(f"{event.kind} {event.name}")


async def call_provider(request: nemo_relay.LLMRequest) -> dict[str, object]:
    return {
        "text": "hello",
        "messages": request.content["messages"],
    }


async def main() -> nemo_relay.Json:
    nemo_relay.subscribers.register("llm-check", log_event)

    try:
        with nemo_relay.scope.scope("agent-run", nemo_relay.ScopeType.Agent) as handle:
            request = nemo_relay.LLMRequest(
                {},
                {"messages": [{"role": "user", "content": "hello"}]},
            )
            result = await nemo_relay.llm.execute(
                "demo-provider",
                request,
                call_provider,
                handle=handle,
                model_name="demo-model",
            )
            print(result)
    finally:
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister("llm-check")

    return result


if __name__ == "__main__":
    asyncio.run(main())
