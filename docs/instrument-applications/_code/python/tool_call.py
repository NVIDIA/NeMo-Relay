# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import TypedDict

import nemo_relay


class SearchArguments(TypedDict):
    query: str


def log_event(event: nemo_relay.Event) -> None:
    print(f"{event.kind} {event.name}")


async def search(tool_arguments: SearchArguments) -> nemo_relay.ToolExecutionResult[dict[str, object]]:
    return nemo_relay.ToolExecutionResult(
        {
            "query": tool_arguments["query"],
            "hits": [{"title": "NeMo Relay"}],
        }
    )


async def main() -> nemo_relay.ToolExecutionResult[dict[str, object]]:
    nemo_relay.subscribers.register("instrumentation-check", log_event)

    try:
        with nemo_relay.scope.scope("agent-run", nemo_relay.ScopeType.Agent) as handle:
            result = await nemo_relay.tools.execute(
                "search",
                {"query": "runtime instrumentation"},
                search,
                handle=handle,
            )
            print(result.result)
    finally:
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister("instrumentation-check")

    return result


if __name__ == "__main__":
    asyncio.run(main())
