# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from typing import TypedDict

import nemo_relay


class SearchArgs(TypedDict):
    query: str


async def search(args: SearchArgs) -> nemo_relay.ToolExecutionResult[dict[str, str]]:
    return nemo_relay.ToolExecutionResult({"marker": args["query"]})


async def model(request: nemo_relay.LLMRequest) -> dict[str, object]:
    return {"content": "RELAY_QUICKSTART_RESPONSE_OK"}


async def main() -> None:
    async with nemo_relay.plugin.activate({}, "plugins.toml"):
        try:
            with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent) as handle:
                nemo_relay.scope.event("initialized", handle=handle, data={"binding": "python"})

                tool_result = await nemo_relay.tools.execute(
                    "emit_marker",
                    {"query": "RELAY_QUICKSTART_TOOL_OK"},
                    search,
                    handle=handle,
                )
                llm_result = await nemo_relay.llm.execute(
                    "demo-provider",
                    nemo_relay.LLMRequest(
                        {},
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "RELAY_QUICKSTART_RESPONSE_OK",
                                }
                            ]
                        },
                    ),
                    model,
                    handle=handle,
                )

                print(tool_result.result["marker"])
                print(llm_result["content"])
        finally:
            await nemo_relay.subscribers.flush_async()

    events = [json.loads(line) for line in Path("logs/events.jsonl").read_text().splitlines()]
    if not events or any(
        not isinstance(event, dict) or event.get("atof_version") != "0.1" or event.get("kind") not in {"scope", "mark"}
        for event in events
    ):
        raise RuntimeError("Expected a nonempty ATOF 0.1 event stream")
    starts = {
        event["uuid"]: event for event in events if event["kind"] == "scope" and event.get("scope_category") == "start"
    }
    closed = [
        event
        for event in events
        if event["kind"] == "scope"
        and event.get("scope_category") == "end"
        and event.get("uuid") in starts
        and all(event.get(key) == starts[event["uuid"]].get(key) for key in ("name", "category", "parent_uuid"))
    ]
    if not {"agent", "tool", "llm"} <= {event.get("category") for event in closed}:
        raise RuntimeError("Expected paired agent, tool, and LLM lifecycle records")
    print("RELAY_QUICKSTART_EVENTS_OK")


asyncio.run(main())
