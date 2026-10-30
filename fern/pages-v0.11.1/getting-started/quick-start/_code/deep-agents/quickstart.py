# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.tools import tool

import nemo_relay
from nemo_relay.integrations.deepagents import (
    NemoRelayDeepAgentsCallbackHandler,
    add_nemo_relay_integration,
)


@tool
def emit_marker() -> str:
    """Return the quick-start validation marker."""
    return "RELAY_QUICKSTART_TOOL_OK"


agent = create_deep_agent(
    **add_nemo_relay_integration(
        model="nvidia:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        tools=[emit_marker],
        name="main-agent",
        system_prompt="Call emit_marker exactly once. Keep the final answer brief.",
    )
)


async def main() -> None:
    """Run the instrumented agent and verify its tool, response, and event markers."""
    async with nemo_relay.plugin.activate({}, "plugins.toml"):
        try:
            with nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent):
                result = await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Use one available tool to produce "
                                    "RELAY_QUICKSTART_TOOL_OK. Then include "
                                    "RELAY_QUICKSTART_RESPONSE_OK in your final response."
                                ),
                            }
                        ]
                    },
                    config={"callbacks": [NemoRelayDeepAgentsCallbackHandler()]},
                )

            tool_outputs = [
                str(message.content) for message in result["messages"] if getattr(message, "type", None) == "tool"
            ]
            if "RELAY_QUICKSTART_TOOL_OK" not in tool_outputs:
                raise RuntimeError("Expected tool result is missing")
            final_message = result["messages"][-1]
            if getattr(final_message, "type", None) != "ai":
                raise RuntimeError("Expected a final assistant message")
            final_response = str(final_message.content)
            if "RELAY_QUICKSTART_RESPONSE_OK" not in final_response:
                raise RuntimeError("Expected final response is missing")
            print("RELAY_QUICKSTART_TOOL_OK")
            print("RELAY_QUICKSTART_RESPONSE_OK")
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
    # Match the required identities, not just categories that unrelated work could satisfy.
    for category, name in (("agent", "deepagents-request"), ("tool", "emit_marker"), ("llm", None)):
        if not any(
            event.get("category") == category and (name is None or event.get("name") == name) for event in closed
        ):
            raise RuntimeError(f"Expected paired {category} scope: {name or 'model call'}")
    print("RELAY_QUICKSTART_EVENTS_OK")


asyncio.run(main())
