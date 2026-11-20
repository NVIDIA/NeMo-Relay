# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

import nemo_relay
from nemo_relay.integrations.langchain import NemoRelayCallbackHandler, NemoRelayMiddleware


@tool
def emit_marker() -> str:
    """Return the quick-start validation marker."""
    return "RELAY_QUICKSTART_TOOL_OK"


@wrap_model_call
async def require_marker_tool(request, handler):
    """Require the marker tool once, then allow the model's final response."""
    # Require a real tool call before the final answer; a prompt alone can be ignored.
    tool_used = any(getattr(message, "type", None) == "tool" for message in request.messages)
    return await handler(request.override(tool_choice="none" if tool_used else "emit_marker"))


agent = create_agent(
    model=init_chat_model("nvidia:nvidia/nemotron-3.5-lightning-30b-a3b", temperature=0),
    tools=[emit_marker],
    middleware=[require_marker_tool, NemoRelayMiddleware()],
    system_prompt="Call emit_marker exactly once. Keep the final answer brief.",
)


async def main() -> None:
    """Run the instrumented agent and verify its response and event lifecycle."""
    async with nemo_relay.plugin.activate({}, "plugins.toml"):
        try:
            with nemo_relay.scope.scope("langchain-request", nemo_relay.ScopeType.Agent):
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
                    config={"callbacks": [NemoRelayCallbackHandler()]},
                )

            tool_outputs = [
                str(message.content) for message in result["messages"] if getattr(message, "type", None) == "tool"
            ]
            if tool_outputs.count("RELAY_QUICKSTART_TOOL_OK") != 1:
                raise RuntimeError("Expected exactly one marker tool result")
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
    if not {"agent", "tool", "llm"} <= {event.get("category") for event in closed}:
        raise RuntimeError("Expected paired agent, tool, and LLM lifecycle records")
    print("RELAY_QUICKSTART_EVENTS_OK")


asyncio.run(main())
