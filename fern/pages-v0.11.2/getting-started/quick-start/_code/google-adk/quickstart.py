# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from google.adk.agents import Context, LlmAgent
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import nemo_relay
from nemo_relay.integrations.google_adk import NemoRelayPlugin


def emit_marker() -> str:
    """Return quick-start validation marker."""
    return "RELAY_QUICKSTART_TOOL_OK"


def require_marker_tool(callback_context: Context, llm_request: LlmRequest) -> None:
    """Force tool call until tool result exists, then disable tools."""
    tool_used = any(
        part.function_response is not None for content in llm_request.contents for part in content.parts or []
    )
    mode = types.FunctionCallingConfigMode.NONE if tool_used else types.FunctionCallingConfigMode.ANY
    llm_request.config.tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))


root_agent = LlmAgent(
    name="marker_agent",
    model="gemini-3.5-flash",
    instruction="Call emit_marker exactly once. Keep the final answer brief.",
    tools=[emit_marker],
    before_model_callback=require_marker_tool,
)
app = App(name="relay_quickstart", root_agent=root_agent, plugins=[NemoRelayPlugin()])


def check_tool_result(events: Sequence[Event]) -> None:
    """Raise unless run produced exactly one marker tool result."""
    results = [
        part.function_response.response
        for event in events
        if event.content is not None
        for part in event.content.parts or []
        if part.function_response is not None
    ]
    if results != [{"result": "RELAY_QUICKSTART_TOOL_OK"}]:
        raise RuntimeError(f"Expected one marker tool result, got {results}")


def check_final_response(events: Sequence[Event]) -> None:
    """Raise unless last event is final model response containing response marker."""
    final = events[-1]
    if not final.is_final_response() or final.content is None:
        raise RuntimeError("Expected final model response as last event")
    text = "".join(part.text for part in final.content.parts or [] if part.text is not None and not part.thought)
    if "RELAY_QUICKSTART_RESPONSE_OK" not in text:
        raise RuntimeError(f"Final response lacks RELAY_QUICKSTART_RESPONSE_OK: {text!r}")


def check_atof_records(path: Path) -> None:
    """Raise unless ATOF file pairs start and end records for `adk-request`, `emit_marker`, and an LLM call."""
    records: list[dict[str, object]] = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or any(
        record.get("atof_version") != "0.1" or record.get("kind") not in {"scope", "mark"} for record in records
    ):
        raise RuntimeError("Expected nonempty ATOF 0.1 event file")
    scopes = [record for record in records if record["kind"] == "scope"]
    starts = {record["uuid"]: record for record in scopes if record["scope_category"] == "start"}
    paired = {
        (record["category"], record["name"])
        for record in scopes
        if record["scope_category"] == "end"
        and record["uuid"] in starts
        and all(record[key] == starts[record["uuid"]][key] for key in ("name", "category", "parent_uuid"))
    }
    categories = {category for category, _ in paired}
    if not {("agent", "adk-request"), ("tool", "emit_marker")} <= paired or "llm" not in categories:
        raise RuntimeError(f"Expected paired adk-request, emit_marker, and LLM scopes, got {paired}")


async def main() -> None:
    """Run instrumented agent, then check tool result, final response, and ATOF records."""
    async with nemo_relay.plugin.activate({}, "plugins.toml"):
        try:
            async with Runner(app=app, session_service=InMemorySessionService()) as runner:
                session = await runner.session_service.create_session(app_name=app.name, user_id="quickstart-user")
                message = types.UserContent(
                    "Use one available tool to produce RELAY_QUICKSTART_TOOL_OK. "
                    "Then include RELAY_QUICKSTART_RESPONSE_OK in your final response."
                )
                with nemo_relay.scope.scope("adk-request", nemo_relay.ScopeType.Agent):
                    events = [
                        event
                        async for event in runner.run_async(
                            user_id=session.user_id, session_id=session.id, new_message=message
                        )
                    ]
        finally:
            await nemo_relay.subscribers.flush_async()

    check_tool_result(events)
    print("RELAY_QUICKSTART_TOOL_OK")
    check_final_response(events)
    print("RELAY_QUICKSTART_RESPONSE_OK")
    check_atof_records(Path("logs/events.jsonl"))
    print("RELAY_QUICKSTART_EVENTS_OK")


if __name__ == "__main__":
    asyncio.run(main())
