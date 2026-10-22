# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import nemo_relay
from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler


class State(TypedDict):
    """Carry the deterministic graph and response markers between nodes."""

    graph_marker: str
    response_marker: str


def emit_marker(_: State) -> State:
    """Produce local validation markers without an inference request."""
    return {
        "graph_marker": "RELAY_QUICKSTART_GRAPH_OK",
        "response_marker": "RELAY_QUICKSTART_RESPONSE_OK",
    }


builder = StateGraph(State)
builder.add_node("emit_marker", emit_marker)
builder.add_edge(START, "emit_marker")
builder.add_edge("emit_marker", END)
graph = builder.compile()


async def main() -> None:
    """Run the graph and verify that Relay exported the completed marker node."""
    async with nemo_relay.plugin.activate({}, "plugins.toml"):
        try:
            with nemo_relay.scope.scope("langgraph-request", nemo_relay.ScopeType.Agent):
                result = await graph.ainvoke(
                    {"graph_marker": "", "response_marker": ""},
                    config={"callbacks": [NemoRelayCallbackHandler()]},
                )

            print(result["graph_marker"])
            print(result["response_marker"])
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
    # Both scopes are part of this guide's contract; an unrelated completed node is insufficient.
    for name in ("langgraph-request", "emit_marker"):
        if not any(event.get("name") == name and event.get("category") == "agent" for event in closed):
            raise RuntimeError(f"Expected the completed {name} scope")
    print("RELAY_QUICKSTART_EVENTS_OK")


asyncio.run(main())
