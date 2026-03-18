# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic LangChain agent with NeMo Agent Toolkit Nexus event logging and ATIF trajectory export.

This example creates a ReAct agent using ChatNVIDIA and registers:
  1. An event subscriber that logs all lifecycle events to the console
  2. An ATIF exporter that collects events and exports a trajectory JSON file

Prerequisites:
    - Set NVIDIA_API_KEY in your environment
    - Install dependencies: uv sync

Usage:
    uv run python agent_with_logging.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import nat_nexus
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Stub implementation for demonstration purposes.
    weather = {
        "san francisco": "62°F, foggy",
        "new york": "45°F, cloudy",
        "london": "50°F, rainy",
    }
    return weather.get(city.lower(), f"No weather data available for {city}")


@tool
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    # Stub implementation for demonstration purposes.
    populations = {
        "san francisco": "873,965",
        "new york": "8,336,817",
        "london": "8,982,000",
    }
    return populations.get(city.lower(), f"No population data available for {city}")


# ---------------------------------------------------------------------------
# NeMo Agent Toolkit Nexus event subscriber
# ---------------------------------------------------------------------------


def log_event(event: nat_nexus.Event) -> None:
    """Log every NeMo Agent Toolkit Nexus lifecycle event to stdout."""
    parts = [
        f"[{event.event_type}]",
        f"name={event.name}",
        f"uuid={event.uuid[:8]}...",
        f"root_uuid={event.root_uuid[:8]}...",
        f"parent_uuid={event.parent_uuid[:8]}...",
    ]
    if event.scope_type is not None:
        parts.append(f"scope_type={event.scope_type}")
    if event.model_name is not None:
        parts.append(f"model={event.model_name}")
    if event.input is not None:
        preview = json.dumps(event.input, default=str)
        if len(preview) > 160:
            preview = preview[: 160 - 3] + "..."
        parts.append(f"input={preview}")
    if event.output is not None:
        preview = json.dumps(event.output, default=str)
        if len(preview) > 160:
            preview = preview[: 160 - 3] + "..."
        parts.append(f"output={preview}")
    print(" | ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def amain() -> None:
    if not os.environ.get("NVIDIA_API_KEY"):
        print("Error: set NVIDIA_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    # Register the console logging subscriber.
    nat_nexus.subscribers.register("console-logger", log_event)

    # Set up ATIF trajectory exporter.
    exporter = nat_nexus.AtifExporter(
        "example-session",
        "example-agent",
        "0.1.0",
        model_name="my-agent",
    )
    exporter.register("atif-exporter")

    # Push a top-level agent scope.
    agent_scope = nat_nexus.scope.push("example-agent", nat_nexus.ScopeType.Agent)
    agent_root_uuid = agent_scope.uuid

    # Create the LLM and agent.
    llm = ChatNVIDIA(model="nvidia/nemotron-3-nano-30b-a3b")
    agent = create_agent(llm, tools=[get_weather, get_population])

    # Run the agent.
    print("\n--- Agent execution ---\n")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the weather and population of San Francisco?"}]}
    )

    # Print the final response.
    print("\n--- Final response ---\n")
    print(result["messages"][-1].content)

    nat_nexus.scope.pop(agent_scope)

    # Export ATIF trajectory filtered to this agent's root scope.
    trajectory = exporter.export(agent_root_uuid)
    trajectory_path = "trajectory.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    print(f"\n--- ATIF trajectory written to {trajectory_path} ---\n")
    print(f"  schema: {trajectory.get('schema_version')}")
    print(f"  steps:  {len(trajectory.get('steps', []))}")

    # Clean up.
    exporter.deregister("atif-exporter")
    nat_nexus.subscribers.deregister("console-logger")


if __name__ == "__main__":
    asyncio.run(amain())
