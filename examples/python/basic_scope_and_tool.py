# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Python example: push a scope, run a tool, and log lifecycle events.

Run from the repository root:

    uv sync
    uv run python examples/python/basic_scope_and_tool.py
"""

from __future__ import annotations

import asyncio

import nat_nexus


async def main() -> None:
    nat_nexus.subscribers.register(
        "example_logger",
        lambda event: print(f"[{event.event_type}] {event.name}"),
    )

    async def search_tool(args: dict) -> dict:
        return {"results": [f"echo:{args['query']}"]}

    with nat_nexus.scope.scope("example-agent", nat_nexus.ScopeType.Agent):
        result = await nat_nexus.tools.execute(
            "search",
            {"query": "hello"},
            search_tool,
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
