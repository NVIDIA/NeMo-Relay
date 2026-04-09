# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Python example: push a scope, run a tool, and log lifecycle events.

Run from the repository root:

    uv sync
    uv run python examples/python/basic_scope_and_tool.py
"""

from __future__ import annotations

import asyncio

import nemo_flow


async def main() -> None:
    nemo_flow.subscribers.register(
        "example_logger",
        lambda event: print(f"[{event.kind}] {event.name}"),
    )

    async def search_tool(args: dict) -> dict:
        return {"results": [f"echo:{args['query']}"]}

    with nemo_flow.scope.scope("example-agent", nemo_flow.ScopeType.Agent):
        result = await nemo_flow.tools.execute(
            "search",
            {"query": "hello"},
            search_tool,
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
