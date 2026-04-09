# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Python example: export an ATIF trajectory for one tool call.

Run from the repository root:

    uv sync
    uv run python examples/python/atif_export.py
"""

from __future__ import annotations

import asyncio
import json

import nemo_flow


async def main() -> None:
    exporter = nemo_flow.AtifExporter(
        session_id="example-session",
        agent_name="example-agent",
        agent_version="1.0",
        model_name="demo-model",
    )
    exporter.register("atif_example")

    async def tool(args: dict) -> dict:
        return {"results": [args["query"]]}

    with nemo_flow.scope.scope("example-agent", nemo_flow.ScopeType.Agent):
        await nemo_flow.tools.execute("search", {"query": "hello"}, tool)

    trajectory = exporter.export_json()
    print(json.dumps(json.loads(trajectory), indent=2))
    exporter.deregister("atif_example")


if __name__ == "__main__":
    asyncio.run(main())
