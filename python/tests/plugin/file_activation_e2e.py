# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise file-backed activation with a lifecycle-managed Python worker."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from nemo_relay import plugin, tools


async def run(plugin_config_path: Path) -> None:
    """Load the prepared file, exercise its worker, and close its owner."""
    activation = await plugin.initialize_from_plugins_toml(plugin_config_path=plugin_config_path)
    try:
        assert activation.is_active
        observed: dict[str, Any] = {}

        async def local_tool(args: Any) -> dict[str, Any]:
            observed["args"] = args
            return {"python_tool_executed": True, "args": args}

        result = await tools.execute(
            "file-backed-python-worker",
            {"query": "relay"},
            local_tool,
        )
        expected_args = {
            "query": "relay",
            "_nemo_relay_plugin": {"tag": "python_grpc_worker"},
        }
        assert observed == {"args": expected_args}, observed
        assert result == {"python_tool_executed": True, "args": expected_args}, result
    finally:
        await activation.close()

    assert not activation.is_active
    after_close = await tools.execute(
        "file-backed-python-worker-after-close",
        {"query": "relay"},
        lambda args: {"args": args},
    )
    assert after_close == {"args": {"query": "relay"}}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} PLUGINS_TOML")
    asyncio.run(run(Path(sys.argv[1])))
