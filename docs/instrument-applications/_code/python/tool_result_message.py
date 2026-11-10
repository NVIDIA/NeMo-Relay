# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import nemo_relay


async def append_tool_result(
    tool_name, tool_arguments, tool_callback, tool_call, messages
) -> nemo_relay.ToolExecutionResult[nemo_relay.Json]:
    """Append only the tool business payload to the model messages."""
    tool_result = await nemo_relay.tools.execute(tool_name, tool_arguments, tool_callback)
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(tool_result.result),
            "tool_call_id": tool_call["id"],
        }
    )
    return tool_result
