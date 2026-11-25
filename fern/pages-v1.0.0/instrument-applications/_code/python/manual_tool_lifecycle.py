# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo_relay


def run_example(query: str = "weather") -> dict[str, int]:
    """Complete the manual tool lifecycle and preserve operation errors."""

    def search(query: str) -> dict[str, int]:
        if not query:
            raise ValueError("query is required")
        return {"hits": 2}

    handle = nemo_relay.tools.call("search", {"query": query}, data={"attempt": 1})
    try:
        result = search(query)
    except Exception as execution_error:
        try:
            nemo_relay.tools.call_end(
                handle,
                nemo_relay.ToolExecutionResult({"error": str(execution_error)}),
            )
        except Exception as completion_error:
            execution_error.add_note(f"Relay could not record tool completion: {completion_error}")
        raise
    else:
        nemo_relay.tools.call_end(handle, nemo_relay.ToolExecutionResult(result))
    return result
