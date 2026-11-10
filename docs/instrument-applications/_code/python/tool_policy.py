# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Awaitable, Callable

import nemo_relay


def redact_api_key(_tool_name: str, tool_arguments: nemo_relay.Json) -> nemo_relay.Json:
    safe_arguments = dict(tool_arguments)
    if "api_key" in safe_arguments:
        safe_arguments["api_key"] = "<redacted>"
    return safe_arguments


def require_query(_tool_name: str, tool_arguments: nemo_relay.Json) -> str | None:
    if not tool_arguments.get("query"):
        return "search.query is required"
    return None


async def measure_tool(
    context: nemo_relay.ToolExecutionContext,
    next_call: Callable[[nemo_relay.Json], Awaitable[nemo_relay.ToolExecutionResult[nemo_relay.Json]]],
) -> nemo_relay.ToolExecutionInterceptOutcome:
    started = time.perf_counter()
    try:
        execution_result = await next_call(context.args)
        return nemo_relay.ToolExecutionInterceptOutcome(
            execution_result.result,
            annotation=execution_result.annotation,
        )
    finally:
        elapsed_milliseconds = round((time.perf_counter() - started) * 1000, 2)
        print(f"{context.tool_name} completed in {elapsed_milliseconds} ms")


def register_tool_policy() -> None:
    nemo_relay.guardrails.register_tool_sanitize_request("search.redact_api_key", 10, redact_api_key)
    nemo_relay.guardrails.register_tool_conditional_execution("search.require_query", 20, require_query)
    nemo_relay.intercepts.register_tool_execution("search.measure", 30, measure_tool)
