# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo_relay


def require_query(tool_name: str, tool_arguments: nemo_relay.Json) -> str | None:
    if tool_name != "search":
        return None
    if not tool_arguments.get("query"):
        return "search.query is required"
    return None


def register_query_policy() -> None:
    nemo_relay.guardrails.register_tool_conditional_execution("search.require_query", 20, require_query)
