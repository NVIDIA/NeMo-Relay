# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared JSON format for tool results."""

from nemo_relay import Json

TOOL_RESULT_SCHEMA = "nemo.relay.tool_result@1"


def projected_tool_result(value: Json) -> tuple[Json, bool] | None:
    """Read content and failure status from a valid tool-result wrapper."""
    if not isinstance(value, dict) or value.get("schema") != TOOL_RESULT_SCHEMA:
        return None
    if "content" not in value or "result" not in value:
        return None
    content, is_error = value["content"], value.get("is_error")
    if not (content is None or isinstance(content, str | list)) or not isinstance(is_error, bool):
        return None
    return content, is_error
