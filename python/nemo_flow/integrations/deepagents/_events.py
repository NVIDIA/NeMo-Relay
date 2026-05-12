# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Deep Agents observability helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import nemo_flow

_logger = logging.getLogger(__name__)

SYNC_SUBAGENT_TOOLS = frozenset({"task"})
ASYNC_SUBAGENT_TOOLS = frozenset(
    {
        "start_async_task",
        "check_async_task",
        "update_async_task",
        "cancel_async_task",
        "list_async_tasks",
    }
)
# Mirrors the Deep Agents built-in filesystem tools listed in the backend docs:
# https://docs.langchain.com/oss/python/deepagents/backends
FILESYSTEM_TOOLS = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})
FILESYSTEM_BACKEND_METHODS = frozenset({"ls", "read", "write", "edit", "glob", "grep"})
# Deep Agents sandbox backends expose execute()/aexecute(); the tool name is execute.
SANDBOX_TOOLS = frozenset({"execute"})


def tool_kind(tool_name: str) -> str | None:
    """Return the Deep Agents semantic category for a known built-in tool."""
    if tool_name in SYNC_SUBAGENT_TOOLS:
        return "subagent"
    if tool_name in ASYNC_SUBAGENT_TOOLS:
        return "async_subagent"
    if tool_name in SANDBOX_TOOLS:
        return "sandbox"
    if tool_name in FILESYSTEM_TOOLS:
        return "filesystem"
    return None


def backend_kind(method_name: str) -> str:
    """Return the Deep Agents semantic category for a backend method."""
    normalized = method_name.removeprefix("a")
    if normalized in SANDBOX_TOOLS:
        return "sandbox"
    if normalized in FILESYSTEM_TOOLS or normalized in FILESYSTEM_BACKEND_METHODS:
        return "filesystem"
    if normalized in {"upload_files", "download_file", "download_files"}:
        return "filesystem"
    return "backend"


def mark_base_name(kind: str) -> str:
    """Return a stable mark-event base name for a Deep Agents category."""
    return {
        "subagent": "DeepAgents Subagent",
        "async_subagent": "DeepAgents Async Subagent",
        "human_in_the_loop": "DeepAgents Human In The Loop",
        "skill": "DeepAgents Skills",
        "sandbox": "DeepAgents Sandbox",
        "filesystem": "DeepAgents Filesystem",
        "backend": "DeepAgents Backend",
    }.get(kind, "DeepAgents")


def json_safe(value: Any) -> nemo_flow.Json:
    """Return a conservative JSON-compatible value."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return f"<{type(value).__name__}: {len(value)} bytes>"
    return repr(value)


def summarize_value(value: Any) -> nemo_flow.Json:
    """Return a JSON-compatible representation for observability payloads."""
    if value is None or isinstance(value, int | float | bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | bytearray):
        return f"<{type(value).__name__}: {len(value)} bytes>"
    if isinstance(value, Mapping):
        return {str(key): summarize_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [summarize_value(item) for item in value]

    content = getattr(value, "content", None)
    if isinstance(content, str):
        summary: dict[str, nemo_flow.Json] = {"content": content}
        for attr in ("name", "id", "tool_call_id"):
            attr_value = getattr(value, attr, None)
            if attr_value is not None:
                summary[attr] = summarize_value(attr_value)
        return summary

    return repr(value)


def tool_event_data(
    tool_name: str,
    args: Mapping[str, Any],
    *,
    result: Any = None,
    error: BaseException | None = None,
) -> dict[str, nemo_flow.Json]:
    """Build a stable Deep Agents tool event payload."""
    data: dict[str, nemo_flow.Json] = {
        "tool_name": tool_name,
        "args": summarize_value(args),
    }

    for key in ("name", "agent_name", "task_id", "thread_id", "run_id", "graph_id", "status"):
        value = args.get(key)
        if value is not None:
            data[key] = summarize_value(value)

    for key in ("path", "file_path", "pattern", "glob", "command"):
        value = args.get(key)
        if value is not None:
            data[key] = summarize_value(value)

    if result is not None:
        data["result"] = summarize_value(result)
    if error is not None:
        data["error"] = repr(error)

    return data


def backend_event_data(
    backend_name: str,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    result: Any = None,
    error: BaseException | None = None,
) -> dict[str, nemo_flow.Json]:
    """Build a stable Deep Agents backend event payload."""
    data: dict[str, nemo_flow.Json] = {
        "backend": backend_name,
        "method": method_name,
        "args": summarize_value(args),
        "kwargs": summarize_value(kwargs),
    }
    if result is not None:
        data["result"] = summarize_value(result)
    if error is not None:
        data["error"] = repr(error)
    return data


def emit_mark(
    base_name: str,
    kind: str,
    phase: str,
    data: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit a Deep Agents mark event without changing framework behavior."""
    event_metadata: dict[str, Any] = {
        "integration": "deepagents",
        "deepagents_kind": kind,
        "phase": phase,
    }
    if metadata:
        event_metadata.update(metadata)

    try:
        nemo_flow.scope.event(
            f"{base_name} {phase.title()}",
            data=json_safe(data),
            metadata=json_safe(event_metadata),
        )
    except Exception:
        _logger.debug("NeMo Flow: Deep Agents mark emission failed", exc_info=True)
