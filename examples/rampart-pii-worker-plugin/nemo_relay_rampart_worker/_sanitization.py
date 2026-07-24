# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Payload sanitization shared by the Rampart worker callbacks."""

from __future__ import annotations

import json
import re
from typing import cast

from nemo_relay_plugin import Json, LlmRequest

from ._detectors import (
    FAILURE_REPLACEMENT,
    RampartSanitizer,
)

JsonObject = dict[str, Json]

BINARY_CONTENT_REPLACEMENT = "[REDACTED:BINARY_CONTENT]"
_CONTENT_KEYS = frozenset(
    {
        "arguments",
        "completion",
        "content",
        "description",
        "input",
        "instructions",
        "message",
        "name",
        "output",
        "output_text",
        "prompt",
        "query",
        "refusal",
        "summary",
        "system",
        "text",
        "url",
    }
)
_BINARY_DATA_KEYS = frozenset(
    {
        "audio_data",
        "data",
        "file_data",
        "image_data",
    }
)
_STRUCTURAL_CONTENT_KEYS = frozenset(
    {
        "call_id",
        "finish_reason",
        "id",
        "model",
        "role",
        "status",
        "tool_call_id",
        "type",
    }
)
_MAX_CONTENT_FIELDS = 64
_MAX_CONTENT_DEPTH = 64
_MAX_PAYLOAD_NODES = 4_096
_FIELD_CONTEXT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_DATA_URI_PATTERN = re.compile(r"^data:[^,\s]{0,128};base64,", re.IGNORECASE)
_BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/_-]+={0,2}")


def _reject_json_constant(value: str) -> Json:
    raise ValueError(f"non-standard JSON constant {value}")


def _is_binary_content(value: str, field_name: str | None) -> bool:
    if _DATA_URI_PATTERN.match(value):
        return True
    return (
        field_name in _BINARY_DATA_KEYS
        and len(value) >= 64
        and len(value) % 4 != 1
        and _BASE64_PATTERN.fullmatch(value) is not None
    )


def _is_opaque_content_field(parent: JsonObject, key: str) -> bool:
    if key == "arguments":
        return True
    item_type = parent.get("type")
    if not isinstance(item_type, str):
        return False
    if key == "input":
        return item_type == "tool_use" or item_type.endswith("_call")
    if key == "output":
        return item_type == "tool_result" or item_type.endswith("_call_output")
    return False


def _preserves_tool_or_function_name(
    parent: JsonObject,
    key: str,
    container: str | None,
) -> bool:
    if key != "name":
        return False
    if container in {"function", "tools"}:
        return True
    item_type = parent.get("type")
    return isinstance(item_type, str) and item_type in {
        "function",
        "function_call",
        "tool_use",
    }


def _content_footprint(
    value: Json,
    *,
    content: bool,
    max_content_chars: int,
    preserve_structural_fields: bool,
    container: str | None = None,
    depth: int = 0,
) -> tuple[int, int, int]:
    if depth > _MAX_CONTENT_DEPTH:
        return _MAX_CONTENT_FIELDS + 1, max_content_chars + 1, _MAX_PAYLOAD_NODES + 1
    if isinstance(value, str):
        if content and _is_binary_content(value, container):
            return 1, 0, 1
        fields, chars = (1, len(value)) if content else (0, 0)
        return fields, chars, 1
    fields = 0
    chars = 0
    nodes = 1
    if isinstance(value, list):
        items = (
            _content_footprint(
                item,
                content=content,
                max_content_chars=max_content_chars,
                preserve_structural_fields=preserve_structural_fields,
                container=container,
                depth=depth + 1,
            )
            for item in value
        )
    elif isinstance(value, dict):

        def dictionary_footprints():
            for key, item in value.items():
                preserve_field = preserve_structural_fields and (
                    key in _STRUCTURAL_CONTENT_KEYS or _preserves_tool_or_function_name(value, key, container)
                )
                if content and preserve_field:
                    yield _content_footprint(
                        item,
                        content=False,
                        max_content_chars=max_content_chars,
                        preserve_structural_fields=preserve_structural_fields,
                        container=key,
                        depth=depth + 1,
                    )
                    continue
                opaque_content = _is_opaque_content_field(value, key)
                yield _content_footprint(
                    item,
                    content=content or key in _CONTENT_KEYS,
                    max_content_chars=max_content_chars,
                    preserve_structural_fields=False if opaque_content else preserve_structural_fields,
                    container=key,
                    depth=depth + 1,
                )

        items = dictionary_footprints()
    else:
        return 0, 0, 1

    for item_fields, item_chars, item_nodes in items:
        fields += item_fields
        chars += item_chars
        nodes += item_nodes
        if fields > _MAX_CONTENT_FIELDS or chars > max_content_chars or nodes > _MAX_PAYLOAD_NODES:
            break
    return fields, chars, nodes


def _payload_is_bounded(
    value: Json,
    *,
    content: bool,
    max_content_chars: int,
    preserve_structural_fields: bool,
) -> bool:
    fields, chars, nodes = _content_footprint(
        value,
        content=content,
        max_content_chars=max_content_chars,
        preserve_structural_fields=preserve_structural_fields,
    )
    return fields <= _MAX_CONTENT_FIELDS and chars <= max_content_chars and nodes <= _MAX_PAYLOAD_NODES


def _sanitize_content(
    value: Json,
    sanitizer: RampartSanitizer,
    *,
    preserve_structural_fields: bool = True,
    container: str | None = None,
) -> Json:
    if isinstance(value, str):
        if _is_binary_content(value, container):
            return BINARY_CONTENT_REPLACEMENT
        return sanitizer.sanitize(value)
    if isinstance(value, list):
        return [
            _sanitize_content(
                item,
                sanitizer,
                preserve_structural_fields=preserve_structural_fields,
                container=container,
            )
            for item in value
        ]
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if preserve_structural_fields and (
                key in _STRUCTURAL_CONTENT_KEYS or _preserves_tool_or_function_name(value, key, container)
            ):
                sanitized[key] = item
                continue
            if _is_opaque_content_field(value, key):
                sanitized[key] = _sanitize_tool_content(item, sanitizer, key)
                continue
            sanitized[key] = _sanitize_content(
                item,
                sanitizer,
                preserve_structural_fields=preserve_structural_fields,
                container=key,
            )
        return sanitized
    return value


def _sanitize_provider_payload(
    value: Json,
    sanitizer: RampartSanitizer,
    *,
    content: bool = False,
    container: str | None = None,
) -> Json:
    if isinstance(value, str):
        if not content:
            return value
        if _is_binary_content(value, container):
            return BINARY_CONTENT_REPLACEMENT
        return sanitizer.sanitize(value)
    if isinstance(value, list):
        return [
            _sanitize_provider_payload(
                item,
                sanitizer,
                content=content,
                container=container,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    sanitized = {}
    for key, item in value.items():
        if key in _STRUCTURAL_CONTENT_KEYS or _preserves_tool_or_function_name(value, key, container):
            sanitized[key] = item
        elif _is_opaque_content_field(value, key):
            sanitized[key] = _sanitize_tool_content(item, sanitizer, key)
        elif key in _CONTENT_KEYS:
            sanitized[key] = _sanitize_content(item, sanitizer, container=key)
        else:
            sanitized[key] = _sanitize_provider_payload(
                item,
                sanitizer,
                content=content,
                container=key,
            )
    return sanitized


def _sanitize_tool_content(
    value: Json,
    sanitizer: RampartSanitizer,
    field_name: str | None = None,
) -> Json:
    if isinstance(value, str):
        if _is_binary_content(value, field_name):
            return BINARY_CONTENT_REPLACEMENT
        if field_name == "arguments":
            try:
                parsed = json.loads(
                    value,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, RecursionError, ValueError):
                pass
            else:
                if not _payload_is_bounded(
                    parsed,
                    content=True,
                    max_content_chars=sanitizer.max_content_chars,
                    preserve_structural_fields=False,
                ):
                    return FAILURE_REPLACEMENT
                return json.dumps(
                    _sanitize_tool_content(parsed, sanitizer),
                    separators=(",", ":"),
                    allow_nan=False,
                )
        if field_name is None or _FIELD_CONTEXT_PATTERN.fullmatch(field_name) is None:
            return sanitizer.sanitize(value)
        prefix = f"{field_name}: "
        if len(prefix) + len(value) > sanitizer.max_content_chars:
            return sanitizer.sanitize(value)
        sanitized = sanitizer.sanitize(f"{prefix}{value}")
        if sanitized == FAILURE_REPLACEMENT:
            return sanitized
        if not sanitized.startswith(prefix):
            return FAILURE_REPLACEMENT
        return sanitized[len(prefix) :]
    if isinstance(value, list):
        return [_sanitize_tool_content(item, sanitizer, field_name) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_tool_content(item, sanitizer, key) for key, item in value.items()}
    return value


def _sanitize_tool_payload(value: Json, sanitizer: RampartSanitizer) -> Json:
    if not _payload_is_bounded(
        value,
        content=True,
        max_content_chars=sanitizer.max_content_chars,
        preserve_structural_fields=False,
    ):
        return FAILURE_REPLACEMENT
    try:
        return _sanitize_tool_content(value, sanitizer)
    except Exception:
        return FAILURE_REPLACEMENT


def _sanitize_llm_payload(value: Json, sanitizer: RampartSanitizer) -> Json:
    root_is_content = isinstance(value, (str, list))
    if not _payload_is_bounded(
        value,
        content=root_is_content,
        max_content_chars=sanitizer.max_content_chars,
        preserve_structural_fields=True,
    ):
        return FAILURE_REPLACEMENT
    try:
        return _sanitize_provider_payload(
            value,
            sanitizer,
            content=root_is_content,
        )
    except Exception:
        return FAILURE_REPLACEMENT


def sanitize_llm_request(request: LlmRequest, sanitizer: RampartSanitizer) -> LlmRequest:
    """Return a request copy with only semantic content fields sanitized."""
    sanitized = request.copy()
    sanitized["headers"] = {}
    content = sanitized.get("content")
    if content is not None:
        sanitized["content"] = _sanitize_llm_payload(cast(Json, content), sanitizer)
    return sanitized


def sanitize_llm_response(response: Json, sanitizer: RampartSanitizer) -> Json:
    """Return an observability-safe LLM response copy."""
    return _sanitize_llm_payload(response, sanitizer)


def sanitize_tool_payload(payload: Json, sanitizer: RampartSanitizer) -> Json:
    """Return an observability-safe tool payload copy."""
    return _sanitize_tool_payload(payload, sanitizer)


__all__ = [
    "BINARY_CONTENT_REPLACEMENT",
    "sanitize_llm_request",
    "sanitize_llm_response",
    "sanitize_tool_payload",
]
