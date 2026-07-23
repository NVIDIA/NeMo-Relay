# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Payload sanitization shared by the Rampart worker callbacks."""

from __future__ import annotations

import re
from typing import cast

from nemo_relay_plugin import Json, LlmRequest

from ._detectors import (
    FAILURE_REPLACEMENT,
    MAX_CONTENT_CHARS,
    RampartSanitizer,
)

JsonObject = dict[str, Json]

_CONTENT_KEYS = frozenset(
    {
        "arguments",
        "content",
        "description",
        "input",
        "instructions",
        "output",
        "output_text",
        "refusal",
        "system",
        "text",
        "url",
    }
)
_STRUCTURAL_CONTENT_KEYS = frozenset(
    {
        "call_id",
        "finish_reason",
        "id",
        "model",
        "name",
        "role",
        "status",
        "tool_call_id",
        "type",
    }
)
_MAX_CONTENT_FIELDS = 64
_MAX_CONTENT_DEPTH = 64
_FIELD_CONTEXT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


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


def _content_footprint(
    value: Json,
    *,
    content: bool,
    preserve_structural_fields: bool,
    depth: int = 0,
) -> tuple[int, int]:
    if depth > _MAX_CONTENT_DEPTH:
        return _MAX_CONTENT_FIELDS + 1, MAX_CONTENT_CHARS + 1
    if isinstance(value, str):
        return (1, len(value)) if content else (0, 0)
    fields = 0
    chars = 0
    if isinstance(value, list):
        items = (
            _content_footprint(
                item,
                content=content and not isinstance(item, dict),
                preserve_structural_fields=preserve_structural_fields,
                depth=depth + 1,
            )
            for item in value
        )
    elif isinstance(value, dict):

        def dictionary_footprints():
            for key, item in value.items():
                if content and preserve_structural_fields and key in _STRUCTURAL_CONTENT_KEYS:
                    continue
                child_preserves_structure = preserve_structural_fields
                if _is_opaque_content_field(value, key):
                    child_preserves_structure = False
                yield _content_footprint(
                    item,
                    content=content or key in _CONTENT_KEYS,
                    preserve_structural_fields=child_preserves_structure,
                    depth=depth + 1,
                )

        items = dictionary_footprints()
    else:
        return 0, 0

    for item_fields, item_chars in items:
        fields += item_fields
        chars += item_chars
        if fields > _MAX_CONTENT_FIELDS or chars > MAX_CONTENT_CHARS:
            break
    return fields, chars


def _payload_is_bounded(
    value: Json,
    *,
    content: bool,
    preserve_structural_fields: bool,
) -> bool:
    fields, chars = _content_footprint(
        value,
        content=content,
        preserve_structural_fields=preserve_structural_fields,
    )
    return fields <= _MAX_CONTENT_FIELDS and chars <= MAX_CONTENT_CHARS


def _sanitize_content(
    value: Json,
    sanitizer: RampartSanitizer,
    *,
    preserve_structural_fields: bool = True,
) -> Json:
    if isinstance(value, str):
        return sanitizer.sanitize(value)
    if isinstance(value, list):
        return [
            _sanitize_content(
                item,
                sanitizer,
                preserve_structural_fields=preserve_structural_fields,
            )
            for item in value
        ]
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if preserve_structural_fields and key in _STRUCTURAL_CONTENT_KEYS:
                sanitized[key] = item
                continue
            if _is_opaque_content_field(value, key):
                sanitized[key] = _sanitize_tool_content(item, sanitizer)
                continue
            sanitized[key] = _sanitize_content(
                item,
                sanitizer,
                preserve_structural_fields=preserve_structural_fields,
            )
        return sanitized
    return value


def _sanitize_provider_payload(
    value: Json,
    sanitizer: RampartSanitizer,
    *,
    content: bool = False,
) -> Json:
    if isinstance(value, str):
        return sanitizer.sanitize(value) if content else value
    if isinstance(value, list):
        return [
            _sanitize_provider_payload(
                item,
                sanitizer,
                content=content and not isinstance(item, dict),
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    return {
        key: (
            _sanitize_tool_content(item, sanitizer)
            if _is_opaque_content_field(value, key)
            else _sanitize_content(item, sanitizer)
            if key in _CONTENT_KEYS
            else _sanitize_provider_payload(item, sanitizer, content=content)
        )
        for key, item in value.items()
    }


def _sanitize_tool_content(
    value: Json,
    sanitizer: RampartSanitizer,
    field_name: str | None = None,
) -> Json:
    if isinstance(value, str):
        if field_name is None or _FIELD_CONTEXT_PATTERN.fullmatch(field_name) is None:
            return sanitizer.sanitize(value)
        prefix = f"{field_name}: "
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
    "sanitize_llm_request",
    "sanitize_llm_response",
    "sanitize_tool_payload",
]
