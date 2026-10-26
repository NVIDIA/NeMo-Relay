# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional readers for configuration-driven, process-local diagnostics.

Readers expose restricted evidence to their caller. They do not enable collection,
install exporters, or confer authority on an external client. Returned dictionaries
are caller-owned; use snapshot_json for canonical wire serialization.
"""

from __future__ import annotations

import json
from typing import Any

from nemo_relay._diagnostics_query_contract import (
    MAX_QUERY_BYTES,
    MAX_QUERY_DEPTH,
    MAX_QUERY_FIELD_NAME_BYTES,
    MAX_QUERY_NODES,
    MAX_QUERY_OBJECT_FIELDS,
    NUMERIC_QUERY_FIELDS,
)
from nemo_relay._native import DiagnosticsReader as _NativeReader
from nemo_relay._native import diagnostics_reader as _reader


class DiagnosticsError(RuntimeError):
    """Reader failure with a stable machine-readable ``reason``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _utf8_size(value: str, remaining: int) -> int:
    # UTF-8 needs at least one byte per character. Bound the temporary copy.
    if len(value) > remaining:
        raise DiagnosticsError("invalid_query")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise DiagnosticsError("invalid_query") from None
    if size > remaining:
        raise DiagnosticsError("invalid_query")
    return size


def _query_json(query: dict[str, Any] | None) -> str:
    if query is None:
        return "{}"
    # Canonical queries encode u64 values as strings. Accept native Python ints
    # at these named positions without converting caller text or small row caps.
    admitted_bytes = 0
    nodes = 0

    def normalize(value: Any, key: str = "", depth: int = 0) -> Any:
        nonlocal admitted_bytes, nodes
        nodes += 1
        if nodes > MAX_QUERY_NODES or depth > MAX_QUERY_DEPTH or not isinstance(key, str):
            raise DiagnosticsError("invalid_query")
        admitted_bytes += _utf8_size(key, min(MAX_QUERY_FIELD_NAME_BYTES, MAX_QUERY_BYTES - admitted_bytes))
        if isinstance(value, str):
            admitted_bytes += _utf8_size(value, MAX_QUERY_BYTES - admitted_bytes)
        if isinstance(value, dict):
            if len(value) > MAX_QUERY_OBJECT_FIELDS:
                raise DiagnosticsError("invalid_query")
            return {k: normalize(v, k, depth + 1) for k, v in value.items()}
        if key in NUMERIC_QUERY_FIELDS and type(value) is int:
            if not 0 <= value <= 2**64 - 1:
                raise DiagnosticsError("invalid_query")
            return str(value)
        if value is not None and type(value) not in (str, int, bool):
            raise DiagnosticsError("invalid_query")
        return value

    chunks = []
    size = 0
    for chunk in json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).iterencode(normalize(query)):
        size += _utf8_size(chunk, MAX_QUERY_BYTES - size)
        chunks.append(chunk)
    return "".join(chunks)


class DiagnosticsReader:
    """Read-only service capability, independent of plugin activation lifetime."""

    __slots__ = ("_native",)

    def __init__(self, native: _NativeReader) -> None:
        self._native = native

    def snapshot(self, query: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return one bounded page; diagnostic counters and durations are int."""
        try:
            return self._native.snapshot(_query_json(query))
        except RuntimeError as exc:
            raise DiagnosticsError(getattr(exc, "reason", "unavailable")) from None

    def snapshot_json(self, query: dict[str, Any] | None = None) -> str:
        """Return canonical JSON with decimal-string diagnostic integers."""
        try:
            return self._native.snapshot_json(_query_json(query))
        except RuntimeError as exc:
            raise DiagnosticsError(getattr(exc, "reason", "unavailable")) from None

    def inspect_memory(self, query: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return one bounded memory page; diagnostic counters and durations are int."""
        try:
            return self._native.inspect_memory(_query_json(query))
        except RuntimeError as exc:
            raise DiagnosticsError(getattr(exc, "reason", "unavailable")) from None

    def inspect_memory_json(self, query: dict[str, Any] | None = None) -> str:
        """Return canonical JSON with decimal-string diagnostic integers."""
        try:
            return self._native.inspect_memory_json(_query_json(query))
        except RuntimeError as exc:
            raise DiagnosticsError(getattr(exc, "reason", "unavailable")) from None

    def info(self) -> dict[str, Any]:
        """Return separate core/copy budgets, occupancy, and logging coverage."""
        try:
            return self._native.info()
        except RuntimeError as exc:
            raise DiagnosticsError(getattr(exc, "reason", "unavailable")) from None


def reader() -> DiagnosticsReader:
    """Obtain a reader for the process diagnostics service.

    Returns:
        DiagnosticsReader: A reader for the service selected during plugin initialization.

    Raises:
        DiagnosticsError: If the service has not been initialized or this process
            inherited it across a fork. The ``reason`` attribute identifies the failure.
    """
    try:
        return DiagnosticsReader(_reader())
    except RuntimeError as exc:
        raise DiagnosticsError(getattr(exc, "reason", "unavailable")) from None


__all__ = ["DiagnosticsError", "DiagnosticsReader", "reader"]
