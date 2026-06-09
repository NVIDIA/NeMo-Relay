# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private helpers for normalizing config helper dataclasses to JSON-like values."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Protocol, cast


class SupportsToDict(Protocol):
    """Private protocol for helper objects that provide ``to_dict()``."""

    def to_dict(self) -> dict[str, Any]: ...


def normalize(value: object) -> Any:
    """Recursively normalize dataclasses, lists, and dicts into JSON-like values."""
    if hasattr(value, "to_dict"):
        return cast(SupportsToDict, value).to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: normalize(field_value)
            for field_info in fields(value)
            if (field_value := getattr(value, field_info.name)) is not None
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {cast(str, key): normalize(val) for key, val in value.items() if val is not None}
    return value


def normalize_object(value: object) -> dict[str, Any]:
    """Normalize a helper value and assert the result is mapping-shaped."""
    return cast(dict[str, Any], normalize(value))
