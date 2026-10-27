# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON helpers shared by integration codecs."""

from __future__ import annotations

import base64
import json
import math
import re
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias, TypeGuard, cast

from pydantic import TypeAdapter, ValidationError
from pydantic.errors import PydanticUserError
from pydantic_core import PydanticSerializationError, to_jsonable_python

if TYPE_CHECKING:
    from nemo_relay import Json, JsonObject, JsonPrimitive

_JsonData: TypeAlias = "Json | Sequence[Mapping[str, Json]] | Mapping[str, Json]"

_JSON_INTS: Final[Annotated[range, "Integers Relay carries as JSON numbers."]] = range(-(2**63), 2**64)
SCALAR_KINDS: Final = frozenset({"json", "float", "int", "str", "bytes", "bytearray"})
_NON_FINITE: Final[Annotated[frozenset[str], "JSON names for NaN and infinity."]] = frozenset(
    {"NaN", "Infinity", "-Infinity"}
)
_SURROGATES: Final = re.compile(r"[\ud800-\udfff]")
_BINARY: Final[Mapping[str, Callable[[bytes], object]]] = {"bytes": bytes, "bytearray": bytearray}
_PYTHON_VALUES: Final = TypeAdapter(object)


def canonical(value: _JsonData) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def same_json(left: _JsonData, right: _JsonData) -> bool:
    """Return whether two values have same JSON text. Unlike `==`, this tells 1, 1.0, and true apart."""
    return canonical(left) == canonical(right)


def paired_items(edited: Sequence[Json], projected: Sequence[Json]) -> list[int | None]:
    """Return position of original item each edited item matches, or None.

    Equal items match in order. Other items match by position between those matches.
    """
    positions: dict[str, list[int]] = {}
    for position, item in enumerate(projected):
        positions.setdefault(canonical(item), []).append(position)
    matches: list[tuple[int, int]] = []
    cursor = 0
    for index, item in enumerate(edited):
        candidates = positions.get(canonical(item), [])
        slot = bisect_left(candidates, cursor)
        if slot < len(candidates):
            matches.append((index, candidates[slot]))
            cursor = candidates[slot] + 1
    paired: list[int | None] = [None] * len(edited)
    for (edited_start, original_start), (edited_end, original_end) in pairwise(
        [(-1, -1), *matches, (len(edited), len(projected))]
    ):
        gap = range(original_start + 1, original_end)
        for offset, index in enumerate(range(edited_start + 1, edited_end)):
            if offset < len(gap):
                paired[index] = gap[offset]
        if edited_end < len(edited):
            paired[edited_end] = original_end
    return paired


def _is_text(value: str) -> bool:
    """Return whether text has no lone surrogate."""
    return value.isascii() or _SURROGATES.search(value) is None


def is_json_key(key: object) -> TypeGuard[str]:
    """Return whether key is text Relay can carry unchanged."""
    return type(key) is str and _is_text(key)


def scalar_json(value: object) -> JsonObject | None:
    """Encode a number, text, or binary value. Return None for any other value."""
    if value is None or isinstance(value, bool):
        return {"kind": "json", "value": value}
    if type(value) is int:
        return {"kind": "json", "value": value} if value in _JSON_INTS else {"kind": "int", "value": str(value)}
    if type(value) is float:
        if math.isfinite(value):
            return {"kind": "json", "value": value}
        return {"kind": "float", "value": json.dumps(value)}
    if type(value) is str:
        if _is_text(value):
            return {"kind": "json", "value": value}
        return {"kind": "str", "value": value.encode("unicode_escape").decode("ascii")}
    if type(value) is bytes or type(value) is bytearray:
        return {"kind": type(value).__name__, "value": base64.b64encode(value).decode("ascii")}
    return None


def scalar_value(kind: str, value: Json) -> object:
    """Decode a scalar. Raise `ValueError` when value does not fit its kind."""
    if kind == "json" and (value is None or isinstance(value, str | int | float | bool)):
        return value
    if kind == "float" and isinstance(value, str) and value in _NON_FINITE:
        return float(value)
    if kind == "int" and isinstance(value, str):
        return int(value)
    if kind == "str" and isinstance(value, str):
        return value.encode("ascii").decode("unicode_escape")
    if kind in _BINARY and isinstance(value, str):
        return _BINARY[kind](base64.b64decode(value, validate=True))
    raise ValueError(f"Invalid encoded {kind} value")


def _relay_text(value: str) -> str:
    """Replace each lone surrogate with U+FFFD."""
    return value if _is_text(value) else _SURROGATES.sub("�", value)


def relay_scalar(value: JsonPrimitive) -> JsonPrimitive:
    """Return scalar in a form Relay can carry."""
    if isinstance(value, str):
        return _relay_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, int) and value not in _JSON_INTS:
        return str(value)
    return value


def _is_relay_json(value: Json) -> bool:
    """Return whether Relay can carry JSON unchanged."""
    pending: list[Json] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if not all(_is_text(key) for key in item):
                return False
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str):
            if not _is_text(item):
                return False
        elif isinstance(item, float):
            if not math.isfinite(item):
                return False
        elif isinstance(item, int) and item not in _JSON_INTS:
            return False
    return True


def _dumpable(item: object, path: set[int]) -> bool:
    if isinstance(item, str | bytes | bytearray):
        return True
    if isinstance(item, Mapping):
        entries = cast(Mapping[object, object], item)
        parts: Iterable[object] = (*entries, *entries.values())
    elif isinstance(item, Sequence):
        parts = cast(Sequence[object], item)
    else:
        return not isinstance(item, Iterable)
    if id(item) in path:
        return False
    path.add(id(item))
    try:
        return all(_dumpable(part, path) for part in parts)
    finally:
        path.discard(id(item))


def typed_json(value: object) -> Json | None:
    """Return JSON that rebuilds an equal object of the same type. None when none exists or Relay cannot carry it."""
    try:
        if not _dumpable(_PYTHON_VALUES.dump_python(value, mode="python", round_trip=True), set()):
            return None
        dumped: Json = to_jsonable_python(value, round_trip=True)
        rebuilt = TypeAdapter(type(value)).validate_python(dumped)
    except (PydanticSerializationError, PydanticUserError, ValidationError):
        return None
    if type(rebuilt) is not type(value) or rebuilt != value:
        return None
    return dumped if _is_relay_json(dumped) else None
