# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use declared response schemas to encode and restore structured answers."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any, cast

from pydantic import BaseModel, JsonValue, TypeAdapter

from nemo_relay import Json
from nemo_relay.typed import Codec


class LangChainStructuredResponseCodec(Codec[object]):
    """Validate declared responses with the same schema API as LangChain."""

    def __init__(self, schemas: Iterable[Any]) -> None:
        self._schemas = [
            (schema, TypeAdapter[Any](schema))
            for schema in schemas
            if isinstance(schema, type) and (issubclass(schema, BaseModel) or dataclasses.is_dataclass(schema))
        ]
        self._json = TypeAdapter(JsonValue)

    def to_json(self, value: object) -> Json:
        """Store the schema position so overlapping union members stay distinct."""
        for index, (schema, adapter) in enumerate(self._schemas):
            if type(value) is schema:
                return {
                    "schema_index": index,
                    "data": cast(
                        Json, adapter.dump_python(value, mode="json", warnings="error", by_alias=False, round_trip=True)
                    ),
                }
        return {"schema_index": None, "data": cast(Json, self._json.validate_python(value, strict=True))}

    def from_json(self, data: Json) -> object:
        """Restore only types declared by this request's response format."""
        if not isinstance(data, dict) or set(data) != {"schema_index", "data"}:
            raise ValueError("Invalid structured response payload")
        index = data["schema_index"]
        if index is None:
            return self._json.validate_python(data["data"], strict=True)
        if type(index) is not int or not 0 <= index < len(self._schemas):
            raise ValueError("Invalid structured response schema index")
        return self._schemas[index][1].validate_python(data["data"], by_alias=False, by_name=True)
