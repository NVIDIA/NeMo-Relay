# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check strict response decoding for declared LangChain schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest


@pytest.mark.parametrize("kind", ["pydantic", "dataclass"])
def test_declared_response_round_trip_and_invalid_data(kind):
    from pydantic import BaseModel, ValidationError

    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    class Model(BaseModel):
        value: int

    @dataclass
    class Data:
        value: int

    schema = Model if kind == "pydantic" else Data
    codec = LangChainStructuredResponseCodec([Model, Data])
    encoded = cast(dict[str, Any], codec.to_json(schema(value=42)))
    assert codec.from_json(encoded) == schema(value=42)
    encoded["data"] = {}
    with pytest.raises(ValidationError):
        codec.from_json(encoded)


def test_declared_pydantic_encoding_does_not_fall_back():
    from pydantic import BaseModel, field_serializer
    from pydantic_core import PydanticSerializationError

    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    class Answer(BaseModel):
        value: str

        @field_serializer("value")
        def broken_dump(self, _value):
            raise ValueError("cannot encode declared answer")

    with pytest.raises(PydanticSerializationError, match="cannot encode declared answer"):
        LangChainStructuredResponseCodec([Answer]).to_json(Answer(value="answer"))


def test_undeclared_python_objects_are_rejected():
    from pydantic import BaseModel, ValidationError

    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    class Declared(BaseModel):
        value: str

    class Unknown(BaseModel):
        count: int

    @dataclass
    class UnknownData:
        count: int

    codec = LangChainStructuredResponseCodec([Declared])
    for value in (Unknown(count=42), UnknownData(count=42), object()):
        with pytest.raises(ValidationError):
            codec.to_json(value)


@pytest.mark.parametrize("value", [None, True, 42, 1.5, "answer", [1, "two"], {"answer": [42]}])
def test_json_values_pass_through_without_a_schema(value):
    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    codec = LangChainStructuredResponseCodec([])
    assert codec.from_json(codec.to_json(value)) == value


def test_json_fields_do_not_trigger_object_reconstruction():
    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    codec = LangChainStructuredResponseCodec([])
    for value in (
        {"schema_index": 0, "data": {"value": "plain JSON"}},
        {"__nv_pickle__": True, "data": "not a pickle"},
        {"schema": "nemo.relay.tool_result@1", "content": "answer", "is_error": False, "result": "answer"},
    ):
        assert codec.from_json(codec.to_json(value)) == value


def test_overlapping_schemas_keep_the_exact_type():
    from dataclasses import make_dataclass

    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    first = make_dataclass("Answer", [("value", int)], namespace={"__module__": __name__})
    second = make_dataclass("Answer", [("value", int)], namespace={"__module__": __name__})
    for schema in (first, second):
        encoded = LangChainStructuredResponseCodec([first, second]).to_json(schema(value=42))
        decoded = LangChainStructuredResponseCodec([first, second]).from_json(encoded)
        assert type(decoded) is schema
        assert decoded == schema(value=42)


@pytest.mark.parametrize("index", [-1, 1, True, "0", "os.system"])
def test_invalid_schema_index_is_rejected(index):
    from pydantic import BaseModel

    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    class Answer(BaseModel):
        value: int

    codec = LangChainStructuredResponseCodec([Answer])
    with pytest.raises(ValueError, match="schema index"):
        codec.from_json({"schema_index": index, "data": {"value": 42}})


def test_nested_dataclass_fields_are_restored():
    from dataclasses import make_dataclass

    from pydantic import BaseModel

    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    class Inner(BaseModel):
        value: int

    Answer = make_dataclass("Answer", [("inner", Inner)])
    codec = LangChainStructuredResponseCodec([Answer])
    decoded = codec.from_json(codec.to_json(Answer(inner=Inner(value=42))))
    assert isinstance(decoded, Answer)
    assert decoded == Answer(inner=Inner(value=42))


@pytest.mark.parametrize("nested_dataclass", [False, True])
@pytest.mark.parametrize("kind", ["alias", "json", "separate_aliases", "distinct_aliases", "alias_path"])
def test_validation_compatible_serialization_round_trips(kind, nested_dataclass):
    from dataclasses import make_dataclass

    from pydantic import AliasChoices, AliasPath, BaseModel, Field, Json, TypeAdapter

    from nemo_relay.integrations.langchain._structured_codec import LangChainStructuredResponseCodec

    class AliasedAnswer(BaseModel):
        value: int = Field(alias="answer")

    class JsonAnswer(BaseModel):
        items: Json[list[int]]

    class SeparateAliasesAnswer(BaseModel):
        value: int = Field(validation_alias=AliasChoices("answer", "output"), serialization_alias="output")

    class DistinctAliasesAnswer(BaseModel):
        value: int = Field(validation_alias="answer", serialization_alias="output")

    class AliasPathAnswer(BaseModel):
        value: int = Field(validation_alias=AliasPath("input", "answer"), serialization_alias="output")

    schema, data = {
        "alias": (AliasedAnswer, {"answer": 42}),
        "json": (JsonAnswer, {"items": "[1,2]"}),
        "separate_aliases": (SeparateAliasesAnswer, {"answer": 42}),
        "distinct_aliases": (DistinctAliasesAnswer, {"answer": 42}),
        "alias_path": (AliasPathAnswer, {"input": {"answer": 42}}),
    }[kind]
    value = TypeAdapter(schema).validate_python(data)
    if nested_dataclass:
        schema = make_dataclass("Envelope", [("answer", schema)])
        value = schema(answer=value)
    codec = LangChainStructuredResponseCodec([schema])
    encoded = codec.to_json(value)
    restored = LangChainStructuredResponseCodec([schema]).from_json(encoded)
    assert type(restored) is schema
    assert restored == value
