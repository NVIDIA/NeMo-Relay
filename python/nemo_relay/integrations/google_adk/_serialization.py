# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate ADK `LlmRequest` and `LlmResponse` dumps with `AdkCodec`. Project other values with `json_safe`."""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter, deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple, Protocol, TypeAlias, TypeVar, cast

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel, TypeAdapter
from pydantic.dataclasses import is_pydantic_dataclass
from pydantic_core import to_jsonable_python
from typing_extensions import TypeIs, override

from nemo_relay import AnnotatedLLMRequest, AnnotatedLLMResponse, Json, JsonObject, LLMRequest
from nemo_relay.codecs import LlmCodec, LlmResponseCodec
from nemo_relay.integrations._json_values import canonical, paired_items, same_json

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_PROVIDER: Final = "google_adk"
_REQUEST_EXCLUDE: Final[Mapping[str, set[str]]] = {"config": {"http_options", "response_schema"}}
_FRAMEWORK_MODULES: Final = ("google.genai.", "google.adk.")
_SCALARS: Final = frozenset({str, int, float, bool, type(None), bytes, bytearray})
_PYTHON_VALUES: Final = TypeAdapter(object)
_PARAMS: Final = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("max_output_tokens", "max_tokens"),
    ("stop_sequences", "stop"),
)
_MODELED_CONFIG_KEYS: Final = frozenset({"system_instruction", "tools", "tool_config", *(key for key, _ in _PARAMS)})
_MODELED_REQUEST_KEYS: Final = frozenset({"model", "contents", "config"})
_REQUEST_FIELDS: Final = frozenset(LlmRequest.model_fields)
_USAGE_FIELDS: Final[Annotated[frozenset[str], "Usage field names, plus camelCase aliases genai accepts."]] = frozenset(
    key
    for name, info in types.GenerateContentResponseUsageMetadata.model_fields.items()
    for key in (name, info.alias)
    if key is not None
)
_API_SPECIFIC_REQUEST_KEYS: Final = _REQUEST_FIELDS - _MODELED_REQUEST_KEYS - {"tools_dict"}
RECORDED_KEY: Final[Annotated[str, "Dump key for fields provider client reports and `LlmResponse` drops."]] = (
    "provider_reported"
)
RECORDED_ID_KEY: Final = "id"
RECORDED_USAGE_KEY: Final[Annotated[str, "Recorded key for usage provider client reports."]] = "usage"
_MODELED_RESPONSE_KEYS: Final = frozenset(
    {"content", "finish_reason", "usage_metadata", "model_version", "interaction_id", RECORDED_KEY}
)
BASE_USAGE: Final = (
    ("prompt_token_count", "prompt_tokens"),
    ("candidates_token_count", "completion_tokens"),
    ("total_token_count", "total_tokens"),
    ("cached_content_token_count", "cache_read_tokens"),
)
GENAI_USAGE: Final = (
    ("thoughts_token_count", "thoughts_tokens"),
    ("tool_use_prompt_token_count", "tool_use_prompt_tokens"),
)
BASE_USAGE_KEYS: Final = frozenset(source for source, _ in BASE_USAGE)
CACHE_WRITE_KEY: Final[
    Annotated[str, "Usage field where ADK puts cache writes. GenAI usage has no field for them."]
] = "cache_creation_input_tokens"


_FinishReason: TypeAlias = Literal["complete", "length", "content_filter", "tool_use"]
_FINISH_REASONS: Final[Mapping[str, _FinishReason]] = dict[str, _FinishReason](
    [
        (types.FinishReason.STOP.value, "complete"),
        (types.FinishReason.MAX_TOKENS.value, "length"),
        (types.FinishReason.SAFETY.value, "content_filter"),
        (types.FinishReason.RECITATION.value, "content_filter"),
        (types.FinishReason.BLOCKLIST.value, "content_filter"),
        (types.FinishReason.PROHIBITED_CONTENT.value, "content_filter"),
        (types.FinishReason.SPII.value, "content_filter"),
        (types.FinishReason.LANGUAGE.value, "content_filter"),
        (types.FinishReason.IMAGE_SAFETY.value, "content_filter"),
        (types.FinishReason.IMAGE_PROHIBITED_CONTENT.value, "content_filter"),
        (types.FinishReason.IMAGE_RECITATION.value, "content_filter"),
    ]
)
_ToolChoice: TypeAlias = Literal["auto", "none", "required"]
_TOOL_CHOICE_MODES: Final[Mapping[str, _ToolChoice]] = dict[str, _ToolChoice](
    [(types.FunctionCallingConfigMode.AUTO.value, "auto"), (types.FunctionCallingConfigMode.NONE.value, "none")]
)
_TOOL_CHOICE_WRITE_BACK: Final[Mapping[str, str]] = {
    "auto": types.FunctionCallingConfigMode.AUTO.value,
    "none": types.FunctionCallingConfigMode.NONE.value,
    "required": types.FunctionCallingConfigMode.ANY.value,
}
_PART_METADATA_KEYS: Final = frozenset(
    {"thought", "thought_signature", "part_metadata", "media_resolution", "video_metadata"}
)
_AdkRole: TypeAlias = Literal["user", "model"]
_MESSAGE_ROLES: Final[Mapping[str, _AdkRole]] = dict[str, _AdkRole](
    user="user", assistant="model", system="user", developer="user"
)
_FUNCTION_RESPONSE_FIELDS: Final = frozenset({"id", "name", "response", "parts"})

_Content: TypeAlias = Json | Sequence[Mapping[str, Json]]
_Blocks: TypeAlias = list[JsonObject]
_Functions: TypeAlias = list[tuple[int, JsonObject]]
_Message: TypeAlias = Mapping[str, Json]
_K = TypeVar("_K", bound=Hashable)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def unserializable(value: object) -> str:
    """Return stand-in text for a value with no JSON form, or for an iterator serializing would consume."""
    if isinstance(value, Iterator):
        return "<unserializable iterator>"
    kind = type(value)
    return f"<unserializable {kind.__module__}.{kind.__qualname__}>"


def _is_framework_model(value: object) -> TypeIs[BaseModel]:
    return isinstance(value, BaseModel) and type(value).__module__.startswith(_FRAMEWORK_MODULES)


def is_plain_dataclass(cls: type[object]) -> TypeIs[type[DataclassInstance]]:
    return dataclasses.is_dataclass(cls) and not is_pydantic_dataclass(cls)


def _stable(value: object) -> object:
    """Return value as Relay records it.

    Replace each set with its members as a list sorted by JSON, and each iterator with stand-in text. Return each
    unchanged part as the same object.
    """
    if type(value) in _SCALARS:
        return value
    if _is_framework_model(value):
        return _stable_model(value)
    if isinstance(value, dict):
        updates = {key: kept for key, item in value.items() if (kept := _stable(item)) is not item}
        return {**value, **updates} if updates else value
    if isinstance(value, list | tuple):
        items = [_stable(item) for item in value]
        return items if any(kept is not item for kept, item in zip(items, value, strict=True)) else value
    if isinstance(value, AbstractSet):
        return [_stable(member) for member in sorted_members(value, wire_json)]
    if isinstance(value, Iterator):
        return unserializable(value)
    if is_plain_dataclass(cls := type(value)):
        fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(cls)}
        kept = _stable(fields)
        return value if kept is fields else kept
    dumped = _PYTHON_VALUES.dump_python(value, mode="python", by_alias=True, exclude_none=True)
    if not holds_iterator(dumped):
        return value
    return unserializable(value) if dumped is value else _stable(dumped)


def _stable_model(model: _ModelT) -> _ModelT:
    updates: dict[str, object] = {}
    for name, field_info in type(model).model_fields.items():
        if not field_info.exclude and (kept := _stable(child := getattr(model, name))) is not child:
            updates[name] = kept
    return model.model_copy(update=updates) if updates else model


def wire_json(value: object) -> Json:
    """Return value's JSON as ADK sends it to the model.

    Encode bytes as base64, sort set members by JSON, and write stand-in text for each part with no JSON form.
    """
    return to_jsonable_python(_stable(value), bytes_mode="base64", exclude_none=True, fallback=unserializable)


def _part_json(value: object, exclude: set[str] | None) -> Json:
    """Return one field value of an ADK request or response as JSON.

    Dump each pydantic model separately, as genai does before sending one.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True, exclude=exclude, fallback=unserializable)
    if isinstance(value, list | tuple):
        return [_part_json(item, None) for item in value]
    return wire_json(value)


def _framework_json(model: BaseModel, exclude: Mapping[str, set[str]]) -> JsonObject:
    """Return model's fields as JSON.

    Sort each set's members. Write stand-in text for each iterator and each object holding one.
    """
    dumped = model.model_dump(mode="python", exclude_none=True, exclude=exclude)
    source = _stable_model(model) if holds_iterator_or_set(dumped) else model
    return {
        name: _part_json(value, exclude.get(name))
        for name, field_info in type(source).model_fields.items()
        if not field_info.exclude and (value := getattr(source, name)) is not None
    }


def request_payload(request: LlmRequest) -> JsonObject:
    """Return plain dump of ADK's request, with output schema as a JSON object.

    Keep only extra request body from HTTP options.
    """
    payload = _framework_json(request, _REQUEST_EXCLUDE)
    config: JsonObject = dict(object_field(payload.get("config"), field="ADK config"))
    if (schema := response_schema_payload(request.config.response_schema)) is not None:
        config["response_schema"] = schema
    options = request.config.http_options
    if options is not None and options.extra_body is not None:
        config["http_options"] = {"extra_body": wire_json(options.extra_body)}
    payload["config"] = config
    return payload


def response_schema_payload(schema: types.SchemaUnion | None) -> Json | None:
    """Return ADK's structured output schema as a JSON object, or None when ADK holds none.

    Accept `Schema`, dict, or a type pydantic renders as JSON Schema. Raise `TypeError` when pydantic cannot render the
    type.
    """
    if schema is None:
        return None
    if isinstance(schema, types.Schema):
        return schema.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(schema, dict):
        return json_safe(schema)
    try:
        return TypeAdapter(schema).json_schema()
    except Exception as error:
        raise TypeError(f"NeMo Relay cannot render the structured output schema {schema!r} as JSON Schema") from error


def response_payload(response: LlmResponse) -> JsonObject:
    """Return plain dump of ADK's response, plus usage counts ADK attaches outside GenAI schema.

    Raise `TypeError` when such a count is not an integer.
    """
    payload = _framework_json(response, {})
    if response.usage_metadata is None:
        return payload
    attached: dict[str, int] = {}
    for key, value in vars(response.usage_metadata).items():
        if key in _USAGE_FIELDS:
            continue
        if not isinstance(value, int):
            raise TypeError(f"ADK usage_metadata {key} must be an integer")
        attached[key] = value
    if attached:
        usage_metadata = object_field(payload.get("usage_metadata"), field="ADK usage_metadata")
        payload["usage_metadata"] = {**usage_metadata, **attached}
    return payload


def write_back_request(request: LlmRequest, *, payload: JsonObject, original_payload: JsonObject) -> bool:
    """Write each field differing from `original_payload` back onto ADK's request.

    Return whether a request field or extra request body changed.
    """
    payload, extra_body = _without_extra_body(payload)
    original_payload, original_extra_body = _without_extra_body(original_payload)
    changed = {
        key
        for key in payload.keys() | original_payload.keys()
        if not same_json(payload.get(key), original_payload.get(key))
    }
    if changed:
        validated = LlmRequest.model_validate(_merged_model(payload, original_payload, request))
        for field_ in LlmRequest.model_fields:
            if field_ in changed:
                setattr(request, field_, getattr(validated, field_))
    if same_json(extra_body, original_extra_body):
        return bool(changed)
    if extra_body is not None and not _is_object(extra_body):
        raise TypeError("ADK http_options extra_body must be a JSON object")
    options = request.config.http_options
    held_body = None if options is None else options.extra_body
    body = None if extra_body is None else _merged(extra_body, original_extra_body, held_body)
    held = options or types.HttpOptions()
    request.config = request.config.model_copy(update={"http_options": held.model_copy(update={"extra_body": body})})
    return True


def _without_extra_body(payload: JsonObject) -> tuple[JsonObject, Json]:
    """Return payload without its HTTP options, plus extra request body those options held.

    Raise `ValueError` when HTTP options hold any key besides `extra_body`.
    """
    config = object_field(payload.get("config"), field="ADK config")
    if "http_options" not in config:
        return payload, None
    options = object_field(config["http_options"], field="ADK http_options")
    if options.keys() - {"extra_body"}:
        raise ValueError(
            "NeMo Relay can write only extra_body to ADK's http_options; request headers belong in the request headers"
        )
    rest = {key: item for key, item in config.items() if key != "http_options"}
    return {**payload, "config": rest}, options.get("extra_body")


def _merged(edited: Json, projected: Json, original: object) -> object:
    """Return `original` when `edited` leaves its JSON unchanged.

    Otherwise return `edited` with each unchanged part restored from `original`.
    """
    if same_json(edited, projected):
        return original
    if _is_object(edited) and _is_object(projected):
        if _is_framework_model(original):
            return _merged_model(edited, projected, original)
        if isinstance(original, Mapping):
            parts = by_json_key(original)
            return {
                key: _merged(item, projected[key], parts[key]) if key in parts else item for key, item in edited.items()
            }
        if is_plain_dataclass(cls := type(original)):
            fields = {field.name: getattr(original, field.name) for field in dataclasses.fields(cls)}
            return {
                key: _merged(item, projected[key], fields[key]) if key in fields else item
                for key, item in edited.items()
            }
    if isinstance(edited, list) and isinstance(projected, list) and is_ordered(original):
        originals = sorted_members(original, wire_json) if isinstance(original, AbstractSet) else [*original]
        return [
            item if position is None else _merged(item, projected[position], originals[position])
            for item, position in zip(edited, paired_items(edited, projected), strict=True)
        ]
    return edited


def _merged_model(edited: JsonObject, projected: JsonObject, original: BaseModel) -> dict[str, object]:
    """Return field values that rebuild a changed model.

    Keep each original field the dump omitted and the edit does not set. Restore each unchanged part of an edited field
    from `original`.
    """
    fields = type(original).model_fields
    merged: dict[str, object] = {
        name: getattr(original, name) for name in fields if name not in projected and name not in edited
    }
    for key, item in edited.items():
        merged[key] = (
            _merged(item, projected[key], getattr(original, key)) if key in projected and key in fields else item
        )
    return merged


def http_headers(request: LlmRequest) -> dict[str, str]:
    """Return HTTP headers from ADK request's HTTP options, or empty headers when options hold none."""
    options = request.config.http_options
    return {} if options is None or options.headers is None else dict(options.headers)


def header_text(headers: Mapping[str, Json]) -> dict[str, str]:
    """Return request headers as HTTP header text Relay's gateway sends.

    Write each non-text JSON value as compact JSON.
    """
    return {
        key: value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        for key, value in headers.items()
    }


def write_headers(request: LlmRequest, headers: dict[str, str]) -> None:
    """Set HTTP headers of ADK's request, keeping its other HTTP options.

    No change when request already carries those headers.
    """
    if http_headers(request) == headers:
        return
    held = request.config.http_options or types.HttpOptions()
    request.config = request.config.model_copy(update={"http_options": held.model_copy(update={"headers": headers})})


def response_from_payload(payload: Json) -> LlmResponse:
    """Rebuild ADK's response from a dump, attaching usage counts outside GenAI schema as ADK does.

    Drop fields recorded from provider client. Raise `TypeError` when payload is not an object.
    """
    if not isinstance(payload, dict):
        raise TypeError("an LLM execution intercept must return an LlmResponse dump")
    return _response_from_fields(payload)


def _response_from_fields(values: Mapping[str, object]) -> LlmResponse:
    fields = {key: value for key, value in values.items() if key != RECORDED_KEY}
    attached: dict[str, int] = {}
    if _is_object(usage_metadata := fields.get("usage_metadata")):
        for key, value in usage_metadata.items():
            if key in _USAGE_FIELDS:
                continue
            if not isinstance(value, int):
                raise TypeError(f"ADK usage_metadata {key} must be an integer")
            attached[key] = value
        fields["usage_metadata"] = {key: value for key, value in usage_metadata.items() if key in _USAGE_FIELDS}
    response = LlmResponse.model_validate(fields)
    for key, value in attached.items():
        object.__setattr__(response.usage_metadata, key, value)
    return response


def is_ordered(value: object) -> TypeIs[Sequence[object] | AbstractSet[object]]:
    return isinstance(value, AbstractSet) or (
        isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
    )


def _is_exact_dict(value: object) -> TypeIs[dict[object, object]]:
    return type(value) is dict


def _is_exact_sequence(value: object) -> TypeIs[list[object] | tuple[object, ...]]:
    return type(value) is list or type(value) is tuple


def _holds(value: object, kinds: tuple[type[object], ...]) -> bool:
    """Return whether a pydantic dump holds a value of one of `kinds`.

    Check exact dict, list, and tuple types a dump consists of before `isinstance` checks.
    """
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if type(item) in _SCALARS:
            continue
        if _is_exact_dict(item):
            pending.extend(item.values())
        elif _is_exact_sequence(item):
            pending.extend(item)
        elif isinstance(item, kinds):
            return True
        elif isinstance(item, Mapping):
            pending.extend(item.values())
        elif is_ordered(item):
            pending.extend(item)
    return False


def holds_iterator(value: object) -> bool:
    return _holds(value, (Iterator,))


def holds_iterator_or_set(value: object) -> bool:
    return _holds(value, (Iterator, AbstractSet))


def sorted_members(members: AbstractSet[object], json_of: Callable[[object], Json]) -> list[object]:
    """Return set members ordered by canonical JSON, as Relay records them."""
    return sorted(members, key=lambda member: canonical(json_of(member)))


def json_key(key: object) -> str:
    if isinstance(key, str):
        return key
    (converted,) = cast("dict[str, None]", to_jsonable_python({key: None}))
    return converted


def by_json_key(source: Mapping[_K, object]) -> dict[str, object]:
    keyed = {json_key(key): item for key, item in source.items()}
    if len(keyed) != len(source):
        raise ValueError("NeMo Relay cannot write an edit back to a dictionary whose keys share a JSON name")
    return keyed


class ProducedResponses:
    """Responses one ADK model call produced.

    Hand back ADK's own response where pipeline left it unchanged.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[str, JsonObject, LlmResponse]] = []

    def add(self, response: LlmResponse, *, recorded: Mapping[str, Json], response_model: str | None) -> JsonObject:
        """Return response dump with `recorded` fields under `RECORDED_KEY`, and remember response under that dump.

        Set `response_model` as `model_version` only when dump has no `model_version`.
        """
        payload = response_payload(response)
        if recorded:
            payload[RECORDED_KEY] = dict(recorded)
        if response_model is not None and "model_version" not in payload:
            payload["model_version"] = response_model
        self._pending.append((canonical(payload), payload, response))
        return payload

    def restore_result(self, result: Json) -> LlmResponse:
        """Return ADK's response for pipeline's result.

        Return latest produced response with same dump. Otherwise merge result over latest produced response, or rebuild
        result when call produced none.
        """
        return self._restored(result, range(len(self._pending) - 1, -1, -1))

    def restore_chunk(self, chunk: Json) -> LlmResponse:
        """Return ADK's response for one streamed chunk.

        Return earliest pending response with same dump. Otherwise merge chunk over earliest pending response, or
        rebuild chunk when none is pending.
        """
        return self._restored(chunk, range(len(self._pending)))

    def _restored(self, payload: Json, order: range) -> LlmResponse:
        key = canonical(payload)
        for index in order:
            if self._pending[index][0] == key:
                return self._pending.pop(index)[2]
        if not order:
            return response_from_payload(payload)
        _, produced, response = self._pending.pop(order[0])
        merged = _merged(payload, produced, response)
        if not _is_field_map(merged):
            raise TypeError("an LLM execution intercept must return an LlmResponse dump")
        return _response_from_fields(merged)


def _is_object(value: object) -> TypeIs[JsonObject]:
    return isinstance(value, dict)


def _is_field_map(value: object) -> TypeIs[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


_LAST_SEEN_FIELDS: Final = (
    "usage_metadata",
    "grounding_metadata",
    RECORDED_KEY,
    "citation_metadata",
    "cache_metadata",
    "custom_metadata",
    "finish_reason",
    "error_code",
    "error_message",
    "model_version",
    "interaction_id",
)


_ARG_PATH_TOKEN: Final = re.compile(
    r"\[\s*(\d+)\s*\]|\['((?:[^'\\]|\\.)*)'\]|\[\"((?:[^\"\\]|\\.)*)\"\]|(?:\.?((?:[^.\[\]\\]|\\.)+))"
)
_ARG_PATH_ESCAPE: Final = re.compile(
    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})|\\u([0-9a-fA-F]{4})|\\(.)"
)
_ARG_PATH_ESCAPES: Final[Mapping[str, str]] = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
_ARG_MAX_INDEX: Final = 10_000
_ARG_MAX_DEPTH: Final = 128
_ArgPath: TypeAlias = tuple[str | int, ...]


def _unescape_arg_path(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if (high := match.group(1)) is not None:
            return chr(0x10000 + ((int(high, 16) - 0xD800) << 10) + (int(match.group(2), 16) - 0xDC00))
        if (single := match.group(3)) is not None:
            code = int(single, 16)
            return "\ufffd" if 0xD800 <= code <= 0xDFFF else chr(code)
        return _ARG_PATH_ESCAPES.get(match.group(4), match.group(4))

    return _ARG_PATH_ESCAPE.sub(replace, text)


def _arg_path(path: str) -> _ArgPath:
    """Return keys and indexes of a `partial_args` JSON path, parsed as ADK parses them."""
    path = path.removeprefix("$.") if path.startswith("$.") else path.removeprefix("$")
    tokens: list[str | int] = []
    for match in _ARG_PATH_TOKEN.finditer(path):
        if match.group(1) is not None:
            tokens.append(int(match.group(1)))
        else:
            tokens.append(_unescape_arg_path(next(group for group in match.group(2, 3, 4) if group is not None)))
    return tuple(tokens)


def _streamed_arg_path(path: str) -> _ArgPath:
    tokens = _arg_path(path)
    if not tokens:
        raise ValueError(f"ADK partial_args json_path {path!r} does not address an argument")
    if len(tokens) > _ARG_MAX_DEPTH:
        raise ValueError(f"ADK partial_args json_path {path!r} nests deeper than {_ARG_MAX_DEPTH} levels")
    if any(isinstance(token, int) and token > _ARG_MAX_INDEX for token in tokens):
        raise ValueError(f"ADK partial_args json_path {path!r} indexes past {_ARG_MAX_INDEX}")
    return tokens


def _arg_text(args: Json, tokens: _ArgPath) -> str | None:
    current = args
    for token in tokens:
        if isinstance(token, str):
            if not _is_object(current) or token not in current:
                return None
            current = current[token]
        elif isinstance(current, list) and 0 <= token < len(current):
            current = current[token]
        else:
            return None
    return current if isinstance(current, str) else None


def _set_arg(args: JsonObject, tokens: _ArgPath, value: Json) -> None:
    current: Json = args
    for position, token in enumerate(tokens[:-1]):
        empty: Json = [] if isinstance(tokens[position + 1], int) else {}
        if isinstance(token, str):
            if not _is_object(current):
                return
            current = current.setdefault(token, empty)
        else:
            if not isinstance(current, list):
                return
            while len(current) <= token:
                current.append(None)
            if current[token] is None:
                current[token] = empty
            current = current[token]
    last = tokens[-1]
    if isinstance(last, str):
        if _is_object(current):
            current[last] = value
    elif isinstance(current, list):
        while len(current) <= last:
            current.append(None)
        current[last] = value


@dataclass(slots=True, kw_only=True)
class _StreamedCall:
    """Function call whose arguments arrive as `partial_args` fragments.

    Holds call's slot in parts until call ends.
    """

    slot: int
    part: JsonObject = field(default_factory=dict)
    call: JsonObject = field(default_factory=dict)
    args: JsonObject = field(default_factory=dict)
    chunks: dict[_ArgPath, list[str]] = field(default_factory=dict)
    paths: dict[str, _ArgPath] = field(default_factory=dict)

    def tokens(self, path: str) -> _ArgPath:
        if (tokens := self.paths.get(path)) is None:
            tokens = self.paths[path] = _streamed_arg_path(path)
        return tokens

    def write(self, tokens: _ArgPath, value: Json) -> None:
        self.chunks = {chunked: pieces for chunked, pieces in self.chunks.items() if chunked[: len(tokens)] != tokens}
        _set_arg(self.args, tokens, value)

    def apply(self, partial: JsonObject) -> None:
        path = partial.get("json_path")
        if not isinstance(path, str) or not path:
            return
        chunk = partial.get("string_value")
        if isinstance(chunk, str):
            tokens = self.tokens(path)
            if tokens not in self.chunks:
                existing = _arg_text(self.args, tokens)
                self.chunks[tokens] = [existing] if existing is not None else []
                _set_arg(self.args, tokens, "")
            self.chunks[tokens].append(chunk)
            return
        value: Json
        if "number_value" in partial:
            value = partial["number_value"]
        elif "bool_value" in partial:
            value = partial["bool_value"]
        elif "null_value" in partial:
            if partial["null_value"] not in (None, "NULL_VALUE"):
                raise ValueError("ADK partial_args null_value must be NULL_VALUE")
            value = None
        else:
            return
        self.write(self.tokens(path), value)

    def into_part(self) -> JsonObject:
        for tokens, pieces in self.chunks.items():
            _set_arg(self.args, tokens, "".join(pieces))
        return {**self.part, "function_call": {**self.call, "args": self.args}}


def _differs(active: Json, fragment: Json) -> bool:
    return isinstance(active, str) and isinstance(fragment, str) and active != fragment


def _joins_call(function: JsonObject, active: _StreamedCall | None) -> bool:
    return active is not None and not any(_differs(active.call.get(key), function.get(key)) for key in ("name", "id"))


def _starts_call(function: JsonObject) -> bool:
    return bool(function.get("partial_args")) or function.get("will_continue") is not None


def _flush_call(parts: list[JsonObject | None], active: _StreamedCall | None) -> None:
    if active is not None and "name" in active.call:
        parts[active.slot] = active.into_part()


def _observe_fragment(
    parts: list[JsonObject | None], active: _StreamedCall | None, part: JsonObject, function: JsonObject
) -> _StreamedCall | None:
    if active is None:
        active = _StreamedCall(slot=len(parts))
        parts.append(None)
    for key in ("name", "id"):
        if key in function and key not in active.call:
            active.call[key] = function[key]
    for key, value in part.items():
        if key != "function_call" and key not in active.part:
            active.part[key] = value
    for key, value in object_field(function.get("args"), field="ADK function_call args").items():
        active.write((key,), value)
    for partial in _objects(function.get("partial_args"), field="ADK function_call partial_args"):
        active.apply(partial)
    if function.get("will_continue") is True:
        return active
    _flush_call(parts, active)
    return None


class _TextRun(NamedTuple):
    """Adjacent text parts of one thought kind, joined into first part when stream ends."""

    part: JsonObject
    pieces: list[str]


def final_response(chunks: Sequence[Json]) -> JsonObject:
    """Return stream's final `LlmResponse` dump.

    Last whole chunk, plus last value of each response-level field later chunks report. Aggregate partial chunks when no
    whole chunk arrived.
    """
    dumps = [_dump(chunk) for chunk in chunks]
    for index in range(len(dumps) - 1, -1, -1):
        if not dumps[index].get("partial"):
            return {**dumps[index], **_last_seen(dumps[index + 1 :])}
    return aggregate_partials(dumps)


def aggregate_partials(chunks: Sequence[Json]) -> JsonObject:
    """Aggregate partial `LlmResponse` dumps into one final dump, as ADK does.

    Join adjacent text of one thought kind, assemble streamed function call arguments, and keep last value of each
    response-level field.
    """
    parts: list[JsonObject | None] = []
    runs: list[_TextRun] = []
    run: _TextRun | None = None
    call: _StreamedCall | None = None
    dumps = [_dump(chunk) for chunk in chunks]
    for chunk in dumps:
        for part in _parts_of(chunk):
            text = part.get("text")
            function = part.get("function_call")
            if isinstance(text, str) and function is None:
                if _append_text_run(run, part, text):
                    continue
                run = _TextRun(dict(part), [text])
                runs.append(run)
                parts.append(run.part)
                continue
            run = None
            if function is not None:
                function = object_field(function, field="ADK function_call")
                if not _joins_call(function, call):
                    _flush_call(parts, call)
                    call = None
                if call is not None or _starts_call(function):
                    call = _observe_fragment(parts, call, part, function)
                    continue
            parts.append(part)
    _flush_call(parts, call)
    return _finish_aggregate(dumps, runs, parts)


def _finish_aggregate(dumps: list[JsonObject], runs: list[_TextRun], parts: list[JsonObject | None]) -> JsonObject:
    for merged in runs:
        merged.part["text"] = "".join(merged.pieces)
    aggregate = _last_seen(dumps)
    kept: list[Json] = [part for part in parts if part is not None]
    if kept:
        aggregate["content"] = {"role": "model", "parts": kept}
    aggregate["partial"] = False
    return aggregate


def _append_text_run(run: _TextRun | None, part: JsonObject, text: str) -> bool:
    if run is None or run.part.get("thought") != part.get("thought"):
        return False
    run.pieces.append(text)
    if "thought_signature" in part and "thought_signature" not in run.part:
        run.part["thought_signature"] = part["thought_signature"]
    return True


def _dump(chunk: Json) -> JsonObject:
    if not _is_object(chunk):
        raise TypeError("a stream execution intercept must yield LlmResponse dumps")
    return chunk


def _last_seen(dumps: Iterable[JsonObject]) -> JsonObject:
    seen: JsonObject = {}
    for dump in dumps:
        for key in _LAST_SEEN_FIELDS:
            if dump.get(key) is not None:
                seen[key] = dump[key]
    return seen


def _parts_of(chunk: JsonObject) -> list[JsonObject]:
    content = object_field(chunk.get("content"), field="ADK response content")
    return _objects(content.get("parts"), field="ADK response content parts")


def _objects(value: Json, *, field: str) -> list[JsonObject]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    objects: list[JsonObject] = []
    for item in value:
        if not _is_object(item):
            raise TypeError(f"{field} entries must be objects")
        objects.append(item)
    return objects


def object_field(value: Json, *, field: str) -> JsonObject:
    if value is None:
        return {}
    if not _is_object(value):
        raise TypeError(f"{field} must be an object")
    return value


def _optional_text(value: Json, *, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    return value


def _name(function: _Message, *, field: str) -> str:
    name = _optional_text(function.get("name"), field=f"{field} name")
    if not name:
        raise ValueError(f"{field} must have a name")
    return name


def _identifier(function: _Message, *, field: str) -> str | None:
    return _optional_text(function.get("id"), field=f"{field} id") or None


def _function_parts(parts: Sequence[JsonObject], field: str) -> _Functions:
    functions: _Functions = []
    for index, part in enumerate(parts):
        function = part.get(field)
        if function is None:
            continue
        if not _is_object(function):
            raise TypeError(f"ADK {field} must be an object")
        functions.append((index, function))
    return functions


def _function_ids(functions: _Functions) -> dict[int, str]:
    """Return an id for each function part.

    Keep a part's own id. ADK leaves calls without ids, so give same-name parts without ids positional ids like
    `name#0`, and any other id-less part its name.
    """
    ids: dict[int, str] = {}
    unidentified: list[tuple[int, str]] = []
    for index, function in functions:
        identifier = _identifier(function, field="ADK function")
        if identifier is None:
            unidentified.append((index, _name(function, field="ADK function")))
        else:
            ids[index] = identifier
    names = Counter(name for _, name in unidentified)
    ordinals: Counter[str] = Counter()
    for index, name in unidentified:
        if names[name] > 1:
            ids[index] = f"{name}#{ordinals[name]}"
            ordinals[name] += 1
        else:
            ids[index] = name
    return ids


def _derived_id(identifier: str, *, name: str) -> bool:
    return identifier == name or (identifier.startswith(f"{name}#") and identifier[len(name) + 1 :].isdigit())


def _native(kind: str, value: Json) -> JsonObject:
    return {"type": "provider_native", "provider": _PROVIDER, "kind": kind, "value": value}


def _part_block(part: JsonObject) -> JsonObject:
    if part.get("thought") is True:
        return _native("thought", part)
    if isinstance(part.get("text"), str):
        return {"type": "text", **part}
    kinds = [key for key in part if key not in _PART_METADATA_KEYS] or [*part]
    if not kinds:
        raise ValueError("An ADK part must have a field")
    return _native(kinds[0], part)


def _visible(parts: Sequence[JsonObject]) -> list[JsonObject]:
    return [part for part in parts if "function_call" not in part and "function_response" not in part]


def _content_blocks(parts: Sequence[JsonObject]) -> str | _Blocks | None:
    content = _visible(parts)
    if len(content) == 1 and content[0].keys() == {"text"} and isinstance(text := content[0]["text"], str):
        return text
    blocks: _Blocks = [_part_block(part) for part in content]
    return blocks or None


def _message_content(parts: Sequence[JsonObject]) -> Json:
    content = _content_blocks(parts)
    return [*content] if isinstance(content, list) else content


def _blocks_to_parts(content: _Content) -> list[JsonObject]:
    """Convert message content to genai parts, restoring ADK provider-native parts verbatim.

    Raise `TypeError` for content or a part of a type an ADK request cannot hold, `ValueError` for an invalid part.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, Sequence):
        raise TypeError(f"NeMo Relay cannot write a {type(content).__name__} message content back to an ADK request")
    parts: list[JsonObject] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"text": item})
            continue
        if not _is_object(item):
            raise TypeError(f"NeMo Relay cannot write a {type(item).__name__} content part back to an ADK request")
        if item.get("type") == "provider_native":
            if item.get("provider") != _PROVIDER:
                raise ValueError(
                    f"NeMo Relay cannot write a {item.get('provider')!r} provider-native part into an ADK request"
                )
            value = item.get("value")
            if not _is_object(value):
                raise ValueError("NeMo Relay provider-native ADK parts must carry an object value")
            parts.append(dict(value))
        elif item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append({key: value for key, value in item.items() if key != "type"})
        else:
            raise ValueError(f"NeMo Relay cannot write a {item.get('type')!r} content part back to an ADK request")
    return parts


def _function_response_content(function_response: JsonObject) -> str | list[Json]:
    response = function_response.get("response")
    if response is None:
        raise ValueError("ADK function_response must have a response")
    text = json.dumps(
        object_field(response, field="ADK function_response response"), ensure_ascii=False, separators=(",", ":")
    )
    nested = _objects(function_response.get("parts"), field="ADK function_response parts")
    if nested:
        return [{"type": "text", "text": text}, *map(_part_block, nested)]
    return text


def _json_object(text: str, *, key: str) -> JsonObject:
    """Return object a JSON text holds, or the text under `key` when it holds no JSON object."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return {key: text}
    return parsed if _is_object(parsed) else {key: text}


def _function_response_fields(content: Json) -> tuple[JsonObject, list[JsonObject] | None]:
    if isinstance(content, str):
        return _json_object(content, key="result"), None
    if not isinstance(content, list):
        raise TypeError(f"NeMo Relay cannot write a {type(content).__name__} tool message content into an ADK request")
    first = content[0] if content else None
    text = first.get("text") if _is_object(first) and first.get("type") == "text" else None
    if not isinstance(text, str):
        raise TypeError(
            "NeMo Relay can write a tool message with blocks into an ADK request only when its first block is the "
            "response text"
        )
    rest = content[1:]
    return _json_object(text, key="result"), _blocks_to_parts(rest) if rest else None


def _content_messages(content: JsonObject) -> list[JsonObject]:
    role = content.get("role")
    parts = _objects(content.get("parts"), field="ADK content parts")
    responses = _function_parts(parts, "function_response")
    if responses:
        ids = _function_ids(responses)
        tools: list[JsonObject] = [
            {"role": "tool", "tool_call_id": ids[index], "content": _function_response_content(function_response)}
            for index, function_response in responses
        ]
        text = _message_content(parts)
        if text is None:
            return tools
        return [{"role": "assistant" if role == "model" else "user", "content": text}, *tools]
    if role not in {None, "user", "model"}:
        return [{"role": "provider_native", "provider": _PROVIDER, "kind": "content", "value": content}]
    text = _message_content(parts)
    if role != "model" and text is None:
        raise ValueError("An ADK user content must have parts")
    message: JsonObject = {"role": "assistant" if role == "model" else "user", "content": text}
    calls = _function_parts(parts, "function_call")
    if calls:
        ids = _function_ids(calls)
        message["tool_calls"] = [
            {
                "id": ids[index],
                "type": "function",
                "function": {
                    "name": _name(call, field="ADK function_call"),
                    "arguments": json.dumps(
                        object_field(call.get("args"), field="ADK function_call args"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for index, call in calls
        ]
    return [message]


def _response_names(contents: Sequence[JsonObject]) -> dict[str, str]:
    names: dict[str, str] = {}
    for content in contents:
        parts = _objects(content.get("parts"), field="ADK content parts")
        for field_ in ("function_call", "function_response"):
            functions = _function_parts(parts, field_)
            ids = _function_ids(functions)
            for index, function in functions:
                names[ids[index]] = _name(function, field="ADK function")
    return names


_ARGUMENTS_TEXT: Final = "NeMo Relay can write only JSON object text as tool call arguments into an ADK request"


def _parsed_arguments(arguments: str) -> JsonObject:
    try:
        parsed = json.loads(arguments) if arguments else {}
    except ValueError as error:
        raise ValueError(_ARGUMENTS_TEXT) from error
    if not _is_object(parsed):
        raise ValueError(_ARGUMENTS_TEXT)
    return parsed


def _function_call_part(call: _Message, names_by_id: dict[str, str]) -> JsonObject:
    function = object_field(call.get("function"), field="tool call function") or call
    name = _name(function, field="tool call")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        args = _parsed_arguments(arguments)
    else:
        args = object_field(arguments, field="tool call arguments")
    part: JsonObject = {"name": name, "args": args}
    identifier = _identifier(call, field="tool call")
    if identifier is not None:
        names_by_id[identifier] = name
        if not _derived_id(identifier, name=name):
            part["id"] = identifier
    return {"function_call": part}


def _function_response(message: _Message, names_by_id: dict[str, str]) -> JsonObject:
    identifier = _optional_text(message.get("tool_call_id"), field="tool message tool_call_id") or ""
    name = names_by_id.get(identifier)
    if name is None:
        raise ValueError(
            f"NeMo Relay cannot write a tool message for an unknown call {identifier!r} into an ADK request"
        )
    response, nested = _function_response_fields(message.get("content"))
    part: JsonObject = {"name": name, "response": response}
    if nested:
        part["parts"] = [*nested]
    if not _derived_id(identifier, name=name):
        part["id"] = identifier
    return part


def _native_content(message: _Message) -> JsonObject:
    if message.get("provider") != _PROVIDER or message.get("kind") != "content":
        raise ValueError(
            f"NeMo Relay cannot write a {message.get('provider')!r} {message.get('kind')!r} provider-native message "
            "into an ADK request"
        )
    value = message.get("value")
    if not _is_object(value):
        raise ValueError("NeMo Relay provider-native ADK messages must carry an object value")
    return dict(value)


def _adk_role(message: _Message) -> _AdkRole:
    role = message.get("role")
    adk_role = _MESSAGE_ROLES.get(role) if isinstance(role, str) else None
    if adk_role is None:
        raise ValueError(f"NeMo Relay cannot write a {role!r} message back to an ADK request")
    return adk_role


def _content(message: _Message, names_by_id: dict[str, str]) -> JsonObject:
    if message.get("role") == "provider_native":
        return _native_content(message)
    parts: list[Json] = [*_blocks_to_parts(message.get("content"))]
    parts.extend(
        _function_call_part(call, names_by_id) for call in _objects(message.get("tool_calls"), field="tool_calls")
    )
    return {"role": _adk_role(message), "parts": parts}


def _result_id(message: _Message) -> str | None:
    if message.get("role") != "tool":
        return None
    return _optional_text(message.get("tool_call_id"), field="tool message tool_call_id") or None


def _call_ids(message: _Message) -> frozenset[str] | None:
    """Return assistant turn's call ids when each call has a distinct id, else None."""
    if message.get("role") != "assistant":
        return None
    ids = [
        _identifier(call, field="tool call") or "" for call in _objects(message.get("tool_calls"), field="tool_calls")
    ]
    distinct = frozenset(ids)
    return distinct if ids and "" not in distinct and len(distinct) == len(ids) else None


def _match_unique(
    aligned: list[int | None],
    used: set[int],
    edited: Sequence[_Message],
    baseline: Sequence[_Message],
    key: Callable[[_Message], _K | None],
) -> bool:
    """Align unmatched edited messages to unused baseline messages by a key unique on both sides.

    Return whether any new match changed a message's position.
    """
    edited_keys = [key(message) for message in edited]
    baseline_keys = [key(message) for message in baseline]
    edited_counts = Counter(identity for identity in edited_keys if identity is not None)
    baseline_counts = Counter(identity for identity in baseline_keys if identity is not None)
    positions = {
        identity: position
        for position, identity in enumerate(baseline_keys)
        if identity is not None and baseline_counts[identity] == 1
    }
    moved = False
    for index, identity in enumerate(edited_keys):
        if aligned[index] is not None or identity is None or edited_counts[identity] != 1:
            continue
        match = positions.get(identity)
        if match is not None and match not in used:
            aligned[index] = match
            used.add(match)
            moved |= match != index
    return moved


def _aligned(
    edited: Sequence[_Message], baseline: Sequence[_Message], edited_keys: Sequence[str], baseline_keys: Sequence[str]
) -> list[int | None]:
    """Return, per edited message, decoded message it aligns with, or None.

    Apply Relay's alignment first, then match remaining assistant turns by call ids.
    """
    aligned: list[int | None] = [
        index if index < len(baseline_keys) and key == baseline_keys[index] else None
        for index, key in enumerate(edited_keys)
    ]
    used = {index for index in aligned if index is not None}
    moved = len(edited) != len(baseline)
    unused: dict[str, deque[int]] = {}
    for position, key in enumerate(baseline_keys):
        if position not in used:
            unused.setdefault(key, deque()).append(position)
    for index, key in enumerate(edited_keys):
        if aligned[index] is None and (equal := unused.get(key)):
            match = equal.popleft()
            aligned[index] = match
            used.add(match)
            moved |= match != index
    moved |= _match_unique(aligned, used, edited, baseline, _result_id)
    unmatched = [index for index, match in enumerate(aligned) if match is None]
    if not moved and len(unmatched) == 1:
        (index,) = unmatched
        if _result_id(edited[index]) is None and _result_id(baseline[index]) is None:
            aligned[index] = index
            used.add(index)
    _match_unique(aligned, used, edited, baseline, _call_ids)
    return aligned


def _rebuilt_calls(parts: Sequence[JsonObject], message: _Message, names_by_id: dict[str, str]) -> list[JsonObject]:
    """Rebuild message's tool calls as parts.

    Merge each call over earliest original part with same call id not taken by an earlier call.
    """
    originals: dict[str, deque[JsonObject]] = {}
    for index, identifier in _function_ids(_function_parts(parts, "function_call")).items():
        originals.setdefault(identifier, deque()).append(parts[index])
    rebuilt: list[JsonObject] = []
    for call in _objects(message.get("tool_calls"), field="tool_calls"):
        part = _function_call_part(call, names_by_id)
        waiting = originals.get(_identifier(call, field="tool call") or "")
        rebuilt.append({**waiting.popleft(), **part} if waiting else part)
    return rebuilt


_Slot: TypeAlias = Literal["call", "text", "keep"]


def _slot(part: JsonObject, *, blocks: bool) -> _Slot:
    if "function_call" in part:
        return "call"
    if "function_response" in part:
        return "keep"
    if blocks or ("text" in part and part.get("thought") is not True):
        return "text"
    return "keep"


def _patched_parts(parts: Sequence[JsonObject], message: _Message, names_by_id: dict[str, str]) -> list[JsonObject]:
    """Return original parts with message's text and calls written into their slots.

    Merge parts as core Gemini codec merges them.
    """
    content = message.get("content")
    blocks = isinstance(content, list)
    replacements = _blocks_to_parts(content)
    calls = _rebuilt_calls(parts, message, names_by_id)
    slots = [_slot(part, blocks=blocks) for part in parts]
    written = dict(zip((index for index, slot in enumerate(slots) if slot == "text"), replacements, strict=False))
    rebuilt = dict(zip((index for index, slot in enumerate(slots) if slot == "call"), calls, strict=False))
    patched: list[JsonObject] = []
    for index, (part, slot) in enumerate(zip(parts, slots, strict=True)):
        if slot == "keep":
            patched.append(part)
        elif index in rebuilt:
            patched.append(rebuilt[index])
        elif index in written:
            patched.append(written[index] if blocks else {**part, "text": written[index]["text"]})
    extra = replacements[slots.count("text") :]
    patched = [*patched, *extra] if written else [*extra, *patched]
    return [*patched, *calls[slots.count("call") :]]


def _rebuilt_responses(
    parts: Sequence[JsonObject], run: Sequence[_Message], names_by_id: dict[str, str]
) -> list[JsonObject]:
    """Rebuild run's tool messages as function response parts.

    Merge each part over earliest original part with same call id. Raise `ValueError` when run holds anything besides
    tool messages with distinct call ids.
    """
    originals: dict[str, JsonObject] = {}
    for index, identifier in _function_ids(_function_parts(parts, "function_response")).items():
        originals.setdefault(identifier, parts[index])
    rebuilt: list[JsonObject] = []
    seen: set[str] = set()
    for message in run:
        identifier = _optional_text(message.get("tool_call_id"), field="tool message tool_call_id") or ""
        if message.get("role") != "tool" or not identifier or identifier in seen:
            raise ValueError("NeMo Relay can write only tool messages with distinct tool call ids into one ADK content")
        seen.add(identifier)
        response = _function_response(message, names_by_id)
        part: JsonObject = {"function_response": response}
        if (original := originals.get(identifier)) is not None:
            kept = {
                key: value
                for key, value in object_field(original.get("function_response"), field="ADK function_response").items()
                if key not in _FUNCTION_RESPONSE_FIELDS
            }
            part = {**original, "function_response": {**kept, **response}}
        rebuilt.append(part)
    return rebuilt


def _patched_visible(parts: Sequence[JsonObject], message: _Message, names_by_id: dict[str, str]) -> list[JsonObject]:
    if same_json(_message_content(parts), message.get("content")):
        return [*parts]
    return _patched_parts(parts, message, names_by_id)


def _patched_responses(
    parts: Sequence[JsonObject], run: Sequence[_Message], names_by_id: dict[str, str]
) -> list[JsonObject]:
    """Write run's visible message over visible slots, and its tool messages over function response slots.

    Raise `ValueError` when run holds more than one visible message, or a visible message that is not a user message.
    """
    visible = [message for message in run if message.get("role") != "tool"]
    tools = [message for message in run if message.get("role") == "tool"]
    if len(visible) > 1:
        raise ValueError("NeMo Relay can write at most one visible message into an ADK function response content")
    if visible and visible[0].get("role") != "user":
        raise ValueError(
            f"NeMo Relay cannot write a {visible[0].get('role')!r} message beside ADK function responses; "
            "only a user message can share the turn"
        )
    kept = (
        _patched_visible(parts, visible[0], names_by_id)
        if visible
        else [part for part in parts if "function_call" in part or "function_response" in part]
    )
    rebuilt = iter(_rebuilt_responses(parts, tools, names_by_id))
    patched: list[JsonObject] = []
    for part in kept:
        if "function_response" not in part:
            patched.append(part)
        elif (response := next(rebuilt, None)) is not None:
            patched.append(response)
    return [*patched, *rebuilt]


def _patched_content(original: JsonObject, run: Sequence[_Message], names_by_id: dict[str, str]) -> JsonObject:
    first = run[0]
    if first.get("role") == "provider_native":
        return _native_content(first)
    parts = _objects(original.get("parts"), field="ADK content parts")
    if _function_parts(parts, "function_response"):
        role = "user" if first.get("role") == "tool" else _adk_role(first)
        patched: list[Json] = [*_patched_responses(parts, run, names_by_id)]
    else:
        role = _adk_role(first)
        patched = [*_patched_parts(parts, first, names_by_id)]
    return {**original, "role": role, "parts": patched}


def _call_key(part: JsonObject) -> str | None:
    call = object_field(part.get("function_call"), field="ADK function_call")
    if not call:
        return None
    return _identifier(call, field="ADK function_call") or _name(call, field="ADK function_call")


def _keep_call_fields(contents: Sequence[JsonObject], originals: Sequence[JsonObject]) -> None:
    """Merge each call part over original part with same call key.

    Merge only when key is unique among both edited and original calls. Call key: its id, or its name when it has no id.
    """
    found: dict[str, JsonObject | None] = {}
    for part in (part for content in originals for part in _objects(content.get("parts"), field="ADK content parts")):
        if (key := _call_key(part)) is not None:
            found[key] = None if key in found else part
    counts = Counter(
        key
        for content in contents
        for part in _objects(content.get("parts"), field="ADK content parts")
        if (key := _call_key(part)) is not None
    )
    for content in contents:
        parts = _objects(content.get("parts"), field="ADK content parts")
        restored = False
        for position, part in enumerate(parts):
            if (key := _call_key(part)) is None or counts[key] != 1:
                continue
            if (original := found.get(key)) is not None:
                parts[position] = {**original, **part}
                restored = True
        if restored:
            content["parts"] = [*parts]


def _message_count(content: JsonObject) -> int:
    parts = _objects(content.get("parts"), field="ADK content parts")
    responses = _function_parts(parts, "function_response")
    if not responses:
        return 1
    return len(responses) + (_message_content(parts) is not None)


def _patched_contents(
    edited: Sequence[_Message], baseline: Sequence[_Message], originals: Sequence[JsonObject]
) -> list[JsonObject]:
    """Return ADK contents for edited messages.

    Keep each unchanged content whole, patch each edited content in place, and build new contents for added messages.
    """
    owners = [index for index, content in enumerate(originals) for _ in range(_message_count(content))]
    sizes = Counter(owners)
    edited_keys = [canonical(message) for message in edited]
    baseline_keys = [canonical(message) for message in baseline]
    aligned = _aligned(edited, baseline, edited_keys, baseline_keys)
    names_by_id = _response_names(originals)
    contents: list[JsonObject] = []
    responses: list[Json] = []
    written: set[int] = set()
    index = 0
    while index < len(edited):
        message, match = edited[index], aligned[index]
        if match is None and message.get("role") == "tool":
            responses.append({"function_response": _function_response(message, names_by_id)})
            index += 1
            continue
        if responses:
            contents.append({"role": "user", "parts": responses})
            responses = []
        if match is None:
            contents.append(_content(message, names_by_id))
            index += 1
            continue
        source = owners[match]
        original = originals[source]
        if source in written:
            fragment: JsonObject = {
                **original,
                "parts": [
                    part
                    for part in _objects(original.get("parts"), field="ADK content parts")
                    if "function_response" in part
                ],
            }
            contents.append(_patched_content(fragment, edited[index : index + 1], names_by_id))
            index += 1
            continue
        end = index + 1
        while end < len(edited) and (following := aligned[end]) is not None and owners[following] == source:
            end += 1
        written.add(source)
        unchanged = end - index == sizes[source] and all(
            aligned[index + offset] == match + offset and edited_keys[index + offset] == baseline_keys[match + offset]
            for offset in range(end - index)
        )
        contents.append(original if unchanged else _patched_content(original, edited[index:end], names_by_id))
        index = end
    if responses:
        contents.append({"role": "user", "parts": responses})
    _keep_call_fields(contents, originals)
    return contents


def _instructions(system_instruction: Json) -> str | _Blocks | None:
    if isinstance(system_instruction, str) or system_instruction is None:
        return system_instruction
    if isinstance(system_instruction, dict):
        parts = system_instruction.get("parts", [system_instruction])
        return _content_blocks(_objects(parts, field="ADK system_instruction parts"))
    if isinstance(system_instruction, list):
        items: list[Json] = [{"text": item} if isinstance(item, str) else item for item in system_instruction]
        return _content_blocks(_objects(items, field="ADK system_instruction parts"))
    raise TypeError("ADK system_instruction must be text, content, or a list of parts")


def _system_instruction(instructions: _Content) -> str | JsonObject | None:
    if instructions is None or isinstance(instructions, str):
        return instructions
    return {"parts": [*_blocks_to_parts(instructions)]}


def _function_tool(declaration: JsonObject) -> JsonObject:
    return {"type": "function", "function": _function_definition(declaration)}


def _function_definition(declaration: JsonObject) -> JsonObject:
    function: JsonObject = {"name": _name(declaration, field="ADK function declaration")}
    if isinstance(declaration.get("description"), str):
        function["description"] = declaration["description"]
    parameters = declaration.get("parameters_json_schema", declaration.get("parameters"))
    if parameters is not None:
        function["parameters"] = parameters
    for key, value in declaration.items():
        if key not in {"name", "description", "parameters", "parameters_json_schema"}:
            function[key] = value
    return function


def _tools(groups: Json) -> list[JsonObject] | None:
    tools: list[JsonObject] = []
    for group in _objects(groups, field="ADK config tools"):
        tools.extend(
            _function_tool(declaration)
            for declaration in _objects(group.get("function_declarations"), field="ADK tool function_declarations")
        )
        native = {key: value for key, value in group.items() if key != "function_declarations"}
        if native:
            tools.append(_native("tool", native))
    return tools or None


_TOOLS_FIXED: Final = "NeMo Relay cannot add, remove, rename, or reorder ADK tools"


def _tool_groups(
    edited: Sequence[Mapping[str, Json]] | None, original_groups: Sequence[JsonObject]
) -> list[Json] | None:
    """Return ADK's tool groups with edited definitions written in.

    Raise `ValueError` when edit adds, removes, renames, or reorders tools.
    """
    remaining = iter(edited or ())
    groups: list[Json] = []
    for group in original_groups:
        rebuilt: JsonObject = {}
        declarations: list[Json] = [
            _declaration(declaration, next(remaining, None))
            for declaration in _objects(group.get("function_declarations"), field="ADK tool function_declarations")
        ]
        if declarations:
            rebuilt["function_declarations"] = declarations
        native = {key: value for key, value in group.items() if key != "function_declarations"}
        if native:
            rebuilt.update(_native_tool(native, next(remaining, None)))
        groups.append(rebuilt)
    if next(remaining, None) is not None:
        raise ValueError(_TOOLS_FIXED)
    return groups or None


def _declaration(original: JsonObject, edited: Mapping[str, Json] | None) -> JsonObject:
    """Return function declaration with edited description, parameters, and provider fields.

    Raise `ValueError` when edit is not a function tool or renames the function.
    """
    if edited is None or edited.get("type") != "function":
        raise ValueError(_TOOLS_FIXED)
    function = edited.get("function")
    if not _is_object(function) or function.get("name") != original.get("name"):
        raise ValueError(_TOOLS_FIXED)
    if set(edited) - {"type", "function"}:
        raise ValueError("NeMo Relay cannot write fields of a function-tool wrapper back to an ADK request")
    declaration: JsonObject = {"name": _name(original, field="ADK function declaration")}
    description = function.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise ValueError("an ADK tool description must be a string")
        declaration["description"] = description
    parameters = function.get("parameters")
    if parameters is not None:
        declaration["parameters_json_schema" if "parameters_json_schema" in original else "parameters"] = parameters
    for key, value in function.items():
        if key not in {"name", "description", "parameters"}:
            declaration[key] = value
    return declaration


def _native_tool(original: JsonObject, edited: Mapping[str, Json] | None) -> JsonObject:
    """Return native tool with its edited value.

    Raise `ValueError` when edit changes tool's kind.
    """
    if edited is None or edited.get("type") != "provider_native":
        raise ValueError(_TOOLS_FIXED)
    value = edited.get("value")
    if edited.get("provider") != _PROVIDER or edited.get("kind") != "tool" or not _is_object(value):
        raise ValueError("NeMo Relay cannot write that tool edit back to an ADK request")
    if set(value) != set(original):
        raise ValueError(_TOOLS_FIXED)
    return dict(value)


def _tool_choice(tool_config: Json) -> _ToolChoice | JsonObject | None:
    calling = object_field(tool_config, field="ADK tool_config").get("function_calling_config")
    if calling is None:
        return None
    if not _is_object(calling):
        raise TypeError("ADK function_calling_config must be an object")
    mode = calling.get("mode")
    names = calling.get("allowed_function_names")
    if calling.keys() <= {"mode", "allowed_function_names"}:
        if isinstance(mode, str) and mode in _TOOL_CHOICE_MODES and not names:
            return _TOOL_CHOICE_MODES[mode]
        if mode == types.FunctionCallingConfigMode.ANY and not names:
            return "required"
        if mode == types.FunctionCallingConfigMode.ANY and isinstance(names, list) and len(names) == 1:
            return {"type": "function", "function": {"name": str(names[0])}}
    return {"provider": _PROVIDER, "kind": "function_calling_config", "value": calling}


def _function_calling_config(choice: str | Mapping[str, Json] | None) -> JsonObject | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        mode = _TOOL_CHOICE_WRITE_BACK.get(choice)
        if mode is not None:
            return {"mode": mode}
    elif choice.get("kind") == "function_calling_config":
        if choice.get("provider") != _PROVIDER:
            raise ValueError(f"NeMo Relay cannot write a {choice.get('provider')!r} tool choice into an ADK request")
        value = choice.get("value")
        if _is_object(value):
            return dict(value)
    else:
        name = object_field(choice.get("function"), field="tool choice function").get("name")
        if name:
            return {"mode": types.FunctionCallingConfigMode.ANY.value, "allowed_function_names": [str(name)]}
    raise ValueError(f"NeMo Relay cannot write tool choice {choice!r} back to an ADK request")


def _config_remainder(config: JsonObject) -> JsonObject:
    remainder = {key: value for key, value in config.items() if key not in _MODELED_CONFIG_KEYS}
    other = {
        key: value
        for key, value in object_field(config.get("tool_config"), field="ADK tool_config").items()
        if key != "function_calling_config"
    }
    if other:
        remainder["tool_config"] = other
    return remainder


def _set_or_pop(target: JsonObject, key: str, value: Json) -> None:
    if value is None:
        target.pop(key, None)
    else:
        target[key] = value


def _api_data(api_specific: Mapping[str, Json] | None) -> JsonObject:
    if api_specific is None:
        return {}
    api, name = api_specific.get("api"), api_specific.get("api_name")
    if api != "custom" or name != _PROVIDER:
        raise ValueError(f"NeMo Relay cannot write {name or api!r} api_specific fields back to an ADK request")
    data = api_specific.get("data")
    if not _is_object(data):
        raise TypeError("ADK api_specific data must be a JSON object")
    return data


def _write_api_specific(
    payload: JsonObject, config: JsonObject, *, edited: Mapping[str, Json] | None, baseline: Mapping[str, Json] | None
) -> None:
    """Write edited ADK request fields and unmodeled config fields from `api_specific` back onto payload.

    Raise `ValueError` for a field the normalized request models.
    """
    data, original = _api_data(edited), _api_data(baseline)
    for key in sorted(data.keys() | original.keys()):
        if key == "config" or same_json(data.get(key), original.get(key)):
            continue
        if key not in _API_SPECIFIC_REQUEST_KEYS:
            raise ValueError(f"NeMo Relay cannot write {key!r} back to an ADK request")
        _set_or_pop(payload, key, data.get(key))
    edited_config = object_field(data.get("config"), field="api_specific data config")
    original_config = object_field(original.get("config"), field="api_specific data config")
    for key in sorted(edited_config.keys() | original_config.keys()):
        value = edited_config.get(key)
        if same_json(value, original_config.get(key)):
            continue
        if key in _MODELED_CONFIG_KEYS - {"tool_config"}:
            raise ValueError(f"NeMo Relay writes the ADK config field {key!r} through the normalized request")
        if key != "tool_config":
            _set_or_pop(config, key, value)
            continue
        remainder = object_field(value, field="ADK tool_config")
        if "function_calling_config" in remainder:
            raise ValueError("NeMo Relay writes the ADK function_calling_config through tool_choice")
        tool_config: JsonObject = dict(remainder)
        calling = object_field(config.get("tool_config"), field="ADK tool_config").get("function_calling_config")
        if calling is not None:
            tool_config["function_calling_config"] = calling
        _set_or_pop(config, "tool_config", tool_config or None)


def token_counts(usage_metadata: JsonObject, fields: Sequence[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source, target in fields:
        value = usage_metadata.get(source)
        if value is None:
            continue
        if not isinstance(value, int):
            raise TypeError(f"ADK usage_metadata {source} must be an integer")
        counts[target] = value
    return counts


def _finish_reason(raw: Json, *, has_tool_calls: bool) -> _FinishReason | JsonObject | None:
    reason = _optional_text(raw, field="ADK finish_reason")
    if reason == types.FinishReason.FINISH_REASON_UNSPECIFIED.value:
        reason = None
    if has_tool_calls and reason in (None, types.FinishReason.STOP.value):
        return "tool_use"
    if reason is None:
        return None
    if (known := _FINISH_REASONS.get(reason)) is not None:
        return known
    return {"unknown": reason}


def _decode(payload: JsonObject) -> AnnotatedLLMRequest:
    config = object_field(payload.get("config"), field="ADK config")
    contents = _objects(payload.get("contents"), field="ADK contents")
    messages = [message for content in contents for message in _content_messages(content)]
    params = {target: config[source] for source, target in _PARAMS if source in config}
    data: JsonObject = {key: value for key, value in payload.items() if key in _API_SPECIFIC_REQUEST_KEYS}
    if remainder := _config_remainder(config):
        data["config"] = remainder
    extra = {key: value for key, value in payload.items() if key not in _REQUEST_FIELDS}
    return AnnotatedLLMRequest(
        messages,
        instructions=_instructions(config.get("system_instruction")),
        model=_optional_text(payload.get("model"), field="ADK model"),
        params=params or None,
        tools=_tools(config.get("tools")),
        tool_choice=_tool_choice(config.get("tool_config")),
        api_specific={"api": "custom", "api_name": _PROVIDER, "data": data} if data else None,
        extra=extra or None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Normalized:
    """Usage, cost, and `api_specific` an API variant reads from an ADK response dump, plus keys it consumed."""

    usage: dict[str, int]
    cost: Annotated[Json, "Cost provider reported, or None."]
    api_specific: JsonObject
    usage_keys: Annotated[AbstractSet[str], "`usage_metadata` keys variant consumed."]
    response_keys: Annotated[AbstractSet[str], "Response keys variant consumed beyond modeled ones."]
    recorded_keys: Annotated[AbstractSet[str], "Recorded keys variant consumed beyond id."]


class ApiVariant(Protocol):
    """Read an ADK response dump by one provider API's usage semantics."""

    def __call__(self, response: JsonObject, recorded: JsonObject, usage_metadata: JsonObject, /) -> Normalized: ...


class AdkCodec(LlmCodec, LlmResponseCodec):
    """Translate plain dumps of ADK's `LlmRequest` and `LlmResponse` for intercepts and events.

    `variant` reads each response by its provider API's usage semantics.
    """

    def __init__(self, variant: ApiVariant) -> None:
        self._variant = variant

    @override
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        return _decode(request.content)

    @override
    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        payload = original.content
        baseline = _decode(payload)
        config = object_field(payload.get("config"), field="ADK config")
        messages, baseline_messages = annotated.messages, baseline.messages
        if not same_json(messages, baseline_messages):
            originals = _objects(payload.get("contents"), field="ADK contents")
            contents = _patched_contents(messages, baseline_messages, originals)
            payload["contents"] = [*contents]
        if not same_json(annotated.instructions, baseline.instructions):
            _set_or_pop(config, "system_instruction", _system_instruction(annotated.instructions))
        if not same_json(annotated.model, baseline.model):
            _set_or_pop(payload, "model", annotated.model)
        params: Mapping[str, Json] = annotated.params or {}
        baseline_params: Mapping[str, Json] = baseline.params or {}
        for source, target in _PARAMS:
            if not same_json(params.get(target), baseline_params.get(target)):
                _set_or_pop(config, source, params.get(target))
        if not same_json(annotated.tool_choice, baseline.tool_choice):
            tool_config = dict(object_field(config.get("tool_config"), field="ADK tool_config"))
            _set_or_pop(tool_config, "function_calling_config", _function_calling_config(annotated.tool_choice))
            _set_or_pop(config, "tool_config", tool_config or None)
        if not same_json(annotated.tools, baseline.tools):
            groups = _objects(config.get("tools"), field="ADK config tools")
            _set_or_pop(config, "tools", _tool_groups(annotated.tools, groups))
        if not same_json(annotated.api_specific, baseline.api_specific):
            _write_api_specific(payload, config, edited=annotated.api_specific, baseline=baseline.api_specific)
        if not same_json(annotated.extra, baseline.extra):
            raise ValueError("ADK requests have no fields outside LlmRequest for NeMo Relay to write extra edits to")
        _set_or_pop(payload, "config", config or None)
        return LLMRequest(original.headers, payload)

    @override
    def decode_response(self, response: Json) -> AnnotatedLLMResponse:
        if not isinstance(response, dict):
            raise TypeError("an ADK response payload must be an LlmResponse dump")
        parts = _parts_of(response)
        calls = _function_parts(parts, "function_call")
        ids = _function_ids(calls)
        tool_calls = [
            {
                "id": ids[index],
                "name": _name(call, field="ADK function_call"),
                "arguments": object_field(call.get("args"), field="ADK function_call args"),
            }
            for index, call in calls
        ]
        usage_metadata = object_field(response.get("usage_metadata"), field="ADK usage_metadata")
        recorded = object_field(response.get(RECORDED_KEY), field=f"ADK response {RECORDED_KEY}")
        read = self._variant(response, recorded, usage_metadata)
        usage: dict[str, Json] = dict(read.usage)
        if read.cost is not None:
            usage["cost"] = read.cost
        modeled = _MODELED_RESPONSE_KEYS | read.response_keys
        extra = {key: value for key, value in response.items() if key not in modeled}
        if usage_details := {key: value for key, value in usage_metadata.items() if key not in read.usage_keys}:
            extra["usage_metadata"] = usage_details
        consumed = {RECORDED_ID_KEY} | read.recorded_keys
        if unread := {key: value for key, value in recorded.items() if key not in consumed}:
            extra[RECORDED_KEY] = unread
        identifier = recorded.get(RECORDED_ID_KEY)
        return AnnotatedLLMResponse(
            id=_optional_text(
                response.get("interaction_id") if identifier is None else identifier, field="ADK response id"
            ),
            model=_optional_text(response.get("model_version"), field="ADK model_version"),
            message=_message_content(parts),
            tool_calls=tool_calls or None,
            finish_reason=_finish_reason(response.get("finish_reason"), has_tool_calls=bool(tool_calls)),
            usage=usage or None,
            api_specific=read.api_specific or None,
            extra=extra or None,
        )


def with_uncached(counts: Mapping[str, int]) -> dict[str, int]:
    """Return counts plus explicit `uncached_input_tokens`, for a prompt count that includes cache reads.

    Omit it when cache reads exceed prompt.
    """
    usage = dict(counts)
    prompt, cached = counts.get("prompt_tokens"), counts.get("cache_read_tokens")
    if prompt is not None and cached is not None and cached <= prompt:
        usage["uncached_input_tokens"] = prompt - cached
    return usage


def without_cache(prompt: int, cached: int | None, written: int | None, *, folded: str) -> int:
    """Return `prompt` minus `cached` and `written`.

    Args:
        prompt: Prompt token count, cache tokens included.
        cached: Cache read tokens, or None.
        written: Cache write tokens, or None.
        folded: Name for cache tokens in the error message.

    Raises:
        ValueError: `cached` plus `written` exceeds `prompt`.
    """
    uncached = prompt - (cached or 0) - (written or 0)
    if uncached < 0:
        raise ValueError(f"ADK prompt_token_count must include {folded}")
    return uncached


def reported_cost(recorded: JsonObject, *, codec: LlmResponseCodec) -> Json:
    """Return cost `codec` reads from recorded provider usage.

    None when no usage is recorded or usage reports no cost.
    """
    usage = recorded.get(RECORDED_USAGE_KEY)
    if usage is None:
        return None
    decoded = codec.decode_response({"usage": usage}).usage
    return None if decoded is None else decoded.get("cost")


def python_dump(value: object) -> object:
    """Return value as pydantic dumps it in Python mode. Keep each value with no JSON form as is."""
    return _PYTHON_VALUES.dump_python(value, mode="python", by_alias=True, exclude_none=True, serialize_as_any=True)


def json_safe(value: object) -> Json:
    """Project a value to JSON for observation.

    Replace binary data with an object holding its byte count under `bytes`, and sort set members by JSON. Replace an
    iterator or a value with no JSON form with its stand-in text.
    """
    return _jsonable(_ordered_and_counted(python_dump(value)))


def _jsonable(value: object) -> Json:
    return to_jsonable_python(value, fallback=unserializable)


def _ordered_and_counted(value: object) -> object:
    if isinstance(value, Iterator):
        return unserializable(value)
    if isinstance(value, bytes | bytearray):
        return {"bytes": len(value)}
    if isinstance(value, Mapping):
        return {key: _ordered_and_counted(item) for key, item in value.items()}
    if isinstance(value, AbstractSet):
        return sorted((_jsonable(_ordered_and_counted(item)) for item in value), key=canonical)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_ordered_and_counted(item) for item in value]
    return value
