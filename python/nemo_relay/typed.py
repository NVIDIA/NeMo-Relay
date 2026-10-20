# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed wrappers around the JSON-based NeMo Relay execution APIs.

The core runtime operates on JSON-like values. This module adds a codec layer
so callers can work with typed Python objects at the boundaries while the
middleware pipeline continues to operate on JSON.

Example::

    from dataclasses import dataclass

    import nemo_relay
    import nemo_relay.typed as typed

    @dataclass
    class SearchArgs:
        query: str

    @dataclass
    class SearchResult:
        answer: str

    async def tool_impl(args: SearchArgs) -> nemo_relay.ToolExecutionResult[SearchResult]:
        return nemo_relay.ToolExecutionResult(SearchResult(answer=args.query.upper()))

    result = await typed.tool_execute(
        "search",
        SearchArgs(query="hello"),
        tool_impl,
        args_codec=typed.DataclassCodec(SearchArgs),
        result_codec=typed.DataclassCodec(SearchResult),
    )
"""

from __future__ import annotations

import base64
import dataclasses
import inspect
import io
import json
import pickle
import typing
from typing import AsyncIterator, Awaitable, Callable, Generic, Iterable, Protocol, TypeVar, cast, overload

from nemo_relay import Json, llm, tools
from nemo_relay._native import LLMAttributes, LLMRequest, LlmStream, ScopeHandle, ToolAttributes, ToolExecutionResult
from nemo_relay._tool_result import TOOL_RESULT_SCHEMA, projected_tool_result
from nemo_relay.codecs import LlmCodec, LlmResponseCodec

T = TypeVar("T")
TArgs = TypeVar("TArgs")
TResult = TypeVar("TResult")
TResponse = TypeVar("TResponse")
TResponseChunk = TypeVar("TResponseChunk")

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = object


class _SupportsModelDump(Protocol):
    def model_dump(self, *, mode: str = "python") -> Json: ...


class _SupportsModelValidate(Protocol[T]):
    @classmethod
    def model_validate(cls, data: Json) -> T: ...


_NO_RECONSTRUCTION = object()


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickle only the builtin containers used by the fallback codec."""

    _ALLOWED_GLOBALS = {
        ("builtins", "dict"),
        ("builtins", "frozenset"),
        ("builtins", "list"),
        ("builtins", "set"),
        ("builtins", "tuple"),
    }

    def find_class(self, module: str, name: str) -> object:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global '{module}.{name}' is not allowed")
        return super().find_class(module, name)


def _serialize_dataclass_instance(value: DataclassInstance) -> Json:
    return {
        field_info.name: _serialize_value(field_value)
        for field_info in dataclasses.fields(value)
        if (field_value := getattr(value, field_info.name)) is not None
    }


def _serialize_value(value: object) -> Json:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _serialize_dataclass_instance(cast(DataclassInstance, value))
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {cast(str, key): _serialize_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return cast(_SupportsModelDump, value).model_dump(mode="json")
        except Exception:
            pass  # Intentional: fall through to the default string encoding
    return cast(Json, value)


# ---------------------------------------------------------------------------
# Codec protocol and built-in implementations
# ---------------------------------------------------------------------------


class Codec(Generic[T]):
    """Bidirectional conversion protocol between a Python type and JSON.

    Implementations convert between ergonomic Python objects and the
    JSON-compatible values used by the underlying NeMo Relay runtime.
    """

    def to_json(self, value: T) -> Json:
        """Convert a typed value to a JSON-serializable object.

        Args:
            value: Typed Python object to serialize.

        Returns:
            Json: JSON-compatible representation of ``value``.
        """
        raise NotImplementedError

    def from_json(self, data: Json) -> T:
        """Reconstruct a typed value from a JSON-serializable object.

        Args:
            data: JSON-compatible value to deserialize.

        Returns:
            T: Typed Python object reconstructed from ``data``.
        """
        raise NotImplementedError


class JsonPassthrough(Codec[Json]):
    """Identity codec for callers already working with JSON values."""

    def to_json(self, value: Json) -> Json:
        """Return ``value`` unchanged.

        Args:
            value: JSON-compatible value to forward into the runtime.

        Returns:
            Json: The original ``value``.
        """
        return value

    def from_json(self, data: Json) -> Json:
        """Return ``data`` unchanged.

        Args:
            data: JSON-compatible value produced by the runtime.

        Returns:
            Json: The original ``data``.
        """
        return data


class PydanticCodec(Codec[T]):
    """Codec for models exposing ``model_dump`` and ``model_validate``.

    Args:
        model_cls: Pydantic-compatible model class implementing
            ``model_validate()``.

    Example::

        codec = PydanticCodec(MyModel)
    """

    def __init__(self, model_cls: type[T]) -> None:
        self._cls = model_cls

    def to_json(self, value: T) -> Json:
        """Serialize a Pydantic model to a JSON-serializable dict.

        Args:
            value: Pydantic model instance to serialize.

        Returns:
            Json: JSON-compatible dictionary produced by ``model_dump()``.
        """
        return cast(_SupportsModelDump, value).model_dump(mode="json")

    def from_json(self, data: Json) -> T:
        """Deserialize JSON data into a Pydantic model.

        Args:
            data: JSON-compatible value to validate.

        Returns:
            T: Model instance produced by ``model_validate()``.
        """
        return cast(_SupportsModelValidate[T], self._cls).model_validate(data)


class DataclassCodec(Codec[T]):
    """Codec for ``dataclasses.dataclass`` models.

    Args:
        dc_cls: Dataclass type to serialize and deserialize.

    Example::

        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        codec = DataclassCodec(Point)
    """

    def __init__(self, dc_cls: type[T]) -> None:
        self._cls = dc_cls

    def to_json(self, value: T) -> Json:
        """Serialize a dataclass instance to a JSON-compatible dictionary.

        Args:
            value: Dataclass instance to serialize.

        Returns:
            Json: JSON-compatible dictionary representation of ``value``.
        """
        return _serialize_dataclass_instance(cast(DataclassInstance, value))

    def from_json(self, data: Json) -> T:
        """Deserialize JSON data into a dataclass instance.

        Args:
            data: JSON-compatible mapping used as keyword arguments for the
                dataclass constructor.

        Returns:
            T: Dataclass instance reconstructed from ``data``.
        """
        return cast(Callable[..., T], self._cls)(**cast(dict[str, object], data))


class BestEffortAnyCodec(Codec[object]):
    """Best-effort codec for arbitrary Python values.

    It tries JSON, dataclasses, and Pydantic models first. It falls back to
    pickle or strings when needed. Pickle decoding allows only a small set of
    Python built-in containers. It never restores arbitrary classes or calls
    other globals through pickle.

    Dataclass and Pydantic tags remain JSON dictionaries by default. To restore
    objects, list their classes in ``allowed_types``. Each codec has its own
    list. Encoding a value does not add its type to the list.

    Set ``tool_result=True`` to wrap the encoded value in Relay's tool-result
    JSON format. The wrapper includes display text or content parts when
    available. It does not guess whether the tool failed. Both settings can
    decode wrappers and older values.

    Args:
        allowed_types: Class objects that this codec can restore from dataclass
            or Pydantic tags. Decoding never imports modules to find a class.
            Allowed constructors and validators run on the input data. Choose
            classes that can safely handle the data you expect.
        tool_result: Wrap encoded values in the tool-result JSON format.

    Notes:
        Encoding falls back to a string if other methods fail. If decoding
        cannot restore an object, it returns the object's JSON value.
    """

    def __init__(self, *, allowed_types: Iterable[type[object]] = (), tool_result: bool = False) -> None:
        self._tool_result = tool_result
        self._allowed_types: dict[str, type[object]] = {}
        for cls in allowed_types:
            if not isinstance(cls, type):
                raise TypeError("allowed_types must contain class objects")
            name = f"{cls.__module__}.{cls.__qualname__}"
            if name in self._allowed_types and self._allowed_types[name] is not cls:
                raise ValueError(f"allowed_types contains distinct classes named {name!r}")
            self._allowed_types[name] = cls

    def to_json(self, value: object) -> Json:
        """Convert a Python value to JSON.

        Args:
            value: Python value to encode.

        Returns:
            Json: Encoded value, with a tool-result wrapper if
            ``tool_result=True``.

        The codec first tries Pydantic ``model_dump()``, then dataclass fields.
        For other values, it tries JSON, then pickle, then ``str()``. Dataclass,
        Pydantic, pickle, and string fallback values use ``__nv_*__`` tags.
        Plain JSON values stay untagged.

        With ``tool_result=True``, the codec places the encoded value in the
        wrapper's ``result`` field.
        """
        encoded = self._encode_value(value)
        if not self._tool_result or projected_tool_result(encoded) is not None:
            return encoded
        content: Json = None
        if isinstance(encoded, str):
            content = encoded
        elif isinstance(encoded, list) and all(
            isinstance(part, str) or (isinstance(part, dict) and part.get("type") in ("text", "image"))
            for part in encoded
        ):
            content = [{"type": "text", "text": part} if isinstance(part, str) else part for part in encoded]
        return {"schema": TOOL_RESULT_SCHEMA, "content": content, "is_error": False, "result": encoded}

    def _encode_value(self, value: object) -> Json:
        """Encode the value before adding any tool-result wrapper."""
        try:
            if hasattr(value, "model_dump"):
                dumped = cast(_SupportsModelDump, value).model_dump(mode="json")
                return {
                    "__nv_pydantic__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                    "data": dumped,
                }
        except Exception:
            pass  # Intentional: fall through to the next encoding strategy

        # Dataclass
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "__nv_dataclass__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                "data": _serialize_dataclass_instance(cast(DataclassInstance, value)),
            }

        # Try JSON encoding directly
        try:
            return cast(Json, json.loads(json.dumps(value)))
        except Exception:
            # Fallback: pickle
            try:
                pickled = pickle.dumps(value)
                encoded = base64.b64encode(pickled).decode("ascii")
                return {
                    "__nv_pickle__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                    "data": encoded,
                }
            except Exception:
                # As last resort, do string (may be lossy, but not error)
                return {
                    "__nv_fallback_str__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                    "data": str(value),
                }

    def _resolve_tagged_type(self, data: dict[str, Json], tag: str) -> type[object] | None:
        name = data.get(tag)
        return self._allowed_types.get(name) if isinstance(name, str) else None

    @staticmethod
    def _mapping_payload(data: dict[str, Json]) -> dict[str, object] | None:
        payload = data.get("data")
        if isinstance(payload, dict):
            return cast(dict[str, object], payload)
        return None

    @staticmethod
    def _string_payload(data: dict[str, Json]) -> str | None:
        payload = data.get("data")
        return payload if isinstance(payload, str) else None

    def _reconstruct_pydantic(self, data: dict[str, Json]) -> object:
        if "__nv_pydantic__" not in data:
            return _NO_RECONSTRUCTION

        try:
            cls = self._resolve_tagged_type(data, "__nv_pydantic__")
            if cls is not None and hasattr(cls, "model_validate"):
                return cast(_SupportsModelValidate[object], cls).model_validate(data["data"])
        except Exception:
            pass
        return _NO_RECONSTRUCTION

    def _reconstruct_dataclass(self, data: dict[str, Json]) -> object:
        if "__nv_dataclass__" not in data:
            return _NO_RECONSTRUCTION

        try:
            cls = self._resolve_tagged_type(data, "__nv_dataclass__")
            payload = self._mapping_payload(data)
            if cls is not None and dataclasses.is_dataclass(cls) and payload is not None:
                return cls(**payload)
        except Exception:
            pass
        return _NO_RECONSTRUCTION

    @staticmethod
    def _reconstruct_pickle(data: dict[str, Json]) -> object:
        if "__nv_pickle__" not in data:
            return _NO_RECONSTRUCTION

        try:
            payload = BestEffortAnyCodec._string_payload(data)
            if payload is None:
                return _NO_RECONSTRUCTION

            decoded = base64.b64decode(payload, validate=True)
            return _RestrictedUnpickler(io.BytesIO(decoded)).load()
        except Exception:
            return _NO_RECONSTRUCTION

    @staticmethod
    def _reconstruct_fallback_string(data: dict[str, Json]) -> object:
        if "__nv_fallback_str__" in data:
            payload = BestEffortAnyCodec._string_payload(data)
            if payload is not None:
                return payload
        return _NO_RECONSTRUCTION

    def from_json(self, data: Json) -> object:
        """Restore a Python value from JSON.

        Args:
            data: Encoded value or plain JSON value.

        Returns:
            object: The restored value, or its JSON value if decoding fails.

        The codec reads the ``__nv_pydantic__``, ``__nv_dataclass__``,
        ``__nv_pickle__``, and ``__nv_fallback_str__`` tags to choose how to
        restore a value. Dataclass and Pydantic types must be in
        ``allowed_types``. The codec ignores old runtime type tokens. Pickle
        decoding allows only a small set of Python built-in containers.

        For a valid tool-result wrapper, the codec reads ``result`` and uses
        the same rules. Content and failure fields do not allow new classes.
        Wrappers with unknown versions or invalid fields stay as JSON.
        Values without recognized tags also stay as JSON.
        """
        if isinstance(data, dict) and data.get("schema") == TOOL_RESULT_SCHEMA:
            if projected_tool_result(data) is None:
                return data
            data = data["result"]
        if isinstance(data, dict) and "data" in data:
            for reconstructor in (
                self._reconstruct_pydantic,
                self._reconstruct_dataclass,
                self._reconstruct_pickle,
                self._reconstruct_fallback_string,
            ):
                result = reconstructor(data)
                if result is not _NO_RECONSTRUCTION:
                    return result

        return data


# ---------------------------------------------------------------------------
# Typed execute wrappers
# ---------------------------------------------------------------------------


@overload
async def tool_execute(
    name: str,
    args: TArgs,
    func: Callable[[TArgs], Awaitable[ToolExecutionResult[TResult]]],
    args_codec: Codec[TArgs],
    result_codec: Codec[TResult],
    *,
    handle: ScopeHandle | None = None,
    attributes: ToolAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    tool_call_id: str | None = None,
) -> ToolExecutionResult[TResult]: ...


@overload
async def tool_execute(
    name: str,
    args: TArgs,
    func: Callable[[TArgs], ToolExecutionResult[TResult]],
    args_codec: Codec[TArgs],
    result_codec: Codec[TResult],
    *,
    handle: ScopeHandle | None = None,
    attributes: ToolAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    tool_call_id: str | None = None,
) -> ToolExecutionResult[TResult]: ...


async def tool_execute(
    name: str,
    args: TArgs,
    func: Callable[[TArgs], ToolExecutionResult[TResult]] | Callable[[TArgs], Awaitable[ToolExecutionResult[TResult]]],
    args_codec: Codec[TArgs],
    result_codec: Codec[TResult],
    *,
    handle: ScopeHandle | None = None,
    attributes: ToolAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    tool_call_id: str | None = None,
) -> ToolExecutionResult[TResult]:
    """Run ``nemo_relay.tools.execute`` with typed arguments and results.

    Args:
        name: Tool name recorded on emitted lifecycle events.
        args: Typed arguments to serialize before entering the runtime.
        func: Tool implementation invoked with deserialized typed arguments.
            It returns ``ToolExecutionResult[TResult]`` and may be synchronous
            or asynchronous.
        args_codec: Codec used to convert ``args`` to and from JSON.
        result_codec: Codec used to convert the tool result to and from JSON.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native tool attributes attached to the start event.
        data: Optional JSON payload recorded on the emitted start event.
        metadata: Optional JSON metadata recorded on the emitted start event.
        tool_call_id: Optional provider-specific tool call identifier recorded
            on the emitted start and end events.

    Returns:
        ToolExecutionResult[TResult]: The decoded typed result and its opaque
        annotation.

    Example::

        from dataclasses import dataclass

        import nemo_relay
        from nemo_relay.typed import DataclassCodec, tool_execute

        @dataclass
        class SearchArgs:
            query: str

        @dataclass
        class SearchResult:
            answer: str

        async def tool_impl(args: SearchArgs):
            return nemo_relay.ToolExecutionResult(SearchResult(answer=args.query.upper()))

        result = await tool_execute(
            "search",
            SearchArgs(query="hello"),
            tool_impl,
            args_codec=DataclassCodec(SearchArgs),
            result_codec=DataclassCodec(SearchResult),
            handle=None,
            attributes=None,
            data={"path": "typed"},
            metadata={"request_id": "req-1"},
        )
    """
    json_args = args_codec.to_json(args)

    async def _json_func(json_args_inner: Json) -> ToolExecutionResult[Json]:
        typed_args = args_codec.from_json(json_args_inner)
        result = func(typed_args)
        if inspect.isawaitable(result):
            result = await typing.cast(Awaitable[ToolExecutionResult[TResult]], result)
        if not isinstance(result, ToolExecutionResult):
            raise TypeError("typed tool callback must return ToolExecutionResult")
        return ToolExecutionResult(result_codec.to_json(result.result), result.annotation)

    json_result = await tools.execute(
        name,
        json_args,
        _json_func,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        tool_call_id=tool_call_id,
    )
    return ToolExecutionResult(result_codec.from_json(json_result.result), json_result.annotation)


@overload
async def llm_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[TResponse]],
    response_json_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: LLMAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> TResponse: ...


@overload
async def llm_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], TResponse],
    response_json_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: LLMAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> TResponse: ...


async def llm_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], TResponse] | Callable[[LLMRequest], Awaitable[TResponse]],
    response_json_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: LLMAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> TResponse:
    """Run ``nemo_relay.llm.execute`` and decode the returned response type.

    Args:
        name: Provider or logical call name recorded on emitted events.
        request: Raw ``LLMRequest`` passed through the managed LLM pipeline.
        func: Provider callback invoked with the possibly intercepted request.
            The implementation may be synchronous or asynchronous.
        response_json_codec: Codec used to convert the provider response to and
            from JSON.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native LLM attributes attached to the start event.
        data: Optional JSON payload recorded on the emitted start event.
        metadata: Optional JSON metadata recorded on the emitted start event.
        model_name: Optional normalized model name to record separately from the
            provider-specific request payload.
        codec: Optional request codec used to expose ``AnnotatedLLMRequest`` to
            request intercepts.
        response_codec: Optional observability codec used to attach an
            annotated response to the emitted ``LLMEnd`` event.

    Returns:
        TResponse: The decoded typed response produced by ``func``.

    Example::

        import nemo_relay
        from dataclasses import dataclass
        from nemo_relay.typed import DataclassCodec, llm_execute

        @dataclass
        class MyResponse:
            text: str

        async def llm_impl(request: nemo_relay.LLMRequest):
            return {"text": "hello"}

        typed_response = await llm_execute(
            "demo-provider",
            nemo_relay.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
            llm_impl,
            response_json_codec=DataclassCodec(MyResponse),
            handle=None,
            attributes=None,
            data={"path": "typed"},
            metadata={"request_id": "req-2"},
            model_name="demo-model",
            codec=None,
            response_codec=None,
        )
    """

    async def _json_func(request_inner: LLMRequest) -> Json:
        result: TResponse | Awaitable[TResponse] = func(request_inner)
        if inspect.isawaitable(result):
            return response_json_codec.to_json(await typing.cast(Awaitable[TResponse], result))
        return response_json_codec.to_json(typing.cast(TResponse, result))

    json_result = await llm.execute(
        name,
        request,
        _json_func,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        model_name=model_name,
        codec=codec,
        response_codec=response_codec,
    )
    return response_json_codec.from_json(json_result)


# SONAR_IGNORE_START
async def llm_stream_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[TResponseChunk]],
    collector: Callable[[TResponseChunk], None],
    finalizer: Callable[[], TResponse],
    chunk_json_codec: Codec[TResponseChunk],
    response_json_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: LLMAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> LlmStream:  # SONAR_IGNORE_STOP
    """Run ``nemo_relay.llm.stream_execute`` with typed chunks and final output.

    Args:
        name: Provider or logical call name recorded on emitted events.
        request: Raw ``LLMRequest`` passed through the managed LLM pipeline.
        func: Async generator invoked with the request and yielding typed chunks.
        collector: Callback invoked for each decoded typed chunk after it passes
            through the runtime's streaming intercept chain.
        finalizer: Callback invoked after the stream completes to build the
            final typed response recorded on the emitted ``LLMEnd`` event.
        chunk_json_codec: Codec used to convert streamed chunks to and from
            JSON.
        response_json_codec: Codec used to convert the final aggregated
            response to and from JSON.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native LLM attributes attached to the start event.
        data: Optional JSON payload recorded on the emitted start event.
        metadata: Optional JSON metadata recorded on the emitted start event.
        model_name: Optional normalized model name to record separately from the
            provider-specific request payload.
        codec: Optional request codec used to expose ``AnnotatedLLMRequest`` to
            request intercepts.
        response_codec: Optional observability codec used to attach an
            annotated response to the emitted ``LLMEnd`` event.

    Returns:
        LlmStream: Async iterator that yields the streamed JSON chunks.

    Notes:
        ``collector`` receives typed chunks, but the returned stream still
        yields the raw JSON chunk values produced by the underlying runtime.

    Example::

        import nemo_relay
        from dataclasses import dataclass
        from nemo_relay.typed import DataclassCodec, llm_stream_execute

        @dataclass
        class MyChunk:
            token: str

        @dataclass
        class MyResponse:
            text: str

        collected = []

        async def stream_impl(request: nemo_relay.LLMRequest):
            yield MyChunk(token="hel")
            yield MyChunk(token="lo")

        def collect_chunk(chunk: MyChunk) -> None:
            collected.append(chunk)

        def finish_response() -> MyResponse:
            return MyResponse(text="".join(chunk.token for chunk in collected))

        stream = await llm_stream_execute(
            "demo-provider",
            nemo_relay.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
            stream_impl,
            collector=collect_chunk,
            finalizer=finish_response,
            chunk_json_codec=DataclassCodec(MyChunk),
            response_json_codec=DataclassCodec(MyResponse),
            handle=None,
            attributes=None,
            data={"path": "typed-stream"},
            metadata={"request_id": "req-3"},
            model_name="demo-model",
            codec=None,
            response_codec=None,
        )
        async for chunk in stream:
            print(chunk)
    """

    async def _json_func(request_inner: LLMRequest) -> AsyncIterator[Json]:
        async for typed_chunk in func(request_inner):
            yield chunk_json_codec.to_json(typed_chunk)

    def _json_collector(json_chunk: Json) -> None:
        collector(chunk_json_codec.from_json(json_chunk))

    def _json_finalizer() -> Json:
        return response_json_codec.to_json(finalizer())

    return await llm.stream_execute(
        name,
        request,
        _json_func,
        _json_collector,
        _json_finalizer,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        model_name=model_name,
        codec=codec,
        response_codec=response_codec,
    )


__all__ = [
    "Codec",
    "BestEffortAnyCodec",
    "DataclassCodec",
    "JsonPassthrough",
    "PydanticCodec",
    "tool_execute",
    "llm_execute",
    "llm_stream_execute",
]
