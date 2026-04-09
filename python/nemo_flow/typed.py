# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed wrappers around the JSON-based NeMo Flow execution APIs.

The core runtime operates on JSON-like values. This module adds a codec layer
so callers can work with typed Python objects at the boundaries while the
middleware pipeline continues to operate on JSON.

Example:
    ```python
    from dataclasses import dataclass

    import nemo_flow.typed as typed

    @dataclass
    class SearchArgs:
        query: str

    @dataclass
    class SearchResult:
        answer: str

    async def tool_impl(args: SearchArgs) -> SearchResult:
        return SearchResult(answer=args.query.upper())

    result = await typed.tool_execute(
        "search",
        SearchArgs(query="hello"),
        tool_impl,
        args_codec=typed.DataclassCodec(SearchArgs),
        result_codec=typed.DataclassCodec(SearchResult),
    )
    ```
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import inspect
import json
import pickle
import typing
import weakref
from typing import AsyncIterator, Awaitable, Callable, Generic, Protocol, TypeVar, cast, overload

from nemo_flow import Json, llm, tools
from nemo_flow._native import LLMRequest, LlmStream, ScopeHandle
from nemo_flow.codecs import LlmCodec, LlmResponseCodec

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


_RUNTIME_TYPE_REGISTRY: weakref.WeakValueDictionary[str, type[object]] = weakref.WeakValueDictionary()


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
    return cast(Json, value)


def _register_runtime_type(type_obj: type[object]) -> str:
    token = f"{type_obj.__module__}.{type_obj.__qualname__}:{id(type_obj)}"
    _RUNTIME_TYPE_REGISTRY[token] = type_obj
    return token


def _resolve_runtime_type(token: object) -> type[object] | None:
    if not isinstance(token, str):
        return None

    resolved = _RUNTIME_TYPE_REGISTRY.get(token)
    return resolved if isinstance(resolved, type) else None


def _resolve_importable_type(path: object) -> type[object] | None:
    if not isinstance(path, str) or not path:
        return None

    parts = path.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split_at])
        qualname_parts = parts[split_at:]
        if "<locals>" in qualname_parts:
            continue

        try:
            resolved: object = importlib.import_module(mod_name)
            for part in qualname_parts:
                resolved = getattr(resolved, part)
        except (ImportError, AttributeError):
            continue

        if isinstance(resolved, type):
            return resolved

    return None


# ---------------------------------------------------------------------------
# Codec protocol and built-in implementations
# ---------------------------------------------------------------------------


class Codec(Generic[T]):
    """Bidirectional conversion protocol between a Python type and JSON.

    Implementations convert between ergonomic Python objects and the
    JSON-compatible values used by the underlying NeMo Flow runtime.
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

    Example:
        ```python
        codec = PydanticCodec(MyModel)
        ```
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

    Example:
        ```python
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        codec = DataclassCodec(Point)
        ```
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

    It prefers JSON-native encodings, then dataclass or Pydantic-style model
    support, and finally falls back to pickle or string encoding when needed.

    Notes:
        This codec favors resilience over strictness. If it cannot preserve the
        original type exactly, it degrades to a string instead of raising.
    """

    def to_json(self, value: object) -> Json:
        """Serialize an arbitrary Python value to a JSON-serializable form.

        Args:
            value: Arbitrary Python object to serialize.

        Returns:
            Json: Tagged JSON-compatible value that ``from_json()`` can
            reconstruct on a best-effort basis.

        Tries, in order: Pydantic ``model_dump()``, ``dataclasses.asdict()``,
        native JSON encoding, pickle fallback, and finally ``str()`` as a last
        resort.  Each encoding is tagged with a ``__nv_*__`` key so that
        ``from_json`` can reconstruct the original type.
        """
        try:
            if hasattr(value, "model_dump"):
                dumped = cast(_SupportsModelDump, value).model_dump(mode="json")
                return {
                    "__nv_pydantic__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                    "__nv_runtime_type__": _register_runtime_type(value.__class__),
                    "data": dumped,
                }
        except Exception:
            pass  # Intentional: fall through to the next encoding strategy

        # Dataclass
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "__nv_dataclass__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                "__nv_runtime_type__": _register_runtime_type(value.__class__),
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

    def from_json(self, data: Json) -> object:
        """Reconstruct a Python value from its tagged JSON representation.

        Args:
            data: Tagged JSON-compatible value produced by ``to_json()`` or a
                plain JSON-compatible value.

        Returns:
            object: Best-effort reconstruction of the original Python value.

        Recognises the ``__nv_pydantic__``, ``__nv_dataclass__``,
        ``__nv_pickle__``, and ``__nv_fallback_str__`` tags produced by
        ``to_json`` and dispatches to the appropriate reconstruction strategy.
        Falls through to returning the raw data if no tag is recognised.
        """
        if isinstance(data, dict) and "data" in data:
            if "__nv_pydantic__" in data:
                try:
                    cls = _resolve_runtime_type(data.get("__nv_runtime_type__")) or _resolve_importable_type(
                        data["__nv_pydantic__"]
                    )
                    if cls is not None and hasattr(cls, "model_validate"):
                        return cast(_SupportsModelValidate[object], cls).model_validate(data["data"])
                except Exception:
                    pass  # Fallback on raw dict

            # Try to reconstruct a dataclass
            if "__nv_dataclass__" in data:
                try:
                    cls = _resolve_runtime_type(data.get("__nv_runtime_type__")) or _resolve_importable_type(
                        data["__nv_dataclass__"]
                    )
                    if cls is not None and dataclasses.is_dataclass(cls):
                        return cls(**data["data"])
                except Exception:
                    pass  # Fallback on raw dict

            # Try to reconstruct from pickle
            if "__nv_pickle__" in data:
                try:
                    decoded = base64.b64decode(data["data"])
                    return pickle.loads(decoded)
                except Exception:
                    pass  # Fall through

            # If fallback string exists and data exists, return the string
            if "__nv_fallback_str__" in data:
                # Only the string is recoverable; return as string
                return data["data"]

        return data


# ---------------------------------------------------------------------------
# Typed execute wrappers
# ---------------------------------------------------------------------------


@overload
async def tool_execute(
    name: str,
    args: TArgs,
    func: Callable[[TArgs], Awaitable[TResult]],
    args_codec: Codec[TArgs],
    result_codec: Codec[TResult],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
) -> TResult: ...


@overload
async def tool_execute(
    name: str,
    args: TArgs,
    func: Callable[[TArgs], TResult],
    args_codec: Codec[TArgs],
    result_codec: Codec[TResult],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
) -> TResult: ...


async def tool_execute(
    name: str,
    args: TArgs,
    func: Callable[[TArgs], TResult] | Callable[[TArgs], Awaitable[TResult]],
    args_codec: Codec[TArgs],
    result_codec: Codec[TResult],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
) -> TResult:
    """Run ``nemo_flow.tools.execute`` with typed arguments and results.

    Args:
        name: Tool name recorded on emitted lifecycle events.
        args: Typed arguments to serialize before entering the runtime.
        func: Tool implementation invoked with deserialized typed arguments.
            The implementation may be synchronous or asynchronous.
        args_codec: Codec used to convert ``args`` to and from JSON.
        result_codec: Codec used to convert the tool result to and from JSON.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native tool attributes attached to the start event.
        data: Optional JSON payload recorded on the emitted start event.
        metadata: Optional JSON metadata recorded on the emitted start event.

    Returns:
        TResult: The decoded typed result produced by ``func``.

    Example:
        ```python
        from dataclasses import dataclass

        from nemo_flow.typed import DataclassCodec, tool_execute

        @dataclass
        class SearchArgs:
            query: str

        @dataclass
        class SearchResult:
            answer: str

        async def tool_impl(args: SearchArgs) -> SearchResult:
            return SearchResult(answer=args.query.upper())

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
        ```
    """
    json_args = args_codec.to_json(args)

    async def _json_func(json_args_inner: Json) -> Json:
        typed_args = args_codec.from_json(json_args_inner)
        result: TResult | Awaitable[TResult] = func(typed_args)
        if inspect.isawaitable(result):
            return result_codec.to_json(await typing.cast(Awaitable[TResult], result))
        return result_codec.to_json(typing.cast(TResult, result))

    json_result = await tools.execute(
        name,
        json_args,
        _json_func,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
    )
    return result_codec.from_json(json_result)


@overload
async def llm_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[TResponse]],
    response_json_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
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
    attributes: int | None = None,
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
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> TResponse:
    """Run ``nemo_flow.llm.execute`` and decode the returned response type.

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

    Example:
        ```python
        import nemo_flow
        from dataclasses import dataclass
        from nemo_flow.typed import DataclassCodec, llm_execute

        @dataclass
        class MyResponse:
            text: str

        async def llm_impl(request: nemo_flow.LLMRequest):
            return {"text": "hello"}

        typed_response = await llm_execute(
            "demo-provider",
            nemo_flow.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
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
        ```
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
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> LlmStream:
    """Run ``nemo_flow.llm.stream_execute`` with typed chunks and final output.

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

    Example:
        ```python
        import nemo_flow
        from dataclasses import dataclass
        from nemo_flow.typed import DataclassCodec, llm_stream_execute

        @dataclass
        class MyChunk:
            token: str

        @dataclass
        class MyResponse:
            text: str

        collected = []

        async def stream_impl(request: nemo_flow.LLMRequest):
            yield MyChunk(token="hel")
            yield MyChunk(token="lo")

        def collect_chunk(chunk: MyChunk) -> None:
            collected.append(chunk)

        def finish_response() -> MyResponse:
            return MyResponse(text="".join(chunk.token for chunk in collected))

        stream = await llm_stream_execute(
            "demo-provider",
            nemo_flow.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
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
        ```
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
