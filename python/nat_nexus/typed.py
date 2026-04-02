# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed wrappers for NeMo Agent Toolkit Nexus execute APIs.

Provides generic typed versions of ``tools.execute``, ``llm.execute``, and
``llm.stream_execute`` that use explicit ``Codec[T]`` objects to
serialize/deserialize typed Python objects at the API boundary.

The Rust core remains unchanged -- these wrappers convert typed objects to/from
JSON (``Any``) at the edges so that the middleware pipeline operates on plain
JSON as before.

Example with a custom codec::

    import nat_nexus.typed as typed
    from nat_nexus.typed import Codec

    class PointCodec(Codec[Point]):
        def to_json(self, value: Point) -> dict:
            return {"x": value.x, "y": value.y}
        def from_json(self, data: dict) -> Point:
            return Point(data["x"], data["y"])

    result = await typed.tool_execute(
        "scale", Point(1, 2), my_func,
        args_codec=PointCodec(), result_codec=PointCodec(),
    )

Built-in codecs:

- ``JsonPassthrough``  -- identity, no conversion (the default)
- ``PydanticCodec(ModelClass)`` -- uses ``model_dump()`` / ``model_validate()``
- ``DataclassCodec(DataclassClass)`` -- uses ``dataclasses.asdict()`` / ``cls(**data)``
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import json
import pickle
import typing
import weakref
from typing import Any, AsyncIterator, Awaitable, Callable, Generic, TypeVar, overload

from nat_nexus import llm, tools
from nat_nexus._native import LLMRequest, LlmStream, ScopeHandle

Json = Any

T = TypeVar("T")
TArgs = TypeVar("TArgs")
TResult = TypeVar("TResult")
TResponse = TypeVar("TResponse")
TResponseChunk = TypeVar("TResponseChunk")

_RUNTIME_TYPE_REGISTRY: weakref.WeakValueDictionary[str, type[Any]] = weakref.WeakValueDictionary()


def _register_runtime_type(type_obj: type[Any]) -> str:
    token = f"{type_obj.__module__}.{type_obj.__qualname__}:{id(type_obj)}"
    _RUNTIME_TYPE_REGISTRY[token] = type_obj
    return token


def _resolve_runtime_type(token: object) -> type[Any] | None:
    if not isinstance(token, str):
        return None

    resolved = _RUNTIME_TYPE_REGISTRY.get(token)
    return resolved if isinstance(resolved, type) else None


def _resolve_importable_type(path: object) -> type[Any] | None:
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
    """Conversion protocol between a typed value ``T`` and JSON (``Any``).

    Subclass and override ``to_json`` / ``from_json`` to provide custom
    serialization for your domain types.
    """

    def to_json(self, value: T) -> Json:
        """Convert a typed value to a JSON-serializable object."""
        raise NotImplementedError

    def from_json(self, data: Json) -> T:
        """Reconstruct a typed value from a JSON-serializable object."""
        raise NotImplementedError


class JsonPassthrough(Codec[Any]):
    """Identity codec -- no conversion, values pass through unchanged.

    This is the default codec when none is specified.
    """

    def to_json(self, value: Any) -> Json:
        return value

    def from_json(self, data: Json) -> Any:
        return data


class PydanticCodec(Codec[T]):
    """Codec for Pydantic ``BaseModel`` subclasses.

    Uses ``model_dump()`` for serialization and ``model_validate()`` for
    deserialization.  Does **not** import Pydantic itself -- it only calls
    methods on user-provided model instances/classes.

    Args:
        model_cls: The Pydantic model class.
    """

    def __init__(self, model_cls: type[T]) -> None:
        self._cls = model_cls

    def to_json(self, value: T) -> Json:
        """Serialize a Pydantic model to a JSON-serializable dict via ``model_dump()``."""
        return value.model_dump()  # type: ignore[union-attr]

    def from_json(self, data: Json) -> T:
        """Deserialize a dict into a Pydantic model via ``model_validate()``."""
        return self._cls.model_validate(data)  # type: ignore[attr-defined]


class DataclassCodec(Codec[T]):
    """Codec for ``dataclasses.dataclass`` types.

    Uses ``dataclasses.asdict()`` for serialization and ``cls(**data)`` for
    deserialization.

    Args:
        dc_cls: The dataclass class.
    """

    def __init__(self, dc_cls: type[T]) -> None:
        self._cls = dc_cls

    def to_json(self, value: T) -> Json:
        """Serialize a dataclass instance to a dict via ``dataclasses.asdict()``."""
        return dataclasses.asdict(value)  # type: ignore[arg-type]

    def from_json(self, data: Json) -> T:
        """Deserialize a dict into a dataclass instance via ``cls(**data)``."""
        return self._cls(**data)


class BestEffortAnyCodec(Codec[Any]):
    """
    Bidirectional (as far as possible) lossless JSON codec for arbitrary Python values.

    Tries:
    1. If it's a dataclass, serialize/deserialize as dict.
    2. If it's a Pydantic BaseModel, use .model_dump() / .model_validate().
    3. Try JSON natively.
    4. If all else fails, fallback to pickling (not portable but lossless).
    """

    def to_json(self, value: Any) -> Any:
        """Serialize an arbitrary Python value to a JSON-serializable form.

        Tries, in order: Pydantic ``model_dump()``, ``dataclasses.asdict()``,
        native JSON encoding, pickle fallback, and finally ``str()`` as a last
        resort.  Each encoding is tagged with a ``__nv_*__`` key so that
        ``from_json`` can reconstruct the original type.
        """
        try:
            if hasattr(value, "model_dump"):
                return {
                    "__nv_pydantic__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                    "__nv_runtime_type__": _register_runtime_type(value.__class__),
                    "data": value.model_dump(mode="json"),
                }
        except Exception:
            pass  # Don't fail if pydantic not available

        # Dataclass
        if dataclasses.is_dataclass(value):
            return {
                "__nv_dataclass__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                "__nv_runtime_type__": _register_runtime_type(value.__class__),
                "data": dataclasses.asdict(value),
            }

        # Try JSON encoding directly
        try:
            return json.loads(json.dumps(value))
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

    def from_json(self, data: Any) -> Any:
        """Reconstruct a Python value from its tagged JSON representation.

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
                        return cls.model_validate(data["data"])
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
    """Execute a tool call with explicit codec-based serialization.

    Converts *args* to JSON via ``args_codec.to_json``, runs the middleware
    pipeline, calls *func* with deserialized typed args (via
    ``args_codec.from_json``), and returns the result deserialized via
    ``result_codec.from_json``.

    Args:
        name: Tool name.
        args: Typed tool arguments.
        func: Async or sync callable ``(typed_args) -> typed_result``.
        args_codec: Codec for args serialization/deserialization.
        result_codec: Codec for result serialization/deserialization.
        handle: Optional parent scope handle.
        attributes: Optional ``ToolAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.

    Returns:
        The typed tool result (deserialized from JSON via *result_codec*).
    """
    json_args = args_codec.to_json(args)

    async def _json_func(json_args_inner: Json) -> Json:
        typed_args = args_codec.from_json(json_args_inner)
        result: TResult | Awaitable[TResult] = func(typed_args)
        if isinstance(result, Awaitable):
            result = await result  # type: ignore[assignment]
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
    response_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
) -> TResponse: ...


@overload
async def llm_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], TResponse],
    response_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
) -> TResponse: ...


async def llm_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], TResponse] | Callable[[LLMRequest], Awaitable[TResponse]],
    response_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
) -> TResponse:
    """Execute an LLM call with explicit codec-based response deserialization.

    The request is an ``LLMRequest`` with headers and content. The response
    is converted via *response_codec*.

    Args:
        name: Model/provider name.
        request: The ``LLMRequest`` object.
        func: Async or sync callable ``(LLMRequest) -> typed_response``.
        response_codec: Codec for response serialization/deserialization.
        handle: Optional parent scope handle.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.
        model_name: Optional model name for ATIF trajectory export.

    Returns:
        The typed LLM response (deserialized from JSON via *response_codec*).
    """

    async def _json_func(request_inner: LLMRequest) -> Json:
        result: TResponse | Awaitable[TResponse] = func(request_inner)
        if isinstance(result, Awaitable):
            result = await result  # type: ignore[assignment]
        return response_codec.to_json(typing.cast(TResponse, result))

    json_result = await llm.execute(
        name,
        request,
        _json_func,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        model_name=model_name,
    )
    return response_codec.from_json(json_result)


async def llm_stream_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[TResponseChunk]],
    collector: Callable[[TResponseChunk], None],
    finalizer: Callable[[], TResponse],
    chunk_codec: Codec[TResponseChunk],
    response_codec: Codec[TResponse],
    *,
    handle: ScopeHandle | None = None,
    attributes: int | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    model_name: str | None = None,
) -> LlmStream:
    """Execute a streaming LLM call with codec-based conversion.

    Individual chunks yielded by *func* are converted to JSON via
    *chunk_codec* before entering the middleware pipeline (stream response
    intercepts operate on plain JSON). After interception, each chunk is
    converted back to ``TResponseChunk`` via *chunk_codec* before being
    passed to *collector*.

    The **finalizer** returns a typed aggregated response which is converted
    to JSON via *response_codec* before flowing through sanitize-response
    guardrails and the END event.

    Args:
        name: Model/provider name.
        request: The ``LLMRequest`` object.
        func: Async callable returning an ``AsyncIterator[TResponseChunk]``
            of typed chunks.
        collector: Called with each typed chunk (after intercepts and
            deserialization via *chunk_codec*).
        finalizer: Called once when the stream is exhausted; returns the
            typed aggregated response.
        chunk_codec: Codec for converting individual stream chunks between
            ``TResponseChunk`` and JSON.
        response_codec: Codec for converting the finalizer's typed result
            to JSON.
        handle: Optional parent scope handle.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional application data.
        metadata: Optional metadata.
        model_name: Optional model name for ATIF trajectory export.

    Returns:
        An ``LlmStream`` async iterator of JSON chunks.
    """

    async def _json_func(request_inner: LLMRequest) -> AsyncIterator[Json]:
        async for typed_chunk in func(request_inner):
            yield chunk_codec.to_json(typed_chunk)

    def _json_collector(json_chunk: Json) -> None:
        collector(chunk_codec.from_json(json_chunk))

    def _json_finalizer() -> Json:
        return response_codec.to_json(finalizer())

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
