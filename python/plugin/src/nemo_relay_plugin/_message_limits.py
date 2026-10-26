# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Callback registration limits carried across worker RPCs and continuations."""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import json
import os
from dataclasses import dataclass
from typing import Any

grpc: Any = importlib.import_module("grpc")
LIMITS_BY_NAME_ENV = "NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES_BY_NAME"
CALLBACK_METADATA = "nemo-relay-worker-callback-bin"
CALLBACK_NAME: contextvars.ContextVar[str | None] = contextvars.ContextVar("worker_callback_name", default=None)


def named_message_limits() -> dict[str, int]:
    """Read exact callback-name overrides resolved by the host, like timeouts."""
    raw = os.environ.get(LIMITS_BY_NAME_ENV)
    if raw is None:
        return {}
    try:
        values = json.loads(raw)
        if isinstance(values, dict) and all(
            isinstance(name, str)
            and bool(name.strip())
            and name == name.strip()
            and type(limit) is int
            and 1 <= limit <= 2_147_483_647
            for name, limit in values.items()
        ):
            return values
    except (ValueError, UnicodeError):
        pass
    raise ValueError(
        f"{LIMITS_BY_NAME_ENV} must map non-empty callback registration names "
        "without surrounding whitespace to integers in 1..=2147483647"
    )


class RpcHostChannel:
    """Select a bounded channel at call time from the task's callback context."""

    def __init__(self, target: str, default: int, by_name: dict[str, int], authority: tuple[tuple[str, str], ...]):
        self._target = target
        self._default = default
        self._by_name = by_name
        self._authority = authority
        self._channels: dict[int, Any] = {}

    def _channel(self, callback: str | None) -> Any:
        limit = self._by_name.get(callback or "", self._default)
        if limit not in self._channels:
            self._channels[limit] = grpc.aio.insecure_channel(
                self._target,
                options=(
                    ("grpc.max_receive_message_length", limit),
                    ("grpc.max_send_message_length", limit),
                    *self._authority,
                ),
            )
        return self._channels[limit]

    def unary_unary(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._callable("unary_unary", method, args, kwargs)

    def unary_stream(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._callable("unary_stream", method, args, kwargs)

    def _callable(self, kind: str, method: str, args: Any, kwargs: Any) -> Any:
        def call(
            request: Any,
            timeout: Any = None,
            metadata: Any = None,
            credentials: Any = None,
            wait_for_ready: Any = None,
            compression: Any = None,
        ) -> Any:
            callback = CALLBACK_NAME.get()
            forwarded = tuple((key, value) for key, value in (metadata or ()) if key != CALLBACK_METADATA)
            if callback is not None:
                forwarded += ((CALLBACK_METADATA, callback.encode("utf-8")),)
            rpc = getattr(self._channel(callback), kind)(method, *args, **kwargs)
            return rpc(
                request,
                timeout=timeout,
                metadata=forwarded,
                credentials=credentials,
                wait_for_ready=wait_for_ready,
                compression=compression,
            )

        return call

    async def close(self) -> None:
        await asyncio.gather(*(channel.close() for channel in self._channels.values()))


@dataclass(frozen=True)
class _OversizedRequest:
    size: int


class MessageLimitInterceptor(grpc.aio.ServerInterceptor):
    """Reject over-limit bytes before protobuf decoding, and bound each reply/chunk."""

    def __init__(self, default: int, by_name: dict[str, int]):
        self._default = default
        self._by_name = by_name

    async def intercept_service(self, continuation: Any, details: Any) -> Any:
        handler = await continuation(details)
        if handler is None:
            return None
        method = details.method.rsplit("/", 1)[-1]
        worker = "/PluginWorker/" in details.method or ".PluginWorker/" in details.method
        invocation = worker and method in ("Invoke", "InvokeStream")
        raw_name = next((value for key, value in details.invocation_metadata if key == CALLBACK_METADATA), None)
        try:
            callback = raw_name.decode("utf-8") if raw_name is not None else None
        except (UnicodeError, AttributeError):
            callback = None
        limit = self._by_name.get(callback or "", self._default) if not worker or invocation else self._default

        def deserialize(raw: bytes) -> Any:
            return _OversizedRequest(len(raw)) if len(raw) > limit else handler.request_deserializer(raw)

        async def reject(context: Any, size: int) -> None:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"message larger than max ({size} vs. {limit})")

        async def validate(request: Any, context: Any) -> None:
            if isinstance(request, _OversizedRequest):
                await reject(context, request.size)
            if invocation and (
                (callback is not None and callback != request.registration_name)
                or (callback is None and request.registration_name in self._by_name)
            ):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "callback metadata does not match registration_name"
                )

        async def unary(request: Any, context: Any) -> Any:
            await validate(request, context)
            response = await handler.unary_unary(request, context)
            if response.ByteSize() > limit:
                await reject(context, response.ByteSize())
            return response

        async def stream(request: Any, context: Any) -> Any:
            await validate(request, context)
            async for response in handler.unary_stream(request, context):
                if response.ByteSize() > limit:
                    await reject(context, response.ByteSize())
                yield response

        factory = (
            grpc.unary_stream_rpc_method_handler if handler.response_streaming else grpc.unary_unary_rpc_method_handler
        )
        return factory(
            stream if handler.response_streaming else unary,
            request_deserializer=deserialize,
            response_serializer=handler.response_serializer,
        )
