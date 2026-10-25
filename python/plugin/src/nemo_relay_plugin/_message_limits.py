# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-method gRPC message limits without changing worker protocol messages."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from dataclasses import dataclass
from typing import Any

grpc: Any = importlib.import_module("grpc")
LIMITS_BY_NAME_ENV = "NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES_BY_NAME"


def named_message_limits() -> dict[str, int]:
    """Read exact RPC overrides forwarded by the host after manifest resolution."""
    raw = os.environ.get(LIMITS_BY_NAME_ENV)
    if raw is None:
        return {}
    try:
        values = json.loads(raw)
        if isinstance(values, dict) and all(
            isinstance(name, str)
            and name.isascii()
            and name.isalnum()
            and name[0].isalpha()
            and type(limit) is int
            and 1 <= limit <= 2_147_483_647
            for name, limit in values.items()
        ):
            return values
    except (ValueError, UnicodeError):
        pass
    raise ValueError(f"{LIMITS_BY_NAME_ENV} must map gRPC method names to integers in 1..=2147483647")


class RpcHostChannel:
    """Share channels by byte limit while binding each stub method to its own cap."""

    def __init__(self, target: str, default: int, by_name: dict[str, int], authority: tuple[tuple[str, str], ...]):
        self._target = target
        self._default = default
        self._by_name = by_name
        self._authority = authority
        self._channels: dict[int, Any] = {}

    def _channel(self, method: str) -> Any:
        limit = self._by_name.get(method.rsplit("/", 1)[-1], self._default)
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
        return self._channel(method).unary_unary(method, *args, **kwargs)

    def unary_stream(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._channel(method).unary_stream(method, *args, **kwargs)

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
        limit = self._by_name.get(details.method.rsplit("/", 1)[-1], self._default)

        def deserialize(raw: bytes) -> Any:
            return _OversizedRequest(len(raw)) if len(raw) > limit else handler.request_deserializer(raw)

        async def reject(context: Any, size: int) -> None:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"message larger than max ({size} vs. {limit})")

        async def unary(request: Any, context: Any) -> Any:
            if isinstance(request, _OversizedRequest):
                await reject(context, request.size)
            response = await handler.unary_unary(request, context)
            if response.ByteSize() > limit:
                await reject(context, response.ByteSize())
            return response

        async def stream(request: Any, context: Any) -> Any:
            if isinstance(request, _OversizedRequest):
                await reject(context, request.size)
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
