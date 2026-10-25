# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-gRPC per-method isolation, including streaming chunks and both directions."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

grpc = pytest.importorskip("grpc")
from nemo_relay_plugin import _api as api  # noqa: E402
from nemo_relay_plugin._message_limits import (  # noqa: E402
    LIMITS_BY_NAME_ENV,
    MessageLimitInterceptor,
    named_message_limits,
)

MIB = 1024 * 1024


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "null",
        "[]",
        "1",
        '{"Invoke":true}',
        '{"Invoke":0}',
        '{"Invoke":-1}',
        '{"Invoke":2147483648}',
        '{"Invoke":"8"}',
        '{"Invoke":1.5}',
        '{"":8}',
        '{" Invoke":8}',
        '{"1Invoke":8}',
        '{"Invöke":8}',
        '{"Invoke/Other":8}',
    ],
)
def test_invalid_named_limits(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject malformed maps without echoing their supplied content."""
    monkeypatch.setenv(LIMITS_BY_NAME_ENV, raw)
    with pytest.raises(ValueError, match=LIMITS_BY_NAME_ENV):
        named_message_limits()


def test_named_limit_defaults_and_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept absent, empty, and finite positive per-RPC maps."""
    monkeypatch.delenv(LIMITS_BY_NAME_ENV, raising=False)
    assert named_message_limits() == {}
    for values in ({}, {"Invoke": 1, "ToolNext": 2_147_483_647}):
        monkeypatch.setenv(LIMITS_BY_NAME_ENV, json.dumps(values))
        assert named_message_limits() == values


def result_value(request: Any, worker: bool) -> Any:
    """Generate a response independently of request size when requested."""
    envelope = request.tool.value if worker else request.request
    value = api._decode_required_envelope(envelope, "request")
    return {"text": "x" * value["response_bytes"]} if "response_bytes" in value else value


class WorkerPeer(api.pb_grpc.PluginWorkerServicer):
    """Permissive synthetic worker; production codec/interceptor supplies the limit."""

    async def Invoke(self, request: Any, context: Any) -> Any:
        return api.pb.InvokeResponse(
            json=api.pb.JsonResult(value=api._json_envelope(api.JSON_SCHEMA, result_value(request, True)))
        )

    async def InvokeStream(self, request: Any, context: Any) -> Any:
        yield api.pb.StreamChunk(value=api._json_envelope(api.JSON_SCHEMA, result_value(request, True)))

    async def Health(self, request: Any, context: Any) -> Any:
        return api.pb.HealthResponse(ok=True)


class HostPeer(api.pb_grpc.RelayHostRuntimeServicer):
    """Permissive continuation peer isolates the maintained SDK channel limits."""

    async def LlmNext(self, request: Any, context: Any) -> Any:
        return api.pb.JsonResult(value=api._json_envelope(api.JSON_SCHEMA, result_value(request, False)))

    async def LlmStreamNext(self, request: Any, context: Any) -> Any:
        yield api.pb.StreamChunk(value=api._json_envelope(api.JSON_SCHEMA, result_value(request, False)))

    async def Log(self, request: Any, context: Any) -> Any:
        return api.pb.HostAck(ok=True)


@asynccontextmanager
async def method_peer(limits: dict[str, int], enforce_server: bool) -> Any:
    """Serve real unary and stream RPCs with a permissive 16 MiB outer transport."""
    server = grpc.aio.server(
        options=api._worker_grpc_options(16 * MIB),
        interceptors=(MessageLimitInterceptor(4 * MIB, limits),) if enforce_server else (),
    )
    api.pb_grpc.add_PluginWorkerServicer_to_server(WorkerPeer(), server)
    api.pb_grpc.add_RelayHostRuntimeServicer_to_server(HostPeer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await server.stop(grace=0)


@pytest.mark.parametrize("worker", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("direction", ["request", "response"])
@pytest.mark.parametrize("enforce_server", [True, False])
async def test_named_rpc_bounds_are_independent(
    worker: bool, stream: bool, direction: str, enforce_server: bool
) -> None:
    """A raised RPC accepts 5 MiB, rejects 9 MiB, and never widens a sibling RPC."""
    rpc = ("InvokeStream" if stream else "Invoke") if worker else ("LlmStreamNext" if stream else "LlmNext")
    limits = {rpc: 8 * MIB}
    async with method_peer(limits, enforce_server) as endpoint:
        channel = (
            grpc.aio.insecure_channel(api._grpc_target(endpoint), options=api._worker_grpc_options(16 * MIB))
            if enforce_server
            else api._open_host_channel(endpoint, max_message_bytes=4 * MIB, by_name=limits)
        )
        try:
            stub = api.pb_grpc.PluginWorkerStub(channel) if worker else api.pb_grpc.RelayHostRuntimeStub(channel)

            async def call(size: int) -> Any:
                value = {"response_bytes": size} if direction == "response" else {"text": "x" * size}
                request = (
                    api.pb.InvokeRequest(tool=api.pb.ToolInvocation(value=api._json_envelope(api.JSON_SCHEMA, value)))
                    if worker
                    else (api.pb.LlmStreamNextRequest if stream else api.pb.LlmNextRequest)(
                        request=api._json_envelope(api.JSON_SCHEMA, value)
                    )
                )
                if direction == "response":
                    assert request.ByteSize() < 128
                response = getattr(stub, rpc)(request, timeout=10)
                if stream:
                    chunks = [item async for item in response]
                    envelope = chunks[0].value
                else:
                    response = await response
                    envelope = response.json.value if worker else response.value
                return api._decode_required_envelope(envelope, "response")

            assert len((await call(5 * MIB))["text"]) == 5 * MIB
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call(9 * MIB)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            control = stub.Health if worker else stub.Log
            request = (
                api.pb.HealthRequest(auth_token="x" * (5 * MIB))
                if worker
                else api.pb.LogRequest(message="x" * (5 * MIB))
            )
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await control(request, timeout=10)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert (await call(0))["text"] == ""
            assert (await control(api.pb.HealthRequest() if worker else api.pb.LogRequest(), timeout=5)).ok
            if enforce_server:
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await channel.unary_unary("/missing/Unknown")(b"", timeout=5)
                assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
        finally:
            await channel.close()
