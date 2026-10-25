# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real gRPC boundary coverage without a model, credentials, or external services."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

grpc = pytest.importorskip("grpc")

from nemo_relay_plugin import PluginContext, WorkerPlugin, WorkerSdkError, serve_plugin  # noqa: E402
from nemo_relay_plugin import _api as api  # noqa: E402

LIMIT_ENV = "NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES"


@pytest.mark.parametrize("value", ["", "0", "-1", "+1", " 10", "10 ", "1.0", "16MiB", "2147483648", "９"])
def test_invalid_limit_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject values that would disable or violate the finite transport contract."""
    monkeypatch.setenv(LIMIT_ENV, value)
    with pytest.raises(WorkerSdkError, match=LIMIT_ENV):
        api._worker_max_message_bytes()


def test_default_and_explicit_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the default and accept the inclusive supported byte boundaries."""
    monkeypatch.delenv(LIMIT_ENV, raising=False)
    assert api._worker_max_message_bytes() == 4 * 1024 * 1024
    for value in (1, 16 * 1024 * 1024, 2_147_483_647):
        monkeypatch.setenv(LIMIT_ENV, str(value))
        assert api._worker_max_message_bytes() == value


class EchoPlugin(WorkerPlugin):
    """Return either the input or a large response generated from a small input."""

    plugin_id = "test.large-message"

    def register(self, ctx: PluginContext, config: object) -> None:
        """Register separate callbacks for request- and response-size isolation."""
        ctx.register_tool_request_intercept("echo", lambda name, args: args)
        ctx.register_tool_request_intercept("response", lambda name, args: {"text": "x" * args["response_bytes"]})


@pytest.mark.parametrize("limit", [None, 8 * 1024 * 1024])
@pytest.mark.parametrize("direction", ["request", "response"])
async def test_worker_request_response_boundary(
    limit: int | None, direction: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolate server receive/send limits with a more permissive test client."""
    if limit is None:
        monkeypatch.delenv(LIMIT_ENV, raising=False)
    else:
        monkeypatch.setenv(LIMIT_ENV, str(limit))
    endpoint_file = tmp_path / "endpoint"
    for key, value in {
        "NEMO_RELAY_WORKER_SOCKET": "tcp://127.0.0.1:0",
        "NEMO_RELAY_HOST_SOCKET": "http://127.0.0.1:9",
        "NEMO_RELAY_WORKER_ID": "test",
        "NEMO_RELAY_WORKER_TOKEN": "synthetic-test-token",
        "NEMO_RELAY_WORKER_ENDPOINT_FILE": str(endpoint_file),
    }.items():
        monkeypatch.setenv(key, value)
    task = asyncio.create_task(serve_plugin(EchoPlugin()))
    channel = None
    try:
        for _ in range(200):
            if endpoint_file.exists() or task.done():
                break
            await asyncio.sleep(0.01)
        assert endpoint_file.exists(), "worker failed to start"
        # The client accepts both 5 and 9 MiB, isolating the SDK server's caps.
        channel = grpc.aio.insecure_channel(
            api._grpc_target(endpoint_file.read_text()), options=api._worker_grpc_options(16 * 1024 * 1024)
        )
        stub = api.pb_grpc.PluginWorkerStub(channel)
        await stub.Register(
            api.pb.RegisterRequest(
                activation_id="test", auth_token="synthetic-test-token", config=api._json_envelope(api.JSON_SCHEMA, {})
            ),
            timeout=5,
        )
        payload = {"text": "x" * (5 * 1024 * 1024)}
        request = api.pb.InvokeRequest(
            activation_id="test",
            auth_token="synthetic-test-token",
            invocation_id="large",
            registration_name="response" if direction == "response" else "echo",
            surface=api.pb.TOOL_REQUEST_INTERCEPT,
            tool=api.pb.ToolInvocation(
                tool_name="echo",
                value=api._json_envelope(
                    api.JSON_SCHEMA, {"response_bytes": 5 * 1024 * 1024} if direction == "response" else payload
                ),
            ),
        )
        if direction == "response":
            assert request.ByteSize() < 1024, "response rejection must not be caused by the request cap"
        if limit is None:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Invoke(request, timeout=10)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        else:
            response = await stub.Invoke(request, timeout=10)
            assert api._decode_required_envelope(response.json.value, "result") == payload
            oversized = (
                {"response_bytes": 9 * 1024 * 1024} if direction == "response" else {"text": "x" * (9 * 1024 * 1024)}
            )
            request.tool.value.CopyFrom(api._json_envelope(api.JSON_SCHEMA, oversized))
            if direction == "response":
                assert request.ByteSize() < 1024
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Invoke(request, timeout=10)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert (
            await stub.Health(api.pb.HealthRequest(activation_id="test", auth_token="synthetic-test-token"), timeout=5)
        ).ok
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if channel is not None:
            await channel.close()


@pytest.mark.parametrize("limit", [None, 8 * 1024 * 1024])
@pytest.mark.parametrize("direction", ["request", "response"])
async def test_worker_host_channel_request_response_boundary(
    limit: int | None, direction: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolate channel send/receive limits with a more permissive host server."""
    if limit is None:
        monkeypatch.delenv(LIMIT_ENV, raising=False)
    else:
        monkeypatch.setenv(LIMIT_ENV, str(limit))

    class Host(api.pb_grpc.RelayHostRuntimeServicer):
        async def LlmNext(self, request: Any, context: Any) -> Any:
            """Echo large requests or generate a large response from a small request."""
            args = api._decode_required_envelope(request.request, "request", expected_schema=api.LLM_REQUEST_SCHEMA)
            if "response_bytes" in args:
                return api.pb.JsonResult(
                    value=api._json_envelope(api.JSON_SCHEMA, {"text": "x" * args["response_bytes"]})
                )
            return api.pb.JsonResult(value=api.pb.JsonEnvelope(schema=api.JSON_SCHEMA, json=request.request.json))

    # Both host caps exceed every test payload, so only the SDK channel can reject.
    server = grpc.aio.server(options=api._worker_grpc_options(16 * 1024 * 1024))
    api.pb_grpc.add_RelayHostRuntimeServicer_to_server(Host(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = api._open_host_channel(f"http://127.0.0.1:{port}")
    try:
        stub = api.pb_grpc.RelayHostRuntimeStub(channel)
        payload = (
            {"text": "x" * (5 * 1024 * 1024)}
            if direction == "response"
            else {"messages": [{"role": "user", "content": "x" * (5 * 1024 * 1024)}]}
        )
        request = api.pb.LlmNextRequest(
            request=api._json_envelope(
                api.LLM_REQUEST_SCHEMA, {"response_bytes": 5 * 1024 * 1024} if direction == "response" else payload
            )
        )
        if direction == "response":
            assert request.ByteSize() < 1024, "response rejection must not be caused by the request cap"
        if limit is None:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.LlmNext(request, timeout=10)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        else:
            response = await stub.LlmNext(request, timeout=10)
            assert api._decode_required_envelope(response.value, "result") == payload
            oversized = (
                {"response_bytes": 9 * 1024 * 1024}
                if direction == "response"
                else {"messages": [{"role": "user", "content": "x" * (9 * 1024 * 1024)}]}
            )
            request.request.CopyFrom(api._json_envelope(api.LLM_REQUEST_SCHEMA, oversized))
            if direction == "response":
                assert request.ByteSize() < 1024
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.LlmNext(request, timeout=10)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        control = {"messages": []}
        response = await stub.LlmNext(
            api.pb.LlmNextRequest(request=api._json_envelope(api.LLM_REQUEST_SCHEMA, control)), timeout=5
        )
        assert api._decode_required_envelope(response.value, "result") == control
    finally:
        await channel.close()
        await server.stop(grace=0)
