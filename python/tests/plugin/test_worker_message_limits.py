# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real gRPC boundary coverage without a model, credentials, or external services."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest

grpc = pytest.importorskip("grpc")

from nemo_relay_plugin import PluginContext, WorkerPlugin, WorkerSdkError, serve_plugin  # noqa: E402
from nemo_relay_plugin import _api as api  # noqa: E402
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    common_pb2 as common_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    diagnostic_read_pb2 as diagnostic_read_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    diagnostics_pb2 as diagnostics_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    llm_pb2 as llm_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    plugin_worker_service_pb2 as worker_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    plugin_worker_service_pb2_grpc as worker_grpc,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    relay_diagnostics_service_pb2 as diagnostics_service_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    relay_diagnostics_service_pb2_grpc as diagnostics_grpc,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    relay_host_runtime_service_pb2 as host_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    relay_host_runtime_service_pb2_grpc as host_grpc,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (  # noqa: E402
    runtime_pb2 as runtime_pb,
)

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
        stub = worker_grpc.PluginWorkerServiceStub(channel)
        await stub.Register(
            worker_pb.RegisterRequest(
                config=api._encode_payload(common_pb.JsonValue, {}),
                credentials=common_pb.Credentials(activation_id="test", auth_token="synthetic-test-token"),
                plugin_id="test.large-message",
            ),
            timeout=5,
        )
        payload = {"text": "x" * (5 * 1024 * 1024)}
        request = worker_pb.InvokeRequest(
            invocation_id="large",
            registration_name="response" if direction == "response" else "echo",
            credentials=common_pb.Credentials(activation_id="test", auth_token="synthetic-test-token"),
            tool_request_intercept=runtime_pb.ToolValueInvocation(
                tool_name="echo",
                value=api._encode_payload(
                    common_pb.JsonValue, {"response_bytes": 5 * 1024 * 1024} if direction == "response" else payload
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
            assert api._decode_payload(response.tool_value, "result") == payload
            oversized = (
                {"response_bytes": 9 * 1024 * 1024} if direction == "response" else {"text": "x" * (9 * 1024 * 1024)}
            )
            request.tool_request_intercept.value.CopyFrom(api._encode_payload(common_pb.JsonValue, oversized))
            if direction == "response":
                assert request.ByteSize() < 1024
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Invoke(request, timeout=10)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert (
            await stub.Health(
                worker_pb.HealthRequest(
                    credentials=common_pb.Credentials(activation_id="test", auth_token="synthetic-test-token")
                ),
                timeout=5,
            )
        ).status == runtime_pb.WORKER_HEALTH_STATUS_READY
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

    class Host(host_grpc.RelayHostRuntimeServiceServicer):
        async def LlmNext(self, request: Any, context: Any) -> Any:
            """Echo large requests or generate a large response from a small request."""
            args = api._decode_payload(request.request, "request")["content"]
            if "response_bytes" in args:
                return host_pb.LlmNextResponse(
                    value=api._encode_payload(common_pb.JsonValue, {"text": "x" * args["response_bytes"]})
                )
            return host_pb.LlmNextResponse(value=api._json_value(args))

    # Both host caps exceed every test payload, so only the SDK channel can reject.
    server = grpc.aio.server(options=api._worker_grpc_options(16 * 1024 * 1024))
    host_grpc.add_RelayHostRuntimeServiceServicer_to_server(Host(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = api._open_host_channel(f"http://127.0.0.1:{port}")
    try:
        stub = host_grpc.RelayHostRuntimeServiceStub(channel)
        payload = (
            {"text": "x" * (5 * 1024 * 1024)}
            if direction == "response"
            else {"messages": [{"role": "user", "content": "x" * (5 * 1024 * 1024)}]}
        )
        request = host_pb.LlmNextRequest(
            request=api._encode_payload(
                llm_pb.LlmRequest,
                {"headers": {}, "content": {"response_bytes": 5 * 1024 * 1024} if direction == "response" else payload},
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
            assert api._decode_payload(response.value, "result") == payload
            oversized = (
                {"response_bytes": 9 * 1024 * 1024}
                if direction == "response"
                else {"messages": [{"role": "user", "content": "x" * (9 * 1024 * 1024)}]}
            )
            request.request.CopyFrom(api._encode_payload(llm_pb.LlmRequest, {"headers": {}, "content": oversized}))
            if direction == "response":
                assert request.ByteSize() < 1024
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.LlmNext(request, timeout=10)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        control = {"messages": []}
        response = await stub.LlmNext(
            host_pb.LlmNextRequest(request=api._encode_payload(llm_pb.LlmRequest, {"headers": {}, "content": control})),
            timeout=5,
        )
        assert api._decode_payload(response.value, "result") == control
    finally:
        await channel.close()
        await server.stop(grace=0)


@pytest.mark.parametrize("policy", ["scalar-small", "named-small", "named-large"])
@pytest.mark.parametrize("transport", ["tcp", "unix"])
async def test_diagnostics_keep_separate_budgets_inside_callbacks(
    policy: str, transport: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """The real worker keeps diagnostic caps independent in both directions."""
    if transport == "unix" and not hasattr(socket, "AF_UNIX"):
        pytest.skip("Unix sockets are unavailable")
    limit = 256 * 1024 if policy == "named-large" else 4096
    monkeypatch.setenv(LIMIT_ENV, str(limit if policy == "scalar-small" else 4 * 1024 * 1024))
    monkeypatch.setenv(
        "NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES_BY_NAME",
        json.dumps({} if policy == "scalar-small" else {"inspect": limit}),
    )

    class Host(diagnostics_grpc.RelayDiagnosticsServiceServicer, host_grpc.RelayHostRuntimeServiceServicer):
        async def ReadDiagnosticPage(self, request: Any, context: Any) -> Any:
            # A permissive peer isolates the SDK's send and receive limits.
            return diagnostics_service_pb.ReadDiagnosticPageResponse(
                snapshot=diagnostics_pb.DiagnosticSnapshot(
                    source=diagnostics_pb.DiagnosticSourceIdentity(process_role="x" * request.query.max_events)
                )
            )

        async def LlmNext(self, request: Any, context: Any) -> Any:
            return host_pb.LlmNextResponse(value=api._encode_payload(common_pb.JsonValue, {"text": "x" * 8192}))

    class DiagnosticPlugin(WorkerPlugin):
        plugin_id = "test.diagnostic-limits"

        def register(self, ctx: PluginContext, config: object) -> None:
            async def inspect(name: str, args: Any) -> Any:
                if args.get("execution"):
                    response = await ctx.runtime._host_stub.LlmNext(host_pb.LlmNextRequest(), timeout=3)
                    return {"bytes": len(api._decode_payload(response.value, "response")["text"])}
                response = await ctx.runtime._diagnostic_stub.ReadDiagnosticPage(
                    diagnostics_service_pb.ReadDiagnosticPageRequest(
                        credentials=common_pb.Credentials(activation_id="test", auth_token="test-token"),
                        query=diagnostic_read_pb.DiagnosticQuery(
                            code="x" * args.get("request_bytes", 0), max_events=args.get("response_bytes", 8192)
                        ),
                    ),
                    timeout=3,
                )
                return {"bytes": len(response.snapshot.source.process_role)}

            ctx.register_tool_request_intercept("inspect", inspect)

    server = grpc.aio.server(options=api._worker_grpc_options(1024 * 1024))
    host = Host()
    diagnostics_grpc.add_RelayDiagnosticsServiceServicer_to_server(host, server)
    host_grpc.add_RelayHostRuntimeServiceServicer_to_server(host, server)
    if transport == "unix":
        # macOS pytest roots can exceed the Unix socket path-length limit.
        directory = tempfile.TemporaryDirectory(prefix="nr-", dir="/tmp")
        request.addfinalizer(directory.cleanup)
        host_endpoint = f"unix://{Path(directory.name) / 'host.sock'}"
    else:
        host_endpoint = "tcp://127.0.0.1:0"
    port = server.add_insecure_port(api._grpc_target(host_endpoint))
    if transport == "tcp":
        host_endpoint = f"http://127.0.0.1:{port}"
    await server.start()
    endpoint_file = tmp_path / "endpoint"
    for key, value in {
        "NEMO_RELAY_WORKER_SOCKET": "tcp://127.0.0.1:0",
        "NEMO_RELAY_HOST_SOCKET": host_endpoint,
        "NEMO_RELAY_WORKER_ID": "test",
        "NEMO_RELAY_WORKER_TOKEN": "test-token",
        "NEMO_RELAY_WORKER_ENDPOINT_FILE": str(endpoint_file),
    }.items():
        monkeypatch.setenv(key, value)
    task = asyncio.create_task(serve_plugin(DiagnosticPlugin()))
    channel = None
    try:
        for _ in range(200):
            if endpoint_file.exists() or task.done():
                break
            await asyncio.sleep(0.01)
        assert endpoint_file.exists(), "worker failed to start"
        channel = grpc.aio.insecure_channel(api._grpc_target(endpoint_file.read_text()))
        stub = worker_grpc.PluginWorkerServiceStub(channel)
        registered = await stub.Register(
            worker_pb.RegisterRequest(
                credentials=common_pb.Credentials(activation_id="test", auth_token="test-token"),
                plugin_id="test.diagnostic-limits",
                config=api._json_value(None),
            ),
            timeout=3,
        )
        assert not registered.HasField("error")

        async def invoke(**args: Any) -> Any:
            return await stub.Invoke(
                worker_pb.InvokeRequest(
                    invocation_id="diagnostic",
                    registration_name="inspect",
                    credentials=common_pb.Credentials(activation_id="test", auth_token="test-token"),
                    tool_request_intercept=runtime_pb.ToolValueInvocation(
                        tool_name="inspect", value=api._encode_payload(common_pb.JsonValue, args)
                    ),
                ),
                metadata=(("nemo-relay-worker-callback-bin", b"inspect"),),
                timeout=3,
            )

        async def healthy_diagnostic() -> None:
            for request_bytes in (0, 8192):
                response = await invoke(request_bytes=request_bytes)
                assert not response.HasField("error"), response.error.message
                assert api._decode_payload(response.tool_value, "result") == {"bytes": 8192}

        await healthy_diagnostic()
        for oversized in ({"request_bytes": 16 * 1024}, {"response_bytes": 128 * 1024}):
            response = await invoke(**oversized)
            assert response.HasField("error"), "diagnostics escaped their own finite transport budget"
            assert "RESOURCE_EXHAUSTED" in response.error.message
            await healthy_diagnostic()
        response = await invoke(execution=True)
        if limit < 8192:
            assert response.HasField("error")
            assert "RESOURCE_EXHAUSTED" in response.error.message
        else:
            assert not response.HasField("error")
            assert api._decode_payload(response.tool_value, "result") == {"bytes": 8192}
        assert (
            await stub.Shutdown(
                worker_pb.ShutdownRequest(
                    credentials=common_pb.Credentials(activation_id="test", auth_token="test-token")
                ),
                timeout=3,
            )
        ) == worker_pb.ShutdownResponse()
        await asyncio.wait_for(task, timeout=3)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if channel is not None:
            await channel.close()
        await server.stop(grace=0)
