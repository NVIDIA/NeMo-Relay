# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker diagnostic delivery contracts, independent of plugin execution."""

import asyncio

import grpc
import pytest

from nemo_relay_plugin import PluginRuntime
from nemo_relay_plugin._diagnostics import DiagnosticDeliveryError, WorkerDiagnostics, pb


class RpcFailure(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class Host:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.requests = []
        self.retry = False
        self.fail = False
        self.gate = None

    async def GetDiagnosticCapabilities(self, request, **kwargs):
        return pb.DiagnosticCapabilities(
            enabled=self.enabled,
            protocol_version=1,
            snapshot_version=1,
            provenance_version=1,
            max_request_bytes=16384,
            max_pending_submissions=32,
        )

    async def OpenDiagnosticSession(self, request, **kwargs):
        return pb.DiagnosticSession(session_id=b"s" * 16)

    async def SubmitDiagnostic(self, request, **kwargs):
        self.requests.append(request.SerializeToString())
        if self.gate is not None:
            await self.gate.wait()
        if self.fail or self.retry:
            self.retry = False
            raise RpcFailure(grpc.StatusCode.UNAVAILABLE)
        operation = request.WhichOneof("operation")
        status = {
            "register_component": pb.DIAGNOSTIC_RECEIPT_STATUS_REGISTERED,
            "record": pb.DIAGNOSTIC_RECEIPT_STATUS_RECORDED,
            "close_session": pb.DIAGNOSTIC_RECEIPT_STATUS_CLOSED,
            "close_component": pb.DIAGNOSTIC_RECEIPT_STATUS_CLOSED,
        }[operation]
        return pb.DiagnosticReceipt(submission_sequence=request.sequence, status=status, component=1)


def observation():
    return pb.DiagnosticObservation(
        code=1,
        count=2,
        outcome=pb.DIAGNOSTIC_OBSERVATION_OUTCOME_FAILURE,
        severity=pb.DIAGNOSTIC_OBSERVATION_SEVERITY_ERROR,
        failure_class="unknown",
        detail="original",
    )


def test_disabled_and_unsupported_skip_factories():
    async def run():
        for stub, availability in [(None, "unsupported"), (Host(False), "disabled")]:
            facade = await WorkerDiagnostics.connect(stub, "activation", "token")
            assert facade.availability == availability
            component = await facade.component("internal", [])
            component.try_record_lazy(lambda: pytest.fail("inert factory ran"))
            await component.close()
            await facade.flush()
            await facade.close()

    asyncio.run(run())


def test_runtime_negotiates_once_and_closes_without_extending_repeated_shutdown():
    async def run():
        runtime = PluginRuntime(activation_id="activation", auth_token="token", host_stub=None, diagnostic_stub=Host())
        first, second = await asyncio.gather(runtime.diagnostics(), runtime.diagnostics())
        assert first is second
        await first.component("internal", [])
        await runtime._close_diagnostics()
        await asyncio.wait_for(runtime._close_diagnostics(), timeout=0.1)
        assert first._task is not None and first._task.done()

    asyncio.run(run())


def test_runtime_shutdown_cancels_pending_negotiation():
    class PendingHost(Host):
        async def GetDiagnosticCapabilities(self, request, **kwargs):
            await asyncio.Future()

    async def run():
        runtime = PluginRuntime(
            activation_id="activation", auth_token="token", host_stub=None, diagnostic_stub=PendingHost()
        )
        pending = asyncio.create_task(runtime.diagnostics())
        await asyncio.sleep(0)
        await runtime._close_diagnostics()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await runtime._close_diagnostics()

    asyncio.run(run())


def test_retry_reuses_sequence_and_frozen_payload():
    async def run():
        host = Host()
        facade = await WorkerDiagnostics.connect(host, "activation", "token")
        component = await facade.component("internal", [])
        host.retry = True
        value = observation()
        component.try_record_lazy(lambda: value)
        value.detail = "mutated-after-admission"
        await facade.flush()
        assert host.requests[1] == host.requests[2]
        submitted = pb.DiagnosticSubmission.FromString(host.requests[1])
        assert submitted.record.detail == "original"
        assert submitted.sequence == 2
        assert facade.delivery_stats().uncertain == 0
        await facade.close()
        assert facade._task is not None and facade._task.done()

    asyncio.run(run())


def test_unknown_acceptance_fences_delivery_and_flush_reports_it():
    async def run():
        host = Host()
        facade = await WorkerDiagnostics.connect(host, "activation", "token")
        component = await facade.component("internal", [])
        host.fail = True
        component.try_record_lazy(observation)
        with pytest.raises(DiagnosticDeliveryError, match="uncertain"):
            await facade.flush()
        assert facade.delivery_stats().uncertain == 1
        assert len(host.requests) == 4  # register, three identical attempts
        assert host.requests[1] == host.requests[2] == host.requests[3]
        with pytest.raises(DiagnosticDeliveryError, match="closed"):
            component.try_record_lazy(lambda: pytest.fail("fenced factory ran"))
        with pytest.raises(DiagnosticDeliveryError, match="uncertain"):
            await facade.close()
        assert facade._task is not None and facade._task.done()

    asyncio.run(run())


def test_queue_and_payload_admission_and_bounded_shutdown():
    async def run():
        host = Host()
        facade = await WorkerDiagnostics.connect(host, "activation", "token")
        component = await facade.component("internal", [])
        large = observation()
        large.detail = "x" * 16384
        with pytest.raises(DiagnosticDeliveryError, match="invalid"):
            component.try_record_lazy(lambda: large)
        host.gate = asyncio.Event()
        component.try_record_lazy(observation)
        await asyncio.sleep(0)  # one request in flight, 32 queued
        for _ in range(32):
            component.try_record_lazy(observation)
        with pytest.raises(DiagnosticDeliveryError, match="full"):
            component.try_record_lazy(lambda: pytest.fail("full factory ran"))
        with pytest.raises(DiagnosticDeliveryError, match="uncertain"):
            await facade.close(timeout=0.01)
        assert facade._task is not None and facade._task.done()
        assert facade._queue is not None and facade._queue.empty()
        assert facade.delivery_stats().uncertain == 1
        assert facade.delivery_stats().dropped == 34

    asyncio.run(run())


@pytest.mark.parametrize("status, consumed", [(grpc.StatusCode.INVALID_ARGUMENT, False), (None, True)])
def test_rejections_and_sequence_consumption(status, consumed):
    class RejectingHost(Host):
        async def SubmitDiagnostic(self, request, **kwargs):
            if request.WhichOneof("operation") == "record" and request.record.count == 2:
                self.requests.append(request.SerializeToString())
                if status is not None:
                    raise RpcFailure(status)
                return pb.DiagnosticReceipt(
                    submission_sequence=request.sequence, status=pb.DIAGNOSTIC_RECEIPT_STATUS_REJECTED
                )
            return await super().SubmitDiagnostic(request, **kwargs)

    async def run():
        host = RejectingHost()
        facade = await WorkerDiagnostics.connect(host, "activation", "token")
        component = await facade.component("internal", [])
        component.try_record_lazy(observation)
        await facade.flush()
        good = observation()
        good.count = 3
        component.try_record_lazy(lambda: good)
        await facade.flush()
        assert pb.DiagnosticSubmission.FromString(host.requests[-1]).sequence == (3 if consumed else 2)
        assert facade.delivery_stats().rejected == int(consumed)
        await facade.close()

    asyncio.run(run())
