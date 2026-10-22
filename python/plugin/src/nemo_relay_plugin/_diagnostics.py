# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded worker-authored diagnostic delivery; no local evidence store."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

grpc: Any = importlib.import_module("grpc")
pb: Any = importlib.import_module("._proto.plugin_worker_pb2", __package__)

DiagnosticCodeDefinition = pb.DiagnosticCodeDefinition
DiagnosticExtensionDefinition = pb.DiagnosticExtensionDefinition
DiagnosticObservation = pb.DiagnosticObservation
DiagnosticObservationOutcome = pb.DiagnosticObservationOutcome
DiagnosticObservationSeverity = pb.DiagnosticObservationSeverity

# Document the SDK's aliases without changing generated protobuf sources.
DiagnosticCodeDefinition.__doc__ = "Built-in canonical name or a DiagnosticExtensionDefinition."
DiagnosticExtensionDefinition.__doc__ = "Bounded name, description, outcome, and allowed_failure_classes."
DiagnosticObservation.__doc__ = (
    "Code handle, positive count, outcome, severity, and optional restricted detail/correlation."
)
DiagnosticObservationOutcome.__doc__ = "Explicit UNKNOWN, SUCCESS, or FAILURE operation outcome."
DiagnosticObservationSeverity.__doc__ = "Evidence severity: TRACE, DEBUG, INFO, WARNING, or ERROR."

_MAX_REQUEST_BYTES = 16 * 1024
_MAX_RESPONSE_BYTES = 128 * 1024
_MIN_RESPONSE_BYTES = 256
_MAX_PENDING_SUBMISSIONS = 32
_MAX_RETRY_ATTEMPTS = 8
_MAX_RETRY_TIMEOUT = 10.0
_BOOTSTRAP_TIMEOUT = 2.0
# Preserve existing flush/shutdown bounds independently of session RPC policy.
_LOCAL_LIFECYCLE_TIMEOUT = 2.0
# Reserve room for credentials, session identity, sequence ordering, and
# protobuf framing. The hydrated request is also checked against the exact cap.
_SUBMISSION_ENVELOPE_ALLOWANCE_BYTES = 256
_MIN_SUBMISSION_BYTES = 2
_MAX_COUNT = (1 << 64) - 1


@dataclass(frozen=True)
class _DeliveryPolicy:
    max_request_bytes: int
    max_response_bytes: int
    max_pending_submissions: int
    retry_attempts: int
    retry_timeout: float

    @classmethod
    def from_capabilities(cls, capability: Any) -> _DeliveryPolicy | None:
        values = (
            capability.max_request_bytes,
            capability.max_response_bytes,
            capability.max_pending_submissions,
            capability.retry_attempts,
            capability.retry_timeout_millis,
        )
        if any(not isinstance(value, int) or value <= 0 for value in values):
            return None
        if (
            capability.max_request_bytes < _SUBMISSION_ENVELOPE_ALLOWANCE_BYTES + _MIN_SUBMISSION_BYTES
            or capability.max_request_bytes > _MAX_REQUEST_BYTES
            or capability.max_response_bytes < _MIN_RESPONSE_BYTES
            or capability.max_response_bytes > _MAX_RESPONSE_BYTES
            or capability.max_pending_submissions > _MAX_PENDING_SUBMISSIONS
            or capability.retry_attempts > _MAX_RETRY_ATTEMPTS
            or capability.retry_timeout_millis > _MAX_RETRY_TIMEOUT * 1000
        ):
            return None
        return cls(
            max_request_bytes=capability.max_request_bytes,
            max_response_bytes=capability.max_response_bytes,
            max_pending_submissions=capability.max_pending_submissions,
            retry_attempts=capability.retry_attempts,
            retry_timeout=capability.retry_timeout_millis / 1000,
        )


class DiagnosticDeliveryError(RuntimeError):
    """Delivery failure with a stable kind, independent of execution results."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"diagnostic delivery: {kind}")
        self.kind = kind


@dataclass(frozen=True)
class DiagnosticDeliveryStats:
    """Local delivery totals; rejected includes host responses that leave the sequence reusable."""

    dropped: int = 0
    rejected: int = 0
    uncertain: int = 0


class WorkerDiagnostics:
    """One negotiated, activation-owned bounded delivery facade."""

    def __init__(self, stub: Any, credentials: Any) -> None:
        self._stub = stub
        self._credentials = credentials
        self.availability = "unavailable"
        self._session = b""
        self._policy: _DeliveryPolicy | None = None
        self._queue: asyncio.Queue[tuple[bytes | None, asyncio.Future[Any] | None]] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopped = False
        self._closed = False
        self._stats = {"dropped": 0, "rejected": 0, "uncertain": 0}

    @classmethod
    def inert(cls, stub: Any, activation_id: str, auth_token: str) -> WorkerDiagnostics:
        """Return an unavailable facade after host negotiation was cancelled."""
        return cls(stub, pb.DiagnosticCredentials(activation_id=activation_id, auth_token=auth_token))

    @classmethod
    async def connect(cls, stub: Any, activation_id: str, auth_token: str) -> WorkerDiagnostics:
        """Negotiate once; old or disabled hosts produce an inert facade."""
        facade = cls(stub, pb.DiagnosticCredentials(activation_id=activation_id, auth_token=auth_token))
        if stub is None:
            facade.availability = "unsupported"
            return facade
        try:
            capability = await stub.GetDiagnosticCapabilities(
                facade._credentials, timeout=_BOOTSTRAP_TIMEOUT
            )
            if not capability.enabled:
                facade.availability = "disabled"
                return facade
            if (
                capability.protocol_version != 1
                or capability.provenance_version != 1
                or capability.snapshot_version != 1
            ):
                facade.availability = "unsupported"
                return facade
            policy = _DeliveryPolicy.from_capabilities(capability)
            if policy is None:
                facade.availability = "unsupported"
                return facade
            session = await stub.OpenDiagnosticSession(
                pb.DiagnosticOpenRequest(
                    credentials=facade._credentials,
                    protocol_version=1,
                    snapshot_version=1,
                    provenance_version=1,
                ),
                timeout=_BOOTSTRAP_TIMEOUT,
            )
        except grpc.RpcError as error:
            if error.code() == grpc.StatusCode.UNIMPLEMENTED:
                facade.availability = "unsupported"
            return facade
        if not session.HasField("capabilities"):
            facade.availability = "unsupported"
            return facade
        if not session.capabilities.enabled:
            facade.availability = "disabled"
            return facade
        if session.capabilities != capability:
            facade.availability = "unsupported"
            return facade
        session_policy = _DeliveryPolicy.from_capabilities(session.capabilities)
        if session_policy is None:
            facade.availability = "unsupported"
            return facade
        if len(session.session_id) != 16:
            return facade
        facade._session = session.session_id
        facade._policy = session_policy
        facade.availability = "available"
        facade._queue = asyncio.Queue(maxsize=session_policy.max_pending_submissions)
        facade._task = asyncio.create_task(facade._deliver())
        return facade

    def delivery_stats(self) -> DiagnosticDeliveryStats:
        """Return saturating loss totals without resetting them."""
        return DiagnosticDeliveryStats(**self._stats)

    def _increment(self, kind: str) -> None:
        self._stats[kind] = min(_MAX_COUNT, self._stats[kind] + 1)

    async def component(self, name: str, codes: Sequence[Any]) -> DiagnosticComponent:
        """Register a bounded vocabulary; code handles are one-based positions."""
        if self._queue is None:
            return DiagnosticComponent(self, 0)
        result = await self._submit(
            pb.DiagnosticSubmission(register_component=pb.DiagnosticComponentRegistration(local_name=name, codes=codes))
        )
        if result.status != pb.DIAGNOSTIC_RECEIPT_STATUS_REGISTERED:
            raise DiagnosticDeliveryError("rejected")
        return DiagnosticComponent(self, result.component)

    def _enqueue(self, request: Any, future: asyncio.Future[Any] | None) -> None:
        if self._stopped:
            raise DiagnosticDeliveryError("closed")
        if self._queue is None:
            return
        policy = self._policy
        if policy is None or request.ByteSize() > policy.max_request_bytes - _SUBMISSION_ENVELOPE_ALLOWANCE_BYTES:
            self._increment("dropped")
            raise DiagnosticDeliveryError("invalid")
        # Frozen bytes bound retained payload and isolate subsequent caller edits.
        frozen = request.SerializeToString(deterministic=True)
        try:
            self._queue.put_nowait((frozen, future))
        except asyncio.QueueFull as error:
            self._increment("dropped")
            raise DiagnosticDeliveryError("full") from error

    async def _submit(self, request: Any) -> Any:
        future = asyncio.get_running_loop().create_future()
        self._enqueue(request, future)
        return await future

    async def flush(self, timeout: float | None = None) -> None:
        """Wait for admitted work within a deadline; no exporter acknowledgment."""
        if self._queue is None:
            return
        if self._closed:
            raise DiagnosticDeliveryError("closed")
        timeout = timeout if timeout is not None else _LOCAL_LIFECYCLE_TIMEOUT
        future = asyncio.get_running_loop().create_future()
        try:
            async with asyncio.timeout(timeout):
                await self._queue.put((None, future))
                await future
        except TimeoutError as error:
            raise DiagnosticDeliveryError("uncertain") from error

    async def close(self, timeout: float | None = None) -> None:
        """Drain diagnostics within the existing plugin shutdown allowance."""
        if self._queue is None or self._closed:
            return
        timeout = timeout if timeout is not None else _LOCAL_LIFECYCLE_TIMEOUT
        try:
            async with asyncio.timeout(timeout):
                await self.flush(timeout)
                await self._submit(pb.DiagnosticSubmission(close_session=pb.DiagnosticSessionClose()))
        except TimeoutError as error:
            raise DiagnosticDeliveryError("uncertain") from error
        finally:
            self._stopped = True
            self._closed = True
            if self._task is not None:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            while not self._queue.empty():
                frozen, future = self._queue.get_nowait()
                if frozen is not None:
                    self._increment("dropped")
                if future is not None and not future.done():
                    future.set_exception(DiagnosticDeliveryError("closed"))

    async def _deliver(self) -> None:
        assert self._queue is not None
        sequence = 1
        while True:
            frozen, future = await self._queue.get()
            if frozen is None:
                if future is not None and not future.done():
                    if self._stats["uncertain"]:
                        future.set_exception(DiagnosticDeliveryError("uncertain"))
                    elif self._stopped:
                        future.set_exception(DiagnosticDeliveryError("closed"))
                    else:
                        future.set_result(None)
                continue
            result: Any = None
            error = DiagnosticDeliveryError("uncertain")
            if self._stopped:
                self._increment("dropped")
                error = DiagnosticDeliveryError("closed")
            else:
                request = pb.DiagnosticSubmission.FromString(frozen)
                request.credentials.CopyFrom(self._credentials)
                request.session_id = self._session
                request.sequence = sequence
                policy = self._policy
                if policy is None or request.ByteSize() > policy.max_request_bytes:
                    self._increment("dropped")
                    error = DiagnosticDeliveryError("invalid")
                    request = None
                if request is None:
                    pass
                else:
                    try:
                        for _ in range(policy.retry_attempts):
                            try:
                                result = await self._stub.SubmitDiagnostic(
                                    request, timeout=policy.retry_timeout
                                )
                            except grpc.RpcError as failure:
                                if failure.code() == grpc.StatusCode.INVALID_ARGUMENT:
                                    self._increment("rejected")
                                    error = DiagnosticDeliveryError("not_consumed")
                                    break
                                if failure.code() in (
                                    grpc.StatusCode.PERMISSION_DENIED,
                                    grpc.StatusCode.UNAUTHENTICATED,
                                    grpc.StatusCode.FAILED_PRECONDITION,
                                    grpc.StatusCode.OUT_OF_RANGE,
                                    grpc.StatusCode.ALREADY_EXISTS,
                                ):
                                    break
                                continue
                            if result.ByteSize() > policy.max_response_bytes:
                                result = None
                                break
                            if result.submission_sequence != sequence or result.status not in range(1, 6):
                                result = None
                                break
                            sequence += 1
                            if sequence > _MAX_COUNT:
                                self._stopped = True
                            if result.status == pb.DIAGNOSTIC_RECEIPT_STATUS_REJECTED:
                                self._increment("rejected")
                                error = DiagnosticDeliveryError("rejected")
                                result = None
                            break
                    except asyncio.CancelledError:
                        self._increment("uncertain")
                        if future is not None and not future.done():
                            future.set_exception(DiagnosticDeliveryError("uncertain"))
                        raise
            if result is None and error.kind == "uncertain":
                self._increment("uncertain")
                self._stopped = True
            if future is not None and not future.done():
                if result is None:
                    future.set_exception(error)
                else:
                    future.set_result(result)

class DiagnosticComponent:
    """Component-scoped producer; host assigns ownership and provenance."""

    def __init__(self, diagnostics: WorkerDiagnostics, handle: int) -> None:
        self._diagnostics = diagnostics
        self._handle = handle

    def try_record_lazy(self, make: Callable[[], Any]) -> None:
        """Enqueue a distinct internal operation; skip factories while disabled/full."""
        if self._handle == 0:
            return
        if self._diagnostics._stopped:
            raise DiagnosticDeliveryError("closed")
        queue = self._diagnostics._queue
        if queue is not None and queue.full():
            self._diagnostics._increment("dropped")
            raise DiagnosticDeliveryError("full")
        observation = make()
        if observation.count == 0:
            raise DiagnosticDeliveryError("invalid")
        request = pb.DiagnosticSubmission(record=observation)
        request.record.component = self._handle
        self._diagnostics._enqueue(request, None)

    async def close(self) -> None:
        """Close registration through the ordered receipt boundary."""
        if self._handle != 0:
            await self._diagnostics._submit(
                pb.DiagnosticSubmission(close_component=pb.DiagnosticComponentClose(component=self._handle))
            )
