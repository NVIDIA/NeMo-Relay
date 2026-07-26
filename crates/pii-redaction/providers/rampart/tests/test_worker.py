# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import pytest

import nemo_relay_pii_rampart.worker as worker_module
from nemo_relay_pii_rampart.worker import RampartWorker
from nemo_relay_plugin import ConfigDiagnostic, PluginContext


class FakeContext:
    def __init__(self) -> None:
        self.callback: Any = None

    def register_local_model_provider(self, name: str, callback: Any) -> None:
        assert name == "detector"
        self.callback = callback


class FakeDetector:
    def __init__(self, started: threading.Event | None = None, release: threading.Event | None = None) -> None:
        self.started = started
        self.release = release

    def detect_request(self, request: Any) -> dict[str, Any]:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=5)
        return {"version": 1, "detections": [], "echo": request}


class FailingDetector:
    def detect_request(self, request: Any) -> dict[str, Any]:
        del request
        raise RuntimeError("detector failed")


def test_worker_validation_reports_invalid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "resolve_verified_model_root", lambda _settings: None)
    worker = RampartWorker()
    assert worker.validate({}) == []
    diagnostics = worker.validate({"max_pending_requests": "many"})
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert isinstance(diagnostic, ConfigDiagnostic)
    assert diagnostic.code == "nemo_relay.pii_rampart.invalid_config"
    diagnostics = worker.validate({"local_files_only": False})
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert isinstance(diagnostic, ConfigDiagnostic)
    assert diagnostic.code == "nemo_relay.pii_rampart.invalid_config"


def test_worker_validation_reports_bounded_model_readiness_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_settings: Any) -> None:
        raise ValueError("/sensitive/cache/path/model_q4.onnx is corrupt")

    monkeypatch.setattr(worker_module, "resolve_verified_model_root", fail)

    diagnostics = RampartWorker().validate({})

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert isinstance(diagnostic, ConfigDiagnostic)
    assert diagnostic.code == "nemo_relay.pii_rampart.model_unavailable"
    assert "prefetch" in diagnostic.message
    assert "/sensitive" not in diagnostic.message


def test_worker_registers_async_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDetector()
    monkeypatch.setattr(worker_module.RampartDetector, "load", lambda _settings: fake)
    context = FakeContext()
    RampartWorker().register(cast(PluginContext, context), {})

    request = {"version": 1, "texts": [{"id": 0, "text": "hello"}]}
    assert asyncio.run(context.callback(request)) == {
        "version": 1,
        "detections": [],
        "echo": request,
    }


def test_cancelled_callback_holds_admission_until_native_work_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    fake = FakeDetector(started, release)
    monkeypatch.setattr(worker_module.RampartDetector, "load", lambda _settings: fake)
    context = FakeContext()
    RampartWorker().register(cast(PluginContext, context), {"max_pending_requests": 1})

    async def exercise() -> None:
        first = asyncio.create_task(context.callback({"version": 1, "texts": [{"id": 0, "text": "one"}]}))
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(RuntimeError, match="pending-request limit"):
            await context.callback({"version": 1, "texts": [{"id": 1, "text": "two"}]})

        release.set()
        for _ in range(100):
            await asyncio.sleep(0.01)
            try:
                result = await context.callback({"version": 1, "texts": [{"id": 2, "text": "three"}]})
            except RuntimeError:
                continue
            assert result["version"] == 1
            return
        pytest.fail("provider admission was not released after native work completed")

    asyncio.run(exercise())


def test_detector_failure_releases_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module.RampartDetector, "load", lambda _settings: FailingDetector())
    context = FakeContext()
    RampartWorker().register(cast(PluginContext, context), {"max_pending_requests": 1})

    async def exercise() -> None:
        for text_id in range(2):
            with pytest.raises(RuntimeError, match="detector failed"):
                await context.callback(
                    {
                        "version": 1,
                        "texts": [{"id": text_id, "text": "private"}],
                    }
                )

    asyncio.run(exercise())
