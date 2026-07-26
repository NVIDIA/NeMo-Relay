# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest entrypoint for the Rampart PII local-model provider."""

from __future__ import annotations

import asyncio
from typing import Any

from nemo_relay_plugin import ConfigDiagnostic, DiagnosticLevel, Json, PluginContext, WorkerPlugin, serve_plugin

from .detector import RampartDetector, RampartSettings, resolve_verified_model_root


class _Admission:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._active = 0

    def acquire(self) -> None:
        if self._active >= self._limit:
            raise RuntimeError("Rampart provider is at its pending-request limit")
        self._active += 1

    def release(self) -> None:
        self._active -= 1


class RampartWorker(WorkerPlugin):
    """Expose Rampart inference through the PII component's provider contract."""

    plugin_id = "nemo_relay.pii_rampart"

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        try:
            settings = RampartSettings.from_config(config)
        except (TypeError, ValueError) as error:
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nemo_relay.pii_rampart.invalid_config",
                    component=self.plugin_id,
                    message=str(error),
                )
            ]
        try:
            resolve_verified_model_root(settings)
        except Exception:
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nemo_relay.pii_rampart.model_unavailable",
                    component=self.plugin_id,
                    message=(
                        "the pinned Rampart model is unavailable or failed integrity "
                        "verification; prefetch it before enabling the plugin"
                    ),
                )
            ]
        return []

    def register(self, ctx: PluginContext, config: Json) -> None:
        settings = RampartSettings.from_config(config)
        detector = RampartDetector.load(settings)
        admission = _Admission(settings.max_pending_requests)

        async def detect(request: Json) -> Json:
            admission.acquire()
            work = asyncio.create_task(asyncio.to_thread(detector.detect_request, request))
            release_on_completion = False
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                release_on_completion = True

                def release_after_work(_task: asyncio.Task[Json]) -> None:
                    try:
                        _task.exception()
                    except asyncio.CancelledError:
                        pass
                    admission.release()

                work.add_done_callback(release_after_work)
                raise
            finally:
                if not release_on_completion:
                    admission.release()

        ctx.register_local_model_provider("detector", detect)


async def main() -> None:
    """Start the Relay-managed worker."""
    await serve_plugin(RampartWorker())


if __name__ == "__main__":
    asyncio.run(main())
