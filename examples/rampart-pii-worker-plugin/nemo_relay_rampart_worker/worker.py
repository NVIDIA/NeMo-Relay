# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process Rampart sanitizer implemented with Relay's worker SDK."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar, cast

from nemo_relay_plugin import (
    ConfigDiagnostic,
    DiagnosticLevel,
    Event,
    EventSanitizeFields,
    Json,
    PluginContext,
    WorkerPlugin,
    serve_plugin,
)

from ._detectors import (
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_MAX_LATENCY_MS,
    FAILURE_REPLACEMENT,
    MAX_CONTENT_CHARS,
    RampartSanitizer,
)
from ._model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, load_classifier
from ._sanitization import sanitize_llm_request, sanitize_llm_response, sanitize_tool_payload

PLUGIN_ID = "nvidia.rampart_pii"
DEFAULT_MAX_CONCURRENCY = 2
_MIN_PRIORITY = -(2**31)
_MAX_PRIORITY = 2**31 - 2
_CONFIG_FIELDS = frozenset(
    {
        "version",
        "allow_network",
        "max_concurrency",
        "max_content_chars",
        "max_latency_ms",
        "priority",
        "input",
        "output",
        "tool_input",
        "tool_output",
    }
)
_T = TypeVar("_T")


class _SanitizationExecutor:
    """Bound CPU work and fail closed after one end-to-end latency budget."""

    def __init__(self, *, max_concurrency: int, max_latency_ms: int) -> None:
        self._slots = asyncio.Semaphore(max_concurrency)
        self._max_latency_seconds = max_latency_ms / 1_000

    def _release_when_done(self, task: asyncio.Task[_T]) -> None:
        def release(completed: asyncio.Task[_T]) -> None:
            try:
                completed.exception()
            except asyncio.CancelledError:
                pass
            finally:
                self._slots.release()

        task.add_done_callback(release)

    async def run(
        self,
        operation: Callable[..., _T],
        *args: object,
        fallback: Callable[[], _T],
    ) -> _T:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_latency_seconds
        try:
            await asyncio.wait_for(
                self._slots.acquire(),
                timeout=self._max_latency_seconds,
            )
        except TimeoutError:
            return fallback()

        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        release_when_returning = True
        try:
            remaining = deadline - loop.time()
            if remaining <= 0:
                release_when_returning = False
                self._release_when_done(task)
                return fallback()
            return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except TimeoutError:
            release_when_returning = False
            self._release_when_done(task)
            return fallback()
        except asyncio.CancelledError:
            # asyncio cannot preempt a running native inference. Keep the slot
            # occupied until the thread exits so cancellation cannot exceed the
            # configured concurrency bound.
            release_when_returning = False
            self._release_when_done(task)
            raise
        except Exception:
            return fallback()
        finally:
            if release_when_returning:
                self._slots.release()


def _failed_llm_request(request: dict[str, Any]) -> dict[str, Any]:
    failed = request.copy()
    failed["headers"] = {}
    failed["content"] = FAILURE_REPLACEMENT
    return failed


def _diagnostic(field: str, message: str) -> ConfigDiagnostic:
    return ConfigDiagnostic(
        level=DiagnosticLevel.ERROR,
        code=f"{PLUGIN_ID}.invalid_config",
        component=PLUGIN_ID,
        field=field,
        message=message,
    )


def _validate_config(config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
    if not isinstance(config, dict):
        return [_diagnostic("config", "plugin config must be a JSON object")]

    diagnostics: list[ConfigDiagnostic | dict[str, Any]] = [
        _diagnostic(field, f"unknown configuration field '{field}'") for field in sorted(set(config) - _CONFIG_FIELDS)
    ]
    version = config.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        diagnostics.append(_diagnostic("version", "version must be 1"))

    allow_network = config.get("allow_network", False)
    if not isinstance(allow_network, bool):
        diagnostics.append(_diagnostic("allow_network", "allow_network must be a boolean"))

    max_latency_ms = config.get("max_latency_ms", DEFAULT_MAX_LATENCY_MS)
    if not isinstance(max_latency_ms, int) or isinstance(max_latency_ms, bool) or max_latency_ms <= 0:
        diagnostics.append(
            _diagnostic(
                "max_latency_ms",
                "max_latency_ms must be a positive integer",
            )
        )

    max_content_chars = config.get("max_content_chars", DEFAULT_MAX_CONTENT_CHARS)
    if (
        not isinstance(max_content_chars, int)
        or isinstance(max_content_chars, bool)
        or not 1 <= max_content_chars <= MAX_CONTENT_CHARS
    ):
        diagnostics.append(
            _diagnostic(
                "max_content_chars",
                f"max_content_chars must be an integer between 1 and {MAX_CONTENT_CHARS}",
            )
        )

    max_concurrency = config.get("max_concurrency", DEFAULT_MAX_CONCURRENCY)
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or not 1 <= max_concurrency <= 64:
        diagnostics.append(
            _diagnostic(
                "max_concurrency",
                "max_concurrency must be an integer between 1 and 64",
            )
        )

    priority = config.get("priority", 100)
    if not isinstance(priority, int) or isinstance(priority, bool) or not _MIN_PRIORITY <= priority <= _MAX_PRIORITY:
        diagnostics.append(
            _diagnostic(
                "priority",
                f"priority must be between {_MIN_PRIORITY} and {_MAX_PRIORITY}",
            )
        )

    enabled_surfaces = 0
    for field in ("input", "output", "tool_input", "tool_output"):
        value = config.get(field, True)
        if not isinstance(value, bool):
            diagnostics.append(_diagnostic(field, f"{field} must be a boolean"))
        elif value:
            enabled_surfaces += 1
    if enabled_surfaces == 0:
        diagnostics.append(
            _diagnostic(
                "input",
                "at least one managed sanitization surface must be enabled",
            )
        )
    return diagnostics


class RampartWorkerPlugin(WorkerPlugin):
    """Install Rampart on content-bearing managed LLM and tool events."""

    plugin_id = PLUGIN_ID

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        return _validate_config(config)

    async def register(self, ctx: PluginContext, config: Json) -> None:
        if not isinstance(config, dict):
            raise TypeError("plugin config must be a JSON object")

        allow_network = cast(bool, config.get("allow_network", False))
        max_latency_ms = cast(int, config.get("max_latency_ms", DEFAULT_MAX_LATENCY_MS))
        max_concurrency = cast(int, config.get("max_concurrency", DEFAULT_MAX_CONCURRENCY))
        max_content_chars = cast(int, config.get("max_content_chars", DEFAULT_MAX_CONTENT_CHARS))
        priority = cast(int, config.get("priority", 100))
        classifier = await asyncio.to_thread(
            load_classifier,
            DEFAULT_MODEL_ID,
            DEFAULT_MODEL_REVISION,
            allow_network,
        )
        sanitizer = RampartSanitizer(
            classifier,
            max_content_chars=max_content_chars,
            max_latency_ms=max_latency_ms,
        )
        executor = _SanitizationExecutor(
            max_concurrency=max_concurrency,
            max_latency_ms=max_latency_ms,
        )

        def fail_closed_backstop(
            _event: Event,
            fields: EventSanitizeFields,
        ) -> EventSanitizeFields:
            # Worker sanitizer RPC failures preserve specialized payloads, while
            # generic event sanitizer failures clear observability fields.
            return fields

        ctx.register_scope_sanitize_start_guardrail(
            "worker_health",
            fail_closed_backstop,
            priority=priority + 1,
        )
        ctx.register_scope_sanitize_end_guardrail(
            "worker_health",
            fail_closed_backstop,
            priority=priority + 1,
        )

        if config.get("input", True):

            async def sanitize_request(request: dict[str, Any]) -> dict[str, Any]:
                return await executor.run(
                    sanitize_llm_request,
                    request,
                    sanitizer,
                    fallback=lambda: _failed_llm_request(request),
                )

            ctx.register_llm_sanitize_request_guardrail(
                "input",
                sanitize_request,
                priority=priority,
            )

        if config.get("output", True):

            async def sanitize_response(response: Json) -> Json:
                return await executor.run(
                    sanitize_llm_response,
                    response,
                    sanitizer,
                    fallback=lambda: FAILURE_REPLACEMENT,
                )

            ctx.register_llm_sanitize_response_guardrail(
                "output",
                sanitize_response,
                priority=priority,
            )

        if config.get("tool_input", True):

            async def sanitize_tool_input(_name: str, payload: Json) -> Json:
                return await executor.run(
                    sanitize_tool_payload,
                    payload,
                    sanitizer,
                    fallback=lambda: FAILURE_REPLACEMENT,
                )

            ctx.register_tool_sanitize_request_guardrail(
                "tool_input",
                sanitize_tool_input,
                priority=priority,
            )

        if config.get("tool_output", True):

            async def sanitize_tool_output(_name: str, payload: Json) -> Json:
                return await executor.run(
                    sanitize_tool_payload,
                    payload,
                    sanitizer,
                    fallback=lambda: FAILURE_REPLACEMENT,
                )

            ctx.register_tool_sanitize_response_guardrail(
                "tool_output",
                sanitize_tool_output,
                priority=priority,
            )


async def main() -> None:
    """Serve the Rampart worker until the Relay host requests shutdown."""
    await serve_plugin(RampartWorkerPlugin())


if __name__ == "__main__":
    asyncio.run(main())
