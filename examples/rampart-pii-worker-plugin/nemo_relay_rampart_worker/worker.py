# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process Rampart sanitizer implemented with Relay's worker SDK."""

from __future__ import annotations

import asyncio
from typing import Any, cast

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

from ._detectors import DEFAULT_MAX_LATENCY_MS, RampartSanitizer
from ._model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, load_classifier
from ._sanitization import sanitize_llm_request, sanitize_llm_response, sanitize_tool_payload

PLUGIN_ID = "nvidia.rampart_pii"
_MIN_PRIORITY = -(2**31)
_MAX_PRIORITY = 2**31 - 2
_CONFIG_FIELDS = frozenset(
    {
        "version",
        "allow_network",
        "max_latency_ms",
        "priority",
        "input",
        "output",
        "tool_input",
        "tool_output",
    }
)


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
    if version != 1 or isinstance(version, bool):
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
        priority = cast(int, config.get("priority", 100))
        classifier = await asyncio.to_thread(
            load_classifier,
            DEFAULT_MODEL_ID,
            DEFAULT_MODEL_REVISION,
            allow_network,
        )
        sanitizer = RampartSanitizer(classifier, max_latency_ms=max_latency_ms)

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
                return await asyncio.to_thread(sanitize_llm_request, request, sanitizer)

            ctx.register_llm_sanitize_request_guardrail(
                "input",
                sanitize_request,
                priority=priority,
            )

        if config.get("output", True):

            async def sanitize_response(response: Json) -> Json:
                return await asyncio.to_thread(sanitize_llm_response, response, sanitizer)

            ctx.register_llm_sanitize_response_guardrail(
                "output",
                sanitize_response,
                priority=priority,
            )

        if config.get("tool_input", True):

            async def sanitize_tool_input(_name: str, payload: Json) -> Json:
                return await asyncio.to_thread(sanitize_tool_payload, payload, sanitizer)

            ctx.register_tool_sanitize_request_guardrail(
                "tool_input",
                sanitize_tool_input,
                priority=priority,
            )

        if config.get("tool_output", True):

            async def sanitize_tool_output(_name: str, payload: Json) -> Json:
                return await asyncio.to_thread(sanitize_tool_payload, payload, sanitizer)

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
