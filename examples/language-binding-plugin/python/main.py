# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runnable Python host for the application-owned documentation plugin."""

from __future__ import annotations

import asyncio
import os
import tempfile
from copy import deepcopy
from typing import Any

import nemo_relay
from nemo_relay import llm, plugin, subscribers, tools

DEFAULT_CONFIG: dict[str, Any] = {
    "tag": "documentation",
    "observe": {"enabled": True, "redact_keys": ["secret"]},
    "requests": {
        "enabled": True,
        "mode": "enforce",
        "blocked_tools": ["dangerous_tool"],
        "blocked_models": ["restricted-model"],
        "header_name": "x-nemo-relay-plugin",
        "header_value": "documentation",
        "priority": 20,
        "break_chain": False,
    },
    "execution": {"enabled": True, "priority": 30, "emit_pending_marks": True},
    "runtime": {"emit_marks": True, "emit_isolated_scope": True},
}

GROUP_FIELDS = {
    "observe": {"enabled", "redact_keys"},
    "requests": {
        "enabled",
        "mode",
        "blocked_tools",
        "blocked_models",
        "header_name",
        "header_value",
        "priority",
        "break_chain",
    },
    "execution": {"enabled", "priority", "emit_pending_marks"},
    "runtime": {"emit_marks", "emit_isolated_scope"},
}


def _diagnostic(
    level: str,
    code: str,
    field: str | None,
    message: str,
) -> dict[str, str]:
    diagnostic = {
        "level": level,
        "code": f"documentation-plugin.{code}",
        "component": "documentation-plugin",
        "message": message,
    }
    if field is not None:
        diagnostic["field"] = field
    return diagnostic


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = deepcopy(DEFAULT_CONFIG)
    if "tag" in config:
        settings["tag"] = config["tag"]
    for group in GROUP_FIELDS:
        if isinstance(config.get(group), dict):
            settings[group].update(config[group])
    return settings


def validate_documentation_config(config: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    allowed_top_level = {"tag", *GROUP_FIELDS}
    for key in config.keys() - allowed_top_level:
        diagnostics.append(
            _diagnostic(
                "warning",
                "unknown_field",
                key,
                f"unknown field '{key}' is not supported",
            )
        )
    for group, allowed in GROUP_FIELDS.items():
        value = config.get(group)
        if value is not None and not isinstance(value, dict):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "invalid_config",
                    group,
                    f"{group} must be an object",
                )
            )
            continue
        if isinstance(value, dict):
            for key in value.keys() - allowed:
                field = f"{group}.{key}"
                diagnostics.append(
                    _diagnostic(
                        "warning",
                        "unknown_field",
                        field,
                        f"unknown field '{field}' is not supported",
                    )
                )

    settings = normalized_config(config)
    expected_types = {
        "tag": str,
        "observe.enabled": bool,
        "observe.redact_keys": list,
        "requests.enabled": bool,
        "requests.mode": str,
        "requests.blocked_tools": list,
        "requests.blocked_models": list,
        "requests.header_name": str,
        "requests.header_value": str,
        "requests.priority": int,
        "requests.break_chain": bool,
        "execution.enabled": bool,
        "execution.priority": int,
        "execution.emit_pending_marks": bool,
        "runtime.emit_marks": bool,
        "runtime.emit_isolated_scope": bool,
    }
    for field, expected in expected_types.items():
        group, separator, key = field.partition(".")
        value = settings[group][key] if separator else settings[group]
        valid = type(value) is expected
        if not valid:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "invalid_config",
                    field,
                    f"{field} must be a {expected.__name__}",
                )
            )
    for field in ("observe.redact_keys", "requests.blocked_tools", "requests.blocked_models"):
        group, key = field.split(".")
        value = settings[group][key]
        if isinstance(value, list) and not all(isinstance(item, str) for item in value):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "invalid_config",
                    field,
                    f"{field} must contain only strings",
                )
            )
    requests = settings["requests"]
    if isinstance(requests["mode"], str) and requests["mode"] not in {"observe", "enforce"}:
        diagnostics.append(
            _diagnostic(
                "error",
                "unsupported_mode",
                "requests.mode",
                "requests.mode must be either observe or enforce",
            )
        )
    return diagnostics


class DocumentationPlugin:
    def __init__(self) -> None:
        self.events: list[str] = []

    def validate(self, config: dict[str, Any]) -> list[dict[str, str]]:
        return validate_documentation_config(config)

    def register(self, config: dict[str, Any], context: plugin.PluginContext) -> None:
        settings = normalized_config(config)
        tag = settings["tag"]
        observe = settings["observe"]
        requests = settings["requests"]
        execution = settings["execution"]
        if observe["enabled"]:
            context.register_subscriber("events", lambda event: self.events.append(event.name))
        if requests["enabled"]:
            context.register_tool_conditional_execution_guardrail(
                "tool-policy",
                10,
                lambda name, _args: (
                    f"tool '{name}' is blocked"
                    if requests["mode"] == "enforce" and name in requests["blocked_tools"]
                    else None
                ),
            )
            context.register_tool_request_intercept(
                "tool-request",
                requests["priority"],
                requests["break_chain"],
                lambda _name, args: {**args, "plugin_tag": tag},
            )

            def llm_policy(request):
                model = request.content.get("model") if isinstance(request.content, dict) else None
                if requests["mode"] == "enforce" and model in requests["blocked_models"]:
                    return f"model '{model}' is blocked"
                return None

            context.register_llm_conditional_execution_guardrail("llm-policy", 10, llm_policy)

            def llm_request(_name, request, annotated):
                return nemo_relay.LLMRequestInterceptOutcome(
                    nemo_relay.LLMRequest(
                        {**request.headers, requests["header_name"]: requests["header_value"]},
                        request.content,
                    ),
                    annotated,
                )

            context.register_llm_request_intercept(
                "llm-request",
                requests["priority"],
                requests["break_chain"],
                llm_request,
            )

        async def stream_request(_request, next_call):
            async for chunk in await next_call(_request):
                yield {**chunk, "plugin_stream": True}

        if execution["enabled"]:
            context.register_llm_stream_execution_intercept(
                "llm-stream",
                execution["priority"],
                stream_request,
            )


def component(mode: str, *, enabled: bool = True) -> plugin.PluginConfig:
    settings = deepcopy(DEFAULT_CONFIG)
    settings["requests"]["mode"] = mode
    return plugin.PluginConfig(
        components=[
            plugin.ComponentSpec(
                kind="documentation-plugin",
                enabled=enabled,
                config=settings,
            )
        ]
    )


async def main() -> dict[str, Any]:
    implementation = DocumentationPlugin()
    plugin.register("documentation-plugin", implementation)
    print("registered:", plugin.list_kinds())
    invalid = plugin.validate(component("invalid"))["diagnostics"]
    assert invalid[0]["code"] == "documentation-plugin.unsupported_mode"
    disabled_invalid = plugin.validate(component("invalid", enabled=False))["diagnostics"]
    assert disabled_invalid[0]["code"] == "documentation-plugin.unsupported_mode"
    print("invalid:", invalid)
    try:
        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            previous_config_home = os.environ.get("XDG_CONFIG_HOME")
            os.chdir(directory)
            os.environ["XDG_CONFIG_HOME"] = directory
            try:
                report = await plugin.initialize(component("enforce"))
            finally:
                os.chdir(previous_directory)
                if previous_config_home is None:
                    os.environ.pop("XDG_CONFIG_HOME", None)
                else:
                    os.environ["XDG_CONFIG_HOME"] = previous_config_home
        print("active:", report)
        tool_result = await tools.execute("safe_tool", {"value": 1}, lambda args: args)
        assert tool_result == {"value": 1, "plugin_tag": "documentation"}
        print("tool:", tool_result)
        request = nemo_relay.LLMRequest({}, {"model": "allowed-model"})
        llm_result = await llm.execute("allowed-model", request, lambda req: {"headers": req.headers})
        assert llm_result["headers"]["x-nemo-relay-plugin"] == "documentation"
        print("llm:", llm_result)

        async def provider(_request):
            yield {"chunk": 1}
            yield {"chunk": 2}

        chunks: list[dict[str, Any]] = []
        stream = await llm.stream_execute("allowed-model", request, provider, chunks.append, lambda: {"done": True})
        streamed: list[dict[str, Any]] = []
        async for chunk in stream:
            streamed.append(chunk)
            print("stream:", chunk)
        assert len(streamed) == 2
        assert all(chunk["plugin_stream"] is True for chunk in streamed)
        await subscribers.flush_async()
        assert implementation.events
        print("events:", implementation.events)
    finally:
        await plugin.clear_async()
        plugin.deregister("documentation-plugin")
    print("teardown: complete")
    assert "documentation-plugin" not in plugin.list_kinds()
    return {
        "invalid": invalid,
        "report": report,
        "tool": tool_result,
        "llm": llm_result,
        "stream": streamed,
        "events": implementation.events,
    }


if __name__ == "__main__":
    asyncio.run(main())
