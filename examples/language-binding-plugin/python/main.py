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
from nemo_relay import llm, plugin, scope, subscribers, tools
from nemo_relay.runtime_registrations import RuntimeRegistrationKind

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
    "registration_control": {
        "enabled": False,
        "kinds": [RuntimeRegistrationKind.SUBSCRIBER],
        "registration_name": "documentation-controlled-subscriber",
        "reason": "disabled by documentation plugin",
    },
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
    "registration_control": {"enabled", "kinds", "registration_name", "reason"},
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


def redact_json(value: Any, redact_keys: list[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key in redact_keys else redact_json(item, redact_keys) for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_json(item, redact_keys) for item in value]
    return value


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
        "registration_control.enabled": bool,
        "registration_control.kinds": list,
        "registration_control.registration_name": str,
        "registration_control.reason": str,
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
    kinds = settings["registration_control"]["kinds"]
    if isinstance(kinds, list):
        supported_kinds = {kind.value for kind in RuntimeRegistrationKind}
        if not kinds or not all(isinstance(item, str) and item in supported_kinds for item in kinds):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "invalid_config",
                    "registration_control.kinds",
                    "registration_control.kinds must be a non-empty array of supported registration kinds",
                )
            )
    if isinstance(settings["tag"], str) and not settings["tag"]:
        diagnostics.append(_diagnostic("error", "invalid_tag", "tag", "tag must be a non-empty string"))
    for field in ("requests.header_name", "requests.header_value"):
        group, key = field.split(".")
        value = settings[group][key]
        if isinstance(value, str) and not value:
            diagnostics.append(_diagnostic("error", "invalid_header", field, f"{field} must be a non-empty string"))
    for field in ("registration_control.registration_name", "registration_control.reason"):
        group, key = field.split(".")
        value = settings[group][key]
        if isinstance(value, str) and not value:
            diagnostics.append(_diagnostic("error", "invalid_config", field, f"{field} must be a non-empty string"))
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
        registration_control = settings["registration_control"]
        if registration_control["enabled"]:
            context.register_conditional_middleware_guardrail(
                "documentation-registration-control",
                {RuntimeRegistrationKind(kind) for kind in registration_control["kinds"]},
                registration_control["registration_name"],
                lambda _kinds, _registration_name: registration_control["reason"],
            )
        if observe["enabled"]:
            context.register_subscriber("events", lambda event: self.events.append(event.name))

            def sanitize_event(_event, fields):
                return {
                    "data": redact_json(fields["data"], observe["redact_keys"]),
                    "category_profile": redact_json(fields["category_profile"], observe["redact_keys"]),
                    "metadata": {
                        **(redact_json(fields["metadata"], observe["redact_keys"]) or {}),
                        "plugin_tag": tag,
                    },
                }

            context.register_mark_sanitize_guardrail("mark-sanitizer", 10, sanitize_event)
            context.register_scope_sanitize_start_guardrail("scope-start-sanitizer", 10, sanitize_event)
            context.register_scope_sanitize_end_guardrail("scope-end-sanitizer", 10, sanitize_event)
            context.register_tool_sanitize_request_guardrail(
                "tool-request-sanitizer", 10, lambda _name, value: redact_json(value, observe["redact_keys"])
            )
            context.register_tool_sanitize_response_guardrail(
                "tool-response-sanitizer", 10, lambda _name, value: redact_json(value, observe["redact_keys"])
            )

            def sanitize_llm_request(request, _codec_context):
                return nemo_relay.LLMRequest(request.headers, redact_json(request.content, observe["redact_keys"]))

            context.register_llm_sanitize_request_guardrail("llm-request-sanitizer", 10, sanitize_llm_request)
            context.register_llm_sanitize_response_guardrail(
                "llm-response-sanitizer", 10, lambda value, _codec_context: redact_json(value, observe["redact_keys"])
            )
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

        if settings["runtime"]["emit_marks"] or settings["runtime"]["emit_isolated_scope"]:

            async def runtime_events(_name, args, next_call):
                if settings["runtime"]["emit_marks"]:
                    scope.event(
                        "documentation-plugin.request",
                        data={"tag": tag, "secret": "application-value"},
                    )
                if settings["runtime"]["emit_isolated_scope"]:
                    with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
                        with scope.scope("documentation-plugin.isolated", nemo_relay.ScopeType.Custom):
                            pass
                downstream = await next_call(args)
                return nemo_relay.ToolExecutionInterceptOutcome(
                    downstream.result,
                    annotation=downstream.annotation,
                )

            context.register_tool_execution_intercept("runtime-events", 0, runtime_events)

        async def stream_request(_request, next_call):
            async for chunk in await next_call(_request):
                yield {**chunk, "plugin_stream": True}

        if execution["enabled"]:

            async def tool_execution(_name, args, next_call):
                result = await next_call(args)
                marks = (
                    [nemo_relay.PendingMarkSpec("documentation-plugin.tool-complete")]
                    if execution["emit_pending_marks"]
                    else []
                )
                return nemo_relay.ToolExecutionInterceptOutcome(
                    result.result,
                    marks,
                    annotation=result.annotation,
                )

            context.register_tool_execution_intercept("tool-execution", execution["priority"], tool_execution)

            async def llm_execution(_name, request, next_call):
                return await next_call(request)

            context.register_llm_execution_intercept("llm-execution", execution["priority"], llm_execution)
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
        tool_result = await tools.execute(
            "safe_tool",
            {"value": 1},
            lambda args: nemo_relay.ToolExecutionResult(args, {"source": "application"}),
        )
        assert tool_result.result == {"value": 1, "plugin_tag": "documentation"}
        assert tool_result.annotation == {"source": "application"}
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
