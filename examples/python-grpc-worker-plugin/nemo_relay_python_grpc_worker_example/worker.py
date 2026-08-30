# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete Python grpc-v1 worker used by the plugin authoring documentation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, cast

from nemo_relay_plugin import (
    ConfigDiagnostic,
    DiagnosticLevel,
    EventSanitizeFields,
    Json,
    LlmOptimizationContribution,
    LlmRequestInterceptOutcome,
    PendingMarkSpec,
    PluginContext,
    RuntimeRegistrationKind,
    ScopeType,
    ToolExecutionInterceptOutcome,
    ToolExecutionResult,
    WorkerPlugin,
    serve_plugin,
)

DEFAULT_CONFIG: dict[str, Json] = {
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
        "kinds": [RuntimeRegistrationKind.SUBSCRIBER.value],
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


class ExamplePythonWorker(WorkerPlugin):
    """Install every safe worker registration surface from one explicit config."""

    plugin_id = "examples.python_grpc_worker"
    allows_multiple_components = False

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        diagnostics: list[ConfigDiagnostic | dict[str, Any]] = []
        diagnostics.extend(validate_config(config))
        return diagnostics

    def register(self, ctx: PluginContext, config: Json) -> None:
        diagnostics = validate_config(config)
        errors = [diagnostic for diagnostic in diagnostics if diagnostic.level == DiagnosticLevel.ERROR]
        if errors:
            raise ValueError(errors[0].message)
        settings = normalized_config(config)
        tag = cast(str, settings["tag"])
        observe = cast(dict[str, Json], settings["observe"])
        requests = cast(dict[str, Json], settings["requests"])
        execution = cast(dict[str, Json], settings["execution"])
        runtime_settings = cast(dict[str, Json], settings["runtime"])
        registration_control = cast(dict[str, Json], settings["registration_control"])

        if registration_control["enabled"]:
            reason = cast(str, registration_control["reason"])
            ctx.register_conditional_middleware_guardrail(
                "documentation_registration_control",
                {RuntimeRegistrationKind(kind) for kind in cast(list[str], registration_control["kinds"])},
                cast(str, registration_control["registration_name"]),
                lambda _kinds, name: reason if name.startswith("documentation-controlled-") else None,
            )

        if observe["enabled"]:
            self._register_observation(ctx, tag, observe)
        if requests["enabled"]:
            self._register_requests(ctx, tag, requests, execution)
        self._register_runtime(ctx, tag, runtime_settings)
        if execution["enabled"]:
            self._register_execution(ctx, tag, execution)

    @staticmethod
    def _register_observation(ctx: PluginContext, tag: str, observe: dict[str, Json]) -> None:
        redact_keys = cast(list[str], observe["redact_keys"])

        async def inject_event_metadata(event: dict[str, Any]) -> dict[str, Json]:
            if str(event.get("name", "")).startswith("external-plugin-event-metadata-injection"):
                return {"external.injector.transport": "python_grpc_worker"}
            return {}

        async def subscriber(event: dict[str, Any]) -> None:
            if not str(event.get("name", "")).startswith("example.python_worker"):
                await ctx.runtime.emit_mark(
                    "example.python_worker.subscriber.seen",
                    {"event": event.get("name"), "tag": tag},
                )

        def sanitize_event(_event: dict[str, Any], fields: EventSanitizeFields) -> EventSanitizeFields:
            metadata = _redact(fields.get("metadata") or {}, redact_keys)
            if not isinstance(metadata, dict):
                metadata = {"original": metadata}
            return {
                "data": _redact(fields.get("data"), redact_keys),
                "category_profile": cast(
                    dict[str, Any] | None,
                    _redact(fields.get("category_profile"), redact_keys),
                ),
                "metadata": {**metadata, "plugin_tag": tag},
            }

        def sanitize_tool(_name: str, value: Json) -> Json:
            return _redact(value, redact_keys)

        async def sanitize_llm_request(request: dict[str, Any], context: Any) -> dict[str, Any]:
            request = deepcopy(request)
            codec = context.resolve_codec()
            if codec is not None:
                annotated = await codec.decode(request)
                annotated = _redact(annotated, redact_keys)
                request = await codec.encode(annotated, request)
            request["content"] = _redact(request.get("content"), redact_keys)
            return request

        async def sanitize_llm_response(response: Json, context: Any) -> Json:
            codec = context.resolve_codec()
            if codec is not None:
                await codec.decode(response)
            return _redact(response, redact_keys)

        ctx.register_event_metadata_injector("documentation_event_metadata_injector", inject_event_metadata)
        ctx.register_subscriber("documentation_subscriber", subscriber)
        ctx.register_mark_sanitize_guardrail("documentation_mark_sanitizer", sanitize_event, priority=10)
        ctx.register_scope_sanitize_start_guardrail("documentation_scope_start_sanitizer", sanitize_event, priority=10)
        ctx.register_scope_sanitize_end_guardrail("documentation_scope_end_sanitizer", sanitize_event, priority=10)
        ctx.register_tool_sanitize_request_guardrail("documentation_tool_request_sanitizer", sanitize_tool, priority=10)
        ctx.register_tool_sanitize_response_guardrail(
            "documentation_tool_response_sanitizer", sanitize_tool, priority=10
        )
        ctx.register_llm_sanitize_request_guardrail(
            "documentation_llm_request_sanitizer", sanitize_llm_request, priority=10
        )
        ctx.register_llm_sanitize_response_guardrail(
            "documentation_llm_response_sanitizer", sanitize_llm_response, priority=10
        )

    @staticmethod
    def _register_requests(
        ctx: PluginContext,
        tag: str,
        requests: dict[str, Json],
        execution: dict[str, Json],
    ) -> None:
        priority = cast(int, requests["priority"])
        break_chain = cast(bool, requests["break_chain"])
        blocked_tools = cast(list[str], requests["blocked_tools"])
        blocked_models = cast(list[str], requests["blocked_models"])
        mode = cast(str, requests["mode"])

        def tool_policy(name: str, _args: Json) -> str | None:
            if mode == "enforce" and name in blocked_tools:
                return f"tool '{name}' is blocked by documentation policy"
            return None

        async def tool_request(name: str, args: Json) -> Json:
            if not isinstance(args, dict):
                return args
            return {
                **args,
                "_nemo_relay_plugin": {"tag": tag, "tool": name},
                "plugin_tag": tag,
                "plugin_tool": name,
            }

        def llm_policy(request: dict[str, Any]) -> str | None:
            content = request.get("content")
            model = content.get("model") if isinstance(content, dict) else None
            if mode == "enforce" and model in blocked_models:
                return f"model '{model}' is blocked by documentation policy"
            return None

        def llm_request(
            _name: str,
            request: dict[str, Any],
            annotated: dict[str, Any] | None,
        ) -> LlmRequestInterceptOutcome:
            rewritten = deepcopy(request)
            headers = rewritten.get("headers")
            if not isinstance(headers, dict):
                headers = {}
            rewritten["headers"] = {
                **headers,
                cast(str, requests["header_name"]): cast(str, requests["header_value"]),
            }
            marks = (
                [PendingMarkSpec(name="example.python_worker.llm_request", data={"tag": tag})]
                if execution["emit_pending_marks"]
                else []
            )
            return LlmRequestInterceptOutcome(
                request=rewritten,
                annotated_request=annotated,
                pending_marks=marks,
                optimization_contributions=[
                    LlmOptimizationContribution(
                        producer="examples.python_grpc_worker",
                        kind="request_rewrite",
                        applied=True,
                    )
                ],
            )

        ctx.register_tool_conditional_execution_guardrail("documentation_tool_policy", tool_policy, priority=10)
        ctx.register_tool_request_intercept(
            "documentation_tool_request", tool_request, priority=priority, break_chain=break_chain
        )
        ctx.register_llm_conditional_execution_guardrail("documentation_llm_policy", llm_policy, priority=10)
        ctx.register_llm_request_intercept(
            "documentation_llm_request", llm_request, priority=priority, break_chain=break_chain
        )

    @staticmethod
    def _register_runtime(ctx: PluginContext, tag: str, settings: dict[str, Json]) -> None:
        if not settings["emit_marks"] and not settings["emit_isolated_scope"]:
            return

        async def runtime_events(_name: str, args: Json, next_call: Any) -> ToolExecutionInterceptOutcome:
            await _emit_runtime_events(ctx, tag, settings)
            downstream = await next_call.call(args)
            return ToolExecutionInterceptOutcome(
                result=downstream.result,
                annotation=downstream.annotation,
            )

        ctx.register_tool_execution_intercept("documentation_runtime_events", runtime_events, priority=0)

    @staticmethod
    def _register_execution(ctx: PluginContext, tag: str, execution: dict[str, Json]) -> None:
        priority = cast(int, execution["priority"])
        emit_pending_marks = cast(bool, execution["emit_pending_marks"])

        async def tool_execution(name: str, args: Json, next_call: Any) -> ToolExecutionInterceptOutcome:
            result: ToolExecutionResult = await next_call.call(args)
            marks = (
                [
                    PendingMarkSpec(
                        name="example.python_worker.tool_execution",
                        data={"tool_name": name, "tag": tag},
                    )
                ]
                if emit_pending_marks
                else []
            )
            return ToolExecutionInterceptOutcome(
                result=result.result,
                annotation={
                    "upstream": result.annotation,
                    "worker": {"tool_name": name, "tag": tag},
                },
                pending_marks=marks,
            )

        async def llm_execution(_name: str, request: dict[str, Any], next_call: Any) -> Json:
            content = request.get("content")
            repeat = isinstance(content, dict) and content.get("repeat_downstream") is True
            if repeat:
                first, _second = await asyncio.gather(
                    next_call.call(request), next_call.call(request), return_exceptions=True
                )
                if isinstance(first, BaseException):
                    raise first
                return first
            return await next_call.call(request)

        async def llm_stream_execution(
            _name: str,
            request: dict[str, Any],
            next_call: Any,
        ) -> AsyncIterator[Json]:
            async for chunk in next_call.call(request):
                if isinstance(chunk, dict):
                    yield {**chunk, "plugin_stream": True}
                else:
                    yield chunk

        ctx.register_tool_execution_intercept("documentation_tool_execution", tool_execution, priority=priority)
        ctx.register_llm_execution_intercept("documentation_llm_execution", llm_execution, priority=priority)
        ctx.register_llm_stream_execution_intercept(
            "documentation_llm_stream_execution", llm_stream_execution, priority=priority
        )


def validate_config(config: Json) -> list[ConfigDiagnostic]:
    if not isinstance(config, dict):
        return [_diagnostic(DiagnosticLevel.ERROR, "invalid_config", None, "plugin config must be a JSON object")]

    diagnostics: list[ConfigDiagnostic] = []
    allowed_top = {"tag", *GROUP_FIELDS}
    for key in config.keys() - allowed_top:
        diagnostics.append(_diagnostic(DiagnosticLevel.ERROR, "unknown_field", key, f"unknown field '{key}'"))
    if "tag" in config and (not isinstance(config["tag"], str) or not config["tag"]):
        diagnostics.append(_diagnostic(DiagnosticLevel.ERROR, "invalid_tag", "tag", "tag must be a non-empty string"))

    for group, fields in GROUP_FIELDS.items():
        value = config.get(group)
        if value is not None and not isinstance(value, dict):
            diagnostics.append(_diagnostic(DiagnosticLevel.ERROR, "invalid_group", group, f"{group} must be an object"))
            continue
        if isinstance(value, dict):
            for key in value.keys() - fields:
                path = f"{group}.{key}"
                diagnostics.append(_diagnostic(DiagnosticLevel.ERROR, "unknown_field", path, f"unknown field '{path}'"))

    settings = normalized_config(config)
    for path in (
        "observe.enabled",
        "requests.enabled",
        "requests.break_chain",
        "execution.enabled",
        "execution.emit_pending_marks",
        "runtime.emit_marks",
        "runtime.emit_isolated_scope",
        "registration_control.enabled",
    ):
        if not isinstance(_path(settings, path), bool):
            diagnostics.append(_diagnostic(DiagnosticLevel.ERROR, "invalid_type", path, f"{path} must be a boolean"))
    for path in ("requests.priority", "execution.priority"):
        value = _path(settings, path)
        if isinstance(value, bool) or not isinstance(value, int):
            diagnostics.append(_diagnostic(DiagnosticLevel.ERROR, "invalid_type", path, f"{path} must be an integer"))
    for path in ("requests.blocked_tools", "requests.blocked_models", "observe.redact_keys"):
        value = _path(settings, path)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            diagnostics.append(
                _diagnostic(DiagnosticLevel.ERROR, "invalid_type", path, f"{path} must be an array of strings")
            )
    kinds = _path(settings, "registration_control.kinds")
    if (
        not isinstance(kinds, list)
        or not kinds
        or not all(isinstance(item, str) and item in {kind.value for kind in RuntimeRegistrationKind} for item in kinds)
    ):
        diagnostics.append(
            _diagnostic(
                DiagnosticLevel.ERROR,
                "invalid_type",
                "registration_control.kinds",
                "registration_control.kinds must be a non-empty array of supported registration kinds",
            )
        )
    for path in ("requests.header_name", "requests.header_value"):
        value = _path(settings, path)
        if not isinstance(value, str) or not value:
            diagnostics.append(
                _diagnostic(DiagnosticLevel.ERROR, "invalid_type", path, f"{path} must be a non-empty string")
            )
    for path in ("registration_control.registration_name", "registration_control.reason"):
        value = _path(settings, path)
        if not isinstance(value, str) or not value:
            diagnostics.append(
                _diagnostic(DiagnosticLevel.ERROR, "invalid_type", path, f"{path} must be a non-empty string")
            )
    mode = _path(settings, "requests.mode")
    if mode not in {"observe", "enforce"}:
        diagnostics.append(
            _diagnostic(
                DiagnosticLevel.ERROR,
                "unsupported_mode",
                "requests.mode",
                "requests.mode must be either observe or enforce",
            )
        )
    return diagnostics


def normalized_config(config: Json) -> dict[str, Json]:
    settings = deepcopy(DEFAULT_CONFIG)
    if not isinstance(config, dict):
        return settings
    if "tag" in config:
        settings["tag"] = config["tag"]
    for group in GROUP_FIELDS:
        supplied = config.get(group)
        if isinstance(supplied, dict):
            cast(dict[str, Json], settings[group]).update(supplied)
        elif supplied is not None:
            settings[group] = supplied
    return settings


async def _emit_runtime_events(ctx: PluginContext, tag: str, settings: dict[str, Json]) -> None:
    handle = await ctx.runtime.push_scope(
        "example.python_worker.request",
        scope_type=ScopeType.CUSTOM,
        data={"tag": tag},
    )
    try:
        if settings["emit_marks"]:
            await ctx.runtime.emit_mark("example.python_worker.tool_request", {"tag": tag})
    except BaseException:
        try:
            await ctx.runtime.pop_scope(handle, metadata={"failed": True})
        except BaseException:
            pass
        raise
    else:
        await ctx.runtime.pop_scope(handle, output={"done": True})
    if settings["emit_isolated_scope"]:
        stack_id = await ctx.runtime.create_scope_stack()
        try:
            with ctx.runtime.bind_scope_stack(stack_id):
                if settings["emit_marks"]:
                    await ctx.runtime.emit_mark("example.python_worker.isolated.mark", {"tag": tag})
        finally:
            await ctx.runtime.drop_scope_stack(stack_id)


def _redact(value: Json, redact_keys: list[str]) -> Json:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key in redact_keys else _redact(item, redact_keys) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, redact_keys) for item in value]
    return value


def _path(config: dict[str, Json], path: str) -> Json:
    group, field = path.split(".", maxsplit=1)
    value = config[group]
    return value.get(field) if isinstance(value, dict) else None


def _diagnostic(
    level: DiagnosticLevel,
    suffix: str,
    field: str | None,
    message: str,
) -> ConfigDiagnostic:
    return ConfigDiagnostic(
        level=level,
        code=f"examples.python_grpc_worker.{suffix}",
        component=ExamplePythonWorker.plugin_id,
        field=field,
        message=message,
    )


async def main() -> None:
    """Serve the worker entrypoint referenced by relay-plugin.toml."""
    await serve_plugin(ExamplePythonWorker())


if __name__ == "__main__":
    asyncio.run(main())
