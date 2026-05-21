# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation for the NeMo Guardrails example plugin."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from typing import Any, Protocol, cast

from nemo_relay import Json, LLMRequest
from nemo_relay import plugin as relay_plugin
from nemo_relay.codecs import (
    AnthropicMessagesCodec,
    LlmCodec,
    LlmResponseCodec,
    OpenAIChatCodec,
    OpenAIResponsesCodec,
)

DEFAULT_KIND = "nemoguardrails"
_DEFAULT_PRIORITY = 100


class NeMoGuardrailsDependencyError(RuntimeError):
    """Raised when the optional ``nemoguardrails`` dependency is unavailable."""


class NeMoGuardrailsViolation(RuntimeError):
    """Raised when NeMo Guardrails blocks or cannot safely apply a rail result."""

    def __init__(
        self,
        message: str,
        *,
        rail_type: str,
        rail: str | None = None,
        content: str | None = None,
    ) -> None:
        super().__init__(message)
        self.rail_type = rail_type
        self.rail = rail
        self.content = content


class _GuardrailsCodec(LlmCodec, LlmResponseCodec, Protocol):
    """Codec shape required by this example plugin."""


_CODECS: dict[str, Callable[[], _GuardrailsCodec]] = {
    "openai_chat": OpenAIChatCodec,
    "openai_responses": OpenAIResponsesCodec,
    "anthropic_messages": AnthropicMessagesCodec,
}
_CODEC_NAMES = ", ".join(_CODECS)


def _diagnostic(code: str, message: str, *, field: str | None = None) -> dict[str, str]:
    diagnostic = {
        "level": "error",
        "code": code,
        "message": message,
    }
    if field is not None:
        diagnostic["field"] = field
    return diagnostic


def _load_nemoguardrails():
    try:
        guardrails = cast(Any, importlib.import_module("nemoguardrails"))
        options = cast(Any, importlib.import_module("nemoguardrails.rails.llm.options"))
    except ImportError as error:
        raise NeMoGuardrailsDependencyError(
            "NeMo Guardrails is required for the NeMo Guardrails example plugin. "
            "Install it with: pip install nemoguardrails"
        ) from error

    return (
        guardrails.RailsConfig,
        guardrails.LLMRails,
        options.RailType,
        options.RailStatus,
    )


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status)).lower()


def _messages_from_annotated(annotated: Any) -> list[dict[str, Any]]:
    messages = annotated.messages
    return [dict(message) for message in messages]


def _replace_last_role_content(messages: list[dict[str, Any]], role: str, content: str) -> list[dict[str, Any]]:
    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == role:
            updated[index]["content"] = content
            return updated
    raise NeMoGuardrailsViolation(
        f"NeMo Guardrails returned modified {role} content but no {role} message was present.",
        rail_type="input" if role == "user" else "output",
        content=content,
    )


def _tool_input_content(name: str, args: Json) -> str:
    return json.dumps(
        {
            "tool_name": name,
            "arguments": args,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _tool_output_content(name: str, args: Json, result: Json) -> str:
    return json.dumps(
        {
            "tool_name": name,
            "arguments": args,
            "result": result,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _modified_tool_payload(content: str, field: str) -> Json:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise NeMoGuardrailsViolation(
            f"NeMo Guardrails returned modified tool {field} content that is not valid JSON.",
            rail_type=f"tool_{field}",
            content=content,
        ) from error

    if not isinstance(value, dict) or field not in value:
        raise NeMoGuardrailsViolation(
            f"NeMo Guardrails returned modified tool {field} content without a '{field}' field.",
            rail_type=f"tool_{field}",
            content=content,
        )
    return cast(Json, value[field])


def _validate_config(plugin_config: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = []

    has_config_path = "config_path" in plugin_config
    has_config_yaml = "config_yaml" in plugin_config
    if has_config_path == has_config_yaml:
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.config_source",
                "Exactly one of config_path or config_yaml is required.",
            )
        )

    if has_config_path and not isinstance(plugin_config.get("config_path"), str):
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.invalid_config_path",
                "config_path must be a string.",
                field="config_path",
            )
        )
    elif has_config_path and not plugin_config["config_path"].strip():
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.invalid_config_path",
                "config_path must not be empty.",
                field="config_path",
            )
        )

    if has_config_yaml and not isinstance(plugin_config.get("config_yaml"), str):
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.invalid_config_yaml",
                "config_yaml must be a string.",
                field="config_yaml",
            )
        )
    elif has_config_yaml and not plugin_config["config_yaml"].strip():
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.invalid_config_yaml",
                "config_yaml must not be empty.",
                field="config_yaml",
            )
        )

    colang_content = plugin_config.get("colang_content")
    if colang_content is not None and not isinstance(colang_content, str):
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.invalid_colang_content",
                "colang_content must be a string when provided.",
                field="colang_content",
            )
        )
    elif isinstance(colang_content, str) and not colang_content.strip():
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.invalid_colang_content",
                "colang_content must not be empty when provided.",
                field="colang_content",
            )
        )
    if colang_content is not None and not has_config_yaml:
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.colang_requires_config_yaml",
                "colang_content can only be used with config_yaml.",
                field="colang_content",
            )
        )

    rail_switches = {
        "input": plugin_config.get("input", True),
        "output": plugin_config.get("output", True),
        "tool_input": plugin_config.get("tool_input", False),
        "tool_output": plugin_config.get("tool_output", False),
    }
    for field, value in rail_switches.items():
        if not isinstance(value, bool):
            diagnostics.append(
                _diagnostic(f"nemoguardrails.invalid_{field}", f"{field} must be a boolean.", field=field)
            )
    if all(isinstance(value, bool) and not value for value in rail_switches.values()):
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.no_rails_enabled",
                "At least one of input, output, tool_input, or tool_output must be enabled.",
            )
        )

    llm_rails_enabled = rail_switches["input"] is True or rail_switches["output"] is True
    codec = plugin_config.get("codec")
    if llm_rails_enabled and not isinstance(codec, str):
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.invalid_codec",
                f"codec is required when input or output is enabled and must be one of: {_CODEC_NAMES}.",
                field="codec",
            )
        )
    elif isinstance(codec, str) and codec not in _CODECS:
        diagnostics.append(
            _diagnostic(
                "nemoguardrails.unsupported_codec",
                f"Unsupported codec. Expected one of: {_CODEC_NAMES}.",
                field="codec",
            )
        )

    priority = plugin_config.get("priority", _DEFAULT_PRIORITY)
    if not isinstance(priority, int) or isinstance(priority, bool):
        diagnostics.append(
            _diagnostic("nemoguardrails.invalid_priority", "priority must be an integer.", field="priority")
        )

    return diagnostics


def _raise_blocked(result: Any, rail_type: str) -> None:
    rail_value = getattr(result, "rail", None)
    rail = None if rail_value is None else str(rail_value)
    content = getattr(result, "content", "")
    detail = f" by rail '{rail}'" if rail else ""
    subject = "LLM call" if rail_type in {"input", "output"} else "tool call"
    raise NeMoGuardrailsViolation(
        f"NeMo Guardrails {rail_type} rail blocked the {subject}{detail}.",
        rail_type=rail_type,
        rail=rail,
        content="" if content is None else str(content),
    )


class NeMoGuardrailsPlugin:
    """Plugin that applies NeMo Guardrails input/output checks to LLM calls."""

    def validate(self, plugin_config: dict[str, Any]) -> list[dict[str, str]]:
        return _validate_config(plugin_config)

    def register(self, plugin_config: dict[str, Any], context: Any) -> None:
        diagnostics = _validate_config(plugin_config)
        if diagnostics:
            message = "; ".join(diagnostic["message"] for diagnostic in diagnostics)
            raise ValueError(f"Invalid NeMo Guardrails plugin config: {message}")

        RailsConfig, LLMRails, RailType, RailStatus = _load_nemoguardrails()

        if "config_path" in plugin_config:
            guardrails_config = RailsConfig.from_path(plugin_config["config_path"])
        else:
            guardrails_config = RailsConfig.from_content(
                colang_content=plugin_config.get("colang_content"),
                yaml_content=plugin_config["config_yaml"],
            )

        rails = LLMRails(guardrails_config)
        enable_input = bool(plugin_config.get("input", True))
        enable_output = bool(plugin_config.get("output", True))
        enable_tool_input = bool(plugin_config.get("tool_input", False))
        enable_tool_output = bool(plugin_config.get("tool_output", False))
        priority = int(plugin_config.get("priority", _DEFAULT_PRIORITY))

        if enable_input or enable_output:
            codec_name = str(plugin_config["codec"])
            codec = _CODECS[codec_name]()

            async def intercept(_name: str, request: LLMRequest, next_call):
                current_request = request
                annotated_request = codec.decode(current_request)
                messages = _messages_from_annotated(annotated_request)

                if enable_input:
                    input_result = await rails.check_async(messages, rail_types=[RailType.INPUT])
                    input_status = _status_value(input_result.status)
                    if input_status == _status_value(RailStatus.BLOCKED):
                        _raise_blocked(input_result, "input")
                    if input_status == _status_value(RailStatus.MODIFIED):
                        input_content = getattr(input_result, "content", "")
                        annotated_request.messages = _replace_last_role_content(
                            messages,
                            "user",
                            "" if input_content is None else str(input_content),
                        )
                        current_request = codec.encode(annotated_request, current_request)
                        messages = _messages_from_annotated(annotated_request)

                response = await next_call(current_request)

                if not enable_output:
                    return response

                annotated_response = codec.decode_response(response)
                response_text = annotated_response.response_text()
                if response_text is None:
                    return response

                output_messages = [*messages, {"role": "assistant", "content": response_text}]
                output_result = await rails.check_async(output_messages, rail_types=[RailType.OUTPUT])
                output_status = _status_value(output_result.status)
                if output_status == _status_value(RailStatus.BLOCKED):
                    _raise_blocked(output_result, "output")
                if output_status == _status_value(RailStatus.MODIFIED):
                    output_content = getattr(output_result, "content", "")
                    output_rail = getattr(output_result, "rail", None)
                    raise NeMoGuardrailsViolation(
                        "NeMo Guardrails output rail returned modified content, but this example plugin does not "
                        "rewrite provider responses.",
                        rail_type="output",
                        rail=None if output_rail is None else str(output_rail),
                        content="" if output_content is None else str(output_content),
                    )

                return response

            context.register_llm_execution_intercept("nemoguardrails", priority, intercept)

        if enable_tool_input or enable_tool_output:

            async def tool_intercept(tool_name: str, args: Json, next_call):
                current_args = args

                if enable_tool_input:
                    input_result = await rails.check_async(
                        [{"role": "user", "content": _tool_input_content(tool_name, current_args)}],
                        rail_types=[RailType.INPUT],
                    )
                    input_status = _status_value(input_result.status)
                    if input_status == _status_value(RailStatus.BLOCKED):
                        _raise_blocked(input_result, "tool_input")
                    if input_status == _status_value(RailStatus.MODIFIED):
                        input_content = getattr(input_result, "content", "")
                        current_args = _modified_tool_payload(
                            "" if input_content is None else str(input_content),
                            "arguments",
                        )

                tool_result = await next_call(current_args)

                if not enable_tool_output:
                    return tool_result

                output_result = await rails.check_async(
                    [
                        {"role": "user", "content": _tool_input_content(tool_name, current_args)},
                        {"role": "assistant", "content": _tool_output_content(tool_name, current_args, tool_result)},
                    ],
                    rail_types=[RailType.OUTPUT],
                )
                output_status = _status_value(output_result.status)
                if output_status == _status_value(RailStatus.BLOCKED):
                    _raise_blocked(output_result, "tool_output")
                if output_status == _status_value(RailStatus.MODIFIED):
                    output_content = getattr(output_result, "content", "")
                    return _modified_tool_payload("" if output_content is None else str(output_content), "result")

                return tool_result

            context.register_tool_execution_intercept("nemoguardrails", priority, tool_intercept)


def register(kind: str = DEFAULT_KIND) -> None:
    """Register the NeMo Guardrails plugin kind with NeMo Relay."""

    relay_plugin.register(kind, cast(relay_plugin.Plugin, NeMoGuardrailsPlugin()))


def deregister(kind: str = DEFAULT_KIND) -> bool:
    """Deregister the NeMo Guardrails plugin kind from NeMo Relay."""

    return relay_plugin.deregister(kind)


__all__ = [
    "DEFAULT_KIND",
    "NeMoGuardrailsDependencyError",
    "NeMoGuardrailsPlugin",
    "NeMoGuardrailsViolation",
    "deregister",
    "register",
]
