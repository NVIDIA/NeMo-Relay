# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for the built-in NeMo Guardrails local backend."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, cast

from nemo_relay import Json, LLMRequest
from nemo_relay.codecs import (
    AnthropicMessagesCodec,
    LlmCodec,
    LlmResponseCodec,
    OpenAIChatCodec,
    OpenAIResponsesCodec,
)
from nemo_relay.plugin import PluginContext

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
    """Codec shape required by the local backend."""


class _GuardrailsRuntimeImports(NamedTuple):
    """Resolved Python symbols required by the local Guardrails backend."""

    rails_config_cls: Any
    llm_rails_cls: Any
    rail_type: Any
    rail_status: Any


_CODECS: dict[str, Callable[[], _GuardrailsCodec]] = {
    "openai_chat": OpenAIChatCodec,
    "openai_responses": OpenAIResponsesCodec,
    "anthropic_messages": AnthropicMessagesCodec,
}


def _load_nemoguardrails(module_name: str | None) -> _GuardrailsRuntimeImports:
    root_module = module_name or "nemoguardrails"
    try:
        guardrails = cast(Any, importlib.import_module(root_module))
        options = cast(Any, importlib.import_module(f"{root_module}.rails.llm.options"))
    except ImportError as error:
        if error.name == root_module:
            raise NeMoGuardrailsDependencyError(
                "NeMo Guardrails is required for the built-in NeMo Guardrails local backend. "
                "Install it with: pip install nemoguardrails"
            ) from error
        raise NeMoGuardrailsDependencyError(
            "NeMo Guardrails local backend could not import a required dependency: "
            f"{error.name or error}. Install the full NeMo Guardrails runtime dependencies."
        ) from error

    return _GuardrailsRuntimeImports(
        rails_config_cls=guardrails.RailsConfig,
        llm_rails_cls=guardrails.LLMRails,
        rail_type=options.RailType,
        rail_status=options.RailStatus,
    )


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status)).lower()


def _messages_from_annotated(annotated: Any) -> list[dict[str, Any]]:
    return [dict(message) for message in annotated.messages]


async def _apply_input_rails(
    rails: Any,
    rail_type: Any,
    rail_status: Any,
    codec: _GuardrailsCodec,
    request: LLMRequest,
) -> tuple[LLMRequest, list[dict[str, Any]]]:
    annotated_request = codec.decode(request)
    messages = _messages_from_annotated(annotated_request)
    input_result = await rails.check_async(messages, rail_types=[rail_type.INPUT])
    input_status = _status_value(input_result.status)
    if input_status == _status_value(rail_status.BLOCKED):
        _raise_blocked(input_result, "input")
    if input_status == _status_value(rail_status.MODIFIED):
        input_content = getattr(input_result, "content", "")
        annotated_request.messages = _replace_last_role_content(
            messages,
            "user",
            "" if input_content is None else str(input_content),
        )
        request = codec.encode(annotated_request, request)
        messages = _messages_from_annotated(annotated_request)
    return request, messages


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


def _raise_modified_output_not_supported(result: Any) -> None:
    output_content = getattr(result, "content", "")
    output_rail = getattr(result, "rail", None)
    raise NeMoGuardrailsViolation(
        "NeMo Guardrails output rail returned modified content, but the local backend "
        "does not rewrite provider responses yet.",
        rail_type="output",
        rail=None if output_rail is None else str(output_rail),
        content="" if output_content is None else str(output_content),
    )


async def _check_output_rails(
    rails: Any,
    rail_type: Any,
    rail_status: Any,
    messages: list[dict[str, Any]],
    response_text: str | None,
) -> None:
    if response_text is None:
        return

    output_messages = [*messages, {"role": "assistant", "content": response_text}]
    output_result = await rails.check_async(output_messages, rail_types=[rail_type.OUTPUT])
    output_status = _status_value(output_result.status)
    if output_status == _status_value(rail_status.BLOCKED):
        _raise_blocked(output_result, "output")
    if output_status == _status_value(rail_status.MODIFIED):
        _raise_modified_output_not_supported(output_result)


def _has_streaming_output_rails(rails: Any) -> bool:
    return bool(getattr(rails.config.rails.output, "flows", []))


def _output_streaming_config(rails: Any) -> Any | None:
    return getattr(rails.config.rails.output, "streaming", None)


def _guardrails_streaming_enabled(rails: Any) -> bool:
    streaming = _output_streaming_config(rails)
    return bool(streaming is not None and getattr(streaming, "enabled", False))


def _extract_stream_text(codec_name: str, chunk: Json) -> str | None:
    if not isinstance(chunk, dict):
        return None

    if codec_name == "openai_chat":
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return None
        parts: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
        return "".join(parts) if parts else None

    if codec_name == "openai_responses":
        if chunk.get("type") == "response.output_text.delta":
            delta = chunk.get("delta")
            return delta if isinstance(delta, str) and delta else None
        return None

    if codec_name == "anthropic_messages":
        if chunk.get("type") != "content_block_delta":
            return None
        delta = chunk.get("delta")
        if not isinstance(delta, dict):
            return None
        if delta.get("type") != "text_delta":
            return None
        text = delta.get("text")
        return text if isinstance(text, str) and text else None

    return None


def _guardrails_stream_error_message(chunk: str) -> str | None:
    try:
        payload = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    if error.get("type") != "guardrails_violation":
        return None
    message = error.get("message")
    return message if isinstance(message, str) and message else "Blocked by output rails."


async def _queue_string_stream(queue: "asyncio.Queue[str | None]"):
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def _monitor_streaming_output_rails(
    *,
    rails: Any,
    messages: list[dict[str, Any]],
    text_queue: "asyncio.Queue[str | None]",
    blocked: dict[str, str | None],
) -> None:
    guarded_stream = rails.stream_async(
        messages=messages,
        generator=_queue_string_stream(text_queue),
        include_metadata=False,
    )
    async for chunk in guarded_stream:
        if isinstance(chunk, str):
            message = _guardrails_stream_error_message(chunk)
            if message is not None:
                blocked["message"] = message
                return


def _raise_streaming_output_blocked(blocked_message: str) -> None:
    raise NeMoGuardrailsViolation(
        f"NeMo Guardrails output rail blocked the LLM call: {blocked_message}",
        rail_type="output",
        content=blocked_message,
    )


def _build_guardrails_config(config: dict[str, Any], rails_config_cls: Any) -> Any:
    if config.get("config_path") is not None:
        return rails_config_cls.from_path(cast(str, config["config_path"]))
    return rails_config_cls.from_content(
        colang_content=cast(str | None, config.get("colang_content")),
        yaml_content=cast(str, config["config_yaml"]),
    )


def _resolve_codec(config: dict[str, Any]) -> tuple[str, _GuardrailsCodec]:
    codec_name = cast(str | None, config.get("codec"))
    if codec_name is None or codec_name not in _CODECS:
        raise RuntimeError("local NeMo Guardrails backend requires a supported codec")
    return codec_name, _CODECS[codec_name]()


async def _check_tool_input(
    rails: Any,
    rail_type: Any,
    rail_status: Any,
    tool_name: str,
    args: Json,
) -> Json:
    input_result = await rails.check_async(
        [{"role": "user", "content": _tool_input_content(tool_name, args)}],
        rail_types=[rail_type.INPUT],
    )
    input_status = _status_value(input_result.status)
    if input_status == _status_value(rail_status.BLOCKED):
        _raise_blocked(input_result, "tool_input")
    if input_status == _status_value(rail_status.MODIFIED):
        input_content = getattr(input_result, "content", "")
        return _modified_tool_payload(
            "" if input_content is None else str(input_content),
            "arguments",
        )
    return args


async def _check_tool_output(
    rails: Any,
    rail_type: Any,
    rail_status: Any,
    tool_name: str,
    args: Json,
    result: Json,
) -> Json:
    output_result = await rails.check_async(
        [
            {"role": "user", "content": _tool_input_content(tool_name, args)},
            {
                "role": "assistant",
                "content": _tool_output_content(tool_name, args, result),
            },
        ],
        rail_types=[rail_type.OUTPUT],
    )
    output_status = _status_value(output_result.status)
    if output_status == _status_value(rail_status.BLOCKED):
        _raise_blocked(output_result, "tool_output")
    if output_status == _status_value(rail_status.MODIFIED):
        output_content = getattr(output_result, "content", "")
        return _modified_tool_payload(
            "" if output_content is None else str(output_content),
            "result",
        )
    return result


def _make_llm_intercept(
    *,
    rails: Any,
    rail_type: Any,
    rail_status: Any,
    codec: _GuardrailsCodec,
    enable_input: bool,
    enable_output: bool,
):
    async def intercept(_name: str, request: LLMRequest, next_call):
        current_request = request
        messages = _messages_from_annotated(codec.decode(current_request))

        if enable_input:
            current_request, messages = await _apply_input_rails(
                rails,
                rail_type,
                rail_status,
                codec,
                current_request,
            )

        response = await next_call(current_request)
        if not enable_output:
            return response

        annotated_response = codec.decode_response(response)
        await _check_output_rails(
            rails,
            rail_type,
            rail_status,
            messages,
            annotated_response.response_text(),
        )
        return response

    return intercept


def _make_llm_stream_intercept(
    *,
    rails: Any,
    rail_type: Any,
    rail_status: Any,
    codec_name: str,
    codec: _GuardrailsCodec,
    enable_input: bool,
    enable_output: bool,
):
    async def stream_intercept(request: LLMRequest, next_call):
        current_request = request
        messages = _messages_from_annotated(codec.decode(current_request))
        if enable_input:
            current_request, messages = await _apply_input_rails(
                rails,
                rail_type,
                rail_status,
                codec,
                current_request,
            )

        stream = await next_call(current_request)
        if not enable_output:
            return stream
        if not _has_streaming_output_rails(rails):
            return stream
        if not _guardrails_streaming_enabled(rails):
            raise RuntimeError(
                "local NeMo Guardrails streaming output rails require "
                "rails.output.streaming.enabled = true in the Guardrails config."
            )

        streaming_config = _output_streaming_config(rails)
        if streaming_config is None or not getattr(streaming_config, "stream_first", True):
            raise RuntimeError(
                "local NeMo Guardrails streaming output rails currently require "
                "rails.output.streaming.stream_first = true."
            )

        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        block_state: dict[str, str | None] = {"message": None}

        async def guarded_provider_stream():
            monitor = asyncio.create_task(
                _monitor_streaming_output_rails(
                    rails=rails,
                    messages=messages,
                    text_queue=text_queue,
                    blocked=block_state,
                )
            )
            try:
                async for chunk in stream:
                    if block_state["message"] is not None:
                        _raise_streaming_output_blocked(block_state["message"])

                    text = _extract_stream_text(codec_name, chunk)
                    if text is not None:
                        await text_queue.put(text)

                    yield chunk

                    if block_state["message"] is not None:
                        _raise_streaming_output_blocked(block_state["message"])
            finally:
                await text_queue.put(None)
                await monitor
                if block_state["message"] is not None:
                    _raise_streaming_output_blocked(block_state["message"])

        return guarded_provider_stream()

    return stream_intercept


def _make_tool_intercept(
    *,
    rails: Any,
    rail_type: Any,
    rail_status: Any,
    enable_tool_input: bool,
    enable_tool_output: bool,
):
    async def tool_intercept(tool_name: str, args: Json, next_call):
        current_args = args

        if enable_tool_input:
            current_args = await _check_tool_input(
                rails,
                rail_type,
                rail_status,
                tool_name,
                current_args,
            )

        tool_result = await next_call(current_args)
        if not enable_tool_output:
            return tool_result

        return await _check_tool_output(
            rails,
            rail_type,
            rail_status,
            tool_name,
            current_args,
            tool_result,
        )

    return tool_intercept


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


def register_local_backend(config: dict[str, Any], context: PluginContext) -> None:
    """Install the built-in NeMo Guardrails local backend."""

    local = cast(dict[str, Any], config.get("local") or {})
    module_name = cast(str | None, local.get("python_module"))
    runtime_imports = _load_nemoguardrails(module_name)
    guardrails_config = _build_guardrails_config(config, runtime_imports.rails_config_cls)
    rails = runtime_imports.llm_rails_cls(guardrails_config)
    enable_input = bool(config.get("input", True))
    enable_output = bool(config.get("output", True))
    enable_tool_input = bool(config.get("tool_input", False))
    enable_tool_output = bool(config.get("tool_output", False))
    priority = int(config.get("priority", _DEFAULT_PRIORITY))

    if enable_input or enable_output:
        codec_name, codec = _resolve_codec(config)
        intercept = _make_llm_intercept(
            rails=rails,
            rail_type=runtime_imports.rail_type,
            rail_status=runtime_imports.rail_status,
            codec=codec,
            enable_input=enable_input,
            enable_output=enable_output,
        )
        stream_intercept = _make_llm_stream_intercept(
            rails=rails,
            rail_type=runtime_imports.rail_type,
            rail_status=runtime_imports.rail_status,
            codec_name=codec_name,
            codec=codec,
            enable_input=enable_input,
            enable_output=enable_output,
        )
        context.register_llm_execution_intercept("nemo_guardrails_local", priority, intercept)
        context.register_llm_stream_execution_intercept(
            "nemo_guardrails_local_stream",
            priority,
            stream_intercept,
        )

    if enable_tool_input or enable_tool_output:
        tool_intercept = _make_tool_intercept(
            rails=rails,
            rail_type=runtime_imports.rail_type,
            rail_status=runtime_imports.rail_status,
            enable_tool_input=enable_tool_input,
            enable_tool_output=enable_tool_output,
        )
        context.register_tool_execution_intercept("nemo_guardrails_local", priority, tool_intercept)
