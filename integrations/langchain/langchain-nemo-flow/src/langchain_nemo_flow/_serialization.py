# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain request/response conversion helpers for NeMo Flow middleware."""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from langchain_core.tools import BaseTool

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


from nemo_flow.codecs import AnthropicMessagesCodec, LlmCodec, OpenAIChatCodec, OpenAIResponsesCodec

LANGCHAIN_MODEL_RESPONSE_KEY = "__langchain_nemo_flow_model_response"


def get_model_name(model: Any) -> str | None:
    """Best-effort extraction of a model name from a LangChain chat model."""
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def infer_codec_from_model(model: Any) -> LlmCodec | None:
    """Infer a NeMo Flow codec name from a LangChain chat model."""
    if ChatAnthropic is not None and isinstance(model, ChatAnthropic):
        return AnthropicMessagesCodec()

    if ChatOpenAI is not None and isinstance(model, ChatOpenAI):
        if getattr(model, "use_responses_api", None) is True:
            return OpenAIResponsesCodec()
        return OpenAIChatCodec()

    return None


def split_system_message(messages: list[BaseMessage]) -> tuple[SystemMessage | None, list[BaseMessage]]:
    """Split a leading system message into LangChain agent ``ModelRequest`` shape."""
    if messages and isinstance(messages[0], SystemMessage):
        return messages[0], messages[1:]
    return None, messages


def tool_to_json(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain tool descriptor into a JSON-compatible summary."""
    if isinstance(tool, dict):
        return tool

    schema: dict[str, Any] | None = None
    try:
        schema = tool.get_input_schema().model_json_schema()
    except Exception:
        schema = None

    payload: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
    }
    if schema is not None:
        payload["schema"] = schema
    return payload


def model_request_to_payload(request: ModelRequest[Any]) -> dict[str, Any]:
    """Serialize public ``ModelRequest`` fields into a JSON-compatible payload."""
    messages: list[BaseMessage] = []
    if request.system_message is not None:
        messages.append(request.system_message)
    messages.extend(request.messages)

    payload: dict[str, Any] = {
        "messages": messages_to_dict(messages),
    }
    if name := get_model_name(request.model):
        payload["model"] = name
    if request.model_settings:
        payload["model_settings"] = request.model_settings
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice
    if request.tools:
        payload["tools"] = [tool_to_json(tool) for tool in request.tools]
    if request.response_format is not None:
        payload["response_format"] = repr(request.response_format)
    return payload


def payload_to_model_request(
    original: ModelRequest[Any],
    payload: dict[str, Any],
) -> ModelRequest[Any]:
    """Apply supported NeMo Flow request-intercept edits back to ``ModelRequest``."""
    overrides: dict[str, Any] = {}

    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        try:
            system_message, messages = split_system_message(messages_from_dict(raw_messages))
            overrides["system_message"] = system_message
            overrides["messages"] = messages
        except Exception:
            pass

    model_settings = payload.get("model_settings")
    if isinstance(model_settings, dict):
        overrides["model_settings"] = model_settings

    if "tool_choice" in payload:
        overrides["tool_choice"] = payload["tool_choice"]

    return original.override(**overrides) if overrides else original


def _model_response_payload(response: ModelResponse[Any], codec: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages_to_dict(response.result),
    }
    if response.structured_response is not None:
        payload["structured_response"] = codec.to_json(response.structured_response)
    return payload


def _model_response_from_payload(payload: Any, codec: Any) -> ModelResponse[Any] | None:
    if not isinstance(payload, dict):
        return None

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return None

    structured_response = None
    if "structured_response" in payload:
        structured_response = codec.from_json(payload["structured_response"])
    return ModelResponse(
        result=messages_from_dict(raw_messages),
        structured_response=structured_response,
    )


def model_response_to_json(response: ModelResponse[Any], codec: Any) -> Any:
    """Serialize ``ModelResponse`` without losing Python-only fields."""
    return {
        LANGCHAIN_MODEL_RESPONSE_KEY: _model_response_payload(response, codec),
    }


def model_response_from_json(payload: Any, codec: Any) -> ModelResponse[Any]:
    """Deserialize a ``ModelResponse`` serialized by ``best_effort_model_response_to_json``."""
    if isinstance(payload, dict) and LANGCHAIN_MODEL_RESPONSE_KEY in payload:
        decoded = _model_response_from_payload(payload[LANGCHAIN_MODEL_RESPONSE_KEY], codec)
        if decoded is not None:
            return decoded
    decoded = codec.from_json(payload)
    if isinstance(decoded, ModelResponse):
        return decoded
    raise TypeError(f"NeMo Flow model execution returned {type(decoded)!r}, expected ModelResponse")


ModelRequestHeaders = Callable[[ModelRequest[Any]], dict[str, str]]
