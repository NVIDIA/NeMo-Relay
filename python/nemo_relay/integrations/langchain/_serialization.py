# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain request/response conversion helpers for NeMo Relay middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langgraph.types import Command, Send

from nemo_relay.codecs import AnthropicMessagesCodec, LlmCodec, OpenAIChatCodec, OpenAIResponsesCodec

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest


# In order to infer codec support from LangChain chat model types, we need to import them here.
# However these may not be installed in the user's environment.
_HAS_ANTHROPIC = False
_HAS_OPENAI = False
_HAS_NVIDIA = False
try:
    from langchain_anthropic import ChatAnthropic

    _HAS_ANTHROPIC = True
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI

    _HAS_OPENAI = True
except ImportError:
    pass

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _HAS_NVIDIA = True
except ImportError:
    pass

LANGCHAIN_MODEL_RESPONSE_KEY = "__nemo_relay_integrations_langchain_model_response"


def get_model_name(model: Any) -> str | None:
    """Best-effort extraction of a model name from a LangChain chat model."""
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def infer_codec_from_model(model: Any) -> LlmCodec | None:
    """Infer a NeMo Relay codec name from a LangChain chat model."""
    if _HAS_ANTHROPIC:
        if isinstance(model, ChatAnthropic):
            return AnthropicMessagesCodec()

    if _HAS_NVIDIA:
        if isinstance(model, ChatNVIDIA):
            return OpenAIChatCodec()

    if _HAS_OPENAI:
        if isinstance(model, ChatOpenAI):
            if getattr(model, "use_responses_api", None) is True:
                return OpenAIResponsesCodec()

            return OpenAIChatCodec()

    return None


def split_system_message(messages: list[BaseMessage]) -> tuple[SystemMessage | None, list[BaseMessage]]:
    """Split a leading system message into LangChain agent ``ModelRequest`` shape."""
    if messages and isinstance(messages[0], SystemMessage):
        return messages[0], messages[1:]
    return None, messages


def model_request_to_payload(model_name: str | None, request: ModelRequest[Any]) -> dict[str, Any]:
    """Serialize public ``ModelRequest`` fields into a JSON-compatible payload."""
    messages: list[BaseMessage] = []
    if request.system_message is not None:
        messages.append(request.system_message)
    messages.extend(request.messages)

    payload: dict[str, Any] = {
        "messages": messages_to_dict(messages),
    }
    if model_name:
        payload["model"] = model_name
    if request.model_settings:
        payload["model_settings"] = request.model_settings
    if request.response_format is not None:
        payload["response_format"] = repr(request.response_format)
    return payload


def payload_to_model_request(
    original: ModelRequest[Any],
    payload: dict[str, Any],
) -> ModelRequest[Any]:
    """Apply supported NeMo Relay request-intercept edits back to ``ModelRequest``."""
    overrides: dict[str, Any] = {}

    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list) and len(raw_messages) > 0:
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
    raise TypeError(f"NeMo Relay model execution returned {type(decoded)!r}, expected ModelResponse")


def _prepare_outputs(outputs: Any) -> Any:
    """Prepare a NeMo Relay scope output dict for returning to LangChain."""
    if isinstance(outputs, dict):
        prepared_outputs = {}
        for key, value in outputs.items():
            prepared_outputs[key] = _prepare_outputs(value)
    elif isinstance(outputs, list | tuple):
        prepared_outputs = []
        for value in outputs:
            prepared_outputs.append(_prepare_outputs(value))
    elif isinstance(outputs, Command):
        prepared_outputs = {
            "type": "command",
            "command": {
                "graph": _prepare_outputs(outputs.graph),
                "update": _prepare_outputs(outputs.update),
                "resume": _prepare_outputs(outputs.resume),
                "goto": _prepare_outputs(outputs.goto),
            },
        }
    elif isinstance(outputs, Send):
        prepared_outputs = {
            "type": "send",
            "send": {
                "node": outputs.node,
                "arg": _prepare_outputs(outputs.arg),
            },
        }
    elif isinstance(outputs, ToolMessage):
        prepared_outputs = {
            "type": "tool_message",
            "tool_call": {
                "name": outputs.name,
                "id": outputs.id,
                "tool_call_id": outputs.tool_call_id,
                "content": outputs.content,
            },
        }
    elif isinstance(outputs, BaseMessage):
        prepared_outputs = {
            "type": "message",
            "message": messages_to_dict([outputs]),
        }
    else:
        prepared_outputs = outputs

    return prepared_outputs
