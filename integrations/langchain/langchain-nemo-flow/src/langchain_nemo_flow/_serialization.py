# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain request/response conversion helpers for NeMo Flow middleware."""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    convert_to_messages,
    convert_to_openai_messages,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

LANGCHAIN_MODEL_RESPONSE_KEY = "__langchain_nemo_flow_model_response"


def get_model_name(model: Any) -> str | None:
    """Best-effort extraction of a model name from a LangChain chat model."""
    for attr in ("model_name", "model", "model_id", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


# TODO: Remove is not used
def get_model_provider(model: Any) -> str:
    """Best-effort provider/name label for a LangChain chat model."""
    name = model.__class__.__name__
    if name.startswith("Chat") and len(name) > 4:
        return name[4:].lower()
    return name.lower()


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


def openai_chat_tool_to_json(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain tool descriptor into OpenAI Chat tool shape."""
    if isinstance(tool, dict) and tool.get("type") == "function":
        return tool
    return convert_to_openai_tool(tool)


def normalize_openai_tool_choice(tool_choice: Any) -> Any:
    """Convert LangChain tool choice aliases to OpenAI Chat-compatible values."""
    if isinstance(tool_choice, bool):
        return "required" if tool_choice else "none"
    if isinstance(tool_choice, str):
        if tool_choice == "any":
            return "required"
        if tool_choice not in {"auto", "none", "required"}:
            return {"type": "function", "function": {"name": tool_choice}}
    return tool_choice


def default_model_request_payload(request: ModelRequest[Any]) -> dict[str, Any]:
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


def openai_chat_model_request_payload(request: ModelRequest[Any]) -> dict[str, Any]:
    """Serialize public ``ModelRequest`` fields into OpenAI Chat payload shape."""
    messages: list[BaseMessage] = []
    if request.system_message is not None:
        messages.append(request.system_message)
    messages.extend(request.messages)

    payload: dict[str, Any] = {
        "messages": convert_to_openai_messages(messages),
    }
    if name := get_model_name(request.model):
        payload["model"] = name
    if request.model_settings:
        payload.update(request.model_settings)
    if request.tool_choice is not None:
        payload["tool_choice"] = normalize_openai_tool_choice(request.tool_choice)
    if request.tools:
        payload["tools"] = [openai_chat_tool_to_json(tool) for tool in request.tools]
    if request.response_format is not None:
        payload["response_format"] = request.response_format
    return payload


def default_payload_to_model_request(
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


def openai_chat_payload_to_model_request(
    original: ModelRequest[Any],
    payload: dict[str, Any],
) -> ModelRequest[Any]:
    """Apply OpenAI Chat request-intercept edits back to ``ModelRequest``."""
    overrides: dict[str, Any] = {}

    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        try:
            system_message, messages = split_system_message(convert_to_messages(raw_messages))
            overrides["system_message"] = system_message
            overrides["messages"] = messages
        except Exception:
            pass

    if "tools" in payload and isinstance(payload["tools"], list):
        overrides["tools"] = payload["tools"]
    if "tool_choice" in payload:
        overrides["tool_choice"] = payload["tool_choice"]
    if "response_format" in payload:
        overrides["response_format"] = payload["response_format"]

    model_settings = {
        key: value
        for key, value in payload.items()
        if key not in {"messages", "tools", "tool_choice", "response_format", LANGCHAIN_MODEL_RESPONSE_KEY}
    }
    if model_settings:
        overrides["model_settings"] = {**original.model_settings, **model_settings}

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


def best_effort_model_response_to_json(response: ModelResponse[Any], codec: Any) -> Any:
    """Serialize ``ModelResponse`` without losing Python-only fields."""
    return {
        LANGCHAIN_MODEL_RESPONSE_KEY: _model_response_payload(response, codec),
    }


def best_effort_model_response_from_json(payload: Any, codec: Any) -> ModelResponse[Any]:
    """Deserialize a ``ModelResponse`` serialized by ``best_effort_model_response_to_json``."""
    if isinstance(payload, dict) and LANGCHAIN_MODEL_RESPONSE_KEY in payload:
        decoded = _model_response_from_payload(payload[LANGCHAIN_MODEL_RESPONSE_KEY], codec)
        if decoded is not None:
            return decoded
    decoded = codec.from_json(payload)
    if isinstance(decoded, ModelResponse):
        return decoded
    raise TypeError(f"NeMo Flow model execution returned {type(decoded)!r}, expected ModelResponse")


def openai_chat_model_response_to_json(response: ModelResponse[Any], codec: Any) -> dict[str, Any]:
    """Serialize ``ModelResponse`` into OpenAI Chat response shape."""
    message = response.result[0] if response.result else AIMessage(content="")
    try:
        message_dict = convert_to_openai_messages([message])[0]
    except Exception:
        message_dict = {"role": "assistant", "content": str(message.content)}

    content = message_dict.get("content")
    if content is not None and not isinstance(content, str):
        message_dict["content"] = str(content)

    finish_reason = getattr(message, "response_metadata", {}).get("finish_reason")
    payload: dict[str, Any] = {
        "id": getattr(message, "id", None) or "langchain-nemo-flow",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message_dict,
                "finish_reason": finish_reason,
            }
        ],
        LANGCHAIN_MODEL_RESPONSE_KEY: _model_response_payload(response, codec),
    }
    if usage := getattr(message, "usage_metadata", None):
        payload["usage"] = usage
    return payload


def openai_chat_model_response_from_json(payload: Any, codec: Any) -> ModelResponse[Any]:
    """Deserialize OpenAI Chat response shape into ``ModelResponse``."""
    if isinstance(payload, dict) and LANGCHAIN_MODEL_RESPONSE_KEY in payload:
        decoded = _model_response_from_payload(payload[LANGCHAIN_MODEL_RESPONSE_KEY], codec)
        if decoded is not None:
            return decoded

    if not isinstance(payload, dict):
        raise TypeError(f"Expected OpenAI Chat response payload, received {type(payload)!r}")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI Chat response payload does not contain choices")

    message_dict = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    structured_response = None
    if "structured_response" in payload:
        structured_response = codec.from_json(payload["structured_response"])
    return ModelResponse(
        result=convert_to_messages([message_dict]),
        structured_response=structured_response,
    )


ModelRequestToPayload = Callable[[ModelRequest[Any]], dict[str, Any]]
ModelRequestHeaders = Callable[[ModelRequest[Any]], dict[str, str]]
PayloadToModelRequest = Callable[[ModelRequest[Any], dict[str, Any]], ModelRequest[Any]]
ModelResponseToJson = Callable[[ModelResponse[Any], Any], Any]
ModelResponseFromJson = Callable[[Any, Any], ModelResponse[Any]]
