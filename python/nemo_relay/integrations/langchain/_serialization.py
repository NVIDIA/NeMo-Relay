# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain request/response conversion helpers for NeMo Relay middleware."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langgraph.types import Command, Send

from nemo_relay import AnnotatedLLMRequest, LLMRequest
from nemo_relay.codecs import LlmCodec

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest

LANGCHAIN_MODEL_RESPONSE_KEY = "__nemo_relay_integrations_langchain_model_response"
_LANGCHAIN_MODELED_REQUEST_KEYS = {"messages", "model", "tool_choice", "tools"}
_LC_TO_RELAY_MESSAGE_ROLE = {
    "human": "user",
    "ai": "assistant",
}


def get_model_name(model: Any) -> str | None:
    """Best-effort extraction of a model name from a LangChain chat model."""
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


class LangChainCodec(LlmCodec):
    """Translate LangChain ``ModelRequest`` payloads for request intercepts."""

    @classmethod
    def _langchain_tool_calls_to_annotated(cls, tool_calls: list[Any]) -> list[dict[str, Any]]:
        annotated_tool_calls = []
        for tool_call in tool_calls:
            args = tool_call["args"]
            arguments = args if isinstance(args, str) else json.dumps(args)
            annotated_tool_calls.append(
                {
                    "id": tool_call.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": arguments,
                    },
                }
            )

        return annotated_tool_calls

    @classmethod
    def _annotated_tool_calls_to_langchain(cls, tool_calls: Any) -> list[dict[str, Any]] | None:
        if not isinstance(tool_calls, list) or not tool_calls:
            return None

        langchain_tool_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict):
                name = str(function.get("name") or "")
                arguments = function.get("arguments", {})
            else:
                name = str(tool_call.get("name") or "")
                arguments = tool_call.get("args", {})

            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    args = {"arguments": arguments}
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}

            langchain_tool_calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": str(tool_call.get("id") or ""),
                    "type": "tool_call",
                }
            )

        return langchain_tool_calls or None

    @classmethod
    def _langchain_message_to_annotated(cls, message: BaseMessage) -> list[dict[str, Any]]:
        content = message.content
        if content is None:
            content = []
        elif isinstance(content, str):
            content = [content]

        name = message.name
        role = _LC_TO_RELAY_MESSAGE_ROLE.get(message.type, message.type)

        messages = []
        for msg in content:
            relay_message: dict[str, Any] = {"role": role}
            if isinstance(msg, str):
                relay_message["content"] = msg
            elif isinstance(msg, dict):
                relay_message.update(msg)
                if "content" not in relay_message:
                    relay_message["content"] = relay_message.pop("text", "")
            else:
                raise ValueError(f"Unsupported LangChain message content type: {type(content)}")

            if name is not None:
                relay_message["name"] = name

            # Using getattr as we are inferring subclasses of BaseMessage based upon the role
            if role == "assistant":
                tool_calls = getattr(message, "tool_calls", [])
                relay_message["tool_calls"] = cls._langchain_tool_calls_to_annotated(tool_calls)
            elif role == "tool":
                relay_message["tool_call_id"] = getattr(message, "tool_call_id", "")

            messages.append(relay_message)

        return messages

    @classmethod
    def _annotated_message_to_langchain(cls, message: dict[str, Any]) -> BaseMessage:
        role = message.get("role")
        content = message.get("content", "")
        name = message.get("name")

        if role == "system":
            return SystemMessage(content=content, name=name)
        if role == "user":
            return HumanMessage(content=content, name=name)
        if role == "assistant":
            tool_calls = cls._annotated_tool_calls_to_langchain(message.get("tool_calls"))
            return AIMessage(content=content, name=name, tool_calls=tool_calls or [])
        if role == "tool":
            return ToolMessage(content=content, name=name, tool_call_id=str(message.get("tool_call_id") or ""))
        raise ValueError(f"Unsupported annotated LangChain message role: {role!r}")

    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode a LangChain-shaped request payload into an annotated request."""
        payload = request.content
        raw_messages = payload.get("messages", [])
        messages: list[dict[str, Any]] = []
        if isinstance(raw_messages, list):
            for message in messages_from_dict(raw_messages):
                messages.extend(self._langchain_message_to_annotated(message))

        model = payload.get("model")
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")
        extra = {key: value for key, value in payload.items() if key not in _LANGCHAIN_MODELED_REQUEST_KEYS}

        return AnnotatedLLMRequest(
            messages,
            model=model if isinstance(model, str) else None,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice if isinstance(tool_choice, str | dict) else None,
            extra=extra or None,
        )

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode annotated request edits back into a LangChain-shaped payload."""
        payload = dict(original.content)
        payload.update(annotated.extra)
        payload["messages"] = messages_to_dict(
            [self._annotated_message_to_langchain(message) for message in annotated.messages]
        )
        if annotated.model is not None:
            payload["model"] = annotated.model
        if annotated.tools is not None:
            payload["tools"] = annotated.tools
        if annotated.tool_choice is not None:
            payload["tool_choice"] = annotated.tool_choice
        return LLMRequest(dict(original.headers), payload)


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
    llm_request: LLMRequest,
) -> ModelRequest[Any]:
    """Apply supported NeMo Relay request-intercept edits back to ``ModelRequest``."""
    overrides: dict[str, Any] = {}

    raw_messages = llm_request.content.get("messages")
    if isinstance(raw_messages, list) and len(raw_messages) > 0:
        try:
            system_message, messages = split_system_message(messages_from_dict(raw_messages))
            overrides["system_message"] = system_message
            overrides["messages"] = messages
        except Exception:
            pass

    model_settings = llm_request.content.get("model_settings")
    if isinstance(model_settings, dict):
        # Using dict() to ensure we have a copy
        model_settings_copy = dict(model_settings)
        extra_headers = model_settings_copy.get("extra_headers")
        if not isinstance(extra_headers, dict):
            extra_headers = {}
        overrides["model_settings"] = model_settings_copy
    else:
        overrides["model_settings"] = {}
        extra_headers = {}

    if len(llm_request.headers) > 0:
        extra_headers.update(llm_request.headers)
        overrides["model_settings"]["extra_headers"] = extra_headers

    if "tool_choice" in llm_request.content:
        overrides["tool_choice"] = llm_request.content["tool_choice"]

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


def _prepare_lc_payloads(payload: Any) -> Any:
    """
    Convert a LangChain payload to a JSON-serializable structure

    Typically the entry point to this method is a LangChain dictionary containing LC message objects, and the returned
    dictionary should contain the same structure, but the values are JSON serializable representations
    """
    if isinstance(payload, dict):
        prepared = {}
        for key, value in payload.items():
            prepared[key] = _prepare_lc_payloads(value)
    elif isinstance(payload, list | tuple):
        prepared = []
        for value in payload:
            prepared.append(_prepare_lc_payloads(value))
    elif isinstance(payload, Command):
        prepared = {
            "type": "command",
            "command": {
                "graph": _prepare_lc_payloads(payload.graph),
                "update": _prepare_lc_payloads(payload.update),
                "resume": _prepare_lc_payloads(payload.resume),
                "goto": _prepare_lc_payloads(payload.goto),
            },
        }
    elif isinstance(payload, Send):
        prepared = {
            "type": "send",
            "send": {
                "node": payload.node,
                "arg": _prepare_lc_payloads(payload.arg),
            },
        }
    elif isinstance(payload, ToolMessage):
        prepared = {
            "type": "tool_message",
            "tool_call": {
                "name": payload.name,
                "id": payload.id,
                "tool_call_id": payload.tool_call_id,
                "content": payload.content,
            },
        }
    elif isinstance(payload, BaseMessage):
        prepared = {
            "type": "message",
            "message": messages_to_dict([payload]),
        }
    else:
        prepared = payload

    return prepared
