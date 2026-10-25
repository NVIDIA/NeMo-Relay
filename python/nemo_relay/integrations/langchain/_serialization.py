# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain request/response conversion helpers for NeMo Relay middleware."""

from __future__ import annotations

import copy
import functools
import importlib
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
from langchain_core.tools import BaseTool
from langgraph.types import Command, Send
from pydantic import BaseModel

from nemo_relay import AnnotatedLLMRequest, AnnotatedLLMResponse, LLMRequest
from nemo_relay.codecs import LlmCodec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware import ModelRequest
    from typing_extensions import TypeIs

    from nemo_relay import Json, JsonObject

LANGCHAIN_MODEL_RESPONSE_KEY = "__nemo_relay_integrations_langchain_model_response"
_LANGCHAIN_MODELED_REQUEST_KEYS = {"messages", "model", "tool_choice", "tools"}
_LANGCHAIN_PROVIDER = "langchain"
_LC_TO_RELAY_MESSAGE_ROLE = {
    "human": "user",
    "ai": "assistant",
}
_FINISH_REASON_MAP = {
    "stop": "complete",
    "end_turn": "complete",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "max_tokens": "length",
    "length": "length",
    "content_filter": "content_filter",
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
    def _annotated_tool_calls_to_provider(cls, tool_calls: Any) -> list[dict[str, Any]] | None:
        """Return the OpenAI-compatible representation required by some providers."""
        langchain_tool_calls = cls._annotated_tool_calls_to_langchain(tool_calls)
        if langchain_tool_calls is None:
            return None

        return [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["args"], separators=(",", ":")),
                },
            }
            for tool_call in langchain_tool_calls
        ]

    @classmethod
    def _langchain_content_to_annotated(cls, content: Any) -> str | list[dict[str, Any]]:
        """Preserve one LangChain message while exposing its portable text blocks."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise ValueError(f"Unsupported LangChain message content type: {type(content)}")

        content_parts = []
        for content_part in content:
            if isinstance(content_part, str):
                content_parts.append({"type": "text", "text": content_part})
            elif (
                isinstance(content_part, dict)
                and content_part.get("type") == "text"
                and isinstance(content_part.get("text"), str)
            ):
                content_parts.append(dict(content_part))
            elif isinstance(content_part, dict):
                content_parts.append(
                    {
                        "type": "provider_native",
                        "provider": _LANGCHAIN_PROVIDER,
                        "kind": str(content_part.get("type") or "content_block"),
                        "value": dict(content_part),
                    }
                )
            else:
                raise ValueError(f"Unsupported LangChain message content part type: {type(content_part)}")
        return content_parts

    @staticmethod
    def _annotated_content_to_langchain(content: Any) -> Any:
        """Restore LangChain-native blocks after annotated request interception."""
        if not isinstance(content, list):
            return content

        content_parts = []
        for content_part in content:
            if (
                isinstance(content_part, dict)
                and content_part.get("type") == "provider_native"
                and content_part.get("provider") == _LANGCHAIN_PROVIDER
            ):
                value = content_part.get("value")
                if not isinstance(value, str | dict):
                    raise ValueError("LangChain provider-native content must contain a string or object value")
                content_parts.append(value)
            else:
                content_parts.append(content_part)
        return content_parts

    @classmethod
    def _langchain_message_to_annotated(cls, message: BaseMessage) -> dict[str, Any]:
        content = message.content
        name = message.name
        role = _LC_TO_RELAY_MESSAGE_ROLE.get(message.type, message.type)

        relay_message: dict[str, Any] = {
            "role": role,
            "content": cls._langchain_content_to_annotated(content),
        }
        if name is not None:
            relay_message["name"] = name

        # Using getattr as we are inferring subclasses of BaseMessage based upon the role
        if role == "assistant":
            relay_message["tool_calls"] = cls._langchain_tool_calls_to_annotated(getattr(message, "tool_calls", []))
        elif role == "tool":
            relay_message["tool_call_id"] = getattr(message, "tool_call_id", "")

        return relay_message

    @classmethod
    def _annotated_message_to_langchain(
        cls,
        message: dict[str, Any],
        provider_tool_calls: list[dict[str, Any]] | None = None,
        original_message: BaseMessage | None = None,
    ) -> BaseMessage:
        role = message.get("role")
        content = (
            original_message.content
            if original_message is not None
            else cls._annotated_content_to_langchain(message.get("content", ""))
        )
        name = message.get("name")

        if role == "system":
            return SystemMessage(content=content, name=name)
        if role == "user":
            return HumanMessage(content=content, name=name)
        if role == "assistant":
            tool_calls = cls._annotated_tool_calls_to_langchain(message.get("tool_calls"))
            additional_kwargs = {"tool_calls": provider_tool_calls} if provider_tool_calls is not None else {}
            return AIMessage(
                content=content, name=name, tool_calls=tool_calls or [], additional_kwargs=additional_kwargs
            )
        if role == "tool":
            return ToolMessage(content=content, name=name, tool_call_id=str(message.get("tool_call_id") or ""))
        raise ValueError(f"Unsupported annotated LangChain message role: {role!r}")

    @classmethod
    def _original_messages(cls, original: LLMRequest) -> list[tuple[dict[str, Any], BaseMessage]]:
        """Return original LangChain messages with their annotated representations."""
        raw_messages = original.content.get("messages")
        if not isinstance(raw_messages, list):
            return []

        return [
            (cls._langchain_message_to_annotated(message), message) for message in _langchain_messages(raw_messages)
        ]

    @staticmethod
    def _message_content_matches(message: dict[str, Any], original_message: dict[str, Any]) -> bool:
        """Return whether two messages have the same role, content, and tool result identity."""
        return all(message.get(key) == original_message.get(key) for key in ("role", "content", "tool_call_id"))

    @classmethod
    def _original_messages_for_annotated(
        cls,
        messages: list[dict[str, Any]],
        original_messages: list[tuple[dict[str, Any], BaseMessage]],
    ) -> list[BaseMessage | None]:
        """Match unchanged message content to its original LangChain representation."""
        matches: list[BaseMessage | None] = [None] * len(messages)
        matched_original_indexes: set[int] = set()

        if len(messages) == len(original_messages):
            for index, message in enumerate(messages):
                original_annotated_message, original_langchain_message = original_messages[index]
                if cls._message_content_matches(message, original_annotated_message):
                    matches[index] = original_langchain_message
                    matched_original_indexes.add(index)

        for message_index, message in enumerate(messages):
            if matches[message_index] is not None:
                continue
            candidate_indexes = [
                original_index
                for original_index, (original_annotated_message, _) in enumerate(original_messages)
                if original_index not in matched_original_indexes
                and cls._message_content_matches(message, original_annotated_message)
            ]
            if len(candidate_indexes) != 1:
                continue
            original_index = candidate_indexes[0]
            matching_message_count = sum(
                1
                for candidate_index, candidate_message in enumerate(messages)
                if matches[candidate_index] is None
                and cls._message_content_matches(candidate_message, original_messages[original_index][0])
            )
            if matching_message_count != 1:
                continue
            matches[message_index] = original_messages[original_index][1]
            matched_original_indexes.add(original_index)

        return matches

    @classmethod
    def _original_provider_tool_calls(cls, original: LLMRequest) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Return assistant messages associated with their provider tool calls."""
        raw_messages = original.content.get("messages")
        if not isinstance(raw_messages, list):
            return []

        provider_tool_calls: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for message in _langchain_messages(raw_messages):
            raw_tool_calls = None
            if isinstance(message, AIMessage):
                candidate = message.additional_kwargs.get("tool_calls")
                if isinstance(candidate, list):
                    raw_tool_calls = candidate
            if raw_tool_calls is not None:
                annotated_message = cls._langchain_message_to_annotated(message)
                provider_tool_calls.append((annotated_message, raw_tool_calls))
        return provider_tool_calls

    def _provider_tool_calls_for_message(
        self,
        message: dict[str, Any],
        original_provider_tool_calls: list[tuple[dict[str, Any], list[dict[str, Any]]]],
        matched_original_messages: set[int],
    ) -> list[dict[str, Any]] | None:
        """Preserve provider tool-call encoding when an assistant message still matches."""
        for index, (original_message, original_tool_calls) in enumerate(original_provider_tool_calls):
            if index not in matched_original_messages and message == original_message:
                matched_original_messages.add(index)
                return original_tool_calls

        message_without_tool_calls = {key: value for key, value in message.items() if key != "tool_calls"}
        for index, (original_message, original_tool_calls) in enumerate(original_provider_tool_calls):
            original_without_tool_calls = {key: value for key, value in original_message.items() if key != "tool_calls"}
            if index in matched_original_messages or message_without_tool_calls != original_without_tool_calls:
                continue
            matched_original_messages.add(index)
            if message.get("tool_calls") == original_message.get("tool_calls"):
                return original_tool_calls
            return self._annotated_tool_calls_to_provider(message.get("tool_calls"))

        return self._annotated_tool_calls_to_provider(message.get("tool_calls"))

    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode a LangChain-shaped request payload into an annotated request."""
        payload = request.content
        raw_messages = payload.get("messages", [])
        messages: list[dict[str, Any]] = []
        if isinstance(raw_messages, list):
            for message in _langchain_messages(raw_messages):
                messages.append(self._langchain_message_to_annotated(message))

        model = payload.get("model")
        tools = payload.get("tools")
        if isinstance(tools, list) and not _are_objects(tools):
            raise ValueError("LangChain request tools must be objects")
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
        messages = annotated.messages
        original_messages = self._original_messages(original)
        original_messages_by_annotated_index = self._original_messages_for_annotated(messages, original_messages)
        original_provider_tool_calls = self._original_provider_tool_calls(original)
        matched_original_messages: set[int] = set()
        encoded_messages = []
        for index, message in enumerate(messages):
            original_message = original_messages_by_annotated_index[index]
            provider_tool_calls = None
            if message.get("role") == "assistant":
                provider_tool_calls = self._provider_tool_calls_for_message(
                    message, original_provider_tool_calls, matched_original_messages
                )
            encoded_messages.append(
                self._annotated_message_to_langchain(message, provider_tool_calls, original_message)
            )
        payload["messages"] = [*messages_to_dict(encoded_messages)]
        if annotated.model is not None:
            payload["model"] = annotated.model
        if annotated.tools is not None:
            payload["tools"] = [*annotated.tools]
        if annotated.tool_choice is not None:
            payload["tool_choice"] = annotated.tool_choice
        return LLMRequest(dict(original.headers), payload)

    def decode_response(self, response: Any) -> AnnotatedLLMResponse:
        """Decode a serialized LangChain ``ModelResponse`` for observability."""
        payload = _model_response_payload_from_json(response)
        return _model_response_to_annotated(payload)


def split_system_message(messages: list[BaseMessage]) -> tuple[SystemMessage | None, list[BaseMessage]]:
    """Split a leading system message into LangChain agent ``ModelRequest`` shape."""
    if messages and isinstance(messages[0], SystemMessage):
        return messages[0], messages[1:]
    return None, messages


def _native_tool_schema(definition: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "provider_native",
        "provider": _LANGCHAIN_PROVIDER,
        "kind": "tool",
        "value": copy.deepcopy(definition),
    }


def _are_objects(values: Sequence[Json]) -> TypeIs[Sequence[JsonObject]]:
    """Return whether every entry in `values` is a JSON object."""
    return all(isinstance(value, dict) for value in values)


def _langchain_messages(raw_messages: Sequence[Json]) -> list[BaseMessage]:
    """Return LangChain messages from their dict form. Raise `TypeError` when an entry is not an object."""
    if not _are_objects(raw_messages):
        raise TypeError("LangChain messages must be objects")
    return messages_from_dict(raw_messages)


def _model_tool_schemas(tools: list[BaseTool | dict[str, Any]]) -> list[dict[str, Any]]:
    """Create editable Relay views without changing the original tools."""
    schemas: list[dict[str, Any]] = []

    for tool in tools:
        if isinstance(tool, dict):
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                schemas.append(copy.deepcopy(tool))
            else:
                schemas.append(_native_tool_schema(tool))
            continue

        provider_definition = (tool.extras or {}).get("provider_tool_definition")
        if provider_definition is not None:
            if not isinstance(provider_definition, dict):
                raise ValueError("LangChain provider tool definition must be an object")
            schemas.append(_native_tool_schema(provider_definition))
            continue

        input_schema = tool.tool_call_schema
        if isinstance(input_schema, dict):
            parameters = copy.deepcopy(input_schema)
        elif issubclass(input_schema, BaseModel):
            parameters = copy.deepcopy(input_schema.model_json_schema())
        else:
            parameters = copy.deepcopy(input_schema.schema())
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )

    return schemas


def _restore_model_tools(
    original_tools: list[BaseTool | dict[str, Any]],
    edited_schemas: Sequence[dict[str, Any]],
) -> list[BaseTool | dict[str, Any]]:
    original_schemas = _model_tool_schemas(original_tools)
    if len(edited_schemas) != len(original_tools):
        raise ValueError("Relay cannot add or remove LangChain tools")

    restored_tools: list[BaseTool | dict[str, Any]] = []

    for original_tool, original_schema, edited_schema in zip(
        original_tools, original_schemas, edited_schemas, strict=True
    ):
        if edited_schema == original_schema:
            restored_tools.append(original_tool)
            continue

        if edited_schema.get("type") != original_schema["type"]:
            raise ValueError("Relay cannot change a LangChain tool's representation")

        if original_schema["type"] == "provider_native":
            if (
                edited_schema.get("provider") != _LANGCHAIN_PROVIDER
                or edited_schema.get("kind") != "tool"
                or not isinstance(edited_schema.get("value"), dict)
            ):
                raise ValueError("Invalid edited LangChain provider tool")

            edited_definition = copy.deepcopy(edited_schema["value"])
            original_definition = original_schema["value"]
            for identity_field in ("type", "name"):
                if edited_definition.get(identity_field) != original_definition.get(identity_field):
                    raise ValueError("Relay cannot change a LangChain provider tool's identity")
            if "type" not in original_definition and "name" not in original_definition:
                if edited_definition.keys() != original_definition.keys():
                    raise ValueError("Relay cannot change a LangChain provider tool's identity")
            if isinstance(original_tool, BaseTool):
                extras = copy.deepcopy(original_tool.extras or {})
                extras["provider_tool_definition"] = edited_definition
                restored_tools.append(original_tool.model_copy(update={"extras": extras}))
            else:
                restored_tools.append(edited_definition)
            continue

        original_function = original_schema["function"]
        edited_function = edited_schema.get("function")
        if not isinstance(edited_function, dict) or edited_function.get("name") != original_function["name"]:
            raise ValueError("Relay cannot rename or reorder LangChain tools")

        if isinstance(original_tool, dict):
            restored_tools.append(copy.deepcopy(edited_schema))
            continue

        description = edited_function.get("description")
        parameters = edited_function.get("parameters")
        if not isinstance(description, str) or not isinstance(parameters, dict):
            raise ValueError("Edited LangChain function fields have invalid types")

        allowed_schema = copy.deepcopy(original_schema)
        allowed_schema["function"]["description"] = description
        allowed_schema["function"]["parameters"] = parameters
        if edited_schema != allowed_schema:
            raise ValueError("Unsupported edit to a LangChain function tool")

        changes = {}
        if description != original_function["description"]:
            changes["description"] = description
        if parameters != original_function["parameters"]:
            changes["args_schema"] = parameters
        restored_tools.append(original_tool.model_copy(update=changes))

    return restored_tools


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
    if request.tools:
        payload["tools"] = _model_tool_schemas(request.tools)
    return payload


def _chat_nvidia_with_relay_headers(model: Any, headers: dict[str, Any]) -> Any | None:
    """Return a per-request ChatNVIDIA model carrying Relay headers over HTTP."""
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except ImportError:
        return None

    if not isinstance(model, ChatNVIDIA):
        return None

    relay_headers = {
        key: value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        for key, value in headers.items()
    }
    relay_header_names = {key.lower() for key in relay_headers}
    default_headers = {
        key: value for key, value in model.default_headers.items() if key.lower() not in relay_header_names
    }
    return model.model_copy(update={"default_headers": {**default_headers, **relay_headers}})


_EXTRA_HEADERS_MODEL_CLASSES: tuple[tuple[str, str], ...] = (
    # OpenAI and Anthropic Python SDKs accept ``extra_headers`` as a per-request option.
    ("langchain_openai.chat_models.base", "BaseChatOpenAI"),
    ("langchain_anthropic.chat_models", "ChatAnthropic"),
)


@functools.cache
def _optional_class(module_name: str, class_name: str) -> type | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, class_name, None)


def _model_accepts_extra_headers(model: Any, model_settings: dict[str, Any]) -> bool:
    """Return whether ``model`` sends the ``extra_headers`` model setting as HTTP headers.

    ``extra_headers`` is an OpenAI and Anthropic SDK request option. Other provider SDKs
    treat unknown model settings as request-body fields and reject them, so Relay
    headers only travel this way when the SDK is known to accept the option or the
    caller already configures ``extra_headers`` for the model.
    """
    if isinstance(model_settings.get("extra_headers"), dict):
        return True
    return any(
        (model_class := _optional_class(module_name, class_name)) is not None and isinstance(model, model_class)
        for module_name, class_name in _EXTRA_HEADERS_MODEL_CLASSES
    )


def payload_to_model_request(
    original: ModelRequest[Any],
    llm_request: LLMRequest,
) -> ModelRequest[Any]:
    """Apply supported NeMo Relay request-intercept edits back to ``ModelRequest``."""
    overrides: dict[str, Any] = {}

    raw_messages = llm_request.content.get("messages")
    if isinstance(raw_messages, list) and len(raw_messages) > 0:
        try:
            system_message, messages = split_system_message(_langchain_messages(raw_messages))
            overrides["system_message"] = system_message
            overrides["messages"] = messages
        except Exception:
            pass

    model_settings = llm_request.content.get("model_settings")
    if isinstance(model_settings, dict):
        # Using dict() to ensure we have a copy
        model_settings_copy = dict(model_settings)
        extra_headers = model_settings_copy.get("extra_headers")
        extra_headers = dict(extra_headers) if isinstance(extra_headers, dict) else {}
        overrides["model_settings"] = model_settings_copy
    else:
        overrides["model_settings"] = {}
        extra_headers = {}

    if len(llm_request.headers) > 0:
        model_with_headers = _chat_nvidia_with_relay_headers(original.model, llm_request.headers)
        if model_with_headers is not None:
            overrides["model"] = model_with_headers
        elif _model_accepts_extra_headers(original.model, overrides["model_settings"]):
            extra_headers.update(llm_request.headers)
            overrides["model_settings"]["extra_headers"] = extra_headers
        # Other providers get no Relay headers: their SDKs reject unknown request fields,
        # and failing the model call is worse than dropping observability headers.

    if "tool_choice" in llm_request.content:
        overrides["tool_choice"] = llm_request.content["tool_choice"]

    if "tools" in llm_request.content:
        tool_schemas = llm_request.content["tools"]
        if not isinstance(tool_schemas, list) or not _are_objects(tool_schemas):
            raise ValueError("LangChain model request tools must be a list of tool schema objects")
        if tool_schemas != _model_tool_schemas(original.tools):
            overrides["tools"] = _restore_model_tools(original.tools, tool_schemas)

    return original.override(**overrides) if overrides else original


def _model_response_payload(response: ModelResponse[Any], codec: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages_to_dict(response.result),
    }
    if response.structured_response is not None:
        payload["structured_response"] = codec.to_json(response.structured_response)
    return payload


def _model_response_payload_from_json(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or LANGCHAIN_MODEL_RESPONSE_KEY not in payload:
        raise TypeError("expected serialized LangChain ModelResponse payload")

    response_payload = payload[LANGCHAIN_MODEL_RESPONSE_KEY]
    if not isinstance(response_payload, dict):
        raise TypeError("expected serialized LangChain ModelResponse object")

    return response_payload


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


def _message_content_text(message: BaseMessage) -> str | None:
    content = message.content
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) if parts else None
    return str(content)


def _message_finish_reason(message: BaseMessage) -> str | dict[str, str] | None:
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("finish_reason", "stop_reason"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return _FINISH_REASON_MAP.get(value, {"unknown": value})
    return None


def _message_usage(message: BaseMessage) -> dict[str, Any] | None:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None

    mapped: dict[str, Any] = {}
    for source, target in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, int):
            mapped[target] = value

    return mapped or None


def _message_model(message: BaseMessage) -> str | None:
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("model_name", "model", "model_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _message_response_tool_calls(message: BaseMessage) -> list[dict[str, Any]] | None:
    if not isinstance(message, AIMessage):
        return None
    tool_calls = getattr(message, "tool_calls", [])
    if not tool_calls:
        return None

    return [
        {
            "id": str(tool_call.get("id") or ""),
            "name": str(tool_call["name"]),
            "arguments": tool_call.get("args") or {},
        }
        for tool_call in tool_calls
    ]


def _model_response_to_annotated(payload: dict[str, Any]) -> AnnotatedLLMResponse:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise TypeError("expected serialized LangChain ModelResponse messages")

    messages = messages_from_dict(raw_messages)
    message = next((item for item in reversed(messages) if isinstance(item, AIMessage)), messages[-1])
    extra = {}
    if "structured_response" in payload:
        extra["structured_response"] = payload["structured_response"]

    return AnnotatedLLMResponse(
        id=getattr(message, "id", None),
        model=_message_model(message),
        message=_message_content_text(message),
        tool_calls=_message_response_tool_calls(message),
        finish_reason=_message_finish_reason(message),
        usage=_message_usage(message),
        extra=extra or None,
    )


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
    elif isinstance(payload, list | tuple | set):
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
    elif hasattr(payload, "model_dump"):
        prepared = _prepare_lc_payloads(payload.model_dump(mode="json"))
    else:
        prepared = payload

    return prepared
