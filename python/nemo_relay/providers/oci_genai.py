# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and response codecs for Oracle Cloud Infrastructure (OCI) Generative AI.

The OCI Generative AI chat API wraps requests in a ``ChatDetails`` envelope
(``compartmentId``, ``servingMode``, ``chatRequest``) and supports two chat
request formats selected by ``chatRequest.apiFormat``:

- ``GENERIC``: OpenAI-style ``messages`` whose ``content`` is a list of typed
  parts (``{"type": "TEXT", "text": ...}``). Used by Meta Llama, Google,
  xAI, OpenAI, and imported open-weights models (for example NVIDIA Nemotron)
  hosted on dedicated AI clusters.
- ``COHERE``: a single ``message`` string plus ``chatHistory`` turns with
  ``USER``/``CHATBOT``/``SYSTEM`` roles. Used by Cohere Command models.

``OCIGenAIChatCodec`` normalizes both formats into ``AnnotatedLLMRequest`` so
request intercepts (PII redaction, guardrails, policy) can operate on a single
shape, and merges intercept edits back without dropping provider-specific
fields. ``OCIGenAIResponseCodec`` normalizes ``ChatResult`` responses for
``LLMEnd`` event annotation.

Example::

    import nemo_relay
    from nemo_relay.providers.oci_genai import OCIGenAIChatCodec, OCIGenAIResponseCodec

    result = await nemo_relay.llm.execute(
        "oci-genai",
        nemo_relay.LLMRequest({}, chat_details),
        call_oci_chat,
        codec=OCIGenAIChatCodec(),
        response_codec=OCIGenAIResponseCodec(),
    )
"""

from __future__ import annotations

import typing

from nemo_relay import Json
from nemo_relay._native import AnnotatedLLMRequest, AnnotatedLLMResponse, LLMRequest

# OCI parameter name -> normalized GenerationParams field. Parameters without
# a normalized slot (topK, penalties, seed, ...) stay untouched in the raw
# payload, which encode() preserves.
_GENERIC_PARAM_MAP = {"maxTokens": "max_tokens", "temperature": "temperature", "topP": "top_p", "stop": "stop"}
_COHERE_PARAM_MAP = {
    "maxTokens": "max_tokens",
    "temperature": "temperature",
    "topP": "top_p",
    "stopSequences": "stop",
}

_COHERE_ROLE_TO_NORMALIZED = {"USER": "user", "CHATBOT": "assistant", "SYSTEM": "system", "TOOL": "tool"}
_NORMALIZED_ROLE_TO_COHERE = {v: k for k, v in _COHERE_ROLE_TO_NORMALIZED.items()}


def _get_first(mapping: typing.Mapping[str, typing.Any], *keys: str) -> typing.Any:
    """Return the first present key from ``mapping`` across naming conventions.

    The OCI SDKs emit camelCase JSON while the OCI CLI emits kebab-case.
    Callers pass camelCase keys; kebab-case and snake_case fallbacks are
    derived automatically.
    """
    for key in keys:
        for candidate in (key, _camel_to_kebab(key), _camel_to_snake(key)):
            if candidate in mapping:
                return mapping[candidate]
    return None


def _camel_to_kebab(key: str) -> str:
    return "".join(f"-{c.lower()}" if c.isupper() else c for c in key)


def _camel_to_snake(key: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in key)


def _generic_content_to_text(content: typing.Any) -> typing.Any:
    """Flatten a GENERIC content-part list into plain text when possible."""
    if isinstance(content, str) or content is None:
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and _get_first(part, "type") == "TEXT":
                texts.append(str(_get_first(part, "text") or ""))
            else:
                # Non-text part (image, etc.): leave the list untouched.
                return content
        return "".join(texts)
    return content


def _text_to_generic_content(content: typing.Any) -> typing.Any:
    """Wrap plain text back into the GENERIC typed content-part list."""
    if isinstance(content, str):
        return [{"type": "TEXT", "text": content}]
    return content


def _normalize_usage(usage: typing.Any) -> dict[str, typing.Any] | None:
    """Map OCI usage counters onto the normalized ``Usage`` field names."""
    if not isinstance(usage, dict):
        return None
    normalized: dict[str, typing.Any] = {}
    prompt_tokens = _get_first(usage, "promptTokens")
    completion_tokens = _get_first(usage, "completionTokens")
    total_tokens = _get_first(usage, "totalTokens")
    if prompt_tokens is not None:
        normalized["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        normalized["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    return normalized or None


def _oci_tool_call_to_normalized(tool_call: typing.Any) -> typing.Any:
    """Convert a flat OCI ``toolCalls`` entry into the normalized nested shape."""
    if not isinstance(tool_call, dict) or "function" in tool_call:
        return tool_call
    return {
        "id": _get_first(tool_call, "id"),
        "type": "function",
        "function": {
            "name": _get_first(tool_call, "name"),
            "arguments": _get_first(tool_call, "arguments"),
        },
    }


def _normalized_tool_call_to_oci(tool_call: typing.Any) -> typing.Any:
    """Convert a normalized nested tool call back into the flat OCI shape."""
    if not isinstance(tool_call, dict) or "function" not in tool_call:
        return tool_call
    function = tool_call.get("function") or {}
    return {
        "id": tool_call.get("id"),
        "type": "FUNCTION",
        "name": function.get("name"),
        "arguments": function.get("arguments"),
    }


class OCIGenAIChatCodec:
    """Request codec for the OCI Generative AI chat API.

    Accepts either a full ``ChatDetails`` envelope or a bare ``chatRequest``
    payload, in the ``GENERIC`` or ``COHERE`` API format. Provider-specific
    fields that have no normalized equivalent are carried in
    ``AnnotatedLLMRequest.extra`` and restored by ``encode()``.
    """

    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode an OCI chat payload into ``AnnotatedLLMRequest``."""
        content = dict(request.content)
        chat_request = _get_first(content, "chatRequest")
        envelope = content if isinstance(chat_request, dict) else None
        chat_request = dict(chat_request) if isinstance(chat_request, dict) else content

        api_format = str(_get_first(chat_request, "apiFormat") or "GENERIC").upper()
        model = self._model_from_envelope(envelope)

        if api_format == "COHERE":
            messages, params = self._decode_cohere(chat_request)
        else:
            messages, params = self._decode_generic(chat_request)

        tools = _get_first(chat_request, "tools")
        tool_choice = _get_first(chat_request, "toolChoice")

        return AnnotatedLLMRequest(
            messages,
            model=model,
            params=params or None,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            api_specific={"api": "custom", "api_name": "oci_genai", "data": {"apiFormat": api_format}},
            extra={"envelope": envelope, "chatRequest": chat_request},
        )

    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Merge annotated edits back into the original OCI payload.

        Edits are applied by comparing the annotation against a decoded
        baseline of the original request and patching only changed fields, so
        ``encode(decode(original), original)`` returns the original payload at
        the JSON-value level and unmodeled provider fields survive edits.
        """
        content = dict(original.content)
        chat_request_value = _get_first(content, "chatRequest")
        has_envelope = isinstance(chat_request_value, dict)
        chat_request = dict(chat_request_value) if has_envelope else dict(content)

        api_specific = annotated.api_specific or {}
        api_data_value = api_specific.get("data")
        api_data: dict[str, typing.Any] = api_data_value if isinstance(api_data_value, dict) else {}
        api_format = str(api_data.get("apiFormat") or _get_first(chat_request, "apiFormat") or "GENERIC").upper()

        baseline = self.decode(original)
        edited_messages = [dict(message) for message in annotated.messages]
        baseline_messages = [dict(message) for message in baseline.messages]

        if edited_messages != baseline_messages:
            if api_format == "COHERE":
                self._encode_cohere(chat_request, annotated)
            else:
                self._patch_generic_messages(chat_request, edited_messages, baseline_messages)

        edited_params = dict(annotated.params or {})
        baseline_params = dict(baseline.params or {})
        if edited_params != baseline_params:
            param_map = _COHERE_PARAM_MAP if api_format == "COHERE" else _GENERIC_PARAM_MAP
            normalized_to_oci = {v: k for k, v in param_map.items()}
            for key, value in edited_params.items():
                if value is None or baseline_params.get(key) == value:
                    continue
                chat_request[normalized_to_oci.get(key, key)] = value

        if has_envelope:
            content["chatRequest"] = chat_request
            return LLMRequest(original.headers, content)
        return LLMRequest(original.headers, chat_request)

    @classmethod
    def _patch_generic_messages(
        cls,
        chat_request: dict[str, typing.Any],
        edited_messages: list[dict[str, typing.Any]],
        baseline_messages: list[dict[str, typing.Any]],
    ) -> None:
        """Rewrite only the GENERIC messages that intercepts actually changed.

        Unchanged messages are carried over from the raw payload verbatim so
        per-message provider fields without a normalized equivalent survive.
        """
        raw_messages = _get_first(chat_request, "messages") or []
        patched: list[typing.Any] = []
        for index, edited in enumerate(edited_messages):
            unchanged = index < len(baseline_messages) and edited == baseline_messages[index]
            if unchanged and index < len(raw_messages):
                patched.append(raw_messages[index])
            else:
                patched.append(cls._build_generic_message(edited))
        chat_request["messages"] = patched

    @staticmethod
    def _build_generic_message(message: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
        encoded: dict[str, typing.Any] = {
            "role": str(message.get("role") or "user").upper(),
            "content": _text_to_generic_content(message.get("content")),
        }
        if "tool_calls" in message:
            encoded["toolCalls"] = [_normalized_tool_call_to_oci(tc) for tc in message["tool_calls"]]
        if "tool_call_id" in message:
            encoded["toolCallId"] = message["tool_call_id"]
        return encoded

    @staticmethod
    def _model_from_envelope(envelope: typing.Mapping[str, typing.Any] | None) -> str | None:
        if not envelope:
            return None
        serving_mode = _get_first(envelope, "servingMode")
        if not isinstance(serving_mode, dict):
            return None
        model = _get_first(serving_mode, "modelId") or _get_first(serving_mode, "endpointId")
        return str(model) if model is not None else None

    @staticmethod
    def _decode_generic(
        chat_request: typing.Mapping[str, typing.Any],
    ) -> tuple[list[dict[str, typing.Any]], dict[str, typing.Any]]:
        messages: list[dict[str, typing.Any]] = []
        for message in _get_first(chat_request, "messages") or []:
            role = str(_get_first(message, "role") or "USER").lower()
            normalized: dict[str, typing.Any] = {
                "role": role,
                "content": _generic_content_to_text(_get_first(message, "content")),
            }
            tool_calls = _get_first(message, "toolCalls")
            if tool_calls is not None:
                normalized["tool_calls"] = [_oci_tool_call_to_normalized(tc) for tc in tool_calls]
            tool_call_id = _get_first(message, "toolCallId")
            if tool_call_id is not None:
                normalized["tool_call_id"] = tool_call_id
            messages.append(normalized)

        params = {
            normalized_key: chat_request[oci_key]
            for oci_key, normalized_key in _GENERIC_PARAM_MAP.items()
            if oci_key in chat_request
        }
        return messages, params

    @staticmethod
    def _decode_cohere(
        chat_request: typing.Mapping[str, typing.Any],
    ) -> tuple[list[dict[str, typing.Any]], dict[str, typing.Any]]:
        messages: list[dict[str, typing.Any]] = []

        preamble = _get_first(chat_request, "preambleOverride")
        if preamble:
            messages.append({"role": "system", "content": preamble})

        for turn in _get_first(chat_request, "chatHistory") or []:
            role = _COHERE_ROLE_TO_NORMALIZED.get(str(_get_first(turn, "role") or "USER").upper(), "user")
            messages.append({"role": role, "content": _get_first(turn, "message")})

        current = _get_first(chat_request, "message")
        if current is not None:
            messages.append({"role": "user", "content": current})

        params = {
            normalized_key: chat_request[oci_key]
            for oci_key, normalized_key in _COHERE_PARAM_MAP.items()
            if oci_key in chat_request
        }
        return messages, params

    @staticmethod
    def _encode_cohere(chat_request: dict[str, typing.Any], annotated: AnnotatedLLMRequest) -> None:
        messages = list(annotated.messages)

        if messages and str(messages[0].get("role")) == "system":
            chat_request["preambleOverride"] = messages[0].get("content")
            messages = messages[1:]

        current_message = ""
        if messages and str(messages[-1].get("role")) == "user":
            current_message = messages[-1].get("content") or ""
            messages = messages[:-1]

        history = [
            {
                "role": _NORMALIZED_ROLE_TO_COHERE.get(str(turn.get("role")), "USER"),
                "message": turn.get("content"),
            }
            for turn in messages
        ]

        chat_request["message"] = current_message
        if history or "chatHistory" in chat_request:
            chat_request["chatHistory"] = history


class OCIGenAIResponseCodec:
    """Response codec for OCI Generative AI ``ChatResult`` payloads.

    Normalizes both ``GENERIC`` (``choices``-based) and ``COHERE``
    (``text``-based) chat responses for ``LLMEnd`` event annotation. Accepts
    the full ``ChatResult`` (``modelId``, ``chatResponse``) or a bare chat
    response object.
    """

    def decode_response(self, response: Json) -> AnnotatedLLMResponse:
        """Decode a raw OCI chat response into ``AnnotatedLLMResponse``."""
        if not isinstance(response, dict):
            return AnnotatedLLMResponse(extra={"raw": response})

        chat_response = _get_first(response, "chatResponse")
        envelope = response if isinstance(chat_response, dict) else None
        chat_response = dict(chat_response) if isinstance(chat_response, dict) else dict(response)

        model = None
        if envelope is not None:
            model_value = _get_first(envelope, "modelId")
            model = str(model_value) if model_value is not None else None

        api_format = str(_get_first(chat_response, "apiFormat") or "GENERIC").upper()

        message: typing.Any = None
        tool_calls: typing.Any = None
        finish_reason: typing.Any = None

        if api_format == "COHERE":
            message = _get_first(chat_response, "text")
            tool_calls = _get_first(chat_response, "toolCalls")
            finish_reason = _get_first(chat_response, "finishReason")
        else:
            choices = _get_first(chat_response, "choices") or []
            if choices:
                first_choice = choices[0]
                finish_reason = _get_first(first_choice, "finishReason")
                raw_message = _get_first(first_choice, "message")
                if isinstance(raw_message, dict):
                    message = _generic_content_to_text(_get_first(raw_message, "content"))
                    tool_calls = _get_first(raw_message, "toolCalls")

        usage = _normalize_usage(_get_first(chat_response, "usage"))

        return AnnotatedLLMResponse(
            model=model,
            message=message,
            tool_calls=tool_calls if isinstance(tool_calls, list) else None,
            finish_reason=finish_reason,
            usage=usage,
            api_specific={"api": "custom", "api_name": "oci_genai", "data": {"apiFormat": api_format}},
        )


__all__ = [
    "OCIGenAIChatCodec",
    "OCIGenAIResponseCodec",
]
