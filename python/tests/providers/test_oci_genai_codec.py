# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OCI Generative AI request and response codecs."""

from __future__ import annotations

import typing

from nemo_relay import LLMRequest
from nemo_relay.providers.oci_genai import OCIGenAIChatCodec, OCIGenAIResponseCodec

DEDICATED_ENDPOINT = "ocid1.generativeaiendpoint.oc1.us-chicago-1.example"


def _j(value: object) -> typing.Any:
    """Erase JSON-union typing for assertion subscripting."""
    return value


GENERIC_CHAT_DETAILS: dict[str, typing.Any] = {
    "compartmentId": "ocid1.compartment.oc1..example",
    "servingMode": {"servingType": "DEDICATED", "endpointId": DEDICATED_ENDPOINT},
    "chatRequest": {
        "apiFormat": "GENERIC",
        "messages": [
            {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "You are terse."}]},
            {"role": "USER", "content": [{"type": "TEXT", "text": "My SSN is 111-22-3333."}]},
        ],
        "maxTokens": 600,
        "temperature": 0.0,
    },
}

COHERE_CHAT_DETAILS: dict[str, typing.Any] = {
    "compartmentId": "ocid1.compartment.oc1..example",
    "servingMode": {"servingType": "ON_DEMAND", "modelId": "cohere.command-a-03-2025"},
    "chatRequest": {
        "apiFormat": "COHERE",
        "preambleOverride": "You are terse.",
        "chatHistory": [
            {"role": "USER", "message": "hello"},
            {"role": "CHATBOT", "message": "hi"},
        ],
        "message": "What is the weather?",
        "maxTokens": 100,
    },
}

# Shape observed from a live dedicated-endpoint chat (imported NVIDIA Nemotron 3).
GENERIC_CHAT_RESULT: dict[str, typing.Any] = {
    "modelId": DEDICATED_ENDPOINT,
    "modelVersion": "1.0",
    "chatResponse": {
        "apiFormat": "GENERIC",
        "timeCreated": "2026-07-23T22:59:00.000Z",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "NEMOTRON3_OK"}],
                },
                "finishReason": "stop",
            }
        ],
        "usage": {"promptTokens": 18, "completionTokens": 5, "totalTokens": 23},
    },
}

COHERE_CHAT_RESULT: dict[str, typing.Any] = {
    "modelId": "cohere.command-a-03-2025",
    "chatResponse": {
        "apiFormat": "COHERE",
        "text": "Sunny and 72.",
        "finishReason": "COMPLETE",
        "usage": {"promptTokens": 12, "completionTokens": 4, "totalTokens": 16},
    },
}


class TestGenericRequestCodec:
    def test_decode_envelope(self):
        annotated = OCIGenAIChatCodec().decode(LLMRequest({}, GENERIC_CHAT_DETAILS))

        assert [m["role"] for m in annotated.messages] == ["system", "user"]
        assert _j(annotated.messages[1])["content"] == "My SSN is 111-22-3333."
        assert annotated.model == DEDICATED_ENDPOINT
        assert _j(annotated.params)["max_tokens"] == 600
        assert _j(annotated.params)["temperature"] == 0.0
        assert _j(annotated.api_specific)["api_name"] == "oci_genai"
        assert _j(annotated.api_specific)["data"]["apiFormat"] == "GENERIC"

    def test_decode_bare_chat_request(self):
        annotated = OCIGenAIChatCodec().decode(LLMRequest({}, GENERIC_CHAT_DETAILS["chatRequest"]))

        assert [m["role"] for m in annotated.messages] == ["system", "user"]
        assert annotated.model is None

    def test_redaction_round_trip_preserves_envelope(self):
        codec = OCIGenAIChatCodec()
        original = LLMRequest({}, GENERIC_CHAT_DETAILS)
        annotated = codec.decode(original)

        edited_messages = [dict(m) for m in annotated.messages]
        edited_messages[1]["content"] = "My SSN is [REDACTED]."
        from nemo_relay._native import AnnotatedLLMRequest

        edited = AnnotatedLLMRequest(
            edited_messages,
            model=annotated.model,
            params=annotated.params,
            api_specific=annotated.api_specific,
            extra=annotated.extra,
        )

        encoded = codec.encode(edited, original)

        chat_request = _j(encoded.content)["chatRequest"]
        assert chat_request["messages"][1]["role"] == "USER"
        assert chat_request["messages"][1]["content"] == [{"type": "TEXT", "text": "My SSN is [REDACTED]."}]
        # Envelope fields survive untouched.
        assert _j(encoded.content)["compartmentId"] == GENERIC_CHAT_DETAILS["compartmentId"]
        assert _j(encoded.content)["servingMode"] == GENERIC_CHAT_DETAILS["servingMode"]
        assert chat_request["maxTokens"] == 600

    def test_tool_calls_round_trip(self):
        payload: dict[str, typing.Any] = {
            "apiFormat": "GENERIC",
            "messages": [
                {
                    "role": "ASSISTANT",
                    "content": [],
                    "toolCalls": [{"id": "call-1", "type": "FUNCTION", "name": "get_weather", "arguments": "{}"}],
                },
                {"role": "TOOL", "content": [{"type": "TEXT", "text": "72F"}], "toolCallId": "call-1"},
            ],
        }
        codec = OCIGenAIChatCodec()
        original = LLMRequest({}, payload)
        annotated = codec.decode(original)

        assert _j(annotated.messages[0])["tool_calls"][0]["function"]["name"] == "get_weather"
        assert _j(annotated.messages[1])["tool_call_id"] == "call-1"

        encoded = codec.encode(annotated, original)
        assert _j(encoded.content)["messages"][0]["toolCalls"][0] == {
            "id": "call-1",
            "type": "FUNCTION",
            "name": "get_weather",
            "arguments": "{}",
        }
        assert _j(encoded.content)["messages"][1]["toolCallId"] == "call-1"


class TestCohereRequestCodec:
    def test_decode(self):
        annotated = OCIGenAIChatCodec().decode(LLMRequest({}, COHERE_CHAT_DETAILS))

        assert [m["role"] for m in annotated.messages] == ["system", "user", "assistant", "user"]
        assert _j(annotated.messages[0])["content"] == "You are terse."
        assert _j(annotated.messages[-1])["content"] == "What is the weather?"
        assert annotated.model == "cohere.command-a-03-2025"
        assert _j(annotated.api_specific)["data"]["apiFormat"] == "COHERE"

    def test_round_trip(self):
        codec = OCIGenAIChatCodec()
        original = LLMRequest({}, COHERE_CHAT_DETAILS)
        annotated = codec.decode(original)
        encoded = codec.encode(annotated, original)

        chat_request = _j(encoded.content)["chatRequest"]
        assert chat_request["message"] == "What is the weather?"
        assert chat_request["preambleOverride"] == "You are terse."
        assert chat_request["chatHistory"] == [
            {"role": "USER", "message": "hello"},
            {"role": "CHATBOT", "message": "hi"},
        ]
        assert _j(encoded.content)["servingMode"] == COHERE_CHAT_DETAILS["servingMode"]


class TestIdentityInvariant:
    """encode(decode(original), original) == original at the JSON-value level.

    This is the same guarantee the built-in provider codecs document in the
    Provider Codecs guide, including preservation of unmodeled fields.
    """

    UNMODELED_GENERIC: typing.ClassVar[dict[str, typing.Any]] = {
        "compartmentId": "ocid1.compartment.oc1..example",
        "opcRetryToken": "retry-abc",
        "servingMode": {"servingType": "DEDICATED", "endpointId": DEDICATED_ENDPOINT, "futureFlag": True},
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": [
                {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "Be terse."}], "name": "sys-1"},
                {"role": "USER", "content": [{"type": "TEXT", "text": "hello"}], "unknownPerMessage": 7},
            ],
            "maxTokens": 64,
            "topK": 40,
            "seed": 7,
            "unknownFutureField": {"nested": True},
        },
    }

    def test_generic_identity(self):
        codec = OCIGenAIChatCodec()
        original = LLMRequest({}, self.UNMODELED_GENERIC)
        encoded = codec.encode(codec.decode(original), original)

        assert dict(encoded.content) == self.UNMODELED_GENERIC

    def test_cohere_identity(self):
        cohere_payload = dict(COHERE_CHAT_DETAILS)
        cohere_payload["chatRequest"] = {**COHERE_CHAT_DETAILS["chatRequest"], "isForceSingleStep": True}
        codec = OCIGenAIChatCodec()
        original = LLMRequest({}, cohere_payload)
        encoded = codec.encode(codec.decode(original), original)

        assert dict(encoded.content) == cohere_payload

    def test_edit_preserves_unmodeled_fields_on_untouched_messages(self):
        """Editing one message must not disturb unmodeled fields on the others."""
        from nemo_relay._native import AnnotatedLLMRequest

        codec = OCIGenAIChatCodec()
        original = LLMRequest({}, self.UNMODELED_GENERIC)
        annotated = codec.decode(original)

        edited_messages = [dict(m) for m in annotated.messages]
        edited_messages[1]["content"] = "redacted"
        edited = AnnotatedLLMRequest(
            edited_messages,
            model=annotated.model,
            params=annotated.params,
            api_specific=annotated.api_specific,
            extra=annotated.extra,
        )

        encoded = codec.encode(edited, original)
        chat_request = _j(encoded.content)["chatRequest"]

        # Untouched system message keeps its unmodeled per-message field.
        assert chat_request["messages"][0] == self.UNMODELED_GENERIC["chatRequest"]["messages"][0]
        # Edited message carries the redaction.
        assert chat_request["messages"][1]["content"] == [{"type": "TEXT", "text": "redacted"}]
        # Unmodeled request-level fields survive.
        assert chat_request["topK"] == 40
        assert chat_request["seed"] == 7
        assert chat_request["unknownFutureField"] == {"nested": True}
        assert encoded.content["opcRetryToken"] == "retry-abc"

    def test_param_edit_only_touches_changed_param(self):
        from nemo_relay._native import AnnotatedLLMRequest

        codec = OCIGenAIChatCodec()
        original = LLMRequest({}, self.UNMODELED_GENERIC)
        annotated = codec.decode(original)

        params = dict(annotated.params or {})
        params["max_tokens"] = 128
        edited = AnnotatedLLMRequest(
            [dict(m) for m in annotated.messages],
            model=annotated.model,
            params=params,
            api_specific=annotated.api_specific,
            extra=annotated.extra,
        )

        encoded = codec.encode(edited, original)
        chat_request = _j(encoded.content)["chatRequest"]

        assert chat_request["maxTokens"] == 128
        assert chat_request["messages"] == self.UNMODELED_GENERIC["chatRequest"]["messages"]


class TestResponseCodec:
    def test_generic_chat_result(self):
        annotated = OCIGenAIResponseCodec().decode_response(GENERIC_CHAT_RESULT)

        assert annotated.model == DEDICATED_ENDPOINT
        assert annotated.message == "NEMOTRON3_OK"
        assert annotated.finish_reason == "stop"
        assert _j(annotated.usage)["total_tokens"] == 23
        assert _j(annotated.usage)["prompt_tokens"] == 18
        assert _j(annotated.api_specific)["data"]["apiFormat"] == "GENERIC"

    def test_cohere_chat_result(self):
        annotated = OCIGenAIResponseCodec().decode_response(COHERE_CHAT_RESULT)

        assert annotated.message == "Sunny and 72."
        assert annotated.finish_reason == "COMPLETE"
        assert annotated.model == "cohere.command-a-03-2025"
        assert _j(annotated.api_specific)["data"]["apiFormat"] == "COHERE"

    def test_kebab_case_cli_shape(self):
        cli_shaped = {
            "model-id": DEDICATED_ENDPOINT,
            "chat-response": {
                "api-format": "GENERIC",
                "choices": [
                    {
                        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "hello"}]},
                        "finish-reason": "stop",
                    }
                ],
                "usage": {"total-tokens": 9},
            },
        }
        annotated = OCIGenAIResponseCodec().decode_response(cli_shaped)

        assert annotated.message == "hello"
        assert annotated.finish_reason == "stop"

    def test_non_dict_response(self):
        annotated = OCIGenAIResponseCodec().decode_response("plain text")

        assert _j(annotated.extra)["raw"] == "plain text"
