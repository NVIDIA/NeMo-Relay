// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the shared content surface through each built-in decoder.

use nemo_relay_codecs::resolve::{ProviderSurface, request_codec, response_codec};
use nemo_relay_types::api::llm::LlmRequest;
use nemo_relay_types::codec::content::{ContentField, ContentKind, PayloadRef};
use serde_json::json;

#[test]
fn content_views_cover_each_builtin_provider() {
    let cases = [
        (
            ProviderSurface::OpenAIChat,
            json!({"messages":[{"role":"user","content":"question"}]}),
            json!({"choices":[{"message":{"content":"answer","tool_calls":[
                {"id":"c","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}
            ]},"finish_reason":"tool_calls"}]}),
        ),
        (
            ProviderSurface::OpenAIResponses,
            json!({"input":[{"role":"user","content":[{"type":"input_text","text":"question"}]}]}),
            json!({"output":[
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
                {"type":"function_call","call_id":"c","name":"lookup","arguments":"{\"q\":1}"}
            ]}),
        ),
        (
            ProviderSurface::AnthropicMessages,
            json!({"messages":[{"role":"user","content":[{"type":"text","text":"question"}]}]}),
            json!({"type":"message","content":[
                {"type":"text","text":"answer"},
                {"type":"tool_use","id":"c","name":"lookup","input":{"q":1}}
            ]}),
        ),
        (
            ProviderSurface::GeminiGenerateContent,
            json!({"contents":[{"role":"user","parts":[{"text":"question"}]}]}),
            json!({"candidates":[{"content":{"role":"model","parts":[
                {"text":"answer"},{"functionCall":{"name":"lookup","args":{"q":1}}}
            ]},"finishReason":"STOP"}]}),
        ),
        (
            ProviderSurface::OCIGenAI,
            json!({"chatRequest":{"apiFormat":"GENERIC","messages":[
                {"role":"USER","content":[{"type":"TEXT","text":"question"}]}
            ]}}),
            json!({"chatResponse":{"apiFormat":"GENERIC","choices":[{"message":{
                "role":"ASSISTANT","content":[{"type":"TEXT","text":"answer"}],
                "toolCalls":[{"id":"c","type":"FUNCTION","name":"lookup","arguments":"{\"q\":1}"}]
            },"finishReason":"stop"}]}}),
        ),
    ];
    for (surface, request, response) in cases {
        let request = request_codec(surface)
            .decode(&LlmRequest {
                headers: Default::default(),
                content: request,
            })
            .unwrap();
        assert_eq!(
            request.last_message().unwrap().text().as_deref(),
            Some("question"),
            "{surface:?}"
        );
        assert_eq!(
            request.last_user_message().as_deref(),
            Some("question"),
            "{surface:?}"
        );
        let response = response_codec(surface).decode_response(&response).unwrap();
        assert_eq!(
            response.output().text().as_deref(),
            Some("answer"),
            "{surface:?}"
        );
        let calls = response
            .output()
            .parts()
            .filter_map(|entry| match entry.kind {
                ContentKind::ToolCall(call) => {
                    assert_eq!(entry.location.field, ContentField::ToolCalls);
                    Some(call)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 1, "{surface:?}");
        assert_eq!(calls[0].name, "lookup", "{surface:?}");
        assert!(
            matches!(calls[0].arguments, PayloadRef::Json(value) if value == &json!({"q":1})),
            "{surface:?}"
        );
    }
}

#[test]
fn anthropic_view_preserves_already_joined_normalization() {
    let response = response_codec(ProviderSurface::AnthropicMessages)
        .decode_response(&json!({
            "type":"message", "content":[
                {"type":"text", "text":"first"},
                {"type":"tool_use", "id":"c", "name":"lookup", "input":{}},
                {"type":"text", "text":"second"}
            ]
        }))
        .unwrap();
    assert_eq!(
        response.output().text_parts().collect::<Vec<_>>(),
        ["first\nsecond"]
    );
    assert_eq!(response.output().parts().count(), 2);
    assert_eq!(response.response_text().as_deref(), Some("first\nsecond"));
}

#[test]
fn extension_fields_do_not_hide_known_request_text() {
    for (surface, field, text_type) in [
        (ProviderSurface::OpenAIChat, "messages", "text"),
        (ProviderSurface::OpenAIResponses, "input", "input_text"),
        (ProviderSurface::AnthropicMessages, "messages", "text"),
    ] {
        let codec = request_codec(surface);
        for role in ["user", "assistant"] {
            for content in [
                json!("known text"),
                json!([
                    {"type":text_type, "text":"first"},
                    {"type":text_type, "text":"second"}
                ]),
            ] {
                let original = LlmRequest {
                    headers: Default::default(),
                    content: json!({field:[{
                        "role":role, "content":content,
                        "provider_extra":{"content":"not message text", "future":true}
                    }]}),
                };
                let annotated = codec.decode(&original).unwrap();
                let before = serde_json::to_value(&annotated).unwrap();
                assert!(matches!(
                    annotated.messages[0],
                    nemo_relay_types::codec::request::Message::ProviderNative { .. }
                ));
                let expected = if content.is_string() {
                    "known text"
                } else {
                    "first\nsecond"
                };
                assert_eq!(
                    annotated.last_message().unwrap().text().as_deref(),
                    Some(expected),
                    "{surface:?}: {role}"
                );
                assert_eq!(
                    annotated.last_user_message().as_deref(),
                    (role == "user").then_some(expected),
                    "{surface:?}"
                );
                assert_eq!(serde_json::to_value(&annotated).unwrap(), before);
                assert_eq!(codec.encode(&annotated, &original).unwrap(), original);
            }
        }
    }
}
