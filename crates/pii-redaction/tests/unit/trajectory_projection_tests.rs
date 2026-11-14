// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::collections::BTreeMap;

use nemo_relay_runtime::api::llm::LlmRequest;
use nemo_relay_runtime::codec::response_codec;

use crate::trajectory::{CustomMarkPayloadPolicy, TrajectorySanitizer};

const SURFACES: [ProviderSurface; 5] = [
    ProviderSurface::OpenAIChat,
    ProviderSurface::OpenAIResponses,
    ProviderSurface::AnthropicMessages,
    ProviderSurface::OCIGenAI,
    ProviderSurface::GeminiGenerateContent,
];

#[test]
fn strict_response_text_preserves_absence_whitespace_and_empty_parts() {
    for (message, expected) in [
        (None, None),
        (Some(json!("")), Some("")),
        (
            Some(json!(" \t{\"text\":\"literal JSON\"} \n")),
            Some(" \t{\"text\":\"literal JSON\"} \n"),
        ),
        (Some(json!([])), None),
        (
            Some(json!([
                {"type": "text", "text": " first "},
                {"type": "refusal", "refusal": ""},
                {"type": "text", "text": " last "}
            ])),
            Some(" first \n\n last "),
        ),
    ] {
        let response = AnnotatedLlmResponse {
            message: message.map(|message| serde_json::from_value(message).unwrap()),
            ..Default::default()
        };
        assert_eq!(response_text(&response).as_deref(), expected);
    }
}

#[test]
fn response_projections_exclude_native_media_metadata_and_tool_payloads() {
    let excluded = json!([
        {"type": "provider_native", "provider": "custom", "kind": "output_text", "value": {"text": "SECRET"}},
        {"type": "provider_native", "provider": "custom", "kind": "refusal", "value": {"refusal": "SECRET"}},
        {"type": "image", "image": {"text": "SECRET"}},
        {"type": "audio", "audio": {"text": "SECRET"}},
        {"type": "file", "file": {"text": "SECRET"}},
        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"text": "SECRET"}},
        {"type": "tool_result", "tool_use_id": "call-1", "content": "SECRET"}
    ]);
    for with_text in [false, true] {
        let mut parts = excluded.as_array().unwrap().clone();
        if with_text {
            parts.push(
                json!({"type": "text", "text": "[REDACTED]", "extension": {"text": "SECRET"}}),
            );
            parts.push(json!({"type": "refusal", "refusal": "[REDACTED]"}));
        }
        let response: AnnotatedLlmResponse = serde_json::from_value(json!({
            "message": parts,
            "tool_calls": [{"id": "call-1", "name": "lookup", "arguments": {"text": "SECRET"}}],
            "extra_field": "SECRET",
            "finish_reason": "complete",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        }))
        .unwrap();
        for surface in SURFACES {
            let rendered = render_response(surface, &response).unwrap();
            assert!(
                !rendered.to_string().contains("SECRET"),
                "{surface:?}: {rendered}"
            );
            let decoded = response_codec(surface).decode_response(&rendered).unwrap();
            assert_eq!(
                decoded.response_text().as_deref(),
                with_text.then_some("[REDACTED]\n[REDACTED]"),
                "{surface:?}"
            );
            let calls = decoded.tool_calls.unwrap();
            assert_eq!(calls[0].id, "call-1", "{surface:?}");
            assert_eq!(calls[0].name, "lookup", "{surface:?}");
            assert_eq!(calls[0].arguments, json!({}), "{surface:?}");
            assert_eq!(decoded.usage.unwrap().total_tokens, Some(5), "{surface:?}");
        }
    }
}

#[test]
fn projections_are_codec_valid_and_never_copy_normalized_extras() {
    let request: AnnotatedLlmRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "[REDACTED]"}]
    }))
    .unwrap();
    let response: AnnotatedLlmResponse = serde_json::from_value(json!({
        "id": "[REDACTED]",
        "model": "trusted-model",
        "message": "[REDACTED]",
        "tool_calls": [{
            "id": "[REDACTED]",
            "name": "safe_tool",
            "arguments": {"secret": "SECRET"}
        }],
        "finish_reason": "complete",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "future_provider_extension": {"secret": "SECRET"}
    }))
    .unwrap();

    for surface in SURFACES {
        let rendered_request = render_request(surface, &request)
            .unwrap_or_else(|| panic!("request projection failed for {surface:?}"));
        assert!(rendered_request.headers.is_empty());
        request_codec(surface)
            .decode(&rendered_request)
            .unwrap_or_else(|err| {
                panic!("request projection was not decodable for {surface:?}: {err}")
            });

        let rendered_response = render_response(surface, &response)
            .unwrap_or_else(|| panic!("response projection failed for {surface:?}"));
        let serialized = serde_json::to_string(&rendered_response).unwrap();
        assert!(!serialized.contains("SECRET"), "{surface:?}: {serialized}");
        response_codec(surface)
            .decode_response(&rendered_response)
            .unwrap_or_else(|err| {
                panic!("response projection was not decodable for {surface:?}: {err}")
            });
    }
}

#[test]
fn openai_projections_preserve_refusal_fields_empty_values_and_part_order() {
    for (message, chat_text, chat_refusal, response_parts) in [
        (None, None, None, json!([])),
        (Some(json!([])), None, None, json!([])),
        (
            Some(json!("")),
            Some(""),
            None,
            json!([{"type": "output_text", "text": ""}]),
        ),
        (
            Some(json!([{"type": "refusal", "refusal": ""}])),
            None,
            Some(""),
            json!([{"type": "refusal", "refusal": ""}]),
        ),
        (
            Some(json!([
                {"type": "text", "text": " first "},
                {"type": "refusal", "refusal": "no"},
                {"type": "text", "text": ""},
                {"type": "text", "text": "last"},
                {"type": "refusal", "refusal": " again "}
            ])),
            Some(" first \n\nlast"),
            Some("no\n again "),
            json!([
                {"type": "output_text", "text": " first "},
                {"type": "refusal", "refusal": "no"},
                {"type": "output_text", "text": ""},
                {"type": "output_text", "text": "last"},
                {"type": "refusal", "refusal": " again "}
            ]),
        ),
    ] {
        let response = AnnotatedLlmResponse {
            message: message.map(|value| serde_json::from_value(value).unwrap()),
            ..Default::default()
        };
        for surface in [
            ProviderSurface::OpenAIChat,
            ProviderSurface::OpenAIResponses,
        ] {
            let rendered = render_response(surface, &response).unwrap();
            if surface == ProviderSurface::OpenAIChat {
                let message = &rendered["choices"][0]["message"];
                assert_eq!(message.get("content"), chat_text.map(Json::from).as_ref());
                assert_eq!(
                    message.get("refusal"),
                    chat_refusal.map(Json::from).as_ref()
                );
            } else if response_parts.as_array().unwrap().is_empty() {
                assert_eq!(rendered["output"], json!([]));
            } else {
                assert_eq!(rendered["output"][0]["content"], response_parts);
            }
            let decoded = response_codec(surface).decode_response(&rendered).unwrap();
            assert_eq!(
                decoded
                    .output()
                    .parts()
                    .any(|part| matches!(part.kind, ContentKind::Refusal(_))),
                chat_refusal.is_some()
            );
        }
    }
}

#[test]
fn request_projections_accept_sanitized_builtin_decoder_annotations() {
    let cases = [
        (
            ProviderSurface::OpenAIChat,
            json!({
                "model": "trusted-model",
                "messages": [{"role": "user", "content": "[REDACTED]"}],
                "max_tokens": 8,
            }),
        ),
        (
            ProviderSurface::OpenAIResponses,
            json!({
                "model": "trusted-model",
                "input": "[REDACTED]",
                "max_output_tokens": 8,
            }),
        ),
        (
            ProviderSurface::AnthropicMessages,
            json!({
                "model": "trusted-model",
                "messages": [{"role": "user", "content": "[REDACTED]"}],
                "max_tokens": 8,
            }),
        ),
        (
            ProviderSurface::OCIGenAI,
            json!({
                "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "[REDACTED]"}]}],
            }),
        ),
        (
            ProviderSurface::GeminiGenerateContent,
            json!({
                "contents": [{"role": "user", "parts": [{"text": "[REDACTED]"}]}],
                "generationConfig": {"maxOutputTokens": 8},
            }),
        ),
    ];

    for (surface, content) in cases {
        let request = LlmRequest {
            headers: Map::new(),
            content,
        };
        let mut annotated = request_codec(surface).decode(&request).unwrap();
        annotated.api_specific = None;
        annotated.extra.clear();
        let rendered = render_request(surface, &annotated)
            .unwrap_or_else(|| panic!("request projection failed for {surface:?}"));
        request_codec(surface)
            .decode(&rendered)
            .unwrap_or_else(|error| {
                panic!("request projection was invalid for {surface:?}: {error}")
            });
    }
}

#[test]
fn oci_cohere_request_projections_use_format_specific_templates() {
    for api_format in ["COHERE", "COHEREV2"] {
        let request: AnnotatedLlmRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "[REDACTED]"}],
            "params": {"max_tokens": 8},
            "api_specific": {
                "api": "oci_genai",
                "api_format": api_format,
            },
        }))
        .unwrap();

        let rendered = render_request(ProviderSurface::OCIGenAI, &request)
            .unwrap_or_else(|| panic!("{api_format} request projection failed"));
        assert!(rendered.headers.is_empty());
        assert_eq!(rendered.content["apiFormat"], api_format);

        match api_format {
            "COHERE" => {
                assert_eq!(rendered.content["message"], "[REDACTED]");
                assert!(rendered.content.get("messages").is_none());
            }
            "COHEREV2" => {
                assert_eq!(
                    rendered.content["messages"][0]["content"],
                    json!([{"type": "TEXT", "text": "[REDACTED]"}])
                );
                assert!(rendered.content.get("message").is_none());
            }
            _ => unreachable!(),
        }

        let decoded = request_codec(ProviderSurface::OCIGenAI)
            .decode(&rendered)
            .unwrap_or_else(|error| panic!("{api_format} projection was invalid: {error}"));
        assert_eq!(decoded.messages.len(), 1);
    }
}

#[test]
fn oci_enveloped_requests_render_from_sanitized_annotations() {
    let cases = [
        (
            "GENERIC",
            json!({
                "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "SECRET"}]}],
            }),
        ),
        ("COHERE", json!({"message": "SECRET"})),
        (
            "COHEREV2",
            json!({
                "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "SECRET"}]}],
            }),
        ),
    ];
    let sanitizer = TrajectorySanitizer::new(
        "[REDACTED]".into(),
        CustomMarkPayloadPolicy::RedactAllLeaves,
        BTreeMap::new(),
    );

    for (api_format, chat_request) in cases {
        let request = LlmRequest {
            headers: Map::new(),
            content: json!({
                "compartmentId": "SECRET",
                "servingMode": {
                    "servingType": "ON_DEMAND",
                    "modelId": "trusted-model",
                    "future": "SECRET",
                },
                "chatRequest": {
                    "apiFormat": api_format,
                    "future": "SECRET",
                },
            }),
        };
        let mut request = request;
        request.content["chatRequest"]
            .as_object_mut()
            .unwrap()
            .extend(chat_request.as_object().unwrap().clone());
        let annotated = request_codec(ProviderSurface::OCIGenAI)
            .decode(&request)
            .unwrap();
        let sanitized = sanitizer.sanitize_annotated_request(annotated).unwrap();

        let rendered = render_request(ProviderSurface::OCIGenAI, &sanitized)
            .unwrap_or_else(|| panic!("{api_format} enveloped request projection failed"));
        let serialized = serde_json::to_string(&rendered).unwrap();
        assert!(!serialized.contains("SECRET"), "{api_format}: {serialized}");
        assert!(rendered.headers.is_empty());
        assert_eq!(rendered.content["compartmentId"], "[REDACTED]");
        assert_eq!(
            rendered.content["servingMode"],
            json!({"servingType": "ON_DEMAND", "modelId": "trusted-model"})
        );
        assert_eq!(rendered.content["chatRequest"]["apiFormat"], api_format);
        let decoded = request_codec(ProviderSurface::OCIGenAI)
            .decode(&rendered)
            .unwrap_or_else(|error| panic!("{api_format} projection was invalid: {error}"));
        assert_eq!(decoded.model.as_deref(), Some("trusted-model"));
        assert_eq!(decoded.messages.len(), 1);
    }
}
