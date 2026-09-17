// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract-first provider projections for the trajectory-context preset.
//!
//! The projection functions in this module deliberately consume normalized,
//! already-sanitized values only. They never receive the original provider
//! payload, so an unmodeled provider field cannot bypass the trajectory policy.

use serde_json::{Map, Value as Json, json};

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::codec::request::{AnnotatedLlmRequest, ContentPart, MessageContent};
use nemo_relay::codec::resolve::{
    ProviderSurface, detect_request_surface_with_hint, detect_response_surface, request_codec,
    response_codec,
};
use nemo_relay::codec::response::{AnnotatedLlmResponse, FinishReason, ResponseToolCall, Usage};

/// Render a sanitized request through a fresh, provider-specific empty template.
///
/// The built-in request codec owns provider wire details. Starting from an
/// empty template prevents it from carrying original fields that the typed
/// trajectory contract does not model.
pub(super) fn render_request(
    surface: ProviderSurface,
    request: &AnnotatedLlmRequest,
) -> Option<LlmRequest> {
    let mut request = request.clone();
    let template = LlmRequest {
        headers: Map::new(),
        content: request_template(surface, &mut request)?,
    };
    let mut rendered = request_codec(surface).encode(&request, &template).ok()?;
    rendered.headers.clear();
    let provider_hint = match surface {
        ProviderSurface::AnthropicMessages => Some("anthropic.messages"),
        ProviderSurface::OCIGenAI => Some("oci.genai"),
        ProviderSurface::OpenAIChat
        | ProviderSurface::OpenAIResponses
        | ProviderSurface::GeminiGenerateContent
        | ProviderSurface::TypeSafeSystemOne => None,
    };
    (detect_request_surface_with_hint(&rendered.content, provider_hint) == Some(surface))
        .then_some(())?;
    request_codec(surface).decode(&rendered).ok()?;
    Some(rendered)
}

/// Render one minimal, decoder-compatible provider response from normalized data.
///
/// No source-derived field is copied into the result. A source response with
/// multiple choices or candidates is therefore represented by at most the one
/// normalized result that Relay models.
pub(super) fn render_response(
    surface: ProviderSurface,
    response: &AnnotatedLlmResponse,
) -> Option<Json> {
    let rendered = match surface {
        ProviderSurface::OpenAIChat => render_openai_chat_response(response),
        ProviderSurface::OpenAIResponses => render_openai_responses_response(response),
        ProviderSurface::AnthropicMessages => render_anthropic_response(response),
        ProviderSurface::OCIGenAI => render_oci_response(response),
        ProviderSurface::GeminiGenerateContent => render_gemini_response(response),
        ProviderSurface::TypeSafeSystemOne => return None,
    };
    (detect_response_surface(&rendered) == Some(surface)).then_some(())?;
    response_codec(surface).decode_response(&rendered).ok()?;
    Some(rendered)
}

fn request_template(surface: ProviderSurface, request: &mut AnnotatedLlmRequest) -> Option<Json> {
    Some(match surface {
        ProviderSurface::OpenAIChat | ProviderSurface::AnthropicMessages => {
            json!({"messages": []})
        }
        ProviderSurface::OpenAIResponses => json!({"input": []}),
        ProviderSurface::OCIGenAI => oci_request_template(request)?,
        ProviderSurface::GeminiGenerateContent => json!({"contents": []}),
        ProviderSurface::TypeSafeSystemOne => return None,
    })
}

fn oci_request_template(request: &mut AnnotatedLlmRequest) -> Option<Json> {
    use nemo_relay::codec::request::ApiSpecificRequest;

    let (api_format, compartment_id, had_serving_mode) = match request.api_specific.as_ref() {
        Some(ApiSpecificRequest::OCIGenAI {
            compartment_id,
            serving_mode,
            api_format,
        }) => (
            api_format.as_deref().unwrap_or("GENERIC").to_string(),
            compartment_id.clone(),
            serving_mode.is_some(),
        ),
        _ => ("GENERIC".to_string(), None, false),
    };
    let chat_request = match api_format.as_str() {
        "COHERE" => json!({"apiFormat": "COHERE"}),
        "COHEREV2" => json!({"apiFormat": "COHEREV2", "messages": []}),
        _ => json!({"apiFormat": "GENERIC", "messages": []}),
    };
    let needs_envelope = request.model.is_some() || compartment_id.is_some() || had_serving_mode;
    if !needs_envelope {
        return Some(chat_request);
    }

    let model = request.model.clone()?;
    let serving_mode = json!({
        "servingType": "ON_DEMAND",
        "modelId": model,
    });
    match request.api_specific.as_mut() {
        Some(ApiSpecificRequest::OCIGenAI {
            serving_mode: target,
            ..
        }) => *target = Some(serving_mode.clone()),
        None => {
            request.api_specific = Some(ApiSpecificRequest::OCIGenAI {
                compartment_id: None,
                serving_mode: Some(serving_mode.clone()),
                api_format: Some(api_format),
            });
        }
        Some(_) => return None,
    }

    let mut envelope = Map::new();
    if let Some(compartment_id) = compartment_id {
        envelope.insert("compartmentId".into(), Json::String(compartment_id));
    }
    envelope.insert("servingMode".into(), serving_mode);
    envelope.insert("chatRequest".into(), chat_request);
    Some(Json::Object(envelope))
}

fn render_openai_chat_response(response: &AnnotatedLlmResponse) -> Json {
    let mut root = Map::new();
    insert_string(&mut root, "id", response.id.as_deref());
    insert_string(&mut root, "model", response.model.as_deref());

    let mut message = Map::new();
    message.insert("role".into(), Json::String("assistant".into()));
    insert_string(&mut message, "content", response_text(response).as_deref());
    if let Some(calls) = response.tool_calls.as_deref() {
        message.insert("tool_calls".into(), Json::Array(openai_tool_calls(calls)));
    }

    let mut choice = Map::new();
    choice.insert("index".into(), Json::from(0));
    choice.insert("message".into(), Json::Object(message));
    insert_string(
        &mut choice,
        "finish_reason",
        response
            .finish_reason
            .as_ref()
            .and_then(openai_finish_reason),
    );
    root.insert("choices".into(), Json::Array(vec![Json::Object(choice)]));
    insert_usage(
        &mut root,
        "usage",
        response.usage.as_ref(),
        UsageWire::OpenAIChat,
    );
    Json::Object(root)
}

fn render_openai_responses_response(response: &AnnotatedLlmResponse) -> Json {
    let mut root = Map::new();
    insert_string(&mut root, "id", response.id.as_deref());
    insert_string(&mut root, "model", response.model.as_deref());
    insert_string(
        &mut root,
        "status",
        response
            .finish_reason
            .as_ref()
            .and_then(openai_responses_status),
    );

    let mut output = Vec::new();
    if let Some(text) = response_text(response) {
        output.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }));
    }
    output.extend(
        response
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|call| {
                json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": "{}",
                })
            }),
    );
    root.insert("output".into(), Json::Array(output));
    insert_usage(
        &mut root,
        "usage",
        response.usage.as_ref(),
        UsageWire::OpenAIResponses,
    );
    Json::Object(root)
}

fn render_anthropic_response(response: &AnnotatedLlmResponse) -> Json {
    let mut root = Map::new();
    insert_string(&mut root, "id", response.id.as_deref());
    root.insert("type".into(), Json::String("message".into()));
    root.insert("role".into(), Json::String("assistant".into()));
    insert_string(&mut root, "model", response.model.as_deref());
    insert_string(
        &mut root,
        "stop_reason",
        response
            .finish_reason
            .as_ref()
            .and_then(anthropic_stop_reason),
    );

    let mut content = Vec::new();
    if let Some(text) = response_text(response) {
        content.push(json!({"type": "text", "text": text}));
    }
    content.extend(
        response
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|call| {
                json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": {},
                })
            }),
    );
    root.insert("content".into(), Json::Array(content));
    insert_usage(
        &mut root,
        "usage",
        response.usage.as_ref(),
        UsageWire::Anthropic,
    );
    Json::Object(root)
}

fn render_oci_response(response: &AnnotatedLlmResponse) -> Json {
    let mut chat_response = Map::new();
    chat_response.insert("apiFormat".into(), Json::String("GENERIC".into()));

    let mut message = Map::new();
    message.insert("role".into(), Json::String("ASSISTANT".into()));
    if let Some(text) = response_text(response) {
        message.insert(
            "content".into(),
            Json::Array(vec![json!({"type": "TEXT", "text": text})]),
        );
    }
    if let Some(calls) = response.tool_calls.as_deref() {
        message.insert(
            "toolCalls".into(),
            Json::Array(
                calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "name": call.name,
                            "arguments": "{}",
                        })
                    })
                    .collect(),
            ),
        );
    }

    let mut choice = Map::new();
    choice.insert("index".into(), Json::from(0));
    choice.insert("message".into(), Json::Object(message));
    insert_string(
        &mut choice,
        "finishReason",
        response
            .finish_reason
            .as_ref()
            .and_then(openai_finish_reason),
    );
    chat_response.insert("choices".into(), Json::Array(vec![Json::Object(choice)]));
    insert_usage(
        &mut chat_response,
        "usage",
        response.usage.as_ref(),
        UsageWire::Oci,
    );

    let mut root = Map::new();
    insert_string(&mut root, "modelId", response.model.as_deref());
    root.insert("chatResponse".into(), Json::Object(chat_response));
    Json::Object(root)
}

fn render_gemini_response(response: &AnnotatedLlmResponse) -> Json {
    let mut root = Map::new();
    insert_string(&mut root, "responseId", response.id.as_deref());
    insert_string(&mut root, "modelVersion", response.model.as_deref());

    let mut parts = Vec::new();
    if let Some(text) = response_text(response) {
        parts.push(json!({"text": text}));
    }
    parts.extend(
        response
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|call| {
                json!({
                    "functionCall": {"id": call.id, "name": call.name, "args": {}},
                })
            }),
    );

    let mut candidate = Map::new();
    candidate.insert("content".into(), json!({"role": "model", "parts": parts}));
    insert_string(
        &mut candidate,
        "finishReason",
        response
            .finish_reason
            .as_ref()
            .and_then(gemini_finish_reason),
    );
    root.insert(
        "candidates".into(),
        Json::Array(vec![Json::Object(candidate)]),
    );
    insert_usage(
        &mut root,
        "usageMetadata",
        response.usage.as_ref(),
        UsageWire::Gemini,
    );
    Json::Object(root)
}

fn response_text(response: &AnnotatedLlmResponse) -> Option<String> {
    match response.message.as_ref()? {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text, .. } => Some(text.as_str()),
                    ContentPart::Refusal { refusal, .. } => Some(refusal.as_str()),
                    ContentPart::ImageUrl { .. }
                    | ContentPart::Image { .. }
                    | ContentPart::Audio { .. }
                    | ContentPart::File { .. }
                    | ContentPart::ToolUse { .. }
                    | ContentPart::ToolResult { .. }
                    | ContentPart::ProviderNative { .. } => None,
                })
                .collect::<Vec<_>>();
            (!text.is_empty()).then(|| text.join("\n"))
        }
    }
}

fn openai_tool_calls(calls: &[ResponseToolCall]) -> Vec<Json> {
    calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": "{}"},
            })
        })
        .collect()
}

fn insert_string(object: &mut Map<String, Json>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_string(), Json::String(value.to_string()));
    }
}

fn openai_finish_reason(reason: &FinishReason) -> Option<&'static str> {
    match reason {
        FinishReason::Complete => Some("stop"),
        FinishReason::Length => Some("length"),
        FinishReason::ToolUse => Some("tool_calls"),
        FinishReason::ContentFilter => Some("content_filter"),
        FinishReason::Unknown(_) => None,
    }
}

fn openai_responses_status(reason: &FinishReason) -> Option<&'static str> {
    match reason {
        FinishReason::Complete | FinishReason::ToolUse => Some("completed"),
        FinishReason::Length | FinishReason::ContentFilter => Some("incomplete"),
        FinishReason::Unknown(_) => None,
    }
}

fn anthropic_stop_reason(reason: &FinishReason) -> Option<&'static str> {
    match reason {
        FinishReason::Complete => Some("end_turn"),
        FinishReason::Length => Some("max_tokens"),
        FinishReason::ToolUse => Some("tool_use"),
        FinishReason::ContentFilter => Some("refusal"),
        FinishReason::Unknown(_) => None,
    }
}

fn gemini_finish_reason(reason: &FinishReason) -> Option<&'static str> {
    match reason {
        FinishReason::Complete | FinishReason::ToolUse => Some("STOP"),
        FinishReason::Length => Some("MAX_TOKENS"),
        FinishReason::ContentFilter => Some("SAFETY"),
        FinishReason::Unknown(_) => None,
    }
}

#[derive(Clone, Copy)]
enum UsageWire {
    OpenAIChat,
    OpenAIResponses,
    Anthropic,
    Oci,
    Gemini,
}

fn insert_usage(object: &mut Map<String, Json>, key: &str, usage: Option<&Usage>, wire: UsageWire) {
    let Some(usage) = usage else {
        return;
    };
    let mut rendered = Map::new();
    match wire {
        UsageWire::OpenAIChat => {
            insert_u64(&mut rendered, "prompt_tokens", usage.prompt_tokens);
            insert_u64(&mut rendered, "completion_tokens", usage.completion_tokens);
            insert_u64(&mut rendered, "total_tokens", usage.total_tokens);
            if let Some(cache_read) = usage.cache_read_tokens {
                rendered.insert(
                    "prompt_tokens_details".into(),
                    json!({"cached_tokens": cache_read}),
                );
            }
        }
        UsageWire::OpenAIResponses => {
            insert_u64(&mut rendered, "input_tokens", usage.prompt_tokens);
            insert_u64(&mut rendered, "output_tokens", usage.completion_tokens);
            insert_u64(&mut rendered, "total_tokens", usage.total_tokens);
            if let Some(cache_read) = usage.cache_read_tokens {
                rendered.insert(
                    "input_tokens_details".into(),
                    json!({"cached_tokens": cache_read}),
                );
            }
        }
        UsageWire::Anthropic => {
            insert_u64(&mut rendered, "input_tokens", usage.prompt_tokens);
            insert_u64(&mut rendered, "output_tokens", usage.completion_tokens);
            insert_u64(
                &mut rendered,
                "cache_read_input_tokens",
                usage.cache_read_tokens,
            );
            insert_u64(
                &mut rendered,
                "cache_creation_input_tokens",
                usage.cache_write_tokens,
            );
        }
        UsageWire::Oci => {
            insert_u64(&mut rendered, "promptTokens", usage.prompt_tokens);
            insert_u64(&mut rendered, "completionTokens", usage.completion_tokens);
            insert_u64(&mut rendered, "totalTokens", usage.total_tokens);
            if let Some(cache_read) = usage.cache_read_tokens {
                rendered.insert(
                    "promptTokensDetails".into(),
                    json!({"cachedTokens": cache_read}),
                );
            }
        }
        UsageWire::Gemini => {
            insert_u64(&mut rendered, "promptTokenCount", usage.prompt_tokens);
            insert_u64(
                &mut rendered,
                "candidatesTokenCount",
                usage.completion_tokens,
            );
            insert_u64(&mut rendered, "totalTokenCount", usage.total_tokens);
            insert_u64(
                &mut rendered,
                "cachedContentTokenCount",
                usage.cache_read_tokens,
            );
        }
    }
    if let Some(cost) = usage.cost.as_ref() {
        rendered.insert("cost".into(), cost_json(cost));
    }
    object.insert(key.to_string(), Json::Object(rendered));
}

fn insert_u64(object: &mut Map<String, Json>, key: &str, value: Option<u64>) {
    if let Some(value) = value {
        object.insert(key.to_string(), Json::from(value));
    }
}

fn cost_json(cost: &nemo_relay::codec::response::CostEstimate) -> Json {
    let mut rendered = Map::new();
    insert_f64(&mut rendered, "total", cost.total);
    insert_f64(&mut rendered, "input", cost.input);
    insert_f64(&mut rendered, "output", cost.output);
    insert_f64(&mut rendered, "cache_read", cost.cache_read);
    insert_f64(&mut rendered, "cache_write", cost.cache_write);
    rendered.insert("currency".into(), Json::String(cost.currency.clone()));
    Json::Object(rendered)
}

fn insert_f64(object: &mut Map<String, Json>, key: &str, value: Option<f64>) {
    if let Some(value) = value.and_then(serde_json::Number::from_f64) {
        object.insert(key.to_string(), Json::Number(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use nemo_relay::api::llm::LlmRequest;
    use nemo_relay::codec::resolve::response_codec;

    use crate::trajectory::{CustomMarkPayloadPolicy, TrajectorySanitizer};

    const SURFACES: [ProviderSurface; 5] = [
        ProviderSurface::OpenAIChat,
        ProviderSurface::OpenAIResponses,
        ProviderSurface::AnthropicMessages,
        ProviderSurface::OCIGenAI,
        ProviderSurface::GeminiGenerateContent,
    ];

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
}
