// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for provider-surface detection and best-effort normalization.

use super::*;
use crate::api::llm::LlmRequest;
use serde_json::json;

fn req(content: serde_json::Value) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content,
    }
}

// ---------------------------------------------------------------------------
// detect_request_surface (priority order, hoisted from adaptive)
// ---------------------------------------------------------------------------

#[test]
fn detect_request_responses_by_input_or_instructions() {
    assert_eq!(
        detect_request_surface(&json!({"input": []})),
        Some(ProviderSurface::OpenAiResponses)
    );
    assert_eq!(
        detect_request_surface(&json!({"instructions": "x"})),
        Some(ProviderSurface::OpenAiResponses)
    );
}

#[test]
fn detect_request_anthropic_by_system() {
    assert_eq!(
        detect_request_surface(&json!({"system": "x", "messages": []})),
        Some(ProviderSurface::AnthropicMessages)
    );
}

#[test]
fn detect_request_chat_by_messages() {
    assert_eq!(
        detect_request_surface(&json!({"messages": []})),
        Some(ProviderSurface::OpenAiChat)
    );
}

#[test]
fn detect_request_priority_responses_then_anthropic_then_chat() {
    // `input` wins even alongside `system` and `messages`.
    assert_eq!(
        detect_request_surface(&json!({"input": [], "system": "x", "messages": []})),
        Some(ProviderSurface::OpenAiResponses)
    );
    // `system` wins over `messages` (Anthropic carries both).
    assert_eq!(
        detect_request_surface(&json!({"system": "x", "messages": []})),
        Some(ProviderSurface::AnthropicMessages)
    );
}

#[test]
fn detect_request_none_for_unknown_or_non_object() {
    assert_eq!(detect_request_surface(&json!({})), None);
    assert_eq!(detect_request_surface(&json!({"foo": 1})), None);
    assert_eq!(detect_request_surface(&json!([1, 2, 3])), None);
    assert_eq!(detect_request_surface(&json!("string")), None);
}

// ---------------------------------------------------------------------------
// detect_response_surface (strict; ambiguity -> None)
// ---------------------------------------------------------------------------

#[test]
fn detect_response_chat_by_choices() {
    assert_eq!(
        detect_response_surface(&json!({"choices": []})),
        Some(ProviderSurface::OpenAiChat)
    );
}

#[test]
fn detect_response_responses_by_output_or_output_text() {
    assert_eq!(
        detect_response_surface(&json!({"output": []})),
        Some(ProviderSurface::OpenAiResponses)
    );
    assert_eq!(
        detect_response_surface(&json!({"output_text": "hi"})),
        Some(ProviderSurface::OpenAiResponses)
    );
}

#[test]
fn detect_response_output_text_must_be_string() {
    // A non-string `output_text` (null/object) is not a Responses match.
    assert_eq!(detect_response_surface(&json!({"output_text": null})), None);
    assert_eq!(
        detect_response_surface(&json!({"output_text": {"nested": 1}})),
        None
    );
}

#[test]
fn detect_response_anthropic_by_type_message_and_content() {
    assert_eq!(
        detect_response_surface(&json!({"type": "message", "content": []})),
        Some(ProviderSurface::AnthropicMessages)
    );
}

#[test]
fn detect_response_none_for_empty_object_the_decode_trap() {
    // The built-in codecs decode `{}` successfully, so detection must NOT rely
    // on decode success: an empty object classifies to None.
    assert_eq!(detect_response_surface(&json!({})), None);
}

#[test]
fn detect_response_none_for_ambiguous_choices_and_output() {
    assert_eq!(
        detect_response_surface(&json!({"choices": [], "output": []})),
        None
    );
}

#[test]
fn detect_response_none_for_partial_anthropic() {
    // `type == "message"` without a content array does not classify.
    assert_eq!(detect_response_surface(&json!({"type": "message"})), None);
    // A content array without `type == "message"` does not classify.
    assert_eq!(detect_response_surface(&json!({"content": []})), None);
}

#[test]
fn detect_response_none_for_non_object() {
    assert_eq!(detect_response_surface(&json!([1, 2])), None);
}

// ---------------------------------------------------------------------------
// normalize_response (detect -> decode, fail-open)
// ---------------------------------------------------------------------------

#[test]
fn normalize_response_decodes_detected_chat() {
    let raw = json!({
        "id": "r1",
        "model": "gpt-4o",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    });
    let decoded = normalize_response(&raw).expect("chat response decodes");
    assert_eq!(decoded.response_text(), Some("hello"));
}

#[test]
fn normalize_response_decodes_detected_responses_output_text() {
    // Top-level `output_text` (the codec extension) detects + decodes as Responses.
    let raw = json!({
        "model": "gpt-4o",
        "output": [],
        "output_text": "hi there"
    });
    let decoded = normalize_response(&raw).expect("responses output_text decodes");
    assert_eq!(decoded.response_text(), Some("hi there"));
}

#[test]
fn normalize_response_decodes_detected_anthropic() {
    let raw = json!({
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn"
    });
    let decoded = normalize_response(&raw).expect("anthropic response decodes");
    assert_eq!(decoded.response_text(), Some("hi"));
}

#[test]
fn normalize_response_none_for_unrecognized_shape() {
    assert!(normalize_response(&json!({"foo": 1})).is_none());
    // Ambiguous/empty objects do not classify, so they do not decode.
    assert!(normalize_response(&json!({})).is_none());
}

// ---------------------------------------------------------------------------
// normalize_request (detect -> decode, fail-open)
// ---------------------------------------------------------------------------

#[test]
fn normalize_request_decodes_detected_chat() {
    let request = req(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let decoded = normalize_request(&request).expect("chat request decodes");
    assert!(!decoded.messages.is_empty());
}

#[test]
fn normalize_request_decodes_detected_anthropic() {
    // `system` selects the Anthropic surface (priority over `messages`).
    let request = req(json!({
        "model": "claude-3-5-sonnet",
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let decoded = normalize_request(&request).expect("anthropic request decodes");
    assert!(!decoded.messages.is_empty());
}

#[test]
fn normalize_request_none_for_unknown_shape() {
    assert!(normalize_request(&req(json!({"foo": 1}))).is_none());
}
