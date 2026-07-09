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

#[test]
fn builtin_provider_surface_registry_keeps_request_priority() {
    let surfaces: Vec<_> = BUILTIN_PROVIDER_SURFACES
        .iter()
        .map(|descriptor| descriptor.surface)
        .collect();
    assert_eq!(
        surfaces,
        vec![
            ProviderSurface::OpenAIResponses,
            ProviderSurface::AnthropicMessages,
            ProviderSurface::OpenAIChat,
        ]
    );
}

// ---------------------------------------------------------------------------
// detect_request_surface (priority order, hoisted from adaptive)
// ---------------------------------------------------------------------------

#[test]
fn detect_request_responses_by_input_or_instructions() {
    assert_eq!(
        detect_request_surface(&json!({"input": []})),
        Some(ProviderSurface::OpenAIResponses)
    );
    assert_eq!(
        detect_request_surface(&json!({"instructions": "x"})),
        Some(ProviderSurface::OpenAIResponses)
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
        Some(ProviderSurface::OpenAIChat)
    );
}

#[test]
fn detect_request_priority_responses_then_anthropic_then_chat() {
    // `input` wins even alongside `system` and `messages`.
    assert_eq!(
        detect_request_surface(&json!({"input": [], "system": "x", "messages": []})),
        Some(ProviderSurface::OpenAIResponses)
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
        Some(ProviderSurface::OpenAIChat)
    );
}

#[test]
fn detect_response_responses_by_output_or_output_text() {
    assert_eq!(
        detect_response_surface(&json!({"output": []})),
        Some(ProviderSurface::OpenAIResponses)
    );
    assert_eq!(
        detect_response_surface(&json!({"output_text": "hi"})),
        Some(ProviderSurface::OpenAIResponses)
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
    // Multiple matching shapes are ambiguous: detection and normalization share
    // one exactly-one rule, so normalization must also decline (guards the
    // shared classifier against divergence).
    assert!(normalize_response(&json!({"choices": [], "output": []})).is_none());
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
fn normalize_request_decodes_detected_responses() {
    // `input` selects the OpenAI Responses surface (priority over chat/anthropic).
    let request = req(json!({
        "model": "gpt-4o",
        "input": "Hello, world!"
    }));
    let decoded = normalize_request(&request).expect("responses request decodes");
    assert!(!decoded.messages.is_empty());
}

#[test]
fn normalize_request_none_for_unknown_shape() {
    assert!(normalize_request(&req(json!({"foo": 1}))).is_none());
}

// ---------------------------------------------------------------------------
// detect_request_surface_with_hint (provider hint upgrades the ambiguous shape)
// ---------------------------------------------------------------------------

#[test]
fn hint_none_matches_plain_detection() {
    for body in [
        json!({"input": []}),
        json!({"instructions": "x"}),
        json!({"system": "x", "messages": []}),
        json!({"messages": []}),
        json!({"input": [], "system": "x", "messages": []}),
        json!({}),
        json!({"foo": 1}),
        json!([1, 2, 3]),
    ] {
        assert_eq!(
            detect_request_surface_with_hint(&body, None),
            detect_request_surface(&body),
            "hint=None must match plain detection for {body:?}",
        );
    }
}

#[test]
fn hint_anthropic_upgrades_system_less_messages() {
    assert_eq!(
        detect_request_surface(&json!({"messages": []})),
        Some(ProviderSurface::OpenAIChat)
    );
    for hint in [Some("anthropic"), Some("anthropic.messages")] {
        assert_eq!(
            detect_request_surface_with_hint(&json!({"messages": []}), hint),
            Some(ProviderSurface::AnthropicMessages),
            "messages-only with hint {hint:?} should select Anthropic",
        );
    }
}

#[test]
fn hint_anthropic_descriptor_decodes_system_less_messages() {
    let request = req(json!({
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["END"]
    }));

    assert_eq!(
        request_descriptor(&request.content, None).map(|descriptor| descriptor.surface),
        Some(ProviderSurface::OpenAIChat)
    );
    let descriptor = request_descriptor(&request.content, Some("anthropic"))
        .expect("anthropic hint should select descriptor");
    assert_eq!(descriptor.surface, ProviderSurface::AnthropicMessages);

    let decoded = (descriptor.decode_request)(&request).expect("anthropic request decodes");
    let stop = decoded
        .params
        .as_ref()
        .and_then(|params| params.stop.as_ref())
        .expect("anthropic stop_sequences are normalized");
    assert_eq!(stop, &vec!["END".to_string()]);
    assert!(!decoded.extra.contains_key("stop_sequences"));
}

#[test]
fn normalize_request_with_hint_decodes_system_less_anthropic() {
    let request = req(json!({
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["END"]
    }));

    let decoded_without_hint =
        normalize_request(&request).expect("messages-only request decodes as chat by default");
    assert!(decoded_without_hint.extra.contains_key("stop_sequences"));

    let decoded = normalize_request_with_hint(&request, Some("anthropic.messages"))
        .expect("anthropic-hinted request decodes");
    let stop = decoded
        .params
        .as_ref()
        .and_then(|params| params.stop.as_ref())
        .expect("anthropic stop_sequences are normalized");
    assert_eq!(stop, &vec!["END".to_string()]);
    assert!(!decoded.extra.contains_key("stop_sequences"));
}

#[test]
fn hint_other_or_unknown_provider_stays_chat() {
    for hint in [
        Some("openai"),
        Some("openai.chat"),
        Some("anthropic.count_tokens"),
        Some("anthropic.preview"),
        Some("passthrough"),
        Some("gemini"),
        None,
    ] {
        assert_eq!(
            detect_request_surface_with_hint(&json!({"messages": []}), hint),
            Some(ProviderSurface::OpenAIChat),
            "messages-only with hint {hint:?} should stay OpenAIChat",
        );
    }
}

#[test]
fn hint_never_overrides_strong_signals() {
    assert_eq!(
        detect_request_surface_with_hint(&json!({"input": [], "messages": []}), Some("anthropic")),
        Some(ProviderSurface::OpenAIResponses)
    );
    assert_eq!(
        detect_request_surface_with_hint(
            &json!({"instructions": "x", "messages": []}),
            Some("anthropic")
        ),
        Some(ProviderSurface::OpenAIResponses)
    );
    assert_eq!(
        detect_request_surface_with_hint(
            &json!({"system": "x", "messages": []}),
            Some("anthropic")
        ),
        Some(ProviderSurface::AnthropicMessages)
    );
}

#[test]
fn hint_does_not_classify_non_object_or_keyless() {
    assert_eq!(
        detect_request_surface_with_hint(&json!({}), Some("anthropic")),
        None
    );
    assert_eq!(
        detect_request_surface_with_hint(&json!([1, 2]), Some("anthropic")),
        None
    );
}

// ---------------------------------------------------------------------------
// Provider-codec factory (name<->surface mapping + codec construction)
// ---------------------------------------------------------------------------

const ALL_SURFACES: [ProviderSurface; 3] = [
    ProviderSurface::OpenAIChat,
    ProviderSurface::OpenAIResponses,
    ProviderSurface::AnthropicMessages,
];

#[test]
fn codec_name_round_trips_for_every_surface() {
    for surface in ALL_SURFACES {
        assert_eq!(
            ProviderSurface::from_codec_name(surface.codec_name()),
            Some(surface),
            "codec_name/from_codec_name must round-trip for {surface:?}",
        );
    }
}

#[test]
fn codec_name_uses_canonical_spellings() {
    assert_eq!(ProviderSurface::OpenAIChat.codec_name(), "openai_chat");
    assert_eq!(
        ProviderSurface::OpenAIResponses.codec_name(),
        "openai_responses"
    );
    assert_eq!(
        ProviderSurface::AnthropicMessages.codec_name(),
        "anthropic_messages"
    );
}

#[test]
fn from_codec_name_is_none_for_unknown_names() {
    // Unknown providers, empty strings, and non-canonical spellings do not match.
    assert_eq!(ProviderSurface::from_codec_name("gemini"), None);
    assert_eq!(ProviderSurface::from_codec_name(""), None);
    assert_eq!(ProviderSurface::from_codec_name("OpenAIChat"), None);
}

#[test]
fn supported_codec_names_track_the_builtin_registry() {
    // The advertised set is exactly the built-in surfaces' names, in the same
    // request-detection priority order, so it never drifts from the codecs the
    // factory can actually construct.
    assert_eq!(
        supported_codec_names(),
        vec!["openai_responses", "anthropic_messages", "openai_chat"]
    );
    let from_registry: Vec<_> = BUILTIN_PROVIDER_SURFACES
        .iter()
        .map(|descriptor| descriptor.codec_name)
        .collect();
    assert_eq!(supported_codec_names(), from_registry);
}

#[test]
fn request_codec_decodes_each_surface() {
    let chat = req(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert!(
        !request_codec(ProviderSurface::OpenAIChat)
            .decode(&chat)
            .expect("chat request decodes")
            .messages
            .is_empty()
    );

    let anthropic = req(json!({
        "model": "claude-3-5-sonnet",
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert!(
        !request_codec(ProviderSurface::AnthropicMessages)
            .decode(&anthropic)
            .expect("anthropic request decodes")
            .messages
            .is_empty()
    );

    let responses = req(json!({"model": "gpt-4o", "input": "hi"}));
    assert!(
        !request_codec(ProviderSurface::OpenAIResponses)
            .decode(&responses)
            .expect("responses request decodes")
            .messages
            .is_empty()
    );
}

#[test]
fn response_codec_decodes_each_surface() {
    let chat = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    });
    assert_eq!(
        response_codec(ProviderSurface::OpenAIChat)
            .decode_response(&chat)
            .expect("chat response decodes")
            .response_text(),
        Some("hello")
    );

    let anthropic = json!({
        "type": "message",
        "content": [{"type": "text", "text": "hey"}],
        "stop_reason": "end_turn"
    });
    assert_eq!(
        response_codec(ProviderSurface::AnthropicMessages)
            .decode_response(&anthropic)
            .expect("anthropic response decodes")
            .response_text(),
        Some("hey")
    );

    let responses = json!({"output": [], "output_text": "yo"});
    assert_eq!(
        response_codec(ProviderSurface::OpenAIResponses)
            .decode_response(&responses)
            .expect("responses response decodes")
            .response_text(),
        Some("yo")
    );
}

#[test]
fn streaming_codec_round_trips_through_its_response_codec() {
    // The streaming codec must assemble a payload the matching response codec can
    // decode, proving the factory wires the streaming and response codecs of the
    // same surface together.
    let codec = streaming_codec(ProviderSurface::OpenAIChat);
    let mut collect = codec.collector();
    collect(json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    }))
    .unwrap();
    for part in ["Hello, ", "world", "."] {
        collect(json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": null}]
        }))
        .unwrap();
    }
    collect(json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }))
    .unwrap();

    let assembled = codec.finalizer()();
    let decoded = response_codec(ProviderSurface::OpenAIChat)
        .decode_response(&assembled)
        .expect("assembled stream decodes");
    assert_eq!(decoded.response_text(), Some("Hello, world."));
}

#[test]
fn streaming_codec_constructs_a_usable_codec_for_every_surface() {
    for surface in ALL_SURFACES {
        let assembled = streaming_codec(surface).finalizer()();
        assert!(
            assembled.is_object(),
            "{surface:?} streaming codec finalizes to a JSON object",
        );
    }
}
