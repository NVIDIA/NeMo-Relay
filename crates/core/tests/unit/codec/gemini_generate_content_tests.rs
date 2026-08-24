// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for GeminiGenerateContentCodec in the NeMo Relay core crate.

use super::*;
use serde_json::json;

use super::super::request::{ContentPart, Message, MessageContent, ToolDefinition};
use super::super::response::FinishReason;
use super::super::streaming::StreamingCodec;

use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};
use crate::codec::traits::{LlmCodec, LlmResponseCodec};

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

fn make_request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content,
    }
}

// ===================================================================
// codec_identity
// ===================================================================

#[test]
fn test_codec_identity_is_gemini_builtin() {
    let codec = GeminiGenerateContentCodec;
    assert_eq!(
        LlmCodec::codec_identity(&codec),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent),
        "GeminiGenerateContentCodec must not return Opaque; PII sanitization depends on a known identity"
    );
}

#[test]
fn test_response_codec_identity_is_gemini_builtin() {
    let codec = GeminiGenerateContentCodec;
    assert_eq!(
        <GeminiGenerateContentCodec as LlmResponseCodec>::codec_identity(&codec),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent),
        "GeminiGenerateContentCodec response codec must not return Opaque"
    );
}

#[test]
fn test_gemini_private_helpers_reject_ambiguous_provider_shapes() {
    assert_eq!(map_finish_reason(None, false), None);
    assert_eq!(map_finish_reason(None, true), Some(FinishReason::ToolUse));
    assert_eq!(
        map_finish_reason(Some("STOP"), false),
        Some(FinishReason::Complete)
    );
    assert_eq!(
        map_finish_reason(Some("MAX_TOKENS"), true),
        Some(FinishReason::Length)
    );
    assert_eq!(
        map_finish_reason(Some("SAFETY"), true),
        Some(FinishReason::ContentFilter)
    );
    assert_eq!(
        map_finish_reason(Some("FUTURE_REASON"), false),
        Some(FinishReason::Unknown("FUTURE_REASON".into()))
    );
    assert_eq!(
        map_prompt_block_reason(Some("IMAGE_SAFETY")),
        Some(FinishReason::ContentFilter)
    );

    let object = json!({"tool_call_id": "call-1", "id": "part-1"});
    let object = object.as_object().unwrap();
    assert_eq!(extract_tool_call_id(object).unwrap(), "call-1");
    assert_eq!(
        parse_optional_id(object, "part").unwrap().as_deref(),
        Some("part-1")
    );
    for value in [
        json!({}),
        json!({"tool_call_id": ""}),
        json!({"tool_call_id": 1}),
    ] {
        assert!(extract_tool_call_id(value.as_object().unwrap()).is_err());
    }
    for value in [json!({"id": ""}), json!({"id": 1})] {
        assert!(parse_optional_id(value.as_object().unwrap(), "part").is_err());
    }

    let ambiguous = json!({"text": "hello", "functionCall": {"name": "tool"}});
    assert!(
        validate_single_gemini_part_data_field(ambiguous.as_object().unwrap(), "content").is_err()
    );
    assert!(gemini_parts_to_message_content(&[json!("not-an-object")], "content").is_err());
}

#[test]
fn test_gemini_response_helpers_cover_prompt_feedback_and_provider_reasons() {
    for reason in ["", "FINISH_REASON_UNSPECIFIED"] {
        assert_eq!(
            map_finish_reason(Some(reason), true),
            Some(FinishReason::ToolUse)
        );
    }
    assert_eq!(
        map_finish_reason(Some("TOOL_CODE"), false),
        Some(FinishReason::ToolUse)
    );
    for reason in [
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "LANGUAGE",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "ESCALATION",
    ] {
        assert_eq!(
            map_finish_reason(Some(reason), false),
            Some(FinishReason::ContentFilter)
        );
    }
    assert_eq!(map_prompt_block_reason(None), None);
    assert_eq!(map_prompt_block_reason(Some("")), None);
    assert_eq!(
        map_prompt_block_reason(Some("future")),
        Some(FinishReason::Unknown("future".into()))
    );

    let response = json!({"candidates": []});
    assert!(detect_gemini_response(response.as_object().unwrap()));
    let blocked = json!({"promptFeedback": {"blockReason": "SAFETY"}});
    assert!(detect_gemini_response(blocked.as_object().unwrap()));
    let unrelated = json!({"promptFeedback": {"other": true}});
    assert!(!detect_gemini_response(unrelated.as_object().unwrap()));
    assert_eq!(
        prompt_feedback_block_reason(blocked.as_object().unwrap()).unwrap(),
        Some("SAFETY")
    );
    for invalid in [
        json!({"promptFeedback": 1}),
        json!({"promptFeedback": {"blockReason": 1}}),
    ] {
        assert!(prompt_feedback_block_reason(invalid.as_object().unwrap()).is_err());
    }
}

#[test]
fn test_gemini_part_helpers_distinguish_provider_data_fields() {
    assert!(is_gemini_part_data_key("text"));
    assert!(is_gemini_part_data_key("functionResponse"));
    assert!(!is_gemini_part_data_key("thought"));

    let metadata_only = json!({"thought": true});
    assert!(gemini_part_data_keys(metadata_only.as_object().unwrap()).is_empty());
    assert_eq!(
        validate_single_gemini_part_data_field(metadata_only.as_object().unwrap(), "part").unwrap(),
        None
    );
    let text = json!({"text": "hello", "thought": true});
    assert_eq!(
        gemini_part_data_keys(text.as_object().unwrap()),
        vec!["text"]
    );
    assert_eq!(
        validate_single_gemini_part_data_field(text.as_object().unwrap(), "part").unwrap(),
        Some("text")
    );
}

#[test]
fn test_gemini_content_and_function_match_helpers_cover_fallbacks() {
    assert_eq!(extract_content_text(&json!("plain")), "plain");
    assert_eq!(
        extract_content_text(&json!([
            {"type": "text", "text": "first"},
            {"type": "provider_native", "text": "ignored"},
            {"text": "second"}
        ])),
        "first\nsecond"
    );
    assert_eq!(extract_content_text(&json!({"text": "not-content"})), "");
    assert!(json_f64(f64::NAN, "temperature").is_err());

    let entries = vec![(3, Some("call-1"), "first"), (7, None, "same")];
    let mut consumed = std::collections::HashSet::new();
    assert_eq!(
        matching_function_call_entry(&entries, &consumed, Some("call-1"), "ignored"),
        Some((3, Some("call-1".into())))
    );
    consumed.insert(3);
    assert_eq!(
        matching_function_call_entry(&entries, &consumed, None, "same"),
        Some((7, None))
    );
    assert_eq!(
        matching_function_call_entry(&entries, &consumed, Some("unknown"), "other"),
        None
    );
}

#[test]
fn test_gemini_streaming_state_rejects_candidate_and_metadata_inconsistencies() {
    let mut state = GeminiGenerateContentStreamingState::default();
    assert!(
        state
            .observe(&json!({"candidates": [{"index": 1}]}))
            .is_err()
    );
    assert!(
        state
            .observe(&json!({"candidates": [{"index": 0}, {"index": 0}]}))
            .is_err()
    );
    assert!(state.observe(&json!({"modelVersion": 1})).is_err());
    assert!(state.observe(&json!({"responseId": 1})).is_err());

    state
        .observe(&json!({
            "candidates": [{"index": 0, "content": {"parts": [{"text": "first"}]}}],
            "modelVersion": "gemini-test",
            "responseId": "response-1"
        }))
        .unwrap();
    assert!(
        state
            .observe(&json!({"candidates": [{"index": 2}]}))
            .is_err()
    );
}

#[test]
fn test_gemini_missing_content_run_rebuilds_single_messages_only() {
    let user = json!({"role": "user", "content": "added"});
    assert_eq!(
        patch_gemini_missing_content_run(&[&user], &std::collections::HashMap::new()).unwrap(),
        Some(json!({"role": "user", "parts": [{"text": "added"}]}))
    );
    assert_eq!(
        patch_gemini_missing_content_run(&[], &std::collections::HashMap::new()).unwrap(),
        None
    );
    assert!(
        patch_gemini_missing_content_run(
            &[&user, &json!({"role": "assistant", "content": "second"})],
            &std::collections::HashMap::new(),
        )
        .is_err()
    );
}

#[test]
fn test_gemini_serialization_and_generation_decoders_handle_edge_values() {
    let mut object = serde_json::Map::new();
    insert_serialized(
        &mut object,
        "values",
        &vec![String::from("one"), String::from("two")],
        "test values",
    )
    .unwrap();
    assert_eq!(object["values"], json!(["one", "two"]));

    let config = json!({"generationConfig": {
        "temperature": 0.25,
        "topP": 0.5,
        "maxOutputTokens": 7,
        "stopSequences": ["stop"]
    }});
    let params = decode_gemini_generation_params(config.as_object().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(params.temperature, Some(0.25));
    assert_eq!(params.top_p, Some(0.5));
    assert_eq!(params.max_tokens, Some(7));
    assert_eq!(params.stop, Some(vec!["stop".into()]));
    let invalid = json!({"generationConfig": {"topP": "bad"}});
    assert!(decode_gemini_generation_params(invalid.as_object().unwrap()).is_err());
}

#[test]
fn test_gemini_generation_param_patching_preserves_native_fields() {
    let mut request = json!({"generationConfig": {
        "temperature": 0.9,
        "topP": 0.8,
        "maxOutputTokens": 99,
        "stopSequences": ["old"],
        "responseMimeType": "application/json"
    }})
    .as_object()
    .unwrap()
    .clone();
    patch_gemini_params(
        &mut request,
        Some(&GenerationParams {
            temperature: Some(0.2),
            top_p: None,
            max_tokens: Some(12),
            stop: Some(vec!["new".into()]),
        }),
    )
    .unwrap();
    assert_eq!(
        request["generationConfig"],
        json!({
            "temperature": 0.2,
            "maxOutputTokens": 12,
            "stopSequences": ["new"],
            "responseMimeType": "application/json"
        })
    );
    patch_gemini_params(&mut request, None).unwrap();
    assert_eq!(
        request["generationConfig"],
        json!({"responseMimeType": "application/json"})
    );
}

#[test]
fn test_gemini_tool_patching_preserves_native_group_order_and_siblings() {
    let function = ToolDefinition::Function {
        function: FunctionDefinition {
            name: "weather".into(),
            description: Some("lookup".into()),
            parameters: None,
            strict: None,
            extra: Default::default(),
        },
        extra: Default::default(),
    };
    let native = ToolDefinition::ProviderNative {
        provider: GEMINI_PROVIDER.into(),
        kind: "googleSearch".into(),
        value: json!({"googleSearch": {"mode": "dynamic"}}),
    };
    let code_execution = ToolDefinition::ProviderNative {
        provider: GEMINI_PROVIDER.into(),
        kind: "codeExecution".into(),
        value: json!({"codeExecution": {}}),
    };
    let mut request = json!({"tools": [
        {"googleSearch": {"mode": "old"}},
        {"functionDeclarations": [{"name": "old"}], "codeExecution": {}}
    ]})
    .as_object()
    .unwrap()
    .clone();
    patch_gemini_tools(&mut request, Some(&vec![function, native, code_execution])).unwrap();
    assert_eq!(
        request["tools"],
        json!([
            {"googleSearch": {"mode": "dynamic"}},
            {"functionDeclarations": [{"name": "weather", "description": "lookup"}], "codeExecution": {}}
        ])
    );

    let duplicate = json!({"tools": [
        {"functionDeclarations": []}, {"functionDeclarations": []}
    ]})
    .as_object()
    .unwrap()
    .clone();
    assert!(patch_gemini_tools(&mut duplicate.clone(), None).is_err());
}

#[test]
fn test_gemini_content_part_conversion_preserves_metadata_and_native_parts() {
    let parts = vec![
        json!({"text": "visible", "thoughtSignature": "sig"}),
        json!({"inlineData": {"mimeType": "text/plain", "data": "aGk="}}),
        json!({"thought": true, "text": "hidden"}),
    ];
    let content = gemini_parts_to_message_content(&parts, "request").unwrap();
    assert_eq!(
        content,
        Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "visible".into(),
                extra: serde_json::Map::from_iter([("thoughtSignature".into(), json!("sig"))]),
            },
            ContentPart::ProviderNative {
                provider: GEMINI_PROVIDER.into(),
                kind: "inlineData".into(),
                value: json!({"inlineData": {"mimeType": "text/plain", "data": "aGk="}}),
            },
        ]))
    );
    assert_eq!(
        gemini_parts_to_message_content(&[json!({"thought": true})], "request").unwrap(),
        None
    );
    assert!(gemini_parts_to_message_content(&[json!({"text": 1})], "request").is_err());
}

#[test]
fn test_gemini_normalized_content_conversion_handles_text_native_and_invalid_parts() {
    let content = json!([
        {"type": "text", "text": "hello", "thoughtSignature": "sig"},
        {"type": "provider_native", "provider": "gemini", "kind": "inlineData", "value": {"inlineData": {"mimeType": "text/plain", "data": "aGk="}}}
    ]);
    let (parts, is_parts) = gemini_content_parts_from_normalized(&content).unwrap();
    assert!(is_parts);
    assert_eq!(
        parts,
        vec![
            json!({"text": "hello", "thoughtSignature": "sig"}),
            json!({"inlineData": {"mimeType": "text/plain", "data": "aGk="}}),
        ]
    );
    assert_eq!(
        gemini_content_parts_from_normalized(&Json::Null).unwrap(),
        (Vec::new(), false)
    );
    assert!(gemini_content_parts_from_normalized(&json!([{"type": "text"}])).is_err());
    assert!(gemini_content_parts_from_normalized(&json!([{"type": "image"}])).is_err());
    assert!(gemini_content_parts_from_normalized(&json!(42)).is_err());
}

#[test]
fn test_gemini_function_response_content_preserves_nested_provider_parts() {
    let response = json!({
        "response": {"status": "ok"},
        "parts": [
            {"text": "detail"},
            {"inlineData": {"mimeType": "text/plain", "data": "aGk="}}
        ]
    });
    assert_eq!(
        gemini_function_response_to_message_content(&response).unwrap(),
        MessageContent::Parts(vec![
            ContentPart::Text {
                text: "{\"status\":\"ok\"}".into(),
                extra: Default::default(),
            },
            ContentPart::ProviderNative {
                provider: GEMINI_PROVIDER.into(),
                kind: "text".into(),
                value: json!({"text": "detail"}),
            },
            ContentPart::ProviderNative {
                provider: GEMINI_PROVIDER.into(),
                kind: "inlineData".into(),
                value: json!({"inlineData": {"mimeType": "text/plain", "data": "aGk="}}),
            },
        ])
    );
    assert!(
        gemini_function_response_to_message_content(&json!({"response": {}, "parts": {}})).is_err()
    );
    assert!(
        gemini_function_response_to_message_content(
            &json!({"response": {}, "parts": [{"functionCall": {}}]})
        )
        .is_err()
    );
}

#[test]
fn test_gemini_response_function_call_extraction_validates_shape_and_id_fallback() {
    let calls = extract_parts_tool_calls(&[
        json!({"text": "ignored"}),
        json!({"functionCall": {"name": "weather", "args": {"city": "Boston"}}}),
    ])
    .unwrap()
    .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "weather");
    assert_eq!(calls[0].name, "weather");
    assert_eq!(calls[0].arguments, json!({"city": "Boston"}));

    for part in [
        json!({"functionCall": null}),
        json!({"functionCall": {"name": ""}}),
        json!({"functionCall": {"name": 1}}),
        json!({"functionCall": {}}),
        json!({"functionCall": {"name": "fn", "id": ""}}),
        json!({"functionCall": {"name": "fn", "args": []}}),
    ] {
        assert!(extract_parts_tool_calls(&[part]).is_err());
    }
}

#[test]
fn test_gemini_function_call_decoders_reject_duplicate_effective_ids() {
    let duplicate_idless_calls = [
        json!({"functionCall": {"name": "search", "args": {"q": "dogs"}}}),
        json!({"functionCall": {"name": "search", "args": {"q": "cats"}}}),
    ];
    assert!(extract_parts_tool_calls(&duplicate_idless_calls).is_err());
    assert!(
        gemini_function_call_messages(
            None,
            &[&duplicate_idless_calls[0], &duplicate_idless_calls[1]]
        )
        .is_err()
    );

    let duplicate_explicit_calls = [
        json!({"functionCall": {"id": "call_1", "name": "search", "args": {"q": "dogs"}}}),
        json!({"functionCall": {"id": "call_1", "name": "search", "args": {"q": "cats"}}}),
    ];
    assert!(extract_parts_tool_calls(&duplicate_explicit_calls).is_err());

    let duplicate_idless_responses = [
        json!({"functionResponse": {"name": "search", "response": {"result": "dogs"}}}),
        json!({"functionResponse": {"name": "search", "response": {"result": "cats"}}}),
    ];
    assert!(
        gemini_function_response_messages(
            &duplicate_idless_responses,
            &[
                &duplicate_idless_responses[0],
                &duplicate_idless_responses[1]
            ],
        )
        .is_err()
    );
}

#[test]
fn test_gemini_provider_native_part_validation_and_tool_payload_conversion() {
    let valid = json!({
        "type": "provider_native",
        "provider": "gemini",
        "kind": "inlineData",
        "value": {"inlineData": {"mimeType": "text/plain", "data": "aGk="}}
    });
    assert_eq!(
        provider_native_gemini_part_value(valid.as_object().unwrap()).unwrap(),
        json!({"inlineData": {"mimeType": "text/plain", "data": "aGk="}})
    );
    for invalid in [
        json!({"type": "provider_native", "kind": "x", "value": {}}),
        json!({"type": "provider_native", "provider": "other", "kind": "x", "value": {}}),
        json!({"type": "provider_native", "provider": "gemini", "kind": "", "value": {}}),
        json!({"type": "provider_native", "provider": "gemini", "kind": "x"}),
        json!({"type": "provider_native", "provider": "gemini", "kind": "x", "value": []}),
        json!({"type": "provider_native", "provider": "gemini", "kind": "x", "value": {"functionCall": {}}}),
        json!({"type": "provider_native", "provider": "gemini", "kind": "x", "value": {"text": 1}}),
    ] {
        assert!(provider_native_gemini_part_value(invalid.as_object().unwrap()).is_err());
    }

    let payload = function_response_payload_from_tool_content(&json!([
        {"type": "text", "text": "first"},
        {"text": "second"},
        valid,
    ]))
    .unwrap();
    assert_eq!(payload.response, json!({"output": "first\nsecond"}));
    assert_eq!(
        payload.parts.unwrap(),
        vec![json!({"inlineData": {"mimeType": "text/plain", "data": "aGk="}})]
    );
    assert_eq!(
        function_response_payload_from_tool_content(&json!("not json"))
            .unwrap()
            .response,
        json!({"output": "not json"})
    );
    assert!(function_response_payload_from_tool_content(&json!([{"type": "provider_native", "provider": "gemini", "kind": "x", "value": {"functionResponse": {}}}])).is_err());
}

#[test]
fn test_gemini_normalized_message_conversion_handles_roles_tools_and_empty_content() {
    let calls = std::collections::HashMap::from([("call-1".into(), "weather".into())]);
    assert_eq!(
        normalized_to_gemini_content(&json!({"role": "system", "content": "rules"}), &calls)
            .unwrap(),
        None
    );
    assert_eq!(
        normalized_to_gemini_content(&json!({"role": "assistant", "content": null}), &calls)
            .unwrap(),
        Some(json!({"role": "model", "parts": [{"text": ""}]}))
    );
    assert_eq!(
        normalized_to_gemini_content(
            &json!({"role": "tool", "tool_call_id": "call-1", "content": "{\"ok\":true}"}),
            &calls,
        )
        .unwrap(),
        Some(
            json!({"role": "user", "parts": [{"functionResponse": {"id": "call-1", "name": "weather", "response": {"ok": true}}}]})
        )
    );
    for message in [
        json!([]),
        json!({"content": "missing role"}),
        json!({"role": "developer", "content": "unsupported"}),
        json!({"role": "tool", "content": "missing call id"}),
    ] {
        assert!(normalized_to_gemini_content(&message, &calls).is_err());
    }
}

#[test]
fn test_gemini_system_instruction_patching_preserves_thoughts_and_rejects_uneditable_shapes() {
    let mut request = json!({"systemInstruction": {
        "role": "system",
        "parts": [{"thought": true, "text": "hidden"}, {"text": "old", "thoughtSignature": "sig"}]
    }})
    .as_object()
    .unwrap()
    .clone();
    let annotated = json!({"role": "system", "content": "new rules"});
    patch_gemini_system_instruction(&mut request, &[&annotated], &[]).unwrap();
    assert_eq!(
        request["systemInstruction"]["parts"],
        json!([{ "thought": true, "text": "hidden" }, {"text": "new rules", "thoughtSignature": "sig"}])
    );
    patch_gemini_system_instruction(&mut request, &[], &[&annotated]).unwrap();
    assert!(request.get("systemInstruction").is_none());

    let multiple = vec![json!({"text": "one"}), json!({"text": "two"})];
    assert!(validate_editable_gemini_system_parts(&multiple).is_err());
    assert!(validate_editable_gemini_system_parts(&[json!({"inlineData": {}})]).is_err());
    assert_eq!(
        rebuild_gemini_system_parts(&[json!({"thought": true, "text": "hidden"})], "new".into()),
        vec![
            json!({"thought": true, "text": "hidden"}),
            json!({"text": "new"})
        ]
    );
}

#[test]
fn test_gemini_changed_content_patching_preserves_native_parts_and_updates_tool_responses() {
    let original = json!({
        "role": "model",
        "parts": [
            {"thought": true, "text": "reasoning"},
            {"text": "old", "thoughtSignature": "sig"},
            {"functionCall": {"id": "call-1", "name": "weather", "args": {"city": "old"}}}
        ]
    });
    let assistant = json!({
        "role": "assistant", "content": "new",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"new\"}"}}]
    });
    assert_eq!(
        patch_changed_gemini_content(&original, &[&assistant], &std::collections::HashMap::new())
            .unwrap(),
        Some(json!({"role": "model", "parts": [
            {"thought": true, "text": "reasoning"},
            {"text": "new", "thoughtSignature": "sig"},
            {"functionCall": {"id": "call-1", "name": "weather", "args": {"city": "new"}}}
        ]}))
    );

    let original_tool = json!({"role": "user", "parts": [
        {"text": "keep"},
        {"functionResponse": {"id": "call-1", "name": "weather", "response": {"old": true}}}
    ]});
    let tool = json!({"role": "tool", "tool_call_id": "call-1", "content": "{\"new\":true}"});
    assert_eq!(
        patch_changed_gemini_content(&original_tool, &[&tool], &std::collections::HashMap::new())
            .unwrap(),
        Some(json!({"role": "user", "parts": [
            {"text": "keep"},
            {"functionResponse": {"id": "call-1", "name": "weather", "response": {"new": true}}}
        ]}))
    );
}

#[test]
fn test_gemini_streaming_state_merges_text_and_retains_response_metadata() {
    let mut state = GeminiGenerateContentStreamingState::default();
    state
        .observe(&json!({
            "candidates": [{
                "index": 0,
                "content": {"parts": [{"text": "hello"}]},
                "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}]
            }],
            "usageMetadata": {"promptTokenCount": 3},
            "modelVersion": "gemini-2.0",
            "responseId": "response-1"
        }))
        .unwrap();
    state
        .observe(&json!({
            "candidates": [{"index": 0, "content": {"parts": [{"text": " world"}]}, "finishReason": "STOP"}]
        }))
        .unwrap();
    assert_eq!(
        state.finalize(),
        json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hello world"}]},
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}]
            }],
            "usageMetadata": {"promptTokenCount": 3},
            "modelVersion": "gemini-2.0",
            "responseId": "response-1"
        })
    );
}

#[test]
fn test_gemini_function_part_validators_and_tool_call_encoder_cover_invalid_shapes() {
    validate_gemini_function_call_part(&json!({"functionCall": {"name": "weather", "args": {}}}))
        .unwrap();
    validate_gemini_function_response_part(
        &json!({"functionResponse": {"name": "weather", "response": {}}}),
    )
    .unwrap();
    for part in [
        json!({"functionCall": null}),
        json!({"functionCall": {"name": ""}}),
        json!({"functionCall": {"name": "ok", "args": []}}),
    ] {
        assert!(validate_gemini_function_call_part(&part).is_err());
    }
    for part in [
        json!({"functionResponse": null}),
        json!({"functionResponse": {"name": "", "response": {}}}),
        json!({"functionResponse": {"name": "ok"}}),
        json!({"functionResponse": {"name": "ok", "response": []}}),
        json!({"functionResponse": {"name": "ok", "response": {}, "parts": {}}}),
    ] {
        assert!(validate_gemini_function_response_part(&part).is_err());
    }
    let valid = json!({"id": "call-1", "function": {"name": "weather", "arguments": "{\"city\":\"Boston\"}"}});
    assert_eq!(
        tool_call_to_fc_obj(&valid).unwrap(),
        serde_json::Map::from_iter([
            ("name".into(), json!("weather")),
            ("id".into(), json!("call-1")),
            ("args".into(), json!({"city": "Boston"}))
        ])
    );
    for call in [
        json!({"function": {"name": ""}}),
        json!({"id": "", "function": {"name": "ok"}}),
        json!({"id": 1, "function": {"name": "ok"}}),
        json!({"function": {"name": "ok", "arguments": "not-json"}}),
        json!({"function": {"name": "ok", "arguments": "[]"}}),
    ] {
        assert!(tool_call_to_fc_obj(&call).is_err());
    }
}

#[test]
fn test_gemini_function_message_converters_cover_role_mixing_and_plain_messages() {
    let response_part =
        json!({"functionResponse": {"id": "call-1", "name": "weather", "response": {"ok": true}}});
    let call_part = json!({"functionCall": {"id": "call-1", "name": "weather", "args": {}}});
    validate_gemini_content_roles("user", &[&response_part], &[]).unwrap();
    validate_gemini_content_roles("model", &[], &[&call_part]).unwrap();
    assert!(validate_gemini_content_roles("model", &[&response_part], &[]).is_err());
    assert!(validate_gemini_content_roles("user", &[], &[&call_part]).is_err());
    let messages =
        gemini_function_response_messages(std::slice::from_ref(&response_part), &[&response_part])
            .unwrap();
    assert!(matches!(&messages[0], Message::Tool { tool_call_id, .. } if tool_call_id == "call-1"));
    assert!(
        gemini_function_response_messages(
            &[json!({"text": "mixed"}), response_part.clone()],
            &[&response_part]
        )
        .is_err()
    );
    let messages =
        gemini_function_call_messages(Some(MessageContent::Text("before".into())), &[&call_part])
            .unwrap();
    assert!(
        matches!(&messages[0], Message::Assistant { tool_calls: Some(calls), .. } if calls[0].id == "call-1")
    );
    assert!(matches!(
        gemini_plain_message("model", None),
        Message::Assistant { .. }
    ));
    assert!(matches!(
        gemini_plain_message("user", None),
        Message::User { .. }
    ));
}

#[test]
fn test_gemini_streaming_state_rejects_part_and_finish_reason_failures() {
    let mut state = GeminiGenerateContentStreamingState::default();
    for event in [
        json!({"candidates": [{"index": 0, "content": {"parts": [null]}}]}),
        json!({"candidates": [{"index": 0, "content": {"parts": [{"text": 1}]}}]}),
        json!({"candidates": [{"index": 0, "content": {"parts": [{"functionResponse": {}}]}}]}),
        json!({"candidates": [{"index": 0, "finishReason": 1}]}),
    ] {
        assert!(state.observe(&event).is_err());
    }
    let mut multi = GeminiGenerateContentStreamingState::default();
    assert!(multi.observe(&json!({"candidates": [null]})).is_err());
    let mut first = GeminiGenerateContentStreamingState::default();
    first.observe(&json!({"candidates": [{"index": 0, "content": {"parts": [{"thought": true, "text": "thought"}, {"functionCall": {"name": "tool", "args": {}}}]}}]})).unwrap();
    assert_eq!(
        first.finalize()["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn test_gemini_encode_updates_model_and_extra_fields() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "model": "gemini-old",
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "safetySettings": [{"threshold": "BLOCK_NONE"}],
        "cachedContent": "old-cache"
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.model = Some("gemini-new".into());
    annotated.extra.remove("cachedContent");
    annotated.extra.insert("newNative".into(), json!(true));
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(encoded.content["model"], json!("gemini-new"));
    assert!(encoded.content.get("cachedContent").is_none());
    assert_eq!(encoded.content["newNative"], json!(true));

    annotated.model = None;
    let encoded = codec.encode(&annotated, &encoded).unwrap();
    assert!(encoded.content.get("model").is_none());
}

#[test]
fn test_gemini_content_to_messages_decodes_roles_and_rejects_mixed_or_invalid_parts() {
    let messages =
        gemini_content_to_messages(&json!({"parts": [{"text": "default user"}]})).unwrap();
    assert!(
        matches!(&messages[0], Message::User { content: MessageContent::Text(text), .. } if text == "default user")
    );
    let messages = gemini_content_to_messages(
        &json!({"role": "model", "parts": [{"functionCall": {"name": "weather", "args": {}}}]}),
    )
    .unwrap();
    assert!(
        matches!(&messages[0], Message::Assistant { tool_calls: Some(calls), .. } if calls[0].id == "weather")
    );
    for content in [
        Json::Null,
        json!({"role": "developer", "parts": []}),
        json!({"role": 1, "parts": []}),
        json!({"role": "user", "parts": {}}),
        json!({"role": "user", "parts": [null]}),
        json!({"role": "user", "parts": [{"text": 1}]}),
        json!({"role": "user", "parts": [{"functionCall": {"name": "a"}}, {"functionResponse": {"name": "a", "response": {}}}]}),
    ] {
        assert!(gemini_content_to_messages(&content).is_err());
    }
}

#[test]
fn test_gemini_original_part_merging_preserves_thoughts_and_inserts_content() {
    let original = vec![
        json!({"thought": true, "text": "hidden"}),
        json!({"text": "old", "thoughtSignature": "sig"}),
        json!({"functionCall": {"name": "old"}}),
    ];
    let merged = merge_gemini_original_parts(
        &original,
        vec![json!({"text": "new"}), json!({"inlineData": {}})],
        vec![json!({"functionCall": {"name": "new-call"}})],
        false,
    );
    assert_eq!(
        merged,
        vec![
            json!({"thought": true, "text": "hidden"}),
            json!({"text": "new", "thoughtSignature": "sig"}),
            json!({"functionCall": {"name": "new-call"}}),
            json!({"inlineData": {}}),
        ]
    );
    assert_eq!(
        merge_gemini_original_parts(&[], Vec::new(), Vec::new(), true),
        vec![json!({"text": ""})]
    );
    assert_eq!(
        replacement_content_part(
            &json!({"text": "old", "signature": "x"}),
            json!({"text": "new"}),
            false
        ),
        json!({"text": "new", "signature": "x"})
    );
}

#[test]
fn test_gemini_request_decoders_reject_invalid_system_generation_and_tool_shapes() {
    let codec = GeminiGenerateContentCodec;
    for request in [
        json!({"systemInstruction": {"parts": "not-array"}, "contents": []}),
        json!({"systemInstruction": {"role": 1, "parts": []}, "contents": []}),
        json!({"generationConfig": [], "contents": []}),
        json!({"generationConfig": {"maxOutputTokens": -1}, "contents": []}),
        json!({"generationConfig": {"stopSequences": [1]}, "contents": []}),
        json!({"tools": {}, "contents": []}),
        json!({"tools": [null], "contents": []}),
        json!({"tools": [{"functionDeclarations": {}}], "contents": []}),
        json!({"tools": [{"functionDeclarations": [{"description": "missing-name"}]}], "contents": []}),
    ] {
        assert!(codec.decode(&make_request(request)).is_err());
    }
}

#[test]
fn test_gemini_system_and_text_only_validators_reject_remaining_invalid_forms() {
    for value in [
        Json::Null,
        json!({}),
        json!({"parts": [null]}),
        json!({"parts": [{"inlineData": {}}]}),
        json!({"parts": [{"text": 1}]}),
    ] {
        assert!(validate_system_instruction(&value).is_err());
    }
    assert!(reject_non_text_content_parts(&json!([null])).is_err());
    assert!(reject_non_text_content_parts(&json!([{"type": 1, "text": "x"}])).is_err());
    assert!(
        reject_non_text_content_parts(&json!([{"type": "provider_native", "text": "x"}])).is_err()
    );
    assert!(reject_non_text_content_parts(&json!([{"type": "text", "text": 1}])).is_err());
    assert!(reject_non_text_content_parts(&json!([{"type": "text"}])).is_err());
}

#[test]
fn test_gemini_tool_response_patching_handles_new_and_duplicate_call_ids() {
    let existing = vec![
        json!({"text": "keep"}),
        json!({"functionResponse": {"name": "legacy", "response": {"old": true}}}),
    ];
    let new_tool = json!({"role": "tool", "tool_call_id": "new-id", "content": "{\"ok\":true}"});
    assert_eq!(
        patch_gemini_tool_response_content(
            &existing,
            &[&new_tool],
            &std::collections::HashMap::from([("new-id".into(), "new_fn".into())]),
        )
        .unwrap(),
        Some(json!({"role": "user", "parts": [
            {"text": "keep"},
            {"functionResponse": {"id": "new-id", "name": "new_fn", "response": {"ok": true}}}
        ]}))
    );
    let duplicate = json!({"role": "tool", "tool_call_id": "new-id", "content": "{}"});
    assert!(gemini_function_response_updates(&[&new_tool, &duplicate]).is_err());
    assert_eq!(
        function_response_call_id(&json!({"name": "fallback"})),
        Some("fallback")
    );
    assert_eq!(function_response_call_id(&json!({})), None);
}

#[test]
fn test_gemini_patch_non_system_contents_inserts_new_message_and_rejects_invalid_original() {
    let base = [json!({"role": "user", "content": "old"})];
    let annotated = [
        json!({"role": "user", "content": "old"}),
        json!({"role": "assistant", "content": "new"}),
    ];
    let mut obj = serde_json::Map::from_iter([(
        "contents".into(),
        json!([{"role": "user", "parts": [{"text": "old"}]}]),
    )]);
    patch_gemini_non_system_contents(
        &mut obj,
        &[],
        &annotated.iter().collect::<Vec<_>>(),
        &base.iter().collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(obj["contents"].as_array().unwrap().len(), 2);

    let changed = json!({"role": "user", "content": "changed"});
    let mut invalid = serde_json::Map::from_iter([("contents".into(), json!([null]))]);
    assert!(patch_gemini_non_system_contents(&mut invalid, &[], &[&changed], &[&base[0]]).is_err());
}

#[test]
fn test_gemini_function_definition_and_native_tool_decoding_preserve_extra_fields() {
    let mut definitions = Vec::new();
    decode_gemini_tool_group(
        &json!({
            "functionDeclarations": [{
                "name": "weather",
                "description": "lookup",
                "parameters": {"type": "object"},
                "responseJsonSchema": {"type": "string"}
            }],
            "googleSearch": {"mode": "dynamic"}
        }),
        &mut definitions,
    )
    .unwrap();
    assert_eq!(definitions.len(), 2);
    assert!(
        matches!(&definitions[0], ToolDefinition::Function { function, .. } if function.extra.contains_key("responseJsonSchema"))
    );
    assert!(
        matches!(&definitions[1], ToolDefinition::ProviderNative { kind, .. } if kind == "googleSearch")
    );
    assert_eq!(decode_gemini_tools(&serde_json::Map::new()).unwrap(), None);
    assert!(decode_gemini_function_definition(&json!(null)).is_err());
    assert!(decode_gemini_function_definition(&json!({"name": ""})).is_err());
    assert!(decode_gemini_function_definition(&json!({"name": "ok", "description": 1})).is_err());
}

#[test]
fn test_gemini_streaming_default_and_nan_param_patching_cover_remaining_error_paths() {
    let codec = GeminiGenerateContentStreamingCodec::default();
    let mut collect = codec.collector();
    collect(json!({"candidates": [{"index": 0, "content": {"parts": [{"text": "ok"}]}}]})).unwrap();
    assert_eq!(
        codec.finalizer()()["candidates"][0]["content"]["parts"],
        json!([{ "text": "ok" }])
    );
    let mut obj = serde_json::Map::new();
    assert!(
        patch_gemini_params(
            &mut obj,
            Some(&GenerationParams {
                temperature: Some(f64::NAN),
                max_tokens: None,
                top_p: None,
                stop: None
            })
        )
        .is_err()
    );
}

#[test]
fn test_gemini_remaining_content_and_tool_payload_validation_paths() {
    let content =
        gemini_parts_to_message_content(&[json!({"metadata": {"source": "cache"}})], "content")
            .unwrap()
            .unwrap();
    assert!(
        matches!(content, MessageContent::Parts(parts) if matches!(&parts[0], ContentPart::ProviderNative { kind, .. } if kind == "unknown"))
    );

    for content in [json!([null]), json!([{"type": 1, "text": "x"}])] {
        assert!(gemini_content_parts_from_normalized(&content).is_err());
    }

    for content in [
        json!([null]),
        json!([{"type": 1}]),
        json!([{"type": "text", "text": 1}]),
        json!([{"type": "text"}]),
        json!([{"type": "image", "url": "https://example.invalid"}]),
    ] {
        assert!(function_response_payload_from_tool_content(&content).is_err());
    }
}

#[test]
fn test_gemini_remaining_message_and_generation_parameter_branches() {
    let calls = std::collections::HashMap::new();
    assert_eq!(
        normalized_to_gemini_content(
            &json!({"role": "assistant", "content": "answer", "tool_calls": []}),
            &calls,
        )
        .unwrap(),
        Some(json!({"role": "model", "parts": [{"text": "answer"}]}))
    );
    assert!(
        patch_changed_gemini_content(
            &json!({"role": "user", "parts": [{"text": "old"}]}),
            &[&json!({"role": "developer", "content": "new"})],
            &calls,
        )
        .is_err()
    );

    let mut request = serde_json::Map::from_iter([(
        "generationConfig".into(),
        json!({"temperature": 0.2, "topP": 0.8, "maxOutputTokens": 10, "stopSequences": ["stop"]}),
    )]);
    patch_gemini_params(
        &mut request,
        Some(&GenerationParams {
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
        }),
    )
    .unwrap();
    assert!(request.get("generationConfig").is_none());
}

#[test]
fn test_gemini_usage_mapping_handles_absent_and_partial_metadata() {
    assert_eq!(map_usage(None, None), (None, None));
    let (usage, thoughts) = map_usage(
        Some(RawUsageMetadata {
            prompt_token_count: Some(3),
            candidates_token_count: None,
            total_token_count: None,
            cached_content_token_count: Some(2),
            thoughts_token_count: Some(5),
        }),
        None,
    );
    let usage = usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(3));
    assert_eq!(usage.completion_tokens, None);
    assert_eq!(usage.total_tokens, Some(8));
    assert_eq!(usage.cache_read_tokens, Some(2));
    assert_eq!(thoughts, Some(5));

    let (usage, thoughts) = map_usage(
        Some(RawUsageMetadata {
            prompt_token_count: None,
            candidates_token_count: Some(4),
            total_token_count: Some(10),
            cached_content_token_count: None,
            thoughts_token_count: None,
        }),
        Some("unknown-model"),
    );
    let usage = usage.unwrap();
    assert_eq!(usage.completion_tokens, Some(4));
    assert_eq!(usage.total_tokens, Some(10));
    assert_eq!(thoughts, None);
}

#[test]
fn test_gemini_system_instruction_helpers_validate_text_only_parts() {
    let valid = json!({"role": "system", "parts": [{"text": "first"}, {"text": "second"}]});
    validate_system_instruction(&valid).unwrap();
    assert_eq!(
        system_instruction_text(&valid).as_deref(),
        Some("first\nsecond")
    );
    assert_eq!(system_instruction_text(&json!({"parts": []})), None);
    for invalid in [
        json!(null),
        json!({"role": 1, "parts": []}),
        json!({}),
        json!({"parts": 1}),
        json!({"parts": [1]}),
        json!({"parts": [{"functionCall": {}}]}),
        json!({"parts": [{"text": 1}]}),
    ] {
        assert!(validate_system_instruction(&invalid).is_err(), "{invalid}");
    }
}

#[test]
fn test_gemini_content_validation_rejects_invalid_roles_and_function_shapes() {
    let default_role = serde_json::Map::new();
    assert_eq!(gemini_content_role(&default_role).unwrap(), "user");
    for role in [json!("system"), json!(1)] {
        assert!(gemini_content_role(json!({"role": role}).as_object().unwrap()).is_err());
    }
    for parts in [
        vec![json!(1)],
        vec![json!({"text": 1})],
        vec![
            json!({"functionResponse": {"name": "tool", "response": {}}}),
            json!({"functionCall": {"name": "tool"}}),
        ],
    ] {
        assert!(validate_gemini_content_parts(&parts).is_err());
    }
    for part in [
        json!({"functionResponse": {}}),
        json!({"functionResponse": {"name": "tool"}}),
        json!({"functionResponse": {"name": "tool", "response": 1}}),
        json!({"functionCall": {}}),
        json!({"functionCall": {"name": "tool", "args": 1}}),
    ] {
        assert!(gemini_content_to_messages(&json!({"parts": [part]})).is_err());
    }
    assert!(validate_gemini_content_roles("model", &[&json!({})], &[]).is_err());
    assert!(validate_gemini_content_roles("user", &[], &[&json!({})]).is_err());
}

#[test]
fn test_gemini_content_to_messages_maps_default_user_and_model_text() {
    let messages = gemini_content_to_messages(&json!({"parts": [{"text": "hello"}]})).unwrap();
    assert!(matches!(messages.as_slice(), [Message::User { .. }]));
    let messages = gemini_content_to_messages(&json!({
        "role": "model",
        "parts": [{"text": "response"}]
    }))
    .unwrap();
    assert!(matches!(messages.as_slice(), [Message::Assistant { .. }]));
    assert!(gemini_content_to_messages(&json!({"role": "user"})).is_err());
}

#[test]
fn test_gemini_function_response_parts_validate_and_preserve_native_content() {
    assert_eq!(
        validate_gemini_nested_function_response_part(&json!({"inlineData": {}})).unwrap(),
        "inlineData"
    );
    for part in [
        json!(1),
        json!({"functionCall": {}}),
        json!({"functionResponse": {}}),
        json!({"text": 1}),
    ] {
        assert!(validate_gemini_nested_function_response_part(&part).is_err());
    }
    let response =
        json!({"response": {"ok": true}, "parts": [{"inlineData": {"mimeType": "text/plain"}}]});
    assert!(matches!(
        gemini_function_response_to_message_content(&response).unwrap(),
        MessageContent::Parts(parts) if parts.len() == 2
    ));
    assert!(gemini_function_response_to_message_content(&json!({"response": {}})).is_ok());
    assert!(
        gemini_function_response_to_message_content(&json!({"response": {}, "parts": 1})).is_err()
    );
}

#[test]
fn test_gemini_request_decoders_cover_generation_tools_and_model_validation() {
    let request = json!({
        "generationConfig": {
            "temperature": 0.5,
            "topP": 0.9,
            "maxOutputTokens": 42,
            "stopSequences": ["stop"]
        },
        "model": "gemini-test",
        "tools": [{
            "functionDeclarations": [{
                "name": "weather",
                "description": "lookup",
                "parameters": {"type": "object"},
                "x-extra": true
            }],
            "googleSearch": {}
        }]
    });
    let object = request.as_object().unwrap();
    let params = decode_gemini_generation_params(object).unwrap().unwrap();
    assert_eq!(params.temperature, Some(0.5));
    assert_eq!(params.top_p, Some(0.9));
    assert_eq!(params.max_tokens, Some(42));
    assert_eq!(params.stop, Some(vec!["stop".into()]));
    assert_eq!(
        decode_gemini_model(object).unwrap().as_deref(),
        Some("gemini-test")
    );
    assert_eq!(decode_gemini_tools(object).unwrap().unwrap().len(), 2);
    assert_eq!(
        decode_gemini_generation_params(&serde_json::Map::new()).unwrap(),
        None
    );

    for request in [
        json!({"generationConfig": 1}),
        json!({"generationConfig": {"temperature": "hot"}}),
        json!({"generationConfig": {"maxOutputTokens": -1}}),
        json!({"generationConfig": {"stopSequences": [1]}}),
    ] {
        assert!(
            decode_gemini_generation_params(request.as_object().unwrap()).is_err(),
            "{request}"
        );
    }
    assert!(decode_gemini_model(json!({"model": 1}).as_object().unwrap()).is_err());
    for request in [
        json!({"tools": 1}),
        json!({"tools": [{"functionDeclarations": [{"name": ""}]}]}),
    ] {
        assert!(
            decode_gemini_tools(request.as_object().unwrap()).is_err(),
            "{request}"
        );
    }
}

#[test]
fn test_gemini_response_part_extractors_cover_content_and_tool_calls() {
    let parts = vec![
        json!({"thought": true, "text": "hidden"}),
        json!({"text": "visible"}),
        json!({"functionCall": {"name": "weather", "args": {"city": "NYC"}}}),
        json!({"functionCall": {"id": "call-2", "name": "time"}}),
    ];
    assert!(matches!(
        extract_parts_message_content(&parts).unwrap(),
        Some(MessageContent::Text(text)) if text == "visible"
    ));
    let calls = extract_parts_tool_calls(&parts).unwrap().unwrap();
    assert_eq!(calls[0].id, "weather");
    assert_eq!(calls[1].id, "call-2");
    assert_eq!(calls[0].arguments, json!({"city": "NYC"}));
    assert_eq!(calls[1].arguments, json!({}));
    assert_eq!(
        extract_parts_tool_calls(&[json!({"text": "only"})]).unwrap(),
        None
    );
    for malformed in [
        json!({"functionCall": null}),
        json!({"functionCall": {"name": ""}}),
        json!({"functionCall": {"name": "tool", "args": 1}}),
    ] {
        assert!(extract_parts_tool_calls(&[malformed]).is_err());
    }
    assert!(extract_parts_message_content(&[json!({"functionResponse": {}})]).is_err());
}

#[test]
fn test_gemini_tool_response_helpers_wrap_and_validate_normalized_content() {
    assert_eq!(
        ensure_object_response("{\"ok\":true}".into()),
        json!({"ok": true})
    );
    assert_eq!(ensure_object_response("[1]".into()), json!({"output": [1]}));
    assert_eq!(
        ensure_object_response("plain".into()),
        json!({"output": "plain"})
    );
    assert_eq!(extract_content_text(&json!("text")), "text");
    assert_eq!(
        extract_content_text(&json!([{"type": "text", "text": "one"}, {"text": "two"}])),
        "one\ntwo"
    );

    let native = json!({"provider": "gemini", "kind": "inlineData", "value": {"inlineData": {}}});
    assert_eq!(
        provider_native_gemini_part_value(native.as_object().unwrap()).unwrap(),
        json!({"inlineData": {}})
    );
    for part in [
        json!({}),
        json!({"provider": "other", "kind": "text", "value": {"text": "x"}}),
        json!({"provider": "gemini", "kind": "", "value": {"text": "x"}}),
        json!({"provider": "gemini", "kind": "text", "value": {"functionCall": {}}}),
    ] {
        assert!(provider_native_gemini_part_value(part.as_object().unwrap()).is_err());
    }
    let payload = function_response_payload_from_tool_content(&json!([
        {"type": "text", "text": "answer"},
        {"type": "provider_native", "provider": "gemini", "kind": "inlineData", "value": {"inlineData": {}}}
    ]))
    .unwrap();
    assert_eq!(payload.response, json!({"output": "answer"}));
    assert_eq!(payload.parts.unwrap().len(), 1);
    assert!(function_response_payload_from_tool_content(&json!(true)).is_err());
}

#[test]
fn test_gemini_serialization_and_native_tool_helpers_handle_edge_cases() {
    let mut object = serde_json::Map::new();
    insert_serialized(&mut object, "value", &vec!["one"], "test").unwrap();
    assert_eq!(object["value"], json!(["one"]));
    assert_eq!(json_f64(0.5, "temperature").unwrap(), json!(0.5));
    assert!(json_f64(f64::NAN, "temperature").is_err());

    let group = json!({"functionDeclarations": [], "googleSearch": {}, "codeExecution": {}});
    let fields = gemini_native_tool_fields(group.as_object().unwrap()).unwrap();
    assert_eq!(gemini_native_tool_kind(&fields), "codeExecution");
    assert_eq!(
        gemini_native_tool_keys(&fields),
        vec!["codeExecution".to_string(), "googleSearch".to_string()]
    );
    assert_eq!(gemini_native_tool_kind(&json!(null)), "unknown");
    assert_eq!(gemini_native_tool_keys(&json!(null)), Vec::<String>::new());
    assert_eq!(
        gemini_native_tool_fields(&serde_json::Map::from_iter([(
            "functionDeclarations".to_string(),
            json!([]),
        )])),
        None
    );

    let groups = vec![json!({"googleSearch": {}}), fields.clone()];
    let mut used = vec![false; groups.len()];
    assert_eq!(
        take_matching_native_group(&groups, &mut used, &gemini_native_tool_keys(&fields)),
        Some(fields)
    );
    assert_eq!(
        take_matching_native_group(&groups, &mut used, &["missing".into()]),
        None
    );
}

#[test]
fn test_gemini_tool_group_encoders_preserve_supported_native_and_function_tools() {
    let function = ToolDefinition::Function {
        function: FunctionDefinition {
            name: "weather".into(),
            description: Some("lookup".into()),
            parameters: Some(json!({"type": "object"})),
            strict: None,
            extra: serde_json::Map::from_iter([("x-extra".into(), json!(true))]),
        },
        extra: Default::default(),
    };
    let native = ToolDefinition::ProviderNative {
        provider: "gemini".into(),
        kind: "googleSearch".into(),
        value: json!({"googleSearch": {}}),
    };
    let tools = vec![function.clone(), native.clone()];
    assert_eq!(gemini_function_declarations(Some(&tools)).unwrap().len(), 1);
    assert_eq!(
        gemini_native_tool_groups(Some(&tools)).unwrap(),
        vec![json!({"googleSearch": {}})]
    );
    assert!(validate_gemini_native_tool_group(&json!(null)).is_err());
    assert!(validate_gemini_native_tool_group(&json!({"functionDeclarations": []})).is_err());
    let invalid = ToolDefinition::Function {
        function: FunctionDefinition {
            name: "".into(),
            description: None,
            parameters: None,
            strict: None,
            extra: Default::default(),
        },
        extra: Default::default(),
    };
    assert!(gemini_function_declarations(Some(&vec![invalid])).is_err());
    let foreign = ToolDefinition::ProviderNative {
        provider: "other".into(),
        kind: "x".into(),
        value: json!({}),
    };
    assert!(gemini_function_declarations(Some(&vec![foreign])).is_err());
}

#[test]
fn test_gemini_extra_field_patching_removes_replaces_and_adds_values() {
    let mut target = serde_json::Map::from_iter([
        ("keep".into(), json!(1)),
        ("remove".into(), json!(2)),
        ("unchanged".into(), json!(3)),
    ]);
    let baseline = target.clone();
    let annotated = serde_json::Map::from_iter([
        ("keep".into(), json!(4)),
        ("unchanged".into(), json!(3)),
        ("add".into(), json!(5)),
    ]);
    patch_extra_fields(&mut target, &baseline, &annotated);
    assert_eq!(target, annotated);
}

// ===================================================================
// Response decode tests
// ===================================================================

#[test]
fn test_decode_response_text() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "Hello, world!"}]
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        },
        "modelVersion": "gemini-2.0-flash"
    });

    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.message,
        Some(MessageContent::Text("Hello, world!".into()))
    );
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));
    assert_eq!(resp.model, Some("gemini-2.0-flash".into()));

    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(10));
    assert_eq!(usage.completion_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(15));
}

#[test]
fn test_decode_response_native_part_as_provider_native_content() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "ran code"},
                    {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "sk-code-secret"}}
                ]
            },
            "finishReason": "STOP"
        }]
    });

    let resp = codec.decode_response(&response).unwrap();
    let Some(MessageContent::Parts(parts)) = resp.message else {
        panic!("expected mixed Gemini response parts to decode as MessageContent::Parts");
    };

    assert!(matches!(
        &parts[0],
        ContentPart::Text { text, .. } if text == "ran code"
    ));
    match &parts[1] {
        ContentPart::ProviderNative {
            provider,
            kind,
            value,
        } => {
            assert_eq!(provider, "gemini");
            assert_eq!(kind, "codeExecutionResult");
            assert_eq!(
                value["codeExecutionResult"]["output"],
                json!("sk-code-secret")
            );
        }
        other => panic!("expected Gemini ProviderNative response part, got {other:?}"),
    }
}

#[test]
fn test_decode_response_response_id() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "responseId": "resp-abc-123",
        "usageMetadata": {"promptTokenCount": 1}
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.id.as_deref(),
        Some("resp-abc-123"),
        "responseId must be mapped to AnnotatedLlmResponse.id"
    );
}

#[test]
fn test_decode_response_function_call() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "NYC"}
                    }
                }]
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30
        }
    });

    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::ToolUse));
    assert!(resp.message.is_none());

    let tool_calls = resp.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"location": "NYC"}));
    // id is derived from name when Gemini omits one
    assert_eq!(tool_calls[0].id, "get_weather");
}

#[test]
fn test_decode_response_finish_reason_table() {
    let codec = GeminiGenerateContentCodec;

    // (finishReason string, expected FinishReason)
    // NOTE: these cases have no functionCall parts in the response, so has_tool_calls = false.
    let cases: &[(&str, Option<FinishReason>)] = &[
        ("STOP", Some(FinishReason::Complete)),
        ("MAX_TOKENS", Some(FinishReason::Length)),
        ("TOOL_CODE", Some(FinishReason::ToolUse)),
        // Error / malfunction codes must not be overridden by tool-call heuristic.
        (
            "MALFORMED_FUNCTION_CALL",
            Some(FinishReason::Unknown("MALFORMED_FUNCTION_CALL".into())),
        ),
        (
            "UNEXPECTED_TOOL_CALL",
            Some(FinishReason::Unknown("UNEXPECTED_TOOL_CALL".into())),
        ),
        // Text safety / policy reasons — all must map to ContentFilter, not Unknown.
        ("SAFETY", Some(FinishReason::ContentFilter)),
        ("RECITATION", Some(FinishReason::ContentFilter)),
        ("BLOCKLIST", Some(FinishReason::ContentFilter)),
        ("PROHIBITED_CONTENT", Some(FinishReason::ContentFilter)),
        ("SPII", Some(FinishReason::ContentFilter)),
        // Image safety and other policy reasons — also ContentFilter.
        ("LANGUAGE", Some(FinishReason::ContentFilter)),
        ("IMAGE_SAFETY", Some(FinishReason::ContentFilter)),
        (
            "IMAGE_PROHIBITED_CONTENT",
            Some(FinishReason::ContentFilter),
        ),
        ("IMAGE_RECITATION", Some(FinishReason::ContentFilter)),
        ("ESCALATION", Some(FinishReason::ContentFilter)),
        // Unspecified / not-yet-finished must map to None.
        ("FINISH_REASON_UNSPECIFIED", None),
        // Unknown future values map to Unknown.
        (
            "FUTURE_REASON",
            Some(FinishReason::Unknown("FUTURE_REASON".into())),
        ),
    ];

    for (reason_str, expected) in cases {
        let response = json!({
            "candidates": [{
                "content": {"role": "model", "parts": []},
                "finishReason": reason_str,
                "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0}
        });
        let resp = codec.decode_response(&response).unwrap();
        assert_eq!(
            &resp.finish_reason, expected,
            "finishReason={reason_str} must decode to {expected:?}"
        );
    }
}

#[test]
fn test_decode_response_prompt_feedback_block_reason_content_filter() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "promptFeedback": {
            "blockReason": "SAFETY",
            "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}]
        },
        "usageMetadata": {"promptTokenCount": 12},
        "modelVersion": "gemini-2.0-flash"
    });

    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::ContentFilter));
    assert!(resp.message.is_none());
    assert!(resp.tool_calls.is_none());
    assert_eq!(
        resp.extra
            .get("promptFeedback")
            .and_then(|feedback| feedback.get("blockReason"))
            .and_then(Json::as_str),
        Some("SAFETY"),
        "promptFeedback must remain available as top-level response extra"
    );
}

#[test]
fn test_decode_response_prompt_feedback_block_reason_must_be_string() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "promptFeedback": {"blockReason": 123}
    });

    let err = codec.decode_response(&response).unwrap_err();
    assert!(
        err.to_string().contains("blockReason must be a string"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_decode_response_cached_content_token_count() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "cached response"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "totalTokenCount": 110,
            "cachedContentTokenCount": 80
        }
    });

    let resp = codec.decode_response(&response).unwrap();
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(100));
    assert_eq!(usage.completion_tokens, Some(10));
    assert_eq!(usage.cache_read_tokens, Some(80));
    assert_eq!(usage.cache_write_tokens, None);
}

#[test]
fn test_decode_response_no_candidates() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0, "totalTokenCount": 5}
    });

    let resp = codec.decode_response(&response).unwrap();
    assert!(resp.message.is_none());
    assert!(resp.tool_calls.is_none());
    assert!(resp.finish_reason.is_none());
}

#[test]
fn test_decode_response_extra_fields_preserved() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        "unknownFutureField": "value"
    });

    let resp = codec.decode_response(&response).unwrap();
    assert!(resp.extra.contains_key("unknownFutureField"));
}

#[test]
fn test_decode_response_candidate_extra_removes_reserved_api_key() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "index": 0,
            "api": "bad_discriminator",
            "futureField": true
        }]
    });

    let resp = codec.decode_response(&response).unwrap();
    let Some(super::super::response::ApiSpecificResponse::GeminiGenerateContent { extra, .. }) =
        resp.api_specific
    else {
        panic!("expected Gemini api_specific metadata");
    };
    assert!(extra.get("api").is_none());
    assert_eq!(extra.get("futureField"), Some(&json!(true)));
}

#[test]
fn test_decode_response_candidate_only_reserved_api_key_has_no_api_specific() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "api": "bad_discriminator"
        }]
    });

    let resp = codec.decode_response(&response).unwrap();
    assert!(
        resp.api_specific.is_none(),
        "reserved api discriminator alone must not create empty Gemini api_specific metadata"
    );
}

// ===================================================================
// Request decode tests
// ===================================================================

#[test]
fn test_decode_contents_with_system_instruction() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there"}]},
            {"role": "user", "parts": [{"text": "What's the weather?"}]}
        ],
        "systemInstruction": {
            "parts": [{"text": "You are a helpful assistant."}]
        }
    }));

    let annotated = codec.decode(&request).unwrap();

    assert!(
        matches!(&annotated.messages[0], Message::System { content: MessageContent::Text(t), .. } if t == "You are a helpful assistant.")
    );
    assert!(
        matches!(&annotated.messages[1], Message::User { content: MessageContent::Text(t), .. } if t == "Hello")
    );
    assert!(
        matches!(&annotated.messages[2], Message::Assistant { content: Some(MessageContent::Text(t)), .. } if t == "Hi there")
    );
    assert!(
        matches!(&annotated.messages[3], Message::User { content: MessageContent::Text(t), .. } if t == "What's the weather?")
    );
    assert_eq!(annotated.messages.len(), 4);
}

#[test]
fn test_decode_function_declarations() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }]
        }]
    }));

    let annotated = codec.decode(&request).unwrap();
    let tools = annotated.tools.unwrap();
    assert_eq!(tools.len(), 1);
    let nemo_relay_types::codec::request::ToolDefinition::Function { ref function, .. } = tools[0]
    else {
        panic!("expected Function variant");
    };
    assert_eq!(function.name, "get_weather");
    assert_eq!(
        function.description.as_deref(),
        Some("Get the weather for a location")
    );
    assert!(function.parameters.is_some());
}

#[test]
fn test_decode_function_declaration_preserves_provider_fields() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "my_tool",
                "description": "A tool",
                "parameters": {"type": "object"},
                "parametersJsonSchema": {"$schema": "draft-2020-12"},
                "responseJsonSchema": {"type": "object"},
                "behavior": "BLOCKING"
            }]
        }]
    }));

    let annotated = codec.decode(&request).unwrap();
    let tools = annotated.tools.unwrap();
    let nemo_relay_types::codec::request::ToolDefinition::Function { ref function, .. } = tools[0]
    else {
        panic!("expected Function variant");
    };
    assert!(
        function.extra.contains_key("parametersJsonSchema"),
        "parametersJsonSchema must be in extra"
    );
    assert!(
        function.extra.contains_key("responseJsonSchema"),
        "responseJsonSchema must be in extra"
    );
    assert!(
        function.extra.contains_key("behavior"),
        "behavior must be in extra"
    );
    assert!(
        !function.extra.contains_key("name"),
        "modeled fields must not appear in extra"
    );
}

#[test]
fn test_decode_generation_config() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 1024,
            "stopSequences": ["stop1", "stop2"]
        }
    }));

    let annotated = codec.decode(&request).unwrap();
    let params = annotated.params.unwrap();
    assert!((params.temperature.unwrap() - 0.7).abs() < 1e-9);
    assert!((params.top_p.unwrap() - 0.9).abs() < 1e-9);
    assert_eq!(params.max_tokens, Some(1024));
    assert_eq!(
        params.stop.as_deref(),
        Some(&["stop1".to_string(), "stop2".to_string()][..])
    );
}

#[test]
fn test_decode_extra_fields_captured() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
        "cachedContent": "cachedContent/abc"
    }));

    let annotated = codec.decode(&request).unwrap();
    assert!(annotated.extra.contains_key("safetySettings"));
    assert!(annotated.extra.contains_key("cachedContent"));
    assert!(!annotated.extra.contains_key("contents"));
}

// ===================================================================
// Request encode tests
// ===================================================================

#[test]
fn test_encode_round_trip_preserves_extra_fields() {
    let codec = GeminiGenerateContentCodec;
    let original_json = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
    });
    let original = make_request(original_json);

    let annotated = codec.decode(&original).unwrap();
    let re_encoded = codec.encode(&annotated, &original).unwrap();

    assert!(re_encoded.content.get("safetySettings").is_some());
}

/// Verify that "assistant" → "model" mapping works when a fresh Message::Assistant
/// is encoded (not decoded from an existing Gemini request).
#[test]
fn test_encode_role_assistant_becomes_model() {
    let codec = GeminiGenerateContentCodec;
    // Decode a single-user-turn request to get a valid baseline.
    let original_json = json!({
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
    });
    let original = make_request(original_json);
    let mut annotated = codec.decode(&original).unwrap();

    // Intercept adds a fresh assistant reply (not decoded from Gemini — this
    // exercises the normalized_to_gemini_content path directly).
    annotated.messages.push(Message::Assistant {
        content: Some(MessageContent::Text("Hi there".into())),
        tool_calls: None,
        name: None,
    });

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded.content.get("contents").unwrap().as_array().unwrap();

    assert_eq!(contents.len(), 2);
    let second_role = contents[1].get("role").unwrap().as_str().unwrap();
    assert_eq!(
        second_role, "model",
        "Message::Assistant must encode as role 'model'"
    );
}

/// Decode-then-encode preserves systemInstruction when unchanged.
#[test]
fn test_encode_preserves_system_instruction_when_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let original_json = json!({
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "systemInstruction": {"parts": [{"text": "You are an assistant."}]}
    });
    let original = make_request(original_json);

    let annotated = codec.decode(&original).unwrap();
    let re_encoded = codec.encode(&annotated, &original).unwrap();

    let sys = re_encoded.content.get("systemInstruction").unwrap();
    let parts = sys.get("parts").unwrap().as_array().unwrap();
    assert_eq!(
        parts[0].get("text").unwrap().as_str().unwrap(),
        "You are an assistant."
    );
}

/// Decode-then-encode preserves tools when unchanged (round-trip).
#[test]
fn test_encode_preserves_tools_when_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let original_json = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }]
        }]
    });
    let original = make_request(original_json);

    let annotated = codec.decode(&original).unwrap();
    let re_encoded = codec.encode(&annotated, &original).unwrap();

    let tools = re_encoded.content.get("tools").unwrap().as_array().unwrap();
    assert_eq!(tools.len(), 1);
    let fds = tools[0]
        .get("functionDeclarations")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(fds.len(), 1);
    assert_eq!(fds[0].get("name").unwrap().as_str().unwrap(), "search");
    assert!(fds[0].get("parameters").is_some());
}

/// Decode-then-encode preserves generationConfig when unchanged (round-trip).
#[test]
fn test_encode_preserves_generation_config_when_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let original_json = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 512
        }
    });
    let original = make_request(original_json);

    let annotated = codec.decode(&original).unwrap();
    let re_encoded = codec.encode(&annotated, &original).unwrap();

    let gc = re_encoded.content.get("generationConfig").unwrap();
    assert!((gc.get("temperature").unwrap().as_f64().unwrap() - 0.5).abs() < 1e-9);
    assert_eq!(gc.get("maxOutputTokens").unwrap().as_u64().unwrap(), 512);
}

/// Interceptor can edit the system message; the new text appears in the output.
#[test]
fn test_encode_system_instruction_edit() {
    let codec = GeminiGenerateContentCodec;
    let original_json = json!({
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "systemInstruction": {"parts": [{"text": "old prompt"}]}
    });
    let original = make_request(original_json);

    let mut annotated = codec.decode(&original).unwrap();
    for msg in annotated.messages.iter_mut() {
        if let Message::System { content, .. } = msg {
            *content = MessageContent::Text("new prompt".into());
        }
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let sys_text = encoded
        .content
        .get("systemInstruction")
        .and_then(|s| s.get("parts"))
        .and_then(|p| p.as_array())
        .and_then(|a| a.first())
        .and_then(|p| p.get("text"))
        .and_then(Json::as_str)
        .unwrap();
    assert_eq!(sys_text, "new prompt");
}

// ===================================================================
// Streaming tests
// ===================================================================

#[test]
fn test_streaming_two_chunks_accumulated() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Hello "}]},
            "index": 0
        }],
        "modelVersion": "gemini-2.0-flash"
    }))
    .unwrap();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "world!"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8
        }
    }))
    .unwrap();

    let assembled = finalizer();
    assert_eq!(assembled["candidates"][0]["index"].as_u64(), Some(0));

    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();
    assert_eq!(
        resp.message,
        Some(MessageContent::Text("Hello world!".into()))
    );
    assert_eq!(resp.finish_reason, Some(FinishReason::Complete));

    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(5));
    assert_eq!(usage.completion_tokens, Some(3));
    assert_eq!(usage.total_tokens, Some(8));
}

#[test]
fn test_streaming_rejects_missing_candidate_index() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();

    let err = collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "orphan"}]}
        }]
    }))
    .expect_err("candidate index is required to avoid corrupt streaming aggregates");

    assert!(
        err.to_string().contains("candidate index is required"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_streaming_rejects_nonzero_candidate_index() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();

    let err = collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "second"}]},
            "index": 1
        }]
    }))
    .expect_err("nonzero candidate indexes cannot be reassembled losslessly");

    assert!(
        err.to_string().contains("only supports candidate index 0"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_streaming_rejects_candidate_index_change() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "first"}]},
            "index": 0
        }]
    }))
    .unwrap();

    let err = collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "second"}]},
            "index": 1
        }]
    }))
    .expect_err("candidate index changes must not be merged into candidate 0");

    assert!(
        err.to_string()
            .contains("candidate index changed across chunks"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_streaming_rejects_multiple_candidates_in_one_chunk() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();

    let err = collector(json!({
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "first"}]},
                "index": 0
            },
            {
                "content": {"role": "model", "parts": [{"text": "second"}]},
                "index": 1
            }
        ]
    }))
    .expect_err("multi-candidate chunks cannot be reassembled by the single-candidate state");

    assert!(
        err.to_string().contains("multiple candidates"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_streaming_preserves_text_part_metadata_from_empty_final_chunk() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "answer"}]},
            "index": 0
        }]
    }))
    .unwrap();

    collector(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "", "thoughtSignature": "sig_STREAM=="}]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    }))
    .unwrap();

    let assembled = finalizer();
    let parts = assembled["candidates"][0]["content"]["parts"]
        .as_array()
        .unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["text"].as_str(), Some("answer"));
    assert!(
        parts[0].get("thoughtSignature").is_none(),
        "signature from the empty final chunk must not be merged onto the previous part"
    );
    assert_eq!(parts[1]["text"].as_str(), Some(""));
    assert_eq!(
        parts[1]["thoughtSignature"].as_str(),
        Some("sig_STREAM=="),
        "streamed text metadata must survive finalization"
    );

    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();
    let Some(MessageContent::Parts(parts)) = resp.message else {
        panic!("metadata-bearing streamed text must decode as content parts");
    };
    assert!(matches!(
        &parts[0],
        ContentPart::Text { text, extra } if text == "answer" && extra.is_empty()
    ));
    assert!(matches!(
        &parts[1],
        ContentPart::Text { text, extra }
            if text.is_empty()
                && extra.get("thoughtSignature").and_then(Json::as_str) == Some("sig_STREAM==")
    ));
}

#[test]
fn test_streaming_keeps_non_empty_signed_text_part_separate_from_plain_text() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "first"}]},
            "index": 0
        }]
    }))
    .unwrap();

    collector(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "second", "thoughtSignature": "sig_SECOND=="}]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    }))
    .unwrap();

    let assembled = finalizer();
    let parts = assembled["candidates"][0]["content"]["parts"]
        .as_array()
        .unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["text"].as_str(), Some("first"));
    assert!(
        parts[0].get("thoughtSignature").is_none(),
        "signature from the second text part must not be moved onto the first"
    );
    assert_eq!(parts[1]["text"].as_str(), Some("second"));
    assert_eq!(parts[1]["thoughtSignature"].as_str(), Some("sig_SECOND=="));
}

#[test]
fn test_streaming_finalize_valid_response_shape() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "chunk"}]},
            "finishReason": "MAX_TOKENS",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 12
        },
        "modelVersion": "gemini-1.5-pro"
    }))
    .unwrap();

    let assembled = finalizer();

    assert!(assembled.get("candidates").is_some_and(Json::is_array));
    assert!(assembled.get("usageMetadata").is_some());
    assert_eq!(
        assembled.get("modelVersion").and_then(Json::as_str),
        Some("gemini-1.5-pro")
    );

    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();
    assert_eq!(resp.finish_reason, Some(FinishReason::Length));
    assert_eq!(resp.model, Some("gemini-1.5-pro".into()));
}

#[test]
fn test_streaming_last_usage_metadata_wins() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "a"}]}, "index": 0}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
    }))
    .unwrap();

    collector(json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "b"}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    }))
    .unwrap();

    let assembled = finalizer();
    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();

    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(10));
    assert_eq!(usage.completion_tokens, Some(5));
    assert_eq!(resp.message, Some(MessageContent::Text("ab".into())));
}

#[test]
fn test_streaming_preserves_native_response_parts() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "ran code"},
                    {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "sk-stream-code"}}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    }))
    .unwrap();

    let assembled = finalizer();
    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();
    let Some(MessageContent::Parts(parts)) = resp.message else {
        panic!("expected streamed native Gemini part to survive finalization");
    };

    assert!(matches!(
        &parts[0],
        ContentPart::Text { text, .. } if text == "ran code"
    ));
    match &parts[1] {
        ContentPart::ProviderNative {
            provider,
            kind,
            value,
        } => {
            assert_eq!(provider, "gemini");
            assert_eq!(kind, "codeExecutionResult");
            assert_eq!(
                value["codeExecutionResult"]["output"],
                json!("sk-stream-code")
            );
        }
        other => panic!("expected Gemini ProviderNative streaming part, got {other:?}"),
    }
}

#[test]
fn test_streaming_preserves_native_before_text_order() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": "abc123=="}},
                    {"text": "caption", "thoughtSignature": "sig_TEXT=="}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    }))
    .unwrap();

    let assembled = finalizer();
    let parts = assembled["candidates"][0]["content"]["parts"]
        .as_array()
        .unwrap();
    assert!(
        parts[0].get("inlineData").is_some(),
        "streaming finalizer must preserve native part position before text"
    );
    assert_eq!(parts[1]["text"].as_str(), Some("caption"));
    assert_eq!(parts[1]["thoughtSignature"].as_str(), Some("sig_TEXT=="));
}

#[test]
fn test_streaming_does_not_merge_text_across_native_part() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "before"},
                    {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "42"}},
                    {"text": "after", "thoughtSignature": "sig_AFTER=="}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    }))
    .unwrap();

    let assembled = finalizer();
    let parts = assembled["candidates"][0]["content"]["parts"]
        .as_array()
        .unwrap();
    assert_eq!(parts[0]["text"].as_str(), Some("before"));
    assert!(
        parts[1].get("codeExecutionResult").is_some(),
        "native part must stay between the two streamed text parts"
    );
    assert_eq!(parts[2]["text"].as_str(), Some("after"));
    assert_eq!(parts[2]["thoughtSignature"].as_str(), Some("sig_AFTER=="));
}

#[test]
fn test_streaming_preserves_adjacent_signed_text_parts() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "first", "thoughtSignature": "sig_FIRST=="},
                    {"text": "second", "thoughtSignature": "sig_SECOND=="}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    }))
    .unwrap();

    let assembled = finalizer();
    let parts = assembled["candidates"][0]["content"]["parts"]
        .as_array()
        .unwrap();
    assert_eq!(parts[0]["text"].as_str(), Some("first"));
    assert_eq!(parts[0]["thoughtSignature"].as_str(), Some("sig_FIRST=="));
    assert_eq!(parts[1]["text"].as_str(), Some("second"));
    assert_eq!(parts[1]["thoughtSignature"].as_str(), Some("sig_SECOND=="));
}

// ===================================================================
// Thought-signature and native-part preservation
// ===================================================================

#[test]
fn test_encode_lossless_when_messages_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "what is 2+2?"},
                    {"inlineData": {"mimeType": "image/png", "data": "abc123=="}}
                ]
            },
            {
                "role": "model",
                "parts": [
                    {"thought": true, "text": "let me think..."},
                    {"thought": true, "thoughtSignature": "sig_XYZ_abc=="},
                    {"functionCall": {"name": "calculator", "id": "call_1", "args": {"op": "add", "a": 2, "b": 2}}}
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": "Be helpful."}]},
        "generationConfig": {"temperature": 0.5},
        "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
    }));

    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();

    assert_eq!(
        encoded.content, original.content,
        "encode(decode(req), req) must be byte-identical to req"
    );
}

#[test]
fn test_encode_thought_signature_preserved_when_system_message_changes() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "call something"}]},
            {
                "role": "model",
                "parts": [
                    {"thought": true, "thoughtSignature": "sig_CRITICAL=="},
                    {"functionCall": {"name": "my_fn", "id": "call_99", "args": {"x": 1}}}
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": "old system prompt"}]}
    }));

    let mut annotated = codec.decode(&original).unwrap();

    for msg in annotated.messages.iter_mut() {
        if let Message::System { content, .. } = msg {
            *content = MessageContent::Text("new system prompt".into());
        }
    }

    let encoded = codec.encode(&annotated, &original).unwrap();

    let sys_text = encoded
        .content
        .get("systemInstruction")
        .and_then(|s| s.get("parts"))
        .and_then(|p| p.as_array())
        .and_then(|a| a.first())
        .and_then(|p| p.get("text"))
        .and_then(Json::as_str)
        .unwrap();
    assert_eq!(sys_text, "new system prompt");

    assert_eq!(
        encoded.content.get("contents"),
        original.content.get("contents"),
        "thoughtSignature must survive when only the system message changed"
    );
}

/// Multi-turn continuation: interceptor appends a tool result and a new user turn.
/// Earlier turns must be preserved exactly; the new tool-result turn must carry
/// the correct id, name, and response fields.
#[test]
fn test_encode_thought_signature_preserved_in_multi_turn_continuation() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "call the tool"}]},
            {
                "role": "model",
                "parts": [
                    {"thought": true, "thoughtSignature": "sig_MUST_SURVIVE=="},
                    {"functionCall": {"name": "my_fn", "id": "call_1", "args": {}}}
                ]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    annotated.messages.push(Message::Tool {
        content: MessageContent::Text(r#"{"output": "done"}"#.into()),
        tool_call_id: "call_1".into(),
    });
    annotated.messages.push(Message::User {
        content: MessageContent::Text("thanks, continue".into()),
        name: None,
    });

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();

    // Unchanged turns are preserved byte-identically.
    let orig_contents = original
        .content
        .get("contents")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        &contents[0], &orig_contents[0],
        "unchanged user turn at position 0 must be preserved byte-identically"
    );
    assert_eq!(
        &contents[1], &orig_contents[1],
        "model turn with thoughtSignature at position 1 must be preserved byte-identically"
    );
    assert_eq!(contents.len(), 4);

    // New tool-result turn at position 2 must have correct id, name, role, and response.
    let tool_turn = &contents[2];
    assert_eq!(
        tool_turn.get("role").and_then(Json::as_str),
        Some("user"),
        "tool result must use role 'user'"
    );
    let fr = tool_turn
        .get("parts")
        .and_then(Json::as_array)
        .and_then(|p| p.first())
        .and_then(|p| p.get("functionResponse"))
        .expect("tool result must contain a functionResponse part");
    assert_eq!(
        fr.get("id").and_then(Json::as_str),
        Some("call_1"),
        "functionResponse.id must be the actual call ID, not the function name"
    );
    assert_eq!(
        fr.get("name").and_then(Json::as_str),
        Some("my_fn"),
        "functionResponse.name must be the function name looked up from the assistant turn"
    );
    assert!(
        fr.get("response").is_some(),
        "functionResponse.response must be present"
    );

    // New user turn at position 3.
    assert_eq!(contents[3].get("role").and_then(Json::as_str), Some("user"));
}

/// thoughtSignature on a functionCall part itself must survive when an interceptor
/// edits the function call arguments (triggers the rebuild path).
#[test]
fn test_encode_thought_signature_on_function_call_part_survives_edit() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "run the tool"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "my_fn", "id": "call_1", "args": {"x": 1}},
                        "thoughtSignature": "ABCDEF=="
                    }
                ]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Intercept changes the function call arguments.
    if let Message::Assistant {
        tool_calls: Some(tcs),
        ..
    } = &mut annotated.messages[1]
        && let Some(tc) = tcs.first_mut()
    {
        tc.function.arguments = r#"{"x": 42}"#.to_string();
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    let model_parts = contents[1].get("parts").and_then(Json::as_array).unwrap();

    let fc_part = model_parts
        .iter()
        .find(|p| p.get("functionCall").is_some())
        .expect("encoded model turn must contain a functionCall part");
    assert_eq!(
        fc_part.get("thoughtSignature").and_then(Json::as_str),
        Some("ABCDEF=="),
        "thoughtSignature on the functionCall part must survive when args are edited"
    );
    // Verify the args were actually updated.
    let args = fc_part
        .get("functionCall")
        .and_then(|fc| fc.get("args"))
        .unwrap();
    assert_eq!(args.get("x").and_then(Json::as_i64), Some(42));
}

// ===================================================================
// inlineData round-trip
// ===================================================================

#[test]
fn test_encode_inline_data_preserved_when_messages_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQ==";
    let original = make_request(json!({
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "what is in this image?"},
                    {"inlineData": {"mimeType": "image/png", "data": image_data}}
                ]
            }
        ]
    }));

    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();

    let parts = encoded.content.get("contents").unwrap().as_array().unwrap()[0]
        .get("parts")
        .unwrap()
        .as_array()
        .unwrap();

    let inline = parts
        .iter()
        .find(|p| p.get("inlineData").is_some())
        .unwrap();
    assert_eq!(
        inline
            .get("inlineData")
            .unwrap()
            .get("data")
            .and_then(Json::as_str),
        Some(image_data),
        "inlineData must be byte-identical after round-trip"
    );
}

#[test]
fn test_encode_inline_data_preserved_when_text_changes() {
    let codec = GeminiGenerateContentCodec;
    let image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQ==";
    let original = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "what is in this image?"},
                {"inlineData": {"mimeType": "image/png", "data": image_data}}
            ]
        }]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    if let Message::User { content, .. } = &mut annotated.messages[0] {
        *content = MessageContent::Text("Please describe this image in detail.".into());
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let parts = encoded.content.get("contents").unwrap().as_array().unwrap()[0]
        .get("parts")
        .unwrap()
        .as_array()
        .unwrap();

    assert_eq!(
        parts[0].get("text").and_then(Json::as_str),
        Some("Please describe this image in detail."),
        "text part must be updated to the interceptor's new text"
    );
    let inline = parts
        .iter()
        .find(|p| p.get("inlineData").is_some())
        .expect("inlineData must survive when interceptor only changed the text");
    assert_eq!(
        inline
            .get("inlineData")
            .unwrap()
            .get("data")
            .and_then(Json::as_str),
        Some(image_data)
    );
}

#[test]
fn test_decode_inline_data_as_provider_native_content_part() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "what is in this file?"},
                {"inlineData": {"mimeType": "text/plain", "data": "sk-file-secret"}}
            ]
        }]
    }));

    let annotated = codec.decode(&request).unwrap();
    let Message::User {
        content: MessageContent::Parts(parts),
        ..
    } = &annotated.messages[0]
    else {
        panic!("expected mixed Gemini content to decode as MessageContent::Parts");
    };

    assert!(matches!(
        &parts[0],
        ContentPart::Text { text, .. } if text == "what is in this file?"
    ));
    match &parts[1] {
        ContentPart::ProviderNative {
            provider,
            kind,
            value,
        } => {
            assert_eq!(provider, "gemini");
            assert_eq!(kind, "inlineData");
            assert_eq!(
                value["inlineData"]["data"],
                json!("sk-file-secret"),
                "native inlineData must be visible to normalized middleware"
            );
        }
        other => panic!("expected Gemini ProviderNative content part, got {other:?}"),
    }
}

#[test]
fn test_encode_patches_provider_native_inline_data_content_part() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "inspect this"},
                {"inlineData": {"mimeType": "text/plain", "data": "sk-file-secret"}}
            ]
        }]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    let Message::User {
        content: MessageContent::Parts(parts),
        ..
    } = &mut annotated.messages[0]
    else {
        panic!("expected native content parts");
    };
    match &mut parts[1] {
        ContentPart::ProviderNative { value, .. } => {
            value["inlineData"]["data"] = json!("[REDACTED]");
        }
        other => panic!("expected Gemini ProviderNative content part, got {other:?}"),
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content["contents"][0]["parts"][1]["inlineData"]["data"],
        json!("[REDACTED]"),
        "editing the normalized provider-native content part must update the raw Gemini part"
    );
}

// ===================================================================
// Thought part filtering
// ===================================================================

#[test]
fn test_decode_response_filters_thought_parts() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"thought": true, "text": "internal reasoning the user should not see"},
                    {"text": "this is the actual answer"}
                ]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
    });
    let ann = codec.decode_response(&response).unwrap();
    assert_eq!(
        ann.message,
        Some(MessageContent::Text("this is the actual answer".into())),
        "thought parts must not leak into the normalized message"
    );
}

#[test]
fn test_streaming_filters_thought_parts() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"thought": true, "text": "reasoning"}]},
            "index": 0
        }]
    }))
    .unwrap();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "final answer"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3}
    }))
    .unwrap();

    let assembled = finalizer();
    let assembled_parts = assembled["candidates"][0]["content"]["parts"]
        .as_array()
        .unwrap();
    assert_eq!(
        assembled_parts[0]["thought"].as_bool(),
        Some(true),
        "thought chunks must survive in the provider-native streaming aggregate"
    );
    assert_eq!(assembled_parts[0]["text"].as_str(), Some("reasoning"));
    assert_eq!(assembled_parts[1]["text"].as_str(), Some("final answer"));

    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();
    assert_eq!(
        resp.message,
        Some(MessageContent::Text("final answer".into())),
        "thought chunks must not appear in the streamed message"
    );
}

/// When a model turn in the request history has both a thought part (thought: true)
/// and a functionCall part, the thought text must NOT appear in the decoded
/// Message::Assistant.content. Only visible (non-thought) text should appear.
#[test]
fn test_decode_request_thought_text_does_not_leak_into_assistant_content() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "search for it"}]},
            {
                "role": "model",
                "parts": [
                    {"thought": true, "text": "I should call the search function"},
                    {"functionCall": {"id": "call_1", "name": "search", "args": {"q": "test"}}}
                ]
            }
        ]
    }));

    let annotated = codec.decode(&request).unwrap();
    let asst = annotated
        .messages
        .iter()
        .find(|m| matches!(m, Message::Assistant { .. }))
        .expect("must have assistant message");
    if let Message::Assistant {
        content,
        tool_calls,
        ..
    } = asst
    {
        assert!(
            content.is_none(),
            "thought text must not appear in Message::Assistant.content; got: {:?}",
            content
        );
        assert!(tool_calls.is_some(), "functionCall must still be decoded");
    }
}

/// A model turn with both visible text and a functionCall: visible text appears in
/// content, thought text does not.
#[test]
fn test_decode_request_visible_text_survives_when_thought_present() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{
            "role": "model",
            "parts": [
                {"thought": true, "text": "let me think"},
                {"text": "here is my answer"},
                {"functionCall": {"id": "c1", "name": "fn", "args": {}}}
            ]
        }]
    }));

    let annotated = codec.decode(&request).unwrap();
    let asst = annotated
        .messages
        .iter()
        .find(|m| matches!(m, Message::Assistant { .. }))
        .expect("must have assistant message");
    if let Message::Assistant { content, .. } = asst {
        assert_eq!(
            content.as_ref().map(|c| match c {
                MessageContent::Text(t) => t.as_str(),
                _ => "",
            }),
            Some("here is my answer"),
            "only non-thought visible text must appear in content"
        );
    }
}

// ===================================================================
// Mixed tool groups
// ===================================================================

#[test]
fn test_encode_preserves_native_tool_groups_when_tools_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [
            {"functionDeclarations": [{"name": "get_weather", "description": "Get weather"}]},
            {"googleSearch": {}},
            {"codeExecution": {}}
        ]
    }));

    let annotated = codec.decode(&original).unwrap();
    let normalized_tools = annotated.tools.as_ref().unwrap();
    assert!(
        normalized_tools.iter().any(
            |td| matches!(td, ToolDefinition::ProviderNative { provider, kind, .. }
                if provider == "gemini" && kind == "googleSearch")
        ),
        "native googleSearch group must be visible as a Gemini ProviderNative tool"
    );
    assert!(
        normalized_tools.iter().any(
            |td| matches!(td, ToolDefinition::ProviderNative { provider, kind, .. }
                if provider == "gemini" && kind == "codeExecution")
        ),
        "native codeExecution group must be visible as a Gemini ProviderNative tool"
    );
    let encoded = codec.encode(&annotated, &original).unwrap();

    let tools = encoded
        .content
        .get("tools")
        .and_then(Json::as_array)
        .unwrap();
    assert!(
        tools.iter().any(|g| g.get("googleSearch").is_some()),
        "googleSearch group must survive unchanged tools encode"
    );
    assert!(
        tools.iter().any(|g| g.get("codeExecution").is_some()),
        "codeExecution group must survive unchanged tools encode"
    );
    assert!(
        tools
            .iter()
            .any(|g| g.get("functionDeclarations").is_some()),
        "functionDeclarations group must survive"
    );
}

#[test]
fn test_decode_native_only_tool_group_as_provider_native() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"googleSearch": {"apiKey": "sk-tool-secret"}}]
    }));

    let annotated = codec.decode(&original).unwrap();
    let tools = annotated.tools.as_ref().unwrap();
    assert_eq!(tools.len(), 1);
    match &tools[0] {
        ToolDefinition::ProviderNative {
            provider,
            kind,
            value,
        } => {
            assert_eq!(provider, "gemini");
            assert_eq!(kind, "googleSearch");
            assert_eq!(
                value,
                &json!({"googleSearch": {"apiKey": "sk-tool-secret"}})
            );
        }
        other => panic!("expected Gemini ProviderNative tool, got {other:?}"),
    }
}

#[test]
fn test_decode_mixed_tool_group_exposes_native_siblings() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{
            "functionDeclarations": [{"name": "lookup"}],
            "googleSearch": {"apiKey": "sk-tool-secret"}
        }]
    }));

    let annotated = codec.decode(&original).unwrap();
    let tools = annotated.tools.as_ref().unwrap();
    assert_eq!(tools.len(), 2);
    assert!(matches!(tools[0], ToolDefinition::Function { .. }));
    match &tools[1] {
        ToolDefinition::ProviderNative {
            provider,
            kind,
            value,
        } => {
            assert_eq!(provider, "gemini");
            assert_eq!(kind, "googleSearch");
            assert_eq!(
                value,
                &json!({"googleSearch": {"apiKey": "sk-tool-secret"}})
            );
        }
        other => panic!("expected Gemini ProviderNative sibling fields, got {other:?}"),
    }
}

#[test]
fn test_encode_preserves_native_tool_groups_when_functions_change() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [
            {"functionDeclarations": [{"name": "old_fn"}]},
            {"googleSearch": {}}
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    if let Some(tools) = annotated.tools.as_mut() {
        for td in tools.iter_mut() {
            if let nemo_relay_types::codec::request::ToolDefinition::Function { function, .. } = td
            {
                function.name = "new_fn".into();
            }
        }
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let tools = encoded
        .content
        .get("tools")
        .and_then(Json::as_array)
        .unwrap();

    assert!(
        tools.iter().any(|g| g.get("googleSearch").is_some()),
        "googleSearch must survive when only function declarations changed"
    );

    let fn_group = tools
        .iter()
        .find(|g| g.get("functionDeclarations").is_some())
        .unwrap();
    let fns = fn_group
        .get("functionDeclarations")
        .and_then(Json::as_array)
        .unwrap();
    assert_eq!(fns[0].get("name").and_then(Json::as_str), Some("new_fn"));
}

/// Provider-native functionDeclaration fields (parametersJsonSchema, responseJsonSchema,
/// response, behavior) must survive a decode → edit description → encode round-trip.
#[test]
fn test_encode_preserves_provider_tool_fields_when_description_changes() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "my_tool",
                "description": "old description",
                "parameters": {"type": "object"},
                "parametersJsonSchema": {"$schema": "draft-2020-12"},
                "responseJsonSchema": {"type": "object"},
                "behavior": "BLOCKING"
            }]
        }]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    if let Some(tools) = annotated.tools.as_mut() {
        for td in tools.iter_mut() {
            if let nemo_relay_types::codec::request::ToolDefinition::Function { function, .. } = td
            {
                function.description = Some("new description".into());
            }
        }
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let tools = encoded
        .content
        .get("tools")
        .and_then(Json::as_array)
        .unwrap();
    let fn_group = tools
        .iter()
        .find(|g| g.get("functionDeclarations").is_some())
        .unwrap();
    let fd = &fn_group
        .get("functionDeclarations")
        .and_then(Json::as_array)
        .unwrap()[0];

    assert_eq!(
        fd.get("description").and_then(Json::as_str),
        Some("new description")
    );
    assert!(
        fd.get("parametersJsonSchema").is_some(),
        "parametersJsonSchema must survive"
    );
    assert!(
        fd.get("responseJsonSchema").is_some(),
        "responseJsonSchema must survive"
    );
    assert_eq!(
        fd.get("behavior").and_then(Json::as_str),
        Some("BLOCKING"),
        "behavior must survive"
    );
}

#[test]
fn test_encode_patches_gemini_provider_native_tool_group() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"googleSearch": {"apiKey": "sk-tool-secret"}}]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    let tools = annotated.tools.as_mut().unwrap();
    match &mut tools[0] {
        ToolDefinition::ProviderNative { value, .. } => {
            *value = json!({"googleSearch": {"apiKey": "[REDACTED]"}});
        }
        other => panic!("expected Gemini ProviderNative tool, got {other:?}"),
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content.get("tools").unwrap(),
        &json!([{"googleSearch": {"apiKey": "[REDACTED]"}}])
    );
}

#[test]
fn test_encode_patches_mixed_gemini_provider_native_tool_group() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{
            "functionDeclarations": [{"name": "lookup"}],
            "googleSearch": {"apiKey": "sk-tool-secret"}
        }]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    let tools = annotated.tools.as_mut().unwrap();
    match &mut tools[1] {
        ToolDefinition::ProviderNative { value, .. } => {
            *value = json!({"googleSearch": {"apiKey": "[REDACTED]"}});
        }
        other => panic!("expected Gemini ProviderNative sibling fields, got {other:?}"),
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content.get("tools").unwrap(),
        &json!([{
            "functionDeclarations": [{"name": "lookup"}],
            "googleSearch": {"apiKey": "[REDACTED]"}
        }])
    );
}

#[test]
fn test_encode_deleting_mixed_native_sibling_does_not_rehome_later_native_group() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [
            {
                "functionDeclarations": [{"name": "lookup"}],
                "googleSearch": {}
            },
            {"codeExecution": {}}
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    annotated.tools.as_mut().unwrap().retain(|tool| {
        !matches!(
            tool,
            ToolDefinition::ProviderNative { provider, kind, .. }
                if provider == "gemini" && kind == "googleSearch"
        )
    });

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content.get("tools").unwrap(),
        &json!([
            {"functionDeclarations": [{"name": "lookup"}]},
            {"codeExecution": {}}
        ]),
        "deleting a native sibling must not merge a later native-only group \
         into the functionDeclarations group"
    );
}

// ===================================================================
// Function-call correlation IDs
// ===================================================================

#[test]
fn test_decode_response_function_call_uses_id_field() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "my_fn", "id": "call_abc123", "args": {"x": 1}}}
                ]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 5}
    });

    let ann = codec.decode_response(&response).unwrap();
    let tc = ann.tool_calls.unwrap();
    assert_eq!(tc.len(), 1);
    assert_eq!(
        tc[0].id, "call_abc123",
        "must use Gemini-provided id, not function name"
    );
    assert_eq!(tc[0].name, "my_fn");
}

#[test]
fn test_decode_response_function_call_fallback_to_name_when_no_id() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "fallback_fn", "args": {}}}]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {}
    });

    let ann = codec.decode_response(&response).unwrap();
    let tc = ann.tool_calls.unwrap();
    assert_eq!(
        tc[0].id, "fallback_fn",
        "must fall back to function name when id is absent"
    );
}

/// Two simultaneous calls to the same function must have distinct IDs.
#[test]
fn test_decode_response_multi_call_same_function_distinct_ids() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "search", "id": "call_001", "args": {"q": "a"}}},
                    {"functionCall": {"name": "search", "id": "call_002", "args": {"q": "b"}}}
                ]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {}
    });

    let ann = codec.decode_response(&response).unwrap();
    let tc = ann.tool_calls.unwrap();
    assert_eq!(tc.len(), 2);
    assert_eq!(tc[0].id, "call_001");
    assert_eq!(tc[1].id, "call_002");
    assert_ne!(
        tc[0].id, tc[1].id,
        "parallel calls to same function must have distinct IDs"
    );
    assert_eq!(tc[0].name, "search");
    assert_eq!(tc[1].name, "search");
}

/// After decode → encode, the functionResponse must contain the actual call ID (not
/// the function name) in the `id` field, and the correct function name in `name`.
#[test]
fn test_encode_function_response_id_not_name() {
    let codec = GeminiGenerateContentCodec;
    // A multi-turn request where Gemini provided a functionCall with an explicit id.
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "call my_fn"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"id": "call_abc123", "name": "my_fn", "args": {"x": 1}}}
                ]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Interceptor appends the tool result, using the ID from the decoded ToolCall.
    annotated.messages.push(Message::Tool {
        content: MessageContent::Text(r#"{"output": 42}"#.into()),
        tool_call_id: "call_abc123".into(),
    });

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    assert_eq!(contents.len(), 3);

    let tool_turn = &contents[2];
    assert_eq!(tool_turn.get("role").and_then(Json::as_str), Some("user"));
    let fr = tool_turn
        .get("parts")
        .and_then(Json::as_array)
        .and_then(|p| p.first())
        .and_then(|p| p.get("functionResponse"))
        .expect("must have functionResponse");
    assert_eq!(
        fr.get("id").and_then(Json::as_str),
        Some("call_abc123"),
        "functionResponse.id must be the actual call ID"
    );
    assert_eq!(
        fr.get("name").and_then(Json::as_str),
        Some("my_fn"),
        "functionResponse.name must be the function name, not the call ID"
    );
    // Sanity check: "call_abc123" must NOT appear as the name.
    assert_ne!(
        fr.get("name").and_then(Json::as_str),
        Some("call_abc123"),
        "functionResponse.name must not be the call ID"
    );
}

/// When a functionResponse is decoded from request history, the `id` field
/// (not the name) is stored in tool_call_id.
#[test]
fn test_decode_function_response_uses_id_not_name() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [
                {"functionCall": {"id": "call_xyz", "name": "my_fn", "args": {}}}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"id": "call_xyz", "name": "my_fn", "response": {"val": 1}}}
            ]}
        ]
    }));

    let annotated = codec.decode(&request).unwrap();
    // messages: [assistant_with_toolcall, tool_result]
    // (no system message, so indices are direct)
    let tool_msg = annotated
        .messages
        .iter()
        .find(|m| matches!(m, Message::Tool { .. }))
        .expect("must decode a Message::Tool");
    if let Message::Tool { tool_call_id, .. } = tool_msg {
        assert_eq!(
            tool_call_id, "call_xyz",
            "tool_call_id must be the actual id field, not the function name"
        );
    }
}

/// When a system message is added (system → systemInstruction, not contents), the
/// original contents items are left untouched.
#[test]
fn test_encode_system_message_added_leaves_contents_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "original user"}]},
            {"role": "model", "parts": [{"text": "original model"}]}
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    annotated.messages.insert(
        0,
        Message::System {
            content: MessageContent::Text("system context".into()),
            name: None,
        },
    );

    let encoded = codec.encode(&annotated, &original).unwrap();

    assert!(encoded.content.get("systemInstruction").is_some());
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    assert_eq!(contents.len(), 2);
    assert_eq!(contents[0].get("role").and_then(Json::as_str), Some("user"));
    assert_eq!(
        contents[1].get("role").and_then(Json::as_str),
        Some("model")
    );
}

/// Inserting a new non-system message increases the contents length; the new
/// message must be encoded correctly and the appended position must use
/// normalized_to_gemini_content (not a wrong original as base).
#[test]
fn test_encode_new_user_message_appended_encodes_correctly() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]}
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Append a new user message (contents grows from 2 → 3).
    annotated.messages.push(Message::User {
        content: MessageContent::Text("follow up question".into()),
        name: None,
    });

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();

    assert_eq!(contents.len(), 3);
    // Original positions unchanged.
    assert_eq!(contents[0].get("role").and_then(Json::as_str), Some("user"));
    assert_eq!(
        contents[1].get("role").and_then(Json::as_str),
        Some("model")
    );
    // New position encoded fresh — correct role and text.
    assert_eq!(contents[2].get("role").and_then(Json::as_str), Some("user"));
    let new_text = contents[2]
        .get("parts")
        .and_then(Json::as_array)
        .and_then(|p| p.first())
        .and_then(|p| p.get("text"))
        .and_then(Json::as_str);
    assert_eq!(
        new_text,
        Some("follow up question"),
        "new message text must be encoded correctly"
    );
}

// ===================================================================
// Streaming: function call parts accumulation
// ===================================================================

#[test]
fn test_streaming_accumulates_function_call_parts() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"thought": true, "text": "deciding..."}]},
            "index": 0
        }]
    }))
    .unwrap();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [
                {"functionCall": {"name": "tool_a", "id": "c1", "args": {"x": 1}}}
            ]},
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14}
    }))
    .unwrap();

    let assembled = finalizer();
    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();

    assert_eq!(resp.finish_reason, Some(FinishReason::ToolUse));
    let tc = resp.tool_calls.unwrap();
    assert_eq!(tc.len(), 1);
    assert_eq!(tc[0].id, "c1");
    assert_eq!(tc[0].name, "tool_a");
}

/// When two parallel calls to the SAME function each carry a distinct
/// thoughtSignature, both signatures must survive a decode → edit-args → encode
/// round-trip.  A HashMap lookup would drop the first entry (same key), leaving
/// both encoded parts with the second part's signature.
#[test]
fn test_encode_thought_signature_preserved_for_parallel_same_function_calls() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "run two searches"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "search", "id": "c1", "args": {"q": "first"}},
                        "thoughtSignature": "sig_FIRST=="
                    },
                    {
                        "functionCall": {"name": "search", "id": "c2", "args": {"q": "second"}},
                        "thoughtSignature": "sig_SECOND=="
                    }
                ]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Interceptor edits the first call's args; second is unchanged.
    if let Message::Assistant {
        tool_calls: Some(tcs),
        ..
    } = &mut annotated.messages[1]
        && let Some(tc) = tcs.first_mut()
    {
        tc.function.arguments = r#"{"q": "first-edited"}"#.to_string();
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    let model_parts = contents[1].get("parts").and_then(Json::as_array).unwrap();

    // There must be exactly two functionCall parts.
    let fc_parts: Vec<&Json> = model_parts
        .iter()
        .filter(|p| p.get("functionCall").is_some())
        .collect();
    assert_eq!(fc_parts.len(), 2, "both functionCall parts must be encoded");

    // The first encoded part must carry sig_FIRST (not sig_SECOND).
    assert_eq!(
        fc_parts[0].get("thoughtSignature").and_then(Json::as_str),
        Some("sig_FIRST=="),
        "first functionCall part must retain its own thoughtSignature"
    );
    // The second encoded part must carry sig_SECOND.
    assert_eq!(
        fc_parts[1].get("thoughtSignature").and_then(Json::as_str),
        Some("sig_SECOND=="),
        "second functionCall part must retain its own thoughtSignature"
    );
    // Verify the first call's args were actually updated.
    let first_args = fc_parts[0]
        .get("functionCall")
        .and_then(|fc| fc.get("args"))
        .unwrap();
    assert_eq!(
        first_args.get("q").and_then(Json::as_str),
        Some("first-edited")
    );
}

/// Reordering same-name parallel function calls must not swap their
/// thoughtSignature fields. Match original parts by Gemini's stable call ID,
/// falling back to name only for provider payloads that omitted an ID.
#[test]
fn test_encode_reordered_same_name_function_calls_match_signatures_by_id() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "run two searches"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "search", "id": "c1", "args": {"q": "first"}},
                        "thoughtSignature": "sig_FIRST=="
                    },
                    {
                        "functionCall": {"name": "search", "id": "c2", "args": {"q": "second"}},
                        "thoughtSignature": "sig_SECOND=="
                    }
                ]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    if let Message::Assistant {
        tool_calls: Some(tcs),
        ..
    } = &mut annotated.messages[1]
    {
        tcs.swap(0, 1);
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    let model_parts = contents[1].get("parts").and_then(Json::as_array).unwrap();
    let fc_parts: Vec<&Json> = model_parts
        .iter()
        .filter(|p| p.get("functionCall").is_some())
        .collect();

    assert_eq!(fc_parts.len(), 2);
    assert_eq!(fc_parts[0]["functionCall"]["id"].as_str(), Some("c2"));
    assert_eq!(
        fc_parts[0].get("thoughtSignature").and_then(Json::as_str),
        Some("sig_SECOND==")
    );
    assert_eq!(fc_parts[1]["functionCall"]["id"].as_str(), Some("c1"));
    assert_eq!(
        fc_parts[1].get("thoughtSignature").and_then(Json::as_str),
        Some("sig_FIRST==")
    );
}

/// An interceptor that inserts a new system message at position 0 shifts the message-list
/// indices but MUST NOT cause thoughtSignature bleed onto the new position or cause the
/// model turn's thoughtSignature to be lost.
///
/// System messages map to `systemInstruction`, not to `contents`, so the contents array
/// indices are unaffected by system-message insertion.  The model turn stays at contents
/// index 1; only the message-list index changes from 1 to 2.
#[test]
fn test_encode_insertion_does_not_bleed_thought_signature() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "user message"}]},
            {
                "role": "model",
                "parts": [{
                    "functionCall": {"id": "c1", "name": "fn", "args": {}},
                    "thoughtSignature": "sig_abc"
                }]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Interceptor inserts a new system message at position 0.
    // Message list becomes: [new_system, user, model]
    annotated.messages.insert(
        0,
        Message::System {
            content: MessageContent::Text("new system prompt".into()),
            name: None,
        },
    );

    let encoded = codec.encode(&annotated, &original).unwrap();

    // systemInstruction must carry the new text.
    let sys = encoded
        .content
        .get("systemInstruction")
        .expect("systemInstruction must be set");
    let sys_text = sys
        .get("parts")
        .and_then(Json::as_array)
        .and_then(|p| p.first())
        .and_then(|p| p.get("text"))
        .and_then(Json::as_str)
        .unwrap();
    assert_eq!(sys_text, "new system prompt");

    // contents array must still have exactly 2 items (system → systemInstruction only).
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    assert_eq!(
        contents.len(),
        2,
        "system insertion must not change the contents array length"
    );

    // Position 0 (user): must NOT have thoughtSignature.
    let user_turn = &contents[0];
    assert_eq!(user_turn.get("role").and_then(Json::as_str), Some("user"));
    let user_parts = user_turn.get("parts").and_then(Json::as_array).unwrap();
    assert!(
        user_parts
            .iter()
            .all(|p| p.get("thoughtSignature").is_none()),
        "user turn must not have thoughtSignature"
    );

    // Position 1 (model): must retain thoughtSignature from the original.
    let model_turn = &contents[1];
    assert_eq!(model_turn.get("role").and_then(Json::as_str), Some("model"));
    let model_parts = model_turn.get("parts").and_then(Json::as_array).unwrap();
    let fc_part = model_parts
        .iter()
        .find(|p| p.get("functionCall").is_some())
        .expect("model turn must still have functionCall part");
    assert_eq!(
        fc_part.get("thoughtSignature").and_then(Json::as_str),
        Some("sig_abc"),
        "thoughtSignature must survive system-message insertion at position 0"
    );
    assert_eq!(
        fc_part
            .get("functionCall")
            .and_then(|fc| fc.get("id"))
            .and_then(Json::as_str),
        Some("c1"),
        "functionCall id must survive system-message insertion"
    );
    assert_eq!(
        fc_part
            .get("functionCall")
            .and_then(|fc| fc.get("name"))
            .and_then(Json::as_str),
        Some("fn"),
        "functionCall name must survive system-message insertion"
    );
}

// ===================================================================
// Prepend non-system user message — content-index insertion regression
// ===================================================================

/// Prepending a new non-system user message must not corrupt the model turn's
/// thoughtSignature. The new message should be encoded fresh; the original user
/// and model turns must be preserved byte-identically.
#[test]
fn test_encode_prepend_user_message_preserves_thought_signature() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "original question"}]},
            {
                "role": "model",
                "parts": [{
                    "functionCall": {"id": "c1", "name": "search", "args": {}},
                    "thoughtSignature": "sig_PREPEND_TEST=="
                }]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Prepend a brand-new user message before the existing user message.
    // Message list becomes: [new_user, old_user, model_with_signature]
    annotated.messages.insert(
        0,
        Message::User {
            content: MessageContent::Text("context preamble".into()),
            name: None,
        },
    );

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();

    assert_eq!(contents.len(), 3, "prepend must produce 3 content items");

    // Position 0: freshly encoded new user message.
    assert_eq!(contents[0].get("role").and_then(Json::as_str), Some("user"));
    let new_text = contents[0]
        .get("parts")
        .and_then(Json::as_array)
        .and_then(|p| p.first())
        .and_then(|p| p.get("text"))
        .and_then(Json::as_str);
    assert_eq!(
        new_text,
        Some("context preamble"),
        "prepended message must have the correct text"
    );

    // Positions 1 and 2: original items preserved byte-identically.
    let orig_contents = original
        .content
        .get("contents")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        &contents[1], &orig_contents[0],
        "original user turn at position 1 must be preserved byte-identically"
    );
    assert_eq!(
        &contents[2], &orig_contents[1],
        "model turn with thoughtSignature at position 2 must be preserved byte-identically"
    );

    // Explicit check that the thoughtSignature survived.
    let model_parts = contents[2].get("parts").and_then(Json::as_array).unwrap();
    let fc_part = model_parts
        .iter()
        .find(|p| p.get("functionCall").is_some())
        .expect("model turn must have functionCall part");
    assert_eq!(
        fc_part.get("thoughtSignature").and_then(Json::as_str),
        Some("sig_PREPEND_TEST=="),
        "thoughtSignature must survive prepend of a non-system user message"
    );
}

// ===================================================================
// thoughtSignature on a regular text part
// ===================================================================

/// A model turn where the response text part itself carries a thoughtSignature
/// must have that signature preserved when an interceptor edits the text.
#[test]
fn test_encode_thought_signature_on_text_part_survives_edit() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "say hello"}]},
            {
                "role": "model",
                "parts": [{"text": "Hello!", "thoughtSignature": "sig_TEXT=="}]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Interceptor changes the assistant's reply text.
    if let Message::Assistant { content, .. } = &mut annotated.messages[1] {
        *content = Some(MessageContent::Text("Goodbye!".into()));
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    let model_parts = contents[1].get("parts").and_then(Json::as_array).unwrap();

    let text_part = model_parts
        .first()
        .expect("model turn must have a text part");
    assert_eq!(
        text_part.get("text").and_then(Json::as_str),
        Some("Goodbye!"),
        "text must be updated to the interceptor's new value"
    );
    assert_eq!(
        text_part.get("thoughtSignature").and_then(Json::as_str),
        Some("sig_TEXT=="),
        "thoughtSignature on a regular text part must survive when text is edited"
    );
}

// ===================================================================
// Parallel functionResponse parts in a single content item
// ===================================================================

/// A single user-role content item with multiple functionResponse parts
/// (Gemini's parallel-call response format) must decode to one Message::Tool
/// per part — not collapse to just the first.
#[test]
fn test_decode_multiple_function_responses_in_one_turn() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "run two tools"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"id": "c1", "name": "tool_a", "args": {}}},
                    {"functionCall": {"id": "c2", "name": "tool_b", "args": {}}}
                ]
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"id": "c1", "name": "tool_a", "response": {"r": 1}}},
                    {"functionResponse": {"id": "c2", "name": "tool_b", "response": {"r": 2}}}
                ]
            }
        ]
    }));

    let annotated = codec.decode(&request).unwrap();

    // Expect two separate Message::Tool entries from the multi-functionResponse turn.
    let tool_msgs: Vec<&Message> = annotated
        .messages
        .iter()
        .filter(|m| matches!(m, Message::Tool { .. }))
        .collect();
    assert_eq!(
        tool_msgs.len(),
        2,
        "each functionResponse part must produce a separate Message::Tool"
    );

    let ids: Vec<&str> = tool_msgs
        .iter()
        .filter_map(|m| {
            if let Message::Tool { tool_call_id, .. } = m {
                Some(tool_call_id.as_str())
            } else {
                None
            }
        })
        .collect();
    assert!(
        ids.contains(&"c1"),
        "first functionResponse id must be decoded"
    );
    assert!(
        ids.contains(&"c2"),
        "second functionResponse id must be decoded"
    );
}

#[test]
fn test_decode_function_response_with_native_sibling_errors() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"functionResponse": {"id": "c1", "name": "lookup", "response": {"ok": true}}},
                {"inlineData": {"mimeType": "text/plain", "data": "sk-hidden-secret"}}
            ]
        }]
    }));

    assert!(
        codec.decode(&request).is_err(),
        "native sibling content beside functionResponse would be hidden by Message::Tool decode"
    );
}

#[test]
fn test_function_response_nested_parts_are_exposed_and_patchable() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "show my ordered instrument"}]},
            {"role": "model", "parts": [{
                "functionCall": {"id": "call_img", "name": "get_image", "args": {}},
                "thoughtSignature": "sig_CALL=="
            }]},
            {"role": "user", "parts": [{
                "functionResponse": {
                    "id": "call_img",
                    "name": "get_image",
                    "response": {"image_ref": {"$ref": "instrument.jpg"}},
                    "parts": [{
                        "inlineData": {
                            "displayName": "instrument.jpg",
                            "mimeType": "image/jpeg",
                            "data": "sk-image-secret"
                        }
                    }]
                }
            }]}
        ]
    }));

    let mut annotated = codec.decode(&request).unwrap();
    let Message::Tool { content, .. } = &mut annotated.messages[2] else {
        panic!("expected functionResponse to decode as Message::Tool");
    };
    let MessageContent::Parts(parts) = content else {
        panic!("functionResponse.parts must be exposed as normalized content parts");
    };
    assert!(matches!(
        &parts[0],
        ContentPart::Text { text, .. }
            if text.contains("instrument.jpg") && text.contains("image_ref")
    ));
    match &mut parts[1] {
        ContentPart::ProviderNative {
            provider,
            kind,
            value,
        } => {
            assert_eq!(provider.as_str(), "gemini");
            assert_eq!(kind.as_str(), "inlineData");
            assert_eq!(value["inlineData"]["data"], json!("sk-image-secret"));
            value["inlineData"]["data"] = json!("[REDACTED]");
        }
        other => panic!("expected nested Gemini functionResponse part, got {other:?}"),
    }

    let encoded = codec.encode(&annotated, &request).unwrap();
    let fr = &encoded.content["contents"][2]["parts"][0]["functionResponse"];
    assert_eq!(
        fr["response"]["image_ref"]["$ref"],
        json!("instrument.jpg"),
        "functionResponse.response must remain object-shaped"
    );
    assert_eq!(
        fr["parts"][0]["inlineData"]["data"],
        json!("[REDACTED]"),
        "nested functionResponse.parts must be editable through normalized ProviderNative content"
    );
}

// ===================================================================
// Request decode validation
// ===================================================================

#[test]
fn test_decode_rejects_missing_contents() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({"model": "gemini-2.0-flash"}));
    assert!(
        codec.decode(&request).is_err(),
        "missing contents must return an error"
    );
}

#[test]
fn test_decode_rejects_non_array_contents() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({"contents": "not an array"}));
    assert!(
        codec.decode(&request).is_err(),
        "non-array contents must return an error"
    );
}

#[test]
fn test_decode_rejects_non_object_generation_config() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": "not an object"
    }));
    assert!(
        codec.decode(&request).is_err(),
        "non-object generationConfig must return an error"
    );
}

#[test]
fn test_decode_rejects_invalid_stop_sequences() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"stopSequences": "not an array"}
    }));
    assert!(
        codec.decode(&request).is_err(),
        "non-array stopSequences must return an error"
    );
}

#[test]
fn test_decode_rejects_non_array_tools() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": "not an array"
    }));
    assert!(
        codec.decode(&request).is_err(),
        "non-array tools must return an error"
    );
}

#[test]
fn test_decode_rejects_function_declaration_without_name() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"functionDeclarations": [{"description": "no name"}]}]
    }));
    assert!(
        codec.decode(&request).is_err(),
        "functionDeclaration without name must return an error"
    );
}

// ===================================================================
// Encoder: invalid role rejection
// ===================================================================

/// Encoding a message with a role unknown to Gemini (e.g. "developer")
/// must return an error — silently dropping it would be data loss.
#[test]
fn test_encode_unsupported_role_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Inject a message with an unsupported Gemini role.
    annotated.messages.push(
        serde_json::from_value(json!({"role": "developer", "content": "system note"})).unwrap(),
    );
    let result = codec.encode(&annotated, &original);
    assert!(
        result.is_err(),
        "encode must return an error for an unsupported role rather than silently dropping the message"
    );
}

// ===================================================================
// Parallel functionResponse: deletion and insertion
// ===================================================================

/// Deleting one of two parallel tool results must produce a single-part
/// functionResponse content item — not resurrect the deleted result.
#[test]
fn test_encode_delete_one_parallel_tool_response() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "run two tools"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"id": "c1", "name": "tool_a", "args": {}}},
                    {"functionCall": {"id": "c2", "name": "tool_b", "args": {}}}
                ]
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"id": "c1", "name": "tool_a", "response": {"r": 1}}},
                    {"functionResponse": {"id": "c2", "name": "tool_b", "response": {"r": 2}}}
                ]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Delete the second tool result (tool_b / c2) from the annotated list.
    annotated
        .messages
        .retain(|m| !matches!(m, Message::Tool { tool_call_id, .. } if tool_call_id == "c2"));

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();

    // Find the functionResponse content item(s).
    let fr_items: Vec<&Json> = contents
        .iter()
        .filter(|c| {
            c.get("parts")
                .and_then(Json::as_array)
                .map(|p| p.iter().any(|part| part.get("functionResponse").is_some()))
                .unwrap_or(false)
        })
        .collect();

    // The deleted tool result must not appear in any functionResponse.
    for item in &fr_items {
        let parts = item.get("parts").and_then(Json::as_array).unwrap();
        for part in parts {
            if let Some(fr) = part.get("functionResponse") {
                assert_ne!(
                    fr.get("id").and_then(Json::as_str),
                    Some("c2"),
                    "deleted tool result c2 must not appear in encoded contents"
                );
                assert_ne!(
                    fr.get("name").and_then(Json::as_str),
                    Some("tool_b"),
                    "deleted tool result tool_b must not appear in encoded contents"
                );
            }
        }
    }
    // Exactly one functionResponse part (c1) must survive.
    let fr_part_count: usize = fr_items
        .iter()
        .map(|item| {
            item.get("parts")
                .and_then(Json::as_array)
                .map(|p| {
                    p.iter()
                        .filter(|part| part.get("functionResponse").is_some())
                        .count()
                })
                .unwrap_or(0)
        })
        .sum();
    assert_eq!(
        fr_part_count, 1,
        "exactly one functionResponse (c1) must survive deletion of c2"
    );
}

/// Inserting a new tool response between two existing parallel results must
/// not duplicate or resurrect the original item.
#[test]
fn test_encode_insert_between_parallel_tool_responses() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "run two tools"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"id": "c1", "name": "tool_a", "args": {}}},
                    {"functionCall": {"id": "c2", "name": "tool_b", "args": {}}}
                ]
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"id": "c1", "name": "tool_a", "response": {"r": 1}}},
                    {"functionResponse": {"id": "c2", "name": "tool_b", "response": {"r": 2}}}
                ]
            }
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();

    // Find positions of c1 and c2 tool messages.
    let c1_pos = annotated
        .messages
        .iter()
        .position(|m| matches!(m, Message::Tool { tool_call_id, .. } if tool_call_id == "c1"))
        .unwrap();

    // Insert a new tool result (c3) between c1 and c2.
    annotated.messages.insert(
        c1_pos + 1,
        Message::Tool {
            content: MessageContent::Text(r#"{"r": 99}"#.into()),
            tool_call_id: "c3".into(),
        },
    );

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();

    // Collect all functionResponse parts across all content items.
    let fr_ids: Vec<&str> = contents
        .iter()
        .flat_map(|c| {
            c.get("parts")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|part| {
            part.get("functionResponse")
                .and_then(|fr| fr.get("id"))
                .and_then(Json::as_str)
        })
        .collect();

    // c1, c2, and the new c3 must all appear exactly once — no duplicates.
    assert_eq!(
        fr_ids.iter().filter(|&&id| id == "c1").count(),
        1,
        "c1 must appear exactly once"
    );
    assert_eq!(
        fr_ids.iter().filter(|&&id| id == "c2").count(),
        1,
        "c2 must appear exactly once"
    );
    assert_eq!(
        fr_ids.iter().filter(|&&id| id == "c3").count(),
        1,
        "new c3 must be encoded"
    );
}

// ===================================================================
// thoughtsTokenCount accounting
// ===================================================================

/// thoughtsTokenCount must be stored in ApiSpecificResponse::GeminiGenerateContent and
/// must be included in the fallback total_tokens calculation.
#[test]
fn test_decode_response_thoughts_token_count() {
    use super::super::response::ApiSpecificResponse;

    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "let me think..."}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 20
            // totalTokenCount intentionally absent to test fallback
        }
    });

    let resp = codec.decode_response(&response).unwrap();
    let usage = resp.usage.unwrap();

    // completion_tokens reflects only candidatesTokenCount (not thinking tokens).
    assert_eq!(usage.completion_tokens, Some(5));
    // Fallback total must include thinking tokens: 10 + 5 + 20 = 35.
    assert_eq!(
        usage.total_tokens,
        Some(35),
        "fallback total_tokens must include thoughtsTokenCount"
    );

    // thoughts_tokens must be in ApiSpecificResponse::GeminiGenerateContent.
    match resp.api_specific {
        Some(ApiSpecificResponse::GeminiGenerateContent {
            thoughts_tokens, ..
        }) => {
            assert_eq!(
                thoughts_tokens,
                Some(20),
                "thoughtsTokenCount must be in api_specific"
            );
        }
        other => panic!("expected ApiSpecificResponse::GeminiGenerateContent, got: {other:?}"),
    }
}

// Roleless contents decode to user messages (Google allows omitting role).
#[test]
fn test_decode_roleless_content_treated_as_user() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [
            {"parts": [{"text": "hello without a role"}]},
            {"role": "model", "parts": [{"text": "hi"}]}
        ]
    }));
    let annotated = codec.decode(&request).unwrap();
    let non_sys: Vec<&Message> = annotated
        .messages
        .iter()
        .filter(|m| !matches!(m, Message::System { .. }))
        .collect();
    assert_eq!(non_sys.len(), 2);
    assert!(
        matches!(non_sys[0], Message::User { .. }),
        "roleless content must decode as a user message"
    );
    if let Message::User { content, .. } = non_sys[0] {
        assert_eq!(
            content,
            &MessageContent::Text("hello without a role".into()),
            "roleless content text must be preserved"
        );
    }
}

// Wrong numeric types in generationConfig must return an error.
#[test]
fn test_decode_rejects_non_numeric_temperature() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"temperature": "hot"}
    }));
    assert!(
        codec.decode(&request).is_err(),
        "non-numeric temperature must return an error"
    );
}

#[test]
fn test_decode_rejects_non_numeric_max_output_tokens() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": "a lot"}
    }));
    assert!(
        codec.decode(&request).is_err(),
        "non-integer maxOutputTokens must return an error"
    );
}

// Streaming finalizer must propagate responseId when present.
#[test]
fn test_streaming_propagates_response_id() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hello"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        "responseId": "stream-resp-xyz",
        "modelVersion": "gemini-2.0-flash"
    }))
    .unwrap();

    let assembled = finalizer();

    assert_eq!(
        assembled.get("responseId").and_then(Json::as_str),
        Some("stream-resp-xyz"),
        "streaming finalizer must propagate responseId from SSE events"
    );

    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();
    assert_eq!(
        resp.id.as_deref(),
        Some("stream-resp-xyz"),
        "responseId from streaming must survive decode_response"
    );
}

// MAX_TOKENS + functionCall parts → Length, not ToolUse.
// Explicit reasons must not be overridden by the tool-call presence heuristic.
#[test]
fn test_decode_response_max_tokens_with_function_call_is_length_not_tool_use() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "fn", "args": {}}}]
            },
            "finishReason": "MAX_TOKENS",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 5}
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.finish_reason,
        Some(FinishReason::Length),
        "MAX_TOKENS must map to Length even when functionCall parts are present"
    );
}

// SAFETY + functionCall parts → ContentFilter, not ToolUse.
#[test]
fn test_decode_response_safety_with_function_call_is_content_filter_not_tool_use() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "fn", "args": {}}}]
            },
            "finishReason": "SAFETY",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 5}
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.finish_reason,
        Some(FinishReason::ContentFilter),
        "SAFETY must map to ContentFilter even when functionCall parts are present"
    );
}

// STOP + functionCall parts → ToolUse (unchanged behaviour).
#[test]
fn test_decode_response_stop_with_function_call_is_tool_use() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "fn", "args": {}}}]
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 5}
    });
    let resp = codec.decode_response(&response).unwrap();
    assert_eq!(
        resp.finish_reason,
        Some(FinishReason::ToolUse),
        "STOP + functionCall must map to ToolUse"
    );
}

// Roleless content items (no explicit "role") that are later edited must not
// be silently dropped by patch_changed_gemini_content.
#[test]
fn test_encode_roleless_original_item_survives_edit() {
    let codec = GeminiGenerateContentCodec;
    // A Gemini request whose user content item has no "role" field.
    let original = make_request(json!({
        "contents": [{"parts": [{"text": "original text"}]}]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    // Change the text (triggers the patch path, not the unchanged path).
    if let Message::User { content, .. } = &mut annotated.messages[0] {
        *content = MessageContent::Text("edited text".into());
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let contents = encoded
        .content
        .get("contents")
        .and_then(Json::as_array)
        .unwrap();
    assert_eq!(
        contents.len(),
        1,
        "roleless item must not be dropped on edit"
    );
    let text = contents[0]
        .get("parts")
        .and_then(Json::as_array)
        .and_then(|p| p.first())
        .and_then(|p| p.get("text"))
        .and_then(Json::as_str);
    assert_eq!(
        text,
        Some("edited text"),
        "edited text must appear in output"
    );
}

// Multiple system messages must be merged into one systemInstruction.
#[test]
fn test_encode_multiple_system_messages_merged() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Inject two system messages.
    annotated.messages.insert(
        0,
        Message::System {
            content: MessageContent::Text("part one".into()),
            name: None,
        },
    );
    annotated.messages.insert(
        1,
        Message::System {
            content: MessageContent::Text("part two".into()),
            name: None,
        },
    );

    let encoded = codec.encode(&annotated, &original).unwrap();
    let sys = encoded
        .content
        .get("systemInstruction")
        .and_then(|s| s.get("parts"))
        .and_then(Json::as_array)
        .and_then(|p| p.first())
        .and_then(|p| p.get("text"))
        .and_then(Json::as_str)
        .unwrap();
    assert!(
        sys.contains("part one") && sys.contains("part two"),
        "multiple system messages must be merged into one systemInstruction"
    );
}

// Encoding unsupported fields must error, not silently ignore.
#[test]
fn test_encode_unsupported_field_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Set a field that has no Gemini equivalent.
    annotated.max_output_tokens = Some(512);
    let result = codec.encode(&annotated, &original);
    assert!(
        result.is_err(),
        "setting max_output_tokens must return an error"
    );
}

// Unparsable tool-call arguments must return an error rather than becoming {}.
#[test]
fn test_encode_invalid_tool_call_arguments_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.messages.push(Message::Assistant {
        content: None,
        name: None,
        tool_calls: Some(vec![super::super::request::ToolCall {
            id: "c1".into(),
            call_type: "function".into(),
            function: super::super::request::FunctionCall {
                name: "fn".into(),
                arguments: "not valid json {{".into(),
            },
        }]),
    });
    let result = codec.encode(&annotated, &original);
    assert!(
        result.is_err(),
        "invalid tool-call arguments must return an error"
    );
}

// Thinking-only partial usage (no candidatesTokenCount).
#[test]
fn test_decode_response_thoughts_only_usage_computes_fallback_total() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": []},
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            // candidatesTokenCount intentionally absent
            "thoughtsTokenCount": 30
        }
    });
    let resp = codec.decode_response(&response).unwrap();
    let usage = resp.usage.unwrap();
    assert_eq!(
        usage.completion_tokens, None,
        "candidatesTokenCount absent → completion_tokens None"
    );
    assert_eq!(
        usage.total_tokens,
        Some(40),
        "fallback total must be prompt(10) + thoughts(30) = 40 even without candidatesTokenCount"
    );
}

// Decode validation: table-driven negative tests for malformed contents items.
#[test]
fn test_decode_contents_item_validation() {
    let codec = GeminiGenerateContentCodec;

    let cases: &[(&str, Json)] = &[
        (
            "contents item not an object",
            json!({"contents": ["not an object"]}),
        ),
        ("parts missing", json!({"contents": [{"role": "user"}]})),
        (
            "parts not an array",
            json!({"contents": [{"role": "user", "parts": "bad"}]}),
        ),
        (
            "explicit unknown role",
            json!({"contents": [{"role": "system", "parts": []}]}),
        ),
        (
            "part not an object",
            json!({"contents": [{"role": "user", "parts": ["not an object"]}]}),
        ),
        (
            "functionCall not an object",
            json!({"contents": [{"role": "model", "parts": [{"functionCall": "bad"}]}]}),
        ),
        (
            "functionCall missing name",
            json!({"contents": [{"role": "model", "parts": [{"functionCall": {"args": {}}}]}]}),
        ),
        (
            "functionCall empty name",
            json!({"contents": [{"role": "model", "parts": [{"functionCall": {"name": "", "args": {}}}]}]}),
        ),
        (
            "functionResponse not an object",
            json!({"contents": [{"role": "user", "parts": [{"functionResponse": "bad"}]}]}),
        ),
        (
            "functionResponse missing name",
            json!({"contents": [{"role": "user", "parts": [{"functionResponse": {"response": {}}}]}]}),
        ),
        (
            "functionResponse.response not an object",
            json!({"contents": [{"role": "user", "parts": [{"functionResponse": {"name": "fn", "response": "bad"}}]}]}),
        ),
    ];

    for (label, body) in cases {
        let req = make_request(body.clone());
        assert!(
            codec.decode(&req).is_err(),
            "case '{label}' must return an error"
        );
    }
}

// Valid roleless contents item must decode successfully as a user message.
#[test]
fn test_decode_roleless_contents_item_is_valid() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"parts": [{"text": "hello"}]}]
    }));
    let ann = codec.decode(&req).unwrap();
    assert!(
        matches!(&ann.messages[0], Message::User { content: MessageContent::Text(t), .. } if t == "hello"),
        "roleless contents item must decode as a user message"
    );
}

// Encoding a normalized message whose content has non-text parts must error.
#[test]
fn test_encode_non_text_content_part_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Replace the user message with one that has an image-url part in its content.
    annotated.messages[0] = serde_json::from_value(json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]
    }))
    .unwrap();
    let result = codec.encode(&annotated, &original);
    assert!(
        result.is_err(),
        "non-text content part (image_url) must return an error, not be silently dropped"
    );
}

// Encoding a newly-inserted user message with non-text content must also error.
#[test]
fn test_encode_inserted_non_text_content_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "original"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.messages.push(
        serde_json::from_value(json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        }))
        .unwrap(),
    );
    let result = codec.encode(&annotated, &original);
    assert!(
        result.is_err(),
        "inserting a message with non-text content must error, not produce empty text"
    );
}

// Editing tools when the original request has multiple functionDeclarations
// groups must return an error rather than silently collapsing them.
#[test]
fn test_encode_multiple_fn_decl_groups_error_on_edit() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [
            {"functionDeclarations": [{"name": "fn_a", "description": "group 1"}]},
            {"functionDeclarations": [{"name": "fn_b", "description": "group 2"}]}
        ]
    }));

    let mut annotated = codec.decode(&original).unwrap();
    // Change a tool to trigger the tools-changed path.
    if let Some(tools) = annotated.tools.as_mut() {
        for td in tools.iter_mut() {
            if let nemo_relay_types::codec::request::ToolDefinition::Function { function, .. } = td
            {
                function.description = Some("edited".into());
            }
        }
    }

    let result = codec.encode(&annotated, &original);
    assert!(
        result.is_err(),
        "editing tools when original has multiple functionDeclarations groups must error"
    );
}

// Streaming: candidate metadata (safetyRatings, groundingMetadata) must survive finalization.
#[test]
fn test_streaming_candidate_extras_survive_finalize() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();

    collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hello"}]},
            "finishReason": "STOP",
            "index": 0,
            "safetyRatings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}
            ],
            "groundingMetadata": {"webSearchQueries": ["example query"]}
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}
    }))
    .unwrap();

    let assembled = finalizer();

    let codec = GeminiGenerateContentCodec;
    let resp = codec.decode_response(&assembled).unwrap();

    use super::super::response::ApiSpecificResponse;
    match &resp.api_specific {
        Some(ApiSpecificResponse::GeminiGenerateContent {
            safety_ratings,
            grounding_metadata,
            ..
        }) => {
            assert!(
                safety_ratings.is_some(),
                "safetyRatings from streaming must survive in ApiSpecificResponse::GeminiGenerateContent"
            );
            assert!(
                grounding_metadata.is_some(),
                "groundingMetadata from streaming must survive in ApiSpecificResponse::GeminiGenerateContent"
            );
        }
        other => panic!("expected ApiSpecificResponse::GeminiGenerateContent, got: {other:?}"),
    }
}

// Present-but-non-string role must error.
#[test]
fn test_decode_non_string_role_errors() {
    let codec = GeminiGenerateContentCodec;
    for bad_role in [json!(123), json!(null), json!(true), json!([])] {
        let req = make_request(json!({
            "contents": [{"role": bad_role, "parts": [{"text": "hi"}]}]
        }));
        assert!(
            codec.decode(&req).is_err(),
            "role={bad_role} must error; only string 'user'/'model' are accepted"
        );
    }
}

// functionResponse.response is required; missing must error.
#[test]
fn test_decode_function_response_missing_response_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"role": "user", "parts": [
            {"functionResponse": {"id": "c1", "name": "fn"}}
        ]}]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "missing functionResponse.response must error"
    );
}

#[test]
fn test_function_response_content_helper_missing_response_errors() {
    assert!(gemini_function_response_to_message_content(&json!({"name": "fn"})).is_err());
}

// functionResponse.response must be an object.
#[test]
fn test_decode_function_response_non_object_response_errors() {
    let codec = GeminiGenerateContentCodec;
    for bad in [json!("string"), json!([1, 2]), json!(42)] {
        let req = make_request(json!({
            "contents": [{"role": "user", "parts": [
                {"functionResponse": {"name": "fn", "response": bad}}
            ]}]
        }));
        assert!(
            codec.decode(&req).is_err(),
            "response={bad} must error; response must be an object"
        );
    }
}

// Tool encode: non-object content is wrapped in {"output": ...}.
#[test]
fn test_encode_tool_content_non_object_is_wrapped() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"functionCall": {"id": "c1", "name": "fn", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"id": "c1", "name": "fn", "response": {"val": 1}}}]}
        ]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Interceptor sets tool content to a bare number (parses as JSON but not object).
    if let Message::Tool { content, .. } = &mut annotated.messages[2] {
        *content = MessageContent::Text("42".into());
    }
    let encoded = codec.encode(&annotated, &original).unwrap();
    let fr = encoded.content["contents"][2]["parts"][0]["functionResponse"]["response"].clone();
    assert!(
        fr.is_object(),
        "non-object tool content must be wrapped as an object"
    );
    assert_eq!(fr["output"], json!(42));
}

// System message with non-text content must error.
#[test]
fn test_encode_system_message_non_text_content_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.messages.insert(
        0,
        serde_json::from_value(json!({
            "role": "system",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
        }))
        .unwrap(),
    );
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "system message with image content must error"
    );
}

// Tool message with non-text normalized content must error.
#[test]
fn test_encode_tool_message_non_text_content_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"functionCall": {"id": "c1", "name": "fn", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"id": "c1", "name": "fn", "response": {}}}]}
        ]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Use image_url — a ContentPart variant that round-trips through serde but has
    // no Gemini encoding, so the guard must reject it.
    if let Message::Tool { content, .. } = &mut annotated.messages[2] {
        *content = serde_json::from_value(json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]))
        .unwrap();
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "tool message with image content must error"
    );
}

// Refusal content must error (not silently drop).
#[test]
fn test_encode_refusal_content_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.messages.push(
        serde_json::from_value(json!({
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "I cannot help"}]
        }))
        .unwrap(),
    );
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "refusal content must error, not silently vanish"
    );
}

// tools[] validation negative cases.
#[test]
fn test_decode_tools_validation_negative() {
    let codec = GeminiGenerateContentCodec;
    let cases: &[(&str, Json)] = &[
        (
            "tools entry not an object",
            json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": ["bad"]}),
        ),
        (
            "functionDeclarations not an array",
            json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": [{"functionDeclarations": "bad"}]}),
        ),
        (
            "functionDeclaration not an object",
            json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": [{"functionDeclarations": ["bad"]}]}),
        ),
        (
            "functionDeclaration empty name",
            json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": [{"functionDeclarations": [{"name": ""}]}]}),
        ),
        (
            "functionDeclaration missing name",
            json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": [{"functionDeclarations": [{"description": "no name"}]}]}),
        ),
    ];
    for (label, body) in cases {
        assert!(
            codec.decode(&make_request(body.clone())).is_err(),
            "case '{label}' must error"
        );
    }
}

// functionCall.args must be an object on request decode.
#[test]
fn test_decode_function_call_non_object_args_errors() {
    let codec = GeminiGenerateContentCodec;
    for bad in [json!([1, 2]), json!("string"), json!(42)] {
        let req = make_request(json!({
            "contents": [{
                "role": "model",
                "parts": [{"functionCall": {"name": "fn", "args": bad}}]
            }]
        }));
        assert!(
            codec.decode(&req).is_err(),
            "functionCall.args={bad} must error on request decode"
        );
    }
}

// functionCall arguments must be a JSON object on encode.
#[test]
fn test_encode_function_call_non_object_args_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({"contents": [{"role": "user", "parts": [{"text": "go"}]}]}));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.messages.push(Message::Assistant {
        content: None,
        name: None,
        tool_calls: Some(vec![super::super::request::ToolCall {
            id: "c1".into(),
            call_type: "function".into(),
            function: super::super::request::FunctionCall {
                name: "fn".into(),
                arguments: r#"["not","object"]"#.into(),
            },
        }]),
    });
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "array arguments must error; Gemini requires object args"
    );
}

// Unchanged payload with native inlineData round-trips byte-identically.
#[test]
fn test_encode_native_inline_data_round_trips_unchanged() {
    let codec = GeminiGenerateContentCodec;
    let image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQ==";
    let original = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "what is this?"},
                {"inlineData": {"mimeType": "image/png", "data": image}}
            ]
        }],
        "generationConfig": {"temperature": 0.5}
    }));
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content, original.content,
        "encode(decode(req), req) must be identical to req"
    );
}

// generationConfig unmodeled keys are preserved when params are cleared.
#[test]
fn test_encode_clearing_params_preserves_unmodeled_gen_config_fields() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json",
            "responseSchema": {"type": "object"}
        }
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Clear params so temperature is removed.
    annotated.params = None;
    let encoded = codec.encode(&annotated, &original).unwrap();
    let gc = encoded
        .content
        .get("generationConfig")
        .expect("generationConfig must remain");
    assert!(
        gc.get("temperature").is_none(),
        "temperature must be removed"
    );
    assert_eq!(
        gc["responseMimeType"],
        json!("application/json"),
        "responseMimeType must survive"
    );
    assert!(
        gc.get("responseSchema").is_some(),
        "responseSchema must survive"
    );
}

// All modeled keys cleared, empty generationConfig is removed.
#[test]
fn test_encode_clearing_params_removes_empty_gen_config() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 256}
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.params = None;
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert!(
        encoded.content.get("generationConfig").is_none(),
        "generationConfig must be removed when all modeled keys are cleared and no unmodeled keys remain"
    );
}

// Response decode: malformed functionCall parts error rather than silently drop.
#[test]
fn test_decode_response_malformed_function_call_errors() {
    let codec = GeminiGenerateContentCodec;
    let cases: &[(&str, Json)] = &[
        (
            "functionCall is non-object",
            json!({
                "candidates": [{"content": {"role": "model",
                    "parts": [{"functionCall": "not-an-object"}]}, "finishReason": "STOP"}]
            }),
        ),
        (
            "functionCall missing name",
            json!({
                "candidates": [{"content": {"role": "model",
                    "parts": [{"functionCall": {"args": {}}}]}, "finishReason": "STOP"}]
            }),
        ),
        (
            "functionCall empty name",
            json!({
                "candidates": [{"content": {"role": "model",
                    "parts": [{"functionCall": {"name": "", "args": {}}}]}, "finishReason": "STOP"}]
            }),
        ),
        (
            "functionCall.id non-string",
            json!({
                "candidates": [{"content": {"role": "model",
                    "parts": [{"functionCall": {"name": "fn", "id": 123, "args": {}}}]}, "finishReason": "STOP"}]
            }),
        ),
        (
            "functionCall.id empty",
            json!({
                "candidates": [{"content": {"role": "model",
                    "parts": [{"functionCall": {"name": "fn", "id": "", "args": {}}}]}, "finishReason": "STOP"}]
            }),
        ),
        (
            "functionCall.args non-object",
            json!({
                "candidates": [{"content": {"role": "model",
                    "parts": [{"functionCall": {"name": "fn", "args": [1, 2]}}]}, "finishReason": "STOP"}]
            }),
        ),
    ];
    for (label, body) in cases {
        assert!(
            codec.decode_response(body).is_err(),
            "case '{label}' must error"
        );
    }
}

// STOP + malformed functionCall must error, not return Complete.
#[test]
fn test_decode_response_stop_with_malformed_function_call_errors() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{"content": {"role": "model",
            "parts": [{"functionCall": {"name": "", "args": {}}}]},
            "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5}
    });
    assert!(
        codec.decode_response(&response).is_err(),
        "STOP + malformed functionCall must error, not return Complete"
    );
}

// Request decode: present functionCall.id must be a non-empty string.
#[test]
fn test_decode_request_function_call_id_validation() {
    let codec = GeminiGenerateContentCodec;
    let cases: &[(&str, Json)] = &[
        (
            "functionCall.id non-string",
            json!({
                "contents": [{"role": "model",
                    "parts": [{"functionCall": {"name": "fn", "id": 99, "args": {}}}]}]
            }),
        ),
        (
            "functionCall.id empty",
            json!({
                "contents": [{"role": "model",
                    "parts": [{"functionCall": {"name": "fn", "id": "", "args": {}}}]}]
            }),
        ),
    ];
    for (label, body) in cases {
        assert!(
            codec.decode(&make_request(body.clone())).is_err(),
            "case '{label}' must error"
        );
    }
}

// Request decode: present functionResponse.id must be a non-empty string.
#[test]
fn test_decode_request_function_response_id_validation() {
    let codec = GeminiGenerateContentCodec;
    let cases: &[(&str, Json)] = &[
        (
            "functionResponse.id non-string",
            json!({
                "contents": [{"role": "user",
                    "parts": [{"functionResponse": {"name": "fn", "id": 99, "response": {}}}]}]
            }),
        ),
        (
            "functionResponse.id empty",
            json!({
                "contents": [{"role": "user",
                    "parts": [{"functionResponse": {"name": "fn", "id": "", "response": {}}}]}]
            }),
        ),
    ];
    for (label, body) in cases {
        assert!(
            codec.decode(&make_request(body.clone())).is_err(),
            "case '{label}' must error"
        );
    }
}

// systemInstruction validation.
#[test]
fn test_decode_system_instruction_validation() {
    let codec = GeminiGenerateContentCodec;
    let base_contents = json!([{"role": "user", "parts": [{"text": "hi"}]}]);
    let cases: &[(&str, Json)] = &[
        (
            "systemInstruction not an object",
            json!({"contents": base_contents, "systemInstruction": "bad"}),
        ),
        (
            "systemInstruction missing parts",
            json!({"contents": base_contents, "systemInstruction": {}}),
        ),
        (
            "systemInstruction.parts not an array",
            json!({"contents": base_contents, "systemInstruction": {"parts": "bad"}}),
        ),
        (
            "systemInstruction.parts entry not an object",
            json!({"contents": base_contents, "systemInstruction": {"parts": ["bad"]}}),
        ),
        (
            "systemInstruction.parts[].text not a string",
            json!({"contents": base_contents, "systemInstruction": {"parts": [{"text": 123}]}}),
        ),
        (
            "systemInstruction.parts native part",
            json!({
                "contents": base_contents,
                "systemInstruction": {
                    "parts": [
                        {"text": "ok"},
                        {
                            "inlineData": {
                                "mimeType": "text/plain",
                                "data": "sk-system-secret"
                            }
                        }
                    ]
                }
            }),
        ),
    ];
    for (label, body) in cases {
        assert!(
            codec.decode(&make_request(body.clone())).is_err(),
            "case '{label}' must error"
        );
    }
}

// model must be a string when present.
#[test]
fn test_decode_non_string_model_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "model": 42
    }));
    assert!(codec.decode(&req).is_err(), "non-string model must error");
}

// Encode: empty tool-call name must error.
#[test]
fn test_encode_empty_tool_call_name_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.messages.push(Message::Assistant {
        content: None,
        name: None,
        tool_calls: Some(vec![super::super::request::ToolCall {
            id: "c1".into(),
            call_type: "function".into(),
            function: super::super::request::FunctionCall {
                name: "".into(), // empty name
                arguments: "{}".into(),
            },
        }]),
    });
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "empty tool-call function name must error"
    );
}

// Encode: empty tool_call_id must error.
#[test]
fn test_encode_empty_tool_call_id_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"functionCall": {"id": "c1", "name": "fn", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"id": "c1", "name": "fn", "response": {}}}]}
        ]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    // Give the tool message an empty tool_call_id (force it via JSON).
    annotated.messages[2] = serde_json::from_value(json!({
        "role": "tool",
        "tool_call_id": "",
        "content": "{}"
    }))
    .unwrap();
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "empty tool_call_id must error"
    );
}

// Encode: empty FunctionDefinition.name must error.
#[test]
fn test_encode_empty_function_definition_name_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"functionDeclarations": [{"name": "good_fn"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    if let Some(tools) = annotated.tools.as_mut() {
        for td in tools.iter_mut() {
            if let nemo_relay_types::codec::request::ToolDefinition::Function { function, .. } = td
            {
                function.name = "".into();
            }
        }
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "empty FunctionDefinition.name must error"
    );
}

// Encode: FunctionDefinition.strict must error when set.
#[test]
fn test_encode_function_definition_strict_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"functionDeclarations": [{"name": "fn"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    if let Some(tools) = annotated.tools.as_mut() {
        for td in tools.iter_mut() {
            if let nemo_relay_types::codec::request::ToolDefinition::Function { function, .. } = td
            {
                function.strict = Some(true);
            }
        }
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "FunctionDefinition.strict is not representable in Gemini and must error"
    );
}

// Decode: non-string description in functionDeclaration must error.
#[test]
fn test_decode_function_declaration_non_string_description_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"functionDeclarations": [{"name": "fn", "description": 42}]}]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "non-string functionDeclaration.description must error"
    );
}

// Encode: systemInstruction sibling fields are preserved when text changes.
#[test]
fn test_encode_system_instruction_preserves_sibling_fields() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "systemInstruction": {
            "role": "user",
            "parts": [{"text": "old prompt"}],
            "nativeField": "keep me"
        }
    }));
    let mut annotated = codec.decode(&original).unwrap();
    for msg in annotated.messages.iter_mut() {
        if let Message::System { content, .. } = msg {
            *content = MessageContent::Text("new prompt".into());
        }
    }
    let encoded = codec.encode(&annotated, &original).unwrap();
    let si = encoded.content.get("systemInstruction").unwrap();
    assert_eq!(
        si.get("parts")
            .and_then(|p| p.as_array())
            .and_then(|a| a.first())
            .and_then(|p| p.get("text"))
            .and_then(Json::as_str),
        Some("new prompt"),
        "new system text must appear in parts"
    );
    assert_eq!(
        si.get("nativeField").and_then(Json::as_str),
        Some("keep me"),
        "systemInstruction native sibling fields must be preserved when text changes"
    );
    assert_eq!(
        si.get("role").and_then(Json::as_str),
        Some("user"),
        "systemInstruction.role must be preserved"
    );
}

// Helper: text parts without an explicit normalized type are still text.
#[test]
fn test_extract_content_text_treats_missing_part_type_as_text() {
    assert_eq!(
        extract_content_text(&json!([{"text": "new prompt"}])),
        "new prompt",
        "missing normalized part type should be interpreted as text, matching encoder validation"
    );
}

// systemInstruction.role non-string must error on decode (validated in validate_system_instruction).
#[test]
fn test_decode_system_instruction_non_string_role_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "systemInstruction": {"role": 123, "parts": [{"text": "system"}]}
    }));
    assert!(
        codec.decode(&req).is_err(),
        "systemInstruction.role non-string must error on decode"
    );
}

// Mixed functionResponse + functionCall in one content item must error.
#[test]
fn test_decode_mixed_fr_and_fc_parts_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "fn", "response": {}}},
                {"functionCall": {"name": "fn", "args": {}}}
            ]
        }]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "content item with both functionResponse and functionCall parts must error"
    );
}

// Non-numeric topP must error.
#[test]
fn test_decode_rejects_non_numeric_top_p() {
    let codec = GeminiGenerateContentCodec;
    let request = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"topP": "high"}
    }));
    assert!(
        codec.decode(&request).is_err(),
        "non-numeric topP must error"
    );
}

// ProviderNative tool variants owned by a different provider must error on encode.
#[test]
fn test_encode_mismatched_provider_native_tool_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.tools = Some(vec![ToolDefinition::ProviderNative {
        provider: "openai_chat".into(),
        kind: "web_search".into(),
        value: serde_json::json!({"type": "web_search_preview"}),
    }]);
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "ProviderNative tool from another provider must error"
    );
}

#[test]
fn test_encode_provider_native_tool_with_function_declarations_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.tools = Some(vec![ToolDefinition::ProviderNative {
        provider: "gemini".into(),
        kind: "functionDeclarations".into(),
        value: serde_json::json!({"functionDeclarations": [{"name": "fn"}]}),
    }]);
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "functionDeclarations must use ToolDefinition::Function"
    );
}

#[test]
fn test_encode_system_message_name_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "systemInstruction": {"parts": [{"text": "be helpful"}]},
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    for msg in annotated.messages.iter_mut() {
        if let Message::System { name, .. } = msg {
            *name = Some("system-name".into());
        }
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "Gemini cannot represent system message names"
    );
}

#[test]
fn test_encode_user_message_name_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    if let Message::User { name, .. } = &mut annotated.messages[0] {
        *name = Some("user-name".into());
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "Gemini cannot represent user message names"
    );
}

#[test]
fn test_encode_assistant_message_name_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]}
        ]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    for msg in annotated.messages.iter_mut() {
        if let Message::Assistant { name, .. } = msg {
            *name = Some("assistant-name".into());
        }
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "Gemini cannot represent assistant message names"
    );
}

#[test]
fn test_encode_assistant_tool_call_message_name_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [
            {"role": "user", "parts": [{"text": "use a tool"}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"id": "call_1", "name": "lookup", "args": {}}}]
            }
        ]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    for msg in annotated.messages.iter_mut() {
        if let Message::Assistant { name, .. } = msg {
            *name = Some("assistant-name".into());
        }
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "Gemini cannot represent assistant names on tool-call messages"
    );
}

#[test]
fn test_encode_previous_response_id_returns_error() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}));
    let mut annotated = codec.decode(&original).unwrap();
    annotated.previous_response_id = Some("prev-123".into());
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "previous_response_id is not representable in Gemini and must error"
    );
}

#[test]
fn test_decode_non_string_text_part_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": 42}]}]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "parts[].text with non-string value must error"
    );
}

// functionResponse on model-role content must error.
#[test]
fn test_decode_function_response_on_model_role_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{
            "role": "model",
            "parts": [{"functionResponse": {"name": "fn", "response": {}}}]
        }]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "functionResponse in a 'model' role content item must error"
    );
}

// functionCall on user-role content must error.
#[test]
fn test_decode_function_call_on_user_role_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [{"functionCall": {"name": "fn", "args": {}}}]
        }]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "functionCall in a 'user' role content item must error"
    );
}

// functionResponse mixed with visible text parts must error.
#[test]
fn test_decode_function_response_mixed_with_text_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "fn", "response": {}}},
                {"text": "visible text"}
            ]
        }]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "functionResponse mixed with visible text parts must error"
    );
}

// Response decode: non-string text value must error.
#[test]
fn test_decode_response_non_string_text_part_errors() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": 42}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 1}
    });
    assert!(
        codec.decode_response(&response).is_err(),
        "response parts[].text with non-string value must error"
    );
}

// systemInstruction part-level metadata is preserved when text changes.
#[test]
fn test_encode_system_instruction_part_metadata_preserved() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "systemInstruction": {
            "parts": [{"text": "old prompt", "nativePartField": "keep-me"}]
        }
    }));
    let mut annotated = codec.decode(&original).unwrap();
    for msg in annotated.messages.iter_mut() {
        if let Message::System { content, .. } = msg {
            *content = MessageContent::Text("new prompt".into());
        }
    }
    let encoded = codec.encode(&annotated, &original).unwrap();
    let part = &encoded.content["systemInstruction"]["parts"][0];
    assert_eq!(part["text"].as_str(), Some("new prompt"));
    assert_eq!(
        part["nativePartField"].as_str(),
        Some("keep-me"),
        "part-level native fields must be preserved when system text changes"
    );
}

// Unknown sibling fields on a known text part are metadata, not additional data-union fields.
#[test]
fn test_encode_text_part_unknown_metadata_preserved() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{
            "role": "user",
            "parts": [{"text": "old text", "nativePartField": "keep-me"}]
        }]
    }));
    let mut annotated = codec.decode(&original).unwrap();
    if let Message::User { content, .. } = &mut annotated.messages[0] {
        match content {
            MessageContent::Parts(parts) => {
                if let ContentPart::Text { text, .. } = &mut parts[0] {
                    *text = "new text".into();
                }
            }
            other => panic!("expected metadata-bearing text part, got {other:?}"),
        }
    }

    let encoded = codec.encode(&annotated, &original).unwrap();
    let part = &encoded.content["contents"][0]["parts"][0];
    assert_eq!(part["text"].as_str(), Some("new text"));
    assert_eq!(part["nativePartField"].as_str(), Some("keep-me"));
}

// Request decode: part with both text and functionResponse must error.
#[test]
fn test_decode_part_with_text_and_function_response_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"role": "user", "parts": [
            {"text": "visible", "functionResponse": {"name": "fn", "response": {}}}
        ]}]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "part with both 'text' and 'functionResponse' must error"
    );
}

// Request decode: part with both text and functionCall must error.
#[test]
fn test_decode_part_with_text_and_function_call_errors() {
    let codec = GeminiGenerateContentCodec;
    let req = make_request(json!({
        "contents": [{"role": "model", "parts": [
            {"text": "visible", "functionCall": {"name": "fn", "args": {}}}
        ]}]
    }));
    assert!(
        codec.decode(&req).is_err(),
        "part with both 'text' and 'functionCall' must error"
    );
}

// Response decode: non-object part must error.
#[test]
fn test_decode_response_non_object_part_errors() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{"content": {"role": "model", "parts": ["not an object"]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1}
    });
    assert!(
        codec.decode_response(&response).is_err(),
        "response part that is not an object must error"
    );
}

// Response decode: part with both text and functionCall must error.
#[test]
fn test_decode_response_part_text_and_function_call_errors() {
    let codec = GeminiGenerateContentCodec;
    let response = json!({
        "candidates": [{"content": {"role": "model", "parts": [
            {"text": "hello", "functionCall": {"name": "fn", "args": {}}}
        ]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1}
    });
    assert!(
        codec.decode_response(&response).is_err(),
        "response part with both 'text' and 'functionCall' must error"
    );
}

// Streaming: part with both text and functionCall must error via collector.
#[test]
fn test_streaming_part_text_and_function_call_errors() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();

    let result = collector(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [
                {"text": "hello", "functionCall": {"name": "fn", "args": {}}}
            ]},
            "index": 0
        }]
    }));
    assert!(
        result.is_err(),
        "streaming part with both 'text' and 'functionCall' must error via collector"
    );
}

// Streaming: non-object part must error.
#[test]
fn test_streaming_non_object_part_errors() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let result = collector(json!({
        "candidates": [{"content": {"role": "model", "parts": ["not an object"]}, "index": 0}]
    }));
    assert!(
        result.is_err(),
        "streaming non-object part must error via collector"
    );
}

// Streaming: non-string text value must error.
#[test]
fn test_streaming_non_string_text_errors() {
    let streaming_codec = GeminiGenerateContentStreamingCodec::new();
    let mut collector = streaming_codec.collector();
    let result = collector(json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": 42}]}, "index": 0}]
    }));
    assert!(
        result.is_err(),
        "streaming part with non-string text must error via collector"
    );
}

// systemInstruction with multiple text parts must error on edit.
#[test]
fn test_encode_system_instruction_multiple_text_parts_edit_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "systemInstruction": {
            "parts": [{"text": "part one"}, {"text": "part two"}]
        }
    }));
    let mut annotated = codec.decode(&original).unwrap();
    for msg in annotated.messages.iter_mut() {
        if let Message::System { content, .. } = msg {
            *content = MessageContent::Text("new system".into());
        }
    }
    assert!(
        codec.encode(&annotated, &original).is_err(),
        "editing systemInstruction with multiple text parts must error"
    );
}

// systemInstruction with a native non-text part must error on decode.
#[test]
fn test_decode_system_instruction_non_text_part_errors() {
    let codec = GeminiGenerateContentCodec;
    let original = make_request(json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "systemInstruction": {
            "parts": [{"text": "ok"}, {"nativePart": "value"}]
        }
    }));
    assert!(
        codec.decode(&original).is_err(),
        "systemInstruction with non-text native parts must error"
    );
}
