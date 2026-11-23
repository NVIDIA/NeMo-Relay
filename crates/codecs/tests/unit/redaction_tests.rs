// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

#[test]
fn removal_and_replacement_remain_attached_to_original_blocks() {
    let codec = crate::openai_responses::OpenAIResponsesCodec;
    let original = json!({"output_text":"first\ntail", "output":[
        {"type":"reasoning", "summary":[], "marker":"keep reasoning"},
        {"type":"message", "marker":"first message", "content":[
            {"type":"output_text", "text":"first", "marker":"first"},
            {"type":"refusal", "refusal":"no", "marker":"refusal"}
        ]},
        {"type":"message", "marker":"last message", "content":[
            {"type":"output_text", "text":"tail", "marker":"tail", "annotations":[{"id":"tail annotation"}]}
        ]}
    ]});
    let baseline = serde_json::to_value(codec.decode_response(&original).unwrap()).unwrap();
    for removed in [vec![0], vec![0, 1], vec![1], vec![0, 1, 2]] {
        let mut sanitized = baseline.clone();
        sanitized["message"][2]["text"] = json!("safe tail");
        for index in &removed {
            sanitized["message"][*index] = Json::Null;
        }
        let result = LlmResponseCodec::redaction(&codec)
            .unwrap()
            .edit_response(
                &original,
                NormalizedEdits {
                    baseline: &baseline,
                    sanitized: &sanitized,
                    require_complete_output: true,
                },
            )
            .unwrap()
            .expect("anchored removal");
        assert_eq!(result["output"][0], original["output"][0]);
        assert_eq!(result["output"][1]["marker"], "first message");
        assert_eq!(result["output"][2]["marker"], "last message");
        if !removed.contains(&2) {
            let tail = &result["output"][2]["content"][0];
            assert_eq!(tail["text"], "safe tail");
            assert_eq!(tail["marker"], "tail");
            assert_eq!(
                tail["annotations"],
                original["output"][2]["content"][0]["annotations"]
            );
            assert_eq!(
                result["output_text"],
                if removed.contains(&0) {
                    "safe tail"
                } else {
                    "first\nsafe tail"
                }
            );
        } else {
            assert!(codec.decode_response(&result).unwrap().message.is_none());
            assert!(result.get("output_text").is_none());
        }
        if removed.contains(&0) {
            assert!(!result.to_string().contains("\"text\":\"first\""));
        }
    }
}

#[test]
fn conflicting_responses_alias_and_normalized_text_edits_fail_closed() {
    let codec = crate::openai_responses::OpenAIResponsesCodec;
    let wire = json!({"output_text":"already hashed","output":[{"type":"message","content":[
        {"type":"output_text","text":"original"}, {"type":"refusal","refusal":"no"}
    ]}]});
    let baseline = serde_json::to_value(codec.decode_response(&wire).unwrap()).unwrap();
    let mut sanitized = baseline.clone();
    sanitized["message"][0]["text"] = json!("masked");
    let result = LlmResponseCodec::redaction(&codec)
        .unwrap()
        .edit_response(
            &wire,
            NormalizedEdits {
                baseline: &baseline,
                sanitized: &sanitized,
                require_complete_output: true,
            },
        )
        .unwrap();
    assert!(result.is_none());
}

#[test]
fn compacted_parts_invalid_fields_and_foreign_baselines_fail_closed() {
    let codec = crate::openai_responses::OpenAIResponsesCodec;
    let mut original = json!({"output":[{"type":"message", "content":[
        {"type":"output_text", "text":"first"},
        {"type":"refusal", "refusal":"no"},
        {"type":"output_text", "text":"tail"}
    ]}]});
    original["output"].as_array_mut().unwrap().extend([
        json!({"type":"function_call", "call_id":"one", "name":"lookup", "arguments":"{}"}),
        json!({"type":"function_call", "call_id":"two", "name":"lookup", "arguments":"{}"}),
    ]);
    let baseline = serde_json::to_value(codec.decode_response(&original).unwrap()).unwrap();
    for case in 0..5 {
        let mut sanitized = baseline.clone();
        let mut source = baseline.clone();
        match case {
            0 => {
                sanitized["message"].as_array_mut().unwrap().remove(0);
            }
            1 => {
                sanitized["message"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("text");
            }
            2 => {
                source["message"][0]["text"] = json!("different source");
            }
            3 => {
                sanitized["tool_calls"].as_array_mut().unwrap().remove(0);
            }
            _ => {
                sanitized["message"][0]["unsupported"] = json!("new value");
            }
        }
        assert!(
            LlmResponseCodec::redaction(&codec)
                .unwrap()
                .edit_response(
                    &original,
                    NormalizedEdits {
                        baseline: &source,
                        sanitized: &sanitized,
                        require_complete_output: true,
                    }
                )
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn redacting_repeated_idless_gemini_calls_preserves_each_signature() {
    let codec = crate::gemini_generate_content::GeminiGenerateContentCodec;
    let original = LlmRequest {
        headers: Default::default(),
        content: json!({"contents":[
            {"role":"model","parts":[{
                "functionCall":{"name":"lookup","args":{"city":"Paris"}},"thoughtSignature":"cGFyaXM="
            }]},
            {"role":"model","parts":[{
                "functionCall":{"name":"lookup","args":{"city":"Rome"}},"thoughtSignature":"cm9tZQ=="
            }]}
        ]}),
    };
    let decoded = codec.decode(&original).unwrap();
    let baseline = serde_json::to_value(&decoded).unwrap();
    let mut sanitized = baseline.clone();
    let mut expected = original.content.clone();
    for index in 0..2 {
        sanitized["messages"][index]["tool_calls"][0]["function"]["arguments"] =
            json!(r#"{"city":"masked"}"#);
        expected["contents"][index]["parts"][0]["functionCall"]["args"]["city"] = json!("masked");
    }
    // A bulk rewrite cannot identify either source turn. The redaction path
    // can still apply each edit against its original slot, one at a time.
    let edited = serde_json::from_value(sanitized.clone()).unwrap();
    assert!(
        codec
            .encode_with_baseline(&edited, &original, &decoded)
            .is_err()
    );
    let request = LlmCodec::redaction(&codec)
        .unwrap()
        .edit_request(
            &original,
            NormalizedEdits {
                baseline: &baseline,
                sanitized: &sanitized,
                require_complete_output: false,
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(request.content, expected);

    // An appended call has no claim on either existing signature.
    let mut appended = decoded.clone();
    appended.messages.push(edited.messages[0].clone());
    let encoded = codec
        .encode_with_baseline(&appended, &original, &decoded)
        .unwrap();
    let contents = encoded.content["contents"].as_array().unwrap();
    assert_eq!(
        &contents[..2],
        original.content["contents"].as_array().unwrap()
    );
    assert!(contents[2]["parts"][0].get("thoughtSignature").is_none());
}

#[test]
fn every_builtin_edits_text_and_preserves_unrelated_wire_fields() {
    let cases = [
        (
            ProviderSurface::OpenAIChat,
            json!({"choices":[{"message":{"content":"secret"}}]}),
        ),
        (
            ProviderSurface::OpenAIResponses,
            json!({"output":[{"type":"message","content":[{"type":"output_text","text":"secret"}]}]}),
        ),
        (
            ProviderSurface::AnthropicMessages,
            json!({"type":"message","content":[{"type":"text","text":"secret"}]}),
        ),
        (
            ProviderSurface::OCIGenAI,
            json!({"chatResponse":{"apiFormat":"GENERIC","choices":[{"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":"secret"}]}}]}}),
        ),
        (
            ProviderSurface::GeminiGenerateContent,
            json!({"candidates":[{"content":{"parts":[{"text":"secret"}]}}]}),
        ),
    ];
    for (surface, mut original) in cases {
        original["unrelated"] = json!({"marker":"preserved"});
        let codec = crate::resolve::response_codec(surface);
        let baseline = serde_json::to_value(codec.decode_response(&original).unwrap()).unwrap();
        let mut sanitized = baseline.clone();
        if sanitized["message"].is_string() {
            sanitized["message"] = json!("safe");
        } else {
            sanitized["message"][0]["text"] = json!("safe");
        }
        let result = codec
            .redaction()
            .unwrap()
            .edit_response(
                &original,
                NormalizedEdits {
                    baseline: &baseline,
                    sanitized: &sanitized,
                    require_complete_output: true,
                },
            )
            .unwrap()
            .unwrap_or_else(|| panic!("{surface:?}"));
        assert_eq!(
            codec
                .decode_response(&result)
                .unwrap()
                .response_text()
                .as_deref(),
            Some("safe"),
            "{surface:?}"
        );
        assert_eq!(result["unrelated"], original["unrelated"]);
    }
}

#[test]
fn complete_output_policy_rejects_unmodeled_alternatives_even_without_a_text_match() {
    for (surface, original) in [
        (
            ProviderSurface::OpenAIChat,
            json!({"choices":[{"message":{}},{"message":{"content":"unmodeled secret"}}]}),
        ),
        (
            ProviderSurface::GeminiGenerateContent,
            json!({"candidates":[{"content":{"parts":[]}},{"content":{"parts":[{"text":"unmodeled secret"}]}}]}),
        ),
    ] {
        let codec = crate::resolve::response_codec(surface);
        let baseline = serde_json::to_value(codec.decode_response(&original).unwrap()).unwrap();
        assert!(
            codec
                .redaction()
                .unwrap()
                .edit_response(
                    &original,
                    NormalizedEdits {
                        baseline: &baseline,
                        sanitized: &baseline,
                        require_complete_output: true,
                    }
                )
                .unwrap()
                .is_none()
        );
        assert!(
            codec
                .redaction()
                .unwrap()
                .edit_response(
                    &original,
                    NormalizedEdits {
                        baseline: &baseline,
                        sanitized: &baseline,
                        require_complete_output: false,
                    }
                )
                .unwrap()
                .is_some()
        );
    }
}

#[test]
fn every_builtin_edits_requests_without_dropping_unrelated_fields() {
    for (surface, content) in [
        (
            ProviderSurface::OpenAIChat,
            json!({"messages":[{"role":"user","content":"secret"}]}),
        ),
        (ProviderSurface::OpenAIResponses, json!({"input":"secret"})),
        (
            ProviderSurface::AnthropicMessages,
            json!({"messages":[{"role":"user","content":"secret"}]}),
        ),
        (
            ProviderSurface::OCIGenAI,
            json!({"messages":[{"role":"USER","content":[{"type":"TEXT","text":"secret"}]}]}),
        ),
        (
            ProviderSurface::GeminiGenerateContent,
            json!({"contents":[{"role":"user","parts":[{"text":"secret"}]}]}),
        ),
    ] {
        let codec = crate::resolve::request_codec(surface);
        let mut original = LlmRequest {
            headers: Default::default(),
            content,
        };
        original.content["unrelated"] = json!("preserved");
        original
            .headers
            .insert("x-marker".into(), json!("preserved"));
        let baseline = serde_json::to_value(codec.decode(&original).unwrap()).unwrap();
        let mut sanitized = baseline.clone();
        if sanitized["messages"][0]["content"].is_string() {
            sanitized["messages"][0]["content"] = json!("safe");
        } else {
            sanitized["messages"][0]["content"][0]["text"] = json!("safe");
        }
        let result = codec
            .redaction()
            .unwrap()
            .edit_request(
                &original,
                NormalizedEdits {
                    baseline: &baseline,
                    sanitized: &sanitized,
                    require_complete_output: false,
                },
            )
            .unwrap()
            .unwrap_or_else(|| panic!("{surface:?}"));
        assert_eq!(
            codec
                .decode(&result)
                .unwrap()
                .last_user_message()
                .as_deref(),
            Some("safe"),
            "{surface:?}"
        );
        assert_eq!(result.content["unrelated"], "preserved");
        assert_eq!(result.headers, original.headers);
        let mut removed = baseline.clone();
        removed["messages"][0] = Json::Null;
        assert!(
            codec
                .redaction()
                .unwrap()
                .edit_request(
                    &original,
                    NormalizedEdits {
                        baseline: &baseline,
                        sanitized: &removed,
                        require_complete_output: false,
                    }
                )
                .unwrap()
                .is_none(),
            "ambiguous request deletion must fail closed"
        );
    }
}

#[test]
fn each_capability_operation_defaults_to_unsupported() {
    struct Unsupported;
    impl LlmRequestRedaction for Unsupported {}
    impl LlmResponseRedaction for Unsupported {}
    let original = json!({});
    let edits = NormalizedEdits {
        baseline: &original,
        sanitized: &original,
        require_complete_output: true,
    };
    let request = LlmRequest {
        headers: Default::default(),
        content: original.clone(),
    };
    assert!(Unsupported.edit_request(&request, edits).unwrap().is_none());
    assert!(
        Unsupported
            .edit_response(&original, edits)
            .unwrap()
            .is_none()
    );
    assert!(
        Unsupported
            .project_request(&Default::default())
            .unwrap()
            .is_none()
    );
    assert!(
        Unsupported
            .project_response(&Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn metadata_only_edits_preserve_provider_blocks_exactly() {
    for (surface, original) in [
        (
            ProviderSurface::OpenAIResponses,
            json!({"id":"old","output":[
            {"type":"custom_tool_call","call_id":"a","name":"custom","input":"private"},
            {"type":"function_call","call_id":"b","name":"ordinary","arguments":"{  }"}]}),
        ),
        (
            ProviderSurface::AnthropicMessages,
            json!({"id":"old","type":"message","content":[
            {"type":"text","text":"one\ntwo","citations":[{"marker":"first"}]},
            {"type":"text","text":"three","citations":[{"marker":"last"}]}]}),
        ),
    ] {
        let codec = crate::resolve::response_codec(surface);
        let baseline = serde_json::to_value(codec.decode_response(&original).unwrap()).unwrap();
        for id in ["old", "new"] {
            let mut sanitized = baseline.clone();
            sanitized["id"] = json!(id);
            let result = codec
                .redaction()
                .unwrap()
                .edit_response(
                    &original,
                    NormalizedEdits {
                        baseline: &baseline,
                        sanitized: &sanitized,
                        require_complete_output: true,
                    },
                )
                .unwrap();
            let mut expected = original.clone();
            expected["id"] = json!(id);
            assert_eq!(result, Some(expected));
        }
        if surface == ProviderSurface::OpenAIResponses {
            let mut sanitized = baseline.clone();
            sanitized["tool_calls"][0]["arguments"] = json!("approved");
            let mut expected = original.clone();
            expected["output"][0]["input"] = json!("approved");
            assert_eq!(
                codec
                    .redaction()
                    .unwrap()
                    .edit_response(
                        &original,
                        NormalizedEdits {
                            baseline: &baseline,
                            sanitized: &sanitized,
                            require_complete_output: true
                        }
                    )
                    .unwrap(),
                Some(expected)
            );
        }
    }
}

#[test]
fn removing_a_tool_before_a_message_preserves_the_original_message_target() {
    let codec = crate::resolve::response_codec(ProviderSurface::OpenAIResponses);
    let message = json!({"type":"message","content":[{"type":"refusal","refusal":"private","marker":"same block"}]});
    let original = json!({"output":[{"type":"function_call","call_id":"c","name":"lookup","arguments":"{}"},message]});
    let baseline = serde_json::to_value(codec.decode_response(&original).unwrap()).unwrap();
    let mut sanitized = baseline.clone();
    sanitized.as_object_mut().unwrap().remove("tool_calls");
    sanitized["message"][0]["refusal"] = json!("approved");
    let edited = codec
        .redaction()
        .unwrap()
        .edit_response(
            &original,
            NormalizedEdits {
                baseline: &baseline,
                sanitized: &sanitized,
                require_complete_output: true,
            },
        )
        .unwrap();
    let mut expected = json!({"output":[message]});
    expected["output"][0]["content"][0]["refusal"] = json!("approved");
    assert_eq!(edited, Some(expected));
}
