// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization compatibility tests for shared NeMo Relay DTOs.

use std::sync::Arc;

use nemo_relay_types::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, PendingMarkSpec, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings,
};
use nemo_relay_types::api::llm::{LlmAttributes, LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay_types::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};
use nemo_relay_types::codec::request::{AnnotatedLlmRequest, ContentPart, Message, MessageContent};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use serde_json::{Map, json};

#[test]
fn event_round_trips_with_annotated_llm_profiles() {
    let request = AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: Map::new(),
    };
    let response = AnnotatedLlmResponse {
        id: Some("resp_1".into()),
        model: Some("model".into()),
        message: Some(MessageContent::Text("world".into())),
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: Map::new(),
    };
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("llm")
            .data(json!(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "hello" }),
            }))
            .build(),
        ScopeCategory::Start,
        llm_attributes_to_strings(LlmAttributes::STATEFUL),
        EventCategory::llm(),
        Some(CategoryProfile {
            annotated_request: Some(Arc::new(request)),
            annotated_response: Some(Arc::new(response)),
            ..CategoryProfile::default()
        }),
    ));

    let encoded = serde_json::to_value(&event).expect("event should serialize");
    let decoded: Event = serde_json::from_value(encoded).expect("event should deserialize");
    assert_eq!(decoded.name(), "llm");
    assert_eq!(
        decoded
            .annotated_response()
            .and_then(|response| response.id.as_deref()),
        Some("resp_1")
    );
}

#[test]
fn llm_request_intercept_outcome_round_trips_pending_marks() {
    let outcome = LlmRequestInterceptOutcome::new(
        LlmRequest {
            headers: Map::new(),
            content: json!({ "prompt": "hello" }),
        },
        None,
    )
    .with_pending_mark(
        PendingMarkSpec::builder()
            .name("request.optimized")
            .category(EventCategory::custom())
            .category_profile(
                CategoryProfile::builder()
                    .subtype("optimizer.saved_tokens")
                    .build(),
            )
            .data(json!({ "saved_tokens": 12 }))
            .metadata(json!({ "source": "test" }))
            .build(),
    );

    let encoded = serde_json::to_value(&outcome).expect("outcome should serialize");
    assert_eq!(encoded["pending_marks"][0]["name"], "request.optimized");
    assert_eq!(encoded["pending_marks"][0]["category"], "custom");
    assert!(encoded["annotated_request"].is_null());

    let mut encoded_without_pending_marks = encoded.clone();
    encoded_without_pending_marks
        .as_object_mut()
        .unwrap()
        .remove("pending_marks");
    let decoded_without_pending_marks: LlmRequestInterceptOutcome =
        serde_json::from_value(encoded_without_pending_marks)
            .expect("outcome without pending marks should deserialize");
    assert!(decoded_without_pending_marks.pending_marks.is_empty());

    let decoded_defaults: LlmRequestInterceptOutcome = serde_json::from_value(json!({
        "request": {"headers": {}, "content": {"prompt": "hello"}},
        "future_field": true
    }))
    .expect("omitted optional fields and unknown fields should be accepted");
    assert!(decoded_defaults.annotated_request.is_none());
    assert!(decoded_defaults.pending_marks.is_empty());

    assert!(
        serde_json::from_value::<LlmRequestInterceptOutcome>(json!({
            "annotated_request": null,
            "pending_marks": []
        }))
        .is_err(),
        "request is required"
    );

    let decoded: LlmRequestInterceptOutcome =
        serde_json::from_value(encoded).expect("outcome should deserialize");
    assert_eq!(decoded, outcome);
}

#[test]
fn llm_request_intercept_outcome_converts_from_request_inputs() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({ "prompt": "hello" }),
    };
    let annotated_request: AnnotatedLlmRequest = serde_json::from_value(json!({
        "messages": [],
        "model": "model"
    }))
    .expect("annotated request should deserialize");

    let request_only: LlmRequestInterceptOutcome = request.clone().into();
    assert_eq!(
        request_only,
        LlmRequestInterceptOutcome::new(request.clone(), None)
    );

    let required_annotation: LlmRequestInterceptOutcome =
        (request.clone(), annotated_request.clone()).into();
    assert_eq!(
        required_annotation,
        LlmRequestInterceptOutcome::new(request.clone(), Some(annotated_request.clone()))
    );

    let optional_annotation: LlmRequestInterceptOutcome =
        (request.clone(), Some(annotated_request.clone())).into();
    assert_eq!(
        optional_annotation,
        LlmRequestInterceptOutcome::new(request, Some(annotated_request))
    );
}

#[test]
fn tool_execution_intercept_outcome_round_trips_pending_marks() {
    let outcome = ToolExecutionInterceptOutcome::annotated(
        json!({"stdout": "compacted"}),
        json!({"source": "middleware"}),
    )
    .with_pending_mark(
        PendingMarkSpec::builder()
            .name("tool.output.compacted")
            .category(EventCategory::custom())
            .category_profile(
                CategoryProfile::builder()
                    .subtype("optimizer.saved_tokens")
                    .build(),
            )
            .data(json!({"saved_tokens": 12}))
            .metadata(json!({"source": "test"}))
            .build(),
    );

    let encoded = serde_json::to_value(&outcome).expect("outcome should serialize");
    assert_eq!(encoded["result"]["stdout"], "compacted");
    assert_eq!(encoded["annotation"]["source"], "middleware");
    assert_eq!(encoded["pending_marks"][0]["name"], "tool.output.compacted");
    assert_eq!(encoded["pending_marks"][0]["category"], "custom");

    let decoded: ToolExecutionInterceptOutcome =
        serde_json::from_value(encoded).expect("outcome should deserialize");
    assert_eq!(decoded, outcome);

    let defaults: ToolExecutionInterceptOutcome = serde_json::from_value(json!({
        "result": "plain",
        "future_field": true
    }))
    .expect("omitted pending marks and unknown fields should be accepted");
    assert!(defaults.pending_marks.is_empty());
    assert!(defaults.annotation.is_none());
    assert_eq!(defaults.result, json!("plain"));

    assert!(
        serde_json::from_value::<ToolExecutionInterceptOutcome>(json!({
            "pending_marks": []
        }))
        .is_err(),
        "result is required"
    );
}

#[test]
fn tool_execution_result_round_trips_opaque_annotations() {
    for annotation in [
        json!("opaque"),
        json!([true, 7]),
        json!({"nested": {"status": "failed"}}),
    ] {
        let result = ToolExecutionResult::annotated(json!({"value": 42}), annotation.clone());
        let encoded = serde_json::to_value(&result).expect("result should serialize");
        assert_eq!(encoded["annotation"], annotation);
        let decoded: ToolExecutionResult =
            serde_json::from_value(encoded).expect("result should deserialize");
        assert_eq!(decoded, result);
    }

    let missing: ToolExecutionResult = serde_json::from_value(json!({"result": [1, 2]}))
        .expect("missing annotation should deserialize");
    assert!(missing.annotation.is_none());

    let null: ToolExecutionResult = serde_json::from_value(json!({
        "result": "ok",
        "annotation": null,
    }))
    .expect("null annotation should deserialize as absent");
    assert!(null.annotation.is_none());
    assert_eq!(
        serde_json::to_value(ToolExecutionResult::annotated(json!("ok"), json!(null))).unwrap(),
        json!({"result": "ok"})
    );

    assert!(
        serde_json::from_value::<ToolExecutionResult>(json!({"annotation": {}})).is_err(),
        "result is required"
    );
}

#[test]
fn tool_execution_result_helpers_preserve_payloads_and_normalize_annotations() {
    let payload = json!({"value": 42});
    let result: ToolExecutionResult = payload.clone().into();
    assert_eq!(result.result, payload);
    assert!(result.annotation.is_none());

    let result = result.with_annotation(json!({"source": "tool"}));
    assert_eq!(result.annotation, Some(json!({"source": "tool"})));
    assert!(result.clone().without_annotation().annotation.is_none());
    assert!(result.with_annotation(json!(null)).annotation.is_none());

    let outcome = ToolExecutionInterceptOutcome::new(json!("ok"))
        .with_pending_mark(
            PendingMarkSpec::builder()
                .name("tool.mark")
                .category(EventCategory::custom())
                .build(),
        )
        .with_annotation(json!({"source": "middleware"}));
    assert_eq!(outcome.annotation, Some(json!({"source": "middleware"})));
    assert_eq!(outcome.pending_marks.len(), 1);

    let outcome = outcome.without_annotation();
    assert!(outcome.annotation.is_none());
    assert_eq!(outcome.pending_marks.len(), 1);
    assert!(outcome.with_annotation(json!(null)).annotation.is_none());
}

#[test]
fn tool_execution_null_annotations_have_stable_equality_and_round_trips() {
    let result_with_null = ToolExecutionResult {
        result: json!({"value": 42}),
        annotation: Some(serde_json::Value::Null),
    };
    let canonical_result = ToolExecutionResult::new(json!({"value": 42}));
    assert_eq!(result_with_null, canonical_result);

    let encoded = serde_json::to_value(&result_with_null).expect("result should serialize");
    assert_eq!(encoded, json!({"result": {"value": 42}}));
    let decoded: ToolExecutionResult =
        serde_json::from_value(encoded).expect("result should deserialize");
    assert_eq!(decoded, result_with_null);

    let outcome_from_result = ToolExecutionInterceptOutcome::from(result_with_null);
    assert!(outcome_from_result.annotation.is_none());

    let outcome_with_null = ToolExecutionInterceptOutcome {
        result: json!("ok"),
        annotation: Some(serde_json::Value::Null),
        pending_marks: Vec::new(),
    };
    let canonical_outcome = ToolExecutionInterceptOutcome::new(json!("ok"));
    assert_eq!(outcome_with_null, canonical_outcome);

    let encoded = serde_json::to_value(&outcome_with_null).expect("outcome should serialize");
    assert_eq!(encoded, json!({"result": "ok", "pending_marks": []}));
    let decoded: ToolExecutionInterceptOutcome =
        serde_json::from_value(encoded).expect("outcome should deserialize");
    assert_eq!(decoded, outcome_with_null);
    assert!(
        outcome_with_null
            .into_execution_result()
            .annotation
            .is_none()
    );
}

#[test]
fn category_profile_serializes_non_null_tool_result_annotation() {
    let mut profile = CategoryProfile::builder()
        .tool_result_annotation(json!({"opaque": true}))
        .build();
    assert_eq!(
        profile.tool_result_annotation,
        Some(json!({"opaque": true}))
    );
    assert_eq!(
        serde_json::to_value(&profile).unwrap()["tool_result_annotation"],
        json!({"opaque": true})
    );
    profile.tool_result_annotation = Some(serde_json::Value::Null);
    assert!(
        serde_json::to_value(profile)
            .unwrap()
            .get("tool_result_annotation")
            .is_none()
    );
}

#[test]
fn event_returns_an_owned_tool_result_annotation() {
    let annotation = json!({"opaque": true});
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("tool").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        Some(
            CategoryProfile::builder()
                .tool_result_annotation(annotation.clone())
                .build(),
        ),
    ));

    assert_eq!(event.tool_result_annotation(), Some(annotation));
}

#[test]
fn tool_execution_outcome_converts_without_exposing_pending_marks() {
    let result = ToolExecutionResult::annotated(json!({"value": 42}), json!({"source": "tool"}));
    let outcome = ToolExecutionInterceptOutcome::from(result.clone()).with_pending_mark(
        PendingMarkSpec::builder()
            .name("tool.mark")
            .category(EventCategory::custom())
            .build(),
    );
    assert_eq!(outcome.into_execution_result(), result);
}

#[test]
fn tool_execution_intercept_outcome_converts_from_json() {
    let result = json!({"value": 42});
    let outcome: ToolExecutionInterceptOutcome = result.clone().into();
    assert_eq!(outcome, ToolExecutionInterceptOutcome::new(result));
}

#[test]
fn annotated_request_helpers_cover_portable_and_native_components() {
    let mut request = AnnotatedLlmRequest {
        instructions: Some(MessageContent::Parts(vec![ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "refusal".into(),
            value: json!({"refusal": "instruction refusal"}),
        }])),
        ..AnnotatedLlmRequest::default()
    };
    assert_eq!(request.system_prompt(), Some("instruction refusal"));

    request.instructions = None;
    request.messages = vec![Message::Developer {
        content: MessageContent::Parts(vec![ContentPart::Text {
            text: "developer instruction".into(),
            extra: Map::new(),
        }]),
        name: None,
    }];
    assert_eq!(request.system_prompt(), Some("developer instruction"));

    for (content, expected) in [
        (json!("native string"), Some("native string")),
        (
            json!([{"type": "input_text", "text": "native array"}]),
            Some("native array"),
        ),
        (json!({"not": "content"}), None),
    ] {
        request.messages = vec![Message::ProviderNative {
            provider: "openai_responses".into(),
            kind: "message".into(),
            value: json!({"role": "user", "content": content}),
        }];
        assert_eq!(request.last_user_message(), expected);
    }

    request.messages = vec![Message::Assistant {
        content: Some(MessageContent::Parts(vec![ContentPart::ToolUse {
            id: "call_1".into(),
            name: "lookup".into(),
            input: json!({}),
            extra: Map::new(),
        }])),
        tool_calls: Some(vec![]),
        name: None,
    }];
    assert!(request.has_tool_calls());

    request.messages = vec![Message::Assistant {
        content: Some(MessageContent::Parts(vec![ContentPart::ProviderNative {
            provider: "anthropic_messages".into(),
            kind: "server_tool_use".into(),
            value: json!({"type": "server_tool_use"}),
        }])),
        tool_calls: None,
        name: None,
    }];
    assert!(request.has_tool_calls());

    request.messages = vec![Message::ToolCallItem {
        id: None,
        call_id: "call_2".into(),
        name: "lookup".into(),
        arguments: json!({}),
        extra: Map::new(),
    }];
    assert!(request.has_tool_calls());

    request.messages = vec![Message::ProviderNative {
        provider: "openai_responses".into(),
        kind: "function_call".into(),
        value: json!({"type": "function_call"}),
    }];
    assert!(request.has_tool_calls());
}

#[test]
fn annotated_response_text_reads_provider_native_text_and_refusal() {
    for (value, expected) in [
        (json!({"text": "native text"}), "native text"),
        (json!({"refusal": "native refusal"}), "native refusal"),
    ] {
        let response = AnnotatedLlmResponse {
            message: Some(MessageContent::Parts(vec![ContentPart::ProviderNative {
                provider: "openai_responses".into(),
                kind: "native".into(),
                value,
            }])),
            ..AnnotatedLlmResponse::default()
        };
        assert_eq!(response.response_text(), Some(expected));
    }
}
