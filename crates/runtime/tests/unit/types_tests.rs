// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime event normalization through codec and pricing services.
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, EventNormalizationExt, EventSanitizeFields,
    EventTelemetryExt, ScopeCategory, ScopeEvent, llm_attributes_to_strings,
};
use crate::api::llm::LlmHandle;
use crate::api::llm::{LlmAttributes, LlmRequest};
use crate::api::scope::{ScopeAttributes, ScopeHandle, ScopeType};
use crate::api::tool::{ToolAttributes, ToolHandle};
use nemo_relay_types::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use serde_json::{Map, json};
use std::sync::Arc;
use uuid::{Uuid, Version};

fn llm_start_event(data: serde_json::Value, profile: CategoryProfile) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("llm").data(data).build(),
        ScopeCategory::Start,
        llm_attributes_to_strings(LlmAttributes::empty()),
        EventCategory::llm(),
        Some(profile),
    ))
}

fn llm_end_event(data: serde_json::Value, profile: CategoryProfile) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("llm").data(data).build(),
        ScopeCategory::End,
        llm_attributes_to_strings(LlmAttributes::empty()),
        EventCategory::llm(),
        Some(profile),
    ))
}

fn annotated_request(model: &str, text: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![Message::User {
            content: MessageContent::Text(text.into()),
            name: None,
        }],
        model: Some(model.into()),
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
    }
}

fn annotated_response(id: &str, model: &str, text: &str) -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: Some(id.into()),
        model: Some(model.into()),
        message: Some(MessageContent::Text(text.into())),
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: Map::new(),
    }
}

#[test]
fn normalized_llm_response_prefers_annotation_over_raw_output() {
    // Annotation present: returned (borrowed), ignoring the conflicting raw output.
    let response = annotated_response("resp-1", "demo-model", "from-annotation");
    let event = llm_end_event(
        json!({"choices": [{"message": {"role": "assistant", "content": "from-raw"}}]}),
        CategoryProfile::builder()
            .annotated_response(Arc::new(response))
            .build(),
    );
    let normalized = event.normalized_llm_response().expect("annotation present");
    assert_eq!(normalized.response_text(), Some("from-annotation".into()));
}

#[test]
fn normalized_llm_response_falls_back_to_codec_decode() {
    // No annotation: best-effort decode of the raw provider output.
    let event = llm_end_event(
        json!({
            "model": "gpt-4o",
            "choices": [{
                "message": {"role": "assistant", "content": "from-raw"},
                "finish_reason": "stop"
            }]
        }),
        CategoryProfile::default(),
    );
    let normalized = event
        .normalized_llm_response()
        .expect("decodes raw chat output");
    assert_eq!(normalized.response_text(), Some("from-raw".into()));
}

#[test]
fn normalized_llm_response_none_for_non_provider_output() {
    let event = llm_end_event(json!({"answer": "x"}), CategoryProfile::default());
    assert!(event.normalized_llm_response().is_none());
}

#[test]
fn normalized_llm_request_decodes_wrapped_request_when_unannotated() {
    // No annotation: decode the wrapped LlmRequest from the start-event input.
    let event = llm_start_event(
        json!({
            "headers": {},
            "content": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        }),
        CategoryProfile::default(),
    );
    let normalized = event
        .normalized_llm_request()
        .expect("decodes wrapped chat request");
    assert!(!normalized.messages.is_empty());
}

#[test]
fn normalized_llm_request_uses_event_name_provider_hint() {
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("anthropic.messages")
            .data(json!({
                "headers": {},
                "content": {
                    "model": "claude-3-5-sonnet",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stop_sequences": ["END"]
                }
            }))
            .build(),
        ScopeCategory::Start,
        llm_attributes_to_strings(LlmAttributes::empty()),
        EventCategory::llm(),
        Some(CategoryProfile::default()),
    ));
    let normalized = event
        .normalized_llm_request()
        .expect("decodes wrapped anthropic request");
    let stop = normalized
        .params
        .as_ref()
        .and_then(|params| params.stop.as_ref())
        .expect("anthropic stop_sequences are normalized");
    assert_eq!(stop, &vec!["END".to_string()]);
    assert!(!normalized.extra.contains_key("stop_sequences"));
}

#[test]
fn normalized_llm_request_prefers_annotation() {
    let request = annotated_request("demo-model", "annotated");
    let event = llm_start_event(
        json!({"headers": {}, "content": {"messages": []}}),
        CategoryProfile::builder()
            .annotated_request(Arc::new(request))
            .build(),
    );
    assert!(event.normalized_llm_request().is_some());
}

#[test]
fn handle_constructors_preserve_supplied_metadata() {
    let parent_uuid = Some(Uuid::now_v7());
    let data = Some(json!({"trace": "abc"}));
    let metadata = Some(json!({"source": "unit-test"}));

    let scope = ScopeHandle::builder()
        .name("agent".to_string())
        .scope_type(ScopeType::Agent)
        .attributes(ScopeAttributes::PARALLEL)
        .parent_uuid_opt(parent_uuid)
        .data_opt(data.clone())
        .metadata_opt(metadata.clone())
        .build();
    assert_eq!(scope.name, "agent");
    assert_eq!(scope.scope_type, ScopeType::Agent);
    assert_eq!(scope.attributes, ScopeAttributes::PARALLEL);
    assert_eq!(scope.parent_uuid, parent_uuid);
    assert_eq!(scope.data, data);
    assert_eq!(scope.metadata, metadata);
    assert_eq!(scope.uuid.get_version(), Some(Version::SortRand));

    let tool = ToolHandle::builder()
        .name("search".to_string())
        .attributes(ToolAttributes::REMOTE)
        .parent_uuid_opt(parent_uuid)
        .data(json!({"query": "rust"}))
        .metadata(json!({"kind": "tool"}))
        .build();
    assert_eq!(tool.name, "search");
    assert_eq!(tool.attributes, ToolAttributes::REMOTE);
    assert_eq!(tool.parent_uuid, parent_uuid);
    assert_eq!(tool.tool_call_id, None);
    assert_eq!(tool.uuid.get_version(), Some(Version::SortRand));

    let llm = LlmHandle::builder()
        .name("planner".to_string())
        .attributes(LlmAttributes::STATEFUL | LlmAttributes::STREAMING)
        .parent_uuid_opt(parent_uuid)
        .data(json!({"request": 1}))
        .metadata(json!({"provider": "test"}))
        .build();
    assert_eq!(llm.name, "planner");
    assert_eq!(
        llm.attributes,
        LlmAttributes::STATEFUL | LlmAttributes::STREAMING
    );
    assert_eq!(llm.parent_uuid, parent_uuid);
    assert_eq!(llm.model_name, None);
    assert_eq!(llm.uuid.get_version(), Some(Version::SortRand));
}

#[test]
fn telemetry_request_projects_native_items_without_changing_wire_annotations() {
    use nemo_relay_codecs::openai_responses::OpenAIResponsesCodec;
    use nemo_relay_types::codec::traits::LlmCodec;

    let input = r#"{"literal":"keep this as text"}"#;
    let raw = LlmRequest {
        headers: Map::new(),
        content: json!({"input": [
            {"type": "custom_tool_call", "id": "item-1", "call_id": "call-1",
             "name": "exec", "input": input, "future": {"keep": true}},
            {"type": "custom_tool_call_output", "id": "item-2", "call_id": "call-1",
             "output": [{"type": "input_text", "text": "done"}], "future": 2}
        ]}),
    };
    for annotated in [false, true] {
        let annotation = OpenAIResponsesCodec.decode(&raw).unwrap();
        let event = llm_start_event(
            serde_json::to_value(&raw).unwrap(),
            CategoryProfile::builder()
                .annotated_request_opt(annotated.then(|| Arc::new(annotation)))
                .build(),
        );
        let before = event.to_json_string().unwrap();
        let view = event.telemetry_llm_request().unwrap();
        assert!(matches!(&view.messages[0], Message::ToolCallItem {
            id: Some(id), call_id, name, arguments, ..
        } if id == "item-1" && call_id == "call-1" && name == "exec" && arguments == &json!(input)));
        assert!(matches!(&view.messages[1], Message::ToolResultItem {
            id: Some(id), call_id, output, ..
        } if id == "item-2" && call_id == "call-1" && output == &raw.content["input"][1]["output"]));
        let original = event.normalized_llm_request().unwrap();
        assert!(matches!(
            original.messages[0],
            Message::ProviderNative { .. }
        ));
        assert_eq!(
            OpenAIResponsesCodec
                .encode(&original, &raw)
                .unwrap()
                .content,
            raw.content
        );
        assert_eq!(event.to_json_string().unwrap(), before);
    }
}

#[test]
fn telemetry_request_keeps_unknown_and_invalid_native_items_borrowed() {
    let mut request = annotated_request("demo-model", "hello");
    for (provider, kind, value) in [
        (
            "other",
            "custom_tool_call",
            json!({"type": "custom_tool_call", "call_id": "call-1", "name": "exec", "input": "text"}),
        ),
        (
            "openai_responses",
            "custom_tool_call",
            json!({"type": "custom_tool_call", "name": "exec", "input": "text"}),
        ),
        (
            "openai_responses",
            "custom_tool_call",
            json!({"type": "custom_tool_call_output", "call_id": "call-1", "output": "done"}),
        ),
        (
            "openai_responses",
            "future_item",
            json!({"type": "future_item", "payload": "text"}),
        ),
    ] {
        request.messages.push(Message::ProviderNative {
            provider: provider.into(),
            kind: kind.into(),
            value,
        });
    }
    let expected = request.clone();
    let event = llm_start_event(
        json!(null),
        CategoryProfile::builder()
            .annotated_request(Arc::new(request))
            .build(),
    );
    let view = event.telemetry_llm_request().unwrap();
    assert!(matches!(view, std::borrow::Cow::Borrowed(_)));
    assert_eq!(*view, expected);
}

#[test]
fn telemetry_request_uses_sanitized_annotation_and_cannot_restore_removed_payloads() {
    let mut event = llm_start_event(
        json!({"headers": {}, "content": {"input": [
            {"type": "custom_tool_call", "call_id": "raw-call", "name": "exec", "input": "raw-secret"}
        ]}}),
        CategoryProfile::builder().annotated_request(Arc::new(AnnotatedLlmRequest {
            messages: vec![Message::ProviderNative {
                provider: "openai_responses".into(), kind: "custom_tool_call".into(),
                value: json!({"type": "custom_tool_call", "call_id": "safe-call", "name": "exec", "input": "redacted"}),
            }], ..Default::default()
        })).build(),
    );
    let view = event.telemetry_llm_request().unwrap();
    assert!(
        matches!(&view.messages[0], Message::ToolCallItem { call_id, arguments, .. }
        if call_id == "safe-call" && arguments == &json!("redacted"))
    );
    event.apply_sanitize_fields(EventSanitizeFields::default());
    assert!(event.telemetry_llm_request().is_none());
    assert!(event.telemetry_llm_tool_calls().is_none());
}

/// Verifies malformed entries preserve valid calls and positional fallback IDs.
#[test]
fn telemetry_tool_calls_skip_non_object_entries_without_renumbering() {
    for invalid in [
        json!(null),
        json!(false),
        json!(42),
        json!("invalid"),
        json!([]),
    ] {
        let event = llm_end_event(
            json!({"tool_calls": [
                {"id": "call-1", "name": "first", "arguments": {"q": "one"}},
                invalid,
                {"name": "later", "arguments": {"q": "two"}}
            ]}),
            CategoryProfile::default(),
        );
        let calls = event
            .telemetry_llm_tool_calls()
            .expect("valid calls survive");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].arguments, json!({"q": "one"}));
        assert_eq!(calls[1].id, "later:3");
        assert_eq!(calls[1].arguments, json!({"q": "two"}));

        let invalid_only =
            llm_end_event(json!({"tool_calls": [invalid]}), CategoryProfile::default());
        assert!(invalid_only.telemetry_llm_tool_calls().is_none());
    }
}

#[test]
fn telemetry_tool_calls_use_annotated_semantics_and_preserve_matching_metadata() {
    use nemo_relay_types::codec::response::ResponseToolCall;
    let event = llm_end_event(
        json!({"tool_calls": [
            {"id": "call-1", "name": "exec", "arguments": "raw-secret", "provider_data": {"status": "ok"}},
            null,
            {"id": "raw-only", "name": "other", "arguments": "{}"}
        ]}),
        CategoryProfile::builder()
            .annotated_response(Arc::new(AnnotatedLlmResponse {
                tool_calls: Some(vec![ResponseToolCall {
                    id: "call-1".into(),
                    name: "exec".into(),
                    arguments: json!("redacted"),
                }]),
                ..Default::default()
            }))
            .build(),
    );
    let before = event.to_json_string().unwrap();
    let calls = event.telemetry_llm_tool_calls().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-1");
    assert_eq!(calls[0].arguments, json!("redacted"));
    assert_eq!(
        calls[0].extra.as_ref().unwrap()["provider_data"]["status"],
        "ok"
    );
    assert_eq!(event.to_json_string().unwrap(), before);
}

#[test]
fn telemetry_tool_calls_distinguish_empty_lists_from_partial_annotations() {
    for tool_calls in [None, Some(Vec::new())] {
        let partial = tool_calls.is_none();
        let event = llm_end_event(
            json!({"tool_calls": [{"id": "call-1", "name": "exec", "arguments": {"q": "test"}}]}),
            CategoryProfile::builder()
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    tool_calls,
                    ..Default::default()
                }))
                .build(),
        );
        let calls = event
            .telemetry_llm_tool_calls()
            .expect("an explicit call list or raw fallback is available");
        if partial {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].id, "call-1");
            assert_eq!(calls[0].name, "exec");
            assert_eq!(calls[0].arguments, json!({"q": "test"}));
        } else {
            assert!(calls.is_empty());
        }
    }
}
