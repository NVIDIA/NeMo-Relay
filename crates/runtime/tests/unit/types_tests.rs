// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime event normalization through codec and pricing services.
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, EventNormalizationExt, ScopeCategory,
    ScopeEvent, llm_attributes_to_strings,
};
use crate::api::llm::LlmAttributes;
use crate::api::llm::LlmHandle;
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
    assert_eq!(normalized.response_text(), Some("from-annotation"));
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
    assert_eq!(normalized.response_text(), Some("from-raw"));
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
