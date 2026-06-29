// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization compatibility tests for shared NeMo Relay DTOs.

use std::sync::Arc;

use nemo_relay_types::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings,
};
use nemo_relay_types::api::llm::{LlmAttributes, LlmRequest};
use nemo_relay_types::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use serde_json::{Map, json};

#[test]
fn event_round_trips_with_annotated_llm_profiles() {
    let request = AnnotatedLlmRequest {
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
