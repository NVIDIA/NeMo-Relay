// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for types in the NeMo Flow core crate.

use std::sync::Arc;

use serde_json::{Map, json};
use uuid::{Uuid, Version};

use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use crate::api::llm::{LlmAttributes, LlmHandle, LlmRequest};
use crate::api::scope::{ScopeAttributes, ScopeHandle, ScopeType};
use crate::api::tool::{ToolAttributes, ToolHandle};
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use crate::codec::response::AnnotatedLlmResponse;

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
fn llm_request_serializes_explicit_headers_and_content() {
    let mut headers = Map::new();
    headers.insert("x-agent".to_string(), json!("planner"));

    let request = LlmRequest {
        headers,
        content: json!({"messages": [{"role": "user", "content": "hi"}]}),
    };

    let encoded = serde_json::to_value(&request).unwrap();
    assert_eq!(encoded["headers"]["x-agent"], json!("planner"));
    assert_eq!(encoded["content"]["messages"][0]["role"], json!("user"));

    let decoded: LlmRequest = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.headers.get("x-agent"), Some(&json!("planner")));
}

#[test]
fn event_accessors_cover_scope_tool_llm_and_mark_variants() {
    let parent_uuid = Some(Uuid::now_v7());
    let scope_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let mark_uuid = Uuid::now_v7();

    let scope_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(scope_uuid)
            .name("scope")
            .data(json!({"task": "classify"}))
            .metadata(json!({"region": "us"}))
            .build(),
        ScopeCategory::Start,
        scope_attributes_to_strings(ScopeAttributes::RELOCATABLE),
        EventCategory::from(ScopeType::Function),
        None,
    ));
    assert_eq!(scope_event.kind(), "scope");
    assert_eq!(scope_event.scope_category(), Some(ScopeCategory::Start));
    assert_eq!(scope_event.parent_uuid(), parent_uuid);
    assert_eq!(scope_event.uuid(), scope_uuid);
    assert_eq!(scope_event.name(), "scope");
    assert_eq!(scope_event.data(), Some(&json!({"task": "classify"})));
    assert_eq!(scope_event.metadata(), Some(&json!({"region": "us"})));
    assert_eq!(
        scope_event.attributes(),
        Some(["relocatable".to_string()].as_slice())
    );
    assert_eq!(scope_event.scope_type(), Some(ScopeType::Function));
    assert_eq!(scope_event.input(), Some(&json!({"task": "classify"})));
    assert!(scope_event.timestamp().timestamp() > 0);

    let tool_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(tool_uuid)
            .name("search")
            .data(json!({"answer": 42}))
            .build(),
        ScopeCategory::End,
        tool_attributes_to_strings(ToolAttributes::REMOTE),
        EventCategory::tool(),
        Some(
            CategoryProfile::builder()
                .tool_call_id("tool-call-1")
                .build(),
        ),
    ));
    assert_eq!(tool_event.kind(), "scope");
    assert_eq!(tool_event.scope_category(), Some(ScopeCategory::End));
    assert_eq!(
        tool_event.attributes(),
        Some(["remote".to_string()].as_slice())
    );
    assert_eq!(tool_event.output(), Some(&json!({"answer": 42})));
    assert_eq!(tool_event.tool_call_id(), Some("tool-call-1"));
    assert_eq!(tool_event.scope_type(), Some(ScopeType::Tool));
    assert_eq!(tool_event.model_name(), None);

    let llm_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(llm_uuid)
            .name("planner")
            .data(json!({"prompt": "hello"}))
            .build(),
        ScopeCategory::Start,
        llm_attributes_to_strings(LlmAttributes::STREAMING),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("gpt-test").build()),
    ));
    assert_eq!(llm_event.kind(), "scope");
    assert_eq!(
        llm_event.attributes(),
        Some(["streaming".to_string()].as_slice())
    );
    assert_eq!(llm_event.input(), Some(&json!({"prompt": "hello"})));
    assert_eq!(llm_event.model_name(), Some("gpt-test"));
    assert_eq!(llm_event.scope_type(), Some(ScopeType::Llm));
    assert_eq!(llm_event.output(), None);

    let mark_event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(mark_uuid)
            .name("checkpoint")
            .data(json!({"ok": true}))
            .metadata(json!({"source": "types"}))
            .build(),
        None,
        None,
    ));
    assert_eq!(mark_event.kind(), "mark");
    assert_eq!(mark_event.uuid(), mark_uuid);
    assert_eq!(mark_event.attributes(), None);
    assert_eq!(mark_event.scope_type(), None);
    assert_eq!(mark_event.input(), None);
    assert_eq!(mark_event.output(), None);
    assert_eq!(mark_event.tool_call_id(), None);
}

#[test]
fn event_json_value_uses_canonical_subscriber_shape() {
    let request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("hi".into()),
            name: None,
        }],
        model: Some("demo-model".into()),
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
        id: Some("resp-1".into()),
        model: Some("demo-model".into()),
        message: Some(MessageContent::Text("hello".into())),
        tool_calls: None,
        finish_reason: None,
        usage: None,
        api_specific: None,
        extra: Map::new(),
    };
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("llm")
            .data(json!({"input": true}))
            .metadata(json!({"trace": "abc"}))
            .build(),
        ScopeCategory::End,
        llm_attributes_to_strings(LlmAttributes::STATEFUL),
        EventCategory::llm(),
        Some(
            CategoryProfile::builder()
                .model_name("demo-model")
                .annotated_request(Arc::new(request))
                .annotated_response(Arc::new(response))
                .build(),
        ),
    ));

    let value = event.to_json_value();
    assert_eq!(value["kind"], json!("scope"));
    assert_eq!(value["scope_category"], json!("end"));
    assert_eq!(value["category"], json!("llm"));
    assert_eq!(value["data"], json!({"input": true}));
    assert_eq!(value["metadata"], json!({"trace": "abc"}));
    assert_eq!(value["annotated_request"]["model"], json!("demo-model"));
    assert_eq!(value["annotated_response"]["id"], json!("resp-1"));

    let encoded = event.to_json_string().unwrap();
    let decoded: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, value);
}

#[test]
fn atof_event_builders_construct_concrete_events() {
    let parent_uuid = Some(Uuid::now_v7());

    let scope_start = ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name("scope-start")
            .data(json!({"input": true}))
            .metadata(json!({"phase": 1}))
            .build(),
        ScopeCategory::Start,
        scope_attributes_to_strings(ScopeAttributes::RELOCATABLE),
        EventCategory::function(),
        None,
    );
    assert_eq!(scope_start.base.parent_uuid, parent_uuid);
    assert_eq!(scope_start.base.name, "scope-start");
    assert_eq!(scope_start.category, EventCategory::function());
    assert_eq!(scope_start.base.data, Some(json!({"input": true})));
    assert!(scope_start.base.timestamp.timestamp() > 0);

    let llm_end = ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name("llm-end")
            .data(json!({"text": "done"}))
            .build(),
        ScopeCategory::End,
        llm_attributes_to_strings(LlmAttributes::STATEFUL),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("demo-model").build()),
    );
    assert_eq!(llm_end.base.parent_uuid, parent_uuid);
    assert_eq!(llm_end.base.name, "llm-end");
    assert_eq!(llm_end.base.data, Some(json!({"text": "done"})));
    assert_eq!(
        llm_end
            .category_profile
            .as_ref()
            .and_then(|profile| profile.model_name.as_deref()),
        Some("demo-model")
    );
    assert!(llm_end.base.timestamp.timestamp() > 0);

    let mark = MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name("mark")
            .data(json!({"ok": true}))
            .metadata(json!({"source": "unit-test"}))
            .build(),
        None,
        None,
    );
    assert_eq!(mark.base.parent_uuid, parent_uuid);
    assert_eq!(mark.base.name, "mark");
    assert_eq!(mark.base.data, Some(json!({"ok": true})));
    assert_eq!(mark.base.metadata, Some(json!({"source": "unit-test"})));
    assert!(mark.base.timestamp.timestamp() > 0);
}

#[test]
fn base_event_and_flattened_specialized_builders_work() {
    let base = BaseEvent::builder()
        .parent_uuid(Uuid::nil())
        .name("base-name")
        .data(json!({"base": true}))
        .metadata(json!({"layer": "base"}))
        .build();

    assert_eq!(base.parent_uuid, Some(Uuid::nil()));
    assert_eq!(base.name, "base-name");
    assert_eq!(base.data, Some(json!({"base": true})));
    assert_eq!(base.metadata, Some(json!({"layer": "base"})));
    assert!(base.timestamp.timestamp() > 0);

    let tool_start = ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid(Uuid::nil())
            .uuid(base.uuid)
            .name("tool-start")
            .data(json!({"query": "override"}))
            .metadata(json!({"layer": "event"}))
            .build(),
        ScopeCategory::Start,
        tool_attributes_to_strings(ToolAttributes::REMOTE),
        EventCategory::tool(),
        Some(CategoryProfile::builder().tool_call_id("tool-42").build()),
    );

    assert_eq!(tool_start.base.parent_uuid, Some(Uuid::nil()));
    assert_eq!(tool_start.base.uuid, base.uuid);
    assert_eq!(tool_start.base.name, "tool-start");
    assert_eq!(tool_start.base.data, Some(json!({"query": "override"})));
    assert_eq!(tool_start.base.metadata, Some(json!({"layer": "event"})));
    assert_eq!(
        tool_start
            .category_profile
            .as_ref()
            .and_then(|profile| profile.tool_call_id.as_deref()),
        Some("tool-42")
    );

    let tool_end = ScopeEvent::new(
        BaseEvent::builder().name("tool-end").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        None,
    );
    assert_eq!(tool_end.base.name, "tool-end");
    assert_eq!(tool_end.base.data, None);
    assert_eq!(tool_end.base.metadata, None);
    assert_eq!(tool_end.category_profile, None);

    let llm_start = ScopeEvent::new(
        BaseEvent::builder().name("llm-start").build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name("gpt-test").build()),
    );
    assert_eq!(
        llm_start
            .category_profile
            .as_ref()
            .and_then(|profile| profile.model_name.as_deref()),
        Some("gpt-test")
    );

    let llm_end = ScopeEvent::new(
        BaseEvent::builder().name("llm-end").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        None,
    );
    assert_eq!(llm_end.category_profile, None);

    let mark = MarkEvent::new(
        BaseEvent::builder().name("mark-builder").build(),
        None,
        None,
    );
    assert_eq!(mark.base.name, "mark-builder");
    assert!(mark.base.timestamp.timestamp() > 0);
}
