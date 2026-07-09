// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for types in the NeMo Relay core crate.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{Map, json};
use uuid::{Uuid, Version};

use crate::api::event::{
    BaseEvent, CategoryProfile, DataSchema, Event, EventCategory, EventNormalizationExt,
    EventSanitizeFields, MarkEvent, ScopeCategory, ScopeEvent, attributes_from_handle,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use crate::api::llm::{LlmAttributes, LlmHandle, LlmRequest};
use crate::api::scope::{HandleAttributes, ScopeAttributes, ScopeHandle, ScopeType};
use crate::api::tool::{ToolAttributes, ToolHandle};
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use crate::codec::response::AnnotatedLlmResponse;
use crate::config_editor::{EditorConfig, EditorFieldKind};

#[derive(Default, serde::Serialize)]
struct NestedEditorFixture {
    enabled: bool,
}

crate::editor_config! {
    impl NestedEditorFixture {
        enabled => {
            label: "Enabled",
            kind: Boolean,
        },
    }
}

#[derive(Default, serde::Serialize)]
struct EditorFixture {
    enabled: bool,
    name: String,
    count: u64,
    ratio: f64,
    mode: String,
    headers: std::collections::BTreeMap<String, String>,
    payload: serde_json::Value,
    nested: NestedEditorFixture,
}

crate::editor_config! {
    impl EditorFixture {
        enabled => {
            label: "Enabled",
            kind: Boolean,
        },
        name => {
            label: "Name",
            kind: String,
            optional: true,
        },
        count => {
            label: "Count",
            kind: Integer,
        },
        ratio => {
            label: "Ratio",
            kind: Float,
        },
        mode => {
            label: "Mode",
            kind: Enum,
            values: ["fast", "safe"],
        },
        headers => {
            label: "Headers",
            kind: StringMap,
        },
        payload => {
            label: "Payload",
            kind: Json,
        },
        nested => {
            label: "Nested",
            kind: Section,
            nested: NestedEditorFixture,
            default: NestedEditorFixture,
        },
    }
}

fn annotated_request(model: &str, text: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
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
fn event_categories_round_trip_all_scope_variants() {
    let cases = [
        (EventCategory::agent(), ScopeType::Agent, "agent"),
        (EventCategory::function(), ScopeType::Function, "function"),
        (EventCategory::tool(), ScopeType::Tool, "tool"),
        (EventCategory::llm(), ScopeType::Llm, "llm"),
        (
            EventCategory::retriever(),
            ScopeType::Retriever,
            "retriever",
        ),
        (EventCategory::embedder(), ScopeType::Embedder, "embedder"),
        (EventCategory::reranker(), ScopeType::Reranker, "reranker"),
        (
            EventCategory::guardrail(),
            ScopeType::Guardrail,
            "guardrail",
        ),
        (
            EventCategory::evaluator(),
            ScopeType::Evaluator,
            "evaluator",
        ),
        (EventCategory::custom(), ScopeType::Custom, "custom"),
        (EventCategory::unknown(), ScopeType::Unknown, "unknown"),
    ];

    for (category, scope_type, wire_value) in cases {
        assert_eq!(category.as_str(), wire_value);
        assert_eq!(category.to_scope_type(), scope_type);
        assert_eq!(EventCategory::from(scope_type), category);
        assert_eq!(ScopeType::from(&category), scope_type);
    }

    let vendor_category = EventCategory::new("vendor.special");
    assert_eq!(vendor_category.as_str(), "vendor.special");
    assert_eq!(vendor_category.to_scope_type(), ScopeType::Unknown);
}

#[test]
fn event_mutators_schema_flags_and_attribute_helpers_cover_remaining_paths() {
    let mut scope_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("scope")
            .data(json!({"payload": true}))
            .data_schema(DataSchema::builder().name("payload").version("1").build())
            .build(),
        ScopeCategory::Start,
        vec!["relocatable".to_string(), "parallel".to_string()],
        EventCategory::custom(),
        Some(CategoryProfile::builder().subtype("initial").build()),
    ));

    assert!(scope_event.is_scope_start());
    assert!(!scope_event.is_scope_end());
    assert_eq!(
        scope_event
            .data_schema()
            .map(|schema| (schema.name.as_str(), schema.version.as_str())),
        Some(("payload", "1"))
    );
    assert_eq!(
        scope_event.attributes(),
        Some(["parallel".to_string(), "relocatable".to_string()].as_slice())
    );

    let scope_profile = scope_event.category_profile_mut().unwrap();
    scope_profile.subtype = Some("updated".to_string());
    assert_eq!(
        scope_event
            .category_profile()
            .and_then(|profile| profile.subtype.as_deref()),
        Some("updated")
    );

    let mut mark_event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("mark").build(),
        Some(EventCategory::evaluator()),
        Some(CategoryProfile::builder().subtype("score").build()),
    ));
    assert!(!mark_event.is_scope_start());
    assert!(!mark_event.is_scope_end());
    assert_eq!(
        mark_event.category().map(EventCategory::as_str),
        Some("evaluator")
    );

    let mark_profile = mark_event.category_profile_mut().unwrap();
    mark_profile.subtype = Some("quality".to_string());
    assert_eq!(
        mark_event
            .category_profile()
            .and_then(|profile| profile.subtype.as_deref()),
        Some("quality")
    );

    let scope_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("scope-end").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::function(),
        None,
    ));
    assert!(!scope_end.is_scope_start());
    assert!(scope_end.is_scope_end());

    assert_eq!(
        attributes_from_handle(HandleAttributes::Scope(
            ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE
        )),
        vec!["parallel".to_string(), "relocatable".to_string()]
    );
    assert_eq!(
        attributes_from_handle(HandleAttributes::Tool(ToolAttributes::REMOTE)),
        vec!["remote".to_string()]
    );
    assert_eq!(
        attributes_from_handle(HandleAttributes::Llm(
            LlmAttributes::STATEFUL | LlmAttributes::STREAMING
        )),
        vec!["stateful".to_string(), "streaming".to_string()]
    );
    assert!(attributes_from_handle(HandleAttributes::Scope(ScopeAttributes::empty())).is_empty());
    assert!(attributes_from_handle(HandleAttributes::Tool(ToolAttributes::empty())).is_empty());
    assert!(attributes_from_handle(HandleAttributes::Llm(LlmAttributes::empty())).is_empty());
}

#[test]
fn event_timestamp_deserialization_accepts_strings_and_epoch_micros() {
    let timestamp = DateTime::parse_from_rfc3339("2026-01-02T03:04:05Z")
        .unwrap()
        .with_timezone(&Utc);

    let from_string: Event = serde_json::from_value(json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": Uuid::now_v7(),
        "timestamp": timestamp.to_rfc3339(),
        "name": "string-timestamp"
    }))
    .unwrap();
    assert_eq!(*from_string.timestamp(), timestamp);

    let micros = timestamp.timestamp_micros();
    let from_i64: Event = serde_json::from_value(json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": Uuid::now_v7(),
        "timestamp": micros,
        "name": "i64-timestamp"
    }))
    .unwrap();
    assert_eq!(*from_i64.timestamp(), timestamp);

    let from_u64: Event = serde_json::from_value(json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": Uuid::now_v7(),
        "timestamp": micros as u64,
        "name": "u64-timestamp"
    }))
    .unwrap();
    assert_eq!(*from_u64.timestamp(), timestamp);

    let out_of_range = serde_json::from_value::<Event>(json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": Uuid::now_v7(),
        "timestamp": u64::MAX,
        "name": "bad-timestamp"
    }));
    assert!(out_of_range.is_err());

    let invalid_string = serde_json::from_value::<Event>(json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": Uuid::now_v7(),
        "timestamp": "not-a-timestamp",
        "name": "bad-string-timestamp"
    }));
    assert!(invalid_string.is_err());

    let invalid_type = serde_json::from_value::<Event>(json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": Uuid::now_v7(),
        "timestamp": false,
        "name": "bad-type-timestamp"
    }));
    assert!(invalid_type.is_err());
}

#[test]
fn scope_type_strings_and_editor_metadata_cover_public_helpers() {
    let scope_types = [
        (ScopeType::Agent, "agent"),
        (ScopeType::Function, "function"),
        (ScopeType::Tool, "tool"),
        (ScopeType::Llm, "llm"),
        (ScopeType::Retriever, "retriever"),
        (ScopeType::Embedder, "embedder"),
        (ScopeType::Reranker, "reranker"),
        (ScopeType::Guardrail, "guardrail"),
        (ScopeType::Evaluator, "evaluator"),
        (ScopeType::Custom, "custom"),
        (ScopeType::Unknown, "unknown"),
    ];
    for (scope_type, expected) in scope_types {
        assert_eq!(scope_type.as_str(), expected);
    }

    let schema = EditorFixture::editor_schema();
    assert_eq!(
        schema.field("enabled").map(|field| field.kind),
        Some(EditorFieldKind::Boolean)
    );
    assert_eq!(
        schema
            .field("name")
            .map(|field| (field.kind, field.optional)),
        Some((EditorFieldKind::String, true))
    );
    assert_eq!(
        schema.field("count").map(|field| field.kind),
        Some(EditorFieldKind::Integer)
    );
    assert_eq!(
        schema.field("ratio").map(|field| field.kind),
        Some(EditorFieldKind::Float)
    );
    assert_eq!(
        schema
            .field("mode")
            .map(|field| (field.kind, field.enum_values)),
        Some((EditorFieldKind::Enum, &["fast", "safe"][..]))
    );
    assert_eq!(
        schema.field("headers").map(|field| field.kind),
        Some(EditorFieldKind::StringMap)
    );
    assert_eq!(
        schema.field("payload").map(|field| field.kind),
        Some(EditorFieldKind::Json)
    );

    let nested = schema.field("nested").unwrap();
    assert_eq!(nested.kind, EditorFieldKind::Section);
    assert!(nested.schema().unwrap().field("enabled").is_some());
    assert_eq!(nested.default_value().unwrap(), json!({"enabled": false}));
    assert!(schema.field("missing").is_none());
}

#[test]
fn event_json_value_uses_canonical_subscriber_shape() {
    let request = annotated_request("demo-model", "hi");
    let response = annotated_response("resp-1", "demo-model", "hello");
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

    let value = event.try_to_json_value().unwrap();
    assert_eq!(event.to_json_value(), value);
    assert_eq!(value["kind"], json!("scope"));
    assert_eq!(value["scope_category"], json!("end"));
    assert_eq!(value["category"], json!("llm"));
    assert_eq!(value["data"], json!({"input": true}));
    assert_eq!(value["metadata"], json!({"trace": "abc"}));
    assert!(value.get("annotated_request").is_none());
    assert!(value.get("annotated_response").is_none());
    assert_eq!(
        value["category_profile"]["annotated_request"]["model"],
        json!("demo-model")
    );
    assert_eq!(
        value["category_profile"]["annotated_response"]["id"],
        json!("resp-1")
    );

    let encoded = event.to_json_string().unwrap();
    let decoded: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, value);
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

fn llm_start_event(data: serde_json::Value, profile: CategoryProfile) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("llm").data(data).build(),
        ScopeCategory::Start,
        llm_attributes_to_strings(LlmAttributes::empty()),
        EventCategory::llm(),
        Some(profile),
    ))
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
fn category_profile_wire_empty_accounts_for_annotations() {
    assert!(CategoryProfile::default().is_wire_empty());

    let request_profile = CategoryProfile::builder()
        .annotated_request(Arc::new(annotated_request("demo-model", "hi")))
        .build();
    assert!(!request_profile.is_wire_empty());

    let response_profile = CategoryProfile::builder()
        .annotated_response(Arc::new(annotated_response(
            "resp-1",
            "demo-model",
            "hello",
        )))
        .build();
    assert!(!response_profile.is_wire_empty());
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

#[test]
fn event_deserialization_preserves_unknown_profile_fields_and_mark_profiles() {
    let event: Event = serde_json::from_value(json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": Uuid::now_v7(),
        "timestamp": "2026-02-03T04:05:06Z",
        "name": "quality-check",
        "category": "custom.vendor",
        "category_profile": {
            "subtype": "score",
            "vendor_score": 0.97,
            "vendor_tags": ["safe", "fast"]
        }
    }))
    .unwrap();

    assert_eq!(event.kind(), "mark");
    assert_eq!(
        event.category().map(EventCategory::as_str),
        Some("custom.vendor")
    );
    let profile = event.category_profile().unwrap();
    assert_eq!(profile.subtype.as_deref(), Some("score"));
    assert_eq!(profile.extra["vendor_score"], json!(0.97));
    assert_eq!(profile.extra["vendor_tags"], json!(["safe", "fast"]));
    assert_eq!(event.attributes(), None);
    assert_eq!(event.input(), None);
    assert_eq!(event.output(), None);
}

#[test]
fn event_scope_accessors_return_none_for_unprofiled_and_wrong_phase_payloads() {
    let start_without_profile = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("start").build(),
        ScopeCategory::Start,
        vec!["parallel".into(), "parallel".into(), "relocatable".into()],
        EventCategory::function(),
        None,
    ));
    assert_eq!(
        start_without_profile.attributes(),
        Some(["parallel".to_string(), "relocatable".to_string()].as_slice())
    );
    assert_eq!(start_without_profile.category_profile(), None);
    assert_eq!(start_without_profile.model_name(), None);
    assert_eq!(start_without_profile.tool_call_id(), None);
    assert_eq!(start_without_profile.annotated_request(), None);
    assert_eq!(start_without_profile.annotated_response(), None);
    assert_eq!(start_without_profile.output(), None);

    let end_without_data = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("end").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        None,
    ));
    assert_eq!(end_without_data.input(), None);
    assert_eq!(end_without_data.output(), None);
}

#[test]
fn event_sanitize_fields_snapshot_and_apply_cover_scope_and_mark_variants() {
    let profile = CategoryProfile::builder().subtype("raw").build();
    let mut scope = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("scope")
            .data(json!({"secret": "raw"}))
            .metadata(json!({"meta": "raw"}))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::custom(),
        Some(profile),
    ));
    assert_eq!(scope.sanitize_fields().data, Some(json!({"secret": "raw"})));
    scope.apply_sanitize_fields(EventSanitizeFields {
        data: Some(json!({"secret": "clean"})),
        category_profile: None,
        metadata: None,
    });
    assert_eq!(scope.data(), Some(&json!({"secret": "clean"})));
    assert!(scope.category_profile().is_none());
    assert!(scope.metadata().is_none());

    let original_uuid = Uuid::now_v7();
    let mut mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("mark")
            .uuid(original_uuid)
            .build(),
        None,
        None,
    ));
    mark.apply_sanitize_fields(
        EventSanitizeFields::builder()
            .metadata(json!({"added": true}))
            .build(),
    );
    assert_eq!(mark.uuid(), original_uuid);
    assert_eq!(mark.metadata(), Some(&json!({"added": true})));
}
