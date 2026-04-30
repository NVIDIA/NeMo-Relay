// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for types in the NeMo Flow WebAssembly crate.

use super::*;
use nemo_flow::api::event::{BaseEvent, EventCategory, MarkEvent, ScopeCategory, ScopeEvent};
use serde_json::json;
use uuid::Uuid;

#[test]
fn test_scope_type_conversion_round_trip() {
    let pairs = [
        (ScopeType::Agent, CoreScopeType::Agent),
        (ScopeType::Function, CoreScopeType::Function),
        (ScopeType::Tool, CoreScopeType::Tool),
        (ScopeType::Llm, CoreScopeType::Llm),
        (ScopeType::Retriever, CoreScopeType::Retriever),
        (ScopeType::Embedder, CoreScopeType::Embedder),
        (ScopeType::Reranker, CoreScopeType::Reranker),
        (ScopeType::Guardrail, CoreScopeType::Guardrail),
        (ScopeType::Evaluator, CoreScopeType::Evaluator),
        (ScopeType::Custom, CoreScopeType::Custom),
        (ScopeType::Unknown, CoreScopeType::Unknown),
    ];

    for (scope_type, core_scope_type) in pairs {
        assert_eq!(CoreScopeType::from(scope_type), core_scope_type);
        assert_eq!(ScopeType::from(core_scope_type), scope_type);
    }
}

#[test]
fn test_handle_wrappers_and_scope_stack_default() {
    let parent_uuid = Uuid::now_v7();
    #[cfg(target_arch = "wasm32")]
    let parent_uuid_str = parent_uuid.to_string();

    let scope = ScopeHandle::from(
        CoreScopeHandle::builder()
            .name("scope")
            .scope_type(CoreScopeType::Guardrail)
            .attributes(ScopeAttributes::PARALLEL)
            .parent_uuid(parent_uuid)
            .data(json!({"data": true}))
            .metadata(json!({"meta": true}))
            .build(),
    );
    assert_eq!(scope.name(), "scope");
    assert_eq!(scope.scope_type(), ScopeType::Guardrail);
    assert_eq!(scope.attributes(), SCOPE_PARALLEL);
    #[cfg(target_arch = "wasm32")]
    assert_eq!(
        scope.parent_uuid().as_string().as_deref(),
        Some(parent_uuid_str.as_str())
    );
    assert!(!scope.uuid().is_empty());

    let tool = ToolHandle::from(
        CoreToolHandle::builder()
            .name("tool")
            .attributes(ToolAttributes::REMOTE)
            .parent_uuid(parent_uuid)
            .build(),
    );
    assert_eq!(tool.name(), "tool");
    assert_eq!(tool.attributes(), TOOL_REMOTE);
    #[cfg(target_arch = "wasm32")]
    assert_eq!(
        tool.parent_uuid().as_string().as_deref(),
        Some(parent_uuid_str.as_str())
    );
    assert!(!tool.uuid().is_empty());

    let llm = LlmHandle::from(
        CoreLlmHandle::builder()
            .name("llm")
            .attributes(LlmAttributes::STATEFUL | LlmAttributes::STREAMING)
            .parent_uuid(parent_uuid)
            .build(),
    );
    assert_eq!(llm.name(), "llm");
    assert_eq!(llm.attributes(), LLM_STATEFUL | LLM_STREAMING);
    #[cfg(target_arch = "wasm32")]
    assert_eq!(
        llm.parent_uuid().as_string().as_deref(),
        Some(parent_uuid_str.as_str())
    );
    assert!(!llm.uuid().is_empty());

    let scope_stack = ScopeStack::default();
    assert!(std::sync::Arc::strong_count(&scope_stack.inner) >= 1);
}

#[test]
fn test_wasm_event_conversion_maps_fields() {
    let parent_uuid = Some(Uuid::now_v7());
    let uuid = Uuid::now_v7();
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name("wasm-event")
            .data(json!({"data": 1}))
            .metadata(json!({"meta": 2}))
            .build(),
        None,
        None,
    ));

    let wasm_event = WasmEvent::from(&event);
    assert_eq!(wasm_event.0["kind"], json!("mark"));
    assert_eq!(
        wasm_event.0["parent_uuid"],
        json!(parent_uuid.map(|value| value.to_string()))
    );
    assert_eq!(wasm_event.0["uuid"], json!(uuid.to_string()));
    assert_eq!(wasm_event.0["name"], json!("wasm-event"));
    assert_eq!(wasm_event.0["data"], json!({"data": 1}));
    assert_eq!(wasm_event.0["metadata"], json!({"meta": 2}));
    assert!(wasm_event.0["timestamp"].as_str().unwrap().contains('T'));
}

#[test]
fn test_wasm_scope_type_is_only_present_on_scope_events() {
    let scope_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("scope-event").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::from(CoreScopeType::Function),
        None,
    ));
    let wasm_scope = WasmEvent::from(&scope_event);
    assert_eq!(wasm_scope.0["kind"], json!("scope"));
    assert_eq!(wasm_scope.0["scope_category"], json!("end"));
    assert_eq!(wasm_scope.0["category"], json!("function"));

    let tool_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("tool-event").build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::tool(),
        None,
    ));
    let wasm_tool = WasmEvent::from(&tool_event);
    assert_eq!(wasm_tool.0["kind"], json!("scope"));
    assert_eq!(wasm_tool.0["category"], json!("tool"));

    let llm_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("llm-event").build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        None,
    ));
    let wasm_llm = WasmEvent::from(&llm_event);
    assert_eq!(wasm_llm.0["kind"], json!("scope"));
    assert_eq!(wasm_llm.0["category"], json!("llm"));
}
