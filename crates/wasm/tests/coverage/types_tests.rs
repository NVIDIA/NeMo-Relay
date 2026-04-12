// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;
use uuid::Uuid;

#[test]
fn test_scope_type_conversion_round_trip() {
    let pairs = [
        (SCOPE_TYPE_AGENT, ScopeType::Agent),
        (SCOPE_TYPE_FUNCTION, ScopeType::Function),
        (SCOPE_TYPE_TOOL, ScopeType::Tool),
        (SCOPE_TYPE_LLM, ScopeType::Llm),
        (SCOPE_TYPE_RETRIEVER, ScopeType::Retriever),
        (SCOPE_TYPE_EMBEDDER, ScopeType::Embedder),
        (SCOPE_TYPE_RERANKER, ScopeType::Reranker),
        (SCOPE_TYPE_GUARDRAIL, ScopeType::Guardrail),
        (SCOPE_TYPE_EVALUATOR, ScopeType::Evaluator),
        (SCOPE_TYPE_CUSTOM, ScopeType::Custom),
        (SCOPE_TYPE_UNKNOWN, ScopeType::Unknown),
    ];

    for (raw, scope_type) in pairs {
        assert_eq!(i32_to_scope_type(raw), scope_type);
        assert_eq!(scope_type_to_i32(scope_type), raw);
    }
    assert_eq!(i32_to_scope_type(999), ScopeType::Unknown);
}

#[test]
fn test_handle_wrappers_and_scope_stack_default() {
    let parent_uuid = Uuid::now_v7();

    let scope = WasmScopeHandle::from(ScopeHandle::new(
        "scope".into(),
        ScopeType::Guardrail,
        ScopeAttributes::PARALLEL,
        Some(parent_uuid),
        Some(json!({"data": true})),
        Some(json!({"meta": true})),
    ));
    assert_eq!(scope.name(), "scope");
    assert_eq!(scope.scope_type(), SCOPE_TYPE_GUARDRAIL);
    assert_eq!(scope.attributes(), SCOPE_PARALLEL);
    assert_eq!(scope.parent_uuid(), Some(parent_uuid.to_string()));
    assert!(!scope.uuid().is_empty());

    let tool = WasmToolHandle::from(ToolHandle::new(
        "tool".into(),
        ToolAttributes::LOCAL,
        Some(parent_uuid),
        None,
        None,
    ));
    assert_eq!(tool.name(), "tool");
    assert_eq!(tool.attributes(), TOOL_LOCAL);
    assert_eq!(tool.parent_uuid(), Some(parent_uuid.to_string()));
    assert!(!tool.uuid().is_empty());

    let llm = WasmLLMHandle::from(LLMHandle::new(
        "llm".into(),
        LLMAttributes::STATELESS | LLMAttributes::STREAMING,
        Some(parent_uuid),
        None,
        None,
    ));
    assert_eq!(llm.name(), "llm");
    assert_eq!(llm.attributes(), LLM_STATELESS | LLM_STREAMING);
    assert_eq!(llm.parent_uuid(), Some(parent_uuid.to_string()));
    assert!(!llm.uuid().is_empty());

    let scope_stack = WasmScopeStack::default();
    assert!(std::sync::Arc::strong_count(&scope_stack.inner) >= 1);
}

#[test]
fn test_wasm_event_conversion_maps_fields() {
    let parent_uuid = Some(Uuid::now_v7());
    let uuid = Uuid::now_v7();
    let event = Event::mark(
        parent_uuid,
        uuid,
        "wasm-event",
        Some(json!({"data": 1})),
        Some(json!({"meta": 2})),
    );

    let wasm_event = WasmEvent::from(&event);
    match wasm_event {
        WasmEvent::Mark {
            parent_uuid: wasm_parent_uuid,
            uuid: wasm_uuid,
            timestamp,
            name,
            data,
            metadata,
        } => {
            assert_eq!(wasm_parent_uuid, parent_uuid.map(|value| value.to_string()));
            assert_eq!(wasm_uuid, uuid.to_string());
            assert_eq!(name, "wasm-event");
            assert_eq!(data, Some(json!({"data": 1})));
            assert_eq!(metadata, Some(json!({"meta": 2})));
            assert!(!timestamp.is_empty());
        }
        _ => panic!("expected Mark event"),
    }
}

#[test]
fn test_wasm_scope_type_is_only_present_on_scope_events() {
    let scope_event = Event::scope_end(
        None,
        Uuid::now_v7(),
        "scope-event",
        None,
        None,
        ScopeAttributes::empty(),
        ScopeType::Function,
    );
    match WasmEvent::from(&scope_event) {
        WasmEvent::ScopeEnd { scope_type, .. } => assert_eq!(scope_type, SCOPE_TYPE_FUNCTION),
        _ => panic!("expected ScopeEnd event"),
    }

    let tool_event = Event::tool_start(
        None,
        Uuid::now_v7(),
        "tool-event",
        None,
        None,
        ToolAttributes::empty(),
        None,
        None,
    );
    match WasmEvent::from(&tool_event) {
        WasmEvent::ToolStart { .. } => {}
        _ => panic!("expected ToolStart event"),
    }

    let llm_event = Event::llm_start(
        None,
        Uuid::now_v7(),
        "llm-event",
        None,
        None,
        LLMAttributes::empty(),
        None,
        None,
        None,
    );
    match WasmEvent::from(&llm_event) {
        WasmEvent::LLMStart { .. } => {}
        _ => panic!("expected LLMStart event"),
    }
}
