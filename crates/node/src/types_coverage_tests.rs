// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;
use uuid::Uuid;

#[test]
fn test_scope_type_round_trip_all_variants() {
    let variants = [
        (ScopeType::Agent, core_types::ScopeType::Agent),
        (ScopeType::Function, core_types::ScopeType::Function),
        (ScopeType::Tool, core_types::ScopeType::Tool),
        (ScopeType::Llm, core_types::ScopeType::Llm),
        (ScopeType::Retriever, core_types::ScopeType::Retriever),
        (ScopeType::Embedder, core_types::ScopeType::Embedder),
        (ScopeType::Reranker, core_types::ScopeType::Reranker),
        (ScopeType::Guardrail, core_types::ScopeType::Guardrail),
        (ScopeType::Evaluator, core_types::ScopeType::Evaluator),
        (ScopeType::Custom, core_types::ScopeType::Custom),
        (ScopeType::Unknown, core_types::ScopeType::Unknown),
    ];

    for (node_variant, core_variant) in variants {
        let converted_core: core_types::ScopeType = node_variant.into();
        let round_trip: ScopeType = core_variant.into();
        assert_eq!(converted_core, core_variant);
        assert_eq!(node_variant as i32, round_trip as i32);
    }
}

#[test]
fn test_handle_and_request_getters() {
    let parent_uuid = Uuid::new_v4();

    let scope = JsScopeHandle::from(core_types::ScopeHandle::new(
        "scope".into(),
        core_types::ScopeType::Tool,
        core_types::ScopeAttributes::PARALLEL | core_types::ScopeAttributes::RELOCATABLE,
        Some(parent_uuid),
        Some(json!({"data": true})),
        Some(json!({"meta": true})),
    ));
    assert_eq!(scope.name(), "scope");
    assert_eq!(scope.scope_type() as i32, ScopeType::Tool as i32);
    assert_eq!(
        scope.attributes(),
        SCOPE_ATTR_PARALLEL | SCOPE_ATTR_RELOCATABLE
    );
    assert_eq!(scope.parent_uuid(), Some(parent_uuid.to_string()));
    assert_eq!(scope.data(), Some(json!({"data": true})));
    assert_eq!(scope.metadata(), Some(json!({"meta": true})));

    let tool = JsToolHandle::from(core_types::ToolHandle::new(
        "tool".into(),
        core_types::ToolAttributes::LOCAL,
        Some(parent_uuid),
        None,
        None,
    ));
    assert_eq!(tool.name(), "tool");
    assert_eq!(tool.attributes(), TOOL_ATTR_LOCAL);
    assert_eq!(tool.parent_uuid(), Some(parent_uuid.to_string()));

    let llm = JsLLMHandle::from(core_types::LLMHandle::new(
        "llm".into(),
        core_types::LLMAttributes::STATELESS | core_types::LLMAttributes::STREAMING,
        Some(parent_uuid),
        None,
        None,
    ));
    assert_eq!(llm.name(), "llm");
    assert_eq!(llm.attributes(), LLM_ATTR_STATELESS | LLM_ATTR_STREAMING);
    assert_eq!(llm.parent_uuid(), Some(parent_uuid.to_string()));

    let request = JsLLMRequest::new(JsLLMRequestInit {
        headers: json!({"x-trace": "1"}),
        content: json!({"prompt": "hi"}),
    });
    assert_eq!(request.headers(), json!({"x-trace": "1"}));
    assert_eq!(request.content(), json!({"prompt": "hi"}));

    let request_with_non_object_headers = JsLLMRequest::new(JsLLMRequestInit {
        headers: Json::Null,
        content: json!({"prompt": "fallback"}),
    });
    assert_eq!(request_with_non_object_headers.headers(), json!({}));
    assert_eq!(
        request_with_non_object_headers.content(),
        json!({"prompt": "fallback"})
    );
}

#[test]
fn test_js_event_conversion_maps_all_fields() {
    let parent_uuid = Some(Uuid::new_v4());
    let uuid = Uuid::new_v4();
    let event = core_types::Event::llm_end(
        parent_uuid,
        uuid,
        "node-event",
        Some(json!({"data": 1})),
        Some(json!({"meta": 2})),
        core_types::LLMAttributes::STATELESS,
        Some(json!({"output": true})),
        Some("model".into()),
        None,
    );

    let js_event = JsEvent::from(&event);
    match js_event {
        JsEvent::LLMEnd {
            parent_uuid: js_parent_uuid,
            uuid: js_uuid,
            timestamp,
            name,
            data,
            metadata,
            attributes,
            output,
            model_name,
            annotated_response,
        } => {
            assert_eq!(js_parent_uuid, parent_uuid.map(|value| value.to_string()));
            assert_eq!(js_uuid, uuid.to_string());
            assert_eq!(name, "node-event");
            assert_eq!(data, Some(json!({"data": 1})));
            assert_eq!(metadata, Some(json!({"meta": 2})));
            assert_eq!(attributes, LLM_ATTR_STATELESS);
            assert_eq!(output, Some(json!({"output": true})));
            assert_eq!(model_name, Some("model".into()));
            assert!(annotated_response.is_none());
            assert!(!timestamp.is_empty());
        }
        _ => panic!("expected LLMEnd event"),
    }
}

#[test]
fn test_event_and_scope_stack_conversions_cover_remaining_variants() {
    let stack = JsScopeStack::from(nemo_flow_core::create_scope_stack());
    let _ = stack;

    let remaining_scope_types = [
        (core_types::ScopeType::Retriever, 4),
        (core_types::ScopeType::Embedder, 5),
        (core_types::ScopeType::Reranker, 6),
        (core_types::ScopeType::Guardrail, 7),
        (core_types::ScopeType::Evaluator, 8),
        (core_types::ScopeType::Custom, 9),
        (core_types::ScopeType::Unknown, 10),
    ];

    for (scope_type, expected) in remaining_scope_types {
        let event = core_types::Event::scope_start(
            None,
            Uuid::new_v4(),
            "variant-event",
            None,
            None,
            core_types::ScopeAttributes::empty(),
            scope_type,
        );
        let js_event = JsEvent::from(&event);
        match js_event {
            JsEvent::ScopeStart { scope_type, .. } => assert_eq!(scope_type, expected),
            _ => panic!("expected ScopeStart event"),
        }
    }
}

#[test]
fn test_scope_type_is_only_present_on_scope_events() {
    let scope_event = core_types::Event::scope_start(
        None,
        Uuid::new_v4(),
        "scope-event",
        None,
        None,
        core_types::ScopeAttributes::empty(),
        core_types::ScopeType::Function,
    );
    match JsEvent::from(&scope_event) {
        JsEvent::ScopeStart { scope_type, .. } => {
            assert_eq!(scope_type, ScopeType::Function as i32)
        }
        _ => panic!("expected ScopeStart event"),
    }

    let tool_event = core_types::Event::tool_start(
        None,
        Uuid::new_v4(),
        "tool-event",
        None,
        None,
        core_types::ToolAttributes::empty(),
        None,
        None,
    );
    match JsEvent::from(&tool_event) {
        JsEvent::ToolStart { .. } => {}
        _ => panic!("expected ToolStart event"),
    }

    let llm_event = core_types::Event::llm_end(
        None,
        Uuid::new_v4(),
        "llm-event",
        None,
        None,
        core_types::LLMAttributes::empty(),
        None,
        None,
        None,
    );
    match JsEvent::from(&llm_event) {
        JsEvent::LLMEnd { .. } => {}
        _ => panic!("expected LLMEnd event"),
    }
}
