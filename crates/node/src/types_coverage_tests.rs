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
    let mut event = core_types::Event::new(
        Some(Uuid::new_v4()),
        Uuid::new_v4(),
        Some("node-event".into()),
        Some(json!({"data": 1})),
        Some(json!({"meta": 2})),
        None,
        core_types::EventType::End,
        Some(core_types::ScopeType::Llm),
    );
    event.input = Some(json!({"input": true}));
    event.output = Some(json!({"output": true}));
    event.model_name = Some("model".into());
    event.tool_call_id = Some("tool-call-id".into());
    event.root_uuid = Some(Uuid::new_v4());

    let js_event = JsEvent::from(&event);
    assert_eq!(
        js_event.parent_uuid,
        event.parent_uuid.map(|uuid| uuid.to_string())
    );
    assert_eq!(js_event.uuid, event.uuid.to_string());
    assert_eq!(js_event.name, Some("node-event".into()));
    assert_eq!(js_event.data, Some(json!({"data": 1})));
    assert_eq!(js_event.metadata, Some(json!({"meta": 2})));
    assert_eq!(js_event.event_type, 1);
    assert_eq!(js_event.scope_type, Some(3));
    assert_eq!(js_event.input, Some(r#"{"input":true}"#.into()));
    assert_eq!(js_event.output, Some(r#"{"output":true}"#.into()));
    assert_eq!(js_event.model_name, Some("model".into()));
    assert_eq!(js_event.tool_call_id, Some("tool-call-id".into()));
    assert_eq!(
        js_event.root_uuid,
        event.root_uuid.map(|uuid| uuid.to_string())
    );
    assert!(!js_event.timestamp.is_empty());
}

#[test]
fn test_event_and_scope_stack_conversions_cover_remaining_variants() {
    let stack = JsScopeStack::from(nvidia_nat_nexus_core::create_scope_stack());
    let _ = stack;

    let event_types = [
        core_types::EventType::Start,
        core_types::EventType::End,
        core_types::EventType::Mark,
    ];
    for event_type in event_types {
        let converted = EventType::from(event_type);
        match (event_type, converted) {
            (core_types::EventType::Start, EventType::Start)
            | (core_types::EventType::End, EventType::End)
            | (core_types::EventType::Mark, EventType::Mark) => {}
            _ => panic!("event type conversion mismatch"),
        }
    }

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
        let event = core_types::Event::new(
            None,
            Uuid::new_v4(),
            Some("variant-event".into()),
            None,
            None,
            None,
            core_types::EventType::Start,
            Some(scope_type),
        );
        let js_event = JsEvent::from(&event);
        assert_eq!(js_event.scope_type, Some(expected));
    }
}
