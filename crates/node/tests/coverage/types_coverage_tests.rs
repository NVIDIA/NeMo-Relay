// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_flow::context::scope_stack::create_scope_stack;
use nemo_flow::types::event::Event;
use nemo_flow::types::llm::{LLMAttributes, LLMHandle};
use nemo_flow::types::scope::{ScopeAttributes, ScopeHandle, ScopeType as CoreScopeType};
use nemo_flow::types::tool::{ToolAttributes, ToolHandle};
use serde_json::json;
use uuid::Uuid;

#[test]
fn test_scope_type_round_trip_all_variants() {
    let variants = [
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

    for (node_variant, core_variant) in variants {
        let converted_core: CoreScopeType = node_variant.into();
        let round_trip: ScopeType = core_variant.into();
        assert_eq!(converted_core, core_variant);
        assert_eq!(node_variant as i32, round_trip as i32);
    }
}

#[test]
fn test_handle_and_request_getters() {
    let parent_uuid = Uuid::now_v7();

    let scope = JsScopeHandle::from(ScopeHandle::new(
        "scope".into(),
        CoreScopeType::Tool,
        ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE,
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

    let tool = JsToolHandle::from(ToolHandle::new(
        "tool".into(),
        ToolAttributes::LOCAL,
        Some(parent_uuid),
        None,
        None,
    ));
    assert_eq!(tool.name(), "tool");
    assert_eq!(tool.attributes(), TOOL_ATTR_LOCAL);
    assert_eq!(tool.parent_uuid(), Some(parent_uuid.to_string()));

    let llm = JsLLMHandle::from(LLMHandle::new(
        "llm".into(),
        LLMAttributes::STATELESS | LLMAttributes::STREAMING,
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
    let parent_uuid = Some(Uuid::now_v7());
    let uuid = Uuid::now_v7();
    let event = Event::llm_end(
        parent_uuid,
        uuid,
        "node-event",
        Some(json!({"data": 1})),
        Some(json!({"meta": 2})),
        LLMAttributes::STATELESS,
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
    let stack = JsScopeStack::from(create_scope_stack());
    let _ = stack;

    let remaining_scope_types = [
        (CoreScopeType::Retriever, 4),
        (CoreScopeType::Embedder, 5),
        (CoreScopeType::Reranker, 6),
        (CoreScopeType::Guardrail, 7),
        (CoreScopeType::Evaluator, 8),
        (CoreScopeType::Custom, 9),
        (CoreScopeType::Unknown, 10),
    ];

    for (scope_type, expected) in remaining_scope_types {
        let event = Event::scope_start(
            None,
            Uuid::now_v7(),
            "variant-event",
            None,
            None,
            ScopeAttributes::empty(),
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
    let scope_event = Event::scope_start(
        None,
        Uuid::now_v7(),
        "scope-event",
        None,
        None,
        ScopeAttributes::empty(),
        CoreScopeType::Function,
    );
    match JsEvent::from(&scope_event) {
        JsEvent::ScopeStart { scope_type, .. } => {
            assert_eq!(scope_type, ScopeType::Function as i32)
        }
        _ => panic!("expected ScopeStart event"),
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
    match JsEvent::from(&tool_event) {
        JsEvent::ToolStart { .. } => {}
        _ => panic!("expected ToolStart event"),
    }

    let llm_event = Event::llm_end(
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
    match JsEvent::from(&llm_event) {
        JsEvent::LLMEnd { .. } => {}
        _ => panic!("expected LLMEnd event"),
    }
}

#[test]
fn test_additional_event_variants_and_scope_stack_constructor() {
    let stack = JsScopeStack::new();
    let _ = stack;

    match JsEvent::from(&Event::scope_end(
        None,
        Uuid::now_v7(),
        "scope-end",
        Some(json!({"done": true})),
        Some(json!({"meta": true})),
        ScopeAttributes::RELOCATABLE,
        CoreScopeType::Function,
    )) {
        JsEvent::ScopeEnd {
            attributes,
            scope_type,
            ..
        } => {
            assert_eq!(attributes, SCOPE_ATTR_RELOCATABLE);
            assert_eq!(scope_type, ScopeType::Function as i32);
        }
        _ => panic!("expected ScopeEnd event"),
    }

    match JsEvent::from(&Event::tool_start(
        None,
        Uuid::now_v7(),
        "tool-start",
        Some(json!({"input": true})),
        Some(json!({"meta": true})),
        ToolAttributes::LOCAL,
        Some(json!({"args": 1})),
        Some("tool-call".into()),
    )) {
        JsEvent::ToolStart {
            attributes,
            input,
            tool_call_id,
            ..
        } => {
            assert_eq!(attributes, TOOL_ATTR_LOCAL);
            assert_eq!(input, Some(json!({"args": 1})));
            assert_eq!(tool_call_id, Some("tool-call".into()));
        }
        _ => panic!("expected ToolStart event"),
    }

    match JsEvent::from(&Event::tool_end(
        None,
        Uuid::now_v7(),
        "tool-end",
        Some(json!({"data": true})),
        Some(json!({"meta": true})),
        ToolAttributes::LOCAL,
        Some(json!({"result": 2})),
        Some("tool-call".into()),
    )) {
        JsEvent::ToolEnd {
            attributes,
            output,
            tool_call_id,
            ..
        } => {
            assert_eq!(attributes, TOOL_ATTR_LOCAL);
            assert_eq!(output, Some(json!({"result": 2})));
            assert_eq!(tool_call_id, Some("tool-call".into()));
        }
        _ => panic!("expected ToolEnd event"),
    }

    let annotated_request = nemo_flow::codec::request::AnnotatedLLMRequest {
        messages: vec![],
        model: Some("demo".into()),
        params: None,
        tools: None,
        tool_choice: None,
        extra: serde_json::Map::new(),
    };
    match JsEvent::from(&Event::llm_start(
        None,
        Uuid::now_v7(),
        "llm-start",
        Some(json!({"data": true})),
        Some(json!({"meta": true})),
        LLMAttributes::STREAMING,
        Some(json!({"input": true})),
        Some("demo-model".into()),
        Some(std::sync::Arc::new(annotated_request)),
    )) {
        JsEvent::LLMStart {
            attributes,
            input,
            model_name,
            annotated_request,
            ..
        } => {
            assert_eq!(attributes, LLM_ATTR_STREAMING);
            assert_eq!(input, Some(json!({"input": true})));
            assert_eq!(model_name, Some("demo-model".into()));
            assert_eq!(annotated_request.unwrap()["model"], json!("demo"));
        }
        _ => panic!("expected LLMStart event"),
    }

    match JsEvent::from(&Event::mark(
        None,
        Uuid::now_v7(),
        "mark",
        Some(json!({"data": true})),
        Some(json!({"meta": true})),
    )) {
        JsEvent::Mark { data, metadata, .. } => {
            assert_eq!(data, Some(json!({"data": true})));
            assert_eq!(metadata, Some(json!({"meta": true})));
        }
        _ => panic!("expected Mark event"),
    }
}

#[test]
fn test_builtin_codecs_round_trip_and_reject_invalid_inputs() {
    let original = json!({
        "headers": {},
        "content": {
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"}
            ],
            "model": "gpt-4o-mini"
        }
    });

    let openai_chat = JsOpenAIChatCodec::new();
    let annotated = openai_chat.decode(original.clone()).unwrap();
    let reencoded = openai_chat
        .encode(annotated.clone(), original.clone())
        .unwrap();
    assert_eq!(reencoded["content"]["model"], json!("gpt-4o-mini"));
    assert!(openai_chat.decode(json!({"content": {}})).is_err());
    assert!(openai_chat.decode_response(json!({"choices": []})).is_ok());

    let openai_responses = JsOpenAIResponsesCodec::new();
    let original_responses = json!({
        "headers": {},
        "content": {
            "model": "gpt-4.1-mini",
            "input": "hello"
        }
    });
    let annotated = openai_responses.decode(original_responses.clone()).unwrap();
    let reencoded = openai_responses
        .encode(annotated.clone(), original_responses.clone())
        .unwrap();
    assert_eq!(reencoded["content"]["model"], json!("gpt-4.1-mini"));
    assert!(openai_responses.decode(json!({"content": 1})).is_err());
    assert!(
        openai_responses
            .decode_response(json!({"status": "completed", "output": []}))
            .is_ok()
    );

    let anthropic = JsAnthropicMessagesCodec::new();
    let original_anthropic = json!({
        "headers": {},
        "content": {
            "model": "claude-3-5-haiku-latest",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16
        }
    });
    let annotated = anthropic.decode(original_anthropic.clone()).unwrap();
    let reencoded = anthropic
        .encode(annotated, original_anthropic.clone())
        .unwrap();
    assert_eq!(
        reencoded["content"]["model"],
        json!("claude-3-5-haiku-latest")
    );
    assert!(
        anthropic
            .encode(json!({"messages": "bad"}), original_anthropic)
            .is_err()
    );
    assert!(
        anthropic
            .decode_response(json!({"stop_reason": "end_turn", "content": []}))
            .is_ok()
    );
}
