// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::{CStr, CString};
use std::sync::Arc;

use serde_json::json;
use uuid::Uuid;

fn take_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let value = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
    unsafe { crate::convert::nemo_flow_string_free(ptr) };
    Some(value)
}

#[test]
fn test_scope_handle_accessors_and_null_metadata_guard() {
    assert!(unsafe { nemo_flow_scope_handle_metadata(std::ptr::null()) }.is_null());

    let parent_uuid = Uuid::now_v7();
    let handle = FfiScopeHandle(ScopeHandle::new(
        "scope".into(),
        ScopeType::Tool,
        ScopeAttributes::PARALLEL,
        Some(parent_uuid),
        Some(json!({"data": true})),
        Some(json!({"meta": true})),
    ));

    assert_eq!(
        take_string(unsafe { nemo_flow_scope_handle_name(&handle) }),
        Some("scope".into())
    );
    assert_eq!(
        unsafe { nemo_flow_scope_handle_scope_type(&handle) } as i32,
        NemoFlowScopeType::Tool as i32
    );
    assert_eq!(
        unsafe { nemo_flow_scope_handle_attributes(&handle) },
        ScopeAttributes::PARALLEL.bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_scope_handle_parent_uuid(&handle) }),
        Some(parent_uuid.to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_scope_handle_data(&handle) }),
        Some(r#"{"data":true}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_scope_handle_metadata(&handle) }),
        Some(r#"{"meta":true}"#.into())
    );
}

#[test]
fn test_scope_type_conversions_and_handle_null_guards() {
    let scope_types = [
        (NemoFlowScopeType::Agent, ScopeType::Agent),
        (NemoFlowScopeType::Function, ScopeType::Function),
        (NemoFlowScopeType::Tool, ScopeType::Tool),
        (NemoFlowScopeType::Llm, ScopeType::Llm),
        (NemoFlowScopeType::Retriever, ScopeType::Retriever),
        (NemoFlowScopeType::Embedder, ScopeType::Embedder),
        (NemoFlowScopeType::Reranker, ScopeType::Reranker),
        (NemoFlowScopeType::Guardrail, ScopeType::Guardrail),
        (NemoFlowScopeType::Evaluator, ScopeType::Evaluator),
        (NemoFlowScopeType::Custom, ScopeType::Custom),
        (NemoFlowScopeType::Unknown, ScopeType::Unknown),
    ];

    for (ffi, core) in scope_types {
        let round_trip: NemoFlowScopeType = core.into();
        assert_eq!(round_trip as i32, ffi as i32);
        let back: ScopeType = ffi.into();
        assert_eq!(back, core);
    }

    assert!(unsafe { nemo_flow_scope_handle_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_scope_handle_name(std::ptr::null()) }.is_null());
    assert_eq!(
        unsafe { nemo_flow_scope_handle_scope_type(std::ptr::null()) } as i32,
        NemoFlowScopeType::Unknown as i32
    );
    assert_eq!(
        unsafe { nemo_flow_scope_handle_attributes(std::ptr::null()) },
        0
    );
    assert!(unsafe { nemo_flow_scope_handle_parent_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_scope_handle_data(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_scope_handle_metadata(std::ptr::null()) }.is_null());
}

#[test]
fn test_tool_and_llm_handle_accessors_and_null_guards() {
    let parent_uuid = Uuid::now_v7();
    let tool = FfiToolHandle(ToolHandle::new(
        "tool".into(),
        ToolAttributes::LOCAL,
        Some(parent_uuid),
        None,
        None,
    ));
    assert_eq!(
        take_string(unsafe { nemo_flow_tool_handle_uuid(&tool) }),
        Some(tool.0.uuid.to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_tool_handle_name(&tool) }),
        Some("tool".into())
    );
    assert_eq!(
        unsafe { nemo_flow_tool_handle_attributes(&tool) },
        ToolAttributes::LOCAL.bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_tool_handle_parent_uuid(&tool) }),
        Some(parent_uuid.to_string())
    );

    let llm = FfiLLMHandle(LLMHandle::new(
        "llm".into(),
        LLMAttributes::STATELESS | LLMAttributes::STREAMING,
        Some(parent_uuid),
        None,
        None,
    ));
    assert_eq!(
        take_string(unsafe { nemo_flow_llm_handle_uuid(&llm) }),
        Some(llm.0.uuid.to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_llm_handle_name(&llm) }),
        Some("llm".into())
    );
    assert_eq!(
        unsafe { nemo_flow_llm_handle_attributes(&llm) },
        (LLMAttributes::STATELESS | LLMAttributes::STREAMING).bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_llm_handle_parent_uuid(&llm) }),
        Some(parent_uuid.to_string())
    );

    assert!(unsafe { nemo_flow_tool_handle_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_tool_handle_name(std::ptr::null()) }.is_null());
    assert_eq!(
        unsafe { nemo_flow_tool_handle_attributes(std::ptr::null()) },
        0
    );
    assert!(unsafe { nemo_flow_tool_handle_parent_uuid(std::ptr::null()) }.is_null());

    assert!(unsafe { nemo_flow_llm_handle_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_llm_handle_name(std::ptr::null()) }.is_null());
    assert_eq!(
        unsafe { nemo_flow_llm_handle_attributes(std::ptr::null()) },
        0
    );
    assert!(unsafe { nemo_flow_llm_handle_parent_uuid(std::ptr::null()) }.is_null());
}

#[test]
fn test_llm_request_null_inputs_event_null_guards_and_free_nulls() {
    let request_ptr = unsafe { nemo_flow_llm_request_new(std::ptr::null(), std::ptr::null()) };
    assert!(!request_ptr.is_null());
    assert_eq!(
        take_string(unsafe { nemo_flow_llm_request_headers(request_ptr) }),
        Some("{}".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_llm_request_content(request_ptr) }),
        Some("null".into())
    );
    unsafe { nemo_flow_llm_request_free(request_ptr) };

    assert!(unsafe { nemo_flow_llm_request_headers(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_llm_request_content(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_name(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_kind(std::ptr::null()) }.is_null());
    assert_eq!(unsafe { nemo_flow_event_attributes(std::ptr::null()) }, 0);
    assert!(unsafe { nemo_flow_event_data(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_metadata(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_timestamp(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_input(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_output(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_model_name(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_tool_call_id(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_parent_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_flow_event_scope_type(std::ptr::null()) }.is_null());

    unsafe {
        nemo_flow_scope_handle_free(std::ptr::null_mut());
        nemo_flow_tool_handle_free(std::ptr::null_mut());
        nemo_flow_llm_handle_free(std::ptr::null_mut());
        nemo_flow_llm_request_free(std::ptr::null_mut());
        nemo_flow_event_free(std::ptr::null_mut());
        nemo_flow_scope_stack_free(std::ptr::null_mut());
        nemo_flow_atif_exporter_free(std::ptr::null_mut());
        nemo_flow_otel_subscriber_free(std::ptr::null_mut());
        nemo_flow_openinference_subscriber_free(std::ptr::null_mut());
    }
}

#[test]
fn test_llm_request_and_event_accessors() {
    let headers = CString::new(r#"{"header":"value"}"#).unwrap();
    let content = CString::new(r#"{"prompt":"hi"}"#).unwrap();
    let request_ptr = unsafe { nemo_flow_llm_request_new(headers.as_ptr(), content.as_ptr()) };
    assert!(!request_ptr.is_null());
    assert_eq!(
        take_string(unsafe { nemo_flow_llm_request_headers(request_ptr) }),
        Some(r#"{"header":"value"}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_llm_request_content(request_ptr) }),
        Some(r#"{"prompt":"hi"}"#.into())
    );
    unsafe { nemo_flow_llm_request_free(request_ptr) };

    let parent_uuid = Uuid::now_v7();
    let scope_event = Event::scope_start(
        Some(parent_uuid),
        Uuid::now_v7(),
        "ffi-event",
        Some(json!({"data": 1})),
        Some(json!({"meta": 2})),
        ScopeAttributes::empty(),
        ScopeType::Guardrail,
    );
    let ffi_event = FfiEvent(scope_event.clone());

    assert_eq!(
        take_string(unsafe { nemo_flow_event_kind(&ffi_event) }),
        Some("ScopeStart".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_uuid(&ffi_event) }),
        Some(scope_event.uuid().to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_name(&ffi_event) }),
        Some("ffi-event".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_data(&ffi_event) }),
        Some(r#"{"data":1}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_metadata(&ffi_event) }),
        Some(r#"{"meta":2}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_scope_type(&ffi_event) }),
        Some("Guardrail".into())
    );
    assert_eq!(
        unsafe { nemo_flow_event_attributes(&ffi_event) },
        ScopeAttributes::empty().bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_parent_uuid(&ffi_event) }),
        scope_event.parent_uuid().map(|uuid| uuid.to_string())
    );
    assert!(
        take_string(unsafe { nemo_flow_event_timestamp(&ffi_event) })
            .unwrap()
            .contains('T')
    );

    assert_eq!(
        take_string(unsafe { nemo_flow_event_input(&ffi_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_output(&ffi_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_model_name(&ffi_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_tool_call_id(&ffi_event) }),
        None
    );

    let llm_event = Event::llm_start(
        Some(parent_uuid),
        Uuid::now_v7(),
        "ffi-llm",
        None,
        None,
        LLMAttributes::empty(),
        Some(json!({"input": true})),
        Some("model".into()),
        None,
    );
    let ffi_llm_event = FfiEvent(llm_event);
    assert_eq!(
        take_string(unsafe { nemo_flow_event_input(&ffi_llm_event) }),
        Some(r#"{"input":true}"#.into())
    );
    assert_eq!(
        unsafe { nemo_flow_event_attributes(&ffi_llm_event) },
        LLMAttributes::empty().bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_model_name(&ffi_llm_event) }),
        Some("model".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_scope_type(&ffi_llm_event) }),
        None
    );

    let tool_event = Event::tool_end(
        Some(parent_uuid),
        Uuid::now_v7(),
        "ffi-tool",
        None,
        None,
        ToolAttributes::empty(),
        Some(json!({"output": true})),
        Some("tool-call-id".into()),
    );
    let ffi_tool_event = FfiEvent(tool_event);
    assert_eq!(
        take_string(unsafe { nemo_flow_event_output(&ffi_tool_event) }),
        Some(r#"{"output":true}"#.into())
    );
    assert_eq!(
        unsafe { nemo_flow_event_attributes(&ffi_tool_event) },
        ToolAttributes::empty().bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_tool_call_id(&ffi_tool_event) }),
        Some("tool-call-id".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_flow_event_scope_type(&ffi_tool_event) }),
        None
    );

    let mark_event = Event::mark(Some(parent_uuid), Uuid::now_v7(), "ffi-mark", None, None);
    let ffi_mark_event = FfiEvent(mark_event);
    assert_eq!(
        take_string(unsafe { nemo_flow_event_scope_type(&ffi_mark_event) }),
        None
    );
    assert_eq!(unsafe { nemo_flow_event_attributes(&ffi_mark_event) }, 0);
}

#[test]
fn test_annotated_event_accessors_and_codec_handles() {
    let annotated_request = nemo_flow::codec::request::AnnotatedLLMRequest {
        messages: vec![nemo_flow::codec::request::Message::User {
            content: nemo_flow::codec::request::MessageContent::Text("hello".into()),
            name: Some("tester".into()),
        }],
        model: Some("gpt-test".into()),
        params: None,
        tools: None,
        tool_choice: None,
        extra: serde_json::Map::from_iter([("provider".into(), json!("ffi"))]),
    };
    let llm_start = Event::llm_start(
        None,
        Uuid::now_v7(),
        "annotated-start",
        None,
        None,
        LLMAttributes::STREAMING,
        Some(json!({"input": "value"})),
        Some("gpt-test".into()),
        Some(Arc::new(annotated_request)),
    );
    let ffi_start = FfiEvent(llm_start);
    let annotated_request_json =
        take_string(unsafe { nemo_flow_event_annotated_request(&ffi_start) })
            .expect("expected annotated request json");
    let annotated_request_value: serde_json::Value =
        serde_json::from_str(&annotated_request_json).unwrap();
    assert_eq!(annotated_request_value["model"], json!("gpt-test"));
    assert_eq!(annotated_request_value["provider"], json!("ffi"));
    assert!(unsafe { nemo_flow_event_annotated_response(&ffi_start) }.is_null());

    let annotated_response = nemo_flow::codec::response::AnnotatedLLMResponse {
        id: Some("resp_123".into()),
        model: Some("gpt-test".into()),
        message: Some(nemo_flow::codec::request::MessageContent::Text(
            "done".into(),
        )),
        tool_calls: None,
        finish_reason: Some(nemo_flow::codec::response::FinishReason::Complete),
        usage: None,
        api_specific: None,
        extra: serde_json::Map::from_iter([("trace".into(), json!(true))]),
    };
    let llm_end = Event::llm_end(
        None,
        Uuid::now_v7(),
        "annotated-end",
        None,
        None,
        LLMAttributes::STATELESS,
        Some(json!({"output": "value"})),
        Some("gpt-test".into()),
        Some(Arc::new(annotated_response)),
    );
    let ffi_end = FfiEvent(llm_end);
    let annotated_response_json =
        take_string(unsafe { nemo_flow_event_annotated_response(&ffi_end) })
            .expect("expected annotated response json");
    let annotated_response_value: serde_json::Value =
        serde_json::from_str(&annotated_response_json).unwrap();
    assert_eq!(annotated_response_value["id"], json!("resp_123"));
    assert_eq!(annotated_response_value["trace"], json!(true));
    assert!(unsafe { nemo_flow_event_annotated_request(&ffi_end) }.is_null());

    let scope_event = FfiEvent(Event::scope_start(
        None,
        Uuid::now_v7(),
        "plain-scope",
        None,
        None,
        ScopeAttributes::PARALLEL,
        ScopeType::Function,
    ));
    assert!(unsafe { nemo_flow_event_annotated_request(&scope_event) }.is_null());
    assert!(unsafe { nemo_flow_event_annotated_response(&scope_event) }.is_null());

    let openai_chat = crate::api::nemo_flow_openai_chat_codec_new();
    let openai_responses = crate::api::nemo_flow_openai_responses_codec_new();
    let anthropic = crate::api::nemo_flow_anthropic_messages_codec_new();
    assert!(!openai_chat.is_null());
    assert!(!openai_responses.is_null());
    assert!(!anthropic.is_null());

    unsafe {
        nemo_flow_codec_free(openai_chat);
        nemo_flow_codec_free(openai_responses);
        nemo_flow_codec_free(anthropic);
        nemo_flow_codec_free(std::ptr::null_mut());
    }
}
